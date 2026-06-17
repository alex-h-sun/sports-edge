"""Snapshot sync: publish the offline DB+artifacts bundle and pull it onto the volume.

Offline (scripts/publish_snapshot.py) packages sports.db + artifacts + ledger into a
.tar.gz, uploads it with a ``latest.json`` manifest. On container boot and on
POST /api/admin/reload we read that manifest and, if its version is newer than what is
already on the volume, download + verify (sha256) + atomically extract it into
SNAPSHOT_DATA_DIR.

Backends are chosen from SNAPSHOT_URL:
  - ``s3://bucket/prefix``           S3 / Cloudflare R2 (needs boto3 + S3 creds in env)
  - ``file:///abs/dir`` or ``/abs/dir``   a local/NFS directory (tests, rsync target)
Empty SNAPSHOT_URL disables syncing (the volume is then managed manually).

Canonical archive layout (so publish and sync agree regardless of local paths):
  sports.db
  artifacts/<name>.pkl ...
  sims/paper_ledger.csv
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger("webapp.snapshot")

MANIFEST_NAME = "latest.json"
VERSION_MARKER = ".snapshot_version"
ARCHIVE_DB = "sports.db"
ARCHIVE_ARTIFACTS = "artifacts"
ARCHIVE_LEDGER = "sims/paper_ledger.csv"


# ── storage backends ─────────────────────────────────────────────────────────────

class LocalBackend:
    """A plain directory used as the snapshot store (also the no-bucket option)."""

    def __init__(self, root: str):
        self.root = Path(root)

    def get_bytes(self, key: str) -> bytes | None:
        p = self.root / key
        return p.read_bytes() if p.is_file() else None

    def download(self, key: str, dest: str) -> None:
        shutil.copyfile(self.root / key, dest)

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def upload(self, src: str, key: str) -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, p)


class S3Backend:
    """S3 / R2 backend. boto3 is imported lazily so the serve image can skip it when
    SNAPSHOT_URL is local or unset."""

    def __init__(self, bucket: str, prefix: str):
        import boto3  # lazy: only needed when actually syncing from S3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,  # set for Cloudflare R2
            region_name=os.getenv("AWS_REGION") or os.getenv("S3_REGION") or None,
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def get_bytes(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self._key(key))
            return obj["Body"].read()
        except ClientError:
            return None

    def download(self, key: str, dest: str) -> None:
        self.s3.download_file(self.bucket, self._key(key), dest)

    def put_bytes(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def upload(self, src: str, key: str) -> None:
        self.s3.upload_file(src, self.bucket, self._key(key))


def backend_for(url: str):
    """Return a storage backend for SNAPSHOT_URL, or None when syncing is disabled."""
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme == "s3":
        return S3Backend(parsed.netloc, parsed.path)
    if parsed.scheme in ("file", ""):
        return LocalBackend(parsed.path if parsed.scheme == "file" else url)
    raise ValueError(f"Unsupported SNAPSHOT_URL scheme: {parsed.scheme!r}")


# ── archive helpers ──────────────────────────────────────────────────────────────

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_archive(out_path: str, db_path: str, artifacts_dir: str, ledger_path: str) -> None:
    """Write a .tar.gz with the canonical layout from the given local paths.

    The DB is copied via the SQLite backup API first, so a live/WAL database yields a
    consistent standalone snapshot (no torn read, no -wal/-shm needed)."""
    import sqlite3

    with tempfile.TemporaryDirectory() as tmp:
        clean_db = os.path.join(tmp, "sports.db")
        src = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
        dst = sqlite3.connect(clean_db)
        with dst:
            src.backup(dst)
        src.close()
        dst.close()

        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(clean_db, arcname=ARCHIVE_DB)
            adir = Path(artifacts_dir)
            if adir.is_dir():
                for pkl in sorted(adir.glob("*.pkl")):
                    tar.add(str(pkl), arcname=f"{ARCHIVE_ARTIFACTS}/{pkl.name}")
            if ledger_path and Path(ledger_path).is_file():
                tar.add(ledger_path, arcname=ARCHIVE_LEDGER)


def _safe_extract(tar: tarfile.TarFile, dest: str) -> None:
    """Extract guarding against path traversal (members must stay under dest)."""
    base = os.path.realpath(dest)
    for member in tar.getmembers():
        target = os.path.realpath(os.path.join(dest, member.name))
        if not (target == base or target.startswith(base + os.sep)):
            raise ValueError(f"unsafe path in archive: {member.name}")
    tar.extractall(dest)


def extract_into(tar_path: str, data_dir: str) -> None:
    """Atomically(ish) place the archive contents into data_dir: extract to a temp
    dir, swap sports.db (atomic file rename) and replace the artifacts/ + sims/ dirs."""
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(data), prefix=".incoming-"))
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            _safe_extract(tar, str(staging))

        new_db = staging / ARCHIVE_DB
        if new_db.is_file():
            os.replace(new_db, data / ARCHIVE_DB)  # atomic file swap

        for sub in (ARCHIVE_ARTIFACTS, "sims"):
            new_sub = staging / sub
            if new_sub.exists():
                final = data / sub
                final.parent.mkdir(parents=True, exist_ok=True)
                if final.exists():
                    shutil.rmtree(final)
                os.replace(new_sub, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ── publish (offline) ────────────────────────────────────────────────────────────

def publish(url: str, db_path: str, artifacts_dir: str, ledger_path: str) -> dict:
    """Package the local snapshot and upload it + a latest.json manifest."""
    backend = backend_for(url)
    if backend is None:
        raise ValueError("publish requires a SNAPSHOT_URL / --url")

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_key = f"snapshot-{version}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        local_archive = os.path.join(tmp, archive_key)
        build_archive(local_archive, db_path, artifacts_dir, ledger_path)
        manifest = {
            "version": version,
            "archive": archive_key,
            "sha256": _sha256(local_archive),
            "bytes": os.path.getsize(local_archive),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        backend.upload(local_archive, archive_key)
        backend.put_bytes(MANIFEST_NAME, json.dumps(manifest, indent=2).encode())
    log.info("published snapshot %s (%d bytes)", version, manifest["bytes"])
    return manifest


# ── sync (online) ────────────────────────────────────────────────────────────────

def read_manifest(url: str) -> dict | None:
    backend = backend_for(url)
    if backend is None:
        return None
    raw = backend.get_bytes(MANIFEST_NAME)
    return json.loads(raw) if raw else None


def local_version(data_dir: str) -> str | None:
    marker = Path(data_dir) / VERSION_MARKER
    return marker.read_text().strip() if marker.is_file() else None


def sync(url: str, data_dir: str, force: bool = False) -> dict:
    """Pull the latest snapshot onto data_dir if newer (or force). Returns a status
    dict; never raises for the 'already current' / 'nothing published' cases."""
    backend = backend_for(url)
    if backend is None:
        return {"status": "disabled"}

    manifest = read_manifest(url)
    if not manifest:
        return {"status": "no_manifest"}

    have = local_version(data_dir)
    if have == manifest["version"] and not force:
        return {"status": "current", "version": have}

    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, manifest["archive"])
        backend.download(manifest["archive"], archive)
        digest = _sha256(archive)
        if digest != manifest.get("sha256"):
            raise ValueError(f"snapshot checksum mismatch: {digest} != {manifest.get('sha256')}")
        extract_into(archive, data_dir)

    Path(data_dir, VERSION_MARKER).write_text(manifest["version"])
    log.info("synced snapshot %s -> %s", manifest["version"], data_dir)
    return {"status": "updated", "version": manifest["version"], "previous": have}


# ── CLI: python -m webapp.snapshot --sync ────────────────────────────────────────

def _main() -> None:
    import argparse

    from webapp import settings

    ap = argparse.ArgumentParser(description="Sync the published snapshot onto the volume")
    ap.add_argument("--sync", action="store_true", help="pull the latest snapshot (default action)")
    ap.add_argument("--force", action="store_true", help="re-download even if version matches")
    ap.add_argument("--url", default=os.getenv("SNAPSHOT_URL", ""))
    ap.add_argument("--data-dir", default=os.getenv("SNAPSHOT_DATA_DIR")
                    or (os.path.dirname(settings.DB_PATH) or "data"))
    args = ap.parse_args()
    result = sync(args.url, args.data_dir, force=args.force)
    print(json.dumps(result))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _main()
