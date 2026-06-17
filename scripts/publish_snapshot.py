#!/usr/bin/env python
"""Publish the local snapshot (sports.db + artifacts + paper ledger) for the web app.

Run from the project root after the offline ingest/training, e.g. after
``python run.py --train``:

    python scripts/publish_snapshot.py --url s3://my-bucket/sports-edge
    python scripts/publish_snapshot.py --url file:///abs/path/published   # local / rsync target

Paths default to the same env vars the app uses (DB_PATH, ARTIFACTS_DIR, LEDGER_PATH).
The deployed app then pulls this on boot / via POST /api/admin/reload.
"""

import argparse
import os
import sys

# allow running as a plain script (python scripts/publish_snapshot.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from webapp import snapshot

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.getenv("SNAPSHOT_URL", ""),
                    help="s3://bucket/prefix or file:///abs/dir (or SNAPSHOT_URL env)")
    ap.add_argument("--db", default=os.getenv("DB_PATH", "data/sports.db"))
    ap.add_argument("--artifacts", default=os.getenv("ARTIFACTS_DIR", "models/artifacts"))
    ap.add_argument("--ledger", default=os.getenv("LEDGER_PATH", "data/sims/paper_ledger.csv"))
    args = ap.parse_args()

    if not args.url:
        ap.error("no snapshot target: pass --url or set SNAPSHOT_URL")
    if not os.path.isfile(args.db):
        ap.error(f"DB not found: {args.db}")

    manifest = snapshot.publish(args.url, args.db, args.artifacts, args.ledger)
    print(f"Published snapshot {manifest['version']} "
          f"({manifest['bytes'] / 1e6:.1f} MB) to {args.url}")


if __name__ == "__main__":
    main()
