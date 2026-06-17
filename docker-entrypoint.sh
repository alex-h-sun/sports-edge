#!/bin/sh
set -e

# Pull the latest published snapshot onto the volume before serving. This is a no-op
# when SNAPSHOT_URL is empty (manually-managed volume) and tolerates transient
# failures so the app still boots on the last good snapshot already on disk.
mkdir -p "${SNAPSHOT_DATA_DIR:-/data}"
if [ -n "${SNAPSHOT_URL:-}" ]; then
  python -m webapp.snapshot --sync || echo "snapshot sync failed; serving existing volume contents"
fi

exec uvicorn webapp.main:app --host 0.0.0.0 --port "${PORT:-8000}"
