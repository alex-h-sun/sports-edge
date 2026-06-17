"""Admin endpoint: hot-reload the snapshot from storage without redeploying."""

import os

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route

from webapp import auth, settings, snapshot
from webapp.services import edges as edges_svc
from webapp.util import json_response


def _data_dir() -> str:
    return os.getenv("SNAPSHOT_DATA_DIR") or (os.path.dirname(settings.DB_PATH) or "data")


@auth.protected
async def reload_snapshot(request):
    url = os.getenv("SNAPSHOT_URL", "")
    force = request.query_params.get("force") in ("1", "true", "yes")
    try:
        result = await run_in_threadpool(snapshot.sync, url, _data_dir(), force)
    except Exception as e:
        return JSONResponse({"detail": f"snapshot sync failed: {e}"}, status_code=502)
    edges_svc.clear_cache()  # serve fresh edges from the new snapshot
    return json_response(result)


routes = [Route("/api/admin/reload", reload_snapshot, methods=["POST"])]
