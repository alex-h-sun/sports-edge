"""Meta + health endpoints. /api/meta drives the SPA and doubles as the auth probe."""

import os
from datetime import datetime, timezone

from starlette.responses import JSONResponse
from starlette.routing import Route

from webapp import auth, settings
from webapp.util import json_response


async def health(request):
    return JSONResponse({"ok": True})


async def meta(request):
    if not auth.is_authenticated(request):
        return json_response(
            {"detail": "Not authenticated", "auth_required": True}, status_code=401
        )
    snapshot_ts = None
    try:
        snapshot_ts = datetime.fromtimestamp(
            os.path.getmtime(settings.DB_PATH), tz=timezone.utc
        ).isoformat()
    except OSError:
        pass
    return json_response({
        "db_label": os.path.basename(settings.DB_PATH),
        "bankroll": settings.BANKROLL,
        "min_edge": settings.MIN_EDGE,
        "sports": settings.SPORTS,
        "default_sports": settings.DEFAULT_SPORTS,
        "snapshot_ts": snapshot_ts,
        "auth_required": bool(settings.APP_PASSWORD),
    })


routes = [
    Route("/api/health", health, methods=["GET"]),
    Route("/api/meta", meta, methods=["GET"]),
]
