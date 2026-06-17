"""Starlette application: JSON API + the built React SPA.

The heavy pipeline (run.py) stays offline and publishes a snapshot of sports.db +
artifacts; this app reads that snapshot and serves edges. Run it with::

    uvicorn webapp.main:app --host 0.0.0.0 --port 8000
"""

import logging

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from webapp import settings
from webapp.routers import admin as admin_routes
from webapp.routers import auth as auth_routes
from webapp.routers import bankroll as bankroll_routes
from webapp.routers import edges as edges_routes
from webapp.routers import manual as manual_routes
from webapp.routers import meta as meta_routes
from webapp.routers import misc as misc_routes

logging.basicConfig(level=logging.INFO)


def _spa_routes() -> list:
    """Serve the built SPA (assets + index.html history fallback) when present.
    API paths still resolve to JSON 404s rather than index.html."""
    if not settings.STATIC_DIR.is_dir():
        return []
    index = settings.STATIC_DIR / "index.html"
    assets = settings.STATIC_DIR / "assets"

    async def spa(request):
        path = request.path_params.get("path", "")
        if path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        if index.is_file():
            return FileResponse(index)
        return JSONResponse({"detail": "SPA not built"}, status_code=404)

    out: list = []
    if assets.is_dir():
        out.append(Mount("/assets", app=StaticFiles(directory=str(assets)), name="assets"))
    out.append(Route("/{path:path}", spa, methods=["GET"]))
    return out


def create_app() -> Starlette:
    routes = [
        *auth_routes.routes,
        *meta_routes.routes,
        *edges_routes.routes,
        *manual_routes.routes,
        *bankroll_routes.routes,
        *misc_routes.routes,
        *admin_routes.routes,
        *_spa_routes(),
    ]
    middleware = [
        Middleware(
            SessionMiddleware,
            secret_key=settings.SESSION_SECRET or "dev-insecure-change-me",
            https_only=settings.IS_PROD,
            same_site="lax",
        ),
    ]
    return Starlette(routes=routes, middleware=middleware)


app = create_app()
