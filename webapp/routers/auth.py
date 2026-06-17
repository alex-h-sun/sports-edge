"""Auth endpoints: shared-password login / logout."""

from starlette.responses import JSONResponse
from starlette.routing import Route

from webapp import auth


async def login(request):
    ip = request.client.host if request.client else "?"
    if auth.throttled(ip):
        return JSONResponse({"detail": "Too many attempts, try again later"}, status_code=429)
    try:
        body = await request.json()
    except Exception:
        body = {}
    if auth.login(request, (body or {}).get("password", "")):
        return JSONResponse({"ok": True})
    return JSONResponse({"detail": "Invalid password"}, status_code=401)


async def logout(request):
    auth.logout(request)
    return JSONResponse({"ok": True})


routes = [
    Route("/api/auth/login", login, methods=["POST"]),
    Route("/api/auth/logout", logout, methods=["POST"]),
]
