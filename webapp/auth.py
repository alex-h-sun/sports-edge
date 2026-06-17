"""Single shared-password auth backed by a signed session cookie (SessionMiddleware).

There is no user table. `login` compares the posted password to APP_PASSWORD and, on
success, marks the session authenticated. If APP_PASSWORD is empty the gate is disabled
(local dev). Protected endpoints use the `@protected` decorator.
"""

import functools
import hmac
import time

from starlette.responses import JSONResponse

from webapp import settings

# crude per-IP throttle so the shared password can't be brute forced: ip -> (window_start, fails)
_attempts: dict[str, tuple[float, int]] = {}
_MAX_FAILS = 8
_WINDOW = 300.0


def _fails(ip: str) -> int:
    start, n = _attempts.get(ip, (0.0, 0))
    if time.time() - start > _WINDOW:
        return 0
    return n


def _record_failure(ip: str) -> None:
    now = time.time()
    start, n = _attempts.get(ip, (now, 0))
    if now - start > _WINDOW:
        start, n = now, 0
    _attempts[ip] = (start, n + 1)


def throttled(ip: str) -> bool:
    return _fails(ip) >= _MAX_FAILS


def check_password(candidate: str) -> bool:
    if not settings.APP_PASSWORD:
        return True  # gate disabled
    return hmac.compare_digest(candidate or "", settings.APP_PASSWORD)


def is_authenticated(request) -> bool:
    if not settings.APP_PASSWORD:
        return True
    return bool(request.session.get("auth"))


def login(request, password: str) -> bool:
    """Validate the password and, on success, mark the session authenticated.

    Returns True on success, False on a bad password. Caller maps that to 200/401.
    """
    if check_password(password):
        request.session["auth"] = True
        return True
    ip = request.client.host if request.client else "?"
    _record_failure(ip)
    return False


def logout(request) -> None:
    request.session.pop("auth", None)


def protected(fn):
    """Decorate an async Starlette endpoint so it 401s unless authenticated."""

    @functools.wraps(fn)
    async def wrapper(request):
        if not is_authenticated(request):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        return await fn(request)

    return wrapper
