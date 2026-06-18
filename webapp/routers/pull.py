"""POST /api/pull — actively ingest fresh games/odds and recompute edges.

This is the write counterpart to the read-only ``/api/admin/reload`` (which only pulls a
pre-built snapshot). It runs the same pipeline as ``python run.py`` against the shared DB,
so the web app is no longer limited to displaying a stale snapshot. Single-flight (a
module-level flag) keeps one slow pull from running twice concurrently or double-spending
the Odds API quota; the work runs in a threadpool so the event loop stays responsive.

Body (JSON, all optional): ``{"sports": ["nba","nhl"], "odds": true}``. ``sports``
defaults to the server's default sports; ``odds`` controls whether the (quota-spending)
Odds API fetch runs.
"""

from starlette.concurrency import run_in_threadpool
from starlette.routing import Route

from webapp import auth, settings
from webapp.services import pull as pull_svc
from webapp.util import json_response

# Single-flight guard. Async handlers share one event-loop thread and there is no `await`
# between the read and the write below, so this check-and-set is atomic (no lock needed).
_running = False


def _parse_sports(raw) -> list[str]:
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple)):
        return list(settings.DEFAULT_SPORTS)
    wanted = [str(s).strip().lower() for s in raw if str(s).strip()]
    valid = [s for s in wanted if s in settings.SPORTS]
    return valid or list(settings.DEFAULT_SPORTS)


@auth.protected
async def pull(request):
    global _running
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    sports = _parse_sports(body.get("sports"))
    fetch_odds = bool(body.get("odds", True))

    if _running:
        return json_response({"detail": "a data pull is already running"}, status_code=409)
    _running = True
    try:
        result = await run_in_threadpool(pull_svc.do_pull, sports, fetch_odds)
    finally:
        _running = False
    return json_response(result)


routes = [Route("/api/pull", pull, methods=["POST"])]
