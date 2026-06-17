"""Bankroll simulator endpoint (equity summary + curve + ledger)."""

from starlette.concurrency import run_in_threadpool
from starlette.routing import Route

from webapp import auth
from webapp.services import bankroll as svc
from webapp.util import json_response


@auth.protected
async def get_bankroll(request):
    state = await run_in_threadpool(svc.bankroll_state)
    return json_response(state)


routes = [Route("/api/bankroll", get_bankroll, methods=["GET"])]
