"""Injury report + Odds API quota endpoints."""

from starlette.concurrency import run_in_threadpool
from starlette.routing import Route

from webapp import auth
from webapp.services import misc as svc
from webapp.util import json_response


@auth.protected
async def get_injuries(request):
    data = await run_in_threadpool(svc.injuries)
    return json_response(data)


@auth.protected
async def get_quota(request):
    data = await run_in_threadpool(svc.quota)
    return json_response({"quota": data})


routes = [
    Route("/api/injuries", get_injuries, methods=["GET"]),
    Route("/api/quota", get_quota, methods=["GET"]),
]
