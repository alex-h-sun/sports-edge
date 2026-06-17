"""Manual edge calculator endpoints: dropdown options + compute edge for a matchup."""

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route

from webapp import auth
from webapp.services import manual as svc
from webapp.util import json_response


@auth.protected
async def get_options(request):
    sport = request.query_params.get("sport", "nba")
    data = await run_in_threadpool(svc.options, sport)
    return json_response(data)


@auth.protected
async def post_edge(request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid JSON body"}, status_code=400)
    try:
        edges = await run_in_threadpool(svc.compute, body or {})
    except KeyError as e:
        return JSONResponse({"detail": f"missing field: {e.args[0]}"}, status_code=400)
    except (ValueError, TypeError, FileNotFoundError) as e:
        return JSONResponse({"detail": str(e) or "Invalid request"}, status_code=400)
    return json_response({"edges": edges, "count": len(edges)})


routes = [
    Route("/api/manual/options", get_options, methods=["GET"]),
    Route("/api/manual/edge", post_edge, methods=["POST"]),
]
