"""Value-bet endpoints: list edges (JSON) and export the current selection as CSV."""

import csv
import io

from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse
from starlette.routing import Route

from webapp import auth, settings
from webapp.services import edges as edges_svc
from webapp.util import json_response


def _parse_sports(raw: str | None) -> list[str]:
    if not raw:
        return settings.DEFAULT_SPORTS
    req = [s.strip().lower() for s in raw.split(",") if s.strip()]
    valid = [s for s in req if s in settings.SPORTS]
    return valid or settings.DEFAULT_SPORTS


def _qfloat(query, key: str, default: float) -> float:
    v = query.get(key)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _filter(all_edges: list[dict], threshold: float, markets: str | None) -> list[dict]:
    tokens = None
    if markets:
        tokens = [m.strip().lower() for m in markets.split(",") if m.strip()]

    def matches(e: dict) -> bool:
        if not tokens:
            return True
        m = str(e.get("market", "")).lower()
        return any(tok in m for tok in tokens)

    out = [e for e in all_edges if float(e.get("edge", 0) or 0) >= threshold and matches(e)]
    out.sort(key=lambda e: float(e.get("edge", 0) or 0), reverse=True)
    return out


async def _selected(request) -> tuple[list[dict], list[str], list[str], float]:
    q = request.query_params
    sports = _parse_sports(q.get("sports"))
    threshold = _qfloat(q, "min_edge", settings.MIN_EDGE)
    all_edges, errors = await run_in_threadpool(edges_svc.edges_for, sports)
    return _filter(all_edges, threshold, q.get("markets")), errors, sports, threshold


@auth.protected
async def get_edges(request):
    edges, errors, sports, threshold = await _selected(request)
    return json_response({
        "edges": edges,
        "count": len(edges),
        "errors": errors,
        "sports": sports,
        "min_edge": threshold,
    })


@auth.protected
async def export_edges_csv(request):
    edges, _, _, _ = await _selected(request)
    cols = ["sport", "market", "game", "bet", "odds", "edge", "kelly_stake"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for e in edges:
        writer.writerow({c: e.get(c, "") for c in cols})
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=edges.csv"},
    )


routes = [
    Route("/api/edges", get_edges, methods=["GET"]),
    Route("/api/edges/export.csv", export_edges_csv, methods=["GET"]),
]
