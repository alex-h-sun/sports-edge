"""Read-only edge finding for the web API.

Mirrors ``run.find_edges`` but never ingests: it builds features from the snapshot
DB and scores the loaded ``.pkl`` artifacts. Because rebuilding features scans the
game-log tables (a few seconds), results are cached in-process per sport with a short
TTL. The cache holds the *unfiltered* edge list (computed at min_edge 0); the API
applies the min-edge / market / sport filters afterwards, exactly like dashboard.py.
"""

import logging
import threading
import time

from webapp import settings

log = logging.getLogger("webapp.edges")

_lock = threading.Lock()
_cache: dict[str, tuple[float, list[dict]]] = {}


def _compute(sport: str) -> list[dict]:
    """All edges for one sport, no min-edge filter. May raise FileNotFoundError when
    that sport's artifacts have not been imported yet."""
    db = settings.DB_PATH

    if sport == "tennis":
        from edge.calculator import find_tennis_edges
        return find_tennis_edges(db, min_edge=0.0, bankroll=settings.BANKROLL)

    from features.pipeline import (
        build_nba_game_features, build_nba_player_features,
        build_nhl_game_features, build_nhl_player_features,
        add_injury_features, add_teammate_wowy_features, add_market_features,
    )
    from edge.calculator import find_moneyline_edges, find_totals_edges, find_prop_edges

    if sport == "nba":
        game_df = build_nba_game_features(db)
        player_df = build_nba_player_features(db)
    else:
        game_df = build_nhl_game_features(db)
        player_df = build_nhl_player_features(db)

    game_df = add_injury_features(game_df, sport, db)
    game_df = add_market_features(game_df, sport, db)
    player_df = add_teammate_wowy_features(player_df, sport, db)

    edges: list[dict] = []
    edges += find_moneyline_edges(game_df, sport, db, min_edge=0.0, bankroll=settings.BANKROLL)
    edges += find_totals_edges(game_df, sport, db, bankroll=settings.BANKROLL)
    edges += find_prop_edges(player_df, sport, db, bankroll=settings.BANKROLL)
    return edges


def _cached(sport: str) -> tuple[list[dict], str | None]:
    """Return (edges, error) for one sport, using the TTL cache."""
    now = time.time()
    with _lock:
        hit = _cache.get(sport)
        if hit and now - hit[0] < settings.EDGES_TTL:
            return hit[1], None

    try:
        edges = _compute(sport)
    except FileNotFoundError as e:
        return [], f"{sport}: model not trained ({e})"
    except ValueError as e:
        return [], f"{sport}: {e}"
    except Exception as e:  # keep the API up; surface the message to the caller + logs
        log.exception("edge computation failed for %s", sport)
        return [], f"{sport}: {e}"

    with _lock:
        _cache[sport] = (time.time(), edges)
    return edges, None


def edges_for(sports: list[str]) -> tuple[list[dict], list[str]]:
    """Aggregate unfiltered edges across the requested sports. Returns (edges, errors)."""
    out: list[dict] = []
    errors: list[str] = []
    for sport in sports:
        edges, err = _cached(sport)
        out.extend(edges)
        if err:
            errors.append(err)
    return out, errors


def clear_cache() -> None:
    """Drop all cached edges (called after a snapshot reload)."""
    with _lock:
        _cache.clear()
