"""Injury report + Odds API quota (read-only, from the snapshot DB)."""

import logging
import sqlite3

from webapp import settings

log = logging.getLogger("webapp.misc")


def injuries() -> dict:
    """Latest injury snapshot per sport from the DB (no ESPN call)."""
    from ingestion.injuries import get_current_injuries

    out: dict[str, list] = {}
    for sport in ("nba", "nhl"):
        try:
            out[sport] = get_current_injuries(sport, settings.DB_PATH)
        except Exception:
            log.warning("injury read failed for %s", sport, exc_info=True)
            out[sport] = []
    return out


def quota() -> dict | None:
    """Most recent Odds API quota row, or None if not tracked yet."""
    try:
        conn = sqlite3.connect(settings.DB_PATH)
        row = conn.execute(
            "SELECT remaining, used, fetched_at FROM odds_requests_remaining "
            "ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
    except sqlite3.Error:
        log.warning("quota read failed", exc_info=True)
        return None
    if not row:
        return None
    return {"remaining": row[0], "used": row[1], "fetched_at": row[2]}
