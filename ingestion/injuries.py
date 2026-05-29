"""Injury report ingestion via ESPN's internal API. Writes to SQLite."""

import sqlite3
from datetime import datetime, timezone

import requests

_BASE = "https://site.api.espn.com/apis/site/v2/sports"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

SPORT_PATHS = {
    "nba": "basketball/nba",
    "nhl": "hockey/nhl",
}


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS injury_reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            sport           TEXT,
            team_name       TEXT,
            player_name     TEXT,
            position        TEXT,
            status          TEXT,
            injury_type     TEXT,
            short_comment   TEXT,
            reported_at     TEXT,
            fetched_at      TEXT
        )
    """)
    conn.commit()


def fetch_injuries(sport: str, db_path: str) -> list[dict]:
    """Pull current injury report from ESPN and store in DB.

    Returns the parsed list of injury dicts for immediate use
    (e.g. to build same-day features without re-reading the DB).
    """
    path = SPORT_PATHS[sport]
    resp = requests.get(f"{_BASE}/{path}/injuries", headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []

    for team in data.get("injuries", []):
        team_name = team["displayName"]
        for inj in team.get("injuries", []):
            athlete = inj.get("athlete", {})
            position = athlete.get("position", {})
            details = inj.get("details", {})
            rows.append({
                "sport":         sport,
                "team_name":     team_name,
                "player_name":   athlete.get("displayName", ""),
                "position":      position.get("abbreviation", "") if isinstance(position, dict) else "",
                "status":        inj.get("status", ""),
                "injury_type":   details.get("type", "") if isinstance(details, dict) else "",
                "short_comment": inj.get("shortComment", ""),
                "reported_at":   inj.get("date", ""),
                "fetched_at":    fetched_at,
            })

    conn = _get_conn(db_path)
    _ensure_tables(conn)
    conn.executemany("""
        INSERT INTO injury_reports
            (sport, team_name, player_name, position, status, injury_type,
             short_comment, reported_at, fetched_at)
        VALUES
            (:sport, :team_name, :player_name, :position, :status, :injury_type,
             :short_comment, :reported_at, :fetched_at)
    """, rows)
    conn.commit()
    conn.close()

    out_count = sum(1 for r in rows if r["status"] == "Out")
    dtd_count = sum(1 for r in rows if r["status"] == "Day-To-Day")
    print(f"  {sport.upper()} injuries: {len(rows)} players ({out_count} Out, {dtd_count} Day-To-Day)")
    return rows


def get_current_injuries(sport: str, db_path: str) -> list[dict]:
    """Return the most recent injury snapshot from the DB (no API call)."""
    conn = _get_conn(db_path)
    _ensure_tables(conn)
    latest = conn.execute(
        "SELECT MAX(fetched_at) FROM injury_reports WHERE sport = ?", (sport,)
    ).fetchone()[0]

    if not latest:
        conn.close()
        return []

    rows = conn.execute("""
        SELECT team_name, player_name, position, status, injury_type, short_comment
        FROM injury_reports WHERE sport = ? AND fetched_at = ?
    """, (sport, latest)).fetchall()
    conn.close()

    return [
        {"team_name": r[0], "player_name": r[1], "position": r[2],
         "status": r[3], "injury_type": r[4], "short_comment": r[5]}
        for r in rows
    ]
