"""NHL data ingestion via the NHL Stats API (api-web.nhle.com). Writes to SQLite."""

import sqlite3
import time
from datetime import date, timedelta

import requests
import polars as pl

_BASE = "https://api-web.nhle.com/v1"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_RATE_LIMIT = 0.3  # seconds between requests

# gameType 2 = regular season, 3 = playoffs
_GAME_TYPES = {2: "Regular Season", 3: "Playoffs"}

CURRENT_SEASON = "20252026"

# All 32 active franchises as of 2025-26
ACTIVE_TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]


def _get(url: str) -> dict:
    resp = requests.get(url, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nhl_team_games (
            game_id     INTEGER,
            team_id     INTEGER,
            season      INTEGER,
            game_type   INTEGER,
            game_date   TEXT,
            home_away   TEXT,
            opp_team_id INTEGER,
            wl          TEXT,
            goals       INTEGER,
            opp_goals   INTEGER,
            shots       INTEGER,
            opp_shots   INTEGER,
            PRIMARY KEY (game_id, team_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nhl_player_games (
            game_id     INTEGER,
            player_id   INTEGER,
            season      INTEGER,
            game_type   INTEGER,
            game_date   TEXT,
            team_id     INTEGER,
            position    TEXT,
            goals       INTEGER,
            assists     INTEGER,
            points      INTEGER,
            plus_minus  INTEGER,
            pim         INTEGER,
            hits        INTEGER,
            shots       INTEGER,
            blocked_shots INTEGER,
            pp_goals    INTEGER,
            faceoff_pct REAL,
            toi         TEXT,
            PRIMARY KEY (game_id, player_id)
        )
    """)
    conn.commit()


def _game_ids_for_season(season: str, game_type: int) -> list[int]:
    """Return all game IDs for a season/type by pulling one team's schedule and deduplicating."""
    seen: set[int] = set()
    game_ids: list[int] = []
    # Pull from all teams so we get every game even if one team's schedule is incomplete
    for abbrev in ACTIVE_TEAMS:
        url = f"{_BASE}/club-schedule-season/{abbrev}/{season}"
        try:
            data = _get(url)
        except Exception:
            continue
        for g in data.get("games", []):
            gid = g["id"]
            if g["gameType"] == game_type and g["gameState"] in ("OFF", "FINAL") and gid not in seen:
                seen.add(gid)
                game_ids.append(gid)
        time.sleep(_RATE_LIMIT)
    return game_ids


def _parse_boxscore(game_id: int) -> tuple[list[dict], list[dict]]:
    """Return (team_rows, player_rows) parsed from a boxscore."""
    data = _get(f"{_BASE}/gamecenter/{game_id}/boxscore")

    season = data["season"]
    game_type = data["gameType"]
    game_date = data["gameDate"]
    home = data["homeTeam"]
    away = data["awayTeam"]

    home_goals = home.get("score", 0)
    away_goals = away.get("score", 0)
    home_shots = home.get("sog", 0)
    away_shots = away.get("sog", 0)

    team_rows = [
        {
            "game_id": game_id, "team_id": home["id"], "season": season,
            "game_type": game_type, "game_date": game_date,
            "home_away": "H", "opp_team_id": away["id"],
            "wl": "W" if home_goals > away_goals else "L",
            "goals": home_goals, "opp_goals": away_goals,
            "shots": home_shots, "opp_shots": away_shots,
        },
        {
            "game_id": game_id, "team_id": away["id"], "season": season,
            "game_type": game_type, "game_date": game_date,
            "home_away": "A", "opp_team_id": home["id"],
            "wl": "W" if away_goals > home_goals else "L",
            "goals": away_goals, "opp_goals": home_goals,
            "shots": away_shots, "opp_shots": home_shots,
        },
    ]

    player_rows: list[dict] = []
    pgs = data.get("playerByGameStats", {})
    for side_key, team in (("homeTeam", home), ("awayTeam", away)):
        side = pgs.get(side_key, {})
        for group in ("forwards", "defense"):
            for p in side.get(group, []):
                player_rows.append({
                    "game_id": game_id, "player_id": p["playerId"],
                    "season": season, "game_type": game_type,
                    "game_date": game_date, "team_id": team["id"],
                    "position": p.get("position", ""),
                    "goals": p.get("goals", 0), "assists": p.get("assists", 0),
                    "points": p.get("points", 0), "plus_minus": p.get("plusMinus", 0),
                    "pim": p.get("pim", 0), "hits": p.get("hits", 0),
                    "shots": p.get("sog", 0), "blocked_shots": p.get("blockedShots", 0),
                    "pp_goals": p.get("powerPlayGoals", 0),
                    "faceoff_pct": p.get("faceoffWinningPctg", None),
                    "toi": p.get("toi", ""),
                })
    return team_rows, player_rows


def _upsert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})", rows
    )
    conn.commit()


def fetch_seasons(seasons: list[str], db_path: str) -> None:
    """Fetch team + player game logs for given seasons (e.g. ['20242025', '20252026'])."""
    conn = _get_conn(db_path)
    _ensure_tables(conn)

    for season in seasons:
        for game_type, label in _GAME_TYPES.items():
            print(f"  NHL {season} {label}: collecting game IDs...", end=" ", flush=True)
            game_ids = _game_ids_for_season(season, game_type)
            print(f"{len(game_ids)} games")

            team_buf, player_buf = [], []
            for i, gid in enumerate(game_ids, 1):
                try:
                    t_rows, p_rows = _parse_boxscore(gid)
                    team_buf.extend(t_rows)
                    player_buf.extend(p_rows)
                except Exception as e:
                    print(f"    warning: game {gid} failed ({e})")
                if i % 50 == 0:
                    _upsert(conn, "nhl_team_games", team_buf)
                    _upsert(conn, "nhl_player_games", player_buf)
                    team_buf, player_buf = [], []
                    print(f"    ...{i}/{len(game_ids)}")
                time.sleep(_RATE_LIMIT)

            _upsert(conn, "nhl_team_games", team_buf)
            _upsert(conn, "nhl_player_games", player_buf)
            print(f"    done: {len(game_ids) * 2} team rows, {len(player_buf)} player rows saved")

    conn.close()


def fetch_recent_games(days_back: int, db_path: str, season: str = CURRENT_SEASON) -> None:
    """Fetch games from the last N days (incremental update)."""
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()

    conn = _get_conn(db_path)
    _ensure_tables(conn)

    seen: set[int] = set()
    team_rows, player_rows = [], []

    for abbrev in ACTIVE_TEAMS:
        url = f"{_BASE}/club-schedule-season/{abbrev}/{season}"
        try:
            data = _get(url)
        except Exception:
            continue
        for g in data.get("games", []):
            gid = g["id"]
            if (g["gameType"] in _GAME_TYPES
                    and g["gameState"] in ("OFF", "FINAL")
                    and g["gameDate"] >= cutoff
                    and gid not in seen):
                seen.add(gid)
                try:
                    t, p = _parse_boxscore(gid)
                    team_rows.extend(t)
                    player_rows.extend(p)
                except Exception as e:
                    print(f"  warning: game {gid} failed ({e})")
                time.sleep(_RATE_LIMIT)
        time.sleep(_RATE_LIMIT)

    _upsert(conn, "nhl_team_games", team_rows)
    _upsert(conn, "nhl_player_games", player_rows)
    print(f"  NHL recent ({days_back}d): {len(seen)} games, {len(team_rows)} team rows, {len(player_rows)} player rows")
    conn.close()
