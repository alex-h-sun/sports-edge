"""Live odds ingestion via The Odds API (the-odds-api.com). Writes to SQLite."""

import os
import sqlite3
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

_BASE = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "nba": "basketball_nba",
    "nhl": "icehockey_nhl",
}

# Market keys as used by The Odds API
MARKET_KEYS = {
    "moneyline": "h2h",
    "spread":    "spreads",
    "totals":    "totals",
}

# Player prop market keys (NBA)
NBA_PROP_KEYS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
]

# Player prop market keys (NHL)
NHL_PROP_KEYS = [
    "player_goals",
    "player_assists",
    "player_points",
    "player_shots_on_goal",
]

# Bookmakers to request (free tier supports all of these)
BOOKMAKERS = "draftkings,fanduel,betmgm,caesars"


def _get_api_key() -> str:
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise EnvironmentError("ODDS_API_KEY not set. Add it to your .env file.")
    return key


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            sport        TEXT,
            event_id     TEXT,
            home_team    TEXT,
            away_team    TEXT,
            commence_time TEXT,
            bookmaker    TEXT,
            market       TEXT,
            outcome_name TEXT,
            price        INTEGER,
            point        REAL,
            fetched_at   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_requests_remaining (
            fetched_at   TEXT PRIMARY KEY,
            remaining    INTEGER,
            used         INTEGER
        )
    """)
    conn.commit()


def _log_quota(conn: sqlite3.Connection, response: requests.Response) -> None:
    remaining = response.headers.get("x-requests-remaining")
    used = response.headers.get("x-requests-used")
    if remaining is not None:
        conn.execute(
            "INSERT OR REPLACE INTO odds_requests_remaining VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), int(remaining), int(used or 0))
        )
        conn.commit()
        print(f"    Odds API quota: {remaining} requests remaining")


def fetch_odds(sport: str, markets: list[str], db_path: str) -> None:
    """Pull current odds for a sport and list of markets, store in DB.

    sport:   'nba' or 'nhl'
    markets: subset of ['moneyline', 'spread', 'totals']
    """
    api_key = _get_api_key()
    sport_key = SPORT_KEYS[sport]
    market_param = ",".join(MARKET_KEYS[m] for m in markets if m in MARKET_KEYS)

    conn = _get_conn(db_path)
    _ensure_tables(conn)

    print(f"  Fetching {sport.upper()} odds ({', '.join(markets)})...", end=" ", flush=True)
    resp = requests.get(
        f"{_BASE}/sports/{sport_key}/odds",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": market_param,
            "bookmakers": BOOKMAKERS,
            "oddsFormat": "american",
        },
        timeout=15,
    )
    resp.raise_for_status()
    _log_quota(conn, resp)

    events = resp.json()
    rows = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    for event in events:
        for book in event.get("bookmakers", []):
            for mkt in book.get("markets", []):
                for outcome in mkt.get("outcomes", []):
                    rows.append((
                        sport,
                        event["id"],
                        event["home_team"],
                        event["away_team"],
                        event["commence_time"],
                        book["key"],
                        mkt["key"],
                        outcome["name"],
                        outcome["price"],
                        outcome.get("point"),
                        fetched_at,
                    ))

    conn.executemany(
        "INSERT INTO odds_snapshots VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    print(f"{len(events)} games, {len(rows)} odds rows")
    conn.close()


def fetch_props(sport: str, db_path: str) -> None:
    """Pull player prop odds for a sport. Uses one API request per prop market."""
    api_key = _get_api_key()
    sport_key = SPORT_KEYS[sport]
    prop_keys = NBA_PROP_KEYS if sport == "nba" else NHL_PROP_KEYS

    conn = _get_conn(db_path)
    _ensure_tables(conn)

    # First get the list of event IDs for today's games
    resp = requests.get(
        f"{_BASE}/sports/{sport_key}/odds",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "bookmakers": "draftkings",
            "oddsFormat": "american",
        },
        timeout=15,
    )
    resp.raise_for_status()
    _log_quota(conn, resp)

    event_ids = [e["id"] for e in resp.json()]
    if not event_ids:
        print(f"  No upcoming {sport.upper()} events found for props")
        conn.close()
        return

    print(f"  Fetching {sport.upper()} props for {len(event_ids)} games...")
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []

    for event_id in event_ids:
        for prop_key in prop_keys:
            try:
                resp = requests.get(
                    f"{_BASE}/sports/{sport_key}/events/{event_id}/odds",
                    params={
                        "apiKey": api_key,
                        "regions": "us",
                        "markets": prop_key,
                        "bookmakers": BOOKMAKERS,
                        "oddsFormat": "american",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                _log_quota(conn, resp)
                event = resp.json()

                for book in event.get("bookmakers", []):
                    for mkt in book.get("markets", []):
                        for outcome in mkt.get("outcomes", []):
                            rows.append((
                                sport,
                                event_id,
                                event.get("home_team", ""),
                                event.get("away_team", ""),
                                event.get("commence_time", ""),
                                book["key"],
                                mkt["key"],
                                outcome["name"],
                                outcome["price"],
                                outcome.get("point"),
                                fetched_at,
                            ))
            except Exception as e:
                print(f"    warning: {event_id}/{prop_key} failed ({e})")
            time.sleep(0.2)

    conn.executemany(
        "INSERT INTO odds_snapshots VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    print(f"  Props done: {len(rows)} rows")
    conn.close()


def get_quota(db_path: str) -> None:
    """Print current API quota usage from the last stored snapshot."""
    conn = _get_conn(db_path)
    _ensure_tables(conn)
    row = conn.execute(
        "SELECT remaining, used, fetched_at FROM odds_requests_remaining ORDER BY fetched_at DESC LIMIT 1"
    ).fetchone()
    if row:
        print(f"  Last checked {row[2]}: {row[0]} requests remaining, {row[1]} used")
    else:
        print("  No quota data yet — run fetch_odds first")
    conn.close()
