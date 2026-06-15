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

# The Odds API splits tennis by tournament. These are the active ATP/WTA keys;
# inactive tournaments simply return no events (and cost no quota beyond the call).
TENNIS_SPORT_KEYS = [
    # ATP Grand Slams
    "tennis_atp_aus_open_singles", "tennis_atp_french_open", "tennis_atp_wimbledon",
    "tennis_atp_us_open",
    # ATP 1000 (Masters)
    "tennis_atp_indian_wells", "tennis_atp_miami_open", "tennis_atp_monte_carlo_masters",
    "tennis_atp_madrid_open", "tennis_atp_italian_open", "tennis_atp_canadian_open",
    "tennis_atp_cincinnati_open", "tennis_atp_shanghai_masters", "tennis_atp_paris_masters",
    # ATP 500
    "tennis_atp_rotterdam", "tennis_atp_rio_open", "tennis_atp_dubai", "tennis_atp_acapulco",
    "tennis_atp_barcelona", "tennis_atp_halle", "tennis_atp_queens_club",
    "tennis_atp_hamburg", "tennis_atp_washington", "tennis_atp_tokyo",
    "tennis_atp_vienna", "tennis_atp_swiss_indoors",
    # ATP 250 (selected)
    "tennis_atp_qatar_open",
    # WTA Grand Slams
    "tennis_wta_aus_open_singles", "tennis_wta_french_open", "tennis_wta_wimbledon",
    "tennis_wta_us_open",
    # WTA 1000
    "tennis_wta_indian_wells", "tennis_wta_miami_open", "tennis_wta_madrid_open",
    "tennis_wta_italian_open", "tennis_wta_canadian_open", "tennis_wta_cincinnati_open",
    "tennis_wta_china_open", "tennis_wta_wuhan_open", "tennis_wta_guadalajara",
    # WTA 500
    "tennis_wta_adelaide", "tennis_wta_abu_dhabi", "tennis_wta_dubai", "tennis_wta_qatar_open",
    "tennis_wta_stuttgart", "tennis_wta_berlin", "tennis_wta_eastbourne",
    "tennis_wta_washington", "tennis_wta_san_jose", "tennis_wta_seoul", "tennis_wta_linz",
    # WTA 250 (selected)
    "tennis_wta_queens_club_champ",
]


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


def fetch_tennis_odds(db_path: str, markets: list[str] | None = None) -> None:
    """Pull current tennis odds across all active tournament keys.

    Writes rows with sport='tennis' into the shared odds_snapshots table. The Odds
    API exposes tennis per-tournament, so we loop the keys; inactive ones return [].
    """
    api_key = _get_api_key()
    markets = markets or ["moneyline", "spread", "totals"]
    market_param = ",".join(MARKET_KEYS[m] for m in markets if m in MARKET_KEYS)

    conn = _get_conn(db_path)
    _ensure_tables(conn)

    fetched_at = datetime.now(timezone.utc).isoformat()
    total_events = total_rows = 0

    for sport_key in TENNIS_SPORT_KEYS:
        try:
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
            if resp.status_code in (404, 422):
                continue  # tournament not currently offered / no upcoming events
            resp.raise_for_status()
        except Exception as e:
            print(f"    warning: {sport_key} failed ({e})")
            continue

        _log_quota(conn, resp)
        events = resp.json()
        rows = []
        for event in events:
            for book in event.get("bookmakers", []):
                for mkt in book.get("markets", []):
                    for outcome in mkt.get("outcomes", []):
                        rows.append((
                            "tennis", event["id"], event.get("home_team", ""),
                            event.get("away_team", ""), event.get("commence_time", ""),
                            book["key"], mkt["key"], outcome["name"],
                            outcome["price"], outcome.get("point"), fetched_at,
                        ))
        if rows:
            conn.executemany(
                "INSERT INTO odds_snapshots VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            conn.commit()
        total_events += len(events)
        total_rows += len(rows)
        time.sleep(0.2)

    print(f"  Tennis odds: {total_events} matches, {total_rows} odds rows")
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
