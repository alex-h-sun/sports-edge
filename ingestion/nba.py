"""NBA data ingestion via nba_api. Writes to SQLite."""

import sqlite3
import time
from datetime import date, timedelta

import polars as pl
from nba_api.stats.endpoints import leaguegamelog, playergamelogs

# nba_api is rate-limited; 0.6s between calls is safe
_RATE_LIMIT = 0.6

TEAM_GAME_COLS = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
]

PLAYER_GAME_COLS = [
    "SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "TOV", "STL", "BLK", "PF", "PTS", "PLUS_MINUS",
]


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nba_team_games (
            game_id     TEXT,
            team_id     INTEGER,
            season_id   TEXT,
            game_date   TEXT,
            matchup     TEXT,
            wl          TEXT,
            team_abbr   TEXT,
            team_name   TEXT,
            fgm INTEGER, fga INTEGER, fg_pct REAL,
            fg3m INTEGER, fg3a INTEGER, fg3_pct REAL,
            ftm INTEGER, fta INTEGER, ft_pct REAL,
            oreb INTEGER, dreb INTEGER, reb INTEGER,
            ast INTEGER, stl INTEGER, blk INTEGER,
            tov INTEGER, pf INTEGER, pts INTEGER, plus_minus INTEGER,
            PRIMARY KEY (game_id, team_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nba_player_games (
            game_id     TEXT,
            player_id   INTEGER,
            season_year TEXT,
            game_date   TEXT,
            matchup     TEXT,
            wl          TEXT,
            player_name TEXT,
            team_id     INTEGER,
            team_abbr   TEXT,
            min         TEXT,
            fgm INTEGER, fga INTEGER, fg_pct REAL,
            fg3m INTEGER, fg3a INTEGER, fg3_pct REAL,
            ftm INTEGER, fta INTEGER, ft_pct REAL,
            oreb INTEGER, dreb INTEGER, reb INTEGER,
            ast INTEGER, tov INTEGER, stl INTEGER, blk INTEGER,
            pf INTEGER, pts INTEGER, plus_minus INTEGER,
            PRIMARY KEY (game_id, player_id)
        )
    """)
    conn.commit()


def _upsert_team_games(conn: sqlite3.Connection, df: pl.DataFrame) -> int:
    rows = df.select(TEAM_GAME_COLS).to_dicts()
    conn.executemany("""
        INSERT OR REPLACE INTO nba_team_games VALUES (
            :GAME_ID, :TEAM_ID, :SEASON_ID, :GAME_DATE, :MATCHUP, :WL,
            :TEAM_ABBREVIATION, :TEAM_NAME,
            :FGM, :FGA, :FG_PCT, :FG3M, :FG3A, :FG3_PCT,
            :FTM, :FTA, :FT_PCT, :OREB, :DREB, :REB,
            :AST, :STL, :BLK, :TOV, :PF, :PTS, :PLUS_MINUS
        )
    """, rows)
    conn.commit()
    return len(rows)


def _upsert_player_games(conn: sqlite3.Connection, df: pl.DataFrame) -> int:
    rows = df.select(PLAYER_GAME_COLS).to_dicts()
    conn.executemany("""
        INSERT OR REPLACE INTO nba_player_games VALUES (
            :GAME_ID, :PLAYER_ID, :SEASON_YEAR, :GAME_DATE, :MATCHUP, :WL,
            :PLAYER_NAME, :TEAM_ID, :TEAM_ABBREVIATION, :MIN,
            :FGM, :FGA, :FG_PCT, :FG3M, :FG3A, :FG3_PCT,
            :FTM, :FTA, :FT_PCT, :OREB, :DREB, :REB,
            :AST, :TOV, :STL, :BLK, :PF, :PTS, :PLUS_MINUS
        )
    """, rows)
    conn.commit()
    return len(rows)


def fetch_seasons(seasons: list[str], db_path: str) -> None:
    """Fetch team game logs for given seasons (e.g. ['2022-23', '2023-24']) and store in DB."""
    conn = _get_conn(db_path)
    _ensure_tables(conn)

    for season in seasons:
        for season_type in ("Regular Season", "Playoffs"):
            print(f"  NBA team games: {season} {season_type}...", end=" ", flush=True)
            resp = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star=season_type
            )
            df = pl.from_pandas(resp.get_data_frames()[0])
            if df.is_empty():
                print("no data")
                continue
            n = _upsert_team_games(conn, df)
            print(f"{n} rows")
            time.sleep(_RATE_LIMIT)

    conn.close()


def fetch_player_stats(seasons: list[str], db_path: str) -> None:
    """Fetch per-game player stats for given seasons and store in DB."""
    conn = _get_conn(db_path)
    _ensure_tables(conn)

    for season in seasons:
        for season_type in ("Regular Season", "Playoffs"):
            print(f"  NBA player games: {season} {season_type}...", end=" ", flush=True)
            resp = playergamelogs.PlayerGameLogs(
                season_nullable=season, season_type_nullable=season_type
            )
            df = pl.from_pandas(resp.get_data_frames()[0])
            if df.is_empty():
                print("no data")
                continue
            n = _upsert_player_games(conn, df)
            print(f"{n} rows")
            time.sleep(_RATE_LIMIT)

    conn.close()


def fetch_recent_games(days_back: int, db_path: str) -> None:
    """Fetch team + player games from the last N days (incremental update)."""
    date_from = (date.today() - timedelta(days=days_back)).strftime("%m/%d/%Y")
    date_to = date.today().strftime("%m/%d/%Y")

    conn = _get_conn(db_path)
    _ensure_tables(conn)

    # Team games
    print(f"  NBA recent team games ({days_back}d)...", end=" ", flush=True)
    resp = leaguegamelog.LeagueGameLog(
        season="2024-25",
        season_type_all_star="Regular Season",
        date_from_nullable=date_from,
        date_to_nullable=date_to,
    )
    df = pl.from_pandas(resp.get_data_frames()[0])
    n = _upsert_team_games(conn, df) if not df.is_empty() else 0
    print(f"{n} rows")
    time.sleep(_RATE_LIMIT)

    # Player games
    print(f"  NBA recent player games ({days_back}d)...", end=" ", flush=True)
    resp = playergamelogs.PlayerGameLogs(
        season_nullable="2024-25",
        season_type_nullable="Regular Season",
        date_from_nullable=date_from,
        date_to_nullable=date_to,
    )
    df = pl.from_pandas(resp.get_data_frames()[0])
    n = _upsert_player_games(conn, df) if not df.is_empty() else 0
    print(f"{n} rows")

    conn.close()
