"""NBA data ingestion via nba_api. Writes to SQLite."""


def fetch_seasons(seasons: list[str], db_path: str) -> None:
    """Fetch game logs for given seasons (e.g. ['2023-24']) and store in DB."""
    raise NotImplementedError


def fetch_player_stats(season: str, db_path: str) -> None:
    """Fetch per-game player stats for a season and store in DB."""
    raise NotImplementedError


def fetch_recent_games(days_back: int, db_path: str) -> None:
    """Fetch games from the last N days (for incremental updates)."""
    raise NotImplementedError
