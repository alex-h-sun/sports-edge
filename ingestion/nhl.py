"""NHL data ingestion via the NHL Stats API (api.nhle.com). Writes to SQLite."""


def fetch_seasons(seasons: list[str], db_path: str) -> None:
    """Fetch game logs for given seasons (e.g. ['20232024']) and store in DB."""
    raise NotImplementedError


def fetch_player_stats(season: str, db_path: str) -> None:
    """Fetch per-game skater/goalie stats for a season and store in DB."""
    raise NotImplementedError


def fetch_recent_games(days_back: int, db_path: str) -> None:
    """Fetch games from the last N days (for incremental updates)."""
    raise NotImplementedError
