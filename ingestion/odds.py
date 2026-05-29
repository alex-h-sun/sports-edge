"""Live odds ingestion via The Odds API (the-odds-api.com)."""

SPORT_KEYS = {
    "nba": "basketball_nba",
    "nhl": "icehockey_nhl",
}

MARKETS = ["h2h", "spreads", "totals", "player_props"]


def fetch_odds(sport: str, markets: list[str], db_path: str) -> None:
    """Pull current odds for a sport/market combo and store in DB.

    sport: 'nba' or 'nhl'
    markets: subset of MARKETS
    """
    raise NotImplementedError


def fetch_historical_odds(sport: str, date: str, db_path: str) -> None:
    """Pull historical odds snapshot for a given date (paid tier)."""
    raise NotImplementedError
