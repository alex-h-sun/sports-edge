"""Feature engineering pipeline. Reads from SQLite, returns Polars DataFrames."""

import polars as pl


def build_game_features(sport: str, db_path: str) -> pl.DataFrame:
    """Build team-level features per game: rolling averages, rest, home/away."""
    raise NotImplementedError


def build_player_features(sport: str, db_path: str) -> pl.DataFrame:
    """Build player-level features per game for props modeling."""
    raise NotImplementedError


def add_odds_features(df: pl.DataFrame, db_path: str) -> pl.DataFrame:
    """Join closing/opening line movement onto a game feature DataFrame."""
    raise NotImplementedError
