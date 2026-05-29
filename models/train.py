"""Train LightGBM models for each market. Saves model artifacts to disk."""

import polars as pl


def train_moneyline(df: pl.DataFrame, save_path: str) -> None:
    """Binary classifier: home team win probability."""
    raise NotImplementedError


def train_spread(df: pl.DataFrame, save_path: str) -> None:
    """Regressor: predicted margin of victory."""
    raise NotImplementedError


def train_totals(df: pl.DataFrame, save_path: str) -> None:
    """Regressor: predicted total points scored."""
    raise NotImplementedError


def train_player_prop(stat: str, df: pl.DataFrame, save_path: str) -> None:
    """Regressor for a single player stat (e.g. 'points', 'assists')."""
    raise NotImplementedError
