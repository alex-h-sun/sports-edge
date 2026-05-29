"""Model evaluation: calibration, Brier score, AUC, ROI simulation."""

import polars as pl


def evaluate_classifier(y_true, y_prob) -> dict:
    """Return AUC, Brier score, and log loss for a binary classifier."""
    raise NotImplementedError


def evaluate_regressor(y_true, y_pred) -> dict:
    """Return MAE, RMSE, and R² for a regression model."""
    raise NotImplementedError


def simulate_roi(predictions: pl.DataFrame, kelly_fraction: float = 0.25) -> dict:
    """Backtest flat/Kelly betting on historical predictions. Returns ROI stats."""
    raise NotImplementedError
