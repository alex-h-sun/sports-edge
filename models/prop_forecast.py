"""Torch-free serving hook for the cloud-trained global prop forecaster (Tier 4).

The heavy probabilistic forecaster (DeepAR / Temporal Fusion Transformer) is
trained on Colab GPU — see notebooks/props_forecast_colab.ipynb and
docs/TRAINING_PROPS_FORECAST.md. It learns one global model across *all* players
and emits, for each player's next game, a predictive distribution per stat.

Following the same "portable winner served locally" rule as the bake-off, the
notebook does not ship a torch model for serving. It distills its next-game
predictions into a plain table:

    artifacts/{sport}_prop_forecast.pkl  ->  {
        "sport": str,
        "asof": "YYYY-MM-DD",            # last date seen in training
        "table": polars.DataFrame[player_id, stat, pred_mean, pred_sigma],
    }

Local edge detection consumes that table with zero torch dependency. Everything
here degrades to a no-op when the artifact is absent, so the pipeline keeps
working before the first cloud run.
"""

import pickle
from pathlib import Path

import polars as pl

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_prop_forecast(sport: str) -> dict | None:
    """Load the cloud-trained forecast table for a sport, or None if not present."""
    path = ARTIFACTS_DIR / f"{sport}_prop_forecast.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def prop_forecast_features(df: pl.DataFrame, sport: str) -> pl.DataFrame:
    """Join the forecaster's per-(player, stat) mean + sigma onto player rows.

    Adds, for each stat the forecaster covers, columns `fc_{stat}_mean` and
    `fc_{stat}_sigma`. These give the props edge path a probabilistic prediction
    (distribution, not just a point estimate) to compare against over/under
    lines. No-op (returns df unchanged) until the cloud artifact is imported.
    """
    art = load_prop_forecast(sport)
    if art is None or "player_id" not in df.columns:
        return df

    table = art["table"]
    if table.is_empty():
        return df

    wide = table.pivot(on="stat", index="player_id", values=["pred_mean", "pred_sigma"])
    # polars names pivoted value columns "pred_mean_{stat}" / "pred_sigma_{stat}";
    # normalise to the documented fc_{stat}_{mean,sigma} feature names
    renames = {}
    for c in wide.columns:
        if c.startswith("pred_mean_"):
            renames[c] = f"fc_{c[len('pred_mean_'):]}_mean"
        elif c.startswith("pred_sigma_"):
            renames[c] = f"fc_{c[len('pred_sigma_'):]}_sigma"
    wide = wide.rename(renames)

    return df.join(wide, on="player_id", how="left")
