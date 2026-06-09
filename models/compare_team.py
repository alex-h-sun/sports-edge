"""NBA / NHL team moneyline model bake-off.

Mirror of the tennis bake-off in `models/compare.py`, applied to the team-level
game features (home-win prediction). Reuses the per-family fold trainers and
portable-model helpers from `models.compare` so there is exactly one
implementation of LightGBM / XGBoost / MLP / logistic.

Runs identically locally (CPU) or on Colab GPU:
  - import-safe, no heavy work at import time
  - device-agnostic MLP (CUDA if available, else CPU)
  - optional deps (torch, xgboost) skipped with a printed note if absent
  - accepts an explicit features path (parquet) or a SQLite db_path

Serving note: the production artifact {sport}_moneyline.pkl is always saved as
the best *portable* model (LightGBM or numpy logistic) so the local edge tier
needs no torch. If a neural net wins it is reported but not used for serving.
"""

import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

from models.compare import (
    NumpyLogistic, _impute_standardize, _has,
    _fit_logistic, _fit_lightgbm, _fit_xgboost, _fit_mlp,
)
from models.evaluate import evaluate_classifier, simulate_roi
from models.train import TEAM_FEATURE_COLS

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

_GBM_PARAMS = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
               "num_leaves": 31, "min_data_in_leaf": 20, "feature_fraction": 0.8,
               "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1}


def _load_features(sport: str, db_path: str | None, features_path: str | None) -> pl.DataFrame:
    if features_path:
        return pl.read_parquet(features_path)
    from features.pipeline import (
        build_nba_game_features, build_nhl_game_features, add_injury_features,
    )
    df = build_nba_game_features(db_path) if sport == "nba" else build_nhl_game_features(db_path)
    return add_injury_features(df, sport, db_path)


def _matrices(df: pl.DataFrame):
    """Return (X_raw with nan, y, feature cols). Home perspective, date-sorted."""
    cols = [c for c in TEAM_FEATURE_COLS if c in df.columns]
    sub = (df.filter(pl.col("is_home") == 1)
           .select(cols + ["win", "game_date"])
           .drop_nulls(subset=["win"])
           .sort("game_date"))
    X = sub.select(cols).to_numpy().astype(float)
    y = sub["win"].to_numpy().astype(float)
    return X, y, cols


def run_team_comparison(sport: str, db_path: str | None = None,
                        features_path: str | None = None, n_splits: int = 5) -> pl.DataFrame:
    """Run the bake-off for one sport's moneyline and persist its winner."""
    df = _load_features(sport, db_path, features_path)
    if df.is_empty():
        print(f"  No {sport.upper()} rows in features — nothing to compare")
        return pl.DataFrame()
    X, y, cols = _matrices(df)
    print(f"  {sport.upper()} comparison data: {X.shape[0]} samples, {X.shape[1]} features")

    families = {"logistic": _fit_logistic, "lightgbm": _fit_lightgbm, "xgboost": _fit_xgboost}
    if _has("torch"):
        families["mlp"] = _fit_mlp
    else:
        print("  note: torch not installed -> skipping MLP (install .[tennis] in Colab)")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = {name: np.full(len(y), np.nan) for name in families}
    for tr, val in tscv.split(X):
        for name, fn in families.items():
            try:
                oof[name][val] = fn(X[tr], y[tr], X[val])
            except Exception as e:
                print(f"    warning: {name} fold failed ({e})")

    stack = np.vstack([oof[n] for n in families])
    with np.errstate(invalid="ignore"):
        oof["ensemble"] = np.nanmean(np.where(np.isnan(stack).all(axis=0), np.nan, stack), axis=0)

    rows = []
    for name, preds in oof.items():
        mask = ~np.isnan(preds)
        if mask.sum() == 0:
            continue
        m = evaluate_classifier(y[mask], preds[mask])
        roi = simulate_roi(preds[mask], y[mask])
        rows.append({"model": name, **{k: round(v, 4) for k, v in m.items()},
                     "roi": round(roi["roi"], 4), "bets": roi["bets"]})

    table = pl.DataFrame(rows).sort("log_loss")
    print("\n  === Model comparison (sorted by CV log-loss) ===")
    print(table)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    table.write_csv(ARTIFACTS_DIR / f"{sport}_model_comparison.csv")

    _persist_winner(table, X, y, cols, sport)
    return table


def _persist_winner(table: pl.DataFrame, X, y, cols, sport: str) -> None:
    """Save the best portable model as {sport}_moneyline.pkl (refit on all data)."""
    ranked = table["model"].to_list()
    winner = ranked[0]
    portable = {"lightgbm", "logistic", "ensemble"}
    serve = next((m for m in ranked if m in portable), "lightgbm")

    if serve in ("logistic", "ensemble"):
        from sklearn.linear_model import LogisticRegression
        Xi, _, mean, std, med = _impute_standardize(X, X)
        m = LogisticRegression(max_iter=2000).fit(Xi, y)
        model = NumpyLogistic(m.coef_.ravel(), m.intercept_[0], mean, std, med, cols)
        artifact = {"model": model, "feature_cols": cols, "sport": sport, "family": "logistic"}
    else:
        import lightgbm as lgb
        m = lgb.train(_GBM_PARAMS, lgb.Dataset(X, label=y), num_boost_round=400)
        artifact = {"model": m, "feature_cols": cols, "sport": sport, "family": "lightgbm"}

    with open(ARTIFACTS_DIR / f"{sport}_moneyline.pkl", "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n  {sport.upper()} winner: {winner}. Saved portable server model: "
          f"{artifact['family']} -> {sport}_moneyline.pkl")
    if winner not in portable:
        print(f"  ({winner} won the bake-off but needs its heavy lib at serve time; "
              f"the portable {artifact['family']} model is used for local edge detection.)")
