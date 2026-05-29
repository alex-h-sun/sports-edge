"""Train LightGBM models for each market. Saves artifacts to models/artifacts/."""

import os
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# Feature columns used by team-level models (moneyline, spread, totals)
_TEAM_STATS = ["pts", "fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov", "stl", "blk", "plus_minus"]

TEAM_FEATURE_COLS = (
    [f"{s}_roll5"  for s in _TEAM_STATS] +
    [f"{s}_roll10" for s in _TEAM_STATS] +
    [f"opp_{s}_roll5"  for s in _TEAM_STATS] +
    [f"opp_{s}_roll10" for s in _TEAM_STATS] +
    ["rest_days", "win_streak_5", "is_home",
     "injured_pts_lost", "star_out",
     "injured_pts_lost_opp", "star_out_opp"]
)

# Feature columns for player prop models
_PLAYER_STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "fg_pct", "fg3_pct", "ft_pct", "plus_minus"]

PLAYER_FEATURE_COLS = (
    [f"{s}_roll5"  for s in _PLAYER_STATS] +
    [f"{s}_roll10" for s in _PLAYER_STATS] +
    ["rest_days", "minutes_roll5", "is_home"]
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(obj, name: str) -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved {path}")
    return path


def _prep(df: pl.DataFrame, feature_cols: list[str], target_col: str):
    """Drop nulls on features + target, return numpy arrays."""
    cols = feature_cols + [target_col, "game_date"]
    present = [c for c in cols if c in df.columns]
    sub = df.select(present).drop_nulls(subset=[c for c in feature_cols if c in df.columns] + [target_col])
    sub = sub.sort("game_date")
    X = sub.select([c for c in feature_cols if c in sub.columns]).to_numpy()
    y = sub[target_col].to_numpy()
    return X, y


def _cv_score(X, y, params: dict, task: str) -> float:
    """Time-series cross-validated score. Returns mean val metric."""
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        m = lgb.train(
            params, dtrain,
            num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        pred = m.predict(X_val)
        if task == "binary":
            # log loss
            pred = np.clip(pred, 1e-7, 1 - 1e-7)
            score = -np.mean(y_val * np.log(pred) + (1 - y_val) * np.log(1 - pred))
        else:
            score = np.sqrt(np.mean((pred - y_val) ** 2))
        scores.append(score)
    return float(np.mean(scores))


def _train_final(X, y, params: dict) -> lgb.Booster:
    """Train on full dataset."""
    return lgb.train(
        params,
        lgb.Dataset(X, label=y),
        num_boost_round=400,
        callbacks=[lgb.log_evaluation(0)],
    )


# ── moneyline ─────────────────────────────────────────────────────────────────

def train_moneyline(df: pl.DataFrame, sport: str) -> lgb.Booster:
    """Binary classifier: predict home team win probability.

    Input df must be one row per game (home team perspective),
    e.g. output of build_nba_game_features filtered to is_home == 1.
    """
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    home_df = df.filter(pl.col("is_home") == 1)
    X, y = _prep(home_df, TEAM_FEATURE_COLS, "win")
    print(f"  Moneyline: {X.shape[0]} samples, {X.shape[1]} features")

    cv = _cv_score(X, y, params, "binary")
    print(f"  CV log-loss: {cv:.4f}")

    model = _train_final(X, y, params)
    _save({"model": model, "feature_cols": TEAM_FEATURE_COLS, "sport": sport}, f"{sport}_moneyline")
    return model


# ── spread ────────────────────────────────────────────────────────────────────

def train_spread(df: pl.DataFrame, sport: str) -> lgb.Booster:
    """Regressor: predict home team margin of victory (home_pts - away_pts)."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    # build margin target: join home and away pts per game
    home = df.filter(pl.col("is_home") == 1).select(
        ["game_id", "game_date", "pts", "opp_pts"] + [c for c in TEAM_FEATURE_COLS if c in df.columns]
    ).with_columns(
        (pl.col("pts") - pl.col("opp_pts")).alias("margin")
    )

    X, y = _prep(home, TEAM_FEATURE_COLS, "margin")
    print(f"  Spread: {X.shape[0]} samples, {X.shape[1]} features")

    cv = _cv_score(X, y, params, "regression")
    print(f"  CV RMSE: {cv:.2f}")

    model = _train_final(X, y, params)
    _save({"model": model, "feature_cols": TEAM_FEATURE_COLS, "sport": sport}, f"{sport}_spread")
    return model


# ── totals ────────────────────────────────────────────────────────────────────

def train_totals(df: pl.DataFrame, sport: str) -> lgb.Booster:
    """Regressor: predict total points scored (home_pts + away_pts)."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    home = df.filter(pl.col("is_home") == 1).select(
        ["game_id", "game_date", "pts", "opp_pts"] + [c for c in TEAM_FEATURE_COLS if c in df.columns]
    ).with_columns(
        (pl.col("pts") + pl.col("opp_pts")).alias("total")
    )

    X, y = _prep(home, TEAM_FEATURE_COLS, "total")
    print(f"  Totals: {X.shape[0]} samples, {X.shape[1]} features")

    cv = _cv_score(X, y, params, "regression")
    print(f"  CV RMSE: {cv:.2f}")

    model = _train_final(X, y, params)
    _save({"model": model, "feature_cols": TEAM_FEATURE_COLS, "sport": sport}, f"{sport}_totals")
    return model


# ── player props ──────────────────────────────────────────────────────────────

PROP_STATS = {
    "nba": ["pts", "reb", "ast", "stl", "blk"],
    "nhl": ["goals", "assists", "points", "shots"],
}


def train_player_props(df: pl.DataFrame, sport: str) -> dict[str, lgb.Booster]:
    """Train one regressor per prop stat. Returns dict of stat -> model."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    models = {}
    for stat in PROP_STATS[sport]:
        if stat not in df.columns:
            print(f"  Props [{stat}]: column not found, skipping")
            continue

        X, y = _prep(df, PLAYER_FEATURE_COLS, stat)
        if len(X) < 100:
            print(f"  Props [{stat}]: insufficient data ({len(X)} rows), skipping")
            continue

        print(f"  Props [{stat}]: {X.shape[0]} samples", end=" ")
        cv = _cv_score(X, y, params, "regression")
        print(f"CV RMSE: {cv:.2f}")

        model = _train_final(X, y, params)
        _save(
            {"model": model, "feature_cols": PLAYER_FEATURE_COLS, "sport": sport, "stat": stat},
            f"{sport}_prop_{stat}"
        )
        models[stat] = model

    return models


# ── convenience: train all ────────────────────────────────────────────────────

def train_all(sport: str, db_path: str) -> dict:
    """Build features and train all models for a sport. Returns all models."""
    from features.pipeline import (
        build_nba_game_features, build_nba_player_features,
        build_nhl_game_features, build_nhl_player_features,
        add_injury_features,
    )

    print(f"\n=== Training {sport.upper()} models ===")

    if sport == "nba":
        game_df   = build_nba_game_features(db_path)
        player_df = build_nba_player_features(db_path)
    else:
        game_df   = build_nhl_game_features(db_path)
        player_df = build_nhl_player_features(db_path)

    game_df = add_injury_features(game_df, sport, db_path)

    print("\n-- Moneyline --")
    ml_model = train_moneyline(game_df, sport)

    print("\n-- Spread --")
    sp_model = train_spread(game_df, sport)

    print("\n-- Totals --")
    to_model = train_totals(game_df, sport)

    print("\n-- Player Props --")
    prop_models = train_player_props(player_df, sport)

    return {
        "moneyline": ml_model,
        "spread":    sp_model,
        "totals":    to_model,
        "props":     prop_models,
    }
