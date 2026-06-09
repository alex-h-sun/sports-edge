"""Train LightGBM models for each market. Saves artifacts to models/artifacts/."""

import os
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

from models.evaluate import evaluate_classifier, evaluate_regressor

# Start of the held-out test season. Passing holdout_start to the trainers excludes
# every game on/after this date from fitting and instead scores the trained model on
# them (an honest, never-seen test set). Default training uses ALL data (production /
# live betting); only backtests pass a cutoff. NBA/NHL 2025-26 seasons begin ~October;
# tennis is by calendar year so its test set is the 2025 season onward.
HOLDOUT_START = "2025-10-01"
TENNIS_HOLDOUT_START = "2025-01-01"

# half-life (days) for recency-weighted training: a game one year old counts half
# as much as a game today. Recent seasons reflect current rosters/rules better.
_RECENCY_HALF_LIFE_DAYS = 365.0

# the two quantiles whose half-spread approximates 1 sigma of a normal predictive
# distribution (P84.13 - P15.87) / 2. Used to price totals/props as distributions.
_SIGMA_QLO, _SIGMA_QHI = 0.1587, 0.8413

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def _dedup(cols: list[str]) -> list[str]:
    """Order-preserving dedup (NBA and NHL stat lists share a few names)."""
    return list(dict.fromkeys(cols))

# Feature columns used by team-level models (moneyline, spread, totals)
_TEAM_STATS = ["pts", "fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov", "stl", "blk", "plus_minus"]

# NHL team stats also carry EWMA features (sport-neutral cols are filtered by
# presence in _prep, so listing both sports' names here is harmless).
_NHL_TEAM_STATS = ["goals", "opp_goals", "shots", "opp_shots"]

# NBA pace / four-factors (rolled + opp mirrors) and explicit matchup features
_NBA_ADV = ["pace", "off_rating", "efg", "tov_pct", "ft_rate"]

TEAM_FEATURE_COLS = _dedup(
    [f"{s}_roll5"  for s in _TEAM_STATS] +
    [f"{s}_roll10" for s in _TEAM_STATS] +
    [f"opp_{s}_roll5"  for s in _TEAM_STATS] +
    [f"opp_{s}_roll10" for s in _TEAM_STATS] +
    # EWMA momentum (recency-weighted) — NBA + NHL stat names
    [f"{s}_ewm" for s in _TEAM_STATS] +
    [f"opp_{s}_ewm" for s in _TEAM_STATS] +
    [f"{s}_ewm" for s in _NHL_TEAM_STATS] +
    [f"opp_{s}_ewm" for s in _NHL_TEAM_STATS] +
    # Holt one-step forecast + trend (trajectory) for headline stats + opp mirror
    ["pts_holt", "pts_trend", "plus_minus_holt", "plus_minus_trend",
     "opp_pts_holt", "opp_pts_trend", "opp_plus_minus_holt", "opp_plus_minus_trend",
     "goals_holt", "goals_trend", "opp_goals_holt", "opp_goals_trend",
     # totals-as-a-series: forecast of each team's scoring environment
     "team_total_holt", "team_total_trend",
     "opp_team_total_holt", "opp_team_total_trend"] +
    # NBA pace / four-factors (rolled + EWMA + opp mirrors)
    [f"{s}_roll5" for s in _NBA_ADV] + [f"{s}_roll10" for s in _NBA_ADV] +
    [f"{s}_ewm" for s in _NBA_ADV] +
    [f"opp_{s}_roll5" for s in _NBA_ADV] + [f"opp_{s}_roll10" for s in _NBA_ADV] +
    [f"opp_{s}_ewm" for s in _NBA_ADV] +
    ["pace_matchup", "off_def_edge"] +
    # NHL starting-goalie form (null until nhl_goalie_games is backfilled)
    ["goalie_save_pct_roll", "goalie_ga_roll",
     "opp_goalie_save_pct_roll", "opp_goalie_ga_roll"] +
    # market line as a feature (null until odds history accumulates)
    ["mkt_implied_prob", "mkt_total", "mkt_spread"] +
    ["rest_days", "win_streak_5", "is_home",
     "injured_pts_lost", "star_out",
     "injured_pts_lost_opp", "star_out_opp",
     # WOWY (with-or-without-you) absence features
     "wowy_margin_delta", "key_players_out", "out_min_share",
     "wowy_margin_delta_opp", "key_players_out_opp", "out_min_share_opp"]
)

# Feature columns for player prop models
_PLAYER_STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "fg_pct", "fg3_pct", "ft_pct", "plus_minus"]

_NHL_PLAYER_STATS = ["goals", "assists", "points", "shots", "hits", "blocked_shots", "plus_minus"]

PLAYER_FEATURE_COLS = _dedup(
    [f"{s}_roll5"  for s in _PLAYER_STATS] +
    [f"{s}_roll10" for s in _PLAYER_STATS] +
    # EWMA momentum — NBA + NHL stat names (filtered by presence in _prep)
    [f"{s}_ewm" for s in _PLAYER_STATS] +
    [f"{s}_ewm" for s in _NHL_PLAYER_STATS] +
    # Holt forecast + trend for the headline prop stats
    ["pts_holt", "pts_trend", "reb_holt", "reb_trend", "ast_holt", "ast_trend",
     "goals_holt", "goals_trend", "points_holt", "points_trend",
     "shots_holt", "shots_trend"] +
    ["rest_days", "minutes_roll5", "is_home",
     # teammate-absence (WOWY) features
     "teammate_out_min_share", "lead_teammate_out"]
)

# per-sport (team score, opponent score) columns for spread/totals targets
_SCORE_COLS = {"nba": ("pts", "opp_pts"), "nhl": ("goals", "opp_goals")}


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(obj, name: str) -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved {path}")
    return path


def _prep(df: pl.DataFrame, feature_cols: list[str], target_col: str,
          date_col: str = "game_date", drop_feature_nulls: bool = True):
    """Return numpy arrays (sorted by date_col), always dropping null targets.

    drop_feature_nulls=True also drops rows with any null feature (NBA/NHL default).
    Set False for tree models that handle NaN natively (tennis) so legitimately
    missing features like weather don't discard otherwise-usable rows.
    """
    cols = feature_cols + [target_col, date_col]
    present = [c for c in cols if c in df.columns]
    sub = df.select(present)
    if drop_feature_nulls:
        # Drop rows with null features, but never let an entirely-null optional
        # column (e.g. market line / goalie form before backfill) gate every row.
        feat_present = [c for c in feature_cols if c in sub.columns]
        non_null = [c for c in feat_present if sub[c].null_count() < sub.height]
        subset = non_null + [target_col]
    else:
        subset = [target_col]
    sub = sub.drop_nulls(subset=subset)
    sub = sub.sort(date_col)
    used_cols = [c for c in feature_cols if c in sub.columns]
    X = sub.select(used_cols).to_numpy()
    y = sub[target_col].to_numpy()
    # parse the (already sorted) date column to day numbers for recency weighting
    dates = (
        sub.select(pl.col(date_col).cast(pl.Date).cast(pl.Int32).alias("_d"))["_d"]
        .to_numpy().astype(float)
    )
    # used_cols is what the model is actually trained on; callers must persist it
    # (not the full requested list) so serving aligns columns correctly.
    return X, y, used_cols, dates


def _recency_weights(dates: np.ndarray, half_life_days: float = _RECENCY_HALF_LIFE_DAYS) -> np.ndarray:
    """Exponential-decay sample weights: recent games count more.

    Weight = 0.5 ** (age_in_days / half_life). Newest game has weight 1.0.
    """
    if dates.size == 0:
        return np.ones_like(dates)
    age = dates.max() - dates
    return np.power(0.5, age / half_life_days)


def _cv_score(X, y, params: dict, task: str, weights=None) -> float:
    """Time-series cross-validated score. Returns mean val metric."""
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = None if weights is None else weights[train_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
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


def _train_final(X, y, params: dict, weights=None) -> lgb.Booster:
    """Train on full dataset (optionally recency-weighted)."""
    return lgb.train(
        params,
        lgb.Dataset(X, label=y, weight=weights),
        num_boost_round=400,
        callbacks=[lgb.log_evaluation(0)],
    )


def _fit_calibrator(X, y, params: dict, holdout: float = 0.2) -> IsotonicRegression | None:
    """Fit an isotonic calibrator on a time-ordered holdout slice.

    Raw LightGBM scores are not guaranteed to be calibrated probabilities, but
    `edge = model_prob - fair_prob` only makes sense if they are. We train on the
    first (1-holdout) of the (date-sorted) data, predict the held-out tail, and
    fit isotonic regression mapping raw score -> empirical win rate. Returns None
    if a class is missing from the holdout (calibration would be undefined).
    """
    n = len(y)
    cut = int(n * (1 - holdout))
    if cut < 50 or n - cut < 50:
        return None
    m = _train_final(X[:cut], y[:cut], params)
    raw = m.predict(X[cut:])
    y_hold = y[cut:]
    if len(np.unique(y_hold)) < 2:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, y_hold)
    return iso


def _fit_sigma_models(X, y, params: dict, weights=None) -> dict:
    """Train two quantile regressors (P15.87 / P84.13) for distributional pricing.

    Their half-spread approximates the predictive standard deviation per row, so
    totals/props can be priced as P(Over) = 1 - Phi((line - mean)/sigma) instead
    of the old point-estimate heuristic.
    """
    out = {}
    for tag, alpha in (("q_lo", _SIGMA_QLO), ("q_hi", _SIGMA_QHI)):
        qp = {**params, "objective": "quantile", "alpha": alpha}
        qp.pop("metric", None)
        out[tag] = _train_final(X, y, qp, weights=weights)
    return out


# ── held-out test season ────────────────────────────────────────────────────────

def _split_train_test(df: pl.DataFrame, date_col: str, holdout_start: str | None):
    """Split rows into (train < holdout_start, test >= holdout_start).

    With holdout_start=None the whole frame is the train set (production mode) and
    there is no test slice.
    """
    if not holdout_start:
        return df, None
    cut = pl.col(date_col).cast(pl.Date) >= pl.lit(holdout_start).str.to_date()
    return df.filter(~cut), df.filter(cut)


def _holdout_eval(model: lgb.Booster, test_df, cols: list[str], target: str,
                  task: str, calibrator=None) -> None:
    """Score a freshly-trained model on the never-seen test season and print metrics.

    Uses exactly the columns the model trained on so serving alignment holds; tree
    models tolerate nulls in the test slice so we do not drop feature-null rows.
    """
    if test_df is None or test_df.is_empty():
        return
    sub = test_df.select(cols + [target]).drop_nulls(subset=[target])
    if sub.is_empty():
        return
    X = sub.select(cols).to_numpy()
    y = sub[target].to_numpy()
    pred = model.predict(X)
    if task == "binary":
        if calibrator is not None:
            pred = np.clip(calibrator.predict(pred), 0.0, 1.0)
        m = evaluate_classifier(y, pred)
        print(f"  HOLDOUT test [{len(y)}]: log-loss {m['log_loss']:.4f}  "
              f"acc {m['accuracy']:.3f}  auc {m['auc']:.3f}")
    else:
        m = evaluate_regressor(y, pred)
        print(f"  HOLDOUT test [{len(y)}]: RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}")


# ── moneyline ─────────────────────────────────────────────────────────────────

def train_moneyline(df: pl.DataFrame, sport: str,
                    holdout_start: str | None = None) -> lgb.Booster:
    """Binary classifier: predict home team win probability.

    Input df must be one row per game (home team perspective),
    e.g. output of build_nba_game_features filtered to is_home == 1.
    """
    suffix = "_holdout" if holdout_start else ""
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
    train_df, test_df = _split_train_test(home_df, "game_date", holdout_start)
    X, y, cols, dates = _prep(train_df, TEAM_FEATURE_COLS, "win")
    w = _recency_weights(dates)
    print(f"  Moneyline: {X.shape[0]} samples, {X.shape[1]} features")

    cv = _cv_score(X, y, params, "binary", weights=w)
    print(f"  CV log-loss: {cv:.4f}")

    model = _train_final(X, y, params, weights=w)
    calibrator = _fit_calibrator(X, y, params)
    print(f"  Calibrator: {'fitted (isotonic)' if calibrator else 'skipped'}")
    _holdout_eval(model, test_df, cols, "win", "binary", calibrator=calibrator)
    _save({"model": model, "feature_cols": cols, "sport": sport,
           "calibrator": calibrator}, f"{sport}_moneyline{suffix}")
    return model


# ── spread ────────────────────────────────────────────────────────────────────

def train_spread(df: pl.DataFrame, sport: str,
                 holdout_start: str | None = None) -> lgb.Booster:
    """Regressor: predict home team margin of victory (home_pts - away_pts)."""
    suffix = "_holdout" if holdout_start else ""
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

    # build margin target from the home team's score vs opponent score
    score, opp_score = _SCORE_COLS[sport]
    home = df.filter(pl.col("is_home") == 1).select(
        ["game_id", "game_date", score, opp_score] + [c for c in TEAM_FEATURE_COLS if c in df.columns]
    ).with_columns(
        (pl.col(score) - pl.col(opp_score)).alias("margin")
    )

    train_df, test_df = _split_train_test(home, "game_date", holdout_start)
    X, y, cols, dates = _prep(train_df, TEAM_FEATURE_COLS, "margin")
    w = _recency_weights(dates)
    print(f"  Spread: {X.shape[0]} samples, {X.shape[1]} features")

    cv = _cv_score(X, y, params, "regression", weights=w)
    print(f"  CV RMSE: {cv:.2f}")

    model = _train_final(X, y, params, weights=w)
    sigma = _fit_sigma_models(X, y, params, weights=w)
    _holdout_eval(model, test_df, cols, "margin", "regression")
    _save({"model": model, "feature_cols": cols, "sport": sport, **sigma}, f"{sport}_spread{suffix}")
    return model


# ── totals ────────────────────────────────────────────────────────────────────

def train_totals(df: pl.DataFrame, sport: str,
                 holdout_start: str | None = None) -> lgb.Booster:
    """Regressor: predict total points scored (home_pts + away_pts)."""
    suffix = "_holdout" if holdout_start else ""
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

    score, opp_score = _SCORE_COLS[sport]
    home = df.filter(pl.col("is_home") == 1).select(
        ["game_id", "game_date", score, opp_score] + [c for c in TEAM_FEATURE_COLS if c in df.columns]
    ).with_columns(
        (pl.col(score) + pl.col(opp_score)).alias("total")
    )

    train_df, test_df = _split_train_test(home, "game_date", holdout_start)
    X, y, cols, dates = _prep(train_df, TEAM_FEATURE_COLS, "total")
    w = _recency_weights(dates)
    print(f"  Totals: {X.shape[0]} samples, {X.shape[1]} features")

    cv = _cv_score(X, y, params, "regression", weights=w)
    print(f"  CV RMSE: {cv:.2f}")

    model = _train_final(X, y, params, weights=w)
    sigma = _fit_sigma_models(X, y, params, weights=w)
    _holdout_eval(model, test_df, cols, "total", "regression")
    _save({"model": model, "feature_cols": cols, "sport": sport, **sigma}, f"{sport}_totals{suffix}")
    return model


# ── player props ──────────────────────────────────────────────────────────────

PROP_STATS = {
    "nba": ["pts", "reb", "ast", "stl", "blk"],
    "nhl": ["goals", "assists", "points", "shots"],
}


def train_player_props(df: pl.DataFrame, sport: str,
                       holdout_start: str | None = None) -> dict[str, lgb.Booster]:
    """Train one regressor per prop stat. Returns dict of stat -> model."""
    suffix = "_holdout" if holdout_start else ""
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

    train_df, test_df = _split_train_test(df, "game_date", holdout_start)
    models = {}
    for stat in PROP_STATS[sport]:
        if stat not in df.columns:
            print(f"  Props [{stat}]: column not found, skipping")
            continue

        X, y, cols, dates = _prep(train_df, PLAYER_FEATURE_COLS, stat)
        if len(X) < 100:
            print(f"  Props [{stat}]: insufficient data ({len(X)} rows), skipping")
            continue
        w = _recency_weights(dates)

        print(f"  Props [{stat}]: {X.shape[0]} samples", end=" ")
        cv = _cv_score(X, y, params, "regression", weights=w)
        print(f"CV RMSE: {cv:.2f}")

        model = _train_final(X, y, params, weights=w)
        sigma = _fit_sigma_models(X, y, params, weights=w)
        _holdout_eval(model, test_df, cols, stat, "regression")
        _save(
            {"model": model, "feature_cols": cols, "sport": sport, "stat": stat, **sigma},
            f"{sport}_prop_{stat}{suffix}"
        )
        models[stat] = model

    return models


# ── tennis ──────────────────────────────────────────────────────────────────────

_GBM_PARAMS = {
    "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1,
}


def _load_tennis_obt(db_path: str) -> pl.DataFrame:
    """Read the persisted tennis_features OBT (rebuilds it if missing)."""
    import sqlite3
    from features.pipeline import persist_tennis_features
    conn = sqlite3.connect(db_path)
    has = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tennis_features'"
    ).fetchone()
    conn.close()
    if not has:
        return persist_tennis_features(db_path)
    return pl.read_database("SELECT * FROM tennis_features",
                            sqlite3.connect(db_path))


TENNIS_TOURS = ["atp", "wta"]


def _tour_slice(df: pl.DataFrame, tour: str) -> pl.DataFrame:
    """Filter the OBT to a single tour. ATP and WTA get fully separate models."""
    return df.filter(pl.col("tour") == tour)


def train_tennis_moneyline(df: pl.DataFrame, tour: str,
                           holdout_start: str | None = None) -> lgb.Booster:
    """Binary classifier: P(player1 beats player2) for one tour's OBT rows."""
    from features.pipeline import TENNIS_FEATURE_COLS
    suffix = "_holdout" if holdout_start else ""
    params = {"objective": "binary", "metric": "binary_logloss", **_GBM_PARAMS}
    sub = _tour_slice(df, tour)
    train_df, test_df = _split_train_test(sub, "match_date", holdout_start)
    X, y, cols, dates = _prep(train_df, TENNIS_FEATURE_COLS, "won", date_col="match_date", drop_feature_nulls=False)
    w = _recency_weights(dates)
    print(f"  {tour.upper()} moneyline: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  CV log-loss: {_cv_score(X, y, params, 'binary', weights=w):.4f}")
    model = _train_final(X, y, params, weights=w)
    calibrator = _fit_calibrator(X, y, params)
    _holdout_eval(model, test_df, cols, "won", "binary", calibrator=calibrator)
    _save({"model": model, "feature_cols": cols, "sport": "tennis", "tour": tour,
           "calibrator": calibrator}, f"tennis_{tour}_moneyline{suffix}")
    return model


def train_tennis_totals(df: pl.DataFrame, tour: str,
                        holdout_start: str | None = None) -> lgb.Booster:
    """Regressor: total games in the match (completed matches only), one tour."""
    from features.pipeline import TENNIS_FEATURE_COLS
    suffix = "_holdout" if holdout_start else ""
    params = {"objective": "regression", "metric": "rmse", **_GBM_PARAMS}
    sub = _tour_slice(df, tour).filter(pl.col("completed") == 1)
    train_df, test_df = _split_train_test(sub, "match_date", holdout_start)
    X, y, cols, dates = _prep(train_df, TENNIS_FEATURE_COLS, "total_games", date_col="match_date", drop_feature_nulls=False)
    w = _recency_weights(dates)
    print(f"  {tour.upper()} totals: {X.shape[0]} samples")
    print(f"  CV RMSE: {_cv_score(X, y, params, 'regression', weights=w):.2f}")
    model = _train_final(X, y, params, weights=w)
    sigma = _fit_sigma_models(X, y, params, weights=w)
    _holdout_eval(model, test_df, cols, "total_games", "regression")
    _save({"model": model, "feature_cols": cols, "sport": "tennis", "tour": tour, **sigma},
          f"tennis_{tour}_totals{suffix}")
    return model


def train_tennis_spread(df: pl.DataFrame, tour: str,
                        holdout_start: str | None = None) -> lgb.Booster:
    """Regressor: game margin from player1 perspective (completed only), one tour."""
    from features.pipeline import TENNIS_FEATURE_COLS
    suffix = "_holdout" if holdout_start else ""
    params = {"objective": "regression", "metric": "rmse", **_GBM_PARAMS}
    sub = _tour_slice(df, tour).filter(pl.col("completed") == 1)
    train_df, test_df = _split_train_test(sub, "match_date", holdout_start)
    X, y, cols, dates = _prep(train_df, TENNIS_FEATURE_COLS, "game_margin", date_col="match_date", drop_feature_nulls=False)
    w = _recency_weights(dates)
    print(f"  {tour.upper()} spread: {X.shape[0]} samples")
    print(f"  CV RMSE: {_cv_score(X, y, params, 'regression', weights=w):.2f}")
    model = _train_final(X, y, params, weights=w)
    sigma = _fit_sigma_models(X, y, params, weights=w)
    _holdout_eval(model, test_df, cols, "game_margin", "regression")
    _save({"model": model, "feature_cols": cols, "sport": "tennis", "tour": tour, **sigma},
          f"tennis_{tour}_spread{suffix}")
    return model


def train_all_tennis(db_path: str, holdout_start: str | None = None) -> dict:
    """Rebuild the OBT and train separate ATP and WTA models for all 3 markets."""
    from features.pipeline import persist_tennis_features
    print("\n=== Training TENNIS models (separate ATP / WTA) ===")
    if holdout_start:
        print(f"  Holdout: training < {holdout_start}, testing on the held-out season")
    df = persist_tennis_features(db_path)  # always rebuild so features are never stale

    out: dict = {}
    for tour in TENNIS_TOURS:
        if _tour_slice(df, tour).is_empty():
            print(f"\n-- {tour.upper()}: no data, skipping --")
            continue
        print(f"\n-- {tour.upper()} Moneyline --")
        ml = train_tennis_moneyline(df, tour, holdout_start)
        print(f"\n-- {tour.upper()} Totals --")
        to = train_tennis_totals(df, tour, holdout_start)
        print(f"\n-- {tour.upper()} Spread --")
        sp = train_tennis_spread(df, tour, holdout_start)
        out[tour] = {"moneyline": ml, "totals": to, "spread": sp}
    return out


# ── convenience: train all ────────────────────────────────────────────────────

def train_all(sport: str, db_path: str, holdout_start: str | None = None) -> dict:
    """Build features and train all models for a sport. Returns all models.

    holdout_start: when set, every game on/after it is excluded from training and
    used as a never-seen test set instead (artifacts are saved with a `_holdout`
    suffix so they never overwrite the production models). Leave None for live
    betting, which trains on all available data.
    """
    from features.pipeline import (
        build_nba_game_features, build_nba_player_features,
        build_nhl_game_features, build_nhl_player_features,
        add_injury_features, add_teammate_wowy_features, add_market_features,
    )

    if sport == "tennis":
        return train_all_tennis(db_path, holdout_start)

    print(f"\n=== Training {sport.upper()} models ===")
    if holdout_start:
        print(f"  Holdout: training < {holdout_start}, testing on the held-out season")

    if sport == "nba":
        game_df   = build_nba_game_features(db_path)
        player_df = build_nba_player_features(db_path)
    else:
        game_df   = build_nhl_game_features(db_path)
        player_df = build_nhl_player_features(db_path)

    game_df   = add_injury_features(game_df, sport, db_path)
    game_df   = add_market_features(game_df, sport, db_path)
    player_df = add_teammate_wowy_features(player_df, sport, db_path)

    print("\n-- Moneyline --")
    ml_model = train_moneyline(game_df, sport, holdout_start)

    print("\n-- Spread --")
    sp_model = train_spread(game_df, sport, holdout_start)

    print("\n-- Totals --")
    to_model = train_totals(game_df, sport, holdout_start)

    print("\n-- Player Props --")
    prop_models = train_player_props(player_df, sport, holdout_start)

    return {
        "moneyline": ml_model,
        "spread":    sp_model,
        "totals":    to_model,
        "props":     prop_models,
    }
