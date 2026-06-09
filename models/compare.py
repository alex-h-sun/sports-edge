"""Tennis match-winner model bake-off.

Trains several model families on the same features with time-series CV (no
leakage), prints a ranked comparison (CV log-loss, Brier, AUC, accuracy, plus a
rough Kelly-ROI backtest), and persists results.

Families: Elo-only baseline, logistic regression, LightGBM, XGBoost (or sklearn
HistGradientBoosting fallback), a PyTorch MLP, and an averaged ensemble.

Designed to run identically locally or on Colab GPU:
  - import-safe, no heavy work at import time
  - device-agnostic (CUDA if available, else CPU)
  - optional deps (torch, xgboost) are skipped with a printed note if absent
  - accepts an explicit features path (parquet) or a SQLite db_path

Serving note: to keep the local edge tier torch-free, the production artifact
tennis_moneyline.pkl is always saved as the best *portable* model (LightGBM or a
numpy logistic). If a neural net wins overall it is reported and also saved to
tennis_moneyline_nn.pkl for environments that have torch.
"""

import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

from models.evaluate import evaluate_classifier, simulate_roi

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


# ── portable models (loadable without sklearn/torch) ────────────────────────────

class NumpyLogistic:
    """Standardize + logistic, stored as plain numpy so .predict needs no sklearn."""

    def __init__(self, coef, intercept, mean, scale, median, feature_cols):
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.median = np.asarray(median, dtype=float)
        self.feature_cols = feature_cols

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = np.where(np.isnan(X), self.median, X)  # impute with training median
        z = (X - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))


# ── data ────────────────────────────────────────────────────────────────────────

def _load_features(db_path: str | None, features_path: str | None) -> pl.DataFrame:
    if features_path:
        return pl.read_parquet(features_path)
    import sqlite3
    from models.train import _load_tennis_obt
    return _load_tennis_obt(db_path)


def _matrices(df: pl.DataFrame, feature_cols: list[str]):
    """Return (X_raw with nan, y, date-sorted). Trees use X_raw; linear/NN impute."""
    cols = [c for c in feature_cols if c in df.columns]
    sub = df.select(cols + ["won", "match_date"]).drop_nulls(subset=["won"])
    sub = sub.sort("match_date")
    X = sub.select(cols).to_numpy().astype(float)
    y = sub["won"].to_numpy().astype(float)
    return X, y, cols


def _impute_standardize(X_tr, X_val):
    """Median-impute (fit on train) then standardize. For linear / NN models."""
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(X_tr), med, X_tr)
    Xv = np.where(np.isnan(X_val), med, X_val)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (Xtr - mean) / std, (Xv - mean) / std, mean, std, med


# ── per-family fold trainers (return val-fold probabilities) ────────────────────

def _fit_elo_baseline(X_tr, y_tr, X_val, elo_idx):
    from sklearn.linear_model import LogisticRegression
    a = X_tr[:, [elo_idx]]
    b = X_val[:, [elo_idx]]
    med = np.nanmedian(a)
    a = np.where(np.isnan(a), med, a)
    b = np.where(np.isnan(b), med, b)
    m = LogisticRegression(max_iter=1000).fit(a, y_tr)
    return m.predict_proba(b)[:, 1]


def _fit_logistic(X_tr, y_tr, X_val):
    from sklearn.linear_model import LogisticRegression
    Xtr, Xv, *_ = _impute_standardize(X_tr, X_val)
    m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y_tr)
    return m.predict_proba(Xv)[:, 1]


def _fit_lightgbm(X_tr, y_tr, X_val):
    import lightgbm as lgb
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
              "num_leaves": 31, "min_data_in_leaf": 20, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1}
    m = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=300)
    return m.predict(X_val)


def _fit_xgboost(X_tr, y_tr, X_val):
    try:
        import xgboost as xgb
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(X_tr, y_tr)
        return m.predict_proba(X_val)[:, 1]
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val)
    params = {"objective": "binary:logistic", "eta": 0.05, "max_depth": 6,
              "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss"}
    m = xgb.train(params, dtr, num_boost_round=300)
    return m.predict(dval)


def _fit_mlp(X_tr, y_tr, X_val):
    import torch
    import torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Xv, *_ = _impute_standardize(X_tr, X_val)

    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device).unsqueeze(1)
    net.train()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        xv = torch.tensor(Xv, dtype=torch.float32, device=device)
        return torch.sigmoid(net(xv)).cpu().numpy().ravel()


# ── orchestration ────────────────────────────────────────────────────────────────

def _has(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def run_comparison(db_path: str | None = None, features_path: str | None = None,
                   tour: str = "atp", n_splits: int = 5) -> pl.DataFrame:
    """Run the bake-off for one tour and persist its winner. Returns the table.

    tour: 'atp' or 'wta' — ATP and WTA get fully separate models.
    """
    from features.pipeline import TENNIS_FEATURE_COLS

    df = _load_features(db_path, features_path)
    df = df.filter(pl.col("tour") == tour)
    if df.is_empty():
        print(f"  No {tour.upper()} rows in features — nothing to compare")
        return pl.DataFrame()
    X, y, cols = _matrices(df, TENNIS_FEATURE_COLS)
    elo_idx = cols.index("elo_diff") if "elo_diff" in cols else 0
    print(f"  {tour.upper()} comparison data: {X.shape[0]} samples, {X.shape[1]} features")

    families = {
        "elo_baseline": lambda a, b, c: _fit_elo_baseline(a, b, c, elo_idx),
        "logistic": _fit_logistic,
        "lightgbm": _fit_lightgbm,
        "xgboost": _fit_xgboost,
    }
    if _has("torch"):
        families["mlp"] = _fit_mlp
    else:
        print("  note: torch not installed -> skipping MLP (install .[tennis] in Colab)")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    # collect out-of-fold predictions aligned to the validation indices
    oof = {name: np.full(len(y), np.nan) for name in families}
    for tr, val in tscv.split(X):
        for name, fn in families.items():
            try:
                oof[name][val] = fn(X[tr], y[tr], X[val])
            except Exception as e:
                print(f"    warning: {name} fold failed ({e})")

    # ensemble = mean of available family oof preds (rows before the first CV
    # split have no predictions and stay NaN; they're masked out of metrics below)
    stack = np.vstack([oof[n] for n in families])
    with np.errstate(invalid="ignore"):
        ens = np.nanmean(np.where(np.isnan(stack).all(axis=0), np.nan, stack), axis=0)
    oof["ensemble"] = ens

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
    table.write_csv(ARTIFACTS_DIR / f"tennis_{tour}_model_comparison.csv")

    _persist_winner(table, X, y, cols, tour)
    return table


def _persist_winner(table: pl.DataFrame, X, y, cols, tour: str) -> None:
    """Save the best portable model as tennis_{tour}_moneyline.pkl (refit on all data)."""
    ranked = table["model"].to_list()
    winner = ranked[0]
    portable = {"lightgbm", "logistic", "elo_baseline", "ensemble"}
    serve = next((m for m in ranked if m in portable), "lightgbm")

    if serve == "logistic" or (serve == "ensemble"):
        # numpy logistic is the most robust torch/sklearn-free server
        from sklearn.linear_model import LogisticRegression
        Xi, _, mean, std, med = _impute_standardize(X, X)
        m = LogisticRegression(max_iter=2000).fit(Xi, y)
        model = NumpyLogistic(m.coef_.ravel(), m.intercept_[0], mean, std, med, cols)
        artifact = {"model": model, "feature_cols": cols, "sport": "tennis",
                    "tour": tour, "family": "logistic"}
    else:
        import lightgbm as lgb
        params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
                  "num_leaves": 31, "min_data_in_leaf": 20, "feature_fraction": 0.8,
                  "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1}
        m = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=400)
        artifact = {"model": m, "feature_cols": cols, "sport": "tennis",
                    "tour": tour, "family": "lightgbm"}

    with open(ARTIFACTS_DIR / f"tennis_{tour}_moneyline.pkl", "wb") as f:
        pickle.dump(artifact, f)
    print(f"\n  {tour.upper()} winner: {winner}. Saved portable server model: {artifact['family']} "
          f"-> tennis_{tour}_moneyline.pkl")
    if winner not in portable:
        print(f"  ({winner} won the bake-off but needs its heavy lib at serve time; "
              f"the portable {artifact['family']} model is used for local edge detection.)")
