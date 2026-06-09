"""Time-series forecasting primitives for form features.

These turn a player's / team's recent history into smarter temporal estimates
than a flat rolling mean:

- EWMA (exponentially weighted mean) — recency-weighted level, momentum.
- Holt (double exponential smoothing) — a one-step-ahead forecast that also
  captures *trend* (a team improving, a player ramping up), exposing both the
  forecast level and the slope as features.

Leakage discipline matches the rest of the pipeline: every value at row i uses
only observations strictly prior to row i (the same guarantee `shift(1)` gives
the rolling features). EWMA does this with `.shift(1)`; Holt's one-step forecast
is by construction made before the current observation is seen.
"""

import numpy as np
import polars as pl


# ── EWMA (vectorised, cheap) ───────────────────────────────────────────────────

def ewm_expr(col: str, group_col: str, half_life: float) -> pl.Expr:
    """Exponentially weighted mean of `col` over a group, using only prior rows.

    The leading `.shift(1)` excludes the current game (no leakage); callers must
    have the frame sorted by [group_col, date] so the recursion runs in order.
    """
    return (
        pl.col(col)
        .shift(1)
        .ewm_mean(half_life=half_life, min_samples=1, ignore_nulls=True)
        .over(group_col)
        .alias(f"{col}_ewm")
    )


# ── Holt double exponential smoothing (one-step-ahead, leakage-safe) ────────────

def _holt_onestep(x: np.ndarray, alpha: float, beta: float):
    """One-step-ahead Holt forecasts using only strictly prior observations.

    Returns (forecast, trend) arrays aligned to `x`. Entry i is the forecast for
    x[i] formed from x[:i] only (NaN until the first observation is seen), so the
    output is safe to use as a feature for predicting row i.
    """
    n = len(x)
    fc = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    level = None
    trend = 0.0
    for i in range(n):
        if level is not None:
            fc[i] = level + trend
            tr[i] = trend
        xi = x[i]
        if xi is None or (isinstance(xi, float) and np.isnan(xi)):
            continue
        if level is None:
            level = float(xi)
            trend = 0.0
        else:
            prev_level = level
            level = alpha * float(xi) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
    return fc, tr


def add_holt_features(
    df: pl.DataFrame,
    cols: list[str],
    group_col: str,
    date_col: str,
    alpha: float = 0.5,
    beta: float = 0.1,
) -> pl.DataFrame:
    """Add `{col}_holt` (one-step forecast) and `{col}_trend` (slope) per group.

    Computed per group on numpy arrays (an O(n) recursive pass, like the Elo
    pass) and concatenated back in the same row order, so it aligns with the
    rest of the frame. Light enough to run locally.
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    df = df.sort([group_col, date_col])
    parts = df.partition_by(group_col, maintain_order=True)
    rebuilt = []
    for part in parts:
        new = {}
        for c in cols:
            fc, tr = _holt_onestep(part[c].to_numpy().astype(float), alpha, beta)
            new[f"{c}_holt"] = pl.Series(f"{c}_holt", fc)
            new[f"{c}_trend"] = pl.Series(f"{c}_trend", tr)
        rebuilt.append(part.with_columns(list(new.values())))
    out = pl.concat(rebuilt, how="vertical")
    # match the rest of the pipeline: no-prior rows are null, not NaN
    return out.with_columns(
        pl.col(f"{c}_{k}").fill_nan(None)
        for c in cols for k in ("holt", "trend")
    )


# ── totals-as-a-series ──────────────────────────────────────────────────────────

def add_totals_series(
    df: pl.DataFrame,
    total_col: str,
    group_col: str = "team_id",
    date_col: str = "game_date",
    alpha: float = 0.4,
    beta: float = 0.1,
) -> pl.DataFrame:
    """Forecast each team's scoring-environment series (points scored + allowed).

    Treats the per-game total points a team is involved in as a univariate time
    series and forecasts the next value with Holt, exposing:
      team_total_holt  - one-step-ahead forecast of the team's total environment
      team_total_trend - whether that environment is trending up/down

    `total_col` must already hold (team score + opponent score) for each row.
    """
    out = add_holt_features(df, [total_col], group_col, date_col, alpha, beta)
    return out.rename({
        f"{total_col}_holt": "team_total_holt",
        f"{total_col}_trend": "team_total_trend",
    })
