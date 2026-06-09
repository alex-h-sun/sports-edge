"""Model evaluation: calibration, Brier score, AUC, ROI simulation."""

import numpy as np
import polars as pl


def evaluate_classifier(y_true, y_prob) -> dict:
    """Return AUC, Brier score, log loss, and accuracy for a binary classifier."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)

    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    acc = float(np.mean((p >= 0.5) == (y >= 0.5)))

    # AUC via the rank (Mann-Whitney) formula — no sklearn dependency
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        order = np.argsort(p)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(p) + 1)
        auc = float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    return {"auc": auc, "brier": brier, "log_loss": logloss, "accuracy": acc}


def evaluate_regressor(y_true, y_pred) -> dict:
    """Return MAE, RMSE, and R^2 for a regression model."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(p - y)))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _american_to_decimal(odds) -> np.ndarray:
    """Vectorised American -> decimal odds."""
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, o / 100 + 1, 100 / np.abs(o) + 1)


def closing_line_value(bet_odds, closing_odds) -> dict:
    """Closing-line value: did we beat the price the market closed at?

    CLV is the single best leading indicator that a model has real edge — you can
    beat the closing line on average long before ROI stabilises, and conversely a
    positive backtest ROI with negative CLV is almost always noise/overfit.

    bet_odds     : American odds we took on each bet
    closing_odds : American odds for that same side at close
    Returns mean CLV%, the beat rate (fraction priced better than close), and the
    implied-probability edge captured (closing implied - our implied).
    """
    bet_dec = _american_to_decimal(bet_odds)
    close_dec = _american_to_decimal(closing_odds)
    if bet_dec.size == 0:
        return {"mean_clv_pct": 0.0, "beat_rate": 0.0, "prob_edge": 0.0, "n": 0}
    clv_pct = bet_dec / close_dec - 1.0           # >0 == we got a better price
    prob_edge = (1.0 / close_dec) - (1.0 / bet_dec)  # closing implied - our implied
    return {
        "mean_clv_pct": float(np.mean(clv_pct)),
        "beat_rate": float(np.mean(bet_dec > close_dec)),
        "prob_edge": float(np.mean(prob_edge)),
        "n": int(bet_dec.size),
    }


def _american_payout(prob: float) -> float:
    """Fair decimal payout multiplier implied by a probability (no vig)."""
    return 1.0 / prob if prob > 0 else 0.0


def simulate_roi(
    model_prob,
    win,
    book_prob=None,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
) -> dict:
    """Backtest fractional-Kelly betting on historical predictions.

    model_prob : model's P(win) per bet
    win        : realized outcome (1/0)
    book_prob  : book's vig-free implied prob. If None, a flat 5% hold synthetic
                 line is assumed around the empirical base rate (rough sanity ROI).
    Returns ROI, total staked, profit, number of bets, and hit rate.
    """
    p = np.asarray(model_prob, dtype=float)
    y = np.asarray(win, dtype=float)
    if book_prob is None:
        # synthetic fair odds = realized base rate; payout from those fair odds
        bp = np.full_like(p, float(np.mean(y)))
    else:
        bp = np.asarray(book_prob, dtype=float)

    edge = p - bp
    bets = edge >= min_edge
    n = int(bets.sum())
    if n == 0:
        return {"roi": 0.0, "bets": 0, "staked": 0.0, "profit": 0.0, "hit_rate": 0.0}

    dec_odds = 1.0 / np.clip(bp[bets], 1e-6, 1 - 1e-6)  # fair decimal odds
    b = dec_odds - 1.0
    pb = p[bets]
    full_kelly = np.clip((b * pb - (1 - pb)) / b, 0, 1)
    stake = kelly_fraction * full_kelly

    outcomes = y[bets]
    profit = np.where(outcomes == 1, stake * b, -stake)
    total_stake = float(stake.sum())
    total_profit = float(profit.sum())
    return {
        "roi": total_profit / total_stake if total_stake > 0 else 0.0,
        "bets": n,
        "staked": total_stake,
        "profit": total_profit,
        "hit_rate": float(outcomes.mean()),
    }
