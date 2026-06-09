"""Bivariate-Poisson goals model for NHL (coherent joint score distribution).

Instead of training moneyline, spread, and totals as three independent models
that can contradict each other, this fits one generative model of the score:

    home_goals ~ Poisson(mu_home),  away_goals ~ Poisson(mu_away)
    log mu_home = home_adv + attack[home] - defense[away]
    log mu_away =           attack[away] - defense[home]

From the fitted (attack, defense, home_adv) we get the full joint distribution
over (home_goals, away_goals), and *every* market is derived from it
consistently: P(home win), P(total over L), P(home -X). A low-score (Dixon-Coles
style) correlation term is included so 0-0/1-0/0-1/1-1 — common in hockey — are
not under-counted.

Fitting is a small convex-ish MLE (scipy), cheap enough to run locally, but it
ships as a portable params artifact so serving never needs scipy. Heavy
retraining/backtesting belongs on Colab — see notebooks/poisson_nhl_colab.ipynb.
"""

import pickle
from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
_MAX_GOALS = 12  # truncation for the score grid (NHL games rarely exceed this)


# ── fitting (scipy; Colab / offline tier) ───────────────────────────────────────

def fit(home_ids, away_ids, home_goals, away_goals, l2: float = 0.01) -> dict:
    """Maximum-likelihood fit of attack/defense/home-advantage + DC correlation.

    Returns a portable params dict (team index -> attack/defense, home_adv, rho).
    """
    from scipy.optimize import minimize

    home_ids = np.asarray(home_ids)
    away_ids = np.asarray(away_ids)
    hg = np.asarray(home_goals, dtype=float)
    ag = np.asarray(away_goals, dtype=float)

    teams = sorted(set(home_ids.tolist()) | set(away_ids.tolist()))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = np.array([idx[t] for t in home_ids])
    ai = np.array([idx[t] for t in away_ids])

    # params: [attack(n), defense(n), home_adv, rho]; attack mean-centred via penalty
    def unpack(p):
        return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]

    def neg_ll(p):
        atk, dfn, home_adv, rho = unpack(p)
        mu_h = np.exp(home_adv + atk[hi] - dfn[ai])
        mu_a = np.exp(atk[ai] - dfn[hi])
        ll = (hg * np.log(mu_h) - mu_h) + (ag * np.log(mu_a) - mu_a)
        # Dixon-Coles low-score adjustment
        tau = np.ones_like(mu_h)
        m00 = (hg == 0) & (ag == 0); m10 = (hg == 1) & (ag == 0)
        m01 = (hg == 0) & (ag == 1); m11 = (hg == 1) & (ag == 1)
        tau = np.where(m00, 1 - mu_h * mu_a * rho, tau)
        tau = np.where(m10, 1 + mu_a * rho, tau)
        tau = np.where(m01, 1 + mu_h * rho, tau)
        tau = np.where(m11, 1 - rho, tau)
        ll = ll + np.log(np.clip(tau, 1e-6, None))
        pen = l2 * (np.sum(atk ** 2) + np.sum(dfn ** 2)) + 100.0 * atk.mean() ** 2
        return -np.sum(ll) + pen

    p0 = np.concatenate([np.zeros(n), np.zeros(n), [0.1], [0.0]])
    res = minimize(neg_ll, p0, method="L-BFGS-B",
                   bounds=[(-3, 3)] * (2 * n) + [(-1, 1), (-0.2, 0.2)])
    atk, dfn, home_adv, rho = unpack(res.x)
    return {
        "teams": teams,
        "attack": {t: float(atk[i]) for t, i in idx.items()},
        "defense": {t: float(dfn[i]) for t, i in idx.items()},
        "home_adv": float(home_adv),
        "rho": float(rho),
    }


def save(params: dict, name: str = "nhl_poisson") -> Path:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(params, f)
    return path


# ── serving (numpy only) ────────────────────────────────────────────────────────

def _score_grid(mu_h: float, mu_a: float, rho: float) -> np.ndarray:
    """Joint P(home=i, away=j) over an (i, j) grid with the DC correction."""
    i = np.arange(_MAX_GOALS + 1)
    # independent Poisson pmfs
    from math import lgamma
    log_pmf_h = i * np.log(mu_h) - mu_h - np.array([lgamma(k + 1) for k in i])
    log_pmf_a = i * np.log(mu_a) - mu_a - np.array([lgamma(k + 1) for k in i])
    grid = np.exp(log_pmf_h)[:, None] * np.exp(log_pmf_a)[None, :]
    # DC low-score correction
    grid[0, 0] *= 1 - mu_h * mu_a * rho
    grid[1, 0] *= 1 + mu_a * rho
    grid[0, 1] *= 1 + mu_h * rho
    grid[1, 1] *= 1 - rho
    return grid / grid.sum()


def predict_game(params: dict, home_id, away_id) -> dict:
    """Derive all markets from the joint score grid for one matchup.

    Returns p_home_win, p_draw (regulation tie), expected totals, and a function
    to query P(total > line) / P(home margin >= x). Unknown teams fall back to
    league-average strength (0).
    """
    atk = params["attack"]; dfn = params["defense"]
    a_h, d_h = atk.get(home_id, 0.0), dfn.get(home_id, 0.0)
    a_a, d_a = atk.get(away_id, 0.0), dfn.get(away_id, 0.0)
    mu_h = float(np.exp(params["home_adv"] + a_h - d_a))
    mu_a = float(np.exp(a_a - d_h))
    grid = _score_grid(mu_h, mu_a, params["rho"])

    i = np.arange(_MAX_GOALS + 1)
    diff = i[:, None] - i[None, :]      # home - away margin
    total = i[:, None] + i[None, :]
    p_home = float(grid[diff > 0].sum())
    p_draw = float(grid[diff == 0].sum())
    # regulation ties are settled in OT/SO; split the tie mass by relative strength
    p_home_ml = p_home + p_draw * (mu_h / (mu_h + mu_a))

    def p_total_over(line: float) -> float:
        return float(grid[total > line].sum())

    def p_home_cover(handicap: float) -> float:
        return float(grid[(diff + handicap) > 0].sum())

    return {
        "mu_home": mu_h, "mu_away": mu_a,
        "exp_total": mu_h + mu_a,
        "p_home_win": p_home_ml,
        "p_total_over": p_total_over,
        "p_home_cover": p_home_cover,
    }


def load(name: str = "nhl_poisson") -> dict | None:
    path = ARTIFACTS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
