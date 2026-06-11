# Accuracy upgrades

A pass focused on prediction quality and, more importantly, *profitability vs the
market*. Everything here uses free data sources (`nba_api`, `api.nhle.com`,
Sackmann, The Odds API free tier) and open-source libraries only.

## Edge-math fixes (highest leverage)

### Probability calibration
Raw LightGBM scores are not guaranteed to be calibrated probabilities, but
`edge = model_prob - fair_prob` is only meaningful if they are. Moneyline models
(NBA/NHL/tennis) now fit an **isotonic calibrator** on a time-ordered holdout and
store it in the artifact (`calibrator`). Serving applies it via
`edge.calculator._predict_proba`. No-op for old artifacts without a calibrator.
Isotonic calibrators saturate their tails to exactly 0/1, so serving additionally
clamps the calibrated probability to `[0.02, 0.98]` (`PROB_FLOOR`/`PROB_CEIL`) —
no real moneyline is a certainty, and an inflated tail probability would inflate
both the edge and the Kelly stake.

### Distributional totals & props pricing
Previously totals/props bet "Over if model > line" with a fabricated edge
(`abs(diff) * 0.03`). Now each regression market trains two **quantile models**
(P15.87 / P84.13) whose half-spread is a per-game sigma. Serving prices
`P(Over) = 1 - Phi((line - mean)/sigma)` and computes a real probability edge vs
the book's vig-free line (`_predict_mean_sigma`, `_over_under_edges`). `min_edge`
is now a probability threshold for these markets.

### Closing-line value (CLV)
`models.evaluate.closing_line_value` measures whether bets beat the price the
market closed at — the single best leading indicator of real edge. Track this
alongside ROI; a positive backtest ROI with negative CLV is almost always noise.

## Training improvements

- **Recency weighting** — `sample_weight = 0.5 ** (age_days / 365)` so recent
  seasons (current rosters/rules) count more. Applied to every market.
- **Hyperparameter tuning** — `notebooks/tuning_colab.ipynb` runs Optuna per
  market under the same `TimeSeriesSplit` CV (Colab/CPU tier, free).

## New features

- **NBA pace & four factors** — possessions, offensive rating, eFG%, TOV%, FT
  rate, derived per game from raw boxscore counts (no new source), rolled + EWMA
  + opponent mirrors, plus explicit `pace_matchup` and `off_def_edge`. Far more
  predictive than raw points because they are pace-adjusted and opponent-comparable.
- **NHL starting goalie** — `nhl_goalie_games` is now ingested from the NHL API
  (free), and `add_goalie_features` attaches the starter's leakage-safe rolling
  save% / goals-against (+ opponent mirror). Backfill with
  `python run.py --ingest-history --sport nhl`. Null until backfilled.
- **Market line as a feature** — `add_market_features` joins the opening vig-free
  implied probability / total / spread. The market is the strongest single signal;
  this lets the model learn where to *disagree*. Best-effort name+date join; mostly
  null until odds history accumulates (odds_snapshots only keeps live pulls).

## Coherent joint model

- **Bivariate-Poisson NHL** (`models/poisson_nhl.py`, `notebooks/poisson_nhl_colab.ipynb`)
  — one generative score model (attack/defense/home-adv + Dixon-Coles low-score
  term). Moneyline, totals, and puck-line are all derived from the same joint
  score grid, so they never contradict each other. Portable params artifact
  (`nhl_poisson.pkl`); serving is numpy-only.

## Held-out test season

CV (`TimeSeriesSplit`) is the working validation signal, but for one honest,
never-seen estimate the trainers accept a `holdout_start`: every game on/after it is
excluded from `fit` and the model is scored on it instead. NBA/NHL hold out the
2025-26 season (`HOLDOUT_START = 2025-10-01`); tennis holds out 2025 onward
(`2025-01-01`). The held-out data is **not** dropped from ingestion — it is still
needed to build rolling features for upcoming games and to find live edges; only the
`fit` excludes it.

```bash
python run.py --backtest --sport all   # train < cutoff, report test-season metrics
```

Backtest artifacts are saved with a `_holdout` suffix so they never clobber the
production models. For live betting, plain `--train` fits on all data (including the
current season) because there is no reason to withhold data from a model you are
actually deploying.

## What to run

```bash
python run.py --train --sport all      # retrain everything with the new stack (all data)
python run.py --backtest --sport all   # honest holdout test on the 2025-26 season
python run.py --ingest-history --sport nhl   # backfill goalie boxscores (once)
```

Backtest with `models/evaluate.py` (now including CLV) to confirm the changes add
edge before betting real stakes.
