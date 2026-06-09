# sports-edge — session context

Accumulated cross-session memory. Keep concise but complete.

## Project shape
ML betting pipeline: `ingestion/ → features/ → models/ → edge/`
(SQLite → Polars → `.pkl` artifacts → stdout/CSV alerts). Sports: NBA, NHL, **Tennis (ATP+WTA)**.
Entry: `run.py`. Config via `.env` (DB_PATH, BANKROLL, MIN_EDGE, ODDS_API_KEY).

---

## Session: 2026-06-08 (latest) — Time-series features, accuracy overhaul, held-out backtest

### What was worked on
- **Time-series forecasting features** — `features/forecast.py` (new): `ewm_expr` (leakage-safe
  EWMA via `shift(1).ewm_mean`), `add_holt_features` (double-exponential one-step forecast + trend),
  `add_totals_series` (team scoring-environment Holt). Wired through `features/pipeline.py` for
  NBA/NHL team + player stats and tennis `won`.
- **Accuracy overhaul ("implement everything, all free")** — all open-source, free data only:
  - **Probability calibration** — isotonic calibrator on a time-ordered holdout, stored in
    moneyline artifacts (`calibrator`), applied at serve (`edge/calculator.py:_predict_proba`).
  - **Distributional totals/props pricing** — replaced fabricated `abs(diff)*0.03` edge with
    quantile models (P15.87/P84.13 → per-game sigma); serve prices `P(Over)=1-Phi((line-mean)/sigma)`
    (`_predict_mean_sigma`, `_over_under_edges`). `min_edge` is now a probability threshold.
  - **CLV** — `models/evaluate.py:closing_line_value` (mean CLV%, beat-rate, prob-edge).
  - **Recency weighting** — `0.5**(age_days/365)` sample weights on every market.
  - **NBA pace + four factors** — `_nba_advanced_exprs` (poss/off_rating/eFG%/TOV%/FT-rate) rolled+
    EWMA+opp mirrors, `pace_matchup`, `off_def_edge`. Derived from existing boxscores (no new source).
  - **NHL starting goalie** — `nhl_goalie_games` ingested from the free NHL API
    (`ingestion/nhl.py:_parse_boxscore` now 3-tuple incl goalies); `add_goalie_features` adds
    leakage-safe starter save%/GA (+opp mirror). Null until backfilled.
  - **Market-line-as-feature** — `add_market_features` joins devigged implied prob/total/spread
    from `odds_snapshots`. Null until odds history accumulates.
  - **Bivariate-Poisson NHL** — `models/poisson_nhl.py` (new): one coherent score model
    (attack/defense/home-adv + Dixon-Coles), yields consistent ML/totals/puck-line. scipy fit,
    numpy-only serving. Colab notebook `poisson_nhl_colab.ipynb`.
  - **Optuna tuning** + **DeepAR/TFT prop forecaster** scaffolds → Colab tier
    (`tuning_colab.ipynb`, `props_forecast_colab.ipynb`, `models/prop_forecast.py` torch-free serving).
- **Held-out test season (this session's last task)** — training-time holdout, NOT ingestion-time:
  `models/train.py` got `HOLDOUT_START="2025-10-01"` / `TENNIS_HOLDOUT_START="2025-01-01"`,
  `_split_train_test`, `_holdout_eval`. Every trainer takes optional `holdout_start`: fits on
  pre-cutoff rows, scores the never-seen season, saves `_holdout`-suffixed artifacts (never clobbers
  production). New `run.py --backtest` flag → `backtest(sport)` (no ingest, no edges).

### Bugs fixed this session
- **Holt emitted NaN not null** → `.fill_nan(None)` to match rolling-feature null semantics.
- **NHL `opp_goals` Holt collision** with `opp_` mirror prefix → `NHL_TEAM_TREND_STATS=["goals"]` only.
- **DuplicateError `plus_minus_ewm`** (shared NBA/NHL stat) → `_dedup` (`dict.fromkeys`) on feature lists.
- **0-samples training bug (CRITICAL)** — always-null market/goalie cols made `_prep`'s
  `drop_nulls(subset=all features)` drop every row. Fixed: exclude entirely-null cols from the
  drop subset (`non_null = [c for c in feat_present if sub[c].null_count() < sub.height]`).
- **Market join SchemaError** (string `d` vs date) → `str.to_date(...,strict=False)`.

### Results / state (verified end-to-end, exit 0)
- **Full NBA+NHL retrain** with the complete ~120-feature stack: all markets train, calibrators fit,
  sigma models save. NBA ML CV log-loss 0.6414; NHL 0.6856. Props all sane RMSE.
- **NBA holdout backtest** ran clean: ML test (1307 games) log-loss 0.6531 / acc 0.624 / auc 0.681 —
  tracks CV (0.6515) closely = no meaningful overfit. Props test RMSE ≈ CV. Saved `nba_*_holdout.pkl`.
- Market features currently ALL-NULL (no historical odds overlap); LightGBM handles nulls — they
  light up as odds accumulate. NHL goalie features null until `--ingest-history --sport nhl` backfill.

### Key decisions / tradeoffs
- **Holdout at TRAIN time, not ingest time** — do NOT remove 2025-26 from season lists. That data
  is needed live to build rolling features for upcoming games and to find edges. Only `fit` excludes it.
- **Two modes kept**: `--train` = all data (live betting); `--backtest` = honest test, separate
  `_holdout` artifacts. Default `holdout_start=None` so production unaffected.
- All accuracy work constrained to **free sources + OSS libs** (user requirement). Heavy training
  (DeepAR, Optuna, Poisson) deferred to free Colab tier per standing no-heavy-local-compute pref.
- `_holdout` artifacts sit in `models/artifacts/` (gitignored) alongside production — harmless
  (serving loads unsuffixed names). Offered to route to a subdir; not done unless requested.

### Docs added/updated
- `docs/ACCURACY.md` (new) — all upgrades + "Held-out test season" section.
- `README.md` — new sections: "Using The Odds API efficiently (free tier)", "How often to run
  (don't skip >7 days)", "Backtest on a held-out season".
- `notebooks/README.md` — Optuna/Poisson/props-forecast notebooks; `docs/TRAINING_PROPS_FORECAST.md`.

### Gotchas worth keeping
- **The Odds API free tier = 500 req/MONTH** (not day). Cost = markets × regions; props = 1 req per
  event per prop market (expensive). Historical-odds endpoints cost 10× — avoid on free tier (this is
  why `mkt_*` features start null; only live pulls are stored, no backfill). Balance auto-logged to
  `odds_requests_remaining` (`get_quota`).
- **Daily run re-ingests only a rolling 7-day window** (`fetch_recent_games(days_back=7)`), deduped by
  `INSERT OR REPLACE` on composite PKs. Run ≥ once/7 days or you get permanent game-log gaps that don't
  self-heal. `--ingest-history` is the only full backfill.

### INSTRUCTIONS TO BRING THIS SESSION TO PRODUCTION
1. **Finish the running NHL ingest** (`python3 run.py --ingest-history --sport nhl`) — populates
   `nhl_goalie_games`, lighting up the goalie features.
2. **Retrain production models on all data**: `.venv/bin/python run.py --train --sport all`
   (use the venv + `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` on macOS).
3. **(Optional) Sanity backtest**: `.venv/bin/python run.py --backtest --sport all` for honest
   test-season metrics (NHL/tennis weren't backtested this session — only NBA was).
4. Heavy/cloud-tier (optional, free Colab): run `tuning_colab.ipynb`, `props_forecast_colab.ipynb`,
   `poisson_nhl_colab.ipynb`; download artifacts → `data/exports/artifacts/`; `run.py --import-models`.
5. Confirm ROI/CLV on the held-out season once odds exist — accuracy is solid but edge-vs-book is
   still unproven (gated on odds history).

---

## Session: 2026-06-08 (later) — WOWY injury features, NBA/NHL retrain, tennis ops fixes, decimal odds

### What was worked on
- **Injury / WOWY features (NBA + NHL)** — `features/pipeline.py`: absences inferred from
  boxscores (seasonal-roster player with no game row = out). New backbone `_wowy_long` /
  `_wowy_team_frame`; `add_injury_features` extended with team-level `wowy_margin_delta`,
  `key_players_out`, `out_min_share` (+ `_opp` mirrors); new `add_teammate_wowy_features`
  adds `teammate_out_min_share`, `lead_teammate_out` for props. All splits use strictly-prior
  games (leakage-safe). `_KEY_MIN_THRESHOLD`, `_parse_min` helpers added.
- **`models/train.py`** — extended `TEAM_FEATURE_COLS` (+6 WOWY) and `PLAYER_FEATURE_COLS` (+2);
  `train_all` now applies `add_teammate_wowy_features`.
- **`models/compare_team.py`** (new) — NBA/NHL moneyline bake-off mirroring tennis; imports the
  single MLP impl from `models/compare.py`. `run_team_comparison(sport, ...)`.
- **`run.py`** — generalized `--export-features` / `--compare` / `--import-models` for NBA/NHL;
  prediction path applies teammate WOWY. **`notebooks/team_train_colab.ipynb`** (new).
- **Decimal (European) odds display** — `edge/calculator.py` new `american_to_decimal` (+100→2.00).
  `edge/alerts.py` console shows decimal; CSV adds `odds_decimal` column (raw American `odds` kept).
  Internal fetch + edge/vig/Kelly math STILL American by design — decimal is display-layer only.
- **Tennis odds keys** — `ingestion/odds.py`: removed bogus `tennis_atp_queens`/`tennis_atp_halle`
  (they 404 — The Odds API has NO ATP Queen's/Halle keys). Added real active key
  `tennis_wta_queens_club_champ`. No-odds message reworded to "no currently tracked tournaments
  are active" (`edge/calculator.py:302`).

### Bugs fixed
- **NHL never trainable before**: `train_spread`/`train_totals` hardcoded NBA `pts`/`opp_pts`.
  Added `_SCORE_COLS = {"nba":("pts","opp_pts"),"nhl":("goals","opp_goals")}`; now sport-aware.
- **Serving feature_cols mismatch**: train fns saved full 53-name list while training on the
  columns actually present (NHL only 13) → misaligned/mislabeled importances at serve. Fixed:
  `_prep` returns `used_cols`; all 7 callers save those. Verified NBA 53=53, NHL 13=13.

### Results / state
- NBA + NHL **fully ingested, trained, validated**. NBA moneyline CV log-loss 0.6515→**0.6386**
  with WOWY (`out_min_share` #3 by gain). NHL 0.6888→0.6875; the **4 WOWY features are NHL's top 4
  by gain** (old `star_out`/`injured_pts_lost` proxies near-zero — WOWY clearly superior).
- Tennis **history fully ingested this session** (2015–2026, 32k ATP + 30k WTA matches; weather
  ~195 outdoor fetches via Open-Meteo). `tennis_features` not yet built/trained on full history.
- Decimal-odds change verified (console + CSV); `/code-review` of the diff = clean (no findings).

### Critical gotchas added this session
- **Use the venv**: deps live in `sports-edge/.venv`, NOT global `python3`. Running `python3
  run.py` directly fails `ModuleNotFoundError: nba_api`. Run `.venv/bin/python run.py` (or
  `source .venv/bin/activate`). Earlier notes saying "python3 locally" are wrong for this repo.
- **`python run.py` excludes tennis**: default `--sport both` = NBA+NHL only; tennis needs
  `--sport all` or `--sport tennis` (gated because tennis ingest is heavier).
- **The Odds API tennis = Grand Slams + Masters/WTA1000 + a few extras only.** No ATP grass
  (Queen's/Halle) keys exist at all. Full key list fetchable via `/v4/sports?all=true`. During a
  dead week (e.g. between French Open & Wimbledon) all keys return empty → "no tracked tournaments
  active" is EXPECTED, not a misconfig.
- **Benign Polars warning** at `features/pipeline.py:304`: "Sortedness ... cannot be checked when
  'by' groups provided" — both frames ARE sorted by game_date just before the `join_asof`; results
  correct. Silence (if wanted) via `.set_sorted("game_date")`.

### Available-but-untracked tennis keys (add if broader coverage wanted, cost-free while inactive)
`tennis_atp_barcelona_open`, `tennis_atp_hamburg_open`, `tennis_atp_munich`,
`tennis_wta_charleston_open`, `tennis_wta_strasbourg`, `tennis_wta_stuttgart_open`.

---

## Session: 2026-06-08 — Added Tennis as a market

### What was worked on
Full tennis market added end-to-end (ATP + WTA; markets: match-winner, totals (game O/U),
spread (game handicap)). New + changed files:
- **`ingestion/tennis.py`** (new) — Sackmann ATP/WTA match CSVs (GitHub raw, keyless),
  curated-CSV loaders, Open-Meteo weather fetch (averaged over the tournament fortnight).
- **`ingestion/tennis_clean.py`** (new) — `tennis_matches` → `tennis_matches_clean`: score
  parsing (total_games/sets/tiebreaks/margin), incomplete flags (RET/W/O/DEF), type coercion,
  name normalization. Has reusable `_write_table(conn, name, df)`.
- **`data/court_speed.csv`**, **`data/tournament_locations.csv`** (new, curated, ~35 stops each).
- **`features/pipeline.py`** (extended) — surface-specific Elo + overall/surface H2H
  (`_compute_elo_h2h`), rolling form/serve/return rates, rest + 14-day fatigue, weather/court/
  altitude joins; `build_tennis_player_features`, `build_tennis_match_features`,
  `persist_tennis_features` (OBT → `tennis_features` table), `add_tennis_odds_features`.
  Exports `TENNIS_FEATURE_COLS`, `_DIFF_FEATURES`, `_CONTEXT_FEATURES`.
- **`models/train.py`** (extended) — SEPARATE ATP/WTA training: `train_tennis_{moneyline,totals,
  spread}(df, tour)` → `tennis_{tour}_{market}.pkl`; `train_all_tennis` loops tours. `_prep` got
  `drop_feature_nulls` param.
- **`models/evaluate.py`** (implemented from stubs) — `evaluate_classifier`, `evaluate_regressor`,
  `simulate_roi` (numpy/no-sklearn AUC, Kelly ROI backtest).
- **`models/compare.py`** (new) — per-tour bake-off: Elo baseline / logistic / LightGBM /
  XGBoost(HistGB fallback) / PyTorch MLP / ensemble; TimeSeriesSplit; saves portable winner +
  `tennis_{tour}_model_comparison.csv`. `NumpyLogistic` = torch/sklearn-free serving model.
- **`edge/calculator.py`** (extended) — `find_tennis_edges`: latest-snapshot-per-player, builds
  matchup diffs on the fly, auto-loads the tour-specific model (cached).
- **`ingestion/odds.py`** (extended) — `TENNIS_SPORT_KEYS`, `fetch_tennis_odds` (loops tournament
  keys, writes sport='tennis' into shared `odds_snapshots`).
- **`run.py`** (extended) — `--sport tennis|all`, `--compare`, `--export-features`,
  `--import-models`; tennis ingest flow = fetch → weather → clean.
- **`notebooks/tennis_train_colab.ipynb`** + **`notebooks/README.md`** (new) — Colab GPU training,
  per-tour, parquet handoff.
- **`docs/TRAINING_MLP.md`** (new) — exact MLP training steps.
- **`pyproject.toml`** — `[tennis]` extra (torch, xgboost).
- Docs: `PLAN.md`, `CLAUDE.md` updated for tennis.

### Current state (working, verified on test DBs)
- Ingest ATP+WTA (≈6k ATP, ≈5.5k WTA per 2 seasons), clean, weather all work.
- OBT builds; **separate** ATP/WTA models train: CV log-loss ATP 0.628 / WTA 0.626 (realistic).
- Per-tour bake-off runs (MLP trains+scores when torch present); ensemble usually best.
- `find_tennis_edges` end-to-end verified: correct tour model auto-selected, vig-strip + Kelly OK.
- Verification used `data/test.db` (ATP only, has weather) and `data/wta_test.db` (ATP+WTA, no
  weather). These are throwaway; safe to delete. **Production artifacts have only been trained on
  2-season test samples — NOT production quality yet.**

### Key decisions / tradeoffs
- **Separate ATP & WTA models** (was briefly a single combined model; user requested split).
  `find_tennis_edges` picks model from the player snapshot's `tour`.
- **One Big Table** (`tennis_features`) at the analytical layer; raw sources stay normalized.
- **Court speed = curated CSV** (no live CPI API exists). **Weather = Open-Meteo** (free/keyless),
  averaged over the tournament fortnight (Sackmann has no per-match dates).
- **Heavy compute (MLP/bake-off) runs in the cloud (Colab GPU), not the laptop** — standing user
  preference. Local serving stays torch-free: bake-off saves the best PORTABLE model
  (LightGBM or NumpyLogistic) as the artifact; MLP is evaluated but not the default server.
- xgboost slot falls back to sklearn HistGradientBoosting when xgboost absent (still labeled
  "xgboost" in the table — minor cosmetic).

### Critical gotchas (don't re-introduce)
- **Leakage via tourney_date**: Sackmann `tourney_date` is the tournament START — every match in
  an event shares one date. A naive 14-day trailing fatigue window then leaked later-round results
  (matches_14d_diff dominated; CV log-loss collapsed to 0.35). FIXED with a per-round synthetic
  `seq_date` (= tourney_date + round_ord days) used for ordering, Elo/H2H, rolling, fatigue.
- **Null-feature row dropping**: `_prep` drops rows with ANY null feature by default. Tennis MUST
  pass `drop_feature_nulls=False` (LightGBM handles NaN) — else null weather wipes the whole
  training set (this caused the WTA "0 samples" bug) and discards ~half the data otherwise.
- **Weather populates only if** `fetch_weather()` ran AND tournament is in
  `tournament_locations.csv` AND it's outdoor (indoor = neutral/null by design).
- **macOS bake-off hang**: LightGBM + sklearn OpenMP double-load deadlocks locally. Run with
  `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`. Does not occur on Colab (Linux).
- `train_all_tennis` always rebuilds the OBT (`persist_tennis_features`) so features never go stale.

### Next steps
1. **Full-history training**: `python run.py --sport tennis --ingest-history` (TENNIS_YEARS =
   2015–2026) then retrain — current artifacts are test-sample only.
2. **Train the MLP for real** in Colab (see `docs/TRAINING_MLP.md`); decide whether to serve it.
3. Expand `court_speed.csv` / `tournament_locations.csv` beyond the ~35 seeded tournaments for
   fuller weather/speed coverage (unlisted tournaments get null context).
4. Optional: real historical odds for a meaningful ROI backtest (current `simulate_roi` uses a
   synthetic base-rate book line — relative comparison only).
5. Optional: probability calibration (Platt/temperature) before edge math if serving non-tree models.
6. Consider tightening the "xgboost" label when the HistGB fallback is used.

### Context to remember
- Sackmann CSV URL: `https://raw.githubusercontent.com/JeffSackmann/tennis_{atp,wta}/master/{atp,wta}_matches_{year}.csv`.
- The Odds API splits tennis by tournament (`tennis_atp_*`, `tennis_wta_*`); odds stored with
  sport='tennis' in `odds_snapshots`.
- Tennis tables: `tennis_matches`, `tennis_matches_clean`, `tennis_court_speed`, `tennis_locations`,
  `tennis_weather`, `tennis_features` (OBT). Artifacts: `tennis_{atp,wta}_{moneyline,totals,spread}.pkl`.
- Python invoked as `python3` locally; core deps (lightgbm, sklearn, polars, dotenv) needed —
  install with `pip install -e .`; heavy with `pip install -e ".[tennis]"`.
