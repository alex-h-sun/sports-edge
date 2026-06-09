# Sports Edge — Project Plan

## Goal
Scrape historical sports data, train ML models to predict game/player outcomes, and identify positive expected-value bets by comparing model probabilities against live sportsbook odds.

## Sports & Markets
- **Sports:** NBA, NHL, Tennis (ATP + WTA)
- **Markets (NBA/NHL):** Moneyline, Spread, Totals (O/U), Player Props
- **Markets (Tennis):** Match winner, Totals (game O/U), Spread (game handicap)

---

## Architecture

```
ingestion/ → features/ → models/ → edge/
   ↓              ↓           ↓         ↓
SQLite        Polars DFs   .pkl      stdout / CSV
```

### Data sources
| Layer | NBA | NHL | Tennis |
|---|---|---|---|
| Game/match logs | `nba_api` (LeagueGameLog) | NHL Stats API (api.nhle.com) | Sackmann tennis_atp/tennis_wta CSVs |
| Player stats | `nba_api` (PlayerGameLogs) | NHL Stats API | (per-match serve/return stats in Sackmann) |
| Weather | — | — | Open-Meteo Historical Archive API |
| Court speed | — | — | Curated `data/court_speed.csv` (ITF CPI) |
| Locations/altitude | — | — | Curated `data/tournament_locations.csv` |
| Live odds | The Odds API | The Odds API | The Odds API (`tennis_atp_*`, `tennis_wta_*`) |

Tennis-derived signals (computed in the feature pipeline, no extra source):
surface-specific Elo, overall + surface H2H, rolling form/serve/return rates,
rest + 14-day fatigue (uses a per-round synthetic date so trailing windows do
not leak later-round results), weather/altitude interactions.

---

## Implementation Phases

### Phase 1 — Ingestion ✅ / 🔲
- [x] `ingestion/nba.py` — team + player game logs, SQLite upsert
- [x] `ingestion/nhl.py` — team + player game logs, SQLite upsert
- [x] `ingestion/odds.py` — live odds via The Odds API (all 4 markets)

### Phase 2 — Feature Engineering
- [x] `features/pipeline.py`
  - Team features: rolling 5/10-game averages (PTS, FG%, REB, AST, TOV), rest days, home/away flag, win streak
  - Player features: rolling 5/10-game averages per stat, minutes trend, days rest, home/away splits
  - Odds features: opening vs. closing line movement, implied probability, vig-adjusted fair odds
  - Injury / WOWY features: absences inferred from boxscores (roster player with no game row = out);
    team-level `wowy_margin_delta` / `key_players_out` / `out_min_share` (+ opp mirrors) via
    leakage-safe per-player with/without margins; teammate-level `teammate_out_min_share` /
    `lead_teammate_out` for the props models. Backbone: `_wowy_long` / `_wowy_team_frame` /
    `add_injury_features` / `add_teammate_wowy_features`.

### Phase 3 — Models
One model per market. All trained with LightGBM, cross-validated on historical seasons.

| Market | Target | Model type |
|---|---|---|
| Moneyline | Home win (0/1) | Binary classifier |
| Spread | Home margin of victory | Regressor |
| Totals | Total points scored | Regressor |
| Player props | Player stat (pts/reb/ast/etc.) | Regressor per stat |

- [x] `models/train.py` — train + serialize each model to `models/artifacts/`
- [x] `models/evaluate.py` — AUC / Brier / MAE / RMSE + ROI backtest
- [x] `models/compare_team.py` — NBA/NHL moneyline bake-off (LightGBM / XGBoost / PyTorch MLP /
  ensemble) with TimeSeriesSplit; mirrors the tennis bake-off and reuses its fold trainers.
  Heavy/MLP training on Colab GPU (`notebooks/team_train_colab.ipynb`); best portable model served.

### Phase 4 — Edge Detection
- [x] `edge/calculator.py`
  - Convert American odds → implied probability
  - Strip vig (additive method)
  - Edge = model_prob − fair_prob
  - Kelly stake = (edge × odds − (1 − edge)) / odds × fraction
- [x] `edge/alerts.py` — print and save positive-EV bets

### Phase 5 — Runner / CLI
- [x] `run.py` — top-level script: ingest recent games → build features → load models → find edges → print/save alerts
- [ ] Schedule via cron or launchd for daily pre-game runs

---

## Data Schema (SQLite)

### `nba_team_games`
game_id, team_id, season_id, game_date, matchup, wl, pts, fgm, fga, fg_pct, fg3m, fg3a, reb, ast, stl, blk, tov, plus_minus, ...

### `nba_player_games`
game_id, player_id, season_year, game_date, player_name, team_id, min, pts, reb, ast, stl, blk, tov, fg_pct, fg3_pct, ...

### `nhl_team_games` *(to build)*
game_id, team_id, season, game_date, matchup, wl, goals, shots, pp_goals, pp_opps, pim, hits, blocks, ...

### `nhl_player_games` *(to build)*
game_id, player_id, season, game_date, player_name, team_id, position, goals, assists, points, shots, toi, plus_minus, ...

### `odds_snapshots` *(to build)*
snapshot_id, sport, game_id, game_date, bookmaker, market, outcome, price, point, fetched_at

### Tennis tables
- `tennis_matches` — raw Sackmann match rows (winner/loser format), PK (tour, tourney_id, match_num)
- `tennis_matches_clean` — cleaned + score-parsed (total_games, game_margin, completed flag)
- `tennis_court_speed`, `tennis_locations`, `tennis_weather` — curated/derived context
- `tennis_features` — the wide one-big-table (one row per match-perspective; all
  diff + context features + targets won/total_games/game_margin); rebuilt on each train/compare

### Tennis model strategies (`models/compare.py`)
**Separate ATP and WTA models** (artifacts `tennis_atp_*` / `tennis_wta_*`), trained
on per-tour slices of the OBT. Bake-off compared by CV log-loss / Brier / AUC / ROI:
Elo-only baseline, logistic regression, LightGBM, XGBoost (HistGradientBoosting
fallback), PyTorch MLP, and an ensemble. Heavy training (incl. the MLP) runs on Colab
GPU (`notebooks/tennis_train_colab.ipynb`, per-tour); the portable winner is saved for
torch-free local serving. Exact MLP steps: `docs/TRAINING_MLP.md`.

---

## Key Design Decisions
- **LightGBM** for all models — fast, handles tabular data well, built-in feature importance
- **Polars** for feature engineering — faster than pandas for rolling window ops
- **SQLite** for storage — zero infrastructure, easy to inspect, sufficient for this data volume
- **The Odds API** for odds — clean JSON, free tier covers ~500 requests/month
- **Fractional Kelly (25%)** for stake sizing — reduces variance vs. full Kelly
- **Vig removal method:** additive (simple, unbiased for two-outcome markets)

---

## Open Questions
- Which bookmakers to target? (DraftKings, FanDuel, BetMGM cover most of the US market)
- How many seasons of history to train on? (Suggest 5 seasons as baseline)
- Retrain frequency? (Weekly during season is likely sufficient)
- Alert delivery beyond console/CSV? (Telegram bot, email, etc.)
