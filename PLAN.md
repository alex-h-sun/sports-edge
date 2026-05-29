# Sports Edge — Project Plan

## Goal
Scrape historical sports data, train ML models to predict game/player outcomes, and identify positive expected-value bets by comparing model probabilities against live sportsbook odds.

## Sports & Markets
- **Sports:** NBA, NHL
- **Markets:** Moneyline, Spread, Totals (O/U), Player Props

---

## Architecture

```
ingestion/ → features/ → models/ → edge/
   ↓              ↓           ↓         ↓
SQLite        Polars DFs   .pkl      stdout / CSV
```

### Data sources
| Layer | NBA | NHL |
|---|---|---|
| Game logs | `nba_api` (LeagueGameLog) | NHL Stats API (api.nhle.com) |
| Player stats | `nba_api` (PlayerGameLogs) | NHL Stats API |
| Live odds | The Odds API | The Odds API |

---

## Implementation Phases

### Phase 1 — Ingestion ✅ / 🔲
- [x] `ingestion/nba.py` — team + player game logs, SQLite upsert
- [ ] `ingestion/nhl.py` — team + player game logs, SQLite upsert
- [ ] `ingestion/odds.py` — live odds via The Odds API (all 4 markets)

### Phase 2 — Feature Engineering
- [ ] `features/pipeline.py`
  - Team features: rolling 5/10-game averages (PTS, FG%, REB, AST, TOV), rest days, home/away flag, win streak
  - Player features: rolling 5/10-game averages per stat, minutes trend, days rest, home/away splits
  - Odds features: opening vs. closing line movement, implied probability, vig-adjusted fair odds

### Phase 3 — Models
One model per market. All trained with LightGBM, cross-validated on historical seasons.

| Market | Target | Model type |
|---|---|---|
| Moneyline | Home win (0/1) | Binary classifier |
| Spread | Home margin of victory | Regressor |
| Totals | Total points scored | Regressor |
| Player props | Player stat (pts/reb/ast/etc.) | Regressor per stat |

- [ ] `models/train.py` — train + serialize each model to `models/artifacts/`
- [ ] `models/evaluate.py` — AUC / Brier / MAE / RMSE + ROI backtest simulation

### Phase 4 — Edge Detection
- [ ] `edge/calculator.py`
  - Convert American odds → implied probability
  - Strip vig (additive method)
  - Edge = model_prob − fair_prob
  - Kelly stake = (edge × odds − (1 − edge)) / odds × fraction
- [ ] `edge/alerts.py` — print and save positive-EV bets

### Phase 5 — Runner / CLI
- [ ] `run.py` — top-level script: ingest recent games → build features → load models → find edges → print/save alerts
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
