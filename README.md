# Sports Edge

ML pipeline that ingests NBA and NHL historical data, trains LightGBM models per betting market, then compares model probabilities against live sportsbook odds to flag positive expected-value bets.

```
ingestion/ → features/ → models/ → edge/
SQLite       Polars DFs   .pkl      stdout / CSV
```

---

## Setup

**Requirements:** Python 3.11+

```bash
git clone <repo>
cd sports-edge
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env`:

```env
ODDS_API_KEY=your_key_here      # from the-odds-api.com (free tier: ~500 req/month)
BANKROLL=1000                   # dollars — used for Kelly stake sizing
MIN_EDGE=0.03                   # minimum edge threshold (3%)
DB_PATH=data/sports.db          # SQLite file path
```

---

## Usage

### First run — pull historical data

```bash
python run.py --ingest-history --sport nba   # ~5 seasons, takes a few minutes
python run.py --ingest-history --sport nhl
```

### Train models

```bash
python run.py --train              # both sports
python run.py --train --sport nba  # NBA only
```

### Daily run — find edges

```bash
python run.py                  # ingest recent games + odds, find edges, print alerts
python run.py --sport nba      # NBA only
python run.py --no-odds        # skip odds fetch, use cached (no book comparison)
```

### Dashboard

```bash
streamlit run dashboard.py
```

### Cron (daily pre-game)

```cron
0 11 * * * cd /path/to/sports-edge && python run.py >> logs/daily.log 2>&1
```

---

## Data Sources

| Source | Data |
|---|---|
| `nba_api` (LeagueGameLog / PlayerGameLogs) | NBA team and player game logs |
| NHL Stats API (api.nhle.com) | NHL team and player game logs |
| The Odds API | Live moneyline, spread, totals, and player prop odds |
| ESPN (scraped) | NBA/NHL injury reports |

---

## Data Tracked

### SQLite tables

**`nba_team_games`**
`game_id`, `team_id`, `season_id`, `game_date`, `matchup`, `wl`, `pts`, `fgm`, `fga`, `fg_pct`, `fg3m`, `fg3a`, `reb`, `ast`, `stl`, `blk`, `tov`, `plus_minus`

**`nba_player_games`**
`game_id`, `player_id`, `season_year`, `game_date`, `player_name`, `team_id`, `min`, `pts`, `reb`, `ast`, `stl`, `blk`, `tov`, `fg_pct`, `fg3_pct`

**`nhl_team_games`**
`game_id`, `team_id`, `season`, `game_date`, `matchup`, `wl`, `goals`, `shots`, `pp_goals`, `pp_opps`, `pim`, `hits`, `blocks`

**`nhl_player_games`**
`game_id`, `player_id`, `season`, `game_date`, `player_name`, `team_id`, `position`, `goals`, `assists`, `points`, `shots`, `toi`, `plus_minus`

**`odds_snapshots`**
`snapshot_id`, `sport`, `game_id`, `game_date`, `bookmaker`, `market`, `outcome`, `price`, `point`, `fetched_at`

**`injuries`**
`sport`, `player_name`, `team`, `status`, `detail`, `fetched_at`

---

## Models

One LightGBM model per market, trained on rolling historical seasons (default: 5).

| Market | Target | Type |
|---|---|---|
| Moneyline | Home win (0/1) | Binary classifier |
| Spread | Home margin of victory | Regressor |
| Totals | Total points scored | Regressor |
| Player props | Per-stat value (pts/reb/ast/etc.) | Regressor per stat |

Serialized to `models/artifacts/<sport>_<market>.pkl`.

### Features

**Team features** (rolling 5 and 10-game windows): points, FG%, rebounds, assists, turnovers, rest days, home/away flag, win streak, injury impact score.

**Player features** (rolling 5 and 10-game windows): per-stat averages, minutes trend, days rest, home/away splits.

**Odds features**: implied probability, vig-adjusted fair odds.

### Evaluation metrics

- Moneyline: AUC, Brier score, ROI backtest
- Spread / Totals / Props: MAE, RMSE, ROI backtest

```bash
python -c "from models.evaluate import evaluate_all; evaluate_all('nba', 'data/sports.db')"
```

---

## Edge Detection

For each upcoming game with available odds, the pipeline:

1. Converts American odds → raw implied probability
2. Strips vig using the additive method → fair probability
3. Computes edge: `model_prob − fair_prob`
4. Filters bets where `edge >= MIN_EDGE`
5. Sizes stake using quarter-Kelly criterion: `bankroll × 0.25 × kelly_fraction`

Alerts print to stdout and are saved to `data/edges/edges_YYYY-MM-DD.csv`.

### Sample output

```
======================================================================
  EDGES FOUND: 3
======================================================================

  NBA | moneyline
  Game:  BOS vs MIA
  Bet:   BOS  (-140)
  Edge:  4.2%  |  Kelly stake: $18.50
  Model: 68.3%  Book fair: 64.1%
```

---

## Project Structure

```
sports-edge/
├── run.py                  # top-level orchestrator
├── dashboard.py            # Streamlit dashboard
├── ingestion/
│   ├── nba.py              # NBA team + player logs (nba_api)
│   ├── nhl.py              # NHL team + player logs (NHL Stats API)
│   ├── odds.py             # live odds (The Odds API)
│   └── injuries.py         # injury reports
├── features/
│   └── pipeline.py         # rolling feature engineering (Polars)
├── models/
│   ├── train.py            # train + serialize all models
│   ├── evaluate.py         # AUC / Brier / MAE / RMSE + ROI backtest
│   └── artifacts/          # .pkl model files (gitignored)
├── edge/
│   ├── calculator.py       # odds math + Kelly stake
│   └── alerts.py           # stdout + CSV output
├── data/
│   ├── sports.db           # SQLite database (gitignored)
│   └── edges/              # dated CSV alert files
└── notebooks/              # exploratory analysis
```

---

## Key Design Decisions

- **LightGBM** — fast gradient boosting, handles tabular data well, built-in feature importance
- **Polars** — faster than pandas for rolling window operations at this data volume
- **SQLite** — zero infrastructure, easy to inspect, sufficient for this scale
- **The Odds API** — clean JSON, free tier covers ~500 requests/month
- **Quarter Kelly (25%)** — reduces variance vs. full Kelly while preserving edge-proportional sizing
- **Additive vig removal** — simple, unbiased method for two-outcome markets
