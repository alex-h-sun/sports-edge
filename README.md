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

### Dashboard (local Streamlit)

```bash
streamlit run dashboard.py
```

The dashboard includes a **Bankroll Simulator** section that renders the paper-trading
equity curve, headline stats, and full bet ledger.

### Web app (deployable)

A web version lives in `webapp/` (Starlette JSON API) + `frontend/` (React + Vite + TS
SPA). It is **read-only**: it serves edges, the manual calculator, injuries, the odds
quota, and the bankroll curve from a published snapshot of `sports.db` + artifacts. The
heavy ingest/training pipeline stays offline (it cannot run from a cloud IP anyway), and
the whole app sits behind a single shared password.

```
offline:  python run.py  -> sports.db + models/artifacts
          python scripts/publish_snapshot.py --url s3://bucket/prefix   (or file:///dir)

online:   one container -> Starlette API + built SPA, reads /data (snapshot volume)
```

Run it locally against your existing data:

```bash
# 1) backend (dev): http://localhost:8000
pip install -e ".[serve]"
APP_PASSWORD=secret SESSION_SECRET=dev uvicorn webapp.main:app --reload

# 2) frontend (dev): http://localhost:5173, proxies /api -> :8000
cd frontend && npm install && npm run dev
```

Or the full image (serves the built SPA + API on one port, parity with production):

```bash
APP_PASSWORD=secret SESSION_SECRET=dev docker compose up --build   # http://localhost:8000
```

**Deploy (Render / Railway / Fly.io):** the multi-stage `Dockerfile` builds the SPA and
the Python serving image; `render.yaml` provisions a web service with a 1 GB persistent
disk mounted at `/data`. Set `APP_PASSWORD`, `SESSION_SECRET`, and (for snapshot sync)
`SNAPSHOT_URL` + S3/R2 credentials. On boot and on `POST /api/admin/reload`, the app pulls
the latest published snapshot onto the volume.

**Snapshot publishing** packages a WAL-checkpointed copy of `sports.db`, the `.pkl`
artifacts, and the paper ledger into a `.tar.gz` (+ a `latest.json` manifest) and uploads
it to S3-compatible storage (Cloudflare R2 works via `S3_ENDPOINT_URL`) or a local/NFS
directory. Re-run it after each `python run.py --train`.

> Note: live moneyline/totals/props edges need **fresh odds and upcoming games** in the
> snapshot, so publish near game time. The **manual calculator works on any snapshot**
> because you supply the odds. Built on Starlette (FastAPI's foundation) to keep the
> serving image lean — no torch, no pydantic.

### Bankroll simulator (forward paper-trading)

Every normal `python run.py` also runs a forward paper-trading ledger: it settles any
previously-open bets against the actual game results, then logs new moneyline edges
(≥ 7% by default) as open positions. Starting from a fixed $1000 bankroll, each bet is
staked flat quarter-Kelly, and a running equity curve is persisted to
`data/sims/paper_ledger.csv`.

```bash
python run.py                  # paper-trading runs automatically (on by default)
python run.py --no-paper       # skip the ledger update for this run
python run.py --sim-status     # print the bankroll summary + open bets (no ingest/betting)
python run.py --sim-history    # print every bet the sim has made (no ingest/betting)
python run.py --sim-min-edge 0.05   # override the logging threshold (default 0.07)
```

It is **forward-only** (not a historical backtest): the free Odds API tier stores only
live odds with no per-game history to replay, so an honest P&L can only accrue from real
live edges going forward. The curve starts empty and grows as you run the pipeline over
time. v1 settles **moneyline only**; stale unmatchable bets are voided after 5 days.

### Cron (daily pre-game)

```cron
0 11 * * * cd /path/to/sports-edge && python run.py >> logs/daily.log 2>&1
```

### Backtest on a held-out season

```bash
python run.py --backtest --sport nba    # train on < 2025-10-01, score the 2025-26 season
python run.py --backtest --sport all    # NBA + NHL + tennis
```

`--backtest` trains every market on data *before* the held-out season
(`HOLDOUT_START = 2025-10-01`, tennis `2025-01-01`) and reports metrics on the
never-seen test season — an honest estimate of real-world accuracy, stricter than
the rolling `TimeSeriesSplit` CV. It saves `_holdout`-suffixed artifacts so it
**never overwrites the production models**, reads the existing DB (no ingest), and
finds no edges. For live betting use plain `--train`, which fits on all data
including the current season.

### How often to run (don't skip more than 7 days)

A plain `python run.py` re-ingests only a **rolling 7-day window** of games
(`fetch_recent_games(days_back=7)`), deduped by primary key via `INSERT OR REPLACE`
— so the full history is never re-pulled, and new games are the only ones added.

The consequence: **run it at least once every 7 days** to keep the game-log table
gap-free. Any game that finished more than 7 days before a run is never requested,
and the hole does not self-heal on later runs (the window has moved past it).

- **Daily** (the cron above) — ideal; 6 days of overlap slack absorbs missed days.
- **≤ 7 days** — fine, no gaps.
- **> 7 days** — you'll miss games. Backfill with
  `python run.py --ingest-history --sport <sport>` (or temporarily raise `days_back`).

This 7-day rule is only about keeping **historical game logs** complete for training.
**Odds are separate**: the free tier stores only live pulls and can't be backfilled,
so for actual betting run near game time on the days you bet — not just weekly.

### Using The Odds API efficiently (free tier)

The free tier is **500 requests per month** (resets monthly), not per day. The
catch: a request is **not** one HTTP call — your quota is debited by
**markets × regions**. One `/odds` pull for the `us` region with `h2h,spreads,totals`
(3 markets) costs **3**, two regions costs **6**. Player props cost **1 request per
event per prop market** (`fetch_props` loops events × prop keys), so they are by far
the most expensive call. Listing sports (`/sports`) is free.

The repo tracks your balance for you: every response's `x-requests-remaining` /
`x-requests-used` headers are logged to the `odds_requests_remaining` table and
printed after each fetch (`ingestion/odds.py:_log_quota`). Check it without spending
a request via `get_quota(db_path)`.

To stay under 500/month:

- **Request only the markets you price.** Drop `spreads` if you only bet totals —
  each market you add is +1 per call.
- **One region (`us`).** Add `uk`/`eu` only for line-shopping; each region multiplies
  the cost.
- **Run props sparingly.** A full props sweep of one slate is `events × prop_markets`
  requests (e.g. 8 games × 5 NBA prop keys = 40). Pull props once near lock, not on
  every cron tick.
- **Batch into one daily pull** instead of polling — the cron above runs once at 11:00.
- **Never use the historical-odds endpoints on the free tier.** They cost **10×** per
  market. This is also why the `mkt_*` model features stay null at first: the pipeline
  only stores live pulls and does not backfill odds history.
- **Skip the odds fetch when iterating** with `python run.py --no-odds` (uses cached
  snapshots, burns zero quota).

A rough budget: NBA + NHL moneyline+totals (2 markets, 1 region) ≈ 4 requests/day ≈
120/month — leaving headroom for occasional props. Adding daily props quickly exhausts
the tier, so reserve them for games you actually intend to bet.

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

**Injury / WOWY features** (with-or-without-you): absences are inferred directly from the boxscores (a seasonal-roster player with no row for a game did not play). From that backbone the team models get `wowy_margin_delta` (the team's historical scoring-margin swing from the players currently out), `key_players_out`, and `out_min_share` (plus opponent mirrors); the player-props models get `teammate_out_min_share` and `lead_teammate_out` (how a player's role opens up when a key teammate is out). All with/without splits use strictly prior games to avoid leakage.

**Odds features**: implied probability, vig-adjusted fair odds.

### Deep-learning bake-off

NBA/NHL moneyline can also be trained as a multi-model bake-off (LightGBM, XGBoost, a PyTorch MLP, and an ensemble) via `models/compare_team.py`, mirroring the tennis bake-off. Heavy/MLP training runs on Colab GPU (`notebooks/team_train_colab.ipynb`); the best **portable** model is saved for torch-free local serving. See `docs/TRAINING_MLP.md`.

```bash
python run.py --export-features --sport nba   # dump features for Colab
python run.py --compare --sport nba           # run the bake-off (local or Colab)
python run.py --import-models                 # pull cloud-trained artifacts back
```

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
│   ├── paper_sim.py        # forward paper-trading bankroll ledger
│   └── artifacts/          # .pkl model files (gitignored)
├── edge/
│   ├── calculator.py       # odds math + Kelly stake
│   └── alerts.py           # stdout + CSV output
├── data/
│   ├── sports.db           # SQLite database (gitignored)
│   ├── edges/              # dated CSV alert files
│   └── sims/               # paper_ledger.csv equity curve (gitignored)
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
