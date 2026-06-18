# CLAUDE.md

ML pipeline that ingests NBA/NHL/tennis data, trains per-market models, and flags positive-EV bets vs live sportsbook odds. The pipeline is a linear stage flow: `ingestion/ → features/ → models/ → edge/` (SQLite → Polars DataFrames → `.pkl` artifacts → stdout/CSV alerts). See `PLAN.md` for phase status.

Tennis (ATP+WTA) is player-vs-player with its own ingest (Sackmann match CSVs + Open-Meteo weather + curated court-speed/location CSVs), cleaning stage, surface-Elo/H2H feature engineering, a materialized `tennis_features` one-big-table, and a multi-model bake-off (`models/compare.py`). **ATP and WTA are separate models** (`tennis_atp_*` / `tennis_wta_*`); `find_tennis_edges` auto-selects by the players' tour. Heavy tennis training (DL + bake-off) is designed to run on Colab GPU, not locally — see `notebooks/` and `docs/TRAINING_MLP.md`. Markets: match-winner, totals (game O/U), spread (game handicap).

## Commands

```bash
pip install -e ".[dev]"              # editable install; needs Python ≥3.11
cp .env.example .env                 # set ODDS_API_KEY, BANKROLL, MIN_EDGE, DB_PATH
python run.py                        # full pipeline, NBA + NHL
python run.py --sport all            # include tennis
python run.py --sport nba
python run.py --sport tennis --ingest-history   # first tennis run: Sackmann history + weather + clean
python run.py --train                # retrain models before finding edges
python run.py --sport tennis --export-features  # dump OBT parquet for Colab training
python run.py --sport tennis --import-models    # pull cloud-trained artifacts back
python run.py --no-paper             # skip the paper-trading ledger update for this run
python run.py --sim-status           # print the bankroll-sim summary and exit (no ingest)
python run.py --sim-history          # print every paper-sim bet and exit (no ingest)
streamlit run dashboard.py           # interactive view (incl. Bankroll Simulator section)
uvicorn webapp.main:app --reload     # deployable web app (React SPA + JSON API)
```

The web app's **Pull data** button (`POST /api/pull`) runs the same ingest→edge→paper
flow as `python run.py`, writing the shared `DB_PATH` — so the terminal and the web app
stay in sync (one database). See the "Web app" section in `README.md`.

## Architecture

- `pipeline.py` — shared ingest→features→edge→paper orchestration (`run_pull`, the ingest/`find_edges` helpers); imported by BOTH `run.py` and `webapp` so the CLI and the web app's pull run one implementation
- `run.py` — CLI entry/arg-parsing over `pipeline` (+ backtest/arb/matchup/sim/export modes); holds season lists
- `webapp/` — Starlette JSON API + built React SPA (`frontend/`); read-only snapshot serving plus live `POST /api/pull` (→ `pipeline.run_pull`)
- `models/artifacts/*.pkl` — one model per market (moneyline/spread/totals + player props per stat)
- `models/train.py` / `models/evaluate.py` — training and scoring (AUC/Brier/MAE + ROI backtest)
- `edge/calculator.py` — edge math: American odds → implied prob → vig-strip → edge → fractional Kelly stake
- `models/paper_sim.py` — forward paper-trading ledger: on every run (unless `--no-paper`) it settles finished bets against game-log results and logs new moneyline edges (≥`SIM_MIN_EDGE`, default 7%) as open positions, staked flat quarter-Kelly off a fixed $1000 bankroll. Persists to `data/sims/paper_ledger.csv`. Forward-only because `odds_snapshots` holds no historical odds to backtest against. Moneyline only in v1.
- Config comes from `.env` via `python-dotenv`, not code constants
