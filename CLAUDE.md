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
streamlit run dashboard.py           # interactive view
```

## Architecture

- `run.py` — orchestrator (ingest → features → predict → edges → alert); holds season lists
- `models/artifacts/*.pkl` — one model per market (moneyline/spread/totals + player props per stat)
- `models/train.py` / `models/evaluate.py` — training and scoring (AUC/Brier/MAE + ROI backtest)
- `edge/calculator.py` — edge math: American odds → implied prob → vig-strip → edge → fractional Kelly stake
- Config comes from `.env` via `python-dotenv`, not code constants
