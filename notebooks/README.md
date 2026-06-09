# Cloud training tier (notebooks/)

Heavy compute (deep-learning training, the multi-model bake-off) runs in the
cloud, not on the laptop. The laptop only does light I/O: ingest, clean, build
features, export, and find edges.

## tennis_train_colab.ipynb

End-to-end tennis model training on a Colab GPU runtime.

### Workflow

1. **Local — export features:**
   ```bash
   python run.py --export-features --sport tennis
   ```
   Writes `data/exports/tennis_features.parquet` (the one-big-table).

2. **Upload** that parquet to your Google Drive at `MyDrive/sports-edge/exports/`.

3. **Colab** — open `tennis_train_colab.ipynb`, set Runtime → GPU, and Run All.
   It installs `torch`/`xgboost`, runs `models.compare.run_comparison` **per tour**
   (Elo / logistic / LightGBM / XGBoost / MLP / ensemble for ATP and WTA separately),
   trains the per-tour totals + spread regressors, and writes artifacts
   (`tennis_atp_*`, `tennis_wta_*`) to `MyDrive/sports-edge/artifacts/`.

   For exact MLP training steps see `docs/TRAINING_MLP.md`.

4. **Download** the artifacts into `data/exports/artifacts/` locally.

5. **Local — import + find edges:**
   ```bash
   python run.py --sport tennis --import-models
   ```

## team_train_colab.ipynb

Same flow for the **NBA and NHL** moneyline bake-off (LightGBM / XGBoost / MLP /
ensemble). The exported game-features parquet already carries the injury / WOWY
absence features.

### Workflow

1. **Local — export features (once per sport):**
   ```bash
   python run.py --export-features --sport nba
   python run.py --export-features --sport nhl
   ```
   Writes `data/exports/nba_game_features.parquet` and `nhl_game_features.parquet`.

2. **Upload** both parquets to `MyDrive/sports-edge/exports/`.

3. **Colab** — open `team_train_colab.ipynb`, set Runtime → GPU, Run All. It runs
   `models.compare_team.run_team_comparison` for each sport and writes
   `{sport}_moneyline.pkl` + `{sport}_model_comparison.csv` to
   `MyDrive/sports-edge/artifacts/`.

4. **Download** the artifacts into `data/exports/artifacts/`, then:
   ```bash
   python run.py --import-models
   ```

## props_forecast_colab.ipynb

Global probabilistic **player-prop forecaster** (DeepAR / TFT) — the Tier-4 time
series model. Trains one model across all players and emits a per-(player, stat)
next-game distribution (mean + sigma) for NBA and NHL props.

### Workflow

1. **Local — export sequences:**
   ```bash
   python run.py --export-sequences --sport nba
   python run.py --export-sequences --sport nhl
   ```
   Writes `data/exports/{sport}_player_sequences.parquet`.

2. **Upload** both to `MyDrive/sports-edge/exports/`.

3. **Colab** — open `props_forecast_colab.ipynb`, GPU runtime, Run All. Writes
   torch-free `{sport}_prop_forecast.pkl` to `MyDrive/sports-edge/artifacts/`.

4. **Download** into `data/exports/artifacts/`, then `python run.py --import-models`.

Full details + serving hook: `docs/TRAINING_PROPS_FORECAST.md`.

## tuning_colab.ipynb

Per-market **Optuna** hyperparameter search under the same `TimeSeriesSplit` CV
the trainer uses (CPU is fine). Writes `{sport}_tuned_params.json` for the laptop
to retrain with. See `docs/ACCURACY.md`.

## poisson_nhl_colab.ipynb

Fits the **bivariate-Poisson** NHL goals model (`models/poisson_nhl.py`) — one
coherent score distribution that yields consistent moneyline / totals / puck-line.
Writes a numpy-only portable `nhl_poisson.pkl`.

### Notes

- No API keys are needed in the cloud (odds ingestion stays on the laptop).
- The production server artifact `tennis_moneyline.pkl` is always saved as the
  best **portable** model (LightGBM or a numpy logistic) so local edge detection
  never needs torch. If a neural net wins the bake-off it is reported but the
  portable model is used for serving.
- The macOS-local OpenMP hang seen with LightGBM + sklearn does not occur on
  Colab (Linux). If you do run the bake-off locally, prefix with
  `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`.
