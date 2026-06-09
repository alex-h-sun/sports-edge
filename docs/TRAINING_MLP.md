# How to train the MLP (deep-learning models)

This document describes **exactly** how the deep-learning MLP is trained, where it
lives, and the precise commands to run it. The MLP is one of the families in the
match/match-winner **bake-off**; it is trained and scored every time the
comparison runs. The same MLP family is shared across two bake-offs:

- **Tennis** (`models/compare.py`) — separate ATP / WTA match-winner models.
- **NBA / NHL** (`models/compare_team.py`) — per-sport home-win moneyline models;
  see Section 7. It imports `_fit_mlp` / `_impute_standardize` from
  `models/compare.py`, so the architecture below is identical.

---

## 1. Where the MLP lives

- **Code:** `models/compare.py`, function `_fit_mlp(X_tr, y_tr, X_val)`.
- **Architecture:** a feed-forward net built per fold:
  ```
  Linear(n_features -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.3)
  Linear(128 -> 64)         -> BatchNorm1d -> ReLU -> Dropout(0.3)
  Linear(64 -> 1)           -> (sigmoid at inference)
  ```
  - Loss: `BCEWithLogitsLoss`; Optimizer: `Adam(lr=1e-3, weight_decay=1e-5)`; 60 epochs full-batch.
  - Inputs are **median-imputed + standardized** per fold (`_impute_standardize`), fit on the
    training fold only (no leakage).
  - Device: `cuda` if available, else `cpu` (`torch.device(...)`). No code change needed for GPU.
- **It runs only if `torch` is installed.** Without torch it is skipped with a printed note;
  every other family still runs.

The MLP is trained **separately for ATP and WTA** because `run_comparison` filters the
one-big-table by tour before training.

---

## 2. Prerequisites (once)

1. Ingest history and build the features (laptop is fine — this is light I/O):
   ```bash
   cd sports-edge
   python run.py --sport tennis --ingest-history     # Sackmann ATP+WTA + weather + clean
   ```
2. Install the heavy deps that include PyTorch:
   ```bash
   pip install -e ".[tennis]"     # adds torch + xgboost
   ```
   On Colab, `torch` is preinstalled; the notebook's first cell also `pip install`s it.

---

## 3. Option A — train the MLP in the cloud (recommended)

This is the intended path (no heavy compute on the laptop).

1. **Export the features locally:**
   ```bash
   python run.py --sport tennis --export-features
   # writes data/exports/tennis_features.parquet
   ```
2. **Upload** `data/exports/tennis_features.parquet` to Google Drive at
   `MyDrive/sports-edge/exports/`.
3. **Open** `notebooks/tennis_train_colab.ipynb` in Colab, set **Runtime → Change runtime
   type → GPU**, and **Run all**. Cell 5 runs the bake-off per tour:
   ```python
   from models.compare import run_comparison
   tables = {t: run_comparison(features_path=FEATURES, tour=t) for t in ('atp', 'wta')}
   ```
   The MLP is trained on GPU inside `run_comparison` for each tour and appears as the
   `mlp` row in the printed comparison table.
4. Artifacts are copied to `MyDrive/sports-edge/artifacts/`. **Download** them into
   `data/exports/artifacts/` locally, then:
   ```bash
   python run.py --sport tennis --import-models
   ```

---

## 4. Option B — train the MLP locally

Only if you accept local compute.

```bash
cd sports-edge
pip install -e ".[tennis]"

# macOS only: avoid the LightGBM + sklearn OpenMP double-load hang
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# run the per-tour bake-off (trains + scores the MLP for each tour)
python run.py --compare --sport tennis
```

Or call it directly for a single tour:
```bash
python -c "from models.compare import run_comparison; run_comparison(db_path='data/sports.db', tour='atp')"
```

---

## 5. What you get

For each tour, `run_comparison` prints a table ranked by CV log-loss, e.g.:

```
model         auc     brier   log_loss  accuracy  roi     bets
ensemble      0.69    0.221   0.632     0.638     ...     ...
logistic      0.69    0.223   0.638     0.641     ...     ...
mlp           0.68    0.229   0.671     0.636     ...     ...
...
```

and writes `models/artifacts/tennis_{tour}_model_comparison.csv`.

---

## 6. Important: the MLP is evaluated, but not the default serving model

To keep **local edge detection torch-free**, `_persist_winner` saves the best
**portable** model (LightGBM or a numpy logistic) as `tennis_{tour}_moneyline.pkl`.
If the MLP wins the bake-off, that is reported, but the portable model is still what
`find_tennis_edges` loads at serve time.

### To actually serve the MLP

If you want the trained neural net used for edge detection, persist it explicitly and
make sure `torch` is installed wherever you run `python run.py` (edge detection). Add a
branch to `_persist_winner` (or save alongside) that pickles the fitted `nn.Module`
plus its imputation/standardization stats into `tennis_{tour}_moneyline.pkl` with a
`.predict(X)` wrapper that returns calibrated `P(win)`. The serving contract that
`edge/calculator.py` expects is simply:

```python
artifact = {"model": <obj with .predict(X)->prob>, "feature_cols": [...], "tour": "atp"}
```

Keep a probability calibration step (Platt/temperature scaling) before serving so the
edge math (`calc_edge`) stays correct.

---

## 7. NBA / NHL moneyline bake-off (same MLP, team features)

The NBA/NHL bake-off lives in `models/compare_team.py` and reuses the tennis
`_fit_mlp`, `_impute_standardize`, and `NumpyLogistic` directly — there is only one
MLP implementation. It trains on the team-level game features (home perspective,
target = home win), which now include the injury / WOWY absence features
(`wowy_margin_delta`, `key_players_out`, `out_min_share`, their `_opp` mirrors).

### Prerequisites (once)

```bash
cd sports-edge
python run.py --ingest-history --sport nba
python run.py --ingest-history --sport nhl
pip install -e ".[tennis]"   # adds torch + xgboost (the MLP only runs if torch is present)
```

### Option A — cloud (recommended)

```bash
python run.py --export-features --sport nba   # data/exports/nba_game_features.parquet
python run.py --export-features --sport nhl   # data/exports/nhl_game_features.parquet
```

Upload both parquets to `MyDrive/sports-edge/exports/`, open
`notebooks/team_train_colab.ipynb` on a GPU runtime, and Run All. It calls:

```python
from models.compare_team import run_team_comparison
run_team_comparison("nba", features_path=".../nba_game_features.parquet")
run_team_comparison("nhl", features_path=".../nhl_game_features.parquet")
```

Artifacts (`{sport}_moneyline.pkl`, `{sport}_model_comparison.csv`) land in
`MyDrive/sports-edge/artifacts/`. Download them into `data/exports/artifacts/` and:

```bash
python run.py --import-models
```

### Option B — local

```bash
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1   # macOS OpenMP guard
python run.py --compare --sport nba
python run.py --compare --sport nhl
```

As with tennis, the served artifact `{sport}_moneyline.pkl` is always the best
**portable** model (LightGBM or numpy logistic); if the MLP wins it is reported but
not used for torch-free local serving.
