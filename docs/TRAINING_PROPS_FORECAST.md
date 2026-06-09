# Player prop forecaster (Tier 4 — cloud)

A **global probabilistic time-series forecaster** (DeepAR by default; a Temporal
Fusion Transformer is a drop-in alternative) trained across *all* players at
once. For each player's next game it predicts a full distribution per stat, so
the props edge path can compare a mean **and** a sigma against the over/under
line instead of a single point estimate.

This is the one place in the pipeline where forecasting a *quantity* is the
actual betting question, which is why it gets a sequence model rather than the
cross-sectional LightGBM the other markets use. Like the rest of the heavy
compute, it trains on Colab GPU and serves a torch-free portable artifact.

## Why not local

DeepAR/TFT training is GPU-bound and pulls in torch — exactly the kind of heavy
job kept off the laptop. The notebook distils its next-game predictions into a
plain table so local serving never imports torch.

## Workflow

1. **Local — export sequences** (per sport):
   ```bash
   python run.py --export-sequences --sport nba
   python run.py --export-sequences --sport nhl
   ```
   Writes `data/exports/{sport}_player_sequences.parquet` — one row per
   player-game with the prop-stat targets and known covariates (minutes, rest,
   home/away, rolling / EWMA / Holt form).

2. **Upload** both parquets to Drive `MyDrive/sports-edge/exports/`.

3. **Colab** — open `notebooks/props_forecast_colab.ipynb`, set Runtime → GPU,
   Run All. It trains one global DeepAR per stat, forecasts each player's next
   game (mean + sigma from sampled paths), and writes portable
   `{sport}_prop_forecast.pkl` to `MyDrive/sports-edge/artifacts/`.

4. **Download** the `.pkl` files into `data/exports/artifacts/`, then:
   ```bash
   python run.py --import-models
   ```
   (`import_models` already globs `nba_*` / `nhl_*`, so the forecast artifacts
   come across with everything else.)

## Artifact schema

`models/artifacts/{sport}_prop_forecast.pkl`:

```python
{
    "sport": "nba",
    "asof":  "2026-06-01",            # last date seen in training
    "table": polars.DataFrame[player_id, stat, pred_mean, pred_sigma],
}
```

## Serving (torch-free)

`models/prop_forecast.py`:

- `load_prop_forecast(sport)` — returns the dict above, or `None` if not imported.
- `prop_forecast_features(df, sport)` — left-joins `fc_{stat}_mean` /
  `fc_{stat}_sigma` onto player feature rows by `player_id`. It is a **no-op**
  until the artifact is present, so the pipeline runs unchanged before the first
  cloud run.

To wire it into edge detection, call `prop_forecast_features` on the player
feature frame in `run.find_edges` and have `edge.calculator.find_prop_edges`
prefer `fc_{stat}_mean` (with `fc_{stat}_sigma` for the over/under probability)
when present, falling back to the LightGBM prop point estimate otherwise. That
last hook is intentionally left for when the first cloud artifact exists.
