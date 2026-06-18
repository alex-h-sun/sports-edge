# ── stage 1: build the React SPA ────────────────────────────────────────────────
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build          # -> /web/dist

# ── stage 2: python serving image ───────────────────────────────────────────────
FROM python:3.11-slim AS serve
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    STATIC_DIR=/app/webapp/static \
    SNAPSHOT_DATA_DIR=/data \
    DB_PATH=/data/sports.db \
    ARTIFACTS_DIR=/data/artifacts \
    LEDGER_PATH=/data/sims/paper_ledger.csv \
    PORT=8000

# libgomp1 is required by lightgbm at import/inference time
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Source first so `pip install .[serve]` can build the package (deps live in pyproject).
COPY pyproject.toml ./
COPY edge/ ./edge/
COPY features/ ./features/
COPY models/ ./models/
COPY ingestion/ ./ingestion/
COPY webapp/ ./webapp/
COPY run.py pipeline.py ./
RUN pip install ".[serve]"

# built SPA from stage 1
COPY --from=web /web/dist ./webapp/static

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
