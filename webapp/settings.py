"""Web app configuration, read from the environment (same convention as run.py)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── data + model snapshot (read-only at serve time) ─────────────────────────────
DB_PATH = os.getenv("DB_PATH", "data/sports.db")
# ARTIFACTS_DIR is consumed by edge.calculator; set here too for /api/meta display.
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "models/artifacts")
# Pass this explicitly to paper_sim.load_ledger so the path does not depend on when
# models.paper_sim was first imported (its module-level default binds env at import).
LEDGER_PATH = os.getenv("LEDGER_PATH", "data/sims/paper_ledger.csv")

# ── betting config (mirrors run.py / dashboard.py) ──────────────────────────────
BANKROLL = float(os.getenv("BANKROLL", "1000"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.03"))

# ── auth (single shared password) ───────────────────────────────────────────────
# Empty APP_PASSWORD disables the gate (handy for local dev); set it in production.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "")

# ── serving behaviour ───────────────────────────────────────────────────────────
EDGES_TTL = int(os.getenv("EDGES_TTL", "300"))          # cache computed edges, seconds
SPORTS = ["nba", "nhl", "tennis"]
DEFAULT_SPORTS = ["nba", "nhl"]
IS_PROD = os.getenv("APP_ENV", "dev").lower() in ("prod", "production")

# ── built SPA (Docker copies the Vite build here) ───────────────────────────────
STATIC_DIR = Path(os.getenv("STATIC_DIR") or (Path(__file__).parent / "static"))

# ── snapshot sync (offline batch -> serving volume) ─────────────────────────────
# s3://bucket/prefix or file:///abs/dir; empty disables syncing (manual volume).
SNAPSHOT_URL = os.getenv("SNAPSHOT_URL", "")
# Base dir the snapshot extracts into; DB_PATH / ARTIFACTS_DIR / LEDGER_PATH live here.
SNAPSHOT_DATA_DIR = os.getenv("SNAPSHOT_DATA_DIR") or (os.path.dirname(DB_PATH) or "data")
