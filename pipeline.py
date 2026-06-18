"""Shared ingest -> features -> edges -> paper-ledger orchestration.

This is the single implementation of "pull fresh data and price today's edges",
used by BOTH the CLI (``run.py``) and the web app (``webapp`` -> ``POST /api/pull``).
Keeping one copy here is what keeps a terminal ``python run.py`` and the web app's
"Pull data" button in lockstep: same ingest calls, same edge math, same paper ledger,
all written to the **same** SQLite file (``DB_PATH``). Point both at one DB and the two
front-ends can never drift.

Every heavy dependency (nba_api, polars, lightgbm, scipy, ...) is imported lazily inside
the functions, so simply importing this module — as the web app does at startup — pulls
in nothing expensive and cannot fail on a missing optional dep.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Callable

log = logging.getLogger("pipeline")

# Season/year coverage for the historical backfill (mirrors the old run.py constants).
SEASONS = ["20212022", "20222023", "20232024", "20242025", "20252026"]
NBA_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TENNIS_YEARS = list(range(2015, 2027))
TENNIS_TOURS = ["atp", "wta"]

DEFAULT_LEDGER = "data/sims/paper_ledger.csv"

Progress = Callable[[str], None]


def _emit(progress: Progress | None, msg: str) -> None:
    if progress is not None:
        progress(msg)
    log.info(msg)


# ── ingest stages (parameterized by db_path; were run.py module functions) ───────

def ingest_history(sport: str, db_path: str) -> None:
    print(f"\n[ingest] Pulling historical data for {sport.upper()}...")
    if sport == "nba":
        from ingestion.nba import fetch_seasons, fetch_player_stats
        fetch_seasons(NBA_SEASONS, db_path)
        fetch_player_stats(NBA_SEASONS, db_path)
    elif sport == "tennis":
        from ingestion.tennis import fetch_seasons, fetch_weather
        from ingestion.tennis_clean import clean
        fetch_seasons(TENNIS_YEARS, TENNIS_TOURS, db_path)
        fetch_weather(db_path)
        clean(db_path)
    else:
        from ingestion.nhl import fetch_seasons
        fetch_seasons(SEASONS, db_path)


def ingest_recent(sport: str, db_path: str) -> None:
    print(f"\n[ingest] Pulling recent games for {sport.upper()}...")
    if sport == "nba":
        from ingestion.nba import fetch_recent_games
        fetch_recent_games(days_back=7, db_path=db_path)
    elif sport == "tennis":
        from ingestion.tennis import fetch_recent, fetch_weather
        from ingestion.tennis_clean import clean
        fetch_recent(db_path, TENNIS_TOURS)
        fetch_weather(db_path)
        clean(db_path)
    else:
        from ingestion.nhl import fetch_recent_games
        fetch_recent_games(days_back=7, db_path=db_path)


def ingest_injuries(sport: str, db_path: str) -> None:
    print(f"\n[ingest] Pulling injury report for {sport.upper()}...")
    from ingestion.injuries import fetch_injuries
    fetch_injuries(sport, db_path)


def ingest_odds(sport: str, db_path: str) -> None:
    print(f"\n[ingest] Pulling odds for {sport.upper()}...")
    if sport == "tennis":
        from ingestion.odds import fetch_tennis_odds
        fetch_tennis_odds(db_path)
        return
    from ingestion.odds import fetch_odds, fetch_props
    fetch_odds(sport, ["moneyline", "spread", "totals"], db_path)
    fetch_props(sport, db_path)


def find_edges(sport: str, db_path: str, *, min_edge: float, bankroll: float) -> list[dict]:
    print(f"\n[edge] Finding edges for {sport.upper()}...")

    if sport == "tennis":
        from edge.calculator import find_tennis_edges
        return find_tennis_edges(db_path, min_edge=min_edge, bankroll=bankroll)

    from features.pipeline import (
        build_nba_game_features, build_nba_player_features,
        build_nhl_game_features, build_nhl_player_features,
        add_injury_features, add_teammate_wowy_features, add_market_features,
    )
    from edge.calculator import find_moneyline_edges, find_totals_edges, find_prop_edges

    if sport == "nba":
        game_df = build_nba_game_features(db_path)
        player_df = build_nba_player_features(db_path)
    else:
        game_df = build_nhl_game_features(db_path)
        player_df = build_nhl_player_features(db_path)

    game_df = add_injury_features(game_df, sport, db_path)
    game_df = add_market_features(game_df, sport, db_path)
    player_df = add_teammate_wowy_features(player_df, sport, db_path)

    edges: list[dict] = []
    edges += find_moneyline_edges(game_df, sport, db_path, min_edge=min_edge, bankroll=bankroll)
    edges += find_totals_edges(game_df, sport, db_path, bankroll=bankroll)
    edges += find_prop_edges(player_df, sport, db_path, bankroll=bankroll)

    return sorted(edges, key=lambda x: x["edge"], reverse=True)


# ── shared concurrency + paper-ledger helpers ───────────────────────────────────

def enable_wal(db_path: str) -> None:
    """Switch the DB to WAL journaling so a terminal ``run.py`` and a web-app pull can
    touch the same file without readers blocking the writer. WAL is a persistent,
    database-level setting, so doing this once benefits every later connection
    (including the CLI's). Best-effort: a missing DB or a transient lock is ignored."""
    if not os.path.exists(db_path):
        return
    try:
        con = sqlite3.connect(db_path, timeout=30)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA busy_timeout=30000")
        finally:
            con.close()
    except sqlite3.Error:
        log.warning("could not enable WAL on %s", db_path, exc_info=True)


def _read_quota(db_path: str) -> dict | None:
    """Latest Odds API quota row after the pull, or None if not tracked."""
    try:
        con = sqlite3.connect(db_path)
        row = con.execute(
            "SELECT remaining, used, fetched_at FROM odds_requests_remaining "
            "ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        con.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    return {"remaining": row[0], "used": row[1], "fetched_at": row[2]}


def update_paper(
    edges: list[dict],
    db_path: str,
    *,
    bankroll: float,
    ledger_path: str | None = None,
    sim_min_edge: float | None = None,
) -> dict:
    """Settle finished paper bets against game results, log today's qualifying
    moneyline edges as new open positions, persist the ledger, and return the equity
    summary. Mirrors the ledger block at the tail of ``run.py``'s default run."""
    from models.paper_sim import (
        load_ledger, settle_ledger, append_edges, save_ledger,
        equity_summary, SIM_MIN_EDGE,
    )

    ledger = ledger_path or os.getenv("LEDGER_PATH", DEFAULT_LEDGER)
    start = bankroll
    min_edge = sim_min_edge if sim_min_edge is not None else SIM_MIN_EDGE
    rows = settle_ledger(load_ledger(ledger), db_path, start=start)
    rows = append_edges(rows, edges, start, min_edge=min_edge, start=start)
    save_ledger(rows, ledger)
    return equity_summary(rows, start)


# ── the shared pull (used by run.py's default path and the web app) ──────────────

def run_pull(
    sports: list[str],
    db_path: str,
    *,
    bankroll: float,
    min_edge: float,
    fetch_odds: bool = True,
    ingest_games: bool = True,
    update_injuries: bool = True,
    update_paper_ledger: bool = True,
    ledger_path: str | None = None,
    sim_min_edge: float | None = None,
    progress: Progress | None = None,
) -> dict:
    """Pull fresh data for ``sports`` into ``db_path`` and price today's edges.

    Runs the same ingest -> feature -> edge -> paper-ledger flow as the tail of
    ``run.py``'s default run (keep the two in sync). Each ingest stage is isolated so a
    single failure (e.g. a cloud IP blocked by stats.nba.com, or a missing
    ``ODDS_API_KEY``) is recorded in ``stages`` and the pull still prices edges from
    whatever data is present. Returns a JSON-serialisable summary; the CLI prints it
    and the web app returns it.
    """
    started = datetime.now(timezone.utc)
    enable_wal(db_path)
    stages: list[dict] = []

    def stage(sport: str | None, name: str, fn: Callable[[], None], *, skip_env=False) -> None:
        label = f"{sport} {name}" if sport else name
        _emit(progress, f"pull: {label}...")
        try:
            fn()
            stages.append({"sport": sport, "stage": name, "status": "ok"})
        except EnvironmentError as e:  # e.g. ODDS_API_KEY missing -> skip, don't fail
            status = "skipped" if skip_env else "error"
            stages.append({"sport": sport, "stage": name, "status": status, "detail": str(e)})
            _emit(progress, f"pull: {label} {status}: {e}")
        except Exception as e:  # keep going; one broken stage shouldn't sink the pull
            log.warning("pull stage %s failed", label, exc_info=True)
            stages.append({"sport": sport, "stage": name, "status": "error", "detail": str(e)})
            _emit(progress, f"pull: {label} error: {e}")

    for sport in sports:
        if ingest_games:
            stage(sport, "games", lambda s=sport: ingest_recent(s, db_path))
            if update_injuries and sport != "tennis":
                stage(sport, "injuries", lambda s=sport: ingest_injuries(s, db_path))
        if fetch_odds:
            stage(sport, "odds", lambda s=sport: ingest_odds(s, db_path), skip_env=True)

    edges: list[dict] = []
    edge_errors: list[str] = []
    for sport in sports:
        _emit(progress, f"pull: {sport} edges...")
        try:
            edges += find_edges(sport, db_path, min_edge=min_edge, bankroll=bankroll)
        except FileNotFoundError as e:
            edge_errors.append(f"{sport}: model not trained ({e})")
        except Exception as e:
            log.warning("edge computation failed for %s", sport, exc_info=True)
            edge_errors.append(f"{sport}: {e}")
    edges.sort(key=lambda x: x["edge"], reverse=True)

    paper = None
    if update_paper_ledger:
        try:
            _emit(progress, "pull: settling paper ledger...")
            paper = update_paper(
                edges, db_path, bankroll=bankroll,
                ledger_path=ledger_path, sim_min_edge=sim_min_edge,
            )
        except Exception as e:
            log.warning("paper ledger update failed", exc_info=True)
            stages.append({"sport": None, "stage": "paper", "status": "error", "detail": str(e)})

    finished = datetime.now(timezone.utc)
    return {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": round((finished - started).total_seconds(), 1),
        "sports": list(sports),
        "fetched_odds": fetch_odds,
        "ingested_games": ingest_games,
        "stages": stages,
        "edges": edges,
        "n_edges": len(edges),
        "edge_errors": edge_errors,
        "quota": _read_quota(db_path),
        "paper": paper,
    }
