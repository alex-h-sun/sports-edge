"""Paper-trading bankroll state for the API (reuses models.paper_sim, read-only).

Settles open bets against the snapshot's game logs and summarizes the equity curve,
exactly like the Streamlit "Bankroll Simulator" panel — but never writes the ledger
back (the offline batch owns ledger writes; the server only serves the snapshot).
"""

from webapp import settings


def bankroll_state() -> dict:
    from models.paper_sim import (
        load_ledger, settle_ledger, equity_summary, START_BANKROLL,
    )

    start = settings.BANKROLL or START_BANKROLL
    rows = settle_ledger(load_ledger(settings.LEDGER_PATH), settings.DB_PATH, start=start)
    if not rows:
        return {"start": start, "summary": None, "curve": [], "ledger": []}

    settled = [r for r in rows if r.get("status") in ("won", "lost")]
    settled.sort(key=lambda r: r.get("game_date") or "")
    curve = [{"game_date": r.get("game_date"), "balance": r.get("balance")} for r in settled]
    ledger = sorted(rows, key=lambda r: r.get("placed_date") or "", reverse=True)
    return {
        "start": start,
        "summary": equity_summary(rows, start),
        "curve": curve,
        "ledger": ledger,
    }
