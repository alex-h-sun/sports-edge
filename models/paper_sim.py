"""Forward paper-trading bankroll simulator.

A persistent CSV ledger that turns the live edges found on each ``python run.py``
into a running equity curve — "if we'd started with $1000 and bet every >=7%
edge, how much would we have now?". Logging runs by default on every edge-finding
run (opt out with ``--no-paper``).

Why forward-only (not a historical backtest): ``odds_snapshots`` stores only
recent *live* odds, so there are no per-game closing prices for past seasons to
replay against (the free Odds API tier cannot backfill them). The honest path is
to log each real live edge as it happens and settle it against the actual game
result once it finishes — a true P&L curve that accrues over time.

Staking is flat quarter-Kelly: every new bet is sized off the *fixed* starting
bankroll ($1000) via ``edge.calculator.kelly_stake`` (not the running balance and
not the edge record's stake, which was sized off the env BANKROLL). Stake amounts
therefore stay constant as the equity curve moves up or down.

v1 logs and settles **moneyline only** — its result (team / player win-loss) maps
cleanly onto the game-log tables. Totals/props settlement needs the realized
total/stat and a game key the edge record does not carry; left as the documented
next step.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from edge.calculator import american_to_decimal, kelly_stake
from edge.manual import NHL_TEAM_NAMES

# Default to the in-repo ledger; override with LEDGER_PATH so a deployed server can
# read/write the ledger on its snapshot volume (e.g. /data/sims/paper_ledger.csv).
LEDGER_PATH = os.getenv("LEDGER_PATH", "data/sims/paper_ledger.csv")
START_BANKROLL = 1000.0
SIM_MIN_EDGE = 0.07
KELLY_FRACTION = 0.25
MIN_STAKE = 1.0          # dust / ruin floor: skip bets the bankroll can't cover
STALE_DAYS = 5           # an unmatched bet older than this is voided (stake refunded)
# tennis_matches_clean.match_date is the tournament START date (shared by every
# match in the event), so a bet placed mid-tournament can carry a placed_date that
# is *after* its own match's start date. Widen the settlement floor by a fortnight
# to absorb that offset; requiring BOTH players to appear in one row keeps it precise.
TENNIS_DATE_SLACK = 16

# columns persisted to the ledger CSV, in order
FIELDNAMES = [
    "bet_id", "placed_date", "game_date", "sport", "market", "game",
    "selection", "odds", "odds_decimal", "model_prob", "edge", "stake",
    "status", "profit", "balance",
]

_SETTLED = {"won", "lost"}


# ── ledger I/O ──────────────────────────────────────────────────────────────────

def load_ledger(path: str = LEDGER_PATH) -> list[dict]:
    """Read the ledger CSV into a list of typed row dicts (empty if absent)."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with open(p, newline="") as f:
        for raw in csv.DictReader(f):
            rows.append(_coerce(raw))
    return rows


def save_ledger(rows: list[dict], path: str = LEDGER_PATH) -> Path:
    """Write rows back to the ledger CSV (creates parent dir)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FIELDNAMES})
    return p


def _coerce(raw: dict) -> dict:
    """Cast CSV strings to numeric types; blank stays None for profit/balance."""
    def num(v, cast):
        if v is None or v == "":
            return None
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    return {
        "bet_id": raw.get("bet_id", ""),
        "placed_date": raw.get("placed_date", ""),
        "game_date": raw.get("game_date", ""),
        "sport": raw.get("sport", ""),
        "market": raw.get("market", ""),
        "game": raw.get("game", ""),
        "selection": raw.get("selection", ""),
        "odds": num(raw.get("odds"), int),
        "odds_decimal": num(raw.get("odds_decimal"), float),
        "model_prob": num(raw.get("model_prob"), float),
        "edge": num(raw.get("edge"), float),
        "stake": num(raw.get("stake"), float),
        "status": raw.get("status", "open"),
        "profit": num(raw.get("profit"), float),
        "balance": num(raw.get("balance"), float),
    }


# ── bankroll accounting ─────────────────────────────────────────────────────────

def _settle_key(r: dict) -> tuple:
    """Chronological order for settled bets: by game date, then when placed."""
    return (r.get("game_date") or "", r.get("placed_date") or "", r.get("bet_id") or "")


def _recompute_balance(rows: list[dict], start: float = START_BANKROLL) -> None:
    """Recompute the running ``balance`` column over settled rows in place.

    Only won/lost bets move the bankroll; open/unsettled/void rows carry no
    balance (they have not resolved or were refunded).
    """
    settled = sorted((r for r in rows if r.get("status") in _SETTLED), key=_settle_key)
    running = start
    for r in settled:
        running += r.get("profit") or 0.0
        r["balance"] = round(running, 2)
    for r in rows:
        if r.get("status") not in _SETTLED:
            r["balance"] = None


def current_bankroll(rows: list[dict], start: float = START_BANKROLL) -> float:
    """The bankroll available to stake now: balance of the last settled bet."""
    settled = sorted((r for r in rows if r.get("status") in _SETTLED), key=_settle_key)
    if not settled:
        return start
    bal = settled[-1].get("balance")
    return float(bal) if bal is not None else start


# ── appending today's edges ─────────────────────────────────────────────────────

def _bet_id(placed_date: str, market: str, game: str, selection: str) -> str:
    raw = f"{placed_date}|{market}|{game}|{selection}".lower()
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _selection_of(edge: dict) -> str:
    """The bet's pick — strip the trailing ' ML' the edge record appends."""
    bet = str(edge.get("bet", "")).strip()
    return bet[:-3].strip() if bet.upper().endswith(" ML") else bet


def append_edges(
    rows: list[dict],
    edges: list[dict],
    bankroll: float,
    placed_date: str | None = None,
    min_edge: float = SIM_MIN_EDGE,
    start: float = START_BANKROLL,
) -> list[dict]:
    """Append today's qualifying moneyline edges as open bets (quarter-Kelly).

    Every new bet is sized off ``bankroll`` via quarter-Kelly. Callers pass the
    fixed starting bankroll (e.g. $1000) for flat sizing, so stake amounts stay
    constant regardless of the running balance. Re-runs the same day are
    idempotent: a bet already present for (placed_date, market, game, selection)
    is skipped. Stakes below ``MIN_STAKE`` are dropped.
    """
    placed_date = placed_date or date.today().isoformat()
    existing = {r["bet_id"] for r in rows}
    if bankroll < MIN_STAKE:
        return rows

    for e in edges:
        if str(e.get("market", "")).lower() != "moneyline":
            continue  # v1: moneyline only (settleable from game logs)
        if float(e.get("edge", 0.0)) < min_edge:
            continue
        selection = _selection_of(e)
        game = str(e.get("game", ""))
        market = "Moneyline"
        bet_id = _bet_id(placed_date, market, game, selection)
        if bet_id in existing:
            continue
        odds = int(e["odds"])
        stake = round(kelly_stake(float(e["edge"]), odds, bankroll, KELLY_FRACTION), 2)
        if stake < MIN_STAKE:
            continue
        rows.append({
            "bet_id": bet_id,
            "placed_date": placed_date,
            "game_date": "",
            "sport": str(e.get("sport", "")).upper(),
            "market": market,
            "game": game,
            "selection": selection,
            "odds": odds,
            "odds_decimal": round(american_to_decimal(odds), 3),
            "model_prob": float(e.get("model_prob", 0.0)),
            "edge": round(float(e.get("edge", 0.0)), 4),
            "stake": stake,
            "status": "open",
            "profit": None,
            "balance": None,
        })
        existing.add(bet_id)

    _recompute_balance(rows, start)
    return rows


# ── settlement ──────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum() or ch == " ").strip()


def _name_match(selection: str, candidate: str) -> bool:
    """True if a bet selection refers to the same team/player as a DB name."""
    a, b = _norm(selection), _norm(candidate)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    # nickname / surname overlap (last token), e.g. "... Knicks" or "... Giron"
    return a.split()[-1] == b.split()[-1]


def _nba_result(conn, selection, game, placed_date, today) -> tuple[str, bool] | None:
    rows = conn.execute(
        """SELECT game_date, team_name, wl FROM nba_team_games
           WHERE game_date >= ? AND game_date < ? ORDER BY game_date ASC""",
        (placed_date, today),
    ).fetchall()
    for game_date, team_name, wl in rows:
        if _name_match(selection, team_name or ""):
            return game_date, (wl == "W")
    return None


def _nhl_result(conn, selection, game, placed_date, today) -> tuple[str, bool] | None:
    team_id = next(
        (tid for tid, name in NHL_TEAM_NAMES.items() if _name_match(selection, name)),
        None,
    )
    if team_id is None:
        return None
    row = conn.execute(
        """SELECT game_date, wl FROM nhl_team_games
           WHERE team_id = ? AND game_date >= ? AND game_date < ?
           ORDER BY game_date ASC LIMIT 1""",
        (team_id, placed_date, today),
    ).fetchone()
    if row is None:
        return None
    return row[0], (row[1] == "W")


def _opponent_from_game(game: str, selection: str) -> str | None:
    """The other player in a tennis matchup stored as ``"A vs B"``.

    Returns the side that is NOT ``selection``, or None if the matchup can't be
    parsed into two players (so the caller can fall back to a name-only scan).
    """
    parts = re.split(r"\s+vs\.?\s+", game or "", flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    a, b = parts[0].strip(), parts[1].strip()
    if _name_match(selection, a):
        return b
    if _name_match(selection, b):
        return a
    return None


def _match_in_rows(rows, selection, opponent) -> tuple[str, bool] | None:
    """Find the bet's result among (match_date, winner, loser) rows.

    With a known ``opponent`` (parsed from the stored matchup), require BOTH
    players in one row so the result is pinned to the bet's actual match rather
    than the earliest match the player appears in. Without one, fall back to a
    single-name scan.
    """
    if opponent is not None:
        for match_date, winner, loser in rows:
            if _name_match(selection, winner or "") and _name_match(opponent, loser or ""):
                return match_date, True
            if _name_match(selection, loser or "") and _name_match(opponent, winner or ""):
                return match_date, False
        return None
    for match_date, winner, loser in rows:
        if _name_match(selection, winner or ""):
            return match_date, True
        if _name_match(selection, loser or ""):
            return match_date, False
    return None


def _wta_results_match(conn, selection, opponent, placed_date, today) -> tuple[str, bool] | None:
    """Settle against the WTA API feed (``tennis_wta_results``), if present.

    This feed carries the *real* per-match date and full names, so a tight floor
    suffices (a few days of slack absorbs timezone/placement timing). Returns None
    when the table is absent (fresh DB / ATP-only) so the caller falls back.
    """
    try:
        floor = (date.fromisoformat(placed_date) - timedelta(days=3)).isoformat()
    except ValueError:
        floor = placed_date
    try:
        rows = conn.execute(
            """SELECT match_date, winner_name, loser_name FROM tennis_wta_results
               WHERE match_date >= ? AND match_date < ? ORDER BY match_date ASC""",
            (floor, today),
        ).fetchall()
    except sqlite3.OperationalError:
        return None  # table not created in this DB
    return _match_in_rows(rows, selection, opponent)


def _tennis_result(conn, selection, game, placed_date, today) -> tuple[str, bool] | None:
    """Settle a tennis moneyline bet, preferring the precise WTA results feed.

    Tries ``tennis_wta_results`` first (real match dates + full names from the WTA
    API). Falls back to the Sackmann-derived ``tennis_matches_clean`` (covers ATP
    and historical WTA), whose ``match_date`` is the tournament START — shared by
    every match in the event — so the floor is widened by ``TENNIS_DATE_SLACK`` to
    absorb that offset. Both paths require BOTH the selection and its stored
    opponent in a single row, pinning the result to the bet's actual match (the old
    name-only scan mis-settled rematches and shared surnames). When the matchup
    can't be parsed into two players, a single-name scan is used.
    """
    opponent = _opponent_from_game(game, selection)

    hit = _wta_results_match(conn, selection, opponent, placed_date, today)
    if hit is not None:
        return hit

    try:
        floor = (date.fromisoformat(placed_date) - timedelta(days=TENNIS_DATE_SLACK)).isoformat()
    except ValueError:
        floor = placed_date
    rows = conn.execute(
        """SELECT match_date, winner_name, loser_name FROM tennis_matches_clean
           WHERE match_date >= ? AND match_date < ? ORDER BY match_date ASC""",
        (floor, today),
    ).fetchall()
    return _match_in_rows(rows, selection, opponent)


_RESULT_FN = {"NBA": _nba_result, "NHL": _nhl_result, "TENNIS": _tennis_result}


def settle_ledger(
    rows: list[dict],
    db_path: str,
    today: str | None = None,
    start: float = START_BANKROLL,
) -> list[dict]:
    """Settle open bets whose games have finished, then recompute the curve.

    For each open moneyline bet, look up the selection's first game on/after the
    placed date that finished before ``today``. Win -> profit = stake*(dec-1),
    loss -> -stake. Bets with no match yet stay open; bets older than
    ``STALE_DAYS`` with no match are voided (stake refunded, no P&L) rather than
    guessed.
    """
    today = today or date.today().isoformat()
    open_rows = [r for r in rows if r.get("status") == "open"]
    if open_rows:
        conn = sqlite3.connect(db_path)
        try:
            for r in open_rows:
                fn = _RESULT_FN.get(str(r.get("sport", "")).upper())
                res = (
                    fn(conn, r["selection"], r.get("game", ""), r["placed_date"], today)
                    if fn else None
                )
                if res is not None:
                    game_date, won = res
                    r["game_date"] = game_date
                    r["status"] = "won" if won else "lost"
                    dec = r.get("odds_decimal") or american_to_decimal(int(r["odds"]))
                    stake = r.get("stake") or 0.0
                    r["profit"] = round(stake * (dec - 1) if won else -stake, 2)
                elif _days_old(r["placed_date"], today) > STALE_DAYS:
                    r["status"] = "void"      # game presumably played but unmatchable
                    r["profit"] = 0.0
        finally:
            conn.close()

    _recompute_balance(rows, start)
    return rows


def _days_old(placed_date: str, today: str) -> int:
    try:
        d0 = datetime.fromisoformat(placed_date).date()
        d1 = datetime.fromisoformat(today).date()
        return (d1 - d0).days
    except (TypeError, ValueError):
        return 0


# ── summary ─────────────────────────────────────────────────────────────────────

def equity_summary(rows: list[dict], start: float = START_BANKROLL) -> dict:
    """Headline stats for the current equity curve."""
    settled = sorted((r for r in rows if r.get("status") in _SETTLED), key=_settle_key)
    n_open = sum(1 for r in rows if r.get("status") == "open")
    n_void = sum(1 for r in rows if r.get("status") == "void")
    current = current_bankroll(rows, start)

    wins = sum(1 for r in settled if r.get("status") == "won")
    staked = sum(r.get("stake") or 0.0 for r in settled)
    net = sum(r.get("profit") or 0.0 for r in settled)

    # max drawdown over the running balance (peak-to-trough)
    peak = start
    max_dd = 0.0
    running = start
    for r in settled:
        running = r.get("balance") if r.get("balance") is not None else running
        peak = max(peak, running)
        if peak > 0:
            max_dd = max(max_dd, (peak - running) / peak)

    return {
        "start": round(start, 2),
        "current": round(current, 2),
        "total_return_pct": round((current / start - 1) * 100, 2) if start else 0.0,
        "peak": round(peak, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_settled": len(settled),
        "n_open": n_open,
        "n_void": n_void,
        "hit_rate": round(wins / len(settled), 3) if settled else 0.0,
        "total_staked": round(staked, 2),
        "net_profit": round(net, 2),
    }


def print_summary(rows: list[dict], start: float = START_BANKROLL) -> None:
    """Pretty-print the bankroll summary + the last few settled bets."""
    s = equity_summary(rows, start)
    print(f"\n{'='*70}")
    print(f"  BANKROLL SIMULATOR  (start ${s['start']:.2f})")
    print(f"{'='*70}")
    print(f"  Balance:    ${s['current']:.2f}   ({s['total_return_pct']:+.2f}%)")
    print(f"  Settled:    {s['n_settled']}  |  hit rate {s['hit_rate']*100:.1f}%  "
          f"|  open {s['n_open']}  |  void {s['n_void']}")
    print(f"  Staked:     ${s['total_staked']:.2f}   net ${s['net_profit']:+.2f}")
    print(f"  Peak ${s['peak']:.2f}  |  max drawdown {s['max_drawdown_pct']:.1f}%")

    settled = sorted((r for r in rows if r.get("status") in _SETTLED), key=_settle_key)
    if settled:
        print(f"\n  Recent settled bets:")
        for r in settled[-5:]:
            mark = "WON " if r["status"] == "won" else "LOST"
            print(f"    {r['game_date']}  {mark}  {r['selection']:<24} "
                  f"${r['stake']:.2f} -> ${r['profit']:+.2f}   bal ${r['balance']:.2f}")

    open_bets = sorted(
        (r for r in rows if r.get("status") == "open"),
        key=lambda r: (r.get("placed_date") or "", r.get("bet_id") or ""),
    )
    if open_bets:
        pending = sum(r.get("stake") or 0.0 for r in open_bets)
        print(f"\n  Open bets (pending settlement) — ${pending:.2f} at stake:")
        for r in open_bets:
            dec = r.get("odds_decimal") or american_to_decimal(int(r["odds"]))
            payout = (r.get("stake") or 0.0) * (dec - 1)
            print(f"    placed {r['placed_date']}  {r['sport']:<6} {r['selection']:<24} "
                  f"@{int(r['odds']):+d}  ${r['stake']:.2f} -> +${payout:.2f} if win")
    print(f"{'='*70}\n")


def print_history(rows: list[dict], start: float = START_BANKROLL) -> None:
    """Print every bet the simulator has ever made, oldest first."""
    if not rows:
        print("\n  No bets logged yet. Run the pipeline to start the paper sim.\n")
        return

    def _key(r: dict):
        return (
            r.get("game_date") or r.get("placed_date") or "",
            r.get("placed_date") or "",
            r.get("bet_id") or "",
        )

    ordered = sorted(rows, key=_key)
    print(f"\n{'='*78}")
    print(f"  PAPER SIM — FULL BET HISTORY  ({len(ordered)} bets)")
    print(f"{'='*78}")
    print(f"  {'placed':<10} {'game':<10} {'sport':<6} {'selection':<24} "
          f"{'odds':>6} {'stake':>8} {'result':>9} {'bal':>9}")
    print(f"  {'-'*76}")
    for r in ordered:
        status = (r.get("status") or "open").lower()
        if status == "won":
            result = f"+${(r.get('profit') or 0.0):.2f}"
        elif status == "lost":
            result = f"-${abs(r.get('profit') or 0.0):.2f}"
        elif status == "void":
            result = "void"
        else:
            result = "open"
        bal = r.get("balance")
        bal_s = f"${bal:.2f}" if isinstance(bal, (int, float)) else "-"
        odds = r.get("odds")
        odds_s = f"{int(odds):+d}" if odds not in (None, "") else "-"
        print(f"  {(r.get('placed_date') or '-'):<10} {(r.get('game_date') or '-'):<10} "
              f"{(r.get('sport') or '-'):<6} {(r.get('selection') or '-'):<24.24} "
              f"{odds_s:>6} ${ (r.get('stake') or 0.0):>7.2f} {result:>9} {bal_s:>9}")
    print(f"{'='*78}\n")
