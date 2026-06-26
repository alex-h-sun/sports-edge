"""WTA match-results ingestion via the public api.wtatennis.com JSON API.

Why this exists separately from ``ingestion/tennis.py`` (Sackmann): Sackmann's
yearly CSV is the canonical source but lags publication (its current-year file may
not exist yet), which leaves the paper-trading simulator unable to settle recent
WTA bets. The WTA's own API exposes finished singles results with a *real
per-match date* and full names, which settles cleanly. ATP has no equivalent open
endpoint (atptour.com is bot-walled, returns 403), so ATP still relies on Sackmann.

Results land in their own table ``tennis_wta_results`` (not ``tennis_matches_clean``,
which ``ingestion/tennis_clean.py`` rebuilds from Sackmann on every run and would
otherwise wipe these rows). The paper simulator's settlement consults this table
first, then falls back to the Sackmann-derived table.
"""

import sqlite3
import time
from datetime import date, timedelta

import requests

_API = "https://api.wtatennis.com/tennis"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_RATE_LIMIT = 0.2  # seconds between HTTP requests


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tennis_wta_results (
            match_uid    TEXT PRIMARY KEY,
            match_date   TEXT,
            tour         TEXT,
            event_id     TEXT,
            event_name   TEXT,
            round_id     TEXT,
            winner_name  TEXT,
            loser_name   TEXT,
            score        TEXT,
            last_updated TEXT
        )
    """)


def _full_name(first, last) -> str:
    return f"{str(first or '').strip()} {str(last or '').strip()}".strip()


def _winner_loser(m: dict) -> tuple[str, str] | None:
    """Return (winner_name, loser_name) for a finished match, or None if undetermined.

    The API's ``Winner`` field is 2 -> player A, 3 -> player B (verified against
    ResultString and set scores). Falls back to counting sets won if Winner is
    missing/unexpected (e.g. odd retirement encodings).
    """
    a_name = _full_name(m.get("PlayerNameFirstA"), m.get("PlayerNameLastA"))
    b_name = _full_name(m.get("PlayerNameFirstB"), m.get("PlayerNameLastB"))
    if not a_name or not b_name:
        return None

    winner = str(m.get("Winner") or "")
    if winner == "2":
        return a_name, b_name
    if winner == "3":
        return b_name, a_name

    sets_a = sets_b = 0
    for i in range(1, 6):
        sa, sb = m.get(f"ScoreSet{i}A"), m.get(f"ScoreSet{i}B")
        try:
            sa, sb = int(sa), int(sb)
        except (TypeError, ValueError):
            continue
        if sa > sb:
            sets_a += 1
        elif sb > sa:
            sets_b += 1
    if sets_a > sets_b:
        return a_name, b_name
    if sets_b > sets_a:
        return b_name, a_name
    return None


def _match_date(m: dict) -> str | None:
    ts = str(m.get("MatchTimeStamp") or m.get("LastUpdated") or "")
    return ts[:10] if len(ts) >= 10 else None


def _fetch_json(url: str):
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _tournaments(cutoff_iso: str | None, years: set[int] | None) -> list[tuple[str, int, str]]:
    """List (group_id, year, name) for finished/in-progress tournaments to scan.

    Restricted to main-tour events (non-empty ``liveScoringId``): only those carry
    sportsbook odds and thus generate settleable bets. This also excludes the long
    tail of ITF $-level events, which otherwise dominate the request count.
    """
    data = _fetch_json(f"{_API}/tournaments/?page=0&pageSize=400&sort=desc")
    out: list[tuple[str, int, str]] = []
    for c in data.get("content", []):
        gid = (c.get("tournamentGroup") or {}).get("id")
        if gid is None or not c.get("liveScoringId"):
            continue
        if years is not None and c.get("year") not in years:
            continue
        if cutoff_iso and (c.get("startDate") or "") < cutoff_iso:
            continue
        if c.get("status") not in ("past", "inProgress"):
            continue
        name = (c.get("tournamentGroup") or {}).get("name") or ""
        out.append((str(gid), c.get("year"), name))
    return out


def fetch_wta_results(db_path: str, years: list[int] | None = None, since_days: int = 45) -> int:
    """Upsert finished WTA singles results into ``tennis_wta_results``.

    Incremental by default: scans tournaments started within the last
    ``since_days``. Pass ``years`` (e.g. ``[2026]``) for a fuller backfill, which
    ignores the day cutoff. Returns the number of matches upserted. Network and
    per-tournament failures are logged and skipped, never raised.
    """
    cutoff = None
    if years is None and since_days:
        cutoff = (date.today() - timedelta(days=since_days)).isoformat()
    try:
        tournaments = _tournaments(cutoff, set(years) if years else None)
    except Exception as e:  # noqa: BLE001 - network best-effort
        print(f"  warning: WTA tournaments fetch failed ({e})")
        return 0

    conn = _get_conn(db_path)
    _ensure_table(conn)
    total = 0
    try:
        for gid, year, name in tournaments:
            try:
                data = _fetch_json(f"{_API}/tournaments/{gid}/{year}/matches")
            except Exception as e:  # noqa: BLE001
                print(f"  warning: WTA {name} {year} matches failed ({e})")
                continue
            rows = []
            for m in data.get("matches", []):
                if m.get("DrawMatchType") != "S" or m.get("MatchState") != "F":
                    continue
                wl = _winner_loser(m)
                md = _match_date(m)
                if not wl or not md:
                    continue
                rows.append((
                    f"wta:{m.get('EventID')}:{m.get('EventYear')}:{m.get('MatchID')}",
                    md, "wta", str(m.get("EventID") or ""), name,
                    str(m.get("RoundID") or ""), wl[0], wl[1],
                    str(m.get("ScoreString") or ""), str(m.get("LastUpdated") or ""),
                ))
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO tennis_wta_results VALUES (?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                total += len(rows)
            time.sleep(_RATE_LIMIT)
        conn.commit()
    finally:
        conn.close()
    print(f"  WTA results: upserted {total} finished singles matches across {len(tournaments)} tournaments")
    return total
