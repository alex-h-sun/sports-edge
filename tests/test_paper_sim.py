"""Tests for the paper-trading bankroll simulator (models/paper_sim.py).

Unit tests exercise the pure ledger logic — appending edges, compounding
quarter-Kelly off the running balance, settlement profit math, dedupe, the
ruin/min-stake floor, balance recomputation, and the equity summary — with no
data dependency. An integration test runs settle_ledger against the local DB and
is skipped when it is absent.
"""

import os
import sqlite3

import pytest

from edge.calculator import american_to_decimal, kelly_stake
from models import paper_sim as ps


# ── helpers ──────────────────────────────────────────────────────────────────────

def _edge(sport, team, odds, edge, market="Moneyline", game=None):
    return {
        "sport": sport, "market": market,
        "game": game or f"{team} @ OPP",
        "bet": f"{team} ML", "odds": odds,
        "model_prob": 0.6, "edge": edge,
        "kelly_stake": 999.0,  # should be ignored / re-sized off running bankroll
    }


def _find_db():
    cands = [os.getenv("DB_PATH")] if os.getenv("DB_PATH") else []
    here = os.path.dirname(__file__)
    cands += [
        "data/sports.db",
        os.path.join(here, "..", "data", "sports.db"),
        os.path.join(here, "..", "..", "..", "data", "sports.db"),  # from a worktree
    ]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


DB = _find_db()


# ── appending edges ──────────────────────────────────────────────────────────────

def test_append_sizes_off_running_bankroll_not_record_stake():
    rows = ps.append_edges([], [_edge("NBA", "Knicks", -110, 0.08)],
                           bankroll=1000.0, placed_date="2026-06-01")
    assert len(rows) == 1
    expected = round(kelly_stake(0.08, -110, 1000.0, ps.KELLY_FRACTION), 2)
    assert rows[0]["stake"] == expected
    assert rows[0]["stake"] != 999.0          # NOT the edge record's stake
    assert rows[0]["status"] == "open"
    assert rows[0]["odds_decimal"] == round(american_to_decimal(-110), 3)


def test_append_filters_below_min_edge_and_non_moneyline():
    edges = [
        _edge("NBA", "Knicks", -110, 0.02),                 # below default 0.05
        _edge("NBA", "Heat", -110, 0.09, market="Totals"),  # wrong market
        _edge("NBA", "Bucks", -110, 0.09),                  # kept
    ]
    rows = ps.append_edges([], edges, bankroll=1000.0, placed_date="2026-06-01")
    assert [r["selection"] for r in rows] == ["Bucks"]


def test_append_is_idempotent_same_day():
    e = [_edge("NBA", "Knicks", -110, 0.08)]
    rows = ps.append_edges([], e, bankroll=1000.0, placed_date="2026-06-01")
    rows = ps.append_edges(rows, e, bankroll=1000.0, placed_date="2026-06-01")
    assert len(rows) == 1


def test_append_drops_dust_when_bankroll_exhausted():
    assert ps.append_edges([], [_edge("NBA", "K", -110, 0.08)], bankroll=0.5) == []


# ── settlement + compounding ─────────────────────────────────────────────────────

def test_settlement_profit_and_running_balance():
    # two bets settled on consecutive days; balance must compound.
    rows = [
        {"bet_id": "a", "placed_date": "2026-06-01", "game_date": "2026-06-01",
         "sport": "NBA", "market": "Moneyline", "game": "g1", "selection": "A",
         "odds": 100, "odds_decimal": 2.0, "model_prob": 0.6, "edge": 0.08,
         "stake": 100.0, "status": "won", "profit": 100.0, "balance": None},
        {"bet_id": "b", "placed_date": "2026-06-02", "game_date": "2026-06-02",
         "sport": "NBA", "market": "Moneyline", "game": "g2", "selection": "B",
         "odds": -110, "odds_decimal": american_to_decimal(-110), "model_prob": 0.6,
         "edge": 0.08, "stake": 50.0, "status": "lost", "profit": -50.0, "balance": None},
    ]
    ps._recompute_balance(rows, start=1000.0)
    # won at +100 (decimal 2.0): profit = 100*(2-1)=100 -> 1100
    assert rows[0]["balance"] == 1100.0
    # lost 50 -> 1050
    assert rows[1]["balance"] == 1050.0
    assert ps.current_bankroll(rows, 1000.0) == 1050.0


def test_void_after_stale_window_refunds_stake(tmp_path):
    # an unmatchable bet older than STALE_DAYS is voided, not guessed.
    rows = [{
        "bet_id": "x", "placed_date": "2026-06-01", "game_date": "",
        "sport": "NBA", "market": "Moneyline", "game": "g", "selection": "Nonexistent Team",
        "odds": -110, "odds_decimal": american_to_decimal(-110), "model_prob": 0.6,
        "edge": 0.08, "stake": 100.0, "status": "open", "profit": None, "balance": None,
    }]
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE nba_team_games (game_date TEXT, team_name TEXT, wl TEXT)")
    conn.commit()
    conn.close()
    rows = ps.settle_ledger(rows, str(db), today="2026-06-30", start=1000.0)
    assert rows[0]["status"] == "void"
    assert ps.current_bankroll(rows, 1000.0) == 1000.0  # refunded, no P&L


def test_open_bet_stays_open_within_window(tmp_path):
    rows = [{
        "bet_id": "x", "placed_date": "2026-06-01", "game_date": "",
        "sport": "NBA", "market": "Moneyline", "game": "g", "selection": "Nonexistent Team",
        "odds": -110, "odds_decimal": american_to_decimal(-110), "model_prob": 0.6,
        "edge": 0.08, "stake": 100.0, "status": "open", "profit": None, "balance": None,
    }]
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE nba_team_games (game_date TEXT, team_name TEXT, wl TEXT)")
    conn.commit()
    conn.close()
    rows = ps.settle_ledger(rows, str(db), today="2026-06-03", start=1000.0)
    assert rows[0]["status"] == "open"


def _tennis_bet(selection, game, placed_date="2026-06-16", odds=-150):
    return {
        "bet_id": selection[:6], "placed_date": placed_date, "game_date": "",
        "sport": "TENNIS", "market": "Moneyline", "game": game, "selection": selection,
        "odds": odds, "odds_decimal": american_to_decimal(odds), "model_prob": 0.6,
        "edge": 0.08, "stake": 100.0, "status": "open", "profit": None, "balance": None,
    }


def _tennis_db(tmp_path, matches):
    # matches: list of (match_date, winner_name, loser_name)
    db = tmp_path / "tennis.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE tennis_matches_clean (match_date TEXT, winner_name TEXT, loser_name TEXT)"
    )
    conn.executemany("INSERT INTO tennis_matches_clean VALUES (?,?,?)", matches)
    conn.commit()
    conn.close()
    return str(db)


def test_tennis_settles_mid_tournament_bet_despite_start_date(tmp_path):
    # match_date is the tournament START (06-15), placed_date is mid-tournament (06-16).
    # The widened floor must still find the match instead of voiding it.
    db = _tennis_db(tmp_path, [("2026-06-15", "Jessica Pegula", "Karolina Muchova")])
    rows = [_tennis_bet("Jessica Pegula", "Jessica Pegula vs Karolina Muchova")]
    rows = ps.settle_ledger(rows, db, today="2026-06-25", start=1000.0)
    assert rows[0]["status"] == "won"
    assert rows[0]["game_date"] == "2026-06-15"
    assert rows[0]["profit"] > 0


def test_tennis_loss_settles_with_negative_profit(tmp_path):
    db = _tennis_db(tmp_path, [("2026-06-15", "Karolina Muchova", "Jessica Pegula")])
    rows = [_tennis_bet("Jessica Pegula", "Jessica Pegula vs Karolina Muchova")]
    rows = ps.settle_ledger(rows, db, today="2026-06-25", start=1000.0)
    assert rows[0]["status"] == "lost"
    assert rows[0]["profit"] == -100.0


def test_tennis_matchup_disambiguates_earlier_match(tmp_path):
    # The selection plays an EARLIER match (different opponent) inside the window.
    # Name-only settlement would have settled against that earlier match; the
    # matchup-aware logic must pin to the row containing BOTH stored players.
    db = _tennis_db(tmp_path, [
        ("2026-06-10", "Other Player", "Jessica Pegula"),       # earlier loss vs someone else
        ("2026-06-15", "Jessica Pegula", "Karolina Muchova"),   # the bet's actual match (win)
    ])
    rows = [_tennis_bet("Jessica Pegula", "Jessica Pegula vs Karolina Muchova")]
    rows = ps.settle_ledger(rows, db, today="2026-06-25", start=1000.0)
    assert rows[0]["status"] == "won"
    assert rows[0]["game_date"] == "2026-06-15"


def test_tennis_voids_when_match_absent(tmp_path):
    # No result row at all -> stays unsettled, voids past the stale window.
    db = _tennis_db(tmp_path, [])
    rows = [_tennis_bet("Jessica Pegula", "Jessica Pegula vs Karolina Muchova")]
    rows = ps.settle_ledger(rows, db, today="2026-06-25", start=1000.0)
    assert rows[0]["status"] == "void"


def _wta_results_db(tmp_path, results):
    # results: list of (match_date, winner_name, loser_name)
    db = tmp_path / "wta.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE tennis_wta_results (match_date TEXT, winner_name TEXT, loser_name TEXT)"
    )
    # also create the Sackmann table so the fallback path has something to scan
    conn.execute(
        "CREATE TABLE tennis_matches_clean (match_date TEXT, winner_name TEXT, loser_name TEXT)"
    )
    conn.executemany("INSERT INTO tennis_wta_results VALUES (?,?,?)", results)
    conn.commit()
    conn.close()
    return str(db)


def test_settles_against_wta_results_feed_with_real_date(tmp_path):
    # The WTA feed carries the real match date (06-17), later than the tournament
    # start. Settlement should use it and resolve precisely.
    db = _wta_results_db(tmp_path, [("2026-06-17", "Eva Lys", "Ann Li")])
    rows = [_tennis_bet("Eva Lys", "Eva Lys vs Ann Li", placed_date="2026-06-16")]
    rows = ps.settle_ledger(rows, db, today="2026-06-25", start=1000.0)
    assert rows[0]["status"] == "won"
    assert rows[0]["game_date"] == "2026-06-17"


def test_wta_feed_takes_precedence_over_sackmann(tmp_path):
    # Both tables have the match; the feed (real date) should win.
    db = _wta_results_db(tmp_path, [("2026-06-17", "Ann Li", "Eva Lys")])  # Lys lost per feed
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO tennis_matches_clean VALUES (?,?,?)",
        ("2026-06-15", "Eva Lys", "Ann Li"),  # Sackmann says Lys won, tournament-start date
    )
    conn.commit()
    conn.close()
    rows = [_tennis_bet("Eva Lys", "Eva Lys vs Ann Li", placed_date="2026-06-16")]
    rows = ps.settle_ledger(rows, db, today="2026-06-25", start=1000.0)
    assert rows[0]["status"] == "lost"          # feed wins
    assert rows[0]["game_date"] == "2026-06-17"


def test_wta_winner_loser_decoding():
    from ingestion import tennis_results as tr
    base = {
        "PlayerNameFirstA": "Jessica", "PlayerNameLastA": "Pegula",
        "PlayerNameFirstB": "Karolina", "PlayerNameLastB": "Muchova",
    }
    assert tr._winner_loser({**base, "Winner": "2"}) == ("Jessica Pegula", "Karolina Muchova")
    assert tr._winner_loser({**base, "Winner": "3"}) == ("Karolina Muchova", "Jessica Pegula")
    # No Winner field -> derive from sets (B wins 2-0)
    derived = tr._winner_loser({
        **base, "Winner": "", "ScoreSet1A": "3", "ScoreSet1B": "6",
        "ScoreSet2A": "4", "ScoreSet2B": "6",
    })
    assert derived == ("Karolina Muchova", "Jessica Pegula")


def test_name_match_handles_substring_and_surname():
    assert ps._name_match("Knicks", "New York Knicks")
    assert ps._name_match("New York Knicks", "Knicks")
    assert ps._name_match("Roger Federer", "R Federer")   # surname-token overlap
    assert not ps._name_match("Lakers", "Boston Celtics")


# ── summary ──────────────────────────────────────────────────────────────────────

def test_equity_summary_drawdown_and_hit_rate():
    rows = [
        {"status": "won", "stake": 100.0, "profit": 100.0, "balance": 1100.0,
         "game_date": "2026-06-01", "placed_date": "2026-06-01", "bet_id": "a"},
        {"status": "lost", "stake": 200.0, "profit": -200.0, "balance": 900.0,
         "game_date": "2026-06-02", "placed_date": "2026-06-02", "bet_id": "b"},
        {"status": "open", "stake": 50.0, "profit": None, "balance": None,
         "game_date": "", "placed_date": "2026-06-03", "bet_id": "c"},
    ]
    s = ps.equity_summary(rows, start=1000.0)
    assert s["current"] == 900.0
    assert s["total_return_pct"] == -10.0
    assert s["n_settled"] == 2 and s["n_open"] == 1
    assert s["hit_rate"] == 0.5
    assert s["peak"] == 1100.0
    # peak 1100 -> trough 900 = 18.18% drawdown
    assert s["max_drawdown_pct"] == pytest.approx(18.18, abs=0.01)


def test_round_trip_csv(tmp_path):
    rows = ps.append_edges([], [_edge("NBA", "Knicks", -110, 0.08)],
                           bankroll=1000.0, placed_date="2026-06-01")
    path = tmp_path / "ledger.csv"
    ps.save_ledger(rows, str(path))
    back = ps.load_ledger(str(path))
    assert back[0]["selection"] == "Knicks"
    assert back[0]["odds"] == -110
    assert back[0]["stake"] == rows[0]["stake"]
    assert back[0]["status"] == "open"


# ── integration (needs the local DB) ─────────────────────────────────────────────

@pytest.mark.skipif(DB is None, reason="local DB not available")
def test_settle_against_real_db_does_not_crash():
    rows = [{
        "bet_id": "z", "placed_date": "2024-01-01", "game_date": "",
        "sport": "NBA", "market": "Moneyline", "game": "g", "selection": "Boston Celtics",
        "odds": -150, "odds_decimal": american_to_decimal(-150), "model_prob": 0.6,
        "edge": 0.08, "stake": 100.0, "status": "open", "profit": None, "balance": None,
    }]
    rows = ps.settle_ledger(rows, DB, today="2026-06-16", start=1000.0)
    assert rows[0]["status"] in {"won", "lost", "void", "open"}
