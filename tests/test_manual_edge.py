"""Tests for the manual matchup edge calculator (edge/manual.py).

Unit tests exercise the pure logic (pricing math, name resolution, matchup-row
assembly) with no data dependency. Integration tests run the full functions against
the local DB + model artifacts and are skipped automatically when those are absent.
"""

import math
import os

import numpy as np
import pytest

from edge.calculator import (
    ARTIFACTS_DIR, _spread_edges, american_to_prob, remove_vig,
    tennis_feature_vector,
)
from edge import manual
from features.pipeline import TENNIS_FEATURE_COLS


# ── locating optional data for integration tests ────────────────────────────────

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


def _has(name):
    return (ARTIFACTS_DIR / f"{name}.pkl").exists()


needs_tennis = pytest.mark.skipif(
    DB is None or not _has("tennis_atp_moneyline"),
    reason="requires local DB + tennis artifacts",
)
needs_nba = pytest.mark.skipif(
    DB is None or not _has("nba_moneyline"),
    reason="requires local DB + NBA artifacts",
)
needs_nhl = pytest.mark.skipif(
    DB is None or not _has("nhl_moneyline"),
    reason="requires local DB + NHL artifacts",
)


# ── unit: pricing math ──────────────────────────────────────────────────────────

def test_spread_edges_probs_are_complementary():
    sides = _spread_edges(mean=5.0, sigma=10.0, line=-3.5,
                          a_odds=-110, b_odds=-110, min_edge=-1.0,
                          a_label="A", b_label="B")
    assert len(sides) == 2
    (_, _, p_a, _), (_, _, p_b, _) = sides
    assert 0.0 < p_a < 1.0 and 0.0 < p_b < 1.0
    assert p_a + p_b == pytest.approx(1.0)


def test_spread_edges_favourite_more_likely_to_cover_small_line():
    # A predicted to win by 5; a -1.5 line should be covered > 50% of the time
    (a, _, p_a, _), _ = _spread_edges(5.0, 10.0, -1.5, -110, -110, -1.0)
    assert p_a > 0.5


def test_spread_edges_min_edge_filters():
    # a tiny mean with symmetric prices => ~0 edge => filtered out at a high threshold
    assert _spread_edges(0.0, 10.0, 0.0, -110, -110, min_edge=0.20) == []


def test_ml_dicts_invariants():
    dicts = manual._ml_dicts("nba", "B @ A", "A", -150, "B", 130, prob_a=0.62, bankroll=1000)
    assert len(dicts) == 2
    a, b = dicts
    assert a["model_prob"] + b["model_prob"] == pytest.approx(1.0, abs=1e-6)
    assert a["fair_prob"] + b["fair_prob"] == pytest.approx(1.0, abs=1e-6)
    # edge = model - fair, so the two edges sum to zero (vig-free book)
    assert a["edge"] + b["edge"] == pytest.approx(0.0, abs=1e-6)
    # only the +EV side gets a non-zero Kelly stake
    assert (a["kelly_stake"] > 0) != (b["kelly_stake"] > 0) or a["edge"] == b["edge"]


def test_ml_dict_edge_matches_model_minus_fair():
    fair_a, _ = remove_vig(american_to_prob(-150), american_to_prob(130))
    dicts = manual._ml_dicts("tennis", "g", "A", -150, "B", 130, prob_a=0.62, bankroll=1000)
    assert dicts[0]["edge"] == pytest.approx(round(0.62 - fair_a, 3), abs=1e-3)


# ── unit: market parsing & validation ───────────────────────────────────────────

@pytest.mark.parametrize("alias,expected", [
    ("moneyline", "moneyline"), ("ML", "moneyline"), ("match winner", "moneyline"),
    ("totals", "totals"), ("o/u", "totals"),
    ("spread", "spread"), ("handicap", "spread"), ("props", "prop"),
])
def test_market_aliases(alias, expected):
    assert manual._market(alias) == expected


def test_market_unknown_raises():
    with pytest.raises(ValueError):
        manual._market("parlay")


def test_require_raises_on_missing():
    with pytest.raises(ValueError) as e:
        manual._require(odds_a=-110, odds_b=None)
    assert "odds_b" in str(e.value)
    manual._require(odds_a=-110, odds_b=120)  # all present -> no raise


# ── unit: name resolution ───────────────────────────────────────────────────────

def test_resolve_exact_and_casefold_and_substring():
    snap = {"Carlos Alcaraz": {}, "Novak Djokovic": {}}
    assert manual._resolve("Carlos Alcaraz", snap, "player") == "Carlos Alcaraz"
    assert manual._resolve("novak djokovic", snap, "player") == "Novak Djokovic"
    assert manual._resolve("Djokovic", snap, "player") == "Novak Djokovic"


def test_resolve_ambiguous_and_missing_and_empty():
    snap = {"John Isner": {}, "John Millman": {}}
    with pytest.raises(ValueError):
        manual._resolve("John", snap, "player")       # ambiguous
    with pytest.raises(ValueError):
        manual._resolve("Federer", snap, "player")    # not found
    with pytest.raises(ValueError):
        manual._resolve("anyone", {}, "player")       # no data


# ── unit: tennis feature vector surface override ─────────────────────────────────

def test_tennis_feature_vector_surface_override():
    cols = TENNIS_FEATURE_COLS
    vec = tennis_feature_vector({}, {}, cols, surface="grass")
    idx = {c: i for i, c in enumerate(cols)}
    assert vec[idx["surface_grass"]] == 1.0
    for other in ("surface_hard", "surface_clay", "surface_carpet"):
        assert vec[idx[other]] == 0.0
    # without an override, surface one-hots stay NaN (proxied from p1's last match)
    vec2 = tennis_feature_vector({}, {}, cols)
    assert math.isnan(vec2[idx["surface_grass"]])


# ── unit: team matchup row assembly ─────────────────────────────────────────────

def test_build_team_matchup_row_mapping():
    from models.train import TEAM_FEATURE_COLS

    # every base stat the builder reads -> distinct value per side so we can assert direction
    bases = set()
    for c in TEAM_FEATURE_COLS:
        bases.add(c[4:] if c.startswith("opp_") else c)
    home = {b: 10.0 for b in bases}
    away = {b: 20.0 for b in bases}

    row = manual._build_team_matchup_row(home, away, rest_days=3)

    assert row["is_home"] == 1.0
    assert row["rest_days"] == 3.0
    # own column comes from home, opp_ column from away
    assert row["pts_roll10"] == 10.0
    assert row["opp_pts_roll10"] == 20.0
    # derived matchup columns recomputed from both sides
    assert row["pace_matchup"] == pytest.approx((10.0 + 20.0) / 2)
    assert row["off_def_edge"] == pytest.approx(10.0 - 20.0)
    # unreconstructable context neutralised
    assert row["wowy_margin_delta"] == 0.0
    assert math.isnan(row["mkt_implied_prob"])


def test_nhl_team_names_cover_known_ids():
    assert manual.NHL_TEAM_NAMES[12] == "Carolina Hurricanes"
    assert manual.NHL_TEAM_NAMES[54] == "Vegas Golden Knights"
    assert len(manual.NHL_TEAM_NAMES) >= 32


# ── integration: full pipeline against local data ───────────────────────────────

@needs_tennis
def test_tennis_moneyline_integration():
    players = manual.list_tennis_players(DB)
    a, b = players[0], players[1]
    edges = manual.tennis_matchup_edge(DB, a, b, "moneyline", odds_a=-150, odds_b=130)
    assert len(edges) == 2
    for e in edges:
        assert 0.0 < e["model_prob"] < 1.0
        assert e["edge"] == pytest.approx(e["model_prob"] - e["fair_prob"], abs=1e-3)
    assert edges[0]["model_prob"] + edges[1]["model_prob"] == pytest.approx(1.0, abs=1e-3)


@needs_tennis
def test_tennis_totals_and_spread_integration():
    players = manual.list_tennis_players(DB)
    a, b = players[0], players[1]
    # distributional markets: probabilities are valid in the closed [0, 1]
    # (unlike the calibrated/clamped moneyline), since tail lines can saturate.
    tot = manual.tennis_matchup_edge(DB, a, b, "totals", line=22.5, over_odds=-110, under_odds=-110)
    assert len(tot) == 2 and all(0.0 <= e["model_prob"] <= 1.0 for e in tot)
    spr = manual.tennis_matchup_edge(DB, a, b, "spread", line=-3.5, odds_a=-110, odds_b=-110)
    assert len(spr) == 2 and all(0.0 <= e["model_prob"] <= 1.0 for e in spr)


@needs_nba
def test_nba_matchup_all_markets_integration():
    teams = manual.list_teams(DB, "nba")
    h, a = teams[0], teams[1]
    for kwargs in (
        dict(market="moneyline", odds_a=-120, odds_b=100),
        dict(market="spread", line=-5.5, odds_a=-110, odds_b=-110),
        dict(market="totals", line=220.5, over_odds=-110, under_odds=-110),
    ):
        edges = manual.team_matchup_edge(DB, "nba", h, a, **kwargs)
        assert len(edges) == 2
        assert all(0.0 <= e["model_prob"] <= 1.0 for e in edges)


@needs_nba
def test_nba_prop_integration():
    players = manual.list_players(DB, "nba")
    edges = manual.player_prop_edge(DB, "nba", players[0], "pts", 20.5, -110, -110)
    assert len(edges) == 2
    assert all(0.0 <= e["model_prob"] <= 1.0 for e in edges)


@needs_nhl
def test_nhl_moneyline_integration():
    teams = manual.list_teams(DB, "nhl")
    edges = manual.team_matchup_edge(DB, "nhl", teams[0], teams[1], "moneyline", odds_a=-130, odds_b=110)
    assert len(edges) == 2


def test_nhl_prop_is_blocked():
    with pytest.raises(ValueError):
        manual.player_prop_edge(DB or ":memory:", "nhl", "x", "goals", 1.5, -110, -110)
