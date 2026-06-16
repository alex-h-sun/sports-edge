"""Manual matchup edge calculator.

On-demand counterpart to the auto edge scan: pick two players (tennis) or two teams
(NBA/NHL), pass in the book's current odds for a market, and get the model's edge +
fractional-Kelly stake for that exact bet. Because the price is supplied by hand, this
path sidesteps the live odds feed entirely (no API quota, no odds-name matching) — names
only need to resolve to a feature snapshot.

All odds/prediction math is reused from edge.calculator; this module only assembles the
right feature row per matchup and formats the result as the same edge-dict shape that
edge.alerts.print_edges renders.

Caveat: manual mode evaluates a *neutral, current-form* matchup. No injuries are assumed,
rest days are defaulted, the market-line feature is null, and (tennis) the surface is
proxied from player1's last match unless overridden. It answers "is this price +EV given
each side's latest form", not a fully-contextualised scheduled-game prediction.
"""

from __future__ import annotations

import numpy as np

from edge.calculator import (
    _load, _predict_proba, _predict_mean_sigma, _over_under_edges, _spread_edges,
    american_to_prob, remove_vig, calc_edge, kelly_stake,
    tennis_player_snapshots, tennis_feature_vector,
)

# NHL stores only numeric team_id; this maps every id present in the history to the
# full name the Odds API uses (fetched from the NHL franchise endpoint). Covers
# relocations/renames (Arizona -> Utah) that appear in older rows.
NHL_TEAM_NAMES: dict[int, str] = {
    1: "New Jersey Devils", 2: "New York Islanders", 3: "New York Rangers",
    4: "Philadelphia Flyers", 5: "Pittsburgh Penguins", 6: "Boston Bruins",
    7: "Buffalo Sabres", 8: "Montréal Canadiens", 9: "Ottawa Senators",
    10: "Toronto Maple Leafs", 12: "Carolina Hurricanes", 13: "Florida Panthers",
    14: "Tampa Bay Lightning", 15: "Washington Capitals", 16: "Chicago Blackhawks",
    17: "Detroit Red Wings", 18: "Nashville Predators", 19: "St. Louis Blues",
    20: "Calgary Flames", 21: "Colorado Avalanche", 22: "Edmonton Oilers",
    23: "Vancouver Canucks", 24: "Anaheim Ducks", 25: "Dallas Stars",
    26: "Los Angeles Kings", 28: "San Jose Sharks", 29: "Columbus Blue Jackets",
    30: "Minnesota Wild", 52: "Winnipeg Jets", 53: "Arizona Coyotes",
    54: "Vegas Golden Knights", 55: "Seattle Kraken", 59: "Utah Hockey Club",
    68: "Utah Mammoth",
}

# team-context columns that cannot be reconstructed for a hypothetical matchup —
# neutralised so they never spuriously swing the prediction.
_ZERO_CTX = {
    "injured_pts_lost", "star_out", "injured_pts_lost_opp", "star_out_opp",
    "wowy_margin_delta", "key_players_out", "out_min_share",
    "wowy_margin_delta_opp", "key_players_out_opp", "out_min_share_opp",
}
_NULL_CTX = {"mkt_implied_prob", "mkt_total", "mkt_spread"}

_MARKET_ALIASES = {
    "moneyline": "moneyline", "ml": "moneyline", "matchwinner": "moneyline",
    "match_winner": "moneyline", "h2h": "moneyline",
    "totals": "totals", "total": "totals", "ou": "totals", "o/u": "totals",
    "spread": "spread", "handicap": "spread", "spreads": "spread",
    "prop": "prop", "props": "prop", "player_prop": "prop",
}


def _market(market: str) -> str:
    key = _MARKET_ALIASES.get(market.strip().lower().replace(" ", "_"))
    if key is None:
        raise ValueError(
            f"Unknown market '{market}'. Use one of: moneyline, totals, spread, prop."
        )
    return key


def _g(snap: dict, key: str) -> float:
    """Best-effort float from a snapshot row (missing/non-numeric -> NaN)."""
    v = snap.get(key)
    try:
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _resolve(name: str, snap: dict, kind: str) -> str:
    """Resolve a typed name to a snapshot key (exact, case-insensitive, or unique substring)."""
    if not snap:
        raise ValueError(f"No {kind} data available — run the pipeline to ingest data first.")
    if name in snap:
        return name
    low = name.strip().lower()
    exact = [k for k in snap if k.lower() == low]
    if exact:
        return exact[0]
    hits = [k for k in snap if low in k.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        sample = ", ".join(sorted(snap)[:4])
        raise ValueError(f"{kind} '{name}' not found. Known examples: {sample}, ...")
    raise ValueError(f"{kind} '{name}' is ambiguous — matches: {', '.join(sorted(hits)[:6])}")


# ── edge-dict assembly (matches edge.alerts.print_edges expectations) ────────────

def _ml_dicts(sport, game, label_a, odds_a, label_b, odds_b, prob_a, bankroll) -> list[dict]:
    """Two moneyline/match-winner edge dicts (one per side)."""
    prob_b = 1.0 - prob_a
    fair_a, fair_b = remove_vig(american_to_prob(odds_a), american_to_prob(odds_b))
    out = []
    for label, odds, mp, fp in (
        (label_a, odds_a, prob_a, fair_a),
        (label_b, odds_b, prob_b, fair_b),
    ):
        edge = calc_edge(mp, fp)
        out.append({
            "sport": sport.upper(), "market": "Moneyline", "game": game,
            "bet": f"{label} ML", "odds": int(odds),
            "model_prob": round(mp, 3), "fair_prob": round(fp, 3),
            "edge": round(edge, 3), "kelly_stake": round(kelly_stake(edge, odds, bankroll), 2),
        })
    return out


def _totals_dicts(sport, game, model_total, line, sides, bankroll) -> list[dict]:
    """Over/Under edge dicts from _over_under_edges output."""
    out = []
    for side, odds, mp, edge in sides:
        out.append({
            "sport": sport.upper(), "market": "Totals", "game": game,
            "bet": f"{side} {line}", "odds": int(odds),
            "model_total": round(model_total, 1), "model_prob": round(mp, 3),
            "book_total": line, "edge": round(edge, 3),
            "kelly_stake": round(kelly_stake(edge, odds, bankroll), 2),
        })
    return out


def _spread_dicts(sport, game, model_margin, line, sides, bankroll) -> list[dict]:
    """Spread/handicap edge dicts from _spread_edges output (side A line = `line`)."""
    out = []
    for i, (side, odds, mp, edge) in enumerate(sides):
        side_line = line if i == 0 else -line
        out.append({
            "sport": sport.upper(), "market": "Spread", "game": game,
            "bet": f"{side} {side_line:+g}", "odds": int(odds),
            "model_pred": round(model_margin, 1), "model_prob": round(mp, 3),
            "book_line": side_line, "edge": round(edge, 3),
            "kelly_stake": round(kelly_stake(edge, odds, bankroll), 2),
        })
    return out


# ── tennis ───────────────────────────────────────────────────────────────────

def tennis_matchup_edge(
    db_path: str, player_a: str, player_b: str, market: str = "moneyline", *,
    odds_a: int | None = None, odds_b: int | None = None,
    line: float | None = None, over_odds: int | None = None, under_odds: int | None = None,
    surface: str | None = None, bankroll: float = 1000.0,
) -> list[dict]:
    """Edge for a hand-picked ATP/WTA matchup. Markets: moneyline, totals, spread.

    moneyline: odds_a (player_a price), odds_b (player_b price).
    totals:    line (game total), over_odds, under_odds.
    spread:    line (player_a game handicap, e.g. -3.5), odds_a, odds_b.
    """
    mkt = _market(market)
    snap = tennis_player_snapshots(db_path)
    a = _resolve(player_a, snap, "player")
    b = _resolve(player_b, snap, "player")
    tour = snap[a].get("tour") or snap[b].get("tour")
    if not tour:
        raise ValueError(f"Could not determine tour (ATP/WTA) for {a} / {b}.")

    name = {"moneyline": "moneyline", "totals": "totals", "spread": "spread"}[mkt]
    artifact = _load(f"tennis_{tour}_{name}")
    X = tennis_feature_vector(snap[a], snap[b], artifact["feature_cols"], surface=surface).reshape(1, -1)
    game = f"{a} vs {b}"

    if mkt == "moneyline":
        _require(odds_a=odds_a, odds_b=odds_b)
        prob_a = float(_predict_proba(artifact, X)[0])
        return _ml_dicts("tennis", game, a, odds_a, b, odds_b, prob_a, bankroll)

    mean, sigma = (float(v[0]) for v in _predict_mean_sigma(artifact, X))
    if mkt == "totals":
        _require(line=line, over_odds=over_odds, under_odds=under_odds)
        sides = _over_under_edges(mean, sigma, line, over_odds, under_odds, min_edge=-1.0)
        return _totals_dicts("tennis", game, mean, line, sides, bankroll)

    # spread (game handicap from player_a perspective)
    _require(line=line, odds_a=odds_a, odds_b=odds_b)
    sides = _spread_edges(mean, sigma, line, odds_a, odds_b, min_edge=-1.0, a_label=a, b_label=b)
    return _spread_dicts("tennis", game, mean, line, sides, bankroll)


# ── NBA / NHL teams ─────────────────────────────────────────────────────────────

def _team_name(sport: str, row: dict) -> str | None:
    if sport == "nba":
        return (row.get("team_name") or "").strip() or None
    return NHL_TEAM_NAMES.get(row.get("team_id"))


def team_snapshots(db_path: str, sport: str) -> dict[str, dict]:
    """Latest game-feature row per team, indexed by full team name."""
    from features.pipeline import build_nba_game_features, build_nhl_game_features

    build = build_nba_game_features if sport == "nba" else build_nhl_game_features
    df = build(db_path)
    if df.is_empty():
        return {}
    latest = df.sort(["team_id", "game_date"]).group_by("team_id").last()
    out: dict[str, dict] = {}
    for row in latest.iter_rows(named=True):
        name = _team_name(sport, row)
        if name:
            out[name] = row
    return out


def _build_team_matchup_row(home: dict, away: dict, rest_days: float) -> dict:
    """Assemble one synthetic feature row for home-vs-away from each side's latest snapshot.

    Own stats come from the home team; every opp_* column is filled from the away team's
    same own-stat; matchup-derived columns are recomputed; unreconstructable context
    (injuries/WOWY/market line) is neutralised.
    """
    from models.train import TEAM_FEATURE_COLS

    row: dict[str, float] = {}
    for c in TEAM_FEATURE_COLS:
        if c in _ZERO_CTX:
            row[c] = 0.0
        elif c in _NULL_CTX:
            row[c] = np.nan
        elif c == "is_home":
            row[c] = 1.0
        elif c == "rest_days":
            row[c] = float(rest_days)
        elif c == "win_streak_5":
            row[c] = _g(home, c)
        elif c == "pace_matchup":
            row[c] = (_g(home, "pace_roll10") + _g(away, "pace_roll10")) / 2.0
        elif c == "off_def_edge":
            row[c] = _g(home, "off_rating_roll10") - _g(away, "off_rating_roll10")
        elif c.startswith("opp_"):
            row[c] = _g(away, c[4:])
        else:
            row[c] = _g(home, c)
    return row


def team_matchup_edge(
    db_path: str, sport: str, home: str, away: str, market: str = "moneyline", *,
    odds_a: int | None = None, odds_b: int | None = None,
    line: float | None = None, over_odds: int | None = None, under_odds: int | None = None,
    rest_days: float = 2.0, bankroll: float = 1000.0,
) -> list[dict]:
    """Edge for a hand-picked NBA/NHL matchup. Markets: moneyline, totals, spread.

    home/away designate the team perspective (home advantage is a model feature).
    moneyline: odds_a (home price), odds_b (away price).
    totals:    line (game total), over_odds, under_odds.
    spread:    line (home spread, e.g. -6.5), odds_a (home), odds_b (away).
    """
    sport = sport.lower()
    if sport not in ("nba", "nhl"):
        raise ValueError("team_matchup_edge supports sport 'nba' or 'nhl'.")
    mkt = _market(market)
    if mkt == "prop":
        raise ValueError("Use player_prop_edge for props, not team_matchup_edge.")

    snap = team_snapshots(db_path, sport)
    h = _resolve(home, snap, "team")
    a = _resolve(away, snap, "team")
    row = _build_team_matchup_row(snap[h], snap[a], rest_days)
    game = f"{a} @ {h}"

    name = {"moneyline": "moneyline", "totals": "totals", "spread": "spread"}[mkt]
    artifact = _load(f"{sport}_{name}")
    X = np.array([[row.get(c, np.nan) for c in artifact["feature_cols"]]], dtype=np.float32)

    if mkt == "moneyline":
        _require(odds_a=odds_a, odds_b=odds_b)
        prob_home = float(_predict_proba(artifact, X)[0])
        return _ml_dicts(sport, game, h, odds_a, a, odds_b, prob_home, bankroll)

    mean, sigma = (float(v[0]) for v in _predict_mean_sigma(artifact, X))
    if mkt == "totals":
        _require(line=line, over_odds=over_odds, under_odds=under_odds)
        sides = _over_under_edges(mean, sigma, line, over_odds, under_odds, min_edge=-1.0)
        return _totals_dicts(sport, game, mean, line, sides, bankroll)

    # spread: model predicts home margin (home_pts - away_pts); line is the home spread
    _require(line=line, odds_a=odds_a, odds_b=odds_b)
    sides = _spread_edges(mean, sigma, line, odds_a, odds_b, min_edge=-1.0, a_label=h, b_label=a)
    return _spread_dicts(sport, game, mean, line, sides, bankroll)


# ── player props (NBA only — NHL has no ingested player names) ───────────────────

def player_prop_edge(
    db_path: str, sport: str, player: str, stat: str,
    line: float, over_odds: int, under_odds: int, bankroll: float = 1000.0,
) -> list[dict]:
    """Edge for a single player's Over/Under prop (e.g. points 27.5)."""
    from models.train import PROP_STATS

    sport = sport.lower()
    if sport == "nhl":
        raise ValueError(
            "NHL props are not supported in manual mode: NHL player names are not "
            "ingested (nhl_player_games stores only player_id)."
        )
    if sport != "nba":
        raise ValueError("player_prop_edge supports sport 'nba'.")
    if stat not in PROP_STATS.get(sport, []):
        raise ValueError(f"Unknown {sport} prop stat '{stat}'. Choose from: {PROP_STATS[sport]}")

    snap = _player_snapshots(db_path, sport)
    p = _resolve(player, snap, "player")
    artifact = _load(f"{sport}_prop_{stat}")
    X = np.array([[_g(snap[p], c) for c in artifact["feature_cols"]]], dtype=np.float32)
    mean, sigma = (float(v[0]) for v in _predict_mean_sigma(artifact, X))

    sides = _over_under_edges(mean, sigma, line, over_odds, under_odds, min_edge=-1.0)
    out = []
    for side, odds, mp, edge in sides:
        out.append({
            "sport": sport.upper(), "market": f"Props ({stat})", "game": p,
            "bet": f"{p} {side} {line} {stat}", "odds": int(odds),
            "model_pred": round(mean, 1), "model_prob": round(mp, 3),
            "book_line": line, "edge": round(edge, 3),
            "kelly_stake": round(kelly_stake(edge, odds, bankroll), 2),
        })
    return out


def _player_snapshots(db_path: str, sport: str) -> dict[str, dict]:
    from features.pipeline import build_nba_player_features, build_nhl_player_features

    build = build_nba_player_features if sport == "nba" else build_nhl_player_features
    df = build(db_path)
    if df.is_empty():
        return {}
    latest = df.sort(["player_id", "game_date"]).group_by("player_id").last()
    out: dict[str, dict] = {}
    for row in latest.iter_rows(named=True):
        nm = (row.get("player_name") or "").strip()
        if nm:
            out[nm] = row
    return out


# ── dropdown enumerators (for the dashboard) ─────────────────────────────────────

def list_tennis_players(db_path: str) -> list[str]:
    return sorted(tennis_player_snapshots(db_path))


def list_teams(db_path: str, sport: str) -> list[str]:
    return sorted(team_snapshots(db_path, sport.lower()))


def list_players(db_path: str, sport: str) -> list[str]:
    return sorted(_player_snapshots(db_path, sport.lower()))


# ── small helpers ───────────────────────────────────────────────────────────────

def _require(**kwargs) -> None:
    """Raise if any required odds/line argument was not supplied."""
    missing = [k for k, v in kwargs.items() if v is None]
    if missing:
        raise ValueError(f"Missing required input(s) for this market: {', '.join(missing)}")
