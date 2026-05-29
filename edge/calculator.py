"""Convert model output to implied probability and calculate edge vs. book."""

import pickle
from pathlib import Path

import numpy as np
import polars as pl

ARTIFACTS_DIR = Path(__file__).parent.parent / "models" / "artifacts"


# ── odds math ─────────────────────────────────────────────────────────────────

def american_to_prob(odds: int) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Strip vig using the additive method. Returns fair probabilities."""
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def calc_edge(model_prob: float, fair_prob: float) -> float:
    """Edge = model probability minus book's fair probability."""
    return model_prob - fair_prob


def kelly_stake(edge: float, odds: int, bankroll: float, fraction: float = 0.25) -> float:
    """Fractional Kelly criterion stake in dollars.

    fraction=0.25 (quarter Kelly) is standard for sports betting to
    reduce variance vs. full Kelly.
    """
    if edge <= 0:
        return 0.0
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / abs(odds)
    # Kelly fraction = (b*p - q) / b  where p=win prob, q=lose prob
    p = edge + american_to_prob(odds)  # model win prob
    q = 1 - p
    full_kelly = (b * p - q) / b
    return max(0.0, bankroll * fraction * full_kelly)


# ── model loading ─────────────────────────────────────────────────────────────

def _load(name: str) -> dict:
    path = ARTIFACTS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}. Run train_all() first.")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── edge finding ──────────────────────────────────────────────────────────────

def find_moneyline_edges(
    game_features: pl.DataFrame,
    sport: str,
    db_path: str,
    min_edge: float = 0.03,
    bankroll: float = 1000.0,
) -> list[dict]:
    """Find +EV moneyline bets for upcoming games.

    game_features: output of build_{sport}_game_features for today's games,
                   one row per team (home + away).
    Returns list of edge dicts sorted by edge descending.
    """
    import sqlite3
    artifact = _load(f"{sport}_moneyline")
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    home_df = game_features.filter(pl.col("is_home") == 1)
    if home_df.is_empty():
        return []

    present = [c for c in feature_cols if c in home_df.columns]
    X = home_df.select(present).to_numpy().astype(np.float32)

    # model gives P(home wins)
    home_win_prob = model.predict(X)

    # pull today's odds from DB
    conn = sqlite3.connect(db_path)
    try:
        odds_rows = conn.execute("""
            SELECT event_id, outcome_name, price
            FROM odds_snapshots
            WHERE sport = ? AND market = 'h2h' AND bookmaker = 'draftkings'
            AND DATE(fetched_at) = DATE('now')
            ORDER BY fetched_at DESC
        """, (sport,)).fetchall()
    except Exception:
        odds_rows = []
    conn.close()

    if not odds_rows:
        print("  No odds data found — run fetch_odds() or add ODDS_API_KEY to .env")
        return []

    # build odds lookup: event_id -> {team_name: american_odds}
    odds_map: dict[str, dict[str, int]] = {}
    for event_id, outcome_name, price in odds_rows:
        odds_map.setdefault(event_id, {})[outcome_name] = int(price)

    edges = []
    for i, row in enumerate(home_df.iter_rows(named=True)):
        model_home_prob = float(home_win_prob[i])
        model_away_prob = 1 - model_home_prob

        game_id = str(row.get("game_id", ""))
        book_odds = odds_map.get(game_id, {})
        if len(book_odds) < 2:
            continue

        teams = list(book_odds.keys())
        home_team = row.get("team_name", row.get("team_abbr", "HOME"))
        away_team = next((t for t in teams if t != home_team), teams[0])

        home_book_odds = book_odds.get(home_team)
        away_book_odds = book_odds.get(away_team)
        if not home_book_odds or not away_book_odds:
            continue

        raw_home = american_to_prob(home_book_odds)
        raw_away = american_to_prob(away_book_odds)
        fair_home, fair_away = remove_vig(raw_home, raw_away)

        home_edge = calc_edge(model_home_prob, fair_home)
        away_edge = calc_edge(model_away_prob, fair_away)

        for team, edge, odds, model_prob in [
            (home_team, home_edge, home_book_odds, model_home_prob),
            (away_team, away_edge, away_book_odds, model_away_prob),
        ]:
            if edge >= min_edge:
                edges.append({
                    "sport":      sport.upper(),
                    "market":     "Moneyline",
                    "game":       f"{away_team} @ {home_team}",
                    "bet":        f"{team} ML",
                    "odds":       odds,
                    "model_prob": round(model_prob, 3),
                    "fair_prob":  round(fair_home if team == home_team else fair_away, 3),
                    "edge":       round(edge, 3),
                    "kelly_stake": round(kelly_stake(edge, odds, bankroll), 2),
                })

    return sorted(edges, key=lambda x: x["edge"], reverse=True)


def find_totals_edges(
    game_features: pl.DataFrame,
    sport: str,
    db_path: str,
    min_edge_pts: float = 2.0,
    bankroll: float = 1000.0,
) -> list[dict]:
    """Find +EV totals bets. min_edge_pts = how far model must differ from book line."""
    import sqlite3
    artifact = _load(f"{sport}_totals")
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    home_df = game_features.filter(pl.col("is_home") == 1)
    if home_df.is_empty():
        return []

    present = [c for c in feature_cols if c in home_df.columns]
    X = home_df.select(present).to_numpy().astype(np.float32)
    predicted_totals = model.predict(X)

    conn = sqlite3.connect(db_path)
    try:
        odds_rows = conn.execute("""
            SELECT event_id, outcome_name, price, point
            FROM odds_snapshots
            WHERE sport = ? AND market = 'totals' AND bookmaker = 'draftkings'
            AND DATE(fetched_at) = DATE('now')
            ORDER BY fetched_at DESC
        """, (sport,)).fetchall()
    except Exception:
        odds_rows = []
    conn.close()

    if not odds_rows:
        return []

    # event_id -> {Over: (price, point), Under: (price, point)}
    odds_map: dict[str, dict] = {}
    for event_id, outcome_name, price, point in odds_rows:
        odds_map.setdefault(event_id, {})[outcome_name] = (int(price), point)

    edges = []
    for i, row in enumerate(home_df.iter_rows(named=True)):
        pred = float(predicted_totals[i])
        game_id = str(row.get("game_id", ""))
        book = odds_map.get(game_id, {})
        if "Over" not in book or "Under" not in book:
            continue

        over_price, book_total = book["Over"]
        under_price, _ = book["Under"]

        diff = pred - book_total
        if abs(diff) < min_edge_pts:
            continue

        bet_side = "Over" if diff > 0 else "Under"
        bet_odds = over_price if bet_side == "Over" else under_price

        # rough edge: each point difference ≈ 3% probability shift
        edge = min(abs(diff) * 0.03, 0.15)

        home_team = row.get("team_name", row.get("team_abbr", "HOME"))
        edges.append({
            "sport":        sport.upper(),
            "market":       "Totals",
            "game":         f"@ {home_team}",
            "bet":          f"{bet_side} {book_total}",
            "odds":         bet_odds,
            "model_total":  round(pred, 1),
            "book_total":   book_total,
            "edge":         round(edge, 3),
            "kelly_stake":  round(kelly_stake(edge, bet_odds, bankroll), 2),
        })

    return sorted(edges, key=lambda x: x["edge"], reverse=True)


def find_prop_edges(
    player_features: pl.DataFrame,
    sport: str,
    db_path: str,
    min_edge_units: float = 1.5,
    bankroll: float = 1000.0,
) -> list[dict]:
    """Find +EV player prop bets. min_edge_units = model must differ from line by N units."""
    import sqlite3
    from models.train import PROP_STATS

    conn = sqlite3.connect(db_path)
    try:
        odds_rows = conn.execute("""
            SELECT event_id, market, outcome_name, price, point
            FROM odds_snapshots
            WHERE sport = ? AND market LIKE 'player_%' AND bookmaker = 'draftkings'
            AND DATE(fetched_at) = DATE('now')
            ORDER BY fetched_at DESC
        """, (sport,)).fetchall()
    except Exception:
        odds_rows = []
    conn.close()

    if not odds_rows:
        return []

    # market+player -> (over_price, under_price, line)
    props_map: dict[tuple, dict] = {}
    for event_id, market, outcome_name, price, point in odds_rows:
        key = (event_id, market, outcome_name.split(" (")[0])
        props_map.setdefault(key, {})["price"] = int(price)
        props_map.setdefault(key, {})["point"] = point

    edges = []
    for stat in PROP_STATS.get(sport, []):
        try:
            artifact = _load(f"{sport}_prop_{stat}")
        except FileNotFoundError:
            continue
        model = artifact["model"]
        feature_cols = artifact["feature_cols"]

        present = [c for c in feature_cols if c in player_features.columns]
        sub = player_features.drop_nulls(subset=present)
        if sub.is_empty():
            continue

        X = sub.select(present).to_numpy().astype(np.float32)
        preds = model.predict(X)

        market_key = f"player_{stat}"

        for i, row in enumerate(sub.iter_rows(named=True)):
            pred = float(preds[i])
            player = row.get("player_name", "")
            game_id = str(row.get("game_id", ""))

            over_key  = (game_id, market_key, f"{player} Over")
            under_key = (game_id, market_key, f"{player} Under")

            over  = props_map.get(over_key, {})
            under = props_map.get(under_key, {})
            if not over or not under:
                continue

            line = over.get("point", 0)
            diff = pred - line
            if abs(diff) < min_edge_units:
                continue

            bet_side  = "Over" if diff > 0 else "Under"
            bet_odds  = over["price"] if bet_side == "Over" else under["price"]
            edge = min(abs(diff) * 0.04, 0.20)

            edges.append({
                "sport":       sport.upper(),
                "market":      f"Props ({stat})",
                "game":        f"game {game_id}",
                "bet":         f"{player} {bet_side} {line} {stat}",
                "odds":        bet_odds,
                "model_pred":  round(pred, 1),
                "book_line":   line,
                "edge":        round(edge, 3),
                "kelly_stake": round(kelly_stake(edge, bet_odds, bankroll), 2),
            })

    return sorted(edges, key=lambda x: x["edge"], reverse=True)
