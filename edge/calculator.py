"""Convert model output to implied probability and calculate edge vs. book."""

import os
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

# Default to the in-repo models/artifacts; override with the ARTIFACTS_DIR env var so a
# deployed server can read artifacts from a mounted snapshot volume (e.g. /data/artifacts).
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR") or Path(__file__).parent.parent / "models" / "artifacts")

# Isotonic calibrators saturate their tails to exactly 0.0/1.0, and no real
# sporting moneyline is a certainty. Clamp to a plausible band so an overconfident
# tail probability cannot inflate the edge (and the Kelly stake) past reason.
PROB_FLOOR, PROB_CEIL = 0.02, 0.98


# ── odds math ─────────────────────────────────────────────────────────────────

def american_to_prob(odds: int) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to European (decimal) odds, e.g. +100 -> 2.00."""
    if odds > 0:
        return odds / 100 + 1
    else:
        return 100 / abs(odds) + 1


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Strip vig using the additive method. Returns fair probabilities."""
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def calc_edge(model_prob: float, fair_prob: float) -> float:
    """Edge = model probability minus book's fair probability."""
    return model_prob - fair_prob


# ── per-book line shopping ──────────────────────────────────────────────────────

def book_odds_breakdown(db_path: str, sport: str, market: str) -> dict:
    """Latest American price per book for every outcome of a market.

    Returns {(event_id, outcome, point): {bookmaker: price}}, keyed by the bare
    outcome name (any " (line)" suffix the props feed appends is stripped). Used to
    attach a full cross-book breakdown to each found edge so the bettor can take the
    best available price rather than the single reference book the edge was priced on.
    """
    import sqlite3
    from collections import defaultdict

    op = "LIKE" if "%" in market else "="
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT event_id, outcome_name, point, bookmaker, price
            FROM odds_snapshots
            WHERE sport = ? AND market {op} ?
            AND DATE(fetched_at) = DATE('now')
            AND datetime(commence_time) > datetime('now')
            ORDER BY fetched_at ASC
            """,
            (sport, market),
        ).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    out: dict = defaultdict(dict)
    for event_id, outcome_name, point, book, price in rows:
        outcome = (outcome_name or "").split(" (")[0]
        out[(str(event_id), outcome, point)][book] = int(price)
    return out


def _attach_books(edge: dict, breakdown: dict, key: tuple) -> dict:
    """Attach the cross-book price map (and the best price) for an edge's outcome."""
    books = breakdown.get(key)
    if books:
        edge["book_odds"] = dict(books)
        best_book, best_price = max(books.items(), key=lambda kv: american_to_decimal(kv[1]))
        edge["best_book"] = best_book
        edge["best_odds"] = best_price
    return edge


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


# ── calibrated / distributional prediction ─────────────────────────────────────

def _predict_proba(artifact: dict, X: np.ndarray) -> np.ndarray:
    """Predict win probability, applying the isotonic calibrator if present.

    Calibrated probabilities are what `edge = model_prob - fair_prob` requires;
    without calibration the edge is systematically biased.
    """
    raw = np.asarray(artifact["model"].predict(X), dtype=float)
    cal = artifact.get("calibrator")
    if cal is not None:
        raw = cal.predict(raw)
    return np.clip(raw, PROB_FLOOR, PROB_CEIL)


def _predict_mean_sigma(artifact: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, sigma) per row for a regression market.

    sigma comes from the P15.87/P84.13 quantile models (half their spread). Falls
    back to a small constant if the quantile models are absent (older artifact).
    """
    mean = np.asarray(artifact["model"].predict(X), dtype=float)
    if "q_lo" in artifact and "q_hi" in artifact:
        lo = np.asarray(artifact["q_lo"].predict(X), dtype=float)
        hi = np.asarray(artifact["q_hi"].predict(X), dtype=float)
        sigma = np.maximum((hi - lo) / 2.0, 1e-3)
    else:
        sigma = np.full_like(mean, max(1.0, float(np.std(mean)) or 1.0))
    return mean, sigma


def _over_under_edges(
    mean: float, sigma: float, line: float,
    over_odds: int, under_odds: int, min_edge: float,
) -> list[tuple[str, int, float, float]]:
    """Price an Over/Under as a normal predictive distribution.

    Returns [(side, odds, model_prob, edge)] for sides whose edge >= min_edge,
    where edge = model P(side) - the book's vig-free P(side).
    """
    p_over = float(1.0 - norm.cdf((line - mean) / sigma))
    p_under = 1.0 - p_over
    fair_over, fair_under = remove_vig(american_to_prob(over_odds), american_to_prob(under_odds))
    out = []
    for side, odds, mp, fp in (
        ("Over", over_odds, p_over, fair_over),
        ("Under", under_odds, p_under, fair_under),
    ):
        edge = mp - fp
        if edge >= min_edge:
            out.append((side, odds, mp, edge))
    return out


def _spread_edges(
    mean: float, sigma: float, line: float,
    a_odds: int, b_odds: int, min_edge: float,
    a_label: str = "Home", b_label: str = "Away",
) -> list[tuple[str, int, float, float]]:
    """Price a point spread / handicap as a normal predictive distribution.

    `mean` is the model's predicted margin from side A's perspective (e.g. home
    points minus away points, or tennis player1 game margin). `line` is side A's
    handicap as written on the book (e.g. -6.5 if A is favoured by 6.5): A covers
    when margin + line > 0, i.e. margin > -line. Returns [(side, odds, model_prob,
    edge)] for sides whose edge >= min_edge, with edge = model P(cover) - the book's
    vig-free P(cover). Pushes are ignored (treated as the complement split).
    """
    p_a = float(1.0 - norm.cdf((-line - mean) / sigma))
    p_b = 1.0 - p_a
    fair_a, fair_b = remove_vig(american_to_prob(a_odds), american_to_prob(b_odds))
    out = []
    for label, odds, mp, fp in (
        (a_label, a_odds, p_a, fair_a),
        (b_label, b_odds, p_b, fair_b),
    ):
        edge = mp - fp
        if edge >= min_edge:
            out.append((label, odds, mp, edge))
    return out


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
    feature_cols = artifact["feature_cols"]

    home_df = game_features.filter(pl.col("is_home") == 1)
    if home_df.is_empty():
        return []

    present = [c for c in feature_cols if c in home_df.columns]
    X = home_df.select(present).to_numpy().astype(np.float32)

    # calibrated P(home wins)
    home_win_prob = _predict_proba(artifact, X)

    # pull today's odds from DB
    conn = sqlite3.connect(db_path)
    try:
        odds_rows = conn.execute("""
            SELECT event_id, outcome_name, price
            FROM odds_snapshots
            WHERE sport = ? AND market = 'h2h' AND bookmaker = 'draftkings'
            AND DATE(fetched_at) = DATE('now')
            AND datetime(commence_time) > datetime('now')
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

    breakdown = book_odds_breakdown(db_path, sport, "h2h")
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
                edges.append(_attach_books({
                    "sport":      sport.upper(),
                    "market":     "Moneyline",
                    "game":       f"{away_team} @ {home_team}",
                    "bet":        f"{team} ML",
                    "odds":       odds,
                    "model_prob": round(model_prob, 3),
                    "fair_prob":  round(fair_home if team == home_team else fair_away, 3),
                    "edge":       round(edge, 3),
                    "kelly_stake": round(kelly_stake(edge, odds, bankroll), 2),
                }, breakdown, (game_id, team, None)))

    return sorted(edges, key=lambda x: x["edge"], reverse=True)


def find_totals_edges(
    game_features: pl.DataFrame,
    sport: str,
    db_path: str,
    min_edge: float = 0.03,
    bankroll: float = 1000.0,
) -> list[dict]:
    """Find +EV totals bets, priced as a distribution.

    The model's predicted total + a per-game sigma (from quantile models) give
    P(Over) = 1 - Phi((line - mean)/sigma); edge = model P(side) - book fair P(side).
    min_edge is a probability threshold (not a points gap).
    """
    import sqlite3
    artifact = _load(f"{sport}_totals")
    feature_cols = artifact["feature_cols"]

    home_df = game_features.filter(pl.col("is_home") == 1)
    if home_df.is_empty():
        return []

    present = [c for c in feature_cols if c in home_df.columns]
    X = home_df.select(present).to_numpy().astype(np.float32)
    predicted_totals, predicted_sigma = _predict_mean_sigma(artifact, X)

    conn = sqlite3.connect(db_path)
    try:
        odds_rows = conn.execute("""
            SELECT event_id, outcome_name, price, point
            FROM odds_snapshots
            WHERE sport = ? AND market = 'totals' AND bookmaker = 'draftkings'
            AND DATE(fetched_at) = DATE('now')
            AND datetime(commence_time) > datetime('now')
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

    breakdown = book_odds_breakdown(db_path, sport, "totals")
    edges = []
    for i, row in enumerate(home_df.iter_rows(named=True)):
        pred = float(predicted_totals[i])
        sigma = float(predicted_sigma[i])
        game_id = str(row.get("game_id", ""))
        book = odds_map.get(game_id, {})
        if "Over" not in book or "Under" not in book:
            continue

        over_price, book_total = book["Over"]
        under_price, _ = book["Under"]

        home_team = row.get("team_name", row.get("team_abbr", "HOME"))
        for side, odds, model_prob, edge in _over_under_edges(
            pred, sigma, book_total, over_price, under_price, min_edge
        ):
            edges.append(_attach_books({
                "sport":        sport.upper(),
                "market":       "Totals",
                "game":         f"@ {home_team}",
                "bet":          f"{side} {book_total}",
                "odds":         odds,
                "model_total":  round(pred, 1),
                "model_prob":   round(model_prob, 3),
                "book_total":   book_total,
                "edge":         round(edge, 3),
                "kelly_stake":  round(kelly_stake(edge, odds, bankroll), 2),
            }, breakdown, (game_id, side, book_total)))

    return sorted(edges, key=lambda x: x["edge"], reverse=True)


def tennis_player_snapshots(db_path: str) -> dict[str, dict]:
    """Latest per-player feature snapshot, indexed by player name.

    One row per player (their most recent match), built from the same feature
    pipeline the tennis models train on. Shared by find_tennis_edges (auto) and the
    manual matchup calculator so both price off identical inputs.
    """
    from features.pipeline import build_tennis_player_features

    pf = build_tennis_player_features(db_path)
    if pf.is_empty():
        return {}
    latest = (
        pf.sort(["player_id", "seq_date"])
          .group_by("player_id")
          .last()
    )
    snap: dict[str, dict] = {}
    for row in latest.iter_rows(named=True):
        name = (row.get("player_name") or "").strip()
        if name:
            snap[name] = row
    return snap


def tennis_feature_vector(
    p1: dict, p2: dict, feature_cols: list[str], surface: str | None = None
) -> np.ndarray:
    """Build the model feature row for a player1-vs-player2 matchup.

    Diff features are player1 minus player2; context is taken from player1's most
    recent match. If `surface` (hard/clay/grass/carpet) is given it overrides the
    surface one-hot context, since odds feeds do not expose the court surface.
    """
    from features.pipeline import _DIFF_FEATURES, _CONTEXT_FEATURES, TENNIS_SURFACES

    vals: dict[str, float] = {}
    for f in _DIFF_FEATURES:
        a, b = p1.get(f), p2.get(f)
        vals[f"{f}_diff"] = (float(a) - float(b)) if a is not None and b is not None else np.nan
    # context proxied from player1's latest match
    for c in _CONTEXT_FEATURES:
        v = p1.get(c)
        vals[c] = float(v) if v is not None else np.nan
    if surface:
        s = surface.strip().lower()
        for name in TENNIS_SURFACES:
            vals[f"surface_{name.lower()}"] = 1.0 if name.lower() == s else 0.0
    return np.array([vals.get(c, np.nan) for c in feature_cols], dtype=np.float32)


def find_tennis_edges(
    db_path: str,
    min_edge: float = 0.03,
    bankroll: float = 1000.0,
) -> list[dict]:
    """Find +EV tennis match-winner bets for upcoming matches.

    Builds a latest feature snapshot per player from history, then for each live
    h2h odds event constructs the matchup diff features on the fly and predicts
    P(player1). Surface/context features use each player's most recent values as a
    best-effort proxy (The Odds API does not expose the surface). Player matching
    is by name, mirroring the NBA/NHL edge functions.
    """
    import sqlite3

    # ATP and WTA have separate models; load lazily per matchup and cache.
    model_cache: dict[str, dict] = {}

    def _tour_model(tour: str) -> dict | None:
        if tour not in model_cache:
            try:
                model_cache[tour] = _load(f"tennis_{tour}_moneyline")
            except FileNotFoundError:
                print(f"  No tennis_{tour}_moneyline model — run: python run.py --train --sport tennis")
                model_cache[tour] = None
        return model_cache[tour]

    snap = tennis_player_snapshots(db_path)
    if not snap:
        return []

    # pull today's tennis h2h odds
    conn = sqlite3.connect(db_path)
    try:
        odds_rows = conn.execute("""
            SELECT event_id, home_team, away_team, outcome_name, price
            FROM odds_snapshots
            WHERE sport = 'tennis' AND market = 'h2h' AND bookmaker = 'draftkings'
            AND DATE(fetched_at) = DATE('now')
            AND datetime(commence_time) > datetime('now')
            ORDER BY fetched_at DESC
        """).fetchall()
    except Exception:
        odds_rows = []
    conn.close()

    if not odds_rows:
        print("  No tennis odds found — no currently tracked tournaments are active")
        return []

    # event_id -> {player_name: price}
    events: dict[str, dict] = {}
    for event_id, home, away, name, price in odds_rows:
        events.setdefault(event_id, {})[name] = int(price)

    breakdown = book_odds_breakdown(db_path, "tennis", "h2h")

    edges: list[dict] = []
    for event_id, book in events.items():
        if len(book) < 2:
            continue
        names = list(book.keys())
        n1, n2 = names[0], names[1]
        if n1 not in snap or n2 not in snap:
            continue

        # both players are on the same tour; pick that tour's model
        tour = snap[n1].get("tour") or snap[n2].get("tour")
        artifact = _tour_model(tour)
        if artifact is None:
            continue
        feature_cols = artifact["feature_cols"]

        X = tennis_feature_vector(snap[n1], snap[n2], feature_cols).reshape(1, -1)
        p1_win = float(_predict_proba(artifact, X)[0])

        for name, model_prob, other in [(n1, p1_win, n2), (n2, 1 - p1_win, n1)]:
            o1, o2 = book[name], book[other]
            fair_self, _ = remove_vig(american_to_prob(o1), american_to_prob(o2))
            edge = calc_edge(model_prob, fair_self)
            if edge >= min_edge:
                edges.append(_attach_books({
                    "sport": "TENNIS",
                    "market": "Moneyline",
                    "game": f"{n1} vs {n2}",
                    "bet": f"{name} ML",
                    "odds": o1,
                    "model_prob": round(model_prob, 3),
                    "fair_prob": round(fair_self, 3),
                    "edge": round(edge, 3),
                    "kelly_stake": round(kelly_stake(edge, o1, bankroll), 2),
                }, breakdown, (str(event_id), name, None)))

    return sorted(edges, key=lambda x: x["edge"], reverse=True)


def find_prop_edges(
    player_features: pl.DataFrame,
    sport: str,
    db_path: str,
    min_edge: float = 0.03,
    bankroll: float = 1000.0,
) -> list[dict]:
    """Find +EV player prop bets, priced as a distribution.

    Each prop is priced with a mean and sigma -> P(Over) vs the book's vig-free
    line, so min_edge is a probability threshold. If the cloud prop forecaster
    (Tier 4) has been imported, its per-(player, stat) mean/sigma are preferred
    over the LightGBM point estimate + quantile sigma.
    """
    import sqlite3
    from models.train import PROP_STATS
    from models.prop_forecast import prop_forecast_features

    # enrich with the cloud forecaster's mean/sigma columns (no-op if not imported)
    player_features = prop_forecast_features(player_features, sport)

    conn = sqlite3.connect(db_path)
    try:
        odds_rows = conn.execute("""
            SELECT event_id, market, outcome_name, price, point
            FROM odds_snapshots
            WHERE sport = ? AND market LIKE 'player_%' AND bookmaker = 'draftkings'
            AND DATE(fetched_at) = DATE('now')
            AND datetime(commence_time) > datetime('now')
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
        means, sigmas = _predict_mean_sigma(artifact, X)

        # prefer the cloud forecaster's distribution when it covers this stat
        fc_mean = f"fc_{stat}_mean"
        fc_sigma = f"fc_{stat}_sigma"
        has_fc = fc_mean in sub.columns and fc_sigma in sub.columns
        fc_mean_col = sub[fc_mean].to_numpy() if has_fc else None
        fc_sigma_col = sub[fc_sigma].to_numpy() if has_fc else None

        market_key = f"player_{stat}"
        breakdown = book_odds_breakdown(db_path, sport, market_key)

        for i, row in enumerate(sub.iter_rows(named=True)):
            pred = float(means[i])
            sigma = float(sigmas[i])
            if has_fc and fc_mean_col[i] is not None and not np.isnan(fc_mean_col[i]):
                pred = float(fc_mean_col[i])
                sigma = max(float(fc_sigma_col[i]), 1e-3)
            player = row.get("player_name", "")
            game_id = str(row.get("game_id", ""))

            over_key  = (game_id, market_key, f"{player} Over")
            under_key = (game_id, market_key, f"{player} Under")

            over  = props_map.get(over_key, {})
            under = props_map.get(under_key, {})
            if not over or not under:
                continue

            line = over.get("point", 0)
            for side, odds, model_prob, edge in _over_under_edges(
                pred, sigma, line, over["price"], under["price"], min_edge
            ):
                edges.append(_attach_books({
                    "sport":       sport.upper(),
                    "market":      f"Props ({stat})",
                    "game":        f"game {game_id}",
                    "bet":         f"{player} {side} {line} {stat}",
                    "odds":        odds,
                    "model_pred":  round(pred, 1),
                    "model_prob":  round(model_prob, 3),
                    "book_line":   line,
                    "edge":        round(edge, 3),
                    "kelly_stake": round(kelly_stake(edge, odds, bankroll), 2),
                }, breakdown, (str(game_id), f"{player} {side}", line)))

    return sorted(edges, key=lambda x: x["edge"], reverse=True)
