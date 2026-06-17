"""Manual matchup edge calculator + dropdown options (reuses edge.manual).

Mirrors the Streamlit "Manual Edge Calculator": pick a matchup and type the book's
odds to get the model's edge + Kelly stake. Pure model-vs-price; no ingest.
"""

import logging

from webapp import settings

log = logging.getLogger("webapp.manual")


def _names(fn, *args) -> list[str]:
    """Resolve a name list, degrading to [] (never a 500) if the snapshot lacks data."""
    try:
        return fn(*args)
    except Exception:
        log.warning("name lookup failed: %s%s", getattr(fn, "__name__", fn), args, exc_info=True)
        return []


def options(sport: str) -> dict:
    """Dropdown data for one sport: entity names (+ players/prop stats for NBA)."""
    from edge import manual
    from models.train import PROP_STATS

    sport = (sport or "").lower()
    db = settings.DB_PATH
    if sport == "tennis":
        return {"sport": "tennis", "entities": _names(manual.list_tennis_players, db),
                "players": [], "prop_stats": []}
    return {
        "sport": sport,
        "entities": _names(manual.list_teams, db, sport),
        "players": _names(manual.list_players, db, sport) if sport == "nba" else [],
        "prop_stats": PROP_STATS.get(sport, []),
    }


def compute(body: dict) -> list[dict]:
    """Dispatch to the right edge.manual.* function. Raises KeyError/ValueError on
    bad input; the router maps those to HTTP 400."""
    from edge import manual

    sport = (body.get("sport") or "").lower()
    market = (body.get("market") or "moneyline").lower()
    bankroll = settings.BANKROLL

    if market in ("prop", "props"):
        return manual.player_prop_edge(
            settings.DB_PATH, sport, body["player"], body["stat"],
            float(body["line"]), int(body["over_odds"]), int(body["under_odds"]),
            bankroll=bankroll,
        )

    kw: dict = {"bankroll": bankroll}
    if market == "totals":
        kw.update(line=float(body["line"]),
                  over_odds=int(body["over_odds"]), under_odds=int(body["under_odds"]))
    elif market == "spread":
        kw.update(line=float(body["line"]),
                  odds_a=int(body["odds_a"]), odds_b=int(body["odds_b"]))
    else:  # moneyline
        kw.update(odds_a=int(body["odds_a"]), odds_b=int(body["odds_b"]))

    if sport == "tennis":
        return manual.tennis_matchup_edge(
            settings.DB_PATH, body["side_a"], body["side_b"], market,
            surface=body.get("surface"), **kw,
        )
    return manual.team_matchup_edge(
        settings.DB_PATH, sport, body["side_a"], body["side_b"], market,
        rest_days=float(body.get("rest_days", 2.0)), **kw,
    )
