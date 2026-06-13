"""Model-free arbitrage finder.

Scans the multi-book odds already stored in `odds_snapshots` for guaranteed-profit
lines. For a two-outcome market an arbitrage exists when the best decimal price for
each outcome, taken across books, satisfies:

    1/decimal(best_A) + 1/decimal(best_B) < 1

The shortfall below 1 is the locked margin; stakes are split inversely to the decimal
odds so the payout is identical whichever outcome hits. This reads stored snapshots
only — it costs no extra Odds API quota. Real arbs between mainstream US books are
rare and short-lived, so treat the output as line-shopping with the occasional true
arb rather than a steady income stream.
"""

import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

from edge.calculator import american_to_decimal

# Strictly two-outcome markets that can be arbed as-is. (Spreads/props are also
# two-way but only arb against the *same* point, which the grouping below handles.)
TWO_WAY_MARKETS = ("h2h", "totals")


def _latest_book_prices(
    db_path: str, sport: str, markets: tuple[str, ...]
) -> dict:
    """Most recent price per (event, market, point, outcome, book).

    Returns {(event_id, market, point): {"teams": (home, away),
                                         "outcomes": {outcome: {book: price}}}}.
    Ordering by fetched_at ascending means the last write per key wins (= latest).

    Only events that have not yet started (`commence_time` in the future) are
    returned — an arb is only real if both legs can still be placed, so stale odds
    from already-completed games sitting in the history table must be excluded.
    """
    now = datetime.now(timezone.utc).isoformat()
    placeholders = ",".join("?" for _ in markets)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT event_id, market, point, outcome_name, bookmaker, price,
                   home_team, away_team, fetched_at
            FROM odds_snapshots
            WHERE sport = ? AND market IN ({placeholders})
            AND commence_time > ?
            ORDER BY fetched_at ASC
            """,
            (sport, *markets, now),
        ).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    grouped: dict = defaultdict(lambda: {"teams": ("", ""), "outcomes": defaultdict(dict)})
    for event_id, market, point, outcome, book, price, home, away, _ in rows:
        key = (str(event_id), market, point)
        grouped[key]["teams"] = (home, away)
        grouped[key]["outcomes"][outcome][book] = int(price)
    return grouped


def find_arbitrage(
    db_path: str,
    sport: str,
    markets: tuple[str, ...] = TWO_WAY_MARKETS,
    min_margin: float = 0.0,
    bankroll: float = 1000.0,
) -> list[dict]:
    """Find cross-book arbitrage opportunities for a sport.

    min_margin is the minimum locked margin (e.g. 0.01 = require >=1% guaranteed).
    Returns a list of arb dicts sorted by ROI descending; each carries the two legs
    (outcome, book, odds, stake) and the guaranteed return on `bankroll`.
    """
    grouped = _latest_book_prices(db_path, sport, markets)

    arbs: list[dict] = []
    for (event_id, market, point), info in grouped.items():
        outcomes = info["outcomes"]
        if len(outcomes) != 2:  # only price clean two-way markets
            continue

        # best (highest decimal) price per outcome, with the book offering it
        best: dict[str, tuple[str, int, float]] = {}
        for outcome, books in outcomes.items():
            book, price = max(books.items(), key=lambda kv: american_to_decimal(kv[1]))
            best[outcome] = (book, price, american_to_decimal(price))

        inv_sum = sum(1.0 / d for (_, _, d) in best.values())
        if inv_sum >= 1.0 - min_margin:  # no arb (or below margin floor)
            continue

        roi = 1.0 / inv_sum - 1.0
        home, away = info["teams"]
        legs = []
        for outcome, (book, price, d) in best.items():
            stake = bankroll * (1.0 / d) / inv_sum
            legs.append({
                "outcome": outcome,
                "book":    book,
                "odds":    price,
                "decimal": round(d, 3),
                "stake":   round(stake, 2),
                "payout":  round(stake * d, 2),
            })

        arbs.append({
            "sport":       sport.upper(),
            "market":      market,
            "event_id":    event_id,
            "game":        f"{away} @ {home}" if home or away else event_id,
            "point":       point,
            "margin":      round(1.0 - inv_sum, 4),
            "roi":         round(roi, 4),
            "profit":      round(bankroll * roi, 2),
            "legs":        legs,
        })

    return sorted(arbs, key=lambda a: a["roi"], reverse=True)
