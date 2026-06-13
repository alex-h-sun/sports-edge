"""Alert delivery for identified edges (console and CSV)."""

import csv
import json
from datetime import date
from pathlib import Path

from edge.calculator import american_to_decimal


def print_edges(edges: list[dict]) -> None:
    """Pretty-print edge bets to stdout."""
    if not edges:
        print("  No edges found above threshold.")
        return

    print(f"\n{'='*70}")
    print(f"  EDGES FOUND: {len(edges)}")
    print(f"{'='*70}")

    for e in edges:
        edge_pct = f"{e['edge']*100:.1f}%"
        stake    = f"${e['kelly_stake']:.2f}"
        odds_str = f"{american_to_decimal(e['odds']):.2f}"

        print(f"\n  {e['sport']} | {e['market']}")
        print(f"  Game:  {e['game']}")
        print(f"  Bet:   {e['bet']}  ({odds_str})")
        print(f"  Edge:  {edge_pct}  |  Kelly stake: {stake}")

        # branch on the market-distinguishing field: moneyline carries fair_prob,
        # totals carries model_total, props carries model_pred (all three also
        # carry model_prob, so it cannot be used to tell them apart).
        if "fair_prob" in e:
            print(f"  Model: {e['model_prob']*100:.1f}%  Book fair: {e['fair_prob']*100:.1f}%")
        elif "model_total" in e:
            print(f"  Model total: {e['model_total']}  Book line: {e['book_total']}")
        elif "model_pred" in e:
            print(f"  Model: {e['model_pred']}  Book line: {e['book_line']}")

        if e.get("book_odds"):
            shop = "  ".join(
                f"{b}:{american_to_decimal(p):.2f}"
                for b, p in sorted(
                    e["book_odds"].items(),
                    key=lambda kv: american_to_decimal(kv[1]),
                    reverse=True,
                )
            )
            print(f"  Books: {shop}")
            print(f"  Best:  {e['best_book']} ({american_to_decimal(e['best_odds']):.2f})")

    print(f"\n{'='*70}\n")


def save_edges(edges: list[dict], output_dir: str = "data/edges") -> Path:
    """Write edge bets to a dated CSV file for review."""
    if not edges:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"edges_{date.today().isoformat()}.csv"

    rows = []
    for e in edges:
        row = {**e, "odds_decimal": round(american_to_decimal(e["odds"]), 2)}
        # dict/list fields (e.g. book_odds) must be flattened for CSV
        for k, v in list(row.items()):
            if isinstance(v, (dict, list)):
                row[k] = json.dumps(v)
        rows.append(row)

    # edges have heterogeneous keys (moneyline vs totals vs props); union them so
    # no row's columns are dropped by keying off only the first row.
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(edges)} edges to {path}")
    return path


def print_arbs(arbs: list[dict]) -> None:
    """Pretty-print cross-book arbitrage opportunities to stdout."""
    if not arbs:
        print("  No arbitrage opportunities found.")
        return

    print(f"\n{'='*70}")
    print(f"  ARBITRAGE FOUND: {len(arbs)}")
    print(f"{'='*70}")

    for a in arbs:
        line = f" ({a['point']})" if a.get("point") is not None else ""
        print(f"\n  {a['sport']} | {a['market']}{line}")
        print(f"  Game:   {a['game']}")
        print(f"  Margin: {a['margin']*100:.2f}%  |  ROI: {a['roi']*100:.2f}%  |  Profit: ${a['profit']:.2f}")
        for leg in a["legs"]:
            print(
                f"    {leg['outcome']:<10} {leg['book']:<12} "
                f"{leg['decimal']:.2f}  stake ${leg['stake']:.2f} -> ${leg['payout']:.2f}"
            )

    print(f"\n{'='*70}\n")


def save_arbs(arbs: list[dict], output_dir: str = "data/arbs") -> Path:
    """Write arbitrage opportunities to a dated CSV file (one row per leg)."""
    if not arbs:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"arbs_{date.today().isoformat()}.csv"

    rows = []
    for a in arbs:
        for leg in a["legs"]:
            rows.append({
                "sport": a["sport"], "market": a["market"], "game": a["game"],
                "point": a["point"], "margin": a["margin"], "roi": a["roi"],
                "profit": a["profit"], **{f"leg_{k}": v for k, v in leg.items()},
            })
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(arbs)} arbs to {path}")
    return path
