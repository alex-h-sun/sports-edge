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

        if "model_prob" in e:
            print(f"  Model: {e['model_prob']*100:.1f}%  Book fair: {e['fair_prob']*100:.1f}%")
        elif "model_total" in e:
            print(f"  Model total: {e['model_total']}  Book line: {e['book_total']}")
        elif "model_pred" in e:
            print(f"  Model: {e['model_pred']}  Book line: {e['book_line']}")

    print(f"\n{'='*70}\n")


def save_edges(edges: list[dict], output_dir: str = "data/edges") -> Path:
    """Write edge bets to a dated CSV file for review."""
    if not edges:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"edges_{date.today().isoformat()}.csv"

    rows = [
        {**e, "odds_decimal": round(american_to_decimal(e["odds"]), 2)}
        for e in edges
    ]
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(edges)} edges to {path}")
    return path
