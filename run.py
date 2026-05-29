"""Top-level runner. Ingest → features → predict → find edges → alert.

Usage:
    python run.py                        # full pipeline, both sports
    python run.py --sport nba            # NBA only
    python run.py --train                # retrain models before finding edges
    python run.py --ingest-history       # pull 5 seasons of history (first run)
"""

import argparse
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH   = os.getenv("DB_PATH", "data/sports.db")
BANKROLL  = float(os.getenv("BANKROLL", "1000"))
MIN_EDGE  = float(os.getenv("MIN_EDGE", "0.03"))
SEASONS   = ["20212022", "20222023", "20232024", "20242025", "20252026"]
NBA_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def ingest_history(sport: str) -> None:
    print(f"\n[ingest] Pulling historical data for {sport.upper()} ({len(SEASONS)} seasons)...")
    if sport == "nba":
        from ingestion.nba import fetch_seasons, fetch_player_stats
        fetch_seasons(NBA_SEASONS, DB_PATH)
        fetch_player_stats(NBA_SEASONS, DB_PATH)
    else:
        from ingestion.nhl import fetch_seasons
        fetch_seasons(SEASONS, DB_PATH)


def ingest_recent(sport: str) -> None:
    print(f"\n[ingest] Pulling recent games for {sport.upper()}...")
    if sport == "nba":
        from ingestion.nba import fetch_recent_games
        fetch_recent_games(days_back=7, db_path=DB_PATH)
    else:
        from ingestion.nhl import fetch_recent_games
        fetch_recent_games(days_back=7, db_path=DB_PATH)


def ingest_injuries(sport: str) -> None:
    print(f"\n[ingest] Pulling injury report for {sport.upper()}...")
    from ingestion.injuries import fetch_injuries
    fetch_injuries(sport, DB_PATH)


def ingest_odds(sport: str) -> None:
    print(f"\n[ingest] Pulling odds for {sport.upper()}...")
    from ingestion.odds import fetch_odds, fetch_props
    fetch_odds(sport, ["moneyline", "spread", "totals"], DB_PATH)
    fetch_props(sport, DB_PATH)


def train(sport: str) -> None:
    print(f"\n[train] Training models for {sport.upper()}...")
    from models.train import train_all
    train_all(sport, DB_PATH)


def find_edges(sport: str) -> list[dict]:
    print(f"\n[edge] Finding edges for {sport.upper()}...")
    from features.pipeline import (
        build_nba_game_features, build_nba_player_features,
        build_nhl_game_features, build_nhl_player_features,
        add_injury_features,
    )
    from edge.calculator import find_moneyline_edges, find_totals_edges, find_prop_edges

    if sport == "nba":
        game_df   = build_nba_game_features(DB_PATH)
        player_df = build_nba_player_features(DB_PATH)
    else:
        game_df   = build_nhl_game_features(DB_PATH)
        player_df = build_nhl_player_features(DB_PATH)

    game_df = add_injury_features(game_df, sport, DB_PATH)

    edges = []
    edges += find_moneyline_edges(game_df, sport, DB_PATH, min_edge=MIN_EDGE, bankroll=BANKROLL)
    edges += find_totals_edges(game_df, sport, DB_PATH, bankroll=BANKROLL)
    edges += find_prop_edges(player_df, sport, DB_PATH, bankroll=BANKROLL)

    return sorted(edges, key=lambda x: x["edge"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Sports edge finder")
    parser.add_argument("--sport",          default="both", choices=["nba", "nhl", "both"])
    parser.add_argument("--train",          action="store_true", help="Retrain models before finding edges")
    parser.add_argument("--ingest-history", action="store_true", help="Pull full historical data (first run only)")
    parser.add_argument("--no-odds",        action="store_true", help="Skip odds fetch (use cached)")
    args = parser.parse_args()

    sports = ["nba", "nhl"] if args.sport == "both" else [args.sport]

    for sport in sports:
        if args.ingest_history:
            ingest_history(sport)

        ingest_recent(sport)
        ingest_injuries(sport)

        if not args.no_odds:
            try:
                ingest_odds(sport)
            except EnvironmentError as e:
                print(f"  Warning: {e} — skipping odds fetch, edges won't include book comparison")

        if args.train:
            train(sport)

    # find and display edges
    all_edges = []
    for sport in sports:
        try:
            all_edges += find_edges(sport)
        except FileNotFoundError as e:
            print(f"  {e}")
            print(f"  Run: python run.py --train --sport {sport}")

    from edge.alerts import print_edges, save_edges
    print_edges(all_edges)
    if all_edges:
        save_edges(all_edges)


if __name__ == "__main__":
    main()
