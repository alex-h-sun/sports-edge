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
TENNIS_YEARS = list(range(2015, 2027))
TENNIS_TOURS = ["atp", "wta"]
EXPORTS_DIR = "data/exports"


def ingest_history(sport: str) -> None:
    print(f"\n[ingest] Pulling historical data for {sport.upper()}...")
    if sport == "nba":
        from ingestion.nba import fetch_seasons, fetch_player_stats
        fetch_seasons(NBA_SEASONS, DB_PATH)
        fetch_player_stats(NBA_SEASONS, DB_PATH)
    elif sport == "tennis":
        from ingestion.tennis import fetch_seasons, fetch_weather
        from ingestion.tennis_clean import clean
        fetch_seasons(TENNIS_YEARS, TENNIS_TOURS, DB_PATH)
        fetch_weather(DB_PATH)
        clean(DB_PATH)
    else:
        from ingestion.nhl import fetch_seasons
        fetch_seasons(SEASONS, DB_PATH)


def ingest_recent(sport: str) -> None:
    print(f"\n[ingest] Pulling recent games for {sport.upper()}...")
    if sport == "nba":
        from ingestion.nba import fetch_recent_games
        fetch_recent_games(days_back=7, db_path=DB_PATH)
    elif sport == "tennis":
        from ingestion.tennis import fetch_recent, fetch_weather
        from ingestion.tennis_clean import clean
        fetch_recent(DB_PATH, TENNIS_TOURS)
        fetch_weather(DB_PATH)
        clean(DB_PATH)
    else:
        from ingestion.nhl import fetch_recent_games
        fetch_recent_games(days_back=7, db_path=DB_PATH)


def ingest_injuries(sport: str) -> None:
    print(f"\n[ingest] Pulling injury report for {sport.upper()}...")
    from ingestion.injuries import fetch_injuries
    fetch_injuries(sport, DB_PATH)


def ingest_odds(sport: str) -> None:
    print(f"\n[ingest] Pulling odds for {sport.upper()}...")
    if sport == "tennis":
        from ingestion.odds import fetch_tennis_odds
        fetch_tennis_odds(DB_PATH)
        return
    from ingestion.odds import fetch_odds, fetch_props
    fetch_odds(sport, ["moneyline", "spread", "totals"], DB_PATH)
    fetch_props(sport, DB_PATH)


def train(sport: str) -> None:
    print(f"\n[train] Training models for {sport.upper()}...")
    from models.train import train_all
    train_all(sport, DB_PATH)


def backtest(sport: str) -> None:
    """Honest holdout backtest: train on pre-2025-26 data, score the held-out season.

    Saves `_holdout`-suffixed artifacts (never overwrites the production models) and
    prints test-season metrics per market. Use to validate the stack before betting.
    """
    print(f"\n[backtest] Holdout evaluation for {sport.upper()}...")
    from models.train import train_all, HOLDOUT_START, TENNIS_HOLDOUT_START
    cutoff = TENNIS_HOLDOUT_START if sport == "tennis" else HOLDOUT_START
    train_all(sport, DB_PATH, holdout_start=cutoff)


def run_matchup(args) -> None:
    """Manual edge for a single hand-picked matchup priced against odds you supply."""
    from edge.manual import tennis_matchup_edge, team_matchup_edge, player_prop_edge
    from edge.alerts import print_edges

    sport = args.sport
    if sport not in ("nba", "nhl", "tennis"):
        print("  --matchup needs a single --sport (nba, nhl, or tennis).")
        return

    market = (args.market or "moneyline").lower()
    try:
        if market in ("prop", "props") or args.player:
            edges = player_prop_edge(
                DB_PATH, sport, args.player, args.stat, args.line,
                args.odds_over, args.odds_under, bankroll=BANKROLL,
            )
        elif sport == "tennis":
            edges = tennis_matchup_edge(
                DB_PATH, args.a, args.b, market,
                odds_a=args.odds_a, odds_b=args.odds_b, line=args.line,
                over_odds=args.odds_over, under_odds=args.odds_under,
                surface=args.surface, bankroll=BANKROLL,
            )
        else:
            edges = team_matchup_edge(
                DB_PATH, sport, args.a, args.b, market,
                odds_a=args.odds_a, odds_b=args.odds_b, line=args.line,
                over_odds=args.odds_over, under_odds=args.odds_under,
                rest_days=args.rest_days, bankroll=BANKROLL,
            )
    except (ValueError, FileNotFoundError, TypeError) as e:
        print(f"  {e}")
        return

    print_edges(edges)


def find_edges(sport: str) -> list[dict]:
    print(f"\n[edge] Finding edges for {sport.upper()}...")

    if sport == "tennis":
        from edge.calculator import find_tennis_edges
        return find_tennis_edges(DB_PATH, min_edge=MIN_EDGE, bankroll=BANKROLL)

    from features.pipeline import (
        build_nba_game_features, build_nba_player_features,
        build_nhl_game_features, build_nhl_player_features,
        add_injury_features, add_teammate_wowy_features, add_market_features,
    )
    from edge.calculator import find_moneyline_edges, find_totals_edges, find_prop_edges

    if sport == "nba":
        game_df   = build_nba_game_features(DB_PATH)
        player_df = build_nba_player_features(DB_PATH)
    else:
        game_df   = build_nhl_game_features(DB_PATH)
        player_df = build_nhl_player_features(DB_PATH)

    game_df   = add_injury_features(game_df, sport, DB_PATH)
    game_df   = add_market_features(game_df, sport, DB_PATH)
    player_df = add_teammate_wowy_features(player_df, sport, DB_PATH)

    edges = []
    edges += find_moneyline_edges(game_df, sport, DB_PATH, min_edge=MIN_EDGE, bankroll=BANKROLL)
    edges += find_totals_edges(game_df, sport, DB_PATH, bankroll=BANKROLL)
    edges += find_prop_edges(player_df, sport, DB_PATH, bankroll=BANKROLL)

    return sorted(edges, key=lambda x: x["edge"], reverse=True)


def export_features(sport: str) -> None:
    """Dump features to parquet for upload to the cloud training tier (Colab)."""
    import os
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    if sport == "tennis":
        from features.pipeline import persist_tennis_features
        df = persist_tennis_features(DB_PATH)
        out = os.path.join(EXPORTS_DIR, "tennis_features.parquet")
    else:
        from features.pipeline import (
            build_nba_game_features, build_nhl_game_features,
            add_injury_features, add_market_features,
        )
        df = build_nba_game_features(DB_PATH) if sport == "nba" else build_nhl_game_features(DB_PATH)
        df = add_injury_features(df, sport, DB_PATH)
        df = add_market_features(df, sport, DB_PATH)
        out = os.path.join(EXPORTS_DIR, f"{sport}_game_features.parquet")
    df.write_parquet(out)
    print(f"  Exported {df.height} rows -> {out} (upload this to Colab)")


def export_sequences(sport: str) -> None:
    """Dump per-player chronological game logs for the cloud prop forecaster.

    Tier-4 (DeepAR / TFT) training runs on Colab, not locally — this only writes
    the input panel: one row per player-game with the prop targets and known
    covariates (minutes, rest, home/away, rolling/EWMA/Holt form). See
    docs/TRAINING_PROPS_FORECAST.md.
    """
    import os
    if sport == "tennis":
        print("  Prop forecaster covers NBA/NHL player props only; skipping tennis.")
        return
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    from features.pipeline import (
        build_nba_player_features, build_nhl_player_features, add_teammate_wowy_features,
    )
    df = build_nba_player_features(DB_PATH) if sport == "nba" else build_nhl_player_features(DB_PATH)
    df = add_teammate_wowy_features(df, sport, DB_PATH)
    out = os.path.join(EXPORTS_DIR, f"{sport}_player_sequences.parquet")
    df.write_parquet(out)
    print(f"  Exported {df.height} player-game rows -> {out} (upload this to Colab)")


def import_models() -> None:
    """Copy cloud-trained artifacts from data/exports/artifacts into models/artifacts."""
    import os, shutil, glob
    src = os.path.join(EXPORTS_DIR, "artifacts")
    dst = "models/artifacts"
    if not os.path.isdir(src):
        print(f"  No {src} found — download cloud artifacts there first")
        return
    os.makedirs(dst, exist_ok=True)
    n = 0
    for pat in ("tennis_*", "nba_*", "nhl_*"):
        for f in glob.glob(os.path.join(src, pat)):
            shutil.copy(f, dst)
            n += 1
    print(f"  Imported {n} cloud artifact(s) into {dst}")


def compare(sport: str) -> None:
    """Run the model bake-off (LightGBM/XGBoost/MLP/ensemble); best on Colab GPU."""
    if sport == "tennis":
        from models.compare import run_comparison
        for tour in TENNIS_TOURS:
            print(f"\n=== {tour.upper()} bake-off ===")
            run_comparison(db_path=DB_PATH, tour=tour)
    else:
        from models.compare_team import run_team_comparison
        print(f"\n=== {sport.upper()} bake-off ===")
        run_team_comparison(sport, db_path=DB_PATH)


def main():
    parser = argparse.ArgumentParser(description="Sports edge finder")
    parser.add_argument("--sport",          default="both", choices=["nba", "nhl", "tennis", "both", "all"])
    parser.add_argument("--train",          action="store_true", help="Retrain models before finding edges")
    parser.add_argument("--backtest",       action="store_true", help="Holdout backtest on the 2025-26 season (no ingest, no betting)")
    parser.add_argument("--ingest-history", action="store_true", help="Pull full historical data (first run only)")
    parser.add_argument("--no-odds",        action="store_true", help="Skip odds fetch (use cached)")
    parser.add_argument("--arb",            action="store_true", help="Arbitrage-only: pull current odds and scan for cross-book arbitrage (no model edges, no game-log ingest)")
    parser.add_argument("--compare",        action="store_true", help="Run the tennis model bake-off (cloud tier)")
    parser.add_argument("--export-features", action="store_true", help="Export tennis OBT to parquet for cloud training")
    parser.add_argument("--export-sequences", action="store_true", help="Export player game-log sequences for the cloud prop forecaster")
    parser.add_argument("--import-models",  action="store_true", help="Import cloud-trained artifacts locally")
    # manual matchup edge calculator (no ingest/odds/training — pure model vs your price)
    parser.add_argument("--matchup",  action="store_true", help="Manual edge for a hand-picked matchup + your odds (needs a single --sport)")
    parser.add_argument("--a", "--home", dest="a", help="Side A: player1 (tennis) or home team")
    parser.add_argument("--b", "--away", dest="b", help="Side B: player2 (tennis) or away team")
    parser.add_argument("--player", help="Player for a prop bet (NBA)")
    parser.add_argument("--market", default="moneyline", help="moneyline | totals | spread | prop")
    parser.add_argument("--stat", help="Prop stat, e.g. pts/reb/ast (with --market prop)")
    parser.add_argument("--line", type=float, help="Line: total, spread handicap, or prop line")
    parser.add_argument("--odds-a", type=int, help="American odds for side A (moneyline/spread)")
    parser.add_argument("--odds-b", type=int, help="American odds for side B (moneyline/spread)")
    parser.add_argument("--odds-over", type=int, help="American odds for Over (totals/prop)")
    parser.add_argument("--odds-under", type=int, help="American odds for Under (totals/prop)")
    parser.add_argument("--surface", help="Tennis surface override: hard|clay|grass|carpet")
    parser.add_argument("--rest-days", type=float, default=2.0, help="Team matchup rest-days assumption (default 2)")
    # paper-trading bankroll simulator (forward equity curve over real live edges).
    # On by default: every normal run settles finished bets and logs new >=5% edges.
    parser.add_argument("--no-paper",     action="store_true", help="Skip the paper-trading ledger update for this run")
    parser.add_argument("--sim-status",   action="store_true", help="Print the paper-trading bankroll summary and exit (no ingest/betting)")
    parser.add_argument("--sim-history",  action="store_true", help="Print every bet the paper sim has made and exit (no ingest/betting)")
    parser.add_argument("--sim-min-edge", type=float, default=None, help="Min edge to log in the paper sim (default 0.07)")
    parser.add_argument("--sim-bankroll", type=float, default=None, help="Starting bankroll for the paper sim (default env BANKROLL)")
    args = parser.parse_args()

    if args.sport == "both":
        sports = ["nba", "nhl"]
    elif args.sport == "all":
        sports = ["nba", "nhl", "tennis"]
    else:
        sports = [args.sport]

    # cloud-tier handoff shortcuts (no ingest needed)
    if args.import_models:
        import_models()
        return

    # holdout backtest: uses the existing DB, trains < cutoff, scores the test season
    if args.backtest:
        for sport in sports:
            backtest(sport)
        return

    # paper-sim status only: read the ledger, settle finished bets, print the curve.
    if args.sim_status:
        from models.paper_sim import load_ledger, settle_ledger, save_ledger, print_summary
        start = args.sim_bankroll if args.sim_bankroll is not None else BANKROLL
        rows = settle_ledger(load_ledger(), DB_PATH, start=start)
        save_ledger(rows)
        print_summary(rows, start)
        return

    # paper-sim full history only: settle finished bets, then list every bet made.
    if args.sim_history:
        from models.paper_sim import load_ledger, settle_ledger, save_ledger, print_history
        start = args.sim_bankroll if args.sim_bankroll is not None else BANKROLL
        rows = settle_ledger(load_ledger(), DB_PATH, start=start)
        save_ledger(rows)
        print_history(rows, start)
        return

    # manual matchup edge: model vs a price you type in. No ingest/odds/training.
    if args.matchup:
        run_matchup(args)
        return

    # arbitrage-only run: pure cross-book scan on right-now odds. No model edges,
    # so we skip game-log ingest/injuries/training and only pull current odds.
    if args.arb:
        if not args.no_odds:
            for sport in sports:
                try:
                    ingest_odds(sport)
                except EnvironmentError as e:
                    print(f"  Warning: {e} — skipping odds fetch, using cached odds")
        from edge.arbitrage import find_arbitrage
        from edge.alerts import print_arbs, save_arbs
        all_arbs = []
        for sport in sports:
            print(f"\n[arb] Scanning {sport.upper()} odds for arbitrage...")
            all_arbs += find_arbitrage(DB_PATH, sport, bankroll=BANKROLL)
        all_arbs.sort(key=lambda a: a["roi"], reverse=True)
        print_arbs(all_arbs)
        if all_arbs:
            save_arbs(all_arbs)
        return

    for sport in sports:
        if args.ingest_history:
            ingest_history(sport)

        ingest_recent(sport)
        if sport != "tennis":
            ingest_injuries(sport)

        if not args.no_odds:
            try:
                ingest_odds(sport)
            except EnvironmentError as e:
                print(f"  Warning: {e} — skipping odds fetch, edges won't include book comparison")

        if args.export_features:
            export_features(sport)
        if args.export_sequences:
            export_sequences(sport)
        if args.compare:
            compare(sport)
        if args.train:
            train(sport)

    if args.export_features or args.export_sequences or args.compare:
        return  # cloud-tier prep runs don't find edges

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

    # paper-trading ledger: settle finished bets, then log today's qualifying
    # moneyline edges as new open positions. Stakes are flat quarter-Kelly off the
    # fixed starting bankroll (not the running balance), so bet sizing never changes
    # as the curve moves. Runs by default; opt out with --no-paper.
    if not args.no_paper:
        from models.paper_sim import (
            load_ledger, settle_ledger, append_edges, save_ledger,
            print_summary, SIM_MIN_EDGE,
        )
        start = args.sim_bankroll if args.sim_bankroll is not None else BANKROLL
        min_edge = args.sim_min_edge if args.sim_min_edge is not None else SIM_MIN_EDGE
        rows = settle_ledger(load_ledger(), DB_PATH, start=start)
        rows = append_edges(rows, all_edges, start,
                            min_edge=min_edge, start=start)
        save_ledger(rows)
        print_summary(rows, start)


if __name__ == "__main__":
    main()
