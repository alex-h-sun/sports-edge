"""Streamlit dashboard for sports edge finder."""

import os
import sqlite3
from datetime import date, datetime

import polars as pl
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DB_PATH  = os.getenv("DB_PATH", "data/test.db")
BANKROLL = float(os.getenv("BANKROLL", "1000"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.03"))

st.set_page_config(
    page_title="Sports Edge Finder",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Sports Edge Finder")
st.caption(f"DB: `{DB_PATH}`  |  Bankroll: ${BANKROLL:,.0f}  |  Min edge: {MIN_EDGE*100:.0f}%")


# ── sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")
    sport_filter = st.multiselect("Sports", ["NBA", "NHL"], default=["NBA", "NHL"])
    market_filter = st.multiselect(
        "Markets",
        ["Moneyline", "Spread", "Totals", "Props"],
        default=["Moneyline", "Totals", "Props"],
    )
    min_edge_slider = st.slider("Min edge %", 0, 20, int(MIN_EDGE * 100), step=1)
    min_edge = min_edge_slider / 100

    st.divider()
    refresh = st.button("🔄 Refresh data", use_container_width=True)
    retrain = st.button("🧠 Retrain models", use_container_width=True)

    st.divider()
    st.caption("**First time setup:**")
    if st.button("⬇️ Pull historical data", use_container_width=True):
        with st.spinner("Pulling 5 seasons of data... (this takes a few minutes)"):
            for sport in ["nba", "nhl"]:
                try:
                    if sport == "nba":
                        from ingestion.nba import fetch_seasons, fetch_player_stats
                        fetch_seasons(["2021-22","2022-23","2023-24","2024-25","2025-26"], DB_PATH)
                        fetch_player_stats(["2021-22","2022-23","2023-24","2024-25","2025-26"], DB_PATH)
                    else:
                        from ingestion.nhl import fetch_seasons
                        fetch_seasons(["20212022","20222023","20232024","20242025","20252026"], DB_PATH)
                except Exception as e:
                    st.error(f"{sport.upper()} ingest failed: {e}")
        st.success("Done! Click 'Retrain models' next.")


# ── cached pipeline ───────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Running pipeline...")
def run_pipeline(sports: list[str], _refresh_key: int) -> tuple[list[dict], list[dict]]:
    """Run full ingestion + edge finding. Returns (edges, injuries)."""
    all_edges = []
    all_injuries = []

    for sport in [s.lower() for s in sports]:
        # recent games + injuries
        try:
            if sport == "nba":
                from ingestion.nba import fetch_recent_games
                fetch_recent_games(days_back=7, db_path=DB_PATH)
            else:
                from ingestion.nhl import fetch_recent_games
                fetch_recent_games(days_back=7, db_path=DB_PATH)
        except Exception:
            pass

        try:
            from ingestion.injuries import fetch_injuries
            injuries = fetch_injuries(sport, DB_PATH)
            all_injuries.extend(injuries)
        except Exception:
            pass

        # odds
        try:
            from ingestion.odds import fetch_odds, fetch_props
            fetch_odds(sport, ["moneyline", "spread", "totals"], DB_PATH)
            fetch_props(sport, DB_PATH)
        except EnvironmentError:
            pass
        except Exception:
            pass

        # features + edges
        try:
            if sport == "nba":
                from features.pipeline import build_nba_game_features, build_nba_player_features, add_injury_features
                game_df   = build_nba_game_features(DB_PATH)
                player_df = build_nba_player_features(DB_PATH)
            else:
                from features.pipeline import build_nhl_game_features, build_nhl_player_features, add_injury_features
                game_df   = build_nhl_game_features(DB_PATH)
                player_df = build_nhl_player_features(DB_PATH)

            game_df = add_injury_features(game_df, sport, DB_PATH)

            from edge.calculator import find_moneyline_edges, find_totals_edges, find_prop_edges
            all_edges += find_moneyline_edges(game_df, sport, DB_PATH, min_edge=0.0, bankroll=BANKROLL)
            all_edges += find_totals_edges(game_df, sport, DB_PATH, bankroll=BANKROLL)
            all_edges += find_prop_edges(player_df, sport, DB_PATH, bankroll=BANKROLL)
        except ValueError as e:
            st.warning(f"{sport.upper()}: {e}")
        except FileNotFoundError as e:
            st.warning(f"{sport.upper()}: {e}")
        except Exception as e:
            st.warning(f"{sport.upper()} pipeline error: {e}")

    return all_edges, all_injuries


# ── refresh key (forces cache invalidation) ───────────────────────────────────

if "refresh_key" not in st.session_state:
    st.session_state.refresh_key = 0
if refresh:
    st.session_state.refresh_key += 1
if retrain:
    with st.spinner("Retraining models..."):
        for sport in [s.lower() for s in sport_filter]:
            try:
                from models.train import train_all
                train_all(sport, DB_PATH)
            except Exception as e:
                st.error(f"Training failed for {sport.upper()}: {e}")
    st.success("Models retrained!")
    st.session_state.refresh_key += 1

all_edges, all_injuries = run_pipeline(
    tuple(sport_filter), st.session_state.refresh_key
)

# ── filter edges ──────────────────────────────────────────────────────────────

filtered = [
    e for e in all_edges
    if e["edge"] >= min_edge
    and e["sport"] in sport_filter
    and any(m in e["market"] for m in market_filter)
]
filtered.sort(key=lambda x: x["edge"], reverse=True)


# ── summary metrics ───────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Edges found", len(filtered))
col2.metric("Total Kelly stake", f"${sum(e['kelly_stake'] for e in filtered):,.2f}")
col3.metric(
    "Best edge",
    f"{max((e['edge'] for e in filtered), default=0)*100:.1f}%"
)
col4.metric(
    "Data as of",
    datetime.now().strftime("%H:%M"),
)

st.divider()


# ── edges table ───────────────────────────────────────────────────────────────

st.subheader("📋 Value Bets")

if not filtered:
    st.info("No edges found above threshold. Add your Odds API key to `.env` to see live bets, or lower the min edge slider.")
else:
    rows = []
    for e in filtered:
        odds_str = f"+{e['odds']}" if e['odds'] > 0 else str(e['odds'])
        detail = ""
        if "model_prob" in e:
            detail = f"Model {e['model_prob']*100:.1f}% vs fair {e['fair_prob']*100:.1f}%"
        elif "model_total" in e:
            detail = f"Model {e['model_total']} vs line {e['book_total']}"
        elif "model_pred" in e:
            detail = f"Model {e['model_pred']} vs line {e['book_line']}"

        rows.append({
            "Sport":    e["sport"],
            "Market":   e["market"],
            "Game":     e["game"],
            "Bet":      e["bet"],
            "Odds":     odds_str,
            "Edge":     f"{e['edge']*100:.1f}%",
            "Kelly $":  f"${e['kelly_stake']:.2f}",
            "Detail":   detail,
        })

    df = pl.DataFrame(rows)
    st.dataframe(
        df.to_pandas(),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Edge":    st.column_config.TextColumn(width="small"),
            "Kelly $": st.column_config.TextColumn(width="small"),
            "Odds":    st.column_config.TextColumn(width="small"),
        }
    )

    # save button
    if st.button("💾 Save to CSV"):
        from edge.alerts import save_edges
        path = save_edges(filtered)
        st.success(f"Saved to `{path}`")


# ── injury report ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("🏥 Injury Report")

if all_injuries:
    inj_cols = st.columns(2)
    for i, sport_name in enumerate(["nba", "nhl"]):
        sport_injuries = [x for x in all_injuries if x["sport"] == sport_name]
        with inj_cols[i]:
            st.markdown(f"**{sport_name.upper()} ({len(sport_injuries)} players)**")
            if sport_injuries:
                inj_df = pl.DataFrame([{
                    "Player":  x["player_name"],
                    "Team":    x["team_name"],
                    "Status":  x["status"],
                    "Injury":  x["injury_type"],
                } for x in sport_injuries])
                st.dataframe(
                    inj_df.to_pandas(),
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )
else:
    st.info("No injury data loaded yet. Click 🔄 Refresh data.")


# ── manual edge calculator ──────────────────────────────────────────────────────

st.divider()
st.subheader("🎯 Manual Edge Calculator")
st.caption(
    "Pick a matchup and type the book's current odds to get the model's edge + Kelly "
    "stake for that exact bet. Evaluates a neutral, current-form matchup (no injuries, "
    "default rest, surface proxied unless set) — good for 'is this price +EV right now'."
)


@st.cache_data(ttl=3600, show_spinner="Loading names...")
def _manual_names(kind: str, sport: str | None = None) -> list[str]:
    from edge import manual
    try:
        if kind == "tennis":
            return manual.list_tennis_players(DB_PATH)
        if kind == "team":
            return manual.list_teams(DB_PATH, sport)
        return manual.list_players(DB_PATH, sport)
    except Exception:
        return []


def _odds_input(col, label: str, default: int = -110):
    return int(col.number_input(label, min_value=-2000, max_value=2000, value=default, step=5))


mc1, mc2 = st.columns(2)
m_sport = mc1.selectbox("Sport", ["Tennis", "NBA", "NHL"], key="m_sport")
_markets = ["Moneyline", "Totals", "Spread"] + (["Prop"] if m_sport == "NBA" else [])
m_market = mc2.selectbox("Market", _markets, key="m_market")
_sport_key = {"Tennis": "tennis", "NBA": "nba", "NHL": "nhl"}[m_sport]
_is_tennis = m_sport == "Tennis"
_entity = "Player" if _is_tennis else "Team"

manual_edges: list[dict] = []
manual_error: str | None = None

if m_market == "Prop":
    from models.train import PROP_STATS
    players = _manual_names("player", _sport_key)
    pc1, pc2 = st.columns(2)
    sel_player = pc1.selectbox("Player", players, key="m_player") if players else None
    sel_stat = pc2.selectbox("Stat", PROP_STATS.get(_sport_key, []), key="m_stat")
    lc1, lc2, lc3 = st.columns(3)
    prop_line = float(lc1.number_input("Line", value=20.5, step=0.5, key="m_prop_line"))
    over_odds = _odds_input(lc2, "Over odds")
    under_odds = _odds_input(lc3, "Under odds")
    if st.button("Calculate edge", type="primary", key="m_go_prop") and sel_player:
        from edge.manual import player_prop_edge
        try:
            manual_edges = player_prop_edge(
                DB_PATH, _sport_key, sel_player, sel_stat, prop_line,
                over_odds, under_odds, bankroll=BANKROLL,
            )
        except Exception as e:
            manual_error = str(e)
else:
    names = _manual_names("tennis") if _is_tennis else _manual_names("team", _sport_key)
    ec1, ec2 = st.columns(2)
    label_a = f"{_entity} A" + (" (player 1)" if _is_tennis else " (home)")
    label_b = f"{_entity} B" + (" (player 2)" if _is_tennis else " (away)")
    side_a = ec1.selectbox(label_a, names, key="m_a") if names else None
    side_b = ec2.selectbox(label_b, names, index=min(1, len(names) - 1), key="m_b") if names else None

    surface = None
    if _is_tennis:
        s = st.selectbox("Surface (optional)", ["(proxy from last match)", "hard", "clay", "grass", "carpet"], key="m_surf")
        surface = None if s.startswith("(") else s

    if m_market == "Totals":
        lc1, lc2, lc3 = st.columns(3)
        line = float(lc1.number_input("Total line", value=(22.5 if _is_tennis else 220.5), step=0.5, key="m_tot_line"))
        over_odds = _odds_input(lc2, "Over odds")
        under_odds = _odds_input(lc3, "Under odds")
    elif m_market == "Spread":
        lc1, lc2, lc3 = st.columns(3)
        line = float(lc1.number_input(f"{_entity} A handicap", value=(-3.5 if _is_tennis else -5.5), step=0.5, key="m_spr_line"))
        odds_a = _odds_input(lc2, f"{_entity} A odds")
        odds_b = _odds_input(lc3, f"{_entity} B odds")
    else:  # Moneyline
        lc1, lc2 = st.columns(2)
        odds_a = _odds_input(lc1, f"{_entity} A odds", default=-150)
        odds_b = _odds_input(lc2, f"{_entity} B odds", default=130)

    if st.button("Calculate edge", type="primary", key="m_go") and side_a and side_b:
        from edge.manual import tennis_matchup_edge, team_matchup_edge
        kw = dict(bankroll=BANKROLL)
        if m_market == "Totals":
            kw.update(line=line, over_odds=over_odds, under_odds=under_odds)
        elif m_market == "Spread":
            kw.update(line=line, odds_a=odds_a, odds_b=odds_b)
        else:
            kw.update(odds_a=odds_a, odds_b=odds_b)
        try:
            if _is_tennis:
                manual_edges = tennis_matchup_edge(DB_PATH, side_a, side_b, m_market.lower(), surface=surface, **kw)
            else:
                manual_edges = team_matchup_edge(DB_PATH, _sport_key, side_a, side_b, m_market.lower(), **kw)
        except Exception as e:
            manual_error = str(e)

if manual_error:
    st.error(manual_error)
elif manual_edges:
    rows = []
    for e in manual_edges:
        if "fair_prob" in e:
            detail = f"Model {e['model_prob']*100:.1f}% vs fair {e['fair_prob']*100:.1f}%"
        elif "model_total" in e:
            detail = f"Model total {e['model_total']} vs line {e['book_total']}"
        else:
            detail = f"Model {e['model_pred']} vs line {e['book_line']}"
        rows.append({
            "Bet": e["bet"],
            "Odds": f"+{e['odds']}" if e["odds"] > 0 else str(e["odds"]),
            "Edge": f"{e['edge']*100:+.1f}%",
            "Kelly $": f"${e['kelly_stake']:.2f}",
            "Detail": detail,
        })
    best = max(manual_edges, key=lambda x: x["edge"])
    if best["edge"] > 0:
        st.success(f"✅ Best +EV side: **{best['bet']}** — edge {best['edge']*100:+.1f}%, Kelly ${best['kelly_stake']:.2f}")
    else:
        st.warning("No +EV side at these prices — both sides are -EV vs the model.")
    st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True, hide_index=True)


# ── quota tracker ─────────────────────────────────────────────────────────────

with st.expander("📊 Odds API quota"):
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT remaining, used, fetched_at FROM odds_requests_remaining ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            st.metric("Requests remaining", row[0])
            st.metric("Requests used", row[1])
            st.caption(f"Last checked: {row[2]}")
        else:
            st.info("No quota data yet. Add ODDS_API_KEY to .env and refresh.")
    except Exception:
        st.info("No quota data yet.")
