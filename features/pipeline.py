"""Feature engineering pipeline. Reads from SQLite, returns Polars DataFrames.

All rolling features are computed from strictly prior games (shift(1) before
the window) to prevent data leakage into model training.
"""

import sqlite3
import polars as pl

from features.forecast import ewm_expr, add_holt_features, add_totals_series


# ── helpers ──────────────────────────────────────────────────────────────────

def _read(db_path: str, query: str) -> pl.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pl.read_database(query, conn)
    conn.close()
    return df


def _rolling_mean(df: pl.DataFrame, col: str, window: int, group_col: str = "team_id") -> pl.Expr:
    """Mean of `col` over the previous `window` games (no leakage)."""
    return (
        pl.col(col)
        .shift(1)
        .rolling_mean(window_size=window, min_samples=1)
        .over(group_col)
        .alias(f"{col}_roll{window}")
    )


def _rest_days(df: pl.DataFrame) -> pl.Expr:
    """Days since each team's previous game."""
    return (
        (pl.col("game_date").cast(pl.Date) - pl.col("game_date").cast(pl.Date).shift(1))
        .dt.total_days()
        .over("team_id")
        .alias("rest_days")
    )


def _add_team_totals_series(df: pl.DataFrame, score_col: str, opp_score_col: str) -> pl.DataFrame:
    """Add totals-as-a-series features: forecast each team's scoring environment.

    Builds the per-game total points a team is involved in (own + opponent
    score) as a univariate series and forecasts it with Holt, then mirrors the
    opponent's forecast onto the row so the totals model sees both sides:
      team_total_holt / team_total_trend      - this team's environment
      opp_team_total_holt / opp_team_total_trend - the opponent's

    Sport-neutral column names so NBA and NHL totals models share them.
    """
    df = df.with_columns(
        (pl.col(score_col) + pl.col(opp_score_col)).alias("_game_total")
    )
    df = add_totals_series(df, "_game_total", group_col="team_id", date_col="game_date")
    df = df.drop("_game_total")

    opp = df.select(
        pl.col("game_id"),
        pl.col("team_id").alias("opp_team_id"),
        pl.col("team_total_holt").alias("opp_team_total_holt"),
        pl.col("team_total_trend").alias("opp_team_total_trend"),
    )
    return df.join(opp, on=["game_id", "opp_team_id"], how="left")


# ── NBA ───────────────────────────────────────────────────────────────────────

# pace / four-factors derived per game from raw boxscore counts (no new source).
# Possessions ~ FGA + 0.44*FTA - OREB + TOV; the four factors (eFG%, TOV%, FT
# rate, plus offensive rating) are far more predictive than raw points because
# they are pace-adjusted and opponent-comparable.
NBA_TEAM_ADV = ["pace", "off_rating", "efg", "tov_pct", "ft_rate"]


def _nba_advanced_exprs() -> list[pl.Expr]:
    poss = (pl.col("fga") + 0.44 * pl.col("fta") - pl.col("oreb") + pl.col("tov"))
    safe_poss = pl.when(poss > 0).then(poss).otherwise(None)
    safe_fga = pl.when(pl.col("fga") > 0).then(pl.col("fga")).otherwise(None)
    return [
        poss.alias("pace"),
        (pl.col("pts") / safe_poss * 100).alias("off_rating"),
        ((pl.col("fgm") + 0.5 * pl.col("fg3m")) / safe_fga).alias("efg"),
        (pl.col("tov") / pl.when((pl.col("fga") + 0.44 * pl.col("fta") + pl.col("tov")) > 0)
                            .then(pl.col("fga") + 0.44 * pl.col("fta") + pl.col("tov"))
                            .otherwise(None)).alias("tov_pct"),
        (pl.col("fta") / safe_fga).alias("ft_rate"),
    ]


NBA_TEAM_STATS = ["pts", "fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov", "stl", "blk", "plus_minus"]
NBA_PLAYER_STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "fg_pct", "fg3_pct", "ft_pct", "plus_minus"]

# stats that also get a Holt one-step forecast + trend (trajectory signal) on top
# of the EWMA momentum feature. Kept to the headline scoring/efficiency stats so
# the feature set stays compact.
NBA_TEAM_TREND_STATS = ["pts", "plus_minus"]
NBA_PLAYER_TREND_STATS = ["pts", "reb", "ast"]
# only the goals-for trajectory (the goals-against side is captured by the
# totals series); including "opp_goals" here would collide with the opp_ mirror.
NHL_TEAM_TREND_STATS = ["goals"]
NHL_PLAYER_TREND_STATS = ["goals", "points", "shots"]

# half-lives (in games) for the EWMA recency weighting
_TEAM_EWM_HL = 3.0
_PLAYER_EWM_HL = 4.0


def build_nba_game_features(db_path: str) -> pl.DataFrame:
    """Build team-level features per game for NBA moneyline/spread/totals models.

    Returns one row per team per game with:
    - 5 and 10-game rolling averages for key stats
    - rest days, home/away flag, win streak
    - target columns: wl (W/L), pts (actual), opp_pts (for totals)
    """
    df = _read(db_path, "SELECT * FROM nba_team_games ORDER BY game_date")

    if df.is_empty():
        raise ValueError("nba_team_games is empty. Run fetch_seasons() first.")

    df = df.with_columns([
        pl.col("game_date").cast(pl.Date),
        pl.col("wl").fill_null("L").eq("W").cast(pl.Int8).alias("win"),
        pl.col("matchup").fill_null("").str.contains(" vs. ").cast(pl.Int8).alias("is_home"),
    ])

    # pace / four-factors derived per game (added before rolling so they roll too)
    df = df.with_columns(_nba_advanced_exprs())

    # rolling stats (5 and 10 game) + EWMA momentum (recency-weighted)
    _nba_roll_stats = NBA_TEAM_STATS + NBA_TEAM_ADV
    roll_exprs = [
        _rolling_mean(df, col, w)
        for col in _nba_roll_stats
        for w in (5, 10)
    ] + [
        ewm_expr(col, "team_id", _TEAM_EWM_HL) for col in _nba_roll_stats
    ]

    # rest days
    rest_expr = _rest_days(df)

    # win streak: cumulative wins, reset on loss
    streak_expr = (
        pl.col("win")
        .shift(1)
        .rolling_sum(window_size=5, min_samples=1)
        .over("team_id")
        .alias("win_streak_5")
    )

    df = df.sort(["team_id", "game_date"]).with_columns(
        roll_exprs + [rest_expr, streak_expr]
    )

    # Holt one-step forecast + trend for the headline stats (trajectory signal)
    df = add_holt_features(df, NBA_TEAM_TREND_STATS, "team_id", "game_date")

    # join opponent stats onto each row so we can compute matchup features
    trend_cols = [f"{c}_{k}" for c in NBA_TEAM_TREND_STATS for k in ("holt", "trend")]
    opp = df.select(
        pl.col("game_id"),
        pl.col("team_id").alias("opp_team_id"),
        pl.col("pts").alias("opp_pts"),
        *[pl.col(f"{c}_roll5").alias(f"opp_{c}_roll5") for c in _nba_roll_stats],
        *[pl.col(f"{c}_roll10").alias(f"opp_{c}_roll10") for c in _nba_roll_stats],
        *[pl.col(f"{c}_ewm").alias(f"opp_{c}_ewm") for c in _nba_roll_stats],
        *[pl.col(c).alias(f"opp_{c}") for c in trend_cols],
    )

    df = df.join(opp, on="game_id", how="left").filter(
        pl.col("team_id") != pl.col("opp_team_id")
    )

    # explicit opponent-adjusted matchup features
    df = df.with_columns([
        ((pl.col("pace_roll10") + pl.col("opp_pace_roll10")) / 2).alias("pace_matchup"),
        (pl.col("off_rating_roll10") - pl.col("opp_off_rating_roll10")).alias("off_def_edge"),
    ])

    return _add_team_totals_series(df, "pts", "opp_pts")


def build_nba_player_features(db_path: str) -> pl.DataFrame:
    """Build player-level features per game for NBA props models.

    Returns one row per player per game with 5 and 10-game rolling averages,
    rest days, home/away, and actual stat targets.
    """
    df = _read(db_path, "SELECT * FROM nba_player_games ORDER BY game_date")

    df = df.with_columns([
        pl.col("game_date").str.slice(0, 10).str.to_date("%Y-%m-%d"),
        pl.col("matchup").str.contains(" vs. ").cast(pl.Int8).alias("is_home"),
        # parse minutes played (format: "MM:SS")
        pl.col("min").str.split(":").list.first().cast(pl.Float32).alias("minutes"),
    ])

    roll_exprs = [
        _rolling_mean(df, col, w, group_col="player_id")
        for col in NBA_PLAYER_STATS
        for w in (5, 10)
    ] + [
        ewm_expr(col, "player_id", _PLAYER_EWM_HL) for col in NBA_PLAYER_STATS
    ]

    rest_expr = (
        (pl.col("game_date").cast(pl.Date) - pl.col("game_date").cast(pl.Date).shift(1))
        .dt.total_days()
        .over("player_id")
        .alias("rest_days")
    )

    minutes_roll_expr = (
        pl.col("minutes")
        .shift(1)
        .rolling_mean(window_size=5, min_samples=1)
        .over("player_id")
        .alias("minutes_roll5")
    )

    df = df.sort(["player_id", "game_date"]).with_columns(
        roll_exprs + [rest_expr, minutes_roll_expr]
    )

    return add_holt_features(df, NBA_PLAYER_TREND_STATS, "player_id", "game_date")


# ── NHL ───────────────────────────────────────────────────────────────────────

NHL_TEAM_STATS = ["goals", "opp_goals", "shots", "opp_shots"]
NHL_PLAYER_STATS = ["goals", "assists", "points", "shots", "hits", "blocked_shots", "plus_minus"]


def build_nhl_game_features(db_path: str) -> pl.DataFrame:
    """Build team-level features per game for NHL moneyline/spread/totals models."""
    df = _read(db_path, "SELECT * FROM nhl_team_games ORDER BY game_date")

    df = df.with_columns([
        pl.col("game_date").cast(pl.Date),
        pl.col("wl").eq("W").cast(pl.Int8).alias("win"),
        pl.col("home_away").eq("H").cast(pl.Int8).alias("is_home"),
    ])

    roll_exprs = [
        _rolling_mean(df, col, w)
        for col in NHL_TEAM_STATS
        for w in (5, 10)
    ] + [
        ewm_expr(col, "team_id", _TEAM_EWM_HL) for col in NHL_TEAM_STATS
    ]

    rest_expr = _rest_days(df)

    streak_expr = (
        pl.col("win")
        .shift(1)
        .rolling_sum(window_size=5, min_samples=1)
        .over("team_id")
        .alias("win_streak_5")
    )

    df = df.sort(["team_id", "game_date"]).with_columns(
        roll_exprs + [rest_expr, streak_expr]
    )

    df = add_holt_features(df, NHL_TEAM_TREND_STATS, "team_id", "game_date")

    trend_cols = [f"{c}_{k}" for c in NHL_TEAM_TREND_STATS for k in ("holt", "trend")]
    opp = df.select(
        pl.col("game_id"),
        pl.col("team_id").alias("opp_team_id"),
        pl.col("goals").alias("opp_goals_actual"),
        *[pl.col(f"{c}_roll5").alias(f"opp_{c}_roll5") for c in NHL_TEAM_STATS],
        *[pl.col(f"{c}_roll10").alias(f"opp_{c}_roll10") for c in NHL_TEAM_STATS],
        *[pl.col(f"{c}_ewm").alias(f"opp_{c}_ewm") for c in NHL_TEAM_STATS],
        *[pl.col(c).alias(f"opp_{c}") for c in trend_cols],
    )

    df = df.join(opp, on="game_id", how="left").filter(
        pl.col("team_id") != pl.col("opp_team_id")
    )

    # NHL carries its own goals-against per row, so the totals series uses them directly
    df = _add_team_totals_series(df, "goals", "opp_goals")
    # starting-goalie form (no-op until nhl_goalie_games is populated)
    return add_goalie_features(df, db_path)


def build_nhl_player_features(db_path: str) -> pl.DataFrame:
    """Build player-level features per game for NHL props models."""
    df = _read(db_path, "SELECT * FROM nhl_player_games ORDER BY game_date")

    df = df.with_columns(pl.col("game_date").cast(pl.Date))

    roll_exprs = [
        _rolling_mean(df, col, w, group_col="player_id")
        for col in NHL_PLAYER_STATS
        for w in (5, 10)
    ] + [
        ewm_expr(col, "player_id", _PLAYER_EWM_HL) for col in NHL_PLAYER_STATS
    ]

    rest_expr = (
        (pl.col("game_date").cast(pl.Date) - pl.col("game_date").cast(pl.Date).shift(1))
        .dt.total_days()
        .over("player_id")
        .alias("rest_days")
    )

    df = df.sort(["player_id", "game_date"]).with_columns(
        roll_exprs + [rest_expr]
    )

    return add_holt_features(df, NHL_PLAYER_TREND_STATS, "player_id", "game_date")


# ── NHL goalie features ────────────────────────────────────────────────────────

_GOALIE_COLS = ["goalie_save_pct_roll", "goalie_ga_roll"]


def add_goalie_features(df: pl.DataFrame, db_path: str) -> pl.DataFrame:
    """Attach the starting goalie's pre-game form to each NHL team-game row.

    Goalie identity is the single biggest swing factor in a hockey game. For each
    team-game we take the starter (the goalie flagged `starter`, else max TOI),
    look up that goalie's leakage-safe rolling save% and goals-against (strictly
    prior games only), and join it on plus an opponent mirror.

    No-op (null columns) until `nhl_goalie_games` is populated — run
    `python run.py --ingest-history --sport nhl` to backfill goalie boxscores.
    """
    new_cols = _GOALIE_COLS + [f"opp_{c}" for c in _GOALIE_COLS]
    try:
        g = _read(db_path, "SELECT * FROM nhl_goalie_games")
    except Exception:
        g = pl.DataFrame()
    if g.is_empty():
        return df.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in new_cols])

    g = g.with_columns([
        pl.col("game_date").cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False),
        _parse_min("toi").alias("_toi_min"),
        pl.col("save_pct").cast(pl.Float64, strict=False),
        pl.col("goals_against").cast(pl.Float64, strict=False),
    ])

    # one starter per (game_id, team_id): the flagged starter, else most TOI
    starters = (
        g.sort(["game_id", "team_id", "starter", "_toi_min"], descending=[False, False, True, True])
         .group_by(["game_id", "team_id"], maintain_order=True).first()
    )

    # leakage-safe rolling form per goalie (strictly prior games)
    starters = starters.sort(["player_id", "game_date"]).with_columns([
        pl.col("save_pct").shift(1).rolling_mean(window_size=10, min_samples=1)
          .over("player_id").alias("goalie_save_pct_roll"),
        pl.col("goals_against").shift(1).rolling_mean(window_size=10, min_samples=1)
          .over("player_id").alias("goalie_ga_roll"),
    ])

    team_g = starters.select(["game_id", "team_id"] + _GOALIE_COLS)
    df = df.join(team_g, on=["game_id", "team_id"], how="left")
    opp_g = team_g.rename({"team_id": "opp_team_id", **{c: f"opp_{c}" for c in _GOALIE_COLS}})
    return df.join(opp_g, on=["game_id", "opp_team_id"], how="left")


# ── absence / WOWY backbone ────────────────────────────────────────────────────
#
# "Who actually missed which game" is inferred directly from the boxscores: a
# player on a team's seasonal roster with no row in the player_games table for a
# game did not play (injured / scratched / rested). From that single backbone we
# derive both the team-level WOWY features (how the team does without its key
# players) and the teammate-level features (how a player's role changes when a
# key teammate is out). Everything is leakage-safe: per-player with/without
# margins use only games strictly prior to the current one.

# rolling-minutes (NBA) / TOI-minutes (NHL) above which an absent player is "key"
_KEY_MIN_THRESHOLD = {"nba": 24.0, "nhl": 15.0}


def _parse_min(col: str) -> pl.Expr:
    """Parse a 'MM:SS' minutes / TOI string to float minutes."""
    return pl.col(col).str.split(":").list.first().cast(pl.Float32, strict=False)


def _wowy_long(db_path: str, sport: str) -> pl.DataFrame:
    """One row per (game_id, team_id, player_id) the seasonal roster *could* have
    fielded, flagged `is_out` (no boxscore row = absent), with a leakage-safe
    importance score (`min_roll10`, prior rolling minutes) and the team's
    per-player with/without margin (`wowy_delta`). Backbone for all WOWY features.

    Returns an empty frame if the underlying tables are missing/empty.
    """
    if sport == "nba":
        p_table, t_table = "nba_player_games", "nba_team_games"
        season_col, min_col = "season_year", "min"
        team_cols = "plus_minus"
    else:
        p_table, t_table = "nhl_player_games", "nhl_team_games"
        season_col, min_col = "season", "toi"
        team_cols = "goals, opp_goals"

    conn = sqlite3.connect(db_path)
    try:
        players = pl.read_database(
            f"SELECT game_id, team_id, player_id, game_date, "
            f"{season_col} AS season, {min_col} AS min_str FROM {p_table}", conn)
        teams = pl.read_database(
            f"SELECT game_id, team_id, {team_cols} FROM {t_table}", conn)
    except Exception:
        conn.close()
        return pl.DataFrame()
    conn.close()

    if players.is_empty() or teams.is_empty():
        return pl.DataFrame()

    players = players.with_columns([
        pl.col("game_date").cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False),
        _parse_min("min_str").alias("minutes"),
    ])
    margin = (pl.col("plus_minus") if sport == "nba"
              else (pl.col("goals") - pl.col("opp_goals")))
    teams = teams.with_columns(margin.cast(pl.Float32).fill_null(0.0).alias("team_margin")) \
                 .select(["game_id", "team_id", "team_margin"])

    # expected roster-games: every seasonal roster player x every team game
    roster = players.select(["season", "team_id", "player_id"]).unique()
    team_games = players.select(["season", "team_id", "game_id", "game_date"]).unique()
    expected = roster.join(team_games, on=["season", "team_id"], how="inner")

    played_keys = (players.select(["game_id", "team_id", "player_id"]).unique()
                   .with_columns(pl.lit(1, dtype=pl.Int8).alias("_present")))
    long = (expected.join(played_keys, on=["game_id", "team_id", "player_id"], how="left")
            .with_columns(pl.col("_present").is_null().cast(pl.Int8).alias("is_out"))
            .drop("_present")
            .join(teams, on=["game_id", "team_id"], how="left")
            .with_columns(pl.col("team_margin").fill_null(0.0)))

    # prior rolling minutes per player -> importance, attached as-of by date
    imp = (players.sort(["player_id", "game_date"]).with_columns(
        pl.col("minutes").shift(1).rolling_mean(window_size=10, min_samples=1)
        .over("player_id").alias("min_roll10")
    ).select(["player_id", "game_date", "min_roll10"]).sort("game_date"))
    long = (long.sort("game_date")
            .join_asof(imp, on="game_date", by="player_id", strategy="backward")
            .with_columns(pl.col("min_roll10").fill_null(0.0)))

    # per-(team, player) with/without margin using strictly prior games
    # (prior = cumulative including current minus current value)
    long = long.sort(["team_id", "player_id", "game_date"]).with_columns(
        pl.col("is_out").cast(pl.Float32).alias("_out")
    )
    g = ["team_id", "player_id"]
    long = long.with_columns([
        (pl.col("_out").cum_sum().over(g) - pl.col("_out")).alias("_n_out"),
        ((1 - pl.col("_out")).cum_sum().over(g) - (1 - pl.col("_out"))).alias("_n_in"),
        ((pl.col("team_margin") * pl.col("_out")).cum_sum().over(g)
         - pl.col("team_margin") * pl.col("_out")).alias("_s_out"),
        ((pl.col("team_margin") * (1 - pl.col("_out"))).cum_sum().over(g)
         - pl.col("team_margin") * (1 - pl.col("_out"))).alias("_s_in"),
    ])
    long = long.with_columns(
        (
            (pl.col("_s_out") / pl.when(pl.col("_n_out") > 0).then(pl.col("_n_out")).otherwise(None))
            - (pl.col("_s_in") / pl.when(pl.col("_n_in") > 0).then(pl.col("_n_in")).otherwise(None))
        ).fill_null(0.0).alias("wowy_delta")
    ).with_columns(
        (pl.col("min_roll10") >= _KEY_MIN_THRESHOLD[sport]).cast(pl.Int8).alias("is_key")
    )

    return long.select(["game_id", "team_id", "player_id",
                        "is_out", "min_roll10", "wowy_delta", "is_key"])


def _wowy_team_frame(db_path: str, sport: str) -> pl.DataFrame:
    """Per (game_id, team_id) team-level WOWY aggregates.

    wowy_margin_delta - summed with/without margin swing of the absent players
                        (negative => team historically worse without who is out)
    key_players_out   - count of absent players above the importance threshold
    out_min_share     - share of the team's rolling minutes that is absent
    """
    long = _wowy_long(db_path, sport)
    if long.is_empty():
        return long
    agg = long.group_by(["game_id", "team_id"]).agg([
        (pl.col("wowy_delta") * pl.col("is_out")).sum().alias("wowy_margin_delta"),
        (pl.col("is_out") * pl.col("is_key")).sum().cast(pl.Int32).alias("key_players_out"),
        (pl.col("min_roll10") * pl.col("is_out")).sum().alias("_out_min"),
        pl.col("min_roll10").sum().alias("_tot_min"),
    ])
    return agg.with_columns(
        (pl.col("_out_min") / pl.when(pl.col("_tot_min") > 0).then(pl.col("_tot_min")).otherwise(None))
        .fill_null(0.0).alias("out_min_share")
    ).select(["game_id", "team_id", "wowy_margin_delta", "key_players_out", "out_min_share"])


# ── injury features ──────────────────────────────────────────────────────────

def add_injury_features(df: pl.DataFrame, sport: str, db_path: str) -> pl.DataFrame:
    """Join injury impact features onto a team game feature DataFrame.

    For historical training rows: derives injury proxy from minutes played
    (players with minutes far below their rolling average were likely limited).

    For live prediction rows (today's games): reads the current ESPN injury
    report and computes team-level impact scores.

    Adds columns:
      injured_pts_lost      - rolling avg pts of players who didn't play / were limited
      injured_pts_lost_opp  - same for the opponent
      star_out              - 1 if a top-3 minutes player on the team missed the game
      star_out_opp          - same for the opponent
      questionable_count    - number of day-to-day players on the team (live only)
    """
    conn = sqlite3.connect(db_path)

    # ── historical proxy (works on past games) ────────────────────────────────
    if sport == "nba":
        player_table = "nba_player_games"
        pts_col = "pts"
        min_col = "min"
    else:
        player_table = "nhl_player_games"
        pts_col = "goals"
        min_col = "toi"

    try:
        players = pl.read_database(
            f"SELECT game_id, team_id, player_id, {pts_col}, {min_col} FROM {player_table}",
            conn
        )
    except Exception:
        conn.close()
        return df

    conn.close()

    if players.is_empty():
        return df

    # parse minutes to float
    players = players.with_columns(
        pl.col(min_col).str.split(":").list.first().cast(pl.Float32, strict=False).alias("minutes_played")
    )

    # rolling average minutes per player (shift to avoid leakage)
    players = players.sort(["player_id", "game_id"]).with_columns(
        pl.col("minutes_played")
        .shift(1)
        .rolling_mean(window_size=10, min_samples=3)
        .over("player_id")
        .alias("minutes_roll10")
    )

    # flag players who played <40% of their rolling average (injured/DNP proxy)
    players = players.with_columns(
        (
            (pl.col("minutes_played") < pl.col("minutes_roll10") * 0.4) |
            pl.col("minutes_played").is_null()
        ).cast(pl.Int8).alias("was_limited")
    )

    # identify top-3 minutes players per team per game (stars)
    players = players.with_columns(
        pl.col("minutes_roll10")
        .rank(method="ordinal", descending=True)
        .over(["game_id", "team_id"])
        .alias("minutes_rank")
    )

    # team-level aggregates per game
    team_injury = players.group_by(["game_id", "team_id"]).agg([
        (pl.col("was_limited") * pl.col(pts_col)).sum().alias("injured_pts_lost"),
        (
            (pl.col("was_limited") == 1) & (pl.col("minutes_rank") <= 3)
        ).any().cast(pl.Int8).alias("star_out"),
    ])

    # join home team injury features
    df = df.join(team_injury, on=["game_id", "team_id"], how="left")

    # join opponent injury features
    opp_injury = team_injury.rename({
        "team_id": "opp_team_id",
        "injured_pts_lost": "injured_pts_lost_opp",
        "star_out": "star_out_opp",
    })
    df = df.join(opp_injury, on=["game_id", "opp_team_id"], how="left")

    # ── with-or-without-you (WOWY) team features ──────────────────────────────
    wowy_cols = ["wowy_margin_delta", "key_players_out", "out_min_share"]
    wowy = _wowy_team_frame(db_path, sport)
    if not wowy.is_empty():
        df = df.join(wowy, on=["game_id", "team_id"], how="left")
        opp_wowy = wowy.rename({
            "team_id": "opp_team_id",
            **{c: f"{c}_opp" for c in wowy_cols},
        })
        df = df.join(opp_wowy, on=["game_id", "opp_team_id"], how="left")
    else:
        df = df.with_columns(
            [pl.lit(0.0).alias(c) for c in wowy_cols]
            + [pl.lit(0.0).alias(f"{c}_opp") for c in wowy_cols]
        )

    # ── live injury report (today's games only) ───────────────────────────────
    # Add questionable_count from the ESPN snapshot if available
    try:
        live_conn = sqlite3.connect(db_path)
        latest = live_conn.execute(
            "SELECT MAX(fetched_at) FROM injury_reports WHERE sport = ?", (sport,)
        ).fetchone()[0]

        if latest:
            live = pl.read_database(
                f"SELECT team_name, status FROM injury_reports WHERE sport = '{sport}' AND fetched_at = '{latest}'",
                live_conn
            )
            dtd = (
                live.filter(pl.col("status") == "Day-To-Day")
                .group_by("team_name")
                .agg(pl.len().alias("questionable_count"))
            )
            # join on team name — only enriches rows where team_name is present
            if "team_name" in df.columns:
                df = df.join(dtd, on="team_name", how="left")
        live_conn.close()
    except Exception:
        pass

    return df.with_columns([
        pl.col("injured_pts_lost").fill_null(0),
        pl.col("injured_pts_lost_opp").fill_null(0),
        pl.col("star_out").fill_null(0),
        pl.col("star_out_opp").fill_null(0),
        pl.col("wowy_margin_delta").fill_null(0),
        pl.col("wowy_margin_delta_opp").fill_null(0),
        pl.col("key_players_out").fill_null(0),
        pl.col("key_players_out_opp").fill_null(0),
        pl.col("out_min_share").fill_null(0),
        pl.col("out_min_share_opp").fill_null(0),
    ])


def add_teammate_wowy_features(df: pl.DataFrame, sport: str, db_path: str) -> pl.DataFrame:
    """Join teammate-absence features onto a player game feature DataFrame.

    Gives the props models the signal "how does this player's role change when a
    key teammate is out". All inputs are leakage-safe (importance is prior
    rolling minutes from `_wowy_long`).

    Adds columns:
      teammate_out_min_share - share of the team's rolling minutes that is absent
                               (the usage opportunity opened up by who's out)
      lead_teammate_out      - 1 if the team's top-minutes player is out and is
                               not this player
    """
    new_cols = ["teammate_out_min_share", "lead_teammate_out"]
    long = _wowy_long(db_path, sport)
    if long.is_empty() or "player_id" not in df.columns or "team_id" not in df.columns:
        return df.with_columns([pl.lit(0.0).alias(c) for c in new_cols])

    team_agg = long.group_by(["game_id", "team_id"]).agg([
        pl.col("min_roll10").sum().alias("_tot"),
        (pl.col("min_roll10") * pl.col("is_out")).sum().alias("_out"),
    ]).with_columns(
        (pl.col("_out") / pl.when(pl.col("_tot") > 0).then(pl.col("_tot")).otherwise(None))
        .fill_null(0.0).alias("teammate_out_min_share")
    ).select(["game_id", "team_id", "teammate_out_min_share"])

    # the team's lead (top rolling-minutes) player and whether he's out
    lead = (long.sort(["game_id", "team_id", "min_roll10"], descending=[False, False, True])
            .group_by(["game_id", "team_id"], maintain_order=True).first()
            .select(["game_id", "team_id",
                     pl.col("player_id").alias("_lead_id"),
                     pl.col("is_out").alias("_lead_out")]))

    df = (df.join(team_agg, on=["game_id", "team_id"], how="left")
          .join(lead, on=["game_id", "team_id"], how="left"))
    return df.with_columns([
        pl.col("teammate_out_min_share").fill_null(0.0),
        ((pl.col("_lead_out") == 1) & (pl.col("player_id") != pl.col("_lead_id")))
        .cast(pl.Int8).fill_null(0).alias("lead_teammate_out"),
    ]).drop(["_lead_id", "_lead_out"])


# ── market line as a feature ───────────────────────────────────────────────────

_MARKET_COLS = ["mkt_implied_prob", "mkt_total", "mkt_spread"]


def add_market_features(df: pl.DataFrame, sport: str, db_path: str) -> pl.DataFrame:
    """Join the opening market line onto each team-game row as model features.

    The market line is the strongest single pre-game signal there is, so feeding
    the vig-free implied probability (+ total / spread points) to the model lets
    it learn where it should *disagree* with the book rather than re-derive the
    base rate. Joined best-effort by team name + date.

    Mostly null until odds history accumulates (odds_snapshots only retains what
    has been pulled live); LightGBM handles the nulls natively. No-op for frames
    without `team_name` (e.g. NHL, which stores only team ids).
    """
    if "team_name" not in df.columns:
        return df.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in _MARKET_COLS])

    try:
        odds = _read(db_path, f"""
            SELECT home_team, away_team, DATE(commence_time) AS d, market,
                   outcome_name, price, point, MIN(fetched_at) AS first_seen
            FROM odds_snapshots
            WHERE sport = '{sport}' AND bookmaker = 'draftkings'
            GROUP BY home_team, away_team, d, market, outcome_name
        """)
    except Exception:
        odds = pl.DataFrame()
    if odds.is_empty():
        return df.with_columns([pl.lit(None).cast(pl.Float64).alias(c) for c in _MARKET_COLS])

    odds = odds.with_columns(pl.col("d").str.to_date("%Y-%m-%d", strict=False))

    # devigged implied prob per (team, date) from the h2h market
    ml = odds.filter(pl.col("market") == "h2h").with_columns(
        pl.col("price").cast(pl.Float64).alias("am")
    ).with_columns(
        pl.when(pl.col("am") > 0).then(100 / (pl.col("am") + 100))
          .otherwise(pl.col("am").abs() / (pl.col("am").abs() + 100)).alias("raw_prob")
    )
    ml = ml.with_columns(
        (pl.col("raw_prob") / pl.col("raw_prob").sum().over(["home_team", "away_team", "d"]))
        .alias("mkt_implied_prob")
    ).select(
        pl.col("outcome_name").alias("team_name"), pl.col("d"),
        pl.col("mkt_implied_prob"),
    )

    tot = (odds.filter((pl.col("market") == "totals") & (pl.col("outcome_name") == "Over"))
              .select(pl.col("home_team"), pl.col("d"), pl.col("point").alias("mkt_total")))
    spr = (odds.filter(pl.col("market") == "spreads")
              .select(pl.col("outcome_name").alias("team_name"), pl.col("d"),
                      pl.col("point").alias("mkt_spread")))

    out = df.with_columns(pl.col("game_date").cast(pl.Date).alias("_d"))
    out = out.join(ml, left_on=["team_name", "_d"], right_on=["team_name", "d"], how="left")
    out = out.join(spr, left_on=["team_name", "_d"], right_on=["team_name", "d"], how="left")
    out = out.join(tot, left_on=["team_name", "_d"], right_on=["home_team", "d"], how="left")
    return out.drop("_d")


# ── odds join ─────────────────────────────────────────────────────────────────

def add_odds_features(df: pl.DataFrame, sport: str, db_path: str) -> pl.DataFrame:
    """Join the most recent pre-game odds snapshot onto a game feature DataFrame.

    Adds: open_ml_home, open_ml_away, open_spread, open_total (DraftKings lines).
    """
    odds = _read(db_path, """
        SELECT event_id, market, outcome_name, price, point,
               MIN(fetched_at) AS first_seen
        FROM odds_snapshots
        WHERE bookmaker = 'draftkings'
          AND sport = '{sport}'
        GROUP BY event_id, market, outcome_name
    """.replace("{sport}", sport))

    if odds.is_empty():
        return df

    # pivot: one row per event with home_ml, away_ml, spread, total
    ml = odds.filter(pl.col("market") == "h2h").pivot(
        on="outcome_name", index="event_id", values="price"
    )
    spread = odds.filter(pl.col("market") == "spreads").pivot(
        on="outcome_name", index="event_id", values="point"
    )
    total = (
        odds.filter((pl.col("market") == "totals") & (pl.col("outcome_name") == "Over"))
        .select("event_id", pl.col("point").alias("open_total"))
    )

    # The Odds API event_id won't directly match game_id — join on home_team + date
    # This is a best-effort join; unmatched games will have null odds columns
    odds_wide = ml.join(spread, on="event_id", how="left").join(total, on="event_id", how="left")

    return df.join(odds_wide, left_on="game_id", right_on="event_id", how="left")


# ── TENNIS ──────────────────────────────────────────────────────────────────────
#
# Tennis is player-vs-player. Each match in tennis_matches_clean is stored in
# Sackmann winner/loser format. The pipeline:
#   1. compute pre-match Elo (overall + per-surface) and H2H in one ordered pass
#   2. join context (court speed, location/altitude, tournament weather)
#   3. melt each match into 2 player-perspective rows
#   4. compute leakage-safe rolling form / serve / fatigue features per player
#   5. self-join the two perspectives into match rows with diff features (both
#      orderings -> symmetric, no player-order bias)
#   6. persist the wide result as the tennis_features one-big-table

TENNIS_SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

_ROUND_ORD = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7,
    "RR": 4, "BR": 6, "Q1": 0, "Q2": 0, "Q3": 0,
}

# serve/return rate features (per match, then rolled)
_SERVE_RATES = ["ace_rate", "df_rate", "first_in", "first_won", "second_won",
                "bp_saved", "ret_bp_conv"]

# numeric features differenced between player and opponent in the match row
_DIFF_FEATURES = (
    ["elo", "surface_elo", "rank", "rank_points", "age", "ht", "h2h", "surface_h2h",
     "won_roll5", "won_roll10", "won_roll25", "surf_won_roll10",
     "won_ewm", "won_trend",
     "rest_days", "matches_14d", "minutes_14d", "minutes_roll10"]
    + [f"{r}_roll20" for r in _SERVE_RATES]
)

# shared match-context features (not differenced)
_CONTEXT_FEATURES = (
    [f"surface_{s.lower()}" for s in TENNIS_SURFACES]
    + ["speed_index", "altitude_m", "indoor", "best_of", "draw_size", "round_ord",
       "is_grand_slam", "temp_mean", "temp_max", "humidity", "wind_max",
       "precip_sum", "pressure", "temp_speed"]
)

# exported for models/train.py and edge/calculator.py
TENNIS_FEATURE_COLS = [f"{f}_diff" for f in _DIFF_FEATURES] + _CONTEXT_FEATURES


def _compute_elo_h2h(df: pl.DataFrame, k: float = 32.0, base: float = 1500.0) -> pl.DataFrame:
    """Single ordered pass adding pre-match Elo (overall + surface) and H2H counts.

    Adds columns: w_elo, l_elo, w_surf_elo, l_surf_elo (pre-match ratings) and
    w_h2h, l_h2h, w_surf_h2h, l_surf_h2h (prior wins of each player vs the other).
    No leakage: every value reflects state strictly before the match.
    """
    elo: dict = {}
    surf_elo: dict = {}
    h2h: dict = {}        # (a,b) sorted -> {a_id: wins, b_id: wins}
    surf_h2h: dict = {}   # (a,b,surface) -> {...}

    w_elo_c, l_elo_c, w_se_c, l_se_c = [], [], [], []
    w_h_c, l_h_c, w_sh_c, l_sh_c = [], [], [], []

    for r in df.iter_rows(named=True):
        w, l = r["winner_id"], r["loser_id"]
        surf = r["surface"] or "Unknown"

        we = elo.get(w, base)
        le = elo.get(l, base)
        wse = surf_elo.get((w, surf), base)
        lse = surf_elo.get((l, surf), base)
        w_elo_c.append(we); l_elo_c.append(le)
        w_se_c.append(wse); l_se_c.append(lse)

        key = (min(w, l, key=str), max(w, l, key=str))
        rec = h2h.get(key, {})
        skey = (key[0], key[1], surf)
        srec = surf_h2h.get(skey, {})
        w_h_c.append(rec.get(w, 0)); l_h_c.append(rec.get(l, 0))
        w_sh_c.append(srec.get(w, 0)); l_sh_c.append(srec.get(l, 0))

        # ── updates (after recording pre-match state) ──
        exp_w = 1.0 / (1.0 + 10 ** ((le - we) / 400))
        elo[w] = we + k * (1 - exp_w)
        elo[l] = le + k * (0 - (1 - exp_w))
        exp_ws = 1.0 / (1.0 + 10 ** ((lse - wse) / 400))
        surf_elo[(w, surf)] = wse + k * (1 - exp_ws)
        surf_elo[(l, surf)] = lse + k * (0 - (1 - exp_ws))
        rec[w] = rec.get(w, 0) + 1
        h2h[key] = rec
        srec[w] = srec.get(w, 0) + 1
        surf_h2h[skey] = srec

    return df.with_columns([
        pl.Series("w_elo", w_elo_c), pl.Series("l_elo", l_elo_c),
        pl.Series("w_surf_elo", w_se_c), pl.Series("l_surf_elo", l_se_c),
        pl.Series("w_h2h", w_h_c), pl.Series("l_h2h", l_h_c),
        pl.Series("w_surf_h2h", w_sh_c), pl.Series("l_surf_h2h", l_sh_c),
    ])


def _join_tennis_context(df: pl.DataFrame, db_path: str) -> pl.DataFrame:
    """Join court speed (with surface fallback), location/altitude, and weather."""
    speed = _read(db_path, "SELECT tourney_name, surface, speed_index FROM tennis_court_speed")
    named = speed.filter(pl.col("tourney_name") != "_DEFAULT_")
    default = (
        speed.filter(pl.col("tourney_name") == "_DEFAULT_")
        .select("surface", pl.col("speed_index").alias("speed_default"))
    )
    df = df.join(named, on=["tourney_name", "surface"], how="left")
    df = df.join(default, on="surface", how="left")
    df = df.with_columns(pl.col("speed_index").fill_null(pl.col("speed_default"))).drop("speed_default")

    locs = _read(db_path, "SELECT tourney_name, indoor, altitude_m FROM tennis_locations")
    df = df.join(locs, on="tourney_name", how="left")

    try:
        wx = _read(db_path,
                   "SELECT tourney_name, tourney_date, temp_mean, temp_max, humidity, "
                   "wind_max, precip_sum, pressure FROM tennis_weather")
        if not wx.is_empty():
            df = df.join(wx, on=["tourney_name", "tourney_date"], how="left")
    except Exception:
        pass

    # ensure weather columns exist even if no weather table yet
    for c in ["temp_mean", "temp_max", "humidity", "wind_max", "precip_sum", "pressure"]:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

    return df.with_columns([
        pl.col("indoor").fill_null(0),
        pl.col("altitude_m").fill_null(0),
    ])


def _melt_matches_to_players(df: pl.DataFrame) -> pl.DataFrame:
    """Expand each match into 2 player-perspective rows with own + opponent raw stats."""
    key = ["tour", "tourney_id", "match_num", "match_date", "seq_date", "tourney_name",
           "surface", "tourney_level", "round", "round_ord", "best_of", "draw_size",
           "minutes", "total_games", "winner_games", "loser_games", "completed",
           "speed_index", "indoor", "altitude_m",
           "temp_mean", "temp_max", "humidity", "wind_max", "precip_sum", "pressure"]

    def side(player: str, opp: str, won: int, pg: str, og: str) -> pl.DataFrame:
        cols = {
            "player_id": pl.col(f"{player}_id"),
            "player_name": pl.col(f"{player}_name"),
            "opp_id": pl.col(f"{opp}_id"),
            "won": pl.lit(won),
            "player_games": pl.col(pg),
            "opp_games": pl.col(og),
            "rank": pl.col(f"{player}_rank"),
            "rank_points": pl.col(f"{player}_rank_points"),
            "age": pl.col(f"{player}_age"),
            "ht": pl.col(f"{player}_ht"),
            "hand": pl.col(f"{player}_hand"),
            "elo": pl.col(f"{'w' if player=='winner' else 'l'}_elo"),
            "surface_elo": pl.col(f"{'w' if player=='winner' else 'l'}_surf_elo"),
            "h2h": pl.col(f"{'w' if player=='winner' else 'l'}_h2h"),
            "surface_h2h": pl.col(f"{'w' if player=='winner' else 'l'}_surf_h2h"),
        }
        sp = "w" if player == "winner" else "l"
        op = "l" if player == "winner" else "w"
        # raw serve counts for this player and opponent (for rate computation)
        for tag, pfx in (("p", sp), ("o", op)):
            for s in ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "bpSaved", "bpFaced"):
                cols[f"{tag}_{s}"] = pl.col(f"{pfx}_{s}")
        return df.select([pl.col(c) for c in key] + [v.alias(k) for k, v in cols.items()])

    win = side("winner", "loser", 1, "winner_games", "loser_games")
    los = side("loser", "winner", 0, "loser_games", "winner_games")
    return pl.concat([win, los], how="vertical")


def _rate(num: str, den: str) -> pl.Expr:
    return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None)


def build_tennis_player_features(db_path: str) -> pl.DataFrame:
    """Per player-match leakage-safe rolling form / serve / fatigue features."""
    df = _read(db_path, "SELECT * FROM tennis_matches_clean")
    if df.is_empty():
        raise ValueError("tennis_matches_clean is empty. Run ingest + clean first.")

    df = df.with_columns(pl.col("match_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False))

    # Sackmann tourney_date is the tournament START, so every match in a tournament
    # shares one date. Build a per-round synthetic sequence date so ordering and
    # trailing fatigue windows respect round progression (earlier rounds first) and
    # never see a player's later-round (future) matches from the same tournament.
    df = df.with_columns(
        pl.col("round").replace_strict(_ROUND_ORD, default=3).alias("round_ord")
    )
    df = df.with_columns(
        (pl.col("match_date") + pl.duration(days=pl.col("round_ord"))).alias("seq_date")
    )
    df = df.sort(["seq_date", "match_num"])

    df = _compute_elo_h2h(df)
    df = _join_tennis_context(df, db_path)

    pf = _melt_matches_to_players(df)

    # per-match serve/return rates
    pf = pf.with_columns([
        _rate("p_ace", "p_svpt").alias("ace_rate"),
        _rate("p_df", "p_svpt").alias("df_rate"),
        _rate("p_1stIn", "p_svpt").alias("first_in"),
        _rate("p_1stWon", "p_1stIn").alias("first_won"),
        (pl.when(pl.col("p_svpt") - pl.col("p_1stIn") > 0)
           .then(pl.col("p_2ndWon") / (pl.col("p_svpt") - pl.col("p_1stIn")))
           .otherwise(None)).alias("second_won"),
        _rate("p_bpSaved", "p_bpFaced").alias("bp_saved"),
        (pl.when(pl.col("o_bpFaced") > 0)
           .then((pl.col("o_bpFaced") - pl.col("o_bpSaved")) / pl.col("o_bpFaced"))
           .otherwise(None)).alias("ret_bp_conv"),
    ])

    pf = pf.sort(["player_id", "seq_date", "match_num"])

    # rolling form (win%) and surface-specific form
    roll = [
        pl.col("won").shift(1).rolling_mean(window_size=w, min_samples=1)
          .over("player_id").alias(f"won_roll{w}")
        for w in (5, 10, 25)
    ]
    roll.append(
        pl.col("won").shift(1).rolling_mean(window_size=10, min_samples=1)
          .over(["player_id", "surface"]).alias("surf_won_roll10")
    )
    # EWMA recency-weighted form (Holt trend added after the rates below)
    roll.append(ewm_expr("won", "player_id", half_life=5.0))
    # rolling serve rates
    roll += [
        pl.col(r).shift(1).rolling_mean(window_size=20, min_samples=2)
          .over("player_id").alias(f"{r}_roll20")
        for r in _SERVE_RATES
    ]
    # recent match length
    roll.append(
        pl.col("minutes").cast(pl.Float64, strict=False).shift(1)
          .rolling_mean(window_size=10, min_samples=1).over("player_id").alias("minutes_roll10")
    )
    pf = pf.with_columns(roll)

    # rest days (since previous match, by synthetic sequence date)
    pf = pf.with_columns(
        (pl.col("seq_date") - pl.col("seq_date").shift(1)).dt.total_days()
          .over("player_id").alias("rest_days")
    )

    # 14-day fatigue (matches played + minutes on court, excluding the current match).
    # The trailing window over seq_date only sees earlier rounds / prior tournaments,
    # never the player's later-round matches in the same event (no leakage).
    pf = pf.sort(["player_id", "seq_date"])
    fatigue = (
        pf.rolling(index_column="seq_date", period="14d", group_by="player_id")
          .agg([pl.len().alias("_m14"),
                pl.col("minutes").cast(pl.Float64, strict=False).sum().alias("_min14")])
    )
    pf = pf.with_columns([
        (fatigue["_m14"] - 1).alias("matches_14d"),
        (fatigue["_min14"] - pl.col("minutes").cast(pl.Float64, strict=False).fill_null(0)).alias("minutes_14d"),
    ])

    # Holt trend of win form (won_trend); leakage-safe one-step pass per player
    pf = add_holt_features(pf, ["won"], "player_id", "seq_date")
    return pf


def build_tennis_match_features(db_path: str) -> pl.DataFrame:
    """Self-join player perspectives into match rows with diff + context features.

    Produces 2 symmetric rows per match (each player as 'player1'). Targets:
    won (binary), total_games (regression), game_margin (player_games - opp_games).
    """
    pf = build_tennis_player_features(db_path)

    match_key = ["tour", "tourney_id", "match_num"]

    # opponent view: same features suffixed _opp, joined on the opponent's id
    opp_cols = _DIFF_FEATURES + ["player_id"]
    opp = pf.select(match_key + [pl.col(c).alias(f"{c}_oppview") for c in opp_cols])
    opp = opp.rename({"player_id_oppview": "opp_id"})

    df = pf.join(opp, on=match_key + ["opp_id"], how="inner")

    # diff features (player - opponent)
    diff_exprs = [
        (pl.col(f).cast(pl.Float64, strict=False)
         - pl.col(f"{f}_oppview").cast(pl.Float64, strict=False)).alias(f"{f}_diff")
        for f in _DIFF_FEATURES
    ]

    # surface one-hot + context
    ctx_exprs = (
        [(pl.col("surface") == s).cast(pl.Int8).alias(f"surface_{s.lower()}")
         for s in TENNIS_SURFACES]
        + [
            (pl.col("tourney_level") == "G").cast(pl.Int8).alias("is_grand_slam"),
            pl.col("best_of").cast(pl.Float64, strict=False),
            pl.col("draw_size").cast(pl.Float64, strict=False),
            (pl.col("temp_mean").cast(pl.Float64, strict=False)
             * pl.col("speed_index").cast(pl.Float64, strict=False)).alias("temp_speed"),
        ]
    )

    df = df.with_columns(diff_exprs + ctx_exprs)

    # targets
    df = df.with_columns([
        pl.col("won").cast(pl.Int8),
        pl.col("total_games").cast(pl.Float64, strict=False),
        (pl.col("player_games").cast(pl.Float64, strict=False)
         - pl.col("opp_games").cast(pl.Float64, strict=False)).alias("game_margin"),
    ])

    return df


def persist_tennis_features(db_path: str) -> pl.DataFrame:
    """Build the wide one-big-table and write it to the tennis_features SQLite table."""
    import sqlite3
    from ingestion.tennis_clean import _write_table

    df = build_tennis_match_features(db_path)

    id_cols = ["tour", "tourney_id", "match_num", "match_date", "tourney_name",
               "surface", "player_id", "player_name", "opp_id", "completed"]
    target_cols = ["won", "total_games", "game_margin"]
    keep = [c for c in id_cols if c in df.columns] + TENNIS_FEATURE_COLS + target_cols
    out = df.select([c for c in keep if c in df.columns])
    out = out.with_columns(pl.col("match_date").cast(pl.Utf8))

    conn = sqlite3.connect(db_path)
    _write_table(conn, "tennis_features", out)
    conn.close()
    print(f"  tennis_features: {out.height} rows x {out.width} cols written")
    return out


def add_tennis_odds_features(df: pl.DataFrame, db_path: str) -> pl.DataFrame:
    """Join the latest pre-match h2h odds (by player name) onto match feature rows."""
    odds = _read(db_path, """
        SELECT outcome_name, price, MIN(fetched_at) AS first_seen
        FROM odds_snapshots
        WHERE sport = 'tennis' AND market = 'h2h' AND bookmaker = 'draftkings'
        GROUP BY outcome_name
    """)
    if odds.is_empty():
        return df
    odds = odds.select(pl.col("outcome_name").alias("player_name"),
                       pl.col("price").alias("player_ml"))
    return df.join(odds, on="player_name", how="left")
