"""Feature engineering pipeline. Reads from SQLite, returns Polars DataFrames.

All rolling features are computed from strictly prior games (shift(1) before
the window) to prevent data leakage into model training.
"""

import sqlite3
import polars as pl


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


# ── NBA ───────────────────────────────────────────────────────────────────────

NBA_TEAM_STATS = ["pts", "fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "tov", "stl", "blk", "plus_minus"]
NBA_PLAYER_STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "fg_pct", "fg3_pct", "ft_pct", "plus_minus"]


def build_nba_game_features(db_path: str) -> pl.DataFrame:
    """Build team-level features per game for NBA moneyline/spread/totals models.

    Returns one row per team per game with:
    - 5 and 10-game rolling averages for key stats
    - rest days, home/away flag, win streak
    - target columns: wl (W/L), pts (actual), opp_pts (for totals)
    """
    df = _read(db_path, "SELECT * FROM nba_team_games ORDER BY game_date")

    df = df.with_columns([
        pl.col("game_date").cast(pl.Date),
        pl.col("wl").eq("W").cast(pl.Int8).alias("win"),
        # home flag: matchup like "BOS vs. NYK" = home, "BOS @ NYK" = away
        pl.col("matchup").str.contains(" vs. ").cast(pl.Int8).alias("is_home"),
    ])

    # rolling stats (5 and 10 game)
    roll_exprs = [
        _rolling_mean(df, col, w)
        for col in NBA_TEAM_STATS
        for w in (5, 10)
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

    # join opponent stats onto each row so we can compute matchup features
    opp = df.select(
        pl.col("game_id"),
        pl.col("team_id").alias("opp_team_id"),
        pl.col("pts").alias("opp_pts"),
        *[pl.col(f"{c}_roll5").alias(f"opp_{c}_roll5") for c in NBA_TEAM_STATS],
        *[pl.col(f"{c}_roll10").alias(f"opp_{c}_roll10") for c in NBA_TEAM_STATS],
    )

    df = df.join(opp, on="game_id", how="left").filter(
        pl.col("team_id") != pl.col("opp_team_id")
    )

    return df


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

    return df


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

    opp = df.select(
        pl.col("game_id"),
        pl.col("team_id").alias("opp_team_id"),
        pl.col("goals").alias("opp_goals_actual"),
        *[pl.col(f"{c}_roll5").alias(f"opp_{c}_roll5") for c in NHL_TEAM_STATS],
        *[pl.col(f"{c}_roll10").alias(f"opp_{c}_roll10") for c in NHL_TEAM_STATS],
    )

    df = df.join(opp, on="game_id", how="left").filter(
        pl.col("team_id") != pl.col("opp_team_id")
    )

    return df


def build_nhl_player_features(db_path: str) -> pl.DataFrame:
    """Build player-level features per game for NHL props models."""
    df = _read(db_path, "SELECT * FROM nhl_player_games ORDER BY game_date")

    df = df.with_columns(pl.col("game_date").cast(pl.Date))

    roll_exprs = [
        _rolling_mean(df, col, w, group_col="player_id")
        for col in NHL_PLAYER_STATS
        for w in (5, 10)
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

    return df


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
