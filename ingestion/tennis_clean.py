"""Clean raw tennis_matches into tennis_matches_clean.

Responsibilities:
  - parse the free-text `score` into total_games / total_sets / tiebreaks / margin
  - flag non-completed matches (retirements, walkovers, defaults)
  - coerce stat / rank / bio columns to numeric, impute missing rank, clip outliers
  - normalize tourney_name for joins to court speed / locations / weather
  - de-duplicate and order chronologically

The cleaned table is the single source the feature pipeline reads from.
"""

import re
import sqlite3

import polars as pl

UNRANKED = 9999  # sentinel rank for unranked / missing players

_INCOMPLETE = ("RET", "W/O", "DEF", "WALKOVER", "ABD", "ABN", "DEFAULT")

_INT_COLS = [
    "draw_size", "match_num", "best_of", "minutes",
    "winner_id", "winner_ht", "winner_rank", "winner_rank_points",
    "loser_id", "loser_ht", "loser_rank", "loser_rank_points",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
]
_FLOAT_COLS = ["winner_age", "loser_age"]

_SET_RE = re.compile(r"^(\d+)-(\d+)")


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _parse_score(score: str | None) -> dict:
    """Parse a Sackmann score string into game/set tallies and a completed flag.

    Returns total_games, total_sets, n_tiebreaks, winner_games, loser_games,
    game_margin (winner - loser), completed (1/0).
    """
    blank = {
        "total_games": None, "total_sets": None, "n_tiebreaks": None,
        "winner_games": None, "loser_games": None, "game_margin": None,
        "completed": 0,
    }
    if not score or not str(score).strip():
        return blank

    s = str(score).strip()
    upper = s.upper()
    completed = 0 if any(tok in upper for tok in _INCOMPLETE) else 1

    w_games = l_games = sets = tiebreaks = 0
    for token in s.split():
        m = _SET_RE.match(token)
        if not m:
            continue  # RET / W/O / score-less tokens
        a, b = int(m.group(1)), int(m.group(2))
        w_games += a
        l_games += b
        sets += 1
        if "(" in token:
            tiebreaks += 1

    if sets == 0:
        return {**blank, "completed": completed}

    return {
        "total_games": w_games + l_games,
        "total_sets": sets,
        "n_tiebreaks": tiebreaks,
        "winner_games": w_games,
        "loser_games": l_games,
        "game_margin": w_games - l_games,
        "completed": completed,
    }


def _normalize_name(col: str) -> pl.Expr:
    """Trim, collapse whitespace, and title-case a tournament name for joins."""
    return (
        pl.col(col)
        .fill_null("")
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .alias(col)
    )


_AFFINITY = {
    pl.Int64: "INTEGER", pl.Int32: "INTEGER", pl.Int16: "INTEGER", pl.Int8: "INTEGER",
    pl.Float64: "REAL", pl.Float32: "REAL",
}


def _write_table(conn: sqlite3.Connection, name: str, df: pl.DataFrame) -> None:
    """Drop and recreate `name` from the DataFrame schema, then bulk insert."""
    cols_sql = ", ".join(
        f"{c} {_AFFINITY.get(dt, 'TEXT')}" for c, dt in zip(df.columns, df.dtypes)
    )
    conn.execute(f"DROP TABLE IF EXISTS {name}")
    conn.execute(f"CREATE TABLE {name} ({cols_sql})")
    rows = df.to_dicts()
    if rows:
        placeholders = ", ".join(f":{c}" for c in df.columns)
        conn.executemany(
            f"INSERT INTO {name} ({', '.join(df.columns)}) VALUES ({placeholders})", rows
        )
    conn.commit()


def clean(db_path: str) -> None:
    """Build tennis_matches_clean from tennis_matches. Prints a cleaning report."""
    conn = _get_conn(db_path)
    df = pl.read_database("SELECT * FROM tennis_matches", conn)

    n_in = df.height
    if n_in == 0:
        print("  tennis_matches is empty — run fetch_seasons() first")
        conn.close()
        return

    # numeric coercion
    df = df.with_columns(
        [pl.col(c).cast(pl.Int64, strict=False) for c in _INT_COLS if c in df.columns]
        + [pl.col(c).cast(pl.Float64, strict=False) for c in _FLOAT_COLS if c in df.columns]
    )

    # tourney_date -> Date; drop rows we can't date (can't order them safely)
    df = df.with_columns(
        pl.col("tourney_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("match_date")
    )

    # normalize join key
    df = df.with_columns(_normalize_name("tourney_name"))

    # impute rank; clip serve outliers (svpt must be > 0 to be meaningful)
    for side in ("winner", "loser"):
        rk = f"{side}_rank"
        if rk in df.columns:
            df = df.with_columns(pl.col(rk).fill_null(UNRANKED))
    for pfx in ("w", "l"):
        sv = f"{pfx}_svpt"
        if sv in df.columns:
            bad = pl.col(sv) <= 0
            stat_cols = [f"{pfx}_{s}" for s in
                         ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon",
                          "SvGms", "bpSaved", "bpFaced")]
            df = df.with_columns(
                [pl.when(bad).then(None).otherwise(pl.col(c)).alias(c)
                 for c in stat_cols if c in df.columns]
            )

    # parse scores (row-wise) and attach derived columns
    parsed = pl.DataFrame([_parse_score(s) for s in df["score"].to_list()])
    df = pl.concat([df, parsed], how="horizontal")

    # de-duplicate and order chronologically
    df = df.unique(subset=["tour", "tourney_id", "match_num"], keep="first")
    df = df.drop_nulls(subset=["match_date"]).sort("match_date")

    n_out = df.height
    n_incomplete = int((df["completed"] == 0).sum())
    n_unparsed = int(df["total_games"].is_null().sum())

    _write_table(conn, "tennis_matches_clean", df)
    conn.close()

    print(f"  Cleaning report: {n_in} raw -> {n_out} clean "
          f"({n_in - n_out} dropped: undated/duplicate)")
    print(f"    incomplete (RET/W/O/DEF) flagged: {n_incomplete}")
    print(f"    unparseable scores (null game tallies): {n_unparsed}")
