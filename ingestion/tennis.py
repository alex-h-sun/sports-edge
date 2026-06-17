"""Tennis data ingestion. Writes to SQLite.

Sources (all keyless):
  - Jeff Sackmann tennis_atp / tennis_wta match CSVs (GitHub raw) -> tennis_matches
  - Curated data/court_speed.csv                                  -> tennis_court_speed
  - Curated data/tournament_locations.csv                        -> tennis_locations
  - Open-Meteo Historical Archive API                            -> tennis_weather

The match table holds one row per match in Sackmann's winner/loser format. Live
odds for tennis are handled separately in ingestion/odds.py.
"""

import io
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
import polars as pl

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_RATE_LIMIT = 0.2  # seconds between HTTP requests

_SACKMANN = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv",
}

_OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"

_DATA_DIR = Path(__file__).parent.parent / "data"

# Columns we keep from each Sackmann match CSV (lowercase = stored name).
_MATCH_COLS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num", "winner_id", "winner_seed", "winner_entry",
    "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "winner_rank", "winner_rank_points", "loser_id", "loser_seed", "loser_entry",
    "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "loser_rank", "loser_rank_points", "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
    "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
]


# ── db helpers ─────────────────────────────────────────────────────────────────

def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    cols_sql = ",\n            ".join(f"{c} TEXT" for c in _MATCH_COLS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS tennis_matches (
            tour        TEXT,
            {cols_sql},
            PRIMARY KEY (tour, tourney_id, match_num)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tennis_court_speed (
            tourney_name TEXT,
            surface      TEXT,
            speed_index  REAL,
            cpi_category TEXT,
            PRIMARY KEY (tourney_name, surface)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tennis_locations (
            tourney_name TEXT PRIMARY KEY,
            city         TEXT,
            country      TEXT,
            lat          REAL,
            lon          REAL,
            indoor       INTEGER,
            altitude_m   REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tennis_weather (
            tourney_name TEXT,
            tourney_date TEXT,
            temp_mean    REAL,
            temp_max     REAL,
            temp_min     REAL,
            humidity     REAL,
            wind_max     REAL,
            precip_sum   REAL,
            pressure     REAL,
            PRIMARY KEY (tourney_name, tourney_date)
        )
    """)
    # HTTP validators (ETag / Last-Modified) per Sackmann CSV URL, so a re-run can
    # send a conditional GET and skip re-downloading a file that has not changed.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tennis_fetch_cache (
            url           TEXT PRIMARY KEY,
            etag          TEXT,
            last_modified TEXT,
            fetched_at    TEXT
        )
    """)
    conn.commit()


def _upsert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    collist = ", ".join(cols)
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({collist}) VALUES ({placeholders})", rows
    )
    conn.commit()


# ── conditional-GET cache ────────────────────────────────────────────────────────

def _cache_get(conn: sqlite3.Connection, url: str) -> tuple[str | None, str | None]:
    """Return the stored (etag, last_modified) for a URL, or (None, None) if unseen."""
    row = conn.execute(
        "SELECT etag, last_modified FROM tennis_fetch_cache WHERE url = ?", (url,)
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _cache_put(conn: sqlite3.Connection, url: str,
               etag: str | None, last_modified: str | None) -> None:
    """Record the validators returned with a successful download of `url`."""
    conn.execute(
        "INSERT OR REPLACE INTO tennis_fetch_cache (url, etag, last_modified, fetched_at) "
        "VALUES (?, ?, ?, ?)",
        (url, etag, last_modified,
         datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    conn.commit()


# ── Sackmann matches ───────────────────────────────────────────────────────────

def _fetch_year(tour: str, year: int,
                conn: sqlite3.Connection | None = None) -> tuple[str, list[dict]]:
    """Download and parse one tour-year of Sackmann matches.

    Returns ``(status, rows)`` where status is:
      - ``"ok"``        — fetched fresh content; ``rows`` are the parsed matches.
      - ``"unchanged"`` — the server answered 304 Not Modified (we already have it);
                          ``rows`` is empty and nothing should be re-ingested.
      - ``"missing"``   — 404 (e.g. a future year not yet published); ``rows`` empty.

    When ``conn`` is given, a stored ETag / Last-Modified for this URL is sent as a
    conditional GET so an unchanged file is not re-downloaded, and the validators
    from a fresh response are persisted for next time. Without ``conn`` (or when the
    server returns no validators) it behaves like a plain unconditional fetch.
    """
    url = _SACKMANN[tour].format(year=year)
    headers = dict(_HEADERS)
    if conn is not None:
        etag, last_modified = _cache_get(conn, url)
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 304:
        return ("unchanged", [])
    if resp.status_code == 404:
        return ("missing", [])
    resp.raise_for_status()

    # Read every column as string; we coerce types later in the cleaning stage.
    df = pl.read_csv(
        io.BytesIO(resp.content),
        infer_schema_length=0,  # all-Utf8, robust to mixed/missing values
    )

    present = [c for c in _MATCH_COLS if c in df.columns]
    df = df.select(present)
    # add any missing expected columns as nulls so the schema stays stable
    for c in _MATCH_COLS:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    df = df.select(_MATCH_COLS).with_columns(pl.lit(tour).alias("tour"))

    if conn is not None:
        _cache_put(conn, url, resp.headers.get("ETag"), resp.headers.get("Last-Modified"))

    return ("ok", df.to_dicts())


def _upsert_matches(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Upsert match rows; return how many were brand new (not just replaced)."""
    before = conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0]
    _upsert(conn, "tennis_matches", rows)
    after = conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0]
    return after - before


def fetch_seasons(years: list[int], tours: list[str], db_path: str) -> None:
    """Fetch full match history for the given years and tours (e.g. range(2000, 2027)).

    Each tour-year is fetched with a conditional GET, so re-running only re-downloads
    files that have actually changed (past seasons are immutable -> all skipped).
    """
    conn = _get_conn(db_path)
    _ensure_tables(conn)

    for tour in tours:
        total = new_total = 0
        for year in years:
            try:
                status, rows = _fetch_year(tour, year, conn)
            except Exception as e:
                print(f"  warning: {tour} {year} failed ({e})")
                continue
            if status == "unchanged":
                print(f"  {tour.upper()} {year}: unchanged since last pull, skipped")
                continue
            new = _upsert_matches(conn, rows)
            total += len(rows)
            new_total += new
            print(f"  {tour.upper()} {year}: {len(rows)} matches ({new} new)")
            time.sleep(_RATE_LIMIT)
        print(f"  {tour.upper()} total: {total} matches ({new_total} new)")

    load_court_speed(db_path, conn)
    load_locations(db_path, conn)
    conn.close()


def fetch_recent(db_path: str, tours: list[str] | None = None, year: int | None = None) -> None:
    """Re-pull the current year's CSV (Sackmann appends in-season) for an incremental update.

    Uses a conditional GET keyed on the stored ETag / Last-Modified, so running this
    more than once a day does not re-download an unchanged file: if nothing new has
    been published the server returns 304 and the fetch is skipped. When the file has
    grown, the whole current-year CSV is re-parsed but only the new matches are added.
    """
    tours = tours or ["atp", "wta"]
    year = year or date.today().year

    conn = _get_conn(db_path)
    _ensure_tables(conn)
    for tour in tours:
        try:
            status, rows = _fetch_year(tour, year, conn)
        except Exception as e:
            print(f"  warning: {tour} {year} failed ({e})")
            continue
        if status == "unchanged":
            print(f"  Tennis recent {tour.upper()} {year}: unchanged since last pull, skipped")
            continue
        new = _upsert_matches(conn, rows)
        print(f"  Tennis recent {tour.upper()} {year}: {len(rows)} matches ({new} new)")
        time.sleep(_RATE_LIMIT)

    load_court_speed(db_path, conn)
    load_locations(db_path, conn)
    conn.close()


# ── curated reference tables ────────────────────────────────────────────────────

def load_court_speed(db_path: str, conn: sqlite3.Connection | None = None) -> None:
    """Load data/court_speed.csv into tennis_court_speed."""
    path = _DATA_DIR / "court_speed.csv"
    if not path.exists():
        print(f"  warning: {path} not found, skipping court speed load")
        return
    df = pl.read_csv(path, comment_prefix="#")
    owns = conn is None
    conn = conn or _get_conn(db_path)
    _ensure_tables(conn)
    _upsert(conn, "tennis_court_speed", df.to_dicts())
    print(f"  Court speed: {df.height} rows loaded")
    if owns:
        conn.close()


def load_locations(db_path: str, conn: sqlite3.Connection | None = None) -> None:
    """Load data/tournament_locations.csv into tennis_locations."""
    path = _DATA_DIR / "tournament_locations.csv"
    if not path.exists():
        print(f"  warning: {path} not found, skipping location load")
        return
    df = pl.read_csv(path, comment_prefix="#")
    owns = conn is None
    conn = conn or _get_conn(db_path)
    _ensure_tables(conn)
    _upsert(conn, "tennis_locations", df.to_dicts())
    print(f"  Locations: {df.height} rows loaded")
    if owns:
        conn.close()


# ── weather ─────────────────────────────────────────────────────────────────────

def _yyyymmdd_to_iso(d: str) -> str | None:
    """Sackmann tourney_date is 'YYYYMMDD'. Return ISO 'YYYY-MM-DD' or None."""
    if not d or len(str(d)) != 8:
        return None
    try:
        return datetime.strptime(str(d), "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def _fetch_open_meteo(lat: float, lon: float, start: str, end: str) -> dict | None:
    """Return averaged daily conditions over [start, end] for a location, or None."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
                 "precipitation_sum,wind_speed_10m_max",
        "hourly": "relative_humidity_2m,surface_pressure",
        "timezone": "auto",
    }
    resp = requests.get(_OPEN_METEO, params=params, headers=_HEADERS, timeout=30)
    if resp.status_code != 200:
        return None
    data = resp.json()
    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    def _avg(vals):
        nums = [v for v in (vals or []) if v is not None]
        return sum(nums) / len(nums) if nums else None

    def _sum(vals):
        nums = [v for v in (vals or []) if v is not None]
        return sum(nums) / len(nums) if nums else None  # mean per-day precip

    return {
        "temp_mean": _avg(daily.get("temperature_2m_mean")),
        "temp_max":  _avg(daily.get("temperature_2m_max")),
        "temp_min":  _avg(daily.get("temperature_2m_min")),
        "precip_sum": _sum(daily.get("precipitation_sum")),
        "wind_max":  _avg(daily.get("wind_speed_10m_max")),
        "humidity":  _avg(hourly.get("relative_humidity_2m")),
        "pressure":  _avg(hourly.get("surface_pressure")),
    }


def fetch_weather(db_path: str, fortnight_days: int = 13) -> None:
    """Fetch average tournament-fortnight weather for each (tourney_name, tourney_date).

    Outdoor tournaments only; indoor venues are stored with null weather (treated
    as neutral downstream). Already-cached tournament-dates are skipped, so this is
    cheap to re-run and stays well under Open-Meteo's free 10k/day cap.
    """
    conn = _get_conn(db_path)
    _ensure_tables(conn)

    locs = {
        r[0]: r for r in conn.execute(
            "SELECT tourney_name, lat, lon, indoor FROM tennis_locations"
        ).fetchall()
    }
    cached = {
        (r[0], r[1]) for r in conn.execute(
            "SELECT tourney_name, tourney_date FROM tennis_weather"
        ).fetchall()
    }

    pairs = conn.execute(
        "SELECT DISTINCT tourney_name, tourney_date FROM tennis_matches"
    ).fetchall()

    _W_KEYS = ("temp_mean", "temp_max", "temp_min", "humidity",
               "wind_max", "precip_sum", "pressure")

    def _row(tname, tdate, w=None):
        base = {"tourney_name": tname, "tourney_date": tdate}
        base.update({k: (w or {}).get(k) for k in _W_KEYS})
        return base

    fetched = skipped = 0
    rows: list[dict] = []
    for tname, tdate in pairs:
        iso = _yyyymmdd_to_iso(tdate)
        if iso is None or (tname, tdate) in cached:
            skipped += 1
            continue
        loc = locs.get(tname)
        if loc is None:
            skipped += 1
            continue
        _, lat, lon, indoor = loc
        if indoor or lat is None or lon is None:
            # store a neutral (null-weather) row so we don't re-query indoor venues
            rows.append(_row(tname, tdate))
            continue

        end = (datetime.fromisoformat(iso) + timedelta(days=fortnight_days)).date().isoformat()
        try:
            w = _fetch_open_meteo(lat, lon, iso, end)
        except Exception as e:
            print(f"    warning: weather {tname} {iso} failed ({e})")
            w = None
        if w:
            rows.append(_row(tname, tdate, w))
            fetched += 1
        time.sleep(_RATE_LIMIT)

        if len(rows) >= 50:
            _upsert(conn, "tennis_weather", rows)
            rows = []

    _upsert(conn, "tennis_weather", rows)
    print(f"  Weather: {fetched} tournament-dates fetched, {skipped} skipped (cached/indoor/no-loc)")
    conn.close()
