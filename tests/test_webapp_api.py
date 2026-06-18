"""Tests for the web API (webapp/, Starlette).

Hermetic tests run against a tiny seeded SQLite DB and cover the auth gate, meta,
quota, injuries, bankroll, manual options/edge error paths, edges-shape, CSV export,
and the snapshot admin reload. A DB-gated integration test exercises the real
feature/model serving path against the production sports.db + artifacts when present
(skipped otherwise, like tests/test_paper_sim.py).

Env is configured *before* importing webapp because settings reads it at import time.
"""

import csv
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

# ── configure env, then seed, then import the app ───────────────────────────────
_TMP = tempfile.mkdtemp(prefix="webapp-test-")
_DB = os.path.join(_TMP, "seed.db")
_LEDGER = os.path.join(_TMP, "ledger.csv")
os.environ.update(
    APP_PASSWORD="testpw",
    SESSION_SECRET="testsecret",
    DB_PATH=_DB,
    LEDGER_PATH=_LEDGER,
    ARTIFACTS_DIR=os.path.join(_TMP, "artifacts"),
    SNAPSHOT_URL="",
)


def _seed_db(path: str) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE injury_reports (
            sport TEXT, team_name TEXT, player_name TEXT, position TEXT, status TEXT,
            injury_type TEXT, short_comment TEXT, reported_at TEXT, fetched_at TEXT);
        CREATE TABLE odds_requests_remaining (remaining INT, used INT, fetched_at TEXT);
        """
    )
    con.execute("INSERT INTO injury_reports VALUES "
                "('nba','Lakers','LeBron James','SF','Out','knee','','','2026-06-17T00:00:00Z')")
    con.execute("INSERT INTO injury_reports VALUES "
                "('nhl','Bruins','David Pastrnak','RW','Day-To-Day','flu','','','2026-06-17T00:00:00Z')")
    con.execute("INSERT INTO odds_requests_remaining VALUES (418, 82, '2026-06-17T05:00:00Z')")
    con.commit()
    con.close()


def _seed_ledger(path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bet_id", "placed_date", "game_date", "sport", "market", "game",
                    "selection", "odds", "odds_decimal", "model_prob", "edge", "stake",
                    "status", "profit", "balance"])
        w.writerow(["a1", "2026-06-01", "2026-06-02", "NBA", "Moneyline", "X @ Y",
                    "X ML", "-150", "1.667", "0.7", "0.08", "10", "won", "6.67", "1006.67"])
        w.writerow(["a2", "2026-06-03", "2026-06-04", "NHL", "Moneyline", "P @ Q",
                    "Q ML", "120", "2.2", "0.55", "0.06", "10", "lost", "-10.0", "996.67"])


_seed_db(_DB)
_seed_ledger(_LEDGER)

from starlette.testclient import TestClient  # noqa: E402

from webapp.main import app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_client():
    c = TestClient(app)
    assert c.post("/api/auth/login", json={"password": "testpw"}).status_code == 200
    return c


# ── auth gate ────────────────────────────────────────────────────────────────────

def test_health_is_open(client):
    assert client.get("/api/health").json() == {"ok": True}


def test_meta_requires_auth(client):
    r = client.get("/api/meta")
    assert r.status_code == 401 and r.json()["auth_required"] is True


def test_login_wrong_then_right_then_logout(client):
    assert client.post("/api/auth/login", json={"password": "nope"}).status_code == 401
    assert client.post("/api/auth/login", json={"password": "testpw"}).status_code == 200
    assert client.get("/api/meta").status_code == 200
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/meta").status_code == 401


def test_meta_payload(auth_client):
    m = auth_client.get("/api/meta").json()
    assert m["db_label"] == "seed.db"
    assert m["auth_required"] is True
    assert set(m["sports"]) == {"nba", "nhl", "tennis"}
    assert m["snapshot_ts"]  # mtime of the seeded DB


# ── read endpoints over seeded data ─────────────────────────────────────────────

def test_quota(auth_client):
    q = auth_client.get("/api/quota").json()["quota"]
    assert q == {"remaining": 418, "used": 82, "fetched_at": "2026-06-17T05:00:00Z"}


def test_injuries(auth_client):
    j = auth_client.get("/api/injuries").json()
    assert [p["player_name"] for p in j["nba"]] == ["LeBron James"]
    assert [p["player_name"] for p in j["nhl"]] == ["David Pastrnak"]


def test_bankroll_summary(auth_client):
    b = auth_client.get("/api/bankroll").json()
    assert b["summary"] is not None
    assert b["summary"]["n_settled"] == 2
    assert len(b["curve"]) == 2 and len(b["ledger"]) == 2


def test_edges_shape_degrades_gracefully(auth_client):
    # seeded DB has no game tables -> empty edges with surfaced per-sport errors, not a 500
    e = auth_client.get("/api/edges?sports=nba,nhl").json()
    assert e["edges"] == [] and e["count"] == 0
    assert len(e["errors"]) == 2 and e["sports"] == ["nba", "nhl"]


def test_edges_export_csv(auth_client):
    r = auth_client.get("/api/edges/export.csv?sports=nba")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert r.text.splitlines()[0] == "sport,market,game,bet,odds,edge,kelly_stake"


def test_manual_options_degrades(auth_client):
    o = auth_client.get("/api/manual/options?sport=nba").json()
    assert o["sport"] == "nba"
    assert isinstance(o["entities"], list)          # [] on the seeded DB, not a 500
    assert o["prop_stats"] == ["pts", "reb", "ast", "stl", "blk"]


def test_manual_edge_missing_field_is_400(auth_client):
    r = auth_client.post("/api/manual/edge",
                         json={"sport": "nba", "market": "moneyline", "side_a": "X"})
    assert r.status_code == 400 and "missing field" in r.json()["detail"]


def test_endpoints_are_gated(client):
    for path in ("/api/edges", "/api/injuries", "/api/quota", "/api/bankroll",
                 "/api/manual/options"):
        assert client.get(path).status_code == 401


# ── snapshot admin reload (local backend) ───────────────────────────────────────

def test_admin_reload(auth_client, tmp_path, monkeypatch):
    from webapp import snapshot

    proj = tmp_path / "proj"
    (proj / "artifacts").mkdir(parents=True)
    (proj / "sims").mkdir()
    src_db = proj / "sports.db"
    con = sqlite3.connect(src_db)
    con.execute("CREATE TABLE t(x)")
    con.execute("INSERT INTO t VALUES (42)")
    con.commit()
    con.close()
    (proj / "artifacts" / "nba_moneyline.pkl").write_bytes(b"PKL")
    (proj / "sims" / "paper_ledger.csv").write_text("bet_id,balance\nx,1000\n")

    bucket = tmp_path / "bucket"
    vol = tmp_path / "vol"
    snapshot.publish(f"file://{bucket}", str(src_db),
                     str(proj / "artifacts"), str(proj / "sims" / "paper_ledger.csv"))

    monkeypatch.setenv("SNAPSHOT_URL", f"file://{bucket}")
    monkeypatch.setenv("SNAPSHOT_DATA_DIR", str(vol))

    r = auth_client.post("/api/admin/reload")
    assert r.status_code == 200 and r.json()["status"] == "updated"
    assert (vol / "sports.db").is_file()
    assert sqlite3.connect(str(vol / "sports.db")).execute("SELECT x FROM t").fetchone()[0] == 42


# ── live pull (POST /api/pull) ──────────────────────────────────────────────────

def test_pull_requires_auth(client):
    assert client.post("/api/pull", json={"sports": ["nba"]}).status_code == 401


def test_pull_runs_pipeline_and_clears_cache(auth_client, monkeypatch):
    import pipeline
    from webapp.services import edges as edges_svc

    edges_svc._cache["nba"] = (0.0, [{"stale": True}])  # primed; the pull must clear it
    captured = {}

    def fake_run_pull(sports, db_path, **kw):
        captured["sports"] = list(sports)
        captured["db_path"] = db_path
        captured["kw"] = kw
        return {
            "duration_s": 1.5, "sports": list(sports), "fetched_odds": kw.get("fetch_odds", True),
            "ingested_games": True, "stages": [], "edges": [{"market": "Moneyline"}],
            "n_edges": 1, "edge_errors": [],
            "quota": {"remaining": 404, "used": 96, "fetched_at": "2026-06-17T05:00:00Z"},
            "paper": None,
        }

    monkeypatch.setattr(pipeline, "run_pull", fake_run_pull)

    r = auth_client.post("/api/pull", json={"sports": ["nba"], "odds": False})
    assert r.status_code == 200
    body = r.json()
    assert body["n_edges"] == 1
    assert "edges" not in body                       # full list dropped from the response
    assert body["quota"]["remaining"] == 404
    # ran against the shared serving DB with the requested options
    assert captured["sports"] == ["nba"]
    assert captured["db_path"] == _DB
    assert captured["kw"]["fetch_odds"] is False
    assert edges_svc._cache == {}                     # cache cleared so /api/edges recomputes


def test_pull_defaults_and_filters_sports(auth_client, monkeypatch):
    import pipeline

    seen = {}

    def fake_run_pull(sports, db_path, **kw):
        seen["sports"] = list(sports)
        return {"duration_s": 0.0, "sports": list(sports), "fetched_odds": True,
                "ingested_games": True, "stages": [], "edges": [], "n_edges": 0,
                "edge_errors": [], "quota": None, "paper": None}

    monkeypatch.setattr(pipeline, "run_pull", fake_run_pull)
    # an unknown sport falls back to the server defaults rather than pulling nothing
    auth_client.post("/api/pull", json={"sports": ["soccer"]})
    assert seen["sports"] == ["nba", "nhl"]


def test_pull_rejects_when_busy(auth_client, monkeypatch):
    from webapp.routers import pull as pull_router
    monkeypatch.setattr(pull_router, "_running", True)
    r = auth_client.post("/api/pull", json={"sports": ["nba"]})
    assert r.status_code == 409 and "already running" in r.json()["detail"]


# ── DB-gated integration: real feature/model serving path ───────────────────────

def _find_real() -> tuple[str | None, str | None]:
    here = Path(__file__).resolve()
    for base in here.parents:
        db = base / "data" / "sports.db"
        art = base / "models" / "artifacts"
        if db.is_file() and art.is_dir() and any(art.glob("*.pkl")):
            return str(db), str(art)
    return None, None


_REAL_DB, _REAL_ART = _find_real()


@pytest.mark.skipif(not _REAL_DB, reason="production sports.db + artifacts not available")
def test_real_serving_path(auth_client, tmp_path, monkeypatch):
    # writable, WAL-checkpointed copy of the read-only production DB
    dst = str(tmp_path / "real.db")
    src = sqlite3.connect(f"file:{_REAL_DB}?immutable=1", uri=True)
    out = sqlite3.connect(dst)
    with out:
        src.backup(out)
    src.close()
    out.close()

    monkeypatch.setattr("webapp.settings.DB_PATH", dst)
    monkeypatch.setattr("edge.calculator.ARTIFACTS_DIR", Path(_REAL_ART))
    from webapp.services import edges as edges_svc
    edges_svc.clear_cache()

    # the full build-features -> load-models -> find-edges path executes (errors empty)
    e = auth_client.get("/api/edges?sports=nba").json()
    assert e["errors"] == [], e["errors"]
    assert isinstance(e["edges"], list)

    # real dropdown names, then a real model-vs-price edge for two of them
    opts = auth_client.get("/api/manual/options?sport=nba").json()
    teams = opts["entities"]
    assert len(teams) >= 2
    r = auth_client.post("/api/manual/edge", json={
        "sport": "nba", "market": "moneyline",
        "side_a": teams[0], "side_b": teams[1], "odds_a": -150, "odds_b": 130,
    })
    assert r.status_code == 200 and r.json()["count"] >= 1
