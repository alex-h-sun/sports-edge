"""Tests for conditional-GET caching in tennis match ingestion (ingestion/tennis.py).

These exercise the ETag / Last-Modified path that lets a second same-day
`fetch_recent` skip re-downloading an unchanged Sackmann CSV, and only ingest
genuinely new matches when the file has grown. The network is fully mocked, so
no real HTTP request is made.
"""

import sqlite3

import pytest
import requests

from ingestion import tennis as tn


# ── fake HTTP layer ──────────────────────────────────────────────────────────────

def _csv_bytes(match_nums, tourney_id="2026-001"):
    """Minimal Sackmann-shaped CSV with one row per match_num."""
    header = ("tourney_id,tourney_name,surface,tourney_date,match_num,"
              "winner_name,loser_name,score")
    lines = [header] + [
        f"{tourney_id},Test Open,Hard,20260106,{m},Winner {m},Loser {m},6-4 6-3"
        for m in match_nums
    ]
    return ("\n".join(lines) + "\n").encode()


class _FakeResp:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class _FakeSackmann:
    """One CSV URL with an ETag: returns 304 when the client's If-None-Match
    matches the current ETag, otherwise 200 with the published content."""

    def __init__(self):
        self.etag = None
        self.content = b""
        self.n_200 = 0
        self.n_304 = 0
        self.last_headers = None

    def publish(self, content, etag):
        self.content = content
        self.etag = etag

    def get(self, url, headers=None, timeout=None, **kw):
        headers = headers or {}
        self.last_headers = headers
        if headers.get("If-None-Match") is not None and headers["If-None-Match"] == self.etag:
            self.n_304 += 1
            return _FakeResp(304, headers={"ETag": self.etag})
        self.n_200 += 1
        out_headers = {"ETag": self.etag} if self.etag else {}
        return _FakeResp(200, content=self.content, headers=out_headers)


@pytest.fixture
def conn(tmp_path):
    c = tn._get_conn(str(tmp_path / "t.db"))
    tn._ensure_tables(c)
    yield c
    c.close()


def _url(tour="atp", year=2026):
    return tn._SACKMANN[tour].format(year=year)


# ── _fetch_year: validators + conditional requests ──────────────────────────────

def test_fetch_year_ok_stores_validators(conn, monkeypatch):
    server = _FakeSackmann()
    server.publish(_csv_bytes([1, 2]), etag='W/"v1"')
    monkeypatch.setattr(tn.requests, "get", server.get)

    status, rows = tn._fetch_year("atp", 2026, conn)

    assert status == "ok"
    assert len(rows) == 2
    etag, _ = tn._cache_get(conn, _url())
    assert etag == 'W/"v1"'


def test_second_fetch_is_conditional_and_skips(conn, monkeypatch):
    server = _FakeSackmann()
    server.publish(_csv_bytes([1, 2]), etag='W/"v1"')
    monkeypatch.setattr(tn.requests, "get", server.get)

    s1, r1 = tn._fetch_year("atp", 2026, conn)
    tn._upsert_matches(conn, r1)
    assert s1 == "ok"

    s2, r2 = tn._fetch_year("atp", 2026, conn)

    assert s2 == "unchanged"
    assert r2 == []
    # the repeat run must send the stored validator and get a 304
    assert server.last_headers.get("If-None-Match") == 'W/"v1"'
    assert (server.n_200, server.n_304) == (1, 1)
    # nothing re-ingested or duplicated
    assert conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0] == 2


def test_changed_file_adds_only_new_matches(conn, monkeypatch):
    server = _FakeSackmann()
    server.publish(_csv_bytes([1, 2]), etag='W/"v1"')
    monkeypatch.setattr(tn.requests, "get", server.get)

    s1, r1 = tn._fetch_year("atp", 2026, conn)
    new1 = tn._upsert_matches(conn, r1)
    assert (s1, new1) == ("ok", 2)

    # a new match is appended and the ETag rolls over
    server.publish(_csv_bytes([1, 2, 3]), etag='W/"v2"')
    s2, r2 = tn._fetch_year("atp", 2026, conn)
    new2 = tn._upsert_matches(conn, r2)

    assert s2 == "ok"
    assert len(r2) == 3
    assert new2 == 1  # only the genuinely new match counts as new
    assert conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0] == 3
    etag, _ = tn._cache_get(conn, _url())
    assert etag == 'W/"v2"'


def test_missing_year_returns_missing(conn, monkeypatch):
    monkeypatch.setattr(
        tn.requests, "get",
        lambda url, headers=None, timeout=None, **kw: _FakeResp(404, content=b"404: Not Found"),
    )
    status, rows = tn._fetch_year("atp", 2099, conn)
    assert status == "missing"
    assert rows == []


def test_no_validators_refetches_each_time(conn, monkeypatch):
    """A server that returns no ETag must degrade to a full fetch every run."""
    calls = {"n": 0}

    def no_etag(url, headers=None, timeout=None, **kw):
        calls["n"] += 1
        return _FakeResp(200, content=_csv_bytes([1, 2]), headers={})

    monkeypatch.setattr(tn.requests, "get", no_etag)

    s1, r1 = tn._fetch_year("atp", 2026, conn)
    s2, r2 = tn._fetch_year("atp", 2026, conn)

    assert s1 == s2 == "ok"
    assert len(r1) == len(r2) == 2
    assert calls["n"] == 2  # no validator -> never a 304, always a real pull
    etag, _ = tn._cache_get(conn, _url())
    assert etag is None  # stored but null; no crash


# ── fetch_recent: twice-in-a-day integration ─────────────────────────────────────

def test_fetch_recent_twice_skips_second(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "t.db")
    server = _FakeSackmann()
    server.publish(_csv_bytes([1, 2]), etag='W/"v1"')
    monkeypatch.setattr(tn.requests, "get", server.get)
    monkeypatch.setattr(tn.time, "sleep", lambda *_: None)

    tn.fetch_recent(db, tours=["atp"], year=2026)
    tn.fetch_recent(db, tours=["atp"], year=2026)

    out = capsys.readouterr().out
    assert "unchanged since last pull, skipped" in out
    assert (server.n_200, server.n_304) == (1, 1)  # one real download, one skip

    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0]
    conn.close()
    assert n == 2  # not duplicated by the second run
