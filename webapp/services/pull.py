"""Live data pull for the web app.

Runs the very same ingest -> features -> edges -> paper-ledger flow as ``python run.py``
(``pipeline.run_pull``), writing the shared ``DB_PATH``. Because the terminal and the web
app point at the same SQLite file, whatever this pulls is immediately visible to
``run.py``/``streamlit``/the API and vice-versa — the two never drift.

This is the synchronous worker; the router wraps it in a threadpool + single-flight so a
slow, network-bound pull never blocks the event loop or runs twice at once.
"""

import logging
from pathlib import Path

import pipeline
from webapp import settings, snapshot
from webapp.services import edges as edges_svc

log = logging.getLogger("webapp.pull")


def _maybe_publish() -> dict | None:
    """Deployed cross-machine sync: re-publish the freshly-pulled snapshot so other
    consumers (e.g. a laptop running ``webapp.snapshot --sync``) can pick it up. No-op
    unless ``PUBLISH_AFTER_PULL`` and ``SNAPSHOT_URL`` are both set. Never raises — a
    publish hiccup must not fail the pull itself."""
    if not (settings.PUBLISH_AFTER_PULL and settings.SNAPSHOT_URL):
        return None
    try:
        manifest = snapshot.publish(
            settings.SNAPSHOT_URL, settings.DB_PATH,
            settings.ARTIFACTS_DIR, settings.LEDGER_PATH,
        )
        # Record the version on our own volume so the next boot/sync does not re-download
        # the snapshot we just produced.
        Path(settings.SNAPSHOT_DATA_DIR, snapshot.VERSION_MARKER).write_text(manifest["version"])
        return {"version": manifest["version"]}
    except Exception as e:
        log.warning("publish-after-pull failed", exc_info=True)
        return {"error": str(e)}


def do_pull(sports: list[str], fetch_odds: bool = True) -> dict:
    """Pull fresh data into the shared DB and return a JSON-safe summary. The full edge
    list is dropped from the response (the SPA re-fetches it via ``/api/edges`` against
    the freshly-cleared cache); only the count and any per-stage errors are kept."""
    summary = pipeline.run_pull(
        sports, settings.DB_PATH,
        bankroll=settings.BANKROLL, min_edge=settings.MIN_EDGE,
        fetch_odds=fetch_odds, ledger_path=settings.LEDGER_PATH,
    )
    edges_svc.clear_cache()  # so the next /api/edges recomputes from the new data
    summary.pop("edges", None)  # keep the response lean; edges come from /api/edges

    published = _maybe_publish()
    if published is not None:
        summary["published"] = published
    return summary
