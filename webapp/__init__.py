"""Web app for sports-edge: a JSON API (Starlette) that serves a read-only snapshot
of the model artifacts + SQLite DB, plus the built React SPA.

The heavy ingest/training pipeline (run.py) stays offline; this layer only reads a
published snapshot and reuses the existing edge logic (edge/, features/, models/).
"""
