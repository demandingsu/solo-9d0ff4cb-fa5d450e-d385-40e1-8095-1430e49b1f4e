"""SQLite helpers for the mosaic application."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS revisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_id       TEXT    NOT NULL,
    revision      INTEGER NOT NULL,
    x             INTEGER NOT NULL,
    y             INTEGER NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    image_rel     TEXT    NOT NULL,
    mask_rel      TEXT    NOT NULL,
    image_sha256  TEXT    NOT NULL,
    mask_sha256   TEXT    NOT NULL,
    cloud_pixels  INTEGER NOT NULL,
    clear_pixels  INTEGER NOT NULL,
    submitted_at  TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'active'
                          CHECK (status IN ('active', 'withdrawn')),
    withdrawn_at  TEXT
);

-- Only one row per tile revision may exist at any time.  Withdrawn rows keep
-- their history but a withdrawn revision cannot be re-submitted.
CREATE UNIQUE INDEX IF NOT EXISTS ux_revisions_tile_revision
    ON revisions(tile_id, revision);

CREATE INDEX IF NOT EXISTS ix_revisions_tile_revision
    ON revisions(tile_id, revision DESC);

CREATE TABLE IF NOT EXISTS published_versions (
    id                  TEXT PRIMARY KEY,
    seq                 INTEGER NOT NULL UNIQUE,
    dir_rel             TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    fingerprint         TEXT NOT NULL,
    canvas_width        INTEGER NOT NULL,
    canvas_height       INTEGER NOT NULL,
    tile_count          INTEGER NOT NULL,
    covered_pixels      INTEGER NOT NULL,
    transparent_pixels  INTEGER NOT NULL,
    png_rel             TEXT NOT NULL,
    csv_rel             TEXT NOT NULL,
    png_sha256          TEXT NOT NULL,
    csv_sha256          TEXT NOT NULL,
    png_bytes           INTEGER NOT NULL,
    csv_bytes           INTEGER NOT NULL
);
"""

DEFAULT_SETTINGS: dict[str, str] = {
    "canvas_width": "512",
    "canvas_height": "512",
    "state_version": "0",
    "preview_version": "-1",
    "preview_error": "",
    "draft_covered": "0",
    "draft_transparent": "262144",
    "published_version_id": "",
}


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(db_path), timeout=30.0, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | str) -> None:
    conn = connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = FULL")
        conn.executescript(SCHEMA)
        for key, value in DEFAULT_SETTINGS.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)",
                (key, value),
            )
        conn.commit()
    finally:
        conn.close()


def get_setting(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return default if row is None else str(row["value"])


def get_int_setting(conn: sqlite3.Connection, key: str, default: int = 0) -> int:
    value = get_setting(conn, key, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def set_setting(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO settings(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
