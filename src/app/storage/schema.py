from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def initialize_database(
    db_path: Path,
    connect: Callable[[], sqlite3.Connection],
) -> None:
    """Create the current SQLite schema and secure the database file."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    db_path.chmod(0o600)
