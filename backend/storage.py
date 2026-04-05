from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "signbridge.db"


def initialize_database() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gesture TEXT NOT NULL,
                text_output TEXT NOT NULL,
                confidence REAL NOT NULL,
                explanation TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def save_prediction(
    gesture: str,
    text_output: str,
    confidence: float,
    explanation: str,
) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO recognition_logs (gesture, text_output, confidence, explanation)
            VALUES (?, ?, ?, ?)
            """,
            (gesture, text_output, confidence, explanation),
        )
        connection.commit()


def fetch_recent_predictions(limit: int = 10) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, gesture, text_output, confidence, explanation, created_at
            FROM recognition_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]
