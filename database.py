"""SQLite persistence for electricity anomaly detection runs.

The module is deliberately independent of Streamlit and the model pipeline.
Call :func:`initialize_database` once at application startup, then use the
insert/save/log functions after a successful analysis.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().with_name("electricity_anomaly.db")
REVIEW_STATUSES = ("Pending", "Under Review", "Resolved")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    upload_date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_records INTEGER NOT NULL CHECK (total_records >= 0)
);

CREATE TABLE IF NOT EXISTS detection_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    risk_score REAL NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
    detection_date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES uploads(upload_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS flagged_users (
    flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    result_id INTEGER NOT NULL,
    risk_score REAL NOT NULL CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
    reason TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'Pending'
        CHECK (review_status IN ('Pending', 'Under Review', 'Resolved')),
    review_notes TEXT,
    flagged_date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (result_id) REFERENCES detection_results(result_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analysis_history (
    analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    analysis_date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    records_analysed INTEGER NOT NULL CHECK (records_analysed >= 0),
    records_flagged INTEGER NOT NULL CHECK (records_flagged >= 0),
    FOREIGN KEY (upload_id) REFERENCES uploads(upload_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_detection_results_upload_id
    ON detection_results(upload_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_user_id
    ON detection_results(user_id);
CREATE INDEX IF NOT EXISTS idx_flagged_users_status
    ON flagged_users(review_status);
CREATE INDEX IF NOT EXISTS idx_analysis_history_upload_id
    ON analysis_history(upload_id);
"""


def _utc_now() -> str:
    """Return a sortable UTC timestamp for records written by this module."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open a configured SQLite connection and ensure the schema exists."""
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.executescript(_SCHEMA)
    return connection


def initialize_database(db_path: str | Path = DB_PATH) -> None:
    """Create the database tables and indexes if they do not already exist."""
    with closing(get_connection(db_path)) as connection:
        connection.commit()


def insert_upload(
    filename: str,
    total_records: int,
    db_path: str | Path = DB_PATH,
) -> int:
    """Insert an upload and return its generated ``upload_id``."""
    if not filename or not str(filename).strip():
        raise ValueError("filename must not be empty.")
    if isinstance(total_records, bool) or total_records < 0:
        raise ValueError("total_records must be a non-negative integer.")

    with closing(get_connection(db_path)) as connection:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO uploads (filename, upload_date, total_records)
                VALUES (?, ?, ?)
                """,
                (str(filename), _utc_now(), int(total_records)),
            )
            return int(cursor.lastrowid)


def _records_as_mappings(
    records: Iterable[Mapping[str, Any]] | Any,
) -> list[Mapping[str, Any]]:
    """Normalize mappings or a pandas-like DataFrame to a list of mappings."""
    if hasattr(records, "to_dict"):
        records = records.to_dict(orient="records")
    normalized = list(records)
    if any(not isinstance(record, Mapping) for record in normalized):
        raise TypeError("records must contain mapping objects.")
    return normalized


def save_detection_results(
    upload_id: int,
    records: Iterable[Mapping[str, Any]] | Any,
    risk_threshold: float,
    db_path: str | Path = DB_PATH,
    reason: str | None = None,
) -> tuple[int, int]:
    """Save results and flag records whose risk score meets ``risk_threshold``.

    Each record must provide ``user_id``, ``prediction``, and ``risk_score``.
    The operation is atomic: detection results and their corresponding audit
    queue entries are committed together. Returns ``(records_saved,
    records_flagged)``.
    """
    if not 0.0 <= float(risk_threshold) <= 1.0:
        raise ValueError("risk_threshold must be between 0.0 and 1.0.")

    normalized = _records_as_mappings(records)
    timestamp = _utc_now()
    flagged_count = 0

    with closing(get_connection(db_path)) as connection:
        with connection:
            upload = connection.execute(
                "SELECT 1 FROM uploads WHERE upload_id = ?", (upload_id,)
            ).fetchone()
            if upload is None:
                raise ValueError(f"Upload {upload_id} does not exist.")

            for record in normalized:
                try:
                    user_id = str(record["user_id"]).strip()
                    prediction = int(record["prediction"])
                    risk_score = float(record["risk_score"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "Each result must contain user_id, prediction, and risk_score."
                    ) from exc

                if not user_id:
                    raise ValueError("user_id must not be empty.")
                if prediction not in (0, 1):
                    raise ValueError("prediction must be 0 or 1.")
                if not 0.0 <= risk_score <= 1.0:
                    raise ValueError("risk_score must be between 0.0 and 1.0.")

                result_cursor = connection.execute(
                    """
                    INSERT INTO detection_results
                        (upload_id, user_id, prediction, risk_score, detection_date)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (upload_id, user_id, prediction, risk_score, timestamp),
                )
                result_id = int(result_cursor.lastrowid)

                if risk_score >= float(risk_threshold):
                    flagged_count += 1
                    flag_reason = reason or (
                        f"Risk score {risk_score:.4f} meets the high-risk "
                        f"threshold {float(risk_threshold):.4f}."
                    )
                    connection.execute(
                        """
                        INSERT INTO flagged_users
                            (user_id, result_id, risk_score, reason, flagged_date)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (user_id, result_id, risk_score, flag_reason, timestamp),
                    )

            return len(normalized), flagged_count


def log_analysis_history(
    upload_id: int,
    model_name: str,
    records_analysed: int,
    records_flagged: int,
    db_path: str | Path = DB_PATH,
) -> int:
    """Log an analysis run and return its generated ``analysis_id``."""
    if not model_name or not str(model_name).strip():
        raise ValueError("model_name must not be empty.")
    if (
        isinstance(records_analysed, bool)
        or isinstance(records_flagged, bool)
        or records_analysed < 0
        or records_flagged < 0
        or records_flagged > records_analysed
    ):
        raise ValueError("Analysis record counts are invalid.")

    with closing(get_connection(db_path)) as connection:
        with connection:
            upload = connection.execute(
                "SELECT 1 FROM uploads WHERE upload_id = ?", (upload_id,)
            ).fetchone()
            if upload is None:
                raise ValueError(f"Upload {upload_id} does not exist.")
            cursor = connection.execute(
                """
                INSERT INTO analysis_history
                    (upload_id, model_name, analysis_date, records_analysed, records_flagged)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    str(model_name),
                    _utc_now(),
                    int(records_analysed),
                    int(records_flagged),
                ),
            )
            return int(cursor.lastrowid)
