from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from typing import Any

from src import DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db() -> None:
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('student', 'company')),
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            target_role TEXT,
            job_description TEXT,
            score REAL NOT NULL,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            must_have TEXT NOT NULL,
            preferred TEXT,
            min_experience REAL DEFAULT 0,
            education TEXT,
            location TEXT,
            weights_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            display_name TEXT NOT NULL,
            parsed_json TEXT NOT NULL,
            match_score REAL NOT NULL,
            explanation_json TEXT NOT NULL,
            shortlisted INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
        );
        """
    )
    now = datetime.utcnow().isoformat(timespec="seconds")
    for username, role in (("student", "student"), ("company", "company")):
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (username, hash_password("demo123"), role, now),
            )
    conn.commit()
    conn.close()


def create_user(username: str, password: str, role: str) -> tuple[bool, str]:
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (username.strip(), hash_password(password), role, datetime.utcnow().isoformat(timespec="seconds")),
        )
        conn.commit()
        return True, "Account created. You can log in now."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    finally:
        conn.close()


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, username, role FROM users WHERE username = ? AND password_hash = ?",
        (username.strip(), hash_password(password)),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_analysis(user_id: int, filename: str, target_role: str, jd: str, score: float, report: dict) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO analyses (user_id, filename, target_role, job_description, score, report_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            filename,
            target_role,
            jd,
            score,
            json.dumps(report),
            datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    analysis_id = int(cur.lastrowid)
    conn.close()
    return analysis_id


def list_analyses(user_id: int) -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, filename, target_role, score, created_at, report_json FROM analyses WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    items = []
    for row in rows:
        item = dict(row)
        item["report"] = json.loads(item.pop("report_json"))
        items.append(item)
    return items


def get_analysis(analysis_id: int, user_id: int) -> dict[str, Any] | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
        (analysis_id, user_id),
    ).fetchone()
    conn.close()
    if not row:
        return None
    item = dict(row)
    item["report"] = json.loads(item.pop("report_json"))
    return item


def save_job(user_id: int, payload: dict[str, Any]) -> int:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO jobs (user_id, title, must_have, preferred, min_experience, education, location, weights_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            payload["title"],
            payload["must_have"],
            payload.get("preferred", ""),
            float(payload.get("min_experience") or 0),
            payload.get("education", ""),
            payload.get("location", ""),
            json.dumps(payload["weights"]),
            datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    job_id = int(cur.lastrowid)
    conn.close()
    return job_id


def list_jobs(user_id: int) -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM jobs WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    items = []
    for row in rows:
        item = dict(row)
        item["weights"] = json.loads(item.pop("weights_json"))
        items.append(item)
    return items


def get_job(job_id: int, user_id: int) -> dict[str, Any] | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id)).fetchone()
    conn.close()
    if not row:
        return None
    item = dict(row)
    item["weights"] = json.loads(item.pop("weights_json"))
    return item


def replace_candidates(job_id: int, candidates: list[dict[str, Any]]) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM candidates WHERE job_id = ?", (job_id,))
    now = datetime.utcnow().isoformat(timespec="seconds")
    for c in candidates:
        conn.execute(
            """INSERT INTO candidates
               (job_id, filename, display_name, parsed_json, match_score, explanation_json, shortlisted, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
            (
                job_id,
                c["filename"],
                c["display_name"],
                json.dumps(c["parsed"]),
                c["match_score"],
                json.dumps(c["explanation"]),
                now,
            ),
        )
    conn.commit()
    conn.close()


def list_candidates(job_id: int) -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM candidates WHERE job_id = ? ORDER BY match_score DESC",
        (job_id,),
    ).fetchall()
    conn.close()
    items = []
    for row in rows:
        item = dict(row)
        item["parsed"] = json.loads(item.pop("parsed_json"))
        item["explanation"] = json.loads(item.pop("explanation_json"))
        items.append(item)
    return items


def set_shortlisted(candidate_id: int, shortlisted: bool) -> None:
    conn = get_conn()
    conn.execute("UPDATE candidates SET shortlisted = ? WHERE id = ?", (1 if shortlisted else 0, candidate_id))
    conn.commit()
    conn.close()
