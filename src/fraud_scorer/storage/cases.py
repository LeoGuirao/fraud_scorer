# src/fraud_scorer/storage/cases.py
from __future__ import annotations
import datetime
from typing import Optional, List
from sqlite3 import Row
from .db import get_conn, _now

def _generate_next_case_id(conn, year: Optional[int] = None):
    year = year or datetime.datetime.now().year
    cur = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM cases WHERE year = ?", (year,))
    max_seq = cur.fetchone()[0] or 0
    next_seq = max_seq + 1
    return f"CASE-{year}-{next_seq:04d}", year, next_seq

def create_case(title: str, base_path: Optional[str] = None, status: str = "new", case_id: Optional[str] = None, notes: Optional[str] = None) -> str:
    with get_conn() as conn:
        if case_id:
            row = conn.execute("SELECT case_id FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row:
                return row["case_id"]
            try:
                year = int(case_id.split("-")[1])
                seq = int(case_id.split("-")[2])
            except Exception:
                year = datetime.datetime.now().year
                cur = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM cases WHERE year = ?", (year,))
                seq = (cur.fetchone()[0] or 0) + 1
        else:
            case_id, year, seq = _generate_next_case_id(conn)

        now = _now()
        conn.execute(
            """
            INSERT INTO cases(case_id, name, base_path, year, seq, status, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (case_id, title, base_path, year, seq, status, notes, now, now),
        )
        return case_id

def get_case_by_id(case_id: str) -> Optional[Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()

def get_case_by_path(base_path: str) -> Optional[Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM cases WHERE base_path = ? ORDER BY created_at DESC LIMIT 1", (base_path,)).fetchone()

def list_cases(limit: int = 50, status: Optional[str] = None) -> List[Row]:
    with get_conn() as conn:
        if status:
            return conn.execute("SELECT * FROM cases WHERE status = ? ORDER BY created_at DESC LIMIT ?", (status, limit)).fetchall()
        return conn.execute("SELECT * FROM cases ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()

def update_case_status(case_id: str, status: str, notes: Optional[str] = None) -> bool:
    allowed = {"new", "processing", "ready", "archived"}
    if status not in allowed:
        raise ValueError(f"status inválido: {status}")
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE cases SET status = ?, notes = COALESCE(?, notes), updated_at = ? WHERE case_id = ?",
            (status, notes, _now(), case_id),
        )
        return cur.rowcount == 1
