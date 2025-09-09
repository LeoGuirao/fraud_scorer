#!/usr/bin/env python3
"""
Mantenimiento y saneamiento de la base de datos SQLite (data/cases.db)

Incluye:
- Limpieza de orfandad: extracted_data y runs
- Normalización opcional de rutas absolutas → relativas
- Creación de triggers de limpieza (cinturón y tirantes)
- Compactación (VACUUM) y ANALYZE

Uso:
  python scripts/db_maintenance.py --dry-run
  python scripts/db_maintenance.py --fix-orphans --vacuum --analyze
  python scripts/db_maintenance.py --normalize-paths "FROM_PREFIX" "TO_PREFIX"
  python scripts/db_maintenance.py --create-triggers
"""
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path
import os

DB_PATH = Path(os.getenv("FRAUD_DB_PATH", "data/cases.db"))

def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def count_orphans(conn: sqlite3.Connection) -> dict:
    c = {}
    c['orphan_extracted'] = conn.execute(
        "SELECT COUNT(*) FROM extracted_data WHERE document_id NOT IN (SELECT id FROM documents)"
    ).fetchone()[0]
    c['orphan_ocr'] = conn.execute(
        "SELECT COUNT(*) FROM ocr_results WHERE document_id NOT IN (SELECT id FROM documents)"
    ).fetchone()[0]
    c['runs_orphan'] = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE case_id NOT IN (SELECT case_id FROM cases)"
    ).fetchone()[0]
    c['docs_without_extracted'] = conn.execute(
        "SELECT COUNT(*) FROM documents d WHERE NOT EXISTS (SELECT 1 FROM extracted_data e WHERE e.document_id=d.id)"
    ).fetchone()[0]
    return c

def cleanup_orphans(conn: sqlite3.Connection) -> dict:
    stats_before = count_orphans(conn)
    conn.execute(
        "DELETE FROM extracted_data WHERE document_id NOT IN (SELECT id FROM documents)"
    )
    conn.execute(
        "DELETE FROM runs WHERE case_id NOT IN (SELECT case_id FROM cases)"
    )
    conn.execute(
        "DELETE FROM ocr_results WHERE document_id NOT IN (SELECT id FROM documents)"
    )
    stats_after = count_orphans(conn)
    return {"before": stats_before, "after": stats_after}

def normalize_paths(conn: sqlite3.Connection, from_prefix: str, to_prefix: str) -> int:
    ch1 = conn.execute(
        "UPDATE cases SET base_path = REPLACE(base_path, ?, ?) WHERE base_path LIKE ?",
        (from_prefix, to_prefix, from_prefix + '%')
    ).rowcount
    ch2 = conn.execute(
        "UPDATE documents SET filepath = REPLACE(filepath, ?, ?) WHERE filepath LIKE ?",
        (from_prefix, to_prefix, from_prefix + '%')
    ).rowcount
    return ch1 + ch2

def create_triggers(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_cleanup_extracted
        AFTER DELETE ON documents
        BEGIN
            DELETE FROM extracted_data WHERE document_id = OLD.id;
            DELETE FROM ocr_results   WHERE document_id = OLD.id;
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_cleanup_runs
        AFTER DELETE ON cases
        BEGIN
            DELETE FROM runs WHERE case_id = OLD.case_id;
        END;
        """
    )

def vacuum(conn: sqlite3.Connection) -> None:
    conn.execute("VACUUM;")

def analyze(conn: sqlite3.Connection) -> None:
    conn.execute("ANALYZE;")

def main():
    p = argparse.ArgumentParser(description="DB Maintenance for data/cases.db")
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--fix-orphans', action='store_true')
    p.add_argument('--normalize-paths', nargs=2, metavar=('FROM_PREFIX','TO_PREFIX'))
    p.add_argument('--create-triggers', action='store_true')
    p.add_argument('--vacuum', action='store_true')
    p.add_argument('--analyze', action='store_true')
    args = p.parse_args()

    print(f"DB: {DB_PATH}")
    with connect() as conn:
        if args.dry_run:
            print("-- Dry run: orphans --")
            print(count_orphans(conn))
            return

        if args.fix_orphans:
            res = cleanup_orphans(conn)
            print("-- Cleanup orphans:", res)

        if args.normalize_paths:
            from_prefix, to_prefix = args.normalize_paths
            changed = normalize_paths(conn, from_prefix, to_prefix)
            print(f"-- Normalize paths: {changed} rows updated")

        if args.create_triggers:
            create_triggers(conn)
            print("-- Triggers ensured")

        if args.vacuum:
            vacuum(conn)
            print("-- VACUUM done")

        if args.analyze:
            analyze(conn)
            print("-- ANALYZE done")

if __name__ == '__main__':
    main()

