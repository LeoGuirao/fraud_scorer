# src/fraud_scorer/storage/db.py
from __future__ import annotations
import sqlite3, json, hashlib, os, datetime, uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

DB_PATH = Path(os.getenv("FRAUD_DB_PATH", "data/cases.db"))

def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db() -> None:
    """Crea todas las tablas necesarias si no existen (esquema nuevo con year/seq)."""
    print(f"Inicializando DB en {DB_PATH.resolve()}")
    with get_conn() as conn:
        # cases
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id     TEXT    NOT NULL UNIQUE,
                name        TEXT    NOT NULL,
                base_path   TEXT,
                year        INTEGER NOT NULL,
                seq         INTEGER NOT NULL,
                status      TEXT    NOT NULL DEFAULT 'new',
                notes       TEXT,
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );
            """
        )
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_cases_case_id ON cases(case_id);")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_cases_year_seq ON cases(year, seq);")

        # documents
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,
                case_id     TEXT NOT NULL,
                filename    TEXT NOT NULL,
                filepath    TEXT,
                file_hash   TEXT NOT NULL,
                mime_type   TEXT,
                size_bytes  INTEGER,
                page_count  INTEGER,
                language    TEXT,
                ocr_success INTEGER,
                created_at  TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_case_hash ON documents(case_id, file_hash);"
        )

        # ocr_results
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_results (
                document_id     TEXT PRIMARY KEY,
                raw_text        TEXT,
                key_value_pairs TEXT,
                tables          TEXT,
                entities        TEXT,
                confidence      TEXT,
                metadata        TEXT,
                errors          TEXT,
                engine          TEXT,
                engine_version  TEXT,
                processed_at    TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            """
        )

        # extracted_data
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extracted_data (
                document_id       TEXT PRIMARY KEY,
                document_type     TEXT,
                entities          TEXT,
                key_value_pairs   TEXT,
                extra             TEXT,
                extractor_version TEXT,
                processed_at      TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            """
        )

        # runs
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id         TEXT PRIMARY KEY,
                case_id    TEXT NOT NULL,
                purpose    TEXT,
                llm_model  TEXT,
                params     TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
            );
            """
        )

        # ai_analyses
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_analyses (
                id                   TEXT PRIMARY KEY,
                document_id          TEXT NOT NULL,
                run_id               TEXT,
                content_analysis     TEXT,
                visual_analysis      TEXT,
                contextual_analysis  TEXT,
                summary              TEXT,
                report_points        TEXT,
                alerts               TEXT,
                model                TEXT,
                temperature          REAL,
                processed_at         TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL
            );
            """
        )
    print("✓ Esquema OK")

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Compat helpers que usan el esquema nuevo
def upsert_document(case_id: str, filepath: str, mime_type: Optional[str], page_count: Optional[int], language: Optional[str]) -> Tuple[str, bool]:
    p = Path(filepath)
    file_hash = sha256_of_file(p)
    size_bytes = p.stat().st_size if p.exists() else None
    doc_id = str(uuid.uuid4())
    with get_conn() as conn:
        cur = conn.execute("SELECT id FROM documents WHERE case_id=? AND file_hash=?", (case_id, file_hash))
        row = cur.fetchone()
        if row:
            return row["id"], False
        conn.execute(
            """INSERT INTO documents(id, case_id, filename, filepath, file_hash, mime_type, size_bytes, page_count, language, ocr_success, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, case_id, p.name, str(p), file_hash, mime_type, size_bytes, page_count, language, None, _now()),
        )
    return doc_id, True

def mark_ocr_success(document_id: str, ok: bool) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE documents SET ocr_success=? WHERE id=?", (1 if ok else 0, document_id))

def save_ocr_result(document_id: str, ocr_dict: Dict[str, Any], engine: str, engine_version: Optional[str]) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO ocr_results(document_id, raw_text, key_value_pairs, tables, entities, confidence, metadata, errors, engine, engine_version, processed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                ocr_dict.get("text", ""),
                json.dumps(ocr_dict.get("key_value_pairs") or {}, ensure_ascii=False),
                json.dumps(ocr_dict.get("tables") or [], ensure_ascii=False),
                json.dumps(ocr_dict.get("entities") or [], ensure_ascii=False),
                json.dumps(ocr_dict.get("confidence_scores") or {}, ensure_ascii=False),
                json.dumps(ocr_dict.get("metadata") or {}, ensure_ascii=False),
                json.dumps(ocr_dict.get("errors") or [], ensure_ascii=False),
                engine,
                engine_version or "",
                _now(),
            ),
        )

def save_extracted_data(document_id: str, extracted: Dict[str, Any], extractor_version: str = "v1") -> None:
    with get_conn() as conn:
        extra = {k: v for k, v in (extracted or {}).items() if k not in {"document_type", "entities", "key_value_pairs"}}
        conn.execute(
            """INSERT OR REPLACE INTO extracted_data(document_id, document_type, entities, key_value_pairs, extra, extractor_version, processed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                (extracted or {}).get("document_type", "desconocido"),
                json.dumps(((extracted or {}).get("entities") or {}), ensure_ascii=False),
                json.dumps(((extracted or {}).get("key_value_pairs") or {}), ensure_ascii=False),
                json.dumps(extra, ensure_ascii=False),
                extractor_version,
                _now(),
            ),
        )

def create_run(case_id: str, purpose: str, llm_model: str, params: Dict[str, Any]) -> str:
    run_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO runs(id, case_id, purpose, llm_model, params, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, case_id, purpose, llm_model, json.dumps(params, ensure_ascii=False), _now()),
        )
    return run_id

def save_ai_analysis(document_id: str, run_id: Optional[str], ai: Dict[str, Any], model: str, temperature: float) -> str:
    analysis_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO ai_analyses(id, document_id, run_id, content_analysis, visual_analysis, contextual_analysis, summary, report_points, alerts, model, temperature, processed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis_id, document_id, run_id,
                json.dumps(ai.get("content_analysis"), ensure_ascii=False),
                json.dumps(ai.get("visual_analysis"), ensure_ascii=False),
                json.dumps(ai.get("contextual_analysis"), ensure_ascii=False),
                ai.get("summary", ""),
                json.dumps(ai.get("report_points") or [], ensure_ascii=False),
                json.dumps(ai.get("alerts") or [], ensure_ascii=False),
                model, float(temperature), _now(),
            ),
        )
    return analysis_id

# --- NUEVOS HELPERS DE CACHE OCR ---

def get_document_by_id(document_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return row

def get_ocr_by_document_id(document_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM ocr_results WHERE document_id = ?", (document_id,)).fetchone()
        return row

def get_document_by_case_and_hash(case_id: str, file_hash: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE case_id = ? AND file_hash = ?",
            (case_id, file_hash)
        ).fetchone()
        return row

def get_any_ocr_by_hash(file_hash: str) -> Optional[sqlite3.Row]:
    """
    Busca en toda la base si existe algún OCR para un archivo con este hash,
    sin importar el caso. Útil para reuso global.
    """
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT ocr.*
            FROM documents d
            JOIN ocr_results ocr ON ocr.document_id = d.id
            WHERE d.file_hash = ?
            ORDER BY ocr.processed_at DESC
            LIMIT 1
            """,
            (file_hash,)
        ).fetchone()
        return row

def copy_ocr_to_document(src_ocr_row: sqlite3.Row, target_document_id: str) -> None:
    """
    Copia un OCR existente (row) a otro document_id (útil para reuso global por hash).
    """
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO ocr_results(
                   document_id, raw_text, key_value_pairs, tables, entities, confidence,
                   metadata, errors, engine, engine_version, processed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                target_document_id,
                src_ocr_row["raw_text"],
                src_ocr_row["key_value_pairs"],
                src_ocr_row["tables"],
                src_ocr_row["entities"],
                src_ocr_row["confidence"],
                src_ocr_row["metadata"],
                src_ocr_row["errors"],
                src_ocr_row["engine"],
                src_ocr_row["engine_version"],
                _now(),
            ),
        )

def get_extracted_by_document_id(document_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM extracted_data WHERE document_id = ?",
            (document_id,)
        ).fetchone()
        return row

if __name__ == "__main__":
    init_db()
