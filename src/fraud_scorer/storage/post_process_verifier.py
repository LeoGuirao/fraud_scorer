"""Utilities to validate and clean case artifacts after processing."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db import get_conn
from .ocr_cache import OCRCacheManager

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Structured payload describing the outcome of a verification."""

    case_id: str
    db_counts: Dict[str, int]
    index_path: Optional[str]
    issues: List[str]
    warnings: List[str]
    duplicates_removed: List[Dict[str, Any]]
    missing_shards: List[Dict[str, Any]]
    index_drift: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "db_counts": self.db_counts,
            "index_path": self.index_path,
            "issues": self.issues,
            "warnings": self.warnings,
            "duplicates_removed": self.duplicates_removed,
            "missing_shards": self.missing_shards,
            "index_drift": self.index_drift,
        }


_CRITICAL_ISSUES = {
    "case_not_found_in_db",
    "no_documents_in_db",
    "missing_case_index",
}


def _summarize_counts(case_id: str) -> Dict[str, int]:
    counts: Dict[str, int] = {
        "cases": 0,
        "documents": 0,
        "ocr_results": 0,
        "extracted_data": 0,
        "fraud_analyses": 0,
        "ai_analyses": 0,
        "runs": 0,
    }
    with get_conn() as conn:
        counts["cases"] = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()[0]
        counts["documents"] = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE case_id = ?",
            (case_id,),
        ).fetchone()[0]
        counts["ocr_results"] = conn.execute(
            "SELECT COUNT(*) FROM ocr_results WHERE document_id IN (SELECT id FROM documents WHERE case_id = ?)",
            (case_id,),
        ).fetchone()[0]
        counts["extracted_data"] = conn.execute(
            "SELECT COUNT(*) FROM extracted_data WHERE document_id IN (SELECT id FROM documents WHERE case_id = ?)",
            (case_id,),
        ).fetchone()[0]
        counts["fraud_analyses"] = conn.execute(
            "SELECT COUNT(*) FROM fraud_analyses WHERE case_id = ?",
            (case_id,),
        ).fetchone()[0]
        counts["ai_analyses"] = conn.execute(
            "SELECT COUNT(*) FROM ai_analyses WHERE document_id IN (SELECT id FROM documents WHERE case_id = ?)",
            (case_id,),
        ).fetchone()[0]
        counts["runs"] = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE case_id = ?",
            (case_id,),
        ).fetchone()[0]
    return counts


def verify_case_artifacts(
    case_id: str,
    *,
    autofix_duplicates: bool = True,
    raise_on_issue: bool = True,
) -> Dict[str, Any]:
    """Validate that the processed artifacts for ``case_id`` are consistent."""

    issues: List[str] = []
    warnings: List[str] = []
    duplicates_removed: List[Dict[str, Any]] = []
    missing_shards: List[Dict[str, Any]] = []
    index_drift: Dict[str, List[str]] = {"missing_in_index": [], "missing_in_db": []}

    counts = _summarize_counts(case_id)
    if counts["cases"] == 0:
        issues.append("case_not_found_in_db")
        result = VerificationResult(
            case_id=case_id,
            db_counts=counts,
            index_path=None,
            issues=issues,
            warnings=warnings,
            duplicates_removed=duplicates_removed,
            missing_shards=missing_shards,
            index_drift=index_drift,
        )
        if raise_on_issue:
            raise RuntimeError(f"Post-process verification failed: {issues}")
        return result.to_dict()

    with get_conn() as conn:
        doc_rows = [
            dict(row)
            for row in conn.execute(
                "SELECT id, filename, filepath, file_hash, created_at FROM documents WHERE case_id = ?",
                (case_id,),
            ).fetchall()
        ]
        if not doc_rows:
            issues.append("no_documents_in_db")

        grouping = defaultdict(list)
        for row in doc_rows:
            file_hash = row.get("file_hash")
            if not file_hash:
                continue
            grouping[file_hash].append(row)

        for file_hash, rows in grouping.items():
            if len(rows) <= 1:
                continue
            rows_sorted = sorted(
                rows,
                key=lambda r: ((r.get("created_at") or ""), r["id"]),
                reverse=True,
            )
            to_keep = rows_sorted[0]
            to_prune = rows_sorted[1:]
            duplicate_entry = {
                "file_hash": file_hash,
                "kept_document_id": to_keep["id"],
                "removed_document_ids": [item["id"] for item in to_prune],
            }
            duplicates_removed.append(duplicate_entry)
            if autofix_duplicates:
                for item in to_prune:
                    conn.execute("DELETE FROM documents WHERE id = ?", (item["id"],))
                conn.commit()
                counts["documents"] -= len(to_prune)

    cache_manager = OCRCacheManager()
    index_path = cache_manager.index_dir / f"{case_id}.json"
    case_index = cache_manager.get_case_index(case_id)
    if not case_index or not index_path.exists():
        issues.append("missing_case_index")
        index_data: Dict[str, Any] = {}
    else:
        index_data = dict(case_index)
        db_hashes = {
            row.get("file_hash")
            for row in doc_rows
            if row.get("file_hash")
        }
        index_hashes = set((index_data.get("document_hashes") or {}).values())
        missing_in_index = sorted(hash_ for hash_ in db_hashes if hash_ not in index_hashes)
        missing_in_db = sorted(hash_ for hash_ in index_hashes if hash_ not in db_hashes)
        if missing_in_index:
            index_drift["missing_in_index"] = missing_in_index
            warnings.append("hashes_missing_in_index")
        if missing_in_db:
            index_drift["missing_in_db"] = missing_in_db
            warnings.append("hashes_missing_in_db")

        doc_hashes = index_data.get("document_hashes") or {}
        for stored_hash in doc_hashes.values():
            if not stored_hash:
                continue
            shard_path = cache_manager.cache_dir / stored_hash[:2] / f"{stored_hash}.json"
            if not shard_path.exists():
                missing_shards.append(
                    {
                        "file_hash": stored_hash,
                        "expected_path": str(shard_path),
                    }
                )
        case_folder = cache_manager.get_case_folder_path(case_id, index_data)
        if not case_folder.exists():
            warnings.append("missing_reorganized_folder")

    result = VerificationResult(
        case_id=case_id,
        db_counts=counts,
        index_path=str(index_path) if index_path.exists() else None,
        issues=issues,
        warnings=warnings,
        duplicates_removed=duplicates_removed,
        missing_shards=missing_shards,
        index_drift=index_drift,
    )

    critical = [issue for issue in issues if issue in _CRITICAL_ISSUES]
    if critical and raise_on_issue:
        raise RuntimeError(f"Post-process verification failed: {critical}")

    if duplicates_removed:
        logger.info("🧹 Se eliminaron %s duplicados para %s", len(duplicates_removed), case_id)
    if missing_shards:
        logger.warning(
            "⚠️ Falta materializar %s shards de OCR para el caso %s",
            len(missing_shards),
            case_id,
        )
    if critical:
        logger.error("❌ Verificación con problemas críticos (%s) para %s", critical, case_id)
    else:
        logger.info("✓ Verificación post-proceso completada para %s", case_id)

    return result.to_dict()
