"""Herramientas de limpieza para índices del Agente Rick."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fraud_scorer.storage.ocr_cache import OCRCacheManager

from ..vector_store import RickVectorStoreManager
from ..vector_store.maintenance import cleanup_missing_cases, list_indexed_cases
from ..ingestion.document_loader import _normalize


logger = logging.getLogger(__name__)


def cleanup_case(case_id: str) -> None:
    manager = RickVectorStoreManager()
    resolved_id = _resolve_case_id(case_id)
    manager.delete_case_index(resolved_id)
    logger.info("Índice eliminado para el caso %s", resolved_id)


def cleanup_orphans() -> None:
    cm = OCRCacheManager()
    cases = cm.list_cached_cases()
    existing_ids = [item.get("case_id") for item in cases if item.get("case_id")]
    removed = cleanup_missing_cases(existing_ids)
    if not removed:
        logger.info("No se encontraron índices huérfanos")
    else:
        logger.info("Se eliminaron índices para: %s", ", ".join(removed))


def show_stats() -> None:
    for item in list_indexed_cases():
        logger.info("Caso %s -> %s bytes", item["case_id"], item["size_bytes"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limpieza de índices del Agente Rick")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-id")
    group.add_argument("--cleanup-missing", action="store_true")
    group.add_argument("--stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    if args.case_id:
        cleanup_case(args.case_id)
    elif args.cleanup_missing:
        cleanup_orphans()
    elif args.stats:
        show_stats()


def _resolve_case_id(identifier: str) -> str:
    cm = OCRCacheManager()
    index = cm.get_case_index(identifier, auto_reconstruct=True)
    if index:
        return identifier

    normalized = _normalize(identifier)
    for candidate in cm.list_cached_cases():
        for key in (candidate.get("case_id"), candidate.get("case_title"), candidate.get("folder_path")):
            if not key:
                continue
            key_str = str(key)
            options = {key_str}
            try:
                options.add(Path(key_str).name)
            except Exception:
                pass
            if any(_normalize(opt) == normalized for opt in options):
                return candidate["case_id"]
    return identifier


if __name__ == "__main__":
    main()
