"""Utilidades de mantenimiento para los índices del Agente Rick."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .manager import RickVectorStoreManager


def list_indexed_cases(manager: RickVectorStoreManager | None = None) -> List[Dict[str, object]]:
    mgr = manager or RickVectorStoreManager()
    base = mgr.config.chroma_base_path
    results: List[Dict[str, object]] = []

    if not base.exists():
        return results

    for path in base.iterdir():
        if not path.is_dir() or path.name.startswith("_"):
            continue
        manifest = path / "index_manifest.json"
        if manifest.exists():
            results.append({
                "case_id": path.name,
                "manifest_path": str(manifest),
                "size_bytes": _dir_size(path),
            })
    return results


def cleanup_missing_cases(existing_case_ids: List[str], manager: RickVectorStoreManager | None = None) -> List[str]:
    mgr = manager or RickVectorStoreManager()
    removed: List[str] = []
    base = mgr.config.chroma_base_path

    if not base.exists():
        return removed

    keep = set(existing_case_ids)
    for path in base.iterdir():
        if not path.is_dir() or path.name.startswith("_"):
            continue
        if path.name not in keep:
            mgr.delete_case_index(path.name)
            removed.append(path.name)
    return removed


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


__all__ = ["list_indexed_cases", "cleanup_missing_cases"]
