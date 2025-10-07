#!/usr/bin/env python3
"""Quick verification script for GPS direct ingestion artifacts.

Usage
-----

    python scripts/check_gps_artifacts.py --case CASE-2025-0001

The script lists all GPS datasets for the case, validates that expected parquet
files exist and prints a short summary with detected headers and row counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from fraud_scorer.storage.gps_cache import GPSCacheManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifica artefactos GPS direct access")
    parser.add_argument("--case", required=True, help="ID del caso a inspeccionar")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache = GPSCacheManager()
    manifest = cache.get_manifest(args.case)
    if not manifest:
        print(f"No se encontraron artefactos GPS para el caso {args.case}")
        return 0

    print(f"Artefactos GPS para {args.case}:")
    for doc_name, entry in manifest.items():
        summary = entry.get("summary") or {}
        tables = entry.get("tables") or []
        print(f"\n- Documento: {doc_name}")
        if summary:
            headers = summary.get("table_headers") or []
            print(f"  Resumen: {summary}")
            if headers:
                print(f"  Headers detectados: {headers}")
        if not tables:
            print("  (sin tablas registradas)")
            continue
        for table in tables:
            table_id = int(table.get("table_id")) if table.get("table_id") is not None else -1
            table_file = table.get("file")
            row_count = table.get("row_count")
            try:
                cache.get_table_path(args.case, doc_name, table_id)
                status = "OK"
            except FileNotFoundError:
                status = "FALTANTE"
            except Exception as exc:  # pragma: no cover - debugging aid
                status = f"ERROR({exc})"
            print(f"  Tabla #{table_id}: filas={row_count} archivo={table_file} [{status}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
