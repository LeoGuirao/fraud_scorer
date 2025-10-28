#!/usr/bin/env python3
"""Reprocesa reportes GPS y persiste los artefactos en el caché del caso."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_src_to_path() -> None:
    project_root = _resolve_project_root()
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reprocesa PDFs de monitoreo GPS y sobreescribe los artefactos almacenados."
    )
    parser.add_argument(
        "--case",
        required=True,
        help="ID del caso destino (ej. CASE-2025-0001)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directorio que contiene los PDFs a reprocesar",
    )
    parser.add_argument(
        "--glob",
        default="*.pdf",
        help="Patrón glob para seleccionar archivos (por defecto *.pdf)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo listar archivos que se reprocesarían sin ejecutar la ingesta",
    )
    return parser.parse_args()


def main() -> int:
    _add_src_to_path()

    from fraud_scorer.parsers.gps_direct_extractor import GPSDirectExtractor
    from fraud_scorer.parsers.processing_hint import ProcessingHintBuilder
    from fraud_scorer.storage.gps_cache import GPSCacheManager

    args = parse_args()
    case_id: str = args.case
    input_dir: Path = args.input_dir
    pattern: str = args.glob

    if not input_dir.exists():
        print(f"[WARN] El directorio {input_dir} no existe.", file=sys.stderr)
        return 1

    files = sorted(input_dir.glob(pattern))
    if not files:
        print(f"[WARN] No se encontraron archivos con patrón {pattern} en {input_dir}.")
        return 1

    if args.dry_run:
        print("Archivos que se reprocesarían:")
        for path in files:
            print(f"  - {path.name}")
        return 0

    extractor = GPSDirectExtractor()
    hint_builder = ProcessingHintBuilder()
    cache = GPSCacheManager()

    successes = 0
    for path in files:
        try:
            hint = hint_builder.build(path, manual_override=True)
            parsed = extractor.extract(path, hint)
            cache.persist_direct_output(case_id, path, parsed)
            rows = parsed["metadata"]["gps_direct"]["normalized_row_count"]
            print(f"[OK] {path.name} → {rows} filas normalizadas")
            successes += 1
        except Exception as exc:  # pragma: no cover - utilitario CLI
            print(f"[ERROR] Falló {path.name}: {exc}", file=sys.stderr)

    print(f"Documentos procesados correctamente: {successes}/{len(files)}")
    return 0 if successes == len(files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
