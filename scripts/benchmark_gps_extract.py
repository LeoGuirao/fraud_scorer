#!/usr/bin/env python3
"""Micro-benchmark for GPS direct ingestion.

Generates a synthetic dataset (CSV) and measures extraction + persistence time.
This helps validate chunking and compression strategies before running full
load tests in staging, following the guidance in BETTER_PRACTICES.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional
import random
import math

from fraud_scorer.parsers.gps_direct_extractor import GPSDirectExtractor
from fraud_scorer.parsers.processing_hint import ProcessingHint
from fraud_scorer.storage.gps_cache import GPSCacheManager


def _generate_dataset(path: Path, rows: int) -> None:
    rng = random.Random(42)
    with path.open("w", encoding="utf-8") as handler:
        handler.write("timestamp,latitude,longitude,speed,event\n")
        for index in range(rows):
            hour = index // 60
            minute = index % 60
            timestamp = f"2025-01-01T{hour:02d}:{minute:02d}:00Z"
            latitude = 19.0 + math.sin(index / 180) * 0.5
            longitude = -99.0 + math.cos(index / 180) * 0.5
            speed = 60 + rng.randint(-15, 15)
            event = "OK" if index % 45 else "STOP"
            handler.write(f"{timestamp},{latitude:.6f},{longitude:.6f},{speed},{event}\n")


def run_benchmark(
    *,
    rows: int,
    chunk_rows: int,
    output_dir: Path,
    persist: bool,
    max_file_mb: float,
) -> None:
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / "gps_benchmark.csv"

    print(f"📦 Generando dataset sintético ({rows} filas)...")
    _generate_dataset(csv_path, rows)

    extractor = GPSDirectExtractor(
        csv_chunk_rows=chunk_rows,
        excel_chunk_rows=chunk_rows,
        max_file_mb=max_file_mb,
    )
    hint = ProcessingHint(
        file_name=csv_path.name,
        file_extension=".csv",
        mime_type="text/csv",
        file_size_bytes=csv_path.stat().st_size,
        manual_override=True,
        is_gps_candidate=True,
        confidence=1.0,
    )

    print("🚀 Ejecutando GPSDirectExtractor...")
    start = time.perf_counter()
    try:
        parsed = extractor.extract(csv_path, hint=hint)
    except ValueError as exc:
        print(f"❌ Error durante la extracción: {exc}")
        print("👉 Ajusta el parámetro --max-file-mb o reduce --rows antes de reintentar.")
        return
    elapsed_extraction = time.perf_counter() - start

    print(
        f"   → Filas normalizadas: {parsed['metadata']['gps_direct']['normalized_row_count']}"
        f" · chunk_count={parsed['metadata']['gps_direct'].get('chunk_count')}"
    )

    if persist:
        cache_dir = output_dir / "gps"
        manager = GPSCacheManager(base_dir=cache_dir)
        print("💾 Persistiendo artefactos Parquet...")
        start = time.perf_counter()
        manager.persist_direct_output("BENCHMARK-CASE", csv_path, parsed)
        elapsed_persist = time.perf_counter() - start
        dataset = manager.get_manifest("BENCHMARK-CASE")[csv_path.name]["dataset"]
        total_size = dataset.get("total_size_bytes")
        print(
            f"   → Persistido en {elapsed_persist:.2f}s · "
            f"particiones={len(dataset.get('partitions', []))} · tamaño={total_size} bytes"
        )
    else:
        elapsed_persist = None

    print(
        f"⏱️  Tiempo de extracción: {elapsed_extraction:.2f}s"
        + (f" | Persistencia: {elapsed_persist:.2f}s" if elapsed_persist is not None else "")
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark GPS direct ingestion")
    parser.add_argument("output", type=Path, help="Directorio donde escribir resultados")
    parser.add_argument("--rows", type=int, default=200_000, help="Número de filas sintéticas a generar")
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=200_000,
        help="Tamaño de chunk utilizado por el extractor",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persistir artefactos Parquet para medir escritura y compresión",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=500.0,
        help="Límite máximo permitido por GPSDirectExtractor (MB)",
    )

    args = parser.parse_args(argv)
    run_benchmark(
        rows=args.rows,
        chunk_rows=args.chunk_rows,
        output_dir=args.output,
        persist=args.persist,
        max_file_mb=args.max_file_mb,
    )


if __name__ == "__main__":
    main()
