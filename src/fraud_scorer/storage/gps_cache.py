"""Storage helpers for GPS direct artifacts (datasets, summaries, metadata)."""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fraud_scorer.parsers.gps_direct_extractor import (
    GPS_SCHEMA_VERSION,
    RAW_PAYLOAD_COLUMN,
    compute_sha256_for_paths,
    normalize_gps_tables,
)

try:  # pragma: no cover - métricas opcionales
    from prometheus_client import Counter, Histogram

    GPS_INGEST_COUNTER = Counter(
        "gps_ingest_documents_total",
        "Total de artefactos GPS persistidos",
        labelnames=("operation",),
    )
    GPS_INGEST_ROWS = Counter(
        "gps_ingest_rows_total",
        "Total de filas normalizadas en la ingesta",
        labelnames=("operation",),
    )
    GPS_INGEST_LATENCY = Histogram(
        "gps_ingest_latency_seconds",
        "Latencia para persistir artefactos GPS",
        labelnames=("operation",),
        buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
    )
except Exception:  # pragma: no cover - fallback silencioso

    class _Noop:  # pylint: disable=too-few-public-methods
        def labels(self, *args, **kwargs):  # noqa: D401
            return self

        def inc(self, *_args, **_kwargs) -> None:
            return None

        def observe(self, *_args, **_kwargs) -> None:
            return None

    GPS_INGEST_COUNTER = GPS_INGEST_ROWS = GPS_INGEST_LATENCY = _Noop()


logger = logging.getLogger(__name__)


class GPSCacheManager:
    """Persists structured GPS datasets (Parquet + metadata) per case/document."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir or Path("data") / "gps")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def persist_direct_output(
        self,
        case_id: Optional[str],
        document_path: Path,
        parsed_document: Dict[str, Any],
    ) -> None:
        if not case_id:
            return

        start = time.perf_counter()
        operation = "direct"

        metadata = parsed_document.get("metadata") or {}
        gps_meta = metadata.get("gps_direct") or {}
        if not gps_meta.get("enabled"):
            return

        doc_dir = self._document_dir(case_id, document_path)
        doc_dir.mkdir(parents=True, exist_ok=True)

        text_content = parsed_document.get("text") or ""
        self._write_text(doc_dir / "raw_text.txt", text_content)

        tables = parsed_document.get("tables") or []
        df, normalization_warnings = normalize_gps_tables(tables)

        schema_version = gps_meta.get("schema_version", GPS_SCHEMA_VERSION)
        dataset_info: Optional[Dict[str, Any]] = None
        tables_manifest: List[Dict[str, Any]] = []

        if not df.empty:
            dataset_info, tables_manifest = self._write_dataset(
                doc_dir=doc_dir,
                dataframe=df,
                schema_version=schema_version,
            )
        else:
            logger.info("Dataset GPS vacío para %s/%s; sólo se guardará resumen", case_id, document_path.name)

        summary = dict(parsed_document.get("gps_summary") or {})
        merged_warnings = sorted(
            set((summary.get("warnings") or []) + list(normalization_warnings))
        )
        summary["warnings"] = merged_warnings
        self._write_json(doc_dir / "gps_summary.json", summary)

        gps_meta.setdefault("normalization_warnings", [])
        gps_meta["normalization_warnings"] = sorted(
            set(gps_meta.get("normalization_warnings", []) + list(normalization_warnings))
        )
        if dataset_info:
            gps_meta["dataset"] = dataset_info
            gps_meta.setdefault("chunk_count", dataset_info.get("partitions") and len(dataset_info["partitions"]))
        metadata["gps_direct"] = gps_meta
        parsed_document["metadata"] = metadata

        entry = {
            "document_name": document_path.name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "hint": gps_meta.get("hint"),
            "summary": summary,
            "dataset": dataset_info,
            "preview_rows": gps_meta.get("preview_rows") or [],
            "normalization_warnings": gps_meta.get("normalization_warnings", []),
            "ingestion_stats": gps_meta.get("ingestion_stats") or {},
            "tables": tables_manifest,
            "chunk_count": gps_meta.get("chunk_count"),
        }
        self._write_json(doc_dir / "_metadata.json", entry)
        self._update_case_manifest(case_id, document_path.name, entry)

        try:
            GPS_INGEST_COUNTER.labels(operation=operation).inc()
            rows = int((dataset_info or {}).get("row_count", 0))
            GPS_INGEST_ROWS.labels(operation=operation).inc(rows)
            GPS_INGEST_LATENCY.labels(operation=operation).observe(time.perf_counter() - start)
        except Exception:  # pragma: no cover - métricas best effort
            logger.debug("No se pudieron actualizar métricas de ingesta para %s", document_path.name)

    def attach_to_case_index(self, case_id: str, case_data: Dict[str, Any]) -> None:
        manifest = self._load_case_manifest(case_id)
        if not manifest:
            return
        case_data.setdefault("gps_direct_documents", {})
        case_data["gps_direct_documents"].update(manifest)

    def get_manifest(self, case_id: str) -> Dict[str, Any]:
        return self._load_case_manifest(case_id)

    def load_table(self, case_id: str, document_name: str, table_id: int) -> pd.DataFrame:
        manifest = self._load_case_manifest(case_id)
        entry = manifest.get(document_name)
        if not entry:
            raise FileNotFoundError(f"No existe metadata GPS para {document_name}")

        table_meta = None
        for item in entry.get("tables", []):
            if int(item.get("table_id", -1)) == table_id:
                table_meta = item
                break
        if not table_meta:
            raise FileNotFoundError(f"Tabla {table_id} no registrada para {document_name}")

        table_path = self._resolve_table_path(case_id, document_name, table_meta["file"])
        if not table_path.exists():
            raise FileNotFoundError(f"No se encontró archivo {table_path}")
        return pd.read_parquet(table_path)

    def load_text(self, case_id: str, document_name: str) -> str:
        doc_dir = self._document_dir(case_id, Path(document_name))
        text_path = doc_dir / "raw_text.txt"
        if not text_path.exists():
            return ""
        return text_path.read_text(encoding="utf-8")

    def load_dataset(self, case_id: str, document_name: str) -> pd.DataFrame:
        manifest = self._load_case_manifest(case_id)
        entry = manifest.get(document_name)
        if not entry:
            raise FileNotFoundError(f"No existe metadata GPS para {document_name}")
        dataset_meta = entry.get("dataset") or {}
        partitions = dataset_meta.get("partitions") or []
        if not partitions:
            raise FileNotFoundError("El documento no tiene dataset registrado")
        frames = []
        for partition in partitions:
            file_path = self._resolve_table_path(case_id, document_name, partition["file"])
            if not file_path.exists():
                continue
            frames.append(pd.read_parquet(file_path))
        if not frames:
            columns = dataset_meta.get("columns") or []
            if RAW_PAYLOAD_COLUMN not in columns:
                columns.append(RAW_PAYLOAD_COLUMN)
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)

    def get_table_path(self, case_id: str, document_name: str, table_id: int) -> Path:
        manifest = self._load_case_manifest(case_id)
        entry = manifest.get(document_name)
        if not entry:
            raise FileNotFoundError(f"No existe metadata GPS para {document_name}")
        for item in entry.get("tables", []):
            if int(item.get("table_id", -1)) == table_id:
                return self._resolve_table_path(case_id, document_name, item["file"])
        raise FileNotFoundError(f"Tabla {table_id} no registrada para {document_name}")

    def collect_global_metrics(
        self,
        *,
        top_documents: int = 5,
        recent_limit: int = 5,
    ) -> Dict[str, Any]:
        totals = {
            "cases": 0,
            "documents": 0,
            "rows": 0,
        }
        chunked_documents = 0
        warned_documents = 0
        recent_candidates: List[Tuple[datetime, str, str, int]] = []
        top_candidates: List[Tuple[int, str, str]] = []

        for case_dir in sorted(self.base_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            manifest = self._load_case_manifest(case_id)
            if not manifest:
                continue
            totals["cases"] += 1

            for doc_name, entry in manifest.items():
                totals["documents"] += 1
                dataset = entry.get("dataset") or {}
                row_count = int(dataset.get("row_count") or 0)
                totals["rows"] += row_count

                chunk_count = entry.get("chunk_count")
                if chunk_count is None:
                    chunk_count = (entry.get("ingestion_stats") or {}).get("table_count")
                if chunk_count and int(chunk_count) > 1:
                    chunked_documents += 1

                warnings = set(entry.get("normalization_warnings") or [])
                summary_warnings = entry.get("summary", {}).get("warnings") or []
                warnings.update(summary_warnings)
                if warnings:
                    warned_documents += 1

                saved_at = entry.get("saved_at")
                saved_dt = _safe_parse_iso(saved_at)
                if saved_dt:
                    recent_candidates.append((saved_dt, case_id, doc_name, row_count))

                top_candidates.append((row_count, case_id, doc_name))

        average_rows = (totals["rows"] / totals["documents"]) if totals["documents"] else 0.0
        recent_documents = [
            {
                "case_id": case_id,
                "document_name": doc_name,
                "row_count": rows,
                "saved_at": recent_dt.isoformat(),
            }
            for recent_dt, case_id, doc_name, rows in sorted(
                recent_candidates, key=lambda item: item[0], reverse=True
            )[:recent_limit]
        ]

        top_documents_list = [
            {
                "case_id": case_id,
                "document_name": doc_name,
                "row_count": rows,
            }
            for rows, case_id, doc_name in sorted(
                top_candidates, key=lambda item: item[0], reverse=True
            )[:top_documents]
        ]

        latest_ingestion = max((item[0] for item in recent_candidates), default=None)

        return {
            "totals": {
                "cases": totals["cases"],
                "documents": totals["documents"],
                "rows": totals["rows"],
                "avg_rows_per_document": average_rows,
                "chunked_documents": chunked_documents,
                "documents_with_warnings": warned_documents,
                "last_ingestion_at": latest_ingestion.isoformat() if latest_ingestion else None,
            },
            "recent_documents": recent_documents,
            "top_documents": top_documents_list,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _document_dir(self, case_id: str, document_path: Path) -> Path:
        case_dir = self.base_dir / self._sanitize(case_id)
        doc_dir = case_dir / self._sanitize(document_path.stem)
        case_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir

    def _write_dataset(
        self,
        *,
        doc_dir: Path,
        dataframe: pd.DataFrame,
        schema_version: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        dataset_dir = doc_dir / "dataset"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir, ignore_errors=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        df = dataframe.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        compression = "zstd"
        partitions = self._partition_dataframe(df)

        partition_manifest: List[Dict[str, Any]] = []
        written_files: List[Path] = []
        total_rows = 0
        total_bytes = 0

        for idx, (partition_key, partition_df) in enumerate(partitions, start=1):
            relative_dir = dataset_dir / partition_key
            relative_dir.mkdir(parents=True, exist_ok=True)
            file_path = relative_dir / "gps_data.parquet"
            try:
                self._write_parquet(partition_df, file_path, compression=compression)
            except Exception as exc:
                if compression != "gzip":
                    compression = "gzip"
                    logger.warning(
                        "Fallo al escribir con zstd (%s); reintentando con gzip", exc
                    )
                    self._write_parquet(partition_df, file_path, compression=compression)
                else:
                    raise
            file_size = file_path.stat().st_size
            checksum = compute_sha256_for_paths([file_path])
            total_rows += int(partition_df.shape[0])
            total_bytes += file_size
            written_files.append(file_path)
            partition_manifest.append(
                {
                    "table_id": idx,
                    "file": str(file_path.relative_to(doc_dir)),
                    "row_count": int(partition_df.shape[0]),
                    "checksum": checksum,
                    "size_bytes": file_size,
                    "partition_key": partition_key,
                }
            )

        dataset_checksum = compute_sha256_for_paths(written_files)
        dataset_info = {
            "schema_version": schema_version,
            "row_count": total_rows,
            "compression": compression,
            "partitions": partition_manifest,
            "total_size_bytes": total_bytes,
            "checksum": dataset_checksum,
            "columns": list(df.columns),
        }
        return dataset_info, partition_manifest

    def _write_parquet(self, dataframe: pd.DataFrame, file_path: Path, *, compression: str) -> None:
        table = pa.Table.from_pandas(dataframe, preserve_index=False)
        pq.write_table(table, file_path, compression=compression)

    def _partition_dataframe(self, dataframe: pd.DataFrame) -> Iterable[Tuple[str, pd.DataFrame]]:
        if "timestamp" not in dataframe.columns or dataframe["timestamp"].isna().all():
            yield ("full", dataframe)
            return

        df = dataframe.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["partition_year"] = df["timestamp"].dt.year.fillna(0).astype(int)
        df["partition_month"] = df["timestamp"].dt.month.fillna(0).astype(int)

        grouped = df.groupby(["partition_year", "partition_month"], dropna=False, sort=True)
        for (year, month), group in grouped:
            key = f"year={int(year):04d}/month={int(month):02d}" if year and month else "full"
            yield key, group.drop(columns=["partition_year", "partition_month"], errors="ignore")

    def _write_text(self, path: Path, content: str) -> None:
        try:
            path.write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.warning("No se pudo guardar texto GPS (%s): %s", path, exc)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        try:
            import orjson

            path.write_text(
                orjson.dumps(payload, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2).decode("utf-8"),
                encoding="utf-8",
            )
        except Exception:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _update_case_manifest(self, case_id: str, document_name: str, entry: Dict[str, Any]) -> None:
        manifest_path = self.base_dir / self._sanitize(case_id) / "_manifest.json"
        data: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[document_name] = entry
        self._write_json(manifest_path, data)

    def _load_case_manifest(self, case_id: str) -> Dict[str, Any]:
        manifest_path = self.base_dir / self._sanitize(case_id) / "_manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("No se pudo leer manifest GPS %s: %s", case_id, exc)
            return {}

    def _resolve_table_path(self, case_id: str, document_name: str, relative_file: str) -> Path:
        return self._document_dir(case_id, Path(document_name)) / relative_file

    def _sanitize(self, value: str) -> str:
        if not value:
            return "SIN_VALOR"
        import re

        sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
        return sanitized.strip("_") or "SIN_VALOR"


def _safe_parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except Exception:
        return None


__all__ = ["GPSCacheManager"]
