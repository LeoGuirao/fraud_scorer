"""Query helpers for GPS direct ingestion artifacts."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from fraud_scorer.parsers.gps_direct_extractor import build_gps_summary
from fraud_scorer.storage.gps_cache import GPSCacheManager


try:  # pragma: no cover - métricas opcionales
    from prometheus_client import Counter, Histogram

    GPS_QUERY_COUNTER = Counter(
        "gps_query_total",
        "Total de consultas al dataset GPS",
        labelnames=("operation",),
    )
    GPS_QUERY_LATENCY = Histogram(
        "gps_query_latency_seconds",
        "Latencia de consultas GPS",
        labelnames=("operation",),
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    )
except Exception:  # pragma: no cover - fallback

    class _Noop:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *_: Any, **__: Any) -> None:
            return None

        def observe(self, *_: Any, **__: Any) -> None:
            return None

    GPS_QUERY_COUNTER = _Noop()
    GPS_QUERY_LATENCY = _Noop()


class GPSDirectQueryService:
    """Provides a higher level API to inspect GPS parquet/text artifacts."""

    def __init__(self, cache_manager: Optional[GPSCacheManager] = None) -> None:
        self.cache = cache_manager or GPSCacheManager()

    def list_documents(self, case_id: str) -> List[Dict[str, Any]]:
        manifest = self.cache.get_manifest(case_id)
        results: List[Dict[str, Any]] = []
        for doc_name, entry in manifest.items():
            results.append(
                {
                    "document_name": doc_name,
                    "tables": entry.get("tables", []),
                    "summary": entry.get("summary", {}),
                    "saved_at": entry.get("saved_at"),
                    "hint": entry.get("hint"),
                    "dataset": entry.get("dataset"),
                }
            )
        return results

    def get_summary(self, case_id: str, document_name: str) -> Dict[str, Any]:
        manifest = self.cache.get_manifest(case_id)
        return manifest.get(document_name, {}).get("summary", {})

    def get_metadata(self, case_id: str, document_name: str) -> Dict[str, Any]:
        manifest = self.cache.get_manifest(case_id)
        entry = manifest.get(document_name)
        if not entry:
            raise FileNotFoundError(f"No se encontró metadata GPS para {document_name}")
        return {
            "document_name": document_name,
            "summary": entry.get("summary", {}),
            "dataset": entry.get("dataset", {}),
            "preview_rows": entry.get("preview_rows", []),
            "ingestion_stats": entry.get("ingestion_stats", {}),
            "hint": entry.get("hint"),
            "normalization_warnings": entry.get("normalization_warnings", []),
        }

    def query_table(self, case_id: str, document_name: str, table_id: int) -> Dict[str, Any]:
        start = time.perf_counter()
        df = self.cache.load_table(case_id, document_name, table_id)
        GPS_QUERY_COUNTER.labels(operation="table").inc()
        GPS_QUERY_LATENCY.labels(operation="table").observe(time.perf_counter() - start)
        preview = _serialize_preview(df.head(20))
        return {
            "columns": list(df.columns),
            "row_count": int(df.shape[0]),
            "preview": preview,
        }

    def query_dataset(
        self,
        case_id: str,
        document_name: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_labels: Optional[Sequence[str]] = None,
        bounding_box: Optional[Dict[str, float]] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        df = self.cache.load_dataset(case_id, document_name)

        filtered = self._apply_filters(df, start_time, end_time, event_labels, bounding_box)
        total_rows = int(filtered.shape[0])
        preview = _serialize_preview(filtered.head(limit))

        summary = build_gps_summary(filtered, page_stats=None, extra_warnings=None)
        result = {
            "row_count": total_rows,
            "preview": preview,
            "summary": summary,
        }

        GPS_QUERY_COUNTER.labels(operation="dataset").inc()
        GPS_QUERY_LATENCY.labels(operation="dataset").observe(time.perf_counter() - start)
        return result

    def get_raw_text(self, case_id: str, document_name: str) -> str:
        return self.cache.load_text(case_id, document_name)

    def global_metrics(
        self,
        *,
        top_documents: int = 5,
        recent_limit: int = 5,
    ) -> Dict[str, Any]:
        return self.cache.collect_global_metrics(
            top_documents=top_documents,
            recent_limit=recent_limit,
        )

    def segment_dataset(
        self,
        case_id: str,
        document_name: str,
        *,
        frequency: str = "60min",
        metrics: Sequence[str] = ("row_count", "avg_speed", "event_distribution"),
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_labels: Optional[Sequence[str]] = None,
        bounding_box: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        df = self.cache.load_dataset(case_id, document_name)
        filtered = self._apply_filters(df, start_time, end_time, event_labels, bounding_box)
        if "timestamp" not in filtered.columns:
            raise ValueError("El dataset no contiene columna timestamp para segmentación")

        if filtered.empty:
            return {"frequency": frequency, "segments": []}

        filtered = filtered.copy()
        filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], errors="coerce")
        filtered.dropna(subset=["timestamp"], inplace=True)
        if filtered.empty:
            return {"frequency": frequency, "segments": []}

        filtered.sort_values("timestamp", inplace=True)
        segments: List[Dict[str, Any]] = []

        grouped = filtered.set_index("timestamp").resample(frequency)
        for window_start, group in grouped:
            if group.empty:
                continue

            segment: Dict[str, Any] = {
                "start": window_start.to_pydatetime().isoformat(),
                "end": group.index.max().to_pydatetime().isoformat(),
            }

            if "row_count" in metrics:
                segment["row_count"] = int(group.shape[0])

            if "avg_speed" in metrics and "speed" in group.columns:
                speed_series = pd.to_numeric(group["speed"], errors="coerce").dropna()
                if not speed_series.empty:
                    segment["avg_speed"] = float(speed_series.mean())

            if "event_distribution" in metrics and "event_label" in group.columns:
                distribution = group["event_label"].value_counts().to_dict()
                segment["event_distribution"] = distribution

            segments.append(segment)

        return {"frequency": frequency, "segments": segments}

    def detect_anomalies(
        self,
        case_id: str,
        document_name: str,
        *,
        gap_threshold_minutes: int = 60,
        speed_threshold: Optional[float] = None,
        max_samples: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_labels: Optional[Sequence[str]] = None,
        bounding_box: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        df = self.cache.load_dataset(case_id, document_name)
        filtered = self._apply_filters(df, start_time, end_time, event_labels, bounding_box)
        filtered = filtered.copy()
        filtered["timestamp"] = pd.to_datetime(filtered.get("timestamp"), errors="coerce")
        filtered.sort_values("timestamp", inplace=True)

        summary = build_gps_summary(filtered, page_stats=None, extra_warnings=None)
        anomalies: List[Dict[str, Any]] = []

        if gap_threshold_minutes > 0:
            for gap in summary.get("time_gaps", []):
                if gap.get("gap_minutes", 0) >= gap_threshold_minutes:
                    anomalies.append({"type": "time_gap", **gap})

        coord_missing = filtered[filtered["latitude"].isna() | filtered["longitude"].isna()] if "latitude" in filtered.columns and "longitude" in filtered.columns else pd.DataFrame()
        if not coord_missing.empty:
            anomalies.append(
                {
                    "type": "missing_coordinates",
                    "count": int(coord_missing.shape[0]),
                    "sample": _serialize_preview(coord_missing.head(min(max_samples, 10))),
                }
            )

        if speed_threshold is not None and "speed" in filtered.columns:
            speed_series = pd.to_numeric(filtered["speed"], errors="coerce")
            high_speed_mask = speed_series > speed_threshold
            if high_speed_mask.any():
                high_speed_rows = filtered.loc[high_speed_mask].head(max_samples)
                anomalies.append(
                    {
                        "type": "high_speed",
                        "threshold": float(speed_threshold),
                        "count": int(high_speed_mask.sum()),
                        "sample": _serialize_preview(high_speed_rows),
                    }
                )

        return {"anomalies": anomalies, "summary": summary}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _apply_filters(
        self,
        dataframe: pd.DataFrame,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        event_labels: Optional[Sequence[str]],
        bounding_box: Optional[Dict[str, float]],
    ) -> pd.DataFrame:
        df = dataframe.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if start_time:
                df = df[df["timestamp"] >= pd.Timestamp(start_time)]
            if end_time:
                df = df[df["timestamp"] <= pd.Timestamp(end_time)]

        if event_labels:
            df = df[df["event_label"].isin(event_labels)]

        if bounding_box:
            min_lat = bounding_box.get("min_lat")
            max_lat = bounding_box.get("max_lat")
            min_lon = bounding_box.get("min_lon")
            max_lon = bounding_box.get("max_lon")
            if min_lat is not None and max_lat is not None and "latitude" in df.columns:
                df = df[df["latitude"].between(min_lat, max_lat, inclusive="both")]
            if min_lon is not None and max_lon is not None and "longitude" in df.columns:
                df = df[df["longitude"].between(min_lon, max_lon, inclusive="both")]

        return df


def _serialize_preview(df: pd.DataFrame) -> List[Dict[str, Any]]:
    safe_df = df.where(pd.notnull(df), None)
    if "timestamp" in safe_df.columns:
        safe_df["timestamp"] = safe_df["timestamp"].apply(
            lambda ts: ts.isoformat() if hasattr(ts, "isoformat") else ts
        )
    return safe_df.to_dict(orient="records")


__all__ = ["GPSDirectQueryService"]
