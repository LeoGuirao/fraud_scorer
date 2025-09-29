"""Métricas y utilidades de observabilidad para el motor de correlación."""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from fraud_scorer.analyzers.correlation.models import CorrelationReport, FindingStatus

try:  # pragma: no cover - dependemos de la instalación en tiempo de ejecución
    from prometheus_client import Counter, Gauge, Histogram
except Exception:  # pragma: no cover - fallback para entornos sin prometheus_client
    class _NoopMetric:
        def __init__(self, *_, **__):
            self._value = 0

        def inc(self, amount: float = 1.0):
            self._value += amount

        def observe(self, value: float):
            self._value = value

        def set(self, value: float):
            self._value = value

        def labels(self, *_, **__):
            return self

    Counter = Gauge = Histogram = _NoopMetric  # type: ignore


logger = logging.getLogger(__name__)

_CORRELATION_REPORTS_TOTAL = Counter(
    "fraud_correlation_reports_total",
    "Total de reportes de correlación generados",
)

_CORRELATION_FINDINGS_PER_REPORT = Histogram(
    "fraud_correlation_findings_per_report",
    "Distribución de hallazgos por reporte",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55),
)

_CORRELATION_NEEDS_CONTEXT_RATIO = Gauge(
    "fraud_correlation_needs_context_ratio",
    "Proporción de hallazgos que requieren contexto adicional",
)

_CORRELATION_RAG_LATENCY_MS = Histogram(
    "fraud_correlation_rag_latency_ms",
    "Latencia del Agente Rick para consultas de correlación (ms)",
    buckets=(10, 25, 50, 75, 100, 150, 250, 400, 600, 1000, 2000, 4000),
)

_CORRELATION_RAG_QUERIES = Counter(
    "fraud_correlation_rag_queries_total",
    "Total de consultas RAG realizadas por estado",
    ["status"],
)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / float(denominator)


def record_report(report: CorrelationReport) -> None:
    """Registra métricas derivadas de un reporte de correlación."""
    total_findings = len(report.findings or [])
    needs_context = report.status_counts.get(FindingStatus.NEEDS_CONTEXT, 0)
    fails = report.status_counts.get(FindingStatus.FAIL, 0)

    _CORRELATION_REPORTS_TOTAL.inc()
    _CORRELATION_FINDINGS_PER_REPORT.observe(float(total_findings))
    _CORRELATION_NEEDS_CONTEXT_RATIO.set(_safe_ratio(needs_context, total_findings))

    report.metadata.setdefault("metrics", {})
    report.metadata["metrics"].update(
        {
            "total_findings": total_findings,
            "needs_context_ratio": _safe_ratio(needs_context, total_findings),
            "fail_ratio": _safe_ratio(fails, total_findings),
        }
    )


def record_rag_event(status: str, latency_ms: Optional[float]) -> None:
    """Expone métricas por cada consulta RAG ejecutada."""
    normalized_status = (status or "unknown").lower()
    try:
        _CORRELATION_RAG_QUERIES.labels(normalized_status).inc()
        if latency_ms is not None:
            _CORRELATION_RAG_LATENCY_MS.observe(float(latency_ms))
    except Exception:  # pragma: no cover - defensivo en caso de fallas del backend de métricas
        logger.debug("No se pudieron registrar métricas RAG", exc_info=True)
