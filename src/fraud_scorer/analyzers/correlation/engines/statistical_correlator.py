"""Correlador estadístico para detectar anomalías numéricas y temporales."""
from __future__ import annotations

import logging
import math
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import yaml

from fraud_scorer.analyzers.correlation.models import (
    CaseContext,
    CorrelationFinding,
    FindingSeverity,
    FindingStatus,
)
from fraud_scorer.analyzers.correlation.utils import (
    is_missing,
    normalize_date,
    normalize_decimal,
)

logger = logging.getLogger(__name__)


class StatisticalCorrelator:
    """Evalúa reglas estadísticas definidas en YAML para detectar anomalías."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        self.config_path = config_path or (base_dir / "rules" / "statistical_config.yaml")
        self.config: Dict[str, Any] = self._load_yaml(self.config_path) or {}

    def analyze(self, context: CaseContext) -> List[CorrelationFinding]:
        findings: List[CorrelationFinding] = []
        findings.extend(self._analyze_numeric_anomalies(context))
        findings.extend(self._analyze_gap_anomalies(context))
        findings.extend(self._analyze_correlation_checks(context))
        return findings

    # ------------------------------------------------------------------
    # Numeric anomalies
    # ------------------------------------------------------------------
    def _analyze_numeric_anomalies(self, context: CaseContext) -> List[CorrelationFinding]:
        entries = self.config.get("numeric_anomalies") or []
        results: List[CorrelationFinding] = []
        for entry in entries:
            try:
                if entry.get("document_type") and entry.get("field"):
                    finding = self._detect_outliers(context, entry)
                    if finding:
                        results.append(finding)
                elif entry.get("source_path") and entry.get("reference_path"):
                    finding = self._detect_ratio(context, entry)
                    if finding:
                        results.append(finding)
                else:
                    logger.warning("Configuración de anomalía numérica incompleta: %s", entry)
            except Exception as exc:  # pragma: no cover - defensivo para reglas futuras
                logger.warning("Error evaluando anomalía numérica %s: %s", entry.get("id"), exc)
        return results

    def _detect_outliers(self, context: CaseContext, entry: Dict[str, Any]) -> Optional[CorrelationFinding]:
        doc_type = entry.get("document_type")
        field = entry.get("field")
        min_samples = int(entry.get("min_samples", 3))
        max_zscore = float(entry.get("max_zscore", 2.5))

        snapshots = context.documents_by_type.get(doc_type or "") or []
        values: List[tuple[str, float]] = []
        for snapshot in snapshots:
            raw = snapshot.extracted_fields.get(field)
            numeric = self._to_float(raw)
            if numeric is not None:
                name = snapshot.document_name or snapshot.document_id or "documento_desconocido"
                values.append((name, numeric))

        if len(values) < min_samples:
            status = (
                FindingStatus.NOT_APPLICABLE
                if len(values) == 0
                else FindingStatus.INSUFFICIENT_DATA
            )
            return self._build_missing_finding(
                entry,
                summary="Datos insuficientes para evaluar outliers.",
                metadata={
                    "collected_samples": len(values),
                    "required_samples": min_samples,
                },
                documents=[doc_type] if doc_type else [],
                status=status,
            )

        numbers = [val for _, val in values]
        std_dev = pstdev(numbers)
        if math.isclose(std_dev, 0.0, abs_tol=1e-9):
            return None

        avg = mean(numbers)
        outliers = [
            {"document": name, "value": value, "zscore": abs((value - avg) / std_dev)}
            for name, value in values
            if abs((value - avg) / std_dev) > max_zscore
        ]
        if not outliers:
            return None

        metadata = {
            "mean": avg,
            "std_dev": std_dev,
            "threshold": max_zscore,
            "outliers": outliers,
        }
        documents_involved = [item["document"] for item in outliers]
        finding = self._build_finding(
            entry,
            status=FindingStatus.FAIL,
            severity=entry.get("severity"),
            summary=entry.get("description") or "Anomalía estadística detectada",
            metadata=metadata,
            documents=documents_involved,
        )
        finding.finding_type = "statistical"
        return finding

    def _detect_ratio(self, context: CaseContext, entry: Dict[str, Any]) -> Optional[CorrelationFinding]:
        source_value = context.resolve(entry.get("source_path"))
        reference_value = context.resolve(entry.get("reference_path"))
        tolerance = float(entry.get("tolerance", 0.0))
        max_ratio = float(entry.get("max_ratio", 1.0))

        status = self._missing_status(source_value, reference_value)
        if status:
            return self._build_missing_finding(
                entry,
                summary="No fue posible calcular el ratio comparativo.",
                metadata={
                    "source_value": source_value,
                    "reference_value": reference_value,
                },
                status=status,
            )

        src = self._to_float(source_value)
        ref = self._to_float(reference_value)

        if src is None or ref is None or ref <= 0:
            metadata = {
                "source_value": source_value,
                "reference_value": reference_value,
                "parse_error": True,
            }
            return self._build_missing_finding(
                entry,
                summary="No fue posible calcular el ratio comparativo.",
                metadata=metadata,
                status=FindingStatus.INSUFFICIENT_DATA,
            )

        ratio = src / ref
        allowed = max_ratio + tolerance
        metadata = {
            "source_value": src,
            "reference_value": ref,
            "ratio": ratio,
            "allowed_ratio": allowed,
        }

        if ratio <= allowed:
            return None

        finding = self._build_finding(
            entry,
            status=FindingStatus.FAIL,
            severity=entry.get("severity"),
            summary=entry.get("description") or "Ratio fuera de tolerancia",
            metadata=metadata,
        )
        finding.finding_type = "statistical"
        return finding

    # ------------------------------------------------------------------
    # Gap anomalies (tiempos)
    # ------------------------------------------------------------------
    def _analyze_gap_anomalies(self, context: CaseContext) -> List[CorrelationFinding]:
        entries = self.config.get("gap_anomalies") or []
        results: List[CorrelationFinding] = []
        for entry in entries:
            try:
                finding = self._detect_gap(context, entry)
                if finding:
                    results.append(finding)
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Error evaluando gap temporal %s: %s", entry.get("id"), exc)
        return results

    def _detect_gap(self, context: CaseContext, entry: Dict[str, Any]) -> Optional[CorrelationFinding]:
        start_raw = context.resolve(entry.get("start"))
        end_raw = context.resolve(entry.get("end"))
        min_days = entry.get("min_days")
        max_days = entry.get("max_days")

        start_date = self._parse_date(start_raw)
        end_date = self._parse_date(end_raw)

        status = self._missing_status(start_raw, end_raw)
        if status:
            return self._build_missing_finding(
                entry,
                summary="No se pudo calcular la diferencia de días (faltan fechas).",
                metadata={"start": start_raw, "end": end_raw},
                status=status,
            )

        if not start_date or not end_date:
            metadata = {"start": start_raw, "end": end_raw, "parse_error": True}
            return self._build_missing_finding(
                entry,
                summary="No se pudo calcular la diferencia de días (faltan fechas).",
                metadata=metadata,
                status=FindingStatus.INSUFFICIENT_DATA,
            )

        delta = (end_date - start_date).days
        metadata = {"difference_days": delta, "start": start_date.isoformat(), "end": end_date.isoformat()}

        violated = False
        if min_days is not None and delta < int(min_days):
            metadata["expected_min"] = int(min_days)
            violated = True
        if max_days is not None and delta > int(max_days):
            metadata["expected_max"] = int(max_days)
            violated = True

        if not violated:
            return None

        finding = self._build_finding(
            entry,
            status=FindingStatus.FAIL,
            severity=entry.get("severity"),
            summary=entry.get("description") or "Diferencia temporal fuera de rango",
            metadata=metadata,
        )

        finding.finding_type = "statistical"
        return finding

    # ------------------------------------------------------------------
    # Correlation checks
    # ------------------------------------------------------------------
    def _analyze_correlation_checks(self, context: CaseContext) -> List[CorrelationFinding]:
        entries = self.config.get("correlation_checks") or []
        results: List[CorrelationFinding] = []
        for entry in entries:
            try:
                finding = self._evaluate_correlation(context, entry)
                if finding:
                    results.append(finding)
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Error evaluando correlación %s: %s", entry.get("id"), exc)
        return results

    def _evaluate_correlation(
        self, context: CaseContext, entry: Dict[str, Any]
    ) -> Optional[CorrelationFinding]:
        left_spec = entry.get("left") or {}
        right_spec = entry.get("right") or {}
        xs, ys, points = self._collect_correlation_pairs(context, left_spec, right_spec, entry)

        min_samples = int(entry.get("min_samples", 3))
        if len(xs) < min_samples:
            status = (
                FindingStatus.NOT_APPLICABLE
                if len(xs) == 0
                else FindingStatus.INSUFFICIENT_DATA
            )
            return self._build_missing_finding(
                entry,
                summary=entry.get("missing_summary") or "Datos insuficientes para evaluar correlación.",
                metadata={
                    "collected_samples": len(xs),
                    "required_samples": min_samples,
                    "series_preview": points[:3],
                },
                status=status,
            )

        method = str(entry.get("method") or "pearson").lower()
        coefficient = self._compute_correlation(xs, ys, method)
        if coefficient is None:
            return self._build_missing_finding(
                entry,
                summary=entry.get("missing_summary") or "No fue posible calcular la correlación.",
                metadata={
                    "method": method,
                    "series_preview": points[:3],
                },
                status=FindingStatus.INSUFFICIENT_DATA,
            )

        abs_coeff = abs(coefficient)
        violations: List[str] = []

        expected_sign = str(entry.get("expected_sign") or entry.get("direction") or "any").lower()
        if expected_sign in {"positive", "pos"} and coefficient < 0:
            violations.append("unexpected_negative_sign")
        if expected_sign in {"negative", "neg"} and coefficient > 0:
            violations.append("unexpected_positive_sign")

        min_abs = entry.get("min_abs_correlation")
        if min_abs is not None and abs_coeff < float(min_abs):
            violations.append("below_min_abs_correlation")

        max_abs = entry.get("max_abs_correlation")
        if max_abs is not None and abs_coeff > float(max_abs):
            violations.append("above_max_abs_correlation")

        if not violations:
            return None

        metadata = {
            "method": method,
            "coefficient": coefficient,
            "abs_coefficient": abs_coeff,
            "sample_size": len(xs),
            "pairs": points,
            "thresholds": {
                "expected_sign": expected_sign,
                "min_abs_correlation": min_abs,
                "max_abs_correlation": max_abs,
            },
            "violations": violations,
        }

        documents_involved = [p.get("document") for p in points if p.get("document")]
        summary = entry.get("description") or "Correlación fuera de los parámetros establecidos"
        finding = self._build_finding(
            entry,
            status=FindingStatus.FAIL,
            severity=entry.get("severity"),
            summary=summary,
            metadata=metadata,
            documents=[doc for doc in documents_involved if doc],
        )
        finding.finding_type = "statistical"
        return finding

    def _collect_correlation_pairs(
        self,
        context: CaseContext,
        left_spec: Dict[str, Any],
        right_spec: Dict[str, Any],
        entry: Dict[str, Any],
    ) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
        pair_mode = str(entry.get("pair_mode") or "auto").lower()
        left_type = str(left_spec.get("type") or ("document" if left_spec.get("document_type") else "path")).lower()
        right_type = str(right_spec.get("type") or ("document" if right_spec.get("document_type") else "path")).lower()

        if pair_mode in {"document", "auto"} and left_type == right_type == "document":
            doc_type = left_spec.get("document_type") or right_spec.get("document_type")
            if doc_type:
                return self._collect_document_pairs(context, doc_type, left_spec.get("field"), right_spec.get("field"))

        left_values, left_points = self._resolve_series(context, left_spec)
        right_values, right_points = self._resolve_series(context, right_spec)
        limit = min(len(left_values), len(right_values))
        xs = left_values[:limit]
        ys = right_values[:limit]
        points: List[Dict[str, Any]] = []
        for idx in range(limit):
            point: Dict[str, Any] = {
                "index": idx,
                "left": xs[idx],
                "right": ys[idx],
            }
            if idx < len(left_points):
                point.update({f"left_{k}": v for k, v in left_points[idx].items()})
            if idx < len(right_points):
                point.update({f"right_{k}": v for k, v in right_points[idx].items()})
            points.append(point)
        return xs, ys, points

    def _collect_document_pairs(
        self,
        context: CaseContext,
        document_type: str,
        left_field: Optional[str],
        right_field: Optional[str],
    ) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
        xs: List[float] = []
        ys: List[float] = []
        points: List[Dict[str, Any]] = []
        snapshots = context.documents_by_type.get(document_type) or []
        for snapshot in snapshots:
            left_value = snapshot.extracted_fields.get(left_field) if left_field else None
            right_value = snapshot.extracted_fields.get(right_field) if right_field else None
            left_num = self._to_float(left_value)
            right_num = self._to_float(right_value)
            if left_num is None or right_num is None:
                continue
            xs.append(left_num)
            ys.append(right_num)
            points.append(
                {
                    "document": snapshot.document_name or snapshot.document_id,
                    "document_type": snapshot.document_type,
                    "left_field": left_field,
                    "right_field": right_field,
                    "left": left_num,
                    "right": right_num,
                }
            )
        return xs, ys, points

    def _resolve_series(
        self, context: CaseContext, spec: Dict[str, Any]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        series_type = str(spec.get("type") or ("document" if spec.get("document_type") else "path")).lower()
        values: List[float] = []
        points: List[Dict[str, Any]] = []

        if series_type == "document":
            doc_type = spec.get("document_type")
            field = spec.get("field")
            snapshots = context.documents_by_type.get(doc_type or "") or []
            for snapshot in snapshots:
                raw = snapshot.extracted_fields.get(field)
                numeric = self._to_float(raw)
                if numeric is None:
                    continue
                values.append(numeric)
                points.append(
                    {
                        "document": snapshot.document_name or snapshot.document_id,
                        "document_type": snapshot.document_type,
                        "field": field,
                    }
                )
            return values, points

        if series_type == "entity":
            field = spec.get("field")
            entries = context.entities.get(field or "") or []
            for raw in entries:
                numeric = self._to_float(raw)
                if numeric is None:
                    continue
                values.append(numeric)
                points.append({"entity": field})
            return values, points

        path = spec.get("path")
        resolved = context.resolve(path) if path else None
        if isinstance(resolved, list):
            for idx, raw in enumerate(resolved):
                numeric = self._to_float(raw)
                if numeric is None:
                    continue
                values.append(numeric)
                points.append({"path": path, "index": idx})
        else:
            numeric = self._to_float(resolved)
            if numeric is not None:
                values.append(numeric)
                points.append({"path": path})
        return values, points

    def _compute_correlation(self, xs: List[float], ys: List[float], method: str) -> Optional[float]:
        if method == "pearson":
            return self._compute_pearson(xs, ys)
        if method == "spearman":
            return self._compute_spearman(xs, ys)
        if method == "kendall":
            return self._compute_kendall(xs, ys)
        logger.warning("Método de correlación desconocido: %s", method)
        return None

    @staticmethod
    def _compute_pearson(xs: List[float], ys: List[float]) -> Optional[float]:
        if len(xs) != len(ys) or len(xs) < 2:
            return None
        mean_x = mean(xs)
        mean_y = mean(ys)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denom_x = sum((x - mean_x) ** 2 for x in xs)
        denom_y = sum((y - mean_y) ** 2 for y in ys)
        denominator = math.sqrt(denom_x * denom_y)
        if math.isclose(denominator, 0.0, abs_tol=1e-12):
            return None
        return numerator / denominator

    def _compute_spearman(self, xs: List[float], ys: List[float]) -> Optional[float]:
        if len(xs) != len(ys) or len(xs) < 2:
            return None
        ranked_x = self._rank_values(xs)
        ranked_y = self._rank_values(ys)
        return self._compute_pearson(ranked_x, ranked_y)

    @staticmethod
    def _compute_kendall(xs: List[float], ys: List[float]) -> Optional[float]:
        n = len(xs)
        if n != len(ys) or n < 2:
            return None
        concordant = discordant = ties_x = ties_y = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                if math.isclose(dx, 0.0) and math.isclose(dy, 0.0):
                    continue
                if math.isclose(dx, 0.0):
                    ties_x += 1
                    continue
                if math.isclose(dy, 0.0):
                    ties_y += 1
                    continue
                product = dx * dy
                if product > 0:
                    concordant += 1
                elif product < 0:
                    discordant += 1
        denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
        if math.isclose(denominator, 0.0, abs_tol=1e-12):
            return None
        return (concordant - discordant) / denominator

    @staticmethod
    def _rank_values(values: List[float]) -> List[float]:
        sorted_pairs = sorted((value, index) for index, value in enumerate(values))
        ranks = [0.0] * len(values)
        idx = 0
        while idx < len(sorted_pairs):
            value = sorted_pairs[idx][0]
            same = [sorted_pairs[idx]]
            j = idx + 1
            while j < len(sorted_pairs) and math.isclose(sorted_pairs[j][0], value):
                same.append(sorted_pairs[j])
                j += 1
            avg_rank = sum(range(idx + 1, j + 1)) / len(same)
            for _, original_index in same:
                ranks[original_index] = avg_rank
            idx = j
        return ranks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            logger.debug("Archivo de configuración estadística no encontrado: %s", path)
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        decimal_value = normalize_decimal(value)
        if isinstance(decimal_value, Decimal):
            return float(decimal_value)
        return None

    @staticmethod
    def _parse_date(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        normalized = normalize_date(value)
        if not normalized:
            return None
        try:
            return datetime.strptime(normalized, "%Y-%m-%d")
        except ValueError:
            return None

    def _build_finding(
        self,
        entry: Dict[str, Any],
        *,
        status: FindingStatus,
        severity: Optional[str],
        summary: str,
        metadata: Dict[str, Any],
        documents: Optional[List[str]] = None,
    ) -> CorrelationFinding:
        rule_id = entry.get("id") or "statistical_rule"
        version = str(entry.get("version", self.config.get("version", "v1.0.0")))
        finding = CorrelationFinding(
            id=self._fingerprint(rule_id, metadata),
            rule_id=rule_id,
            rule_version=version,
            status=status,
            severity=self._map_severity(severity),
            summary=summary,
            documents_involved=documents or entry.get("documents") or [],
            entities_involved=entry.get("entities") or [],
            evidence=[],
            metadata=metadata,
            tags=list(entry.get("tags") or []),
        )
        return finding

    def _build_missing_finding(
        self,
        entry: Dict[str, Any],
        *,
        summary: str,
        metadata: Dict[str, Any],
        status: FindingStatus,
        documents: Optional[List[str]] = None,
    ) -> CorrelationFinding:
        finding = self._build_finding(
            entry,
            status=status,
            severity=entry.get("severity"),
            summary=summary,
            metadata=metadata,
            documents=documents,
        )
        finding.finding_type = "statistical"
        return finding

    @staticmethod
    def _map_severity(value: Optional[str]) -> FindingSeverity:
        mapping = {
            "low": FindingSeverity.LOW,
            "medio": FindingSeverity.MEDIUM,
            "medium": FindingSeverity.MEDIUM,
            "high": FindingSeverity.HIGH,
            "critico": FindingSeverity.CRITICAL,
            "critical": FindingSeverity.CRITICAL,
        }
        if value is None:
            return FindingSeverity.MEDIUM
        return mapping.get(str(value).lower(), FindingSeverity.MEDIUM)

    @staticmethod
    def _missing_status(*values: Any) -> Optional[FindingStatus]:
        flags = [is_missing(value) for value in values]
        if all(flags):
            return FindingStatus.NOT_APPLICABLE
        if any(flags):
            return FindingStatus.INSUFFICIENT_DATA
        return None

    @staticmethod
    def _fingerprint(rule_id: str, metadata: Dict[str, Any]) -> str:
        import hashlib
        payload = f"{rule_id}:{json_dumps_sorted(metadata)}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()


def json_dumps_sorted(data: Dict[str, Any]) -> str:
    import json

    try:
        return json.dumps(data, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(data), ensure_ascii=False)
