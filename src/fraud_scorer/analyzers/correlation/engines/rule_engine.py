"""Motor determinístico de reglas para correlación inter-documental."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from fraud_scorer.analyzers.correlation.models import (
    CaseContext,
    CorrelationFinding,
    FindingSeverity,
    FindingStatus,
)
from fraud_scorer.analyzers.correlation.utils import EntityNormalizer


logger = logging.getLogger(__name__)


class RuleEngine:
    """Evalúa reglas determinísticas definidas en YAML."""

    def __init__(
        self,
        rules_path: Optional[Path] = None,
        entity_mappings_path: Optional[Path] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        self.rules_path = rules_path or (base_dir / "rules" / "correlation_rules.yaml")
        self.entity_mappings_path = entity_mappings_path or (
            base_dir / "rules" / "entity_mappings.yaml"
        )
        raw_rules = self._load_yaml(self.rules_path) or []
        if isinstance(raw_rules, dict):
            self.catalog_metadata = raw_rules.get("meta") or {}
            self._rules = raw_rules.get("rules") or []
        else:
            self.catalog_metadata = {}
            self._rules = raw_rules

        raw_entity_mappings = self._load_yaml(self.entity_mappings_path)
        self._entity_mappings = raw_entity_mappings if isinstance(raw_entity_mappings, dict) else {}
        mappings_version = (
            self._entity_mappings.get("version")
            or self._entity_mappings.get("meta", {}).get("version")
        )
        self._entity_normalizer = EntityNormalizer(
            mappings=self._entity_mappings,
            version=mappings_version,
        )

    @property
    def rules(self) -> List[Dict[str, Any]]:
        return list(self._rules)

    @property
    def catalog_version(self) -> str:
        return str(self.catalog_metadata.get("version", "v0"))

    @property
    def entity_normalizer(self) -> EntityNormalizer:
        return self._entity_normalizer

    @property
    def entity_mappings_version(self) -> str:
        return getattr(self._entity_normalizer, "version", "v0")

    def evaluate(self, context: CaseContext) -> List[CorrelationFinding]:
        findings: List[CorrelationFinding] = []
        for rule in self._rules:
            try:
                result = self._evaluate_rule(rule, context)
                if not result:
                    continue
                if isinstance(result, list):
                    findings.extend(result)
                else:
                    findings.append(result)
            except Exception as exc:  # pragma: no cover - defensivo
                logger.exception("Error evaluando regla %s", rule.get("id"))
                findings.append(
                    self._build_fallback_finding(
                        rule=rule,
                        status=FindingStatus.NEEDS_CONTEXT,
                        summary=f"Regla {rule.get('id')} no pudo evaluarse: {exc}",
                        metadata={"exception": str(exc)},
                    )
                )
        return findings

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: Path) -> Any:
        if not path.exists():
            logger.warning("Archivo YAML no encontrado: %s", path)
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or None

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    def _evaluate_rule(
        self,
        rule: Dict[str, Any],
        context: CaseContext,
    ) -> Optional[CorrelationFinding | List[CorrelationFinding]]:
        condition = rule.get("condition") or {}
        rule_type = (condition.get("type") or "").lower()

        if not rule_type:
            return None

        if rule_type == "equality":
            return self._evaluate_equality(rule, context)
        if rule_type == "temporal_order":
            return self._evaluate_temporal_order(rule, context)
        if rule_type == "set_overlap":
            return self._evaluate_set_overlap(rule, context)
        if rule_type == "exists":
            return self._evaluate_exists(rule, context)
        if rule_type == "numeric_order":
            return self._evaluate_numeric_order(rule, context)

        logger.warning("Tipo de condición desconocida: %s", rule_type)
        return None

    # ------------------------------------------------------------------
    # Rule evaluators
    # ------------------------------------------------------------------
    def _evaluate_equality(self, rule: Dict[str, Any], context: CaseContext) -> CorrelationFinding:
        condition = rule.get("condition") or {}
        source_path = condition.get("source")
        target_path = condition.get("target")
        tolerance = condition.get("tolerance", 0)
        absolute_tolerance = condition.get("absolute_tolerance")

        source_value = context.resolve(source_path)
        target_value = context.resolve(target_path)

        meta = {
            "source_path": source_path,
            "target_path": target_path,
            "source_value": source_value,
            "target_value": target_value,
            "tolerance": tolerance,
        }

        if source_value is None or target_value is None:
            return self._build_missing_data_finding(rule, meta)

        src_num = self._to_float(source_value)
        tgt_num = self._to_float(target_value)

        if src_num is not None and tgt_num is not None:
            limit = absolute_tolerance if absolute_tolerance is not None else None
            if limit is None:
                tol = float(tolerance or 0)
                base = abs(tgt_num) if abs(tgt_num) > 1 else 1.0
                limit = base * tol
            diff = abs(src_num - tgt_num)
            passed = diff <= (limit or 0)
        else:
            passed = self._normalize_str(source_value) == self._normalize_str(target_value)

        severity = self._map_severity(rule)

        if passed:
            return self._build_success_finding(rule, meta, severity)

        meta["diff"] = None if src_num is None or tgt_num is None else abs(src_num - tgt_num)
        recom = self._get_recommendation(rule, on="fail")
        return self._build_failure_finding(rule, meta, severity, recommendation=recom)

    def _evaluate_temporal_order(
        self, rule: Dict[str, Any], context: CaseContext
    ) -> CorrelationFinding:
        condition = rule.get("condition") or {}
        earlier_path = condition.get("earlier")
        later_path = condition.get("later")
        allow_equal = bool(condition.get("allow_equal", True))

        earlier = context.resolve(earlier_path)
        later = context.resolve(later_path)

        meta = {
            "earlier_path": earlier_path,
            "later_path": later_path,
            "earlier": earlier,
            "later": later,
            "allow_equal": allow_equal,
        }

        if earlier is None or later is None:
            return self._build_missing_data_finding(rule, meta)

        earlier_date = self._normalise_date(earlier)
        later_date = self._normalise_date(later)
        if not earlier_date or not later_date:
            return self._build_missing_data_finding(rule, meta)

        if allow_equal:
            passed = earlier_date <= later_date
        else:
            passed = earlier_date < later_date

        severity = self._map_severity(rule)
        if passed:
            return self._build_success_finding(rule, meta, severity)

        recom = self._get_recommendation(rule, on="fail")
        return self._build_failure_finding(rule, meta, severity, recommendation=recom)

    def _evaluate_set_overlap(
        self, rule: Dict[str, Any], context: CaseContext
    ) -> CorrelationFinding:
        condition = rule.get("condition") or {}
        source_path = condition.get("source")
        target_path = condition.get("target")
        min_overlap = int(condition.get("min_overlap", 1))

        source_values = self._ensure_list(context.resolve(source_path))
        target_values = self._ensure_list(context.resolve(target_path))

        meta = {
            "source_path": source_path,
            "target_path": target_path,
            "source_values": source_values,
            "target_values": target_values,
            "min_overlap": min_overlap,
        }

        if not source_values or not target_values:
            return self._build_missing_data_finding(rule, meta)

        norm_source = {self._normalize_str(v) for v in source_values if v is not None}
        norm_target = {self._normalize_str(v) for v in target_values if v is not None}
        intersection = {v for v in norm_source if v and v in norm_target}

        meta["overlap_count"] = len(intersection)

        severity = self._map_severity(rule)
        if len(intersection) >= min_overlap:
            return self._build_success_finding(rule, meta, severity)

        recom = self._get_recommendation(rule, on="fail")
        return self._build_failure_finding(rule, meta, severity, recommendation=recom)

    def _evaluate_exists(self, rule: Dict[str, Any], context: CaseContext) -> CorrelationFinding:
        condition = rule.get("condition") or {}
        path = condition.get("path")
        value = context.resolve(path)
        meta = {"path": path, "value": value}

        severity = self._map_severity(rule)
        if value is None or value == "":
            recom = self._get_recommendation(rule, on="fail")
            return self._build_failure_finding(rule, meta, severity, recommendation=recom)
        return self._build_success_finding(rule, meta, severity)

    def _evaluate_numeric_order(
        self, rule: Dict[str, Any], context: CaseContext
    ) -> CorrelationFinding:
        condition = rule.get("condition") or {}
        lhs_path = condition.get("lhs")
        rhs_path = condition.get("rhs")
        operator = str(condition.get("operator") or "<=").strip()
        tolerance = condition.get("tolerance", 0)
        absolute_tolerance = condition.get("absolute_tolerance")

        lhs_value = context.resolve(lhs_path)
        rhs_value = context.resolve(rhs_path)

        meta = {
            "lhs_path": lhs_path,
            "rhs_path": rhs_path,
            "lhs_value": lhs_value,
            "rhs_value": rhs_value,
            "operator": operator,
            "tolerance": tolerance,
            "absolute_tolerance": absolute_tolerance,
        }

        if lhs_value is None or rhs_value is None:
            return self._build_missing_data_finding(rule, meta)

        lhs_num = self._to_float(lhs_value)
        rhs_num = self._to_float(rhs_value)
        if lhs_num is None or rhs_num is None:
            return self._build_missing_data_finding(rule, meta)

        percent_tol = float(tolerance or 0)
        abs_tol = float(absolute_tolerance) if absolute_tolerance is not None else None
        rhs_reference = abs(rhs_num) if abs(rhs_num) > 1 else 1.0

        def apply_upper_bound(base: float) -> float:
            delta = rhs_reference * percent_tol
            if abs_tol is not None:
                delta = max(delta, abs_tol)
            return base + delta

        def apply_lower_bound(base: float) -> float:
            delta = rhs_reference * percent_tol
            if abs_tol is not None:
                delta = max(delta, abs_tol)
            return base - delta

        passed: bool
        if operator in {"<=", "=<"}:
            threshold = apply_upper_bound(rhs_num)
            passed = lhs_num <= threshold
            meta["threshold"] = threshold
        elif operator == "<":
            threshold = apply_upper_bound(rhs_num)
            passed = lhs_num < threshold
            meta["threshold"] = threshold
        elif operator in {">=", "=>"}:
            threshold = apply_lower_bound(rhs_num)
            passed = lhs_num >= threshold
            meta["threshold"] = threshold
        elif operator == ">":
            threshold = apply_lower_bound(rhs_num)
            passed = lhs_num > threshold
            meta["threshold"] = threshold
        elif operator in {"==", "="}:
            diff = abs(lhs_num - rhs_num)
            base_tol = rhs_reference * percent_tol
            if abs_tol is not None:
                base_tol = max(base_tol, abs_tol)
            passed = diff <= base_tol
            meta["difference"] = diff
            meta["threshold"] = base_tol
        else:
            meta["unsupported_operator"] = operator
            return self._build_fallback_finding(
                rule,
                status=FindingStatus.NEEDS_CONTEXT,
                summary=f"Operador numérico no soportado: {operator}",
                metadata=meta,
            )

        severity = self._map_severity(rule)
        if passed:
            return self._build_success_finding(rule, meta, severity)

        recom = self._get_recommendation(rule, on="fail")
        return self._build_failure_finding(
            rule,
            metadata=meta,
            severity=severity,
            recommendation=recom,
        )

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_success_finding(
        self,
        rule: Dict[str, Any],
        metadata: Dict[str, Any],
        severity: FindingSeverity,
    ) -> CorrelationFinding:
        summary = rule.get("description") or f"Regla {rule.get('id')} cumplida"
        return self._build_finding(
            rule=rule,
            status=FindingStatus.PASS,
            severity=severity,
            summary=summary,
            metadata=metadata,
        )

    def _build_failure_finding(
        self,
        rule: Dict[str, Any],
        metadata: Dict[str, Any],
        severity: FindingSeverity,
        recommendation: Optional[str],
    ) -> CorrelationFinding:
        summary = rule.get("description") or f"Regla {rule.get('id')} incumplida"
        finding = self._build_finding(
            rule=rule,
            status=FindingStatus.FAIL,
            severity=severity,
            summary=summary,
            metadata=metadata,
        )
        if recommendation:
            finding.recommendation = recommendation
        return finding

    def _build_missing_data_finding(
        self, rule: Dict[str, Any], metadata: Dict[str, Any]
    ) -> CorrelationFinding:
        summary = (
            rule.get("missing_summary")
            or f"Información insuficiente para evaluar {rule.get('id')}"
        )
        finding = self._build_finding(
            rule=rule,
            status=FindingStatus.NEEDS_CONTEXT,
            severity=self._map_severity(rule),
            summary=summary,
            metadata=metadata,
        )
        return finding

    def _build_fallback_finding(
        self,
        rule: Dict[str, Any],
        status: FindingStatus,
        summary: str,
        metadata: Dict[str, Any],
    ) -> CorrelationFinding:
        return self._build_finding(
            rule=rule,
            status=status,
            severity=self._map_severity(rule),
            summary=summary,
            metadata=metadata,
        )

    def _build_finding(
        self,
        rule: Dict[str, Any],
        status: FindingStatus,
        severity: FindingSeverity,
        summary: str,
        metadata: Dict[str, Any],
    ) -> CorrelationFinding:
        rule_id = rule.get("id") or "rule"
        rule_version = str(rule.get("version", "1.0"))
        documents_involved = rule.get("documents") or []
        entities = rule.get("entities") or []

        fingerprint_payload = json.dumps(
            {
                "rule_id": rule_id,
                "documents": documents_involved,
                "metadata": metadata,
            },
            sort_keys=True,
        ).encode("utf-8")
        finding_id = hashlib.sha1(fingerprint_payload).hexdigest()

        finding = CorrelationFinding(
            id=finding_id,
            rule_id=rule_id,
            rule_version=rule_version,
            status=status,
            severity=severity,
            summary=summary,
            description=rule.get("description"),
            documents_involved=documents_involved,
            entities_involved=entities,
            metadata=dict(metadata or {}),
        )
        if rule.get("tags"):
            finding.tags = list(rule.get("tags"))
        return finding

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _map_severity(rule: Dict[str, Any]) -> FindingSeverity:
        raw = (rule.get("severity") or "medium").lower()
        mapping = {
            "low": FindingSeverity.LOW,
            "medium": FindingSeverity.MEDIUM,
            "high": FindingSeverity.HIGH,
            "critical": FindingSeverity.CRITICAL,
        }
        return mapping.get(raw, FindingSeverity.MEDIUM)

    @staticmethod
    def _normalize_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip().lower()
        return str(value).strip().lower()

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list):
            if not value:
                return None
            return RuleEngine._to_float(value[0])
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _normalise_date(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
                try:
                    return str(datetime.strptime(text, fmt).date())
                except ValueError:
                    continue
            return text
        dt = getattr(value, "date", None)
        if callable(dt):
            try:
                return str(dt())
            except Exception:  # pragma: no cover
                return None
        return None

    @staticmethod
    def _get_recommendation(rule: Dict[str, Any], on: str) -> Optional[str]:
        section = rule.get(f"on_{on}")
        if isinstance(section, dict):
            return section.get("recommendation")
        return None
