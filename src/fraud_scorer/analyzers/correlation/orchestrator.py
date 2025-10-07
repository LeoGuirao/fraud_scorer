"""Orquestador principal del motor de correlación."""
from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from fraud_scorer.analyzers.correlation.models import (
    CaseContext,
    CorrelationFinding,
    CorrelationReport,
    FindingStatus,
)
from fraud_scorer.models.extraction import ConsolidatedExtraction, DocumentExtraction
from fraud_scorer.models.fraud_analysis import FraudAnalysisResult

if TYPE_CHECKING:
    from fraud_scorer.storage.ocr_cache import OCRCacheManager

from .engines.rule_engine import RuleEngine
from .engines.statistical_correlator import StatisticalCorrelator
from .engines.rag_evidence_builder import RAGEvidenceBuilder
from .metrics import record_report
from .rules import gps_rules

logger = logging.getLogger(__name__)


class CorrelationEngine:
    """Coordina la ejecución de reglas, correlación estadística y evidencia contextual."""

    def __init__(
        self,
        *,
        rule_engine: Optional[RuleEngine] = None,
        statistical_correlator: Optional[StatisticalCorrelator] = None,
        rag_evidence_builder: Optional[RAGEvidenceBuilder] = None,
    ) -> None:
        self.rule_engine = rule_engine or RuleEngine()
        self.statistical_correlator = statistical_correlator or StatisticalCorrelator()
        self.rag_evidence_builder = rag_evidence_builder or RAGEvidenceBuilder()

    def run(
        self,
        *,
        case_id: str,
        consolidated: Optional[ConsolidatedExtraction | Dict[str, Any]],
        extractions: Sequence[DocumentExtraction | Dict[str, Any]],
        fraud_results: Sequence[FraudAnalysisResult | Dict[str, Any]],
        case_index: Optional[Dict[str, Any]] = None,
        enable_rag: bool = True,
        cache_manager: Optional["OCRCacheManager"] = None,
    ) -> CorrelationReport:
        logger.debug("Iniciando CorrelationEngine para %s", case_id)

        context = CaseContext.from_case(
            case_id=case_id,
            consolidated=consolidated,
            extractions=extractions,
            fraud_results=fraud_results,
            case_index=case_index,
            cache_manager=cache_manager,
            entity_normalizer=self.rule_engine.entity_normalizer,
        )

        findings: List[CorrelationFinding] = []

        rule_findings = self.rule_engine.evaluate(context)
        findings.extend(rule_findings)

        gps_findings = gps_rules.evaluate(context)
        findings.extend(gps_findings)

        statistical_findings = self.statistical_correlator.analyze(context)
        findings.extend(statistical_findings)

        if enable_rag:
            rag_candidates = [f for f in findings if f.status == FindingStatus.NEEDS_CONTEXT]
            if rag_candidates:
                logger.debug("Enriqueciendo %s hallazgos con RAG", len(rag_candidates))
                self.rag_evidence_builder.build(
                    rag_candidates,
                    case_id=case_id,
                    context=context,
                )

        report = CorrelationReport(case_id=case_id)
        report.extend_findings(findings)
        metadata_payload = {
            "rule_count": len(rule_findings),
            "statistical_count": len(statistical_findings),
            "gps_rule_count": len(gps_findings),
            "documents_indexed": len(context.documents),
            "rules_catalog_version": getattr(self.rule_engine, "catalog_version", "v0"),
            "entity_mappings_version": getattr(self.rule_engine, "entity_mappings_version", "v0"),
            "statistical_config_version": self.statistical_correlator.config.get("version"),
            "rag_enabled": getattr(self.rag_evidence_builder, "enabled", False),
        }
        rag_summary = self._summarize_rag(findings)
        if rag_summary:
            metadata_payload["rag_summary"] = rag_summary
        report.metadata.update(metadata_payload)
        record_report(report)
        return report

    @staticmethod
    def _summarize_rag(findings: Sequence[CorrelationFinding]) -> Dict[str, Any]:
        rag_entries = []
        for finding in findings:
            metadata = finding.metadata or {}
            rag_meta = metadata.get("rag") if isinstance(metadata.get("rag"), dict) else None
            if rag_meta:
                rag_entries.append(rag_meta)
        if not rag_entries:
            return {}
        status_counts = Counter(entry.get("status", "unknown") for entry in rag_entries)
        latencies = [
            entry.get("latency_ms")
            for entry in rag_entries
            if isinstance(entry.get("latency_ms"), (int, float))
        ]
        summary: Dict[str, Any] = {
            "total": len(rag_entries),
            "by_status": dict(status_counts),
        }
        if latencies:
            summary["avg_latency_ms"] = sum(latencies) / len(latencies)
            summary["max_latency_ms"] = max(latencies)
        return summary
