"""Constructor de evidencia contextual (RAG) para hallazgos de correlación."""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Iterable, List, Optional

from fraud_scorer.ai.orchestration import AgenteRickService
from fraud_scorer.ai.orchestration.agente_rick import NO_CONTEXT_MESSAGE
from fraud_scorer.analyzers.correlation.models import (
    CaseContext,
    CorrelationFinding,
    FindingStatus,
)
from ..metrics import record_rag_event

logger = logging.getLogger(__name__)


class RAGEvidenceBuilder:
    """Enriquece hallazgos con evidencia contextual consultando al Agente Rick."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        service: Optional[AgenteRickService] = None,
        max_findings: Optional[int] = None,
    ) -> None:
        env_enabled = os.getenv("CORRELATION_ENABLE_RAG", "true").lower() in {"1", "true", "yes", "on"}
        self.enabled = enabled if enabled is not None else env_enabled
        self.max_findings = max_findings or int(os.getenv("CORRELATION_RAG_MAX_FINDINGS", "3"))
        self._service = service if self.enabled else None
        if self.enabled and self._service is None:
            try:
                self._service = AgenteRickService()
            except Exception as exc:  # pragma: no cover - si Rick no está disponible
                logger.warning("RAG deshabilitado: no se pudo inicializar Agente Rick (%s)", exc)
                self.enabled = False

    def build(
        self,
        findings: Iterable[CorrelationFinding],
        *,
        case_id: str,
        context: CaseContext,
    ) -> None:
        if not self.enabled or not self._service:
            for finding in findings:
                finding.metadata.setdefault("rag", {"status": "disabled"})
            return

        pending: List[CorrelationFinding] = [f for f in findings if f.status == FindingStatus.NEEDS_CONTEXT]
        if not pending:
            return

        for finding in pending[: self.max_findings]:
            query = self._build_query(finding, context)
            if not query:
                finding.metadata.setdefault("rag", {"status": "unsupported", "module": "correlation"})
                continue

            try:
                result = self._service.query(case_id=case_id, question=query, scope="correlation", module="correlation")
            except Exception as exc:  # pragma: no cover - red restringida
                logger.warning("RAG falló para regla %s: %s", finding.rule_id, exc)
                rag_payload = finding.metadata.setdefault("rag", {"status": "error"})
                rag_payload.update({"detail": str(exc), "module": "correlation"})
                record_rag_event("error", None)
                continue

            rag_metadata = {
                "status": "answered" if result.answer != NO_CONTEXT_MESSAGE else "no_context",
                "latency_ms": result.latency_ms,
                "token_usage": {
                    "prompt": result.tokens_input,
                    "completion": result.tokens_output,
                },
                "module": "correlation",
            }

            if result.answer and result.answer != NO_CONTEXT_MESSAGE:
                evidence_payload = {
                    "type": "rag_answer",
                    "text": result.answer,
                    "sources": result.sources,
                }
                finding.add_evidence(evidence_payload)
                finding.metadata["rag"] = rag_metadata
                finding.tags = list(set(finding.tags + ["rag"]))
                if finding.status == FindingStatus.NEEDS_CONTEXT:
                    finding.status = FindingStatus.FAIL
                finding.prompt_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
            else:
                rag_metadata["answer_preview"] = result.answer
                finding.metadata["rag"] = rag_metadata

            record_rag_event(rag_metadata.get("status", "unknown"), result.latency_ms)

    def _build_query(self, finding: CorrelationFinding, context: CaseContext) -> str:
        description = finding.summary or "Hallazgo de correlación"
        documents = ", ".join(finding.documents_involved or [])
        metadata_parts = []
        meta = finding.metadata or {}
        for key in ("source_value", "target_value", "mean", "std_dev", "ratio"):
            if key in meta:
                metadata_parts.append(f"{key.replace('_', ' ')}: {meta[key]}")
        meta_text = "; ".join(metadata_parts)

        context_clues = []
        for doc_type, snapshots in context.documents_by_type.items():
            if finding.documents_involved and doc_type not in finding.documents_involved:
                continue
            for snapshot in snapshots[:2]:  # limitar para no generar prompts enormes
                summary = []
                if snapshot.document_name:
                    summary.append(snapshot.document_name)
                if snapshot.document_type:
                    summary.append(snapshot.document_type)
                if summary:
                    context_clues.append(" / ".join(summary))

        clues = "; ".join(context_clues)
        question = (
            f"Analiza la regla '{finding.rule_id}' que indica: {description}. "
            f"Documentos involucrados: {documents}. "
            f"Datos relevantes: {meta_text}. "
        )
        if clues:
            question += f"Contexto del caso: {clues}. "
        question += "Confirma si existe evidencia textual que respalde la discrepancia y cita fragmentos concretos."
        return question.strip()
