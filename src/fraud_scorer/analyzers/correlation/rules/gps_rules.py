"""GPS specific correlation rules leveraging dataset summaries."""

from __future__ import annotations

from typing import Dict, List, Optional

from fraud_scorer.analyzers.correlation.models import (
    CaseContext,
    CorrelationFinding,
    FindingSeverity,
    FindingStatus,
)


RULE_VERSION = "2025.10"
GAP_THRESHOLD_MINUTES = 120


def evaluate(context: CaseContext) -> List[CorrelationFinding]:
    gps_documents = context.gps_documents or {}
    if not gps_documents:
        return []

    audit_map: Dict[str, Dict] = {}
    for entry in context.metadata.get("gps_ingestion_audit") or []:
        if isinstance(entry, dict) and entry.get("document_name"):
            audit_map[entry["document_name"]] = entry

    findings: List[CorrelationFinding] = []
    for doc_name, entry in gps_documents.items():
        summary = entry.get("summary") or {}
        dataset_meta = entry.get("dataset") or {}
        warnings = entry.get("normalization_warnings") or summary.get("warnings") or []

        if not dataset_meta or int(dataset_meta.get("row_count") or 0) == 0:
            findings.append(
                _build_finding(
                    rule_id="gps_dataset_available",
                    doc_name=doc_name,
                    status=FindingStatus.INSUFFICIENT_DATA,
                    summary=f"Reporte GPS {doc_name} no pudo generarse (dataset vacío)",
                    severity=FindingSeverity.MEDIUM,
                    metadata={
                        "summary": summary,
                        "dataset": dataset_meta,
                        "warnings": warnings,
                        "audit": audit_map.get(doc_name),
                    },
                )
            )
            continue

        time_gaps = summary.get("time_gaps") or []
        largest_gap = _max_gap_minutes(time_gaps)
        if largest_gap and largest_gap >= GAP_THRESHOLD_MINUTES:
            findings.append(
                _build_finding(
                    rule_id="gps_time_gap",
                    doc_name=doc_name,
                    status=FindingStatus.FAIL,
                    summary=f"Se detectó un gap de {largest_gap} minutos en el monitoreo GPS",
                    severity=FindingSeverity.HIGH if largest_gap >= 240 else FindingSeverity.MEDIUM,
                    metadata={
                        "largest_gap_minutes": largest_gap,
                        "gaps": time_gaps,
                        "summary": summary,
                        "dataset": dataset_meta,
                        "audit": audit_map.get(doc_name),
                    },
                    recommendation="Solicitar evidencia adicional (bitácora, CFDI, soporte manual) para cubrir el gap detectado.",
                )
            )
        else:
            findings.append(
                _build_finding(
                    rule_id="gps_time_gap",
                    doc_name=doc_name,
                    status=FindingStatus.PASS,
                    summary="Monitoreo GPS sin gaps relevantes",
                    severity=FindingSeverity.LOW,
                    metadata={
                        "gaps": time_gaps,
                        "summary": summary,
                        "dataset": dataset_meta,
                        "audit": audit_map.get(doc_name),
                        "warnings": warnings,
                    },
                )
            )

        if "missing_coordinates" in warnings:
            findings.append(
                _build_finding(
                    rule_id="gps_coordinates",
                    doc_name=doc_name,
                    status=FindingStatus.NEEDS_CONTEXT,
                    summary="El dataset GPS presenta registros sin coordenadas",
                    severity=FindingSeverity.MEDIUM,
                    metadata={
                        "warnings": warnings,
                        "summary": summary,
                        "dataset": dataset_meta,
                        "audit": audit_map.get(doc_name),
                    },
                    recommendation="Solicitar al proveedor GPS la exportación completa o confirmar la validez del monitoreo.",
                )
            )

    return findings


def _max_gap_minutes(gaps: List[Dict]) -> Optional[int]:
    values = [int(gap.get("gap_minutes") or 0) for gap in gaps if gap.get("gap_minutes")]
    return max(values) if values else None


def _build_finding(
    *,
    rule_id: str,
    doc_name: str,
    status: FindingStatus,
    summary: str,
    severity: FindingSeverity,
    metadata: Dict[str, object],
    recommendation: Optional[str] = None,
) -> CorrelationFinding:
    return CorrelationFinding(
        id=f"gps::{rule_id}::{doc_name}",
        rule_id=rule_id,
        rule_version=RULE_VERSION,
        status=status,
        severity=severity,
        summary=summary,
        documents_involved=[doc_name],
        metadata=metadata,
        recommendation=recommendation,
        finding_type="gps",
        tags=["gps", rule_id],
    )


__all__ = ["evaluate"]

