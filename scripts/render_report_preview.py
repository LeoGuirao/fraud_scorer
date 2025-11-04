#!/usr/bin/env python3
"""
Genera una vista previa HTML del reporte usando artefactos guardados.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fraud_scorer.models.extraction import ConsolidatedExtraction, ConsolidatedFields
from fraud_scorer.models.fraud_analysis import FraudAnalysisResult
from fraud_scorer.templates.fraud_report_generator import FraudReportGenerator
from fraud_scorer.analyzers.correlation.models import CorrelationReport


def load_case_payload(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_consolidated(case_id: str, payload: Dict[str, Any]) -> ConsolidatedExtraction:
    consolidated_fields = payload.get("consolidated_fields") or payload.get("consolidated") or {}
    return ConsolidatedExtraction(
        case_id=case_id,
        consolidated_fields=ConsolidatedFields.model_validate(consolidated_fields),
        consolidation_sources=payload.get("consolidation_sources") or {},
        conflicts_resolved=payload.get("conflicts_resolved") or [],
        confidence_scores=payload.get("confidence_scores") or {},
    )


def build_analyses(payload: Dict[str, Any]) -> List[FraudAnalysisResult]:
    results: List[FraudAnalysisResult] = []
    for item in payload.get("fraud_analyses", []):
        try:
            results.append(FraudAnalysisResult.model_validate(item))
        except Exception as exc:  # pragma: no cover - robustez CLI
            print(f"⚠️  No se pudo cargar un análisis: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Renderiza un reporte HTML desde un snapshot JSON.")
    parser.add_argument("--case", required=True, type=Path, help="Ruta al JSON de snapshot/case index.")
    parser.add_argument("--output", type=Path, help="Ruta de salida para el HTML.")
    args = parser.parse_args()

    case_payload = load_case_payload(args.case)
    case_id = case_payload.get("case_id") or "CASE-PREVIEW"

    consolidated = build_consolidated(case_id, case_payload)
    analyses = build_analyses(case_payload)
    documents_metadata = case_payload.get("documents_metadata") or []
    correlation_raw = case_payload.get("correlation_report")
    correlation_obj: Optional[CorrelationReport] = None
    if correlation_raw:
        try:
            correlation_obj = CorrelationReport.model_validate(correlation_raw)
        except Exception:
            correlation_obj = None

    generator = FraudReportGenerator()
    report_data = generator.prepare_fraud_report_data(
        consolidated_data=consolidated,
        fraud_analyses=analyses,
        documents_metadata=documents_metadata,
        correlation_report=correlation_obj,
    )
    html = generator.render_html_template("report_template.html", report_data)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html, encoding="utf-8")
        print(f"Reporte HTML guardado en {args.output}")
    else:
        print(html)


if __name__ == "__main__":
    main()
