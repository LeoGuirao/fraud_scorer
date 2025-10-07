from __future__ import annotations

from fraud_scorer.analyzers.correlation.models import CaseContext, CorrelationFinding
from fraud_scorer.analyzers.correlation.rules import gps_rules


def _build_context(gps_documents):
    return CaseContext(
        case_id="CASE-GPS",
        consolidated=None,
        documents=[],
        documents_by_type={},
        fraud_results={},
        aggregates={},
        entities={},
        timeline=[],
        metadata={},
        gps_documents=gps_documents,
    )


def test_gps_rules_detects_gap():
    context = _build_context(
        {
            "reporte.csv": {
                "summary": {
                    "time_gaps": [
                        {"gap_minutes": 150, "start": "2025-01-01T01:00:00Z", "end": "2025-01-01T03:30:00Z"}
                    ]
                },
                "dataset": {"row_count": 200, "checksum": "abc"},
                "normalization_warnings": [],
            }
        }
    )

    findings = gps_rules.evaluate(context)
    assert any(
        isinstance(finding, CorrelationFinding)
        and finding.rule_id == "gps_time_gap"
        and finding.status.value == "fail"
        for finding in findings
    )


def test_gps_rules_handle_empty_dataset():
    context = _build_context(
        {
            "reporte.csv": {
                "summary": {},
                "dataset": {},
            }
        }
    )

    findings = gps_rules.evaluate(context)
    assert any(f.rule_id == "gps_dataset_available" and f.status.value == "insufficient_data" for f in findings)
