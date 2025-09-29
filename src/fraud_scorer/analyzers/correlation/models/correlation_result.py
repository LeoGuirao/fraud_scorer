"""Modelo de salida consolidado para el motor de correlación."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import Field

from fraud_scorer.models.extraction import BaseModelCompat
from .finding import CorrelationFinding, FindingSeverity, FindingStatus


class CorrelationReport(BaseModelCompat):
    """Reporte generado tras ejecutar el motor de correlación."""

    case_id: str
    findings: List[CorrelationFinding] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status_counts: Dict[FindingStatus, int] = Field(default_factory=dict)
    severity_counts: Dict[FindingSeverity, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_finding(self, finding: CorrelationFinding) -> None:
        self.findings.append(finding)
        self._recalculate_counters()

    def extend_findings(self, findings: List[CorrelationFinding]) -> None:
        if not findings:
            return
        self.findings.extend(findings)
        self._recalculate_counters()

    def _recalculate_counters(self) -> None:
        status_counter: Counter = Counter()
        severity_counter: Counter = Counter()
        for finding in self.findings:
            status_counter[finding.status] += 1
            severity_counter[finding.severity] += 1
        self.status_counts = {status: status_counter.get(status, 0) for status in FindingStatus}
        self.severity_counts = {severity: severity_counter.get(severity, 0) for severity in FindingSeverity}

    def as_summary(self) -> Dict[str, Any]:
        return {
            "total_findings": len(self.findings),
            "status_counts": {k.value: v for k, v in self.status_counts.items()},
            "severity_counts": {k.value: v for k, v in self.severity_counts.items()},
            "generated_at": self.generated_at.isoformat(),
        }
