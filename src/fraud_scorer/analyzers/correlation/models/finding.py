"""Modelos de hallazgos generados por el motor de correlación."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from fraud_scorer.models.extraction import BaseModelCompat


class FindingStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_CONTEXT = "needs_context"


class FindingSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CorrelationFinding(BaseModelCompat):
    """Representa un hallazgo inter-documental."""

    id: str
    rule_id: str
    rule_version: str
    status: FindingStatus
    severity: FindingSeverity
    summary: str
    description: Optional[str] = None
    documents_involved: List[str] = Field(default_factory=list)
    entities_involved: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    recommendation: Optional[str] = None
    prompt_hash: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    finding_type: str = "rule"

    def add_evidence(self, item: Dict[str, Any]) -> None:
        self.evidence.append(item)

    def mark_needs_context(self, reason: str) -> None:
        self.status = FindingStatus.NEEDS_CONTEXT
        self.metadata.setdefault("notes", []).append(reason)
