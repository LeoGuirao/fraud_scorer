"""Modelos utilizados por el motor de correlación."""

from .case_context import CaseContext, DocumentSnapshot
from .finding import CorrelationFinding, FindingStatus, FindingSeverity
from .correlation_result import CorrelationReport

__all__ = [
    "CaseContext",
    "DocumentSnapshot",
    "CorrelationFinding",
    "FindingStatus",
    "FindingSeverity",
    "CorrelationReport",
]
