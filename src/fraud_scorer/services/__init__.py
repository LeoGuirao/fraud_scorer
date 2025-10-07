# src/fraud_scorer/services/__init__.py
"""
Lazy load façade for service modules to avoid circular imports.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "ReplayService",
    "FraudDocumentCatalog",
    "FraudDocumentReprocessService",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial
    if name == "ReplayService":
        from .replay_service import ReplayService

        return ReplayService
    if name == "FraudDocumentCatalog":
        from .fraud_document_service import FraudDocumentCatalog

        return FraudDocumentCatalog
    if name == "FraudDocumentReprocessService":
        from .fraud_document_service import FraudDocumentReprocessService

        return FraudDocumentReprocessService
    raise AttributeError(f"module 'fraud_scorer.services' has no attribute '{name}'")
