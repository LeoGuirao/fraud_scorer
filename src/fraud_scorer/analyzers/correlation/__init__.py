"""Subsistema de correlación inter-documental para análisis de fraude."""

__all__ = ["CorrelationEngine"]


def __getattr__(name: str):  # pragma: no cover - acceso perezoso
    if name == "CorrelationEngine":
        from .orchestrator import CorrelationEngine

        return CorrelationEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
