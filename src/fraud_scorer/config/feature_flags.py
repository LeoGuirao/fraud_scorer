"""
Feature flags y toggles de rollout.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

FiscalRolloutStage = Literal["disabled", "shadow", "partial", "enabled"]


@lru_cache(maxsize=1)
def get_fiscal_validation_stage() -> FiscalRolloutStage:
    """
    Retorna el stage actual del feature flag `FISCAL_VALIDATION_ROLLOUT`.
    Valores soportados:
      - disabled: no se ejecuta ninguna validación
      - shadow: se ejecuta pero sin afectar score ni reporte
      - partial: solo en subconjunto de casos/documentos
      - enabled: se habilita completamente
    """
    raw = (os.getenv("FISCAL_VALIDATION_ROLLOUT") or "disabled").strip().lower()
    if raw in {"on", "true", "full", "100%", "100", "enabled"}:
        return "enabled"
    if raw in {"staging", "partial", "gradual", "50", "50%"}:
        return "partial"
    if raw in {"shadow", "observe", "monitor"}:
        return "shadow"
    return "disabled"


def is_fiscal_validation_enabled() -> bool:
    """Indica si la validación fiscal debe impactar en el flujo principal."""
    return get_fiscal_validation_stage() == "enabled"


def should_collect_fiscal_metrics() -> bool:
    """
    Determina si debemos reportar métricas aunque la funcionalidad esté en shadow/partial.
    Útil para observar comportamiento previo a habilitar.
    """
    stage = get_fiscal_validation_stage()
    return stage in {"shadow", "partial", "enabled"}


def invalidate_caches() -> None:
    """Permite limpiar el cache interno (para pruebas)."""
    get_fiscal_validation_stage.cache_clear()
