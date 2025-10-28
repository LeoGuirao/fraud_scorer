"""Heurísticas ligeras para sugerir puntos de referencia geográficos."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple
import re

# Diccionario base de ubicaciones → coordenadas aproximadas (lat, lon)
# Se puede extender según las rutas frecuentes de los casos.
_REFERENCE_POINTS = {
    "carretera matehuala": (22.571049, -100.632222),
    "carretera matehuala san luis potosi": (22.571049, -100.632222),
    "carretera matehuala san luis potosi kilometro 57": (22.571049, -100.632222),
    "kilometro 57": (22.571049, -100.632222),
    "kilometro57": (22.571049, -100.632222),
    "km 57": (22.571049, -100.632222),
    "km57": (22.571049, -100.632222),
    "matehuala san luis potosi": (22.571049, -100.632222),
    "matehuala": (22.571049, -100.632222),
}


def _normalize_text(value: str) -> str:
    normalized = value.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@lru_cache(maxsize=128)
def suggest_reference_point(text: Optional[str]) -> Optional[Tuple[float, float]]:
    """Devuelve una coordenada aproximada para la ubicación declarada."""
    if not text:
        return None

    normalized = _normalize_text(text)
    tokens = normalized.split()

    # Buscar coincidencias completas o parciales
    for key, coords in _REFERENCE_POINTS.items():
        if key in normalized:
            return coords

        key_tokens = key.split()
        if all(token in tokens for token in key_tokens):
            return coords

    return None


__all__ = ["suggest_reference_point"]
