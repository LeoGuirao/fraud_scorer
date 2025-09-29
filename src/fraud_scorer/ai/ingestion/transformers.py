"""Transformaciones y limpieza de texto para el Agente Rick."""

from __future__ import annotations

import re
from html import unescape
from typing import Dict, Any
import json


_CONTROL_CHARS = """\u0000-\u001f\u007f"""
_CONTROL_PATTERN = re.compile(f"[{_CONTROL_CHARS}]")
_MULTISPACE_PATTERN = re.compile(r"[ \t]{2,}")


def clean_text(value: str) -> str:
    """Limpia caracteres de control, HTML simple y espacios extras."""

    if not value:
        return ""

    text = unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)  # remover etiquetas HTML simples
    text = _CONTROL_PATTERN.sub(" ", text)
    text = _MULTISPACE_PATTERN.sub(" ", text)
    return text.replace("\n \n", "\n\n").strip()


def combine_metadata(base: Dict[str, object], extra: Dict[str, object] | None = None) -> Dict[str, object]:
    """Mezcla diccionarios de metadatos evitando sobrescribir valores críticos."""

    if not extra:
        return dict(base)

    result = dict(base)
    for key, value in extra.items():
        if key not in result or value:
            result[key] = value
    return result


__all__ = ["clean_text", "combine_metadata"]
def safe_json_dumps(value: Any) -> str:
    """Convierte estructuras complejas a JSON tolerante a fallos."""

    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        try:
            cleaned = _clean_obj(value)
            return json.dumps(cleaned, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)


def _clean_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_obj(v) for v in value]
    if isinstance(value, set):
        return sorted(_clean_obj(v) for v in value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["clean_text", "combine_metadata", "safe_json_dumps"]
