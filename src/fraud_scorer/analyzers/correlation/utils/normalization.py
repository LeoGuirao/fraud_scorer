"""Helpers de normalización para el motor de correlación."""
from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional


_DECIMAL_SANITIZE_RE = re.compile(r"[^0-9,.-]+")
_DATE_SEPARATORS_RE = re.compile(r"[./\\]")
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%m/%d/%Y",
)


def normalize_decimal(value: Any) -> Optional[Decimal]:
    """Convierte montos con símbolos o separadores mixtos en Decimal."""
    if value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = _DECIMAL_SANITIZE_RE.sub("", text)
    if not cleaned:
        return None
    if cleaned.count(",") and cleaned.count("."):
        # Detectar separador decimal por posición: el último símbolo estadísticamente es decimal.
        last_comma = cleaned.rfind(",")
        last_point = cleaned.rfind(".")
        if last_point > last_comma:
            # Formato tipo 1,234.56 → quitar comas
            cleaned = cleaned.replace(",", "")
        else:
            # Formato tipo 1.234,56 → quitar puntos y reemplazar coma
            cleaned = cleaned.replace(".", "")
            cleaned = cleaned.replace(",", ".")
    elif cleaned.count(",") and not cleaned.count("."):
        cleaned = cleaned.replace(",", ".")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def normalize_decimal_as_str(value: Any) -> Optional[str]:
    number = normalize_decimal(value)
    if number is None:
        return None
    normalized = number.normalize()
    # Evitar notación científica
    return format(normalized, "f").rstrip("0").rstrip(".") or "0"


def normalize_date(value: Any) -> Optional[str]:
    """Normaliza fechas usadas en reglas a formato ISO."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value).strip()
    if not text:
        return None
    candidate = _DATE_SEPARATORS_RE.sub("/", text)
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(candidate, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue
    return None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, tuple, set)):
        if len(value) == 0:
            return True
        return all(is_missing(item) for item in value)
    if isinstance(value, dict):
        if len(value) == 0:
            return True
        return all(is_missing(item) for item in value.values())
    return False
