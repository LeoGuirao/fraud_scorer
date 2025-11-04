"""
Configuración centralizada para la integración con FiscalAPI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


DEFAULT_TEST_BASE_URL = "https://test.fiscalapi.com"
DEFAULT_PROD_BASE_URL = "https://live.fiscalapi.com"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_CACHE_TTL = 60 * 60 * 24  # 24 horas
DEFAULT_NAMESPACE = "fiscal_validation"


def _env(name: str, fallback: Optional[str] = None) -> str:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        if fallback and os.getenv(fallback):
            return os.getenv(fallback, "").strip()
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class FiscalAPIConfig:
    """Valores de configuración para FiscalAPI."""

    base_url: str
    api_key: str
    tenant_id: str
    environment: str
    timeout_seconds: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL
    cache_namespace: str = DEFAULT_NAMESPACE
    document_types: Tuple[str, ...] = (
        "cfdi_carta_porte",
        "factura_comercial_cfdi",
        "poliza_de_la_aseguradora",
    )
    fraud_patterns: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {
            "cfdi_cancelado": {"severity": "critico", "weight": 0.9},
            "emisor_efos": {"severity": "critico", "weight": 0.95},
            "totales_no_coinciden": {"severity": "alto", "weight": 0.7},
            "uuid_no_encontrado": {"severity": "alto", "weight": 0.65},
        }
    )

    def headers(self) -> Dict[str, str]:
        """Headers comunes para FiscalAPI."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
            "x-tenant-id": self.tenant_id,
            "x-tenant-key": self.tenant_id,
            "Content-Type": "application/json",
            "User-Agent": "fraud-scorer/2.0",
        }

    def is_configured(self) -> bool:
        """Indica si contamos con credenciales válidas."""
        return bool(self.api_key and self.tenant_id and self.base_url)


@lru_cache(maxsize=1)
def get_fiscal_api_config() -> FiscalAPIConfig:
    """Obtiene configuración desde variables de entorno."""
    environment = _env("FISCAL_API_ENVIRONMENT") or "test"
    environment = environment.strip().lower()

    base_url = _env("FISCAL_API_BASE_URL")
    if not base_url:
        base_url = DEFAULT_PROD_BASE_URL if environment == "production" else DEFAULT_TEST_BASE_URL

    api_key = _env("FISCAL_API_KEY", fallback="FISCALAPI_KEY")
    tenant = _env("FISCAL_API_TENANT", fallback="FISCALAPI_TID")

    timeout_raw = _env("FISCAL_API_TIMEOUT")
    max_retries_raw = _env("FISCAL_API_MAX_RETRIES")
    cache_ttl_raw = _env("FISCAL_API_CACHE_TTL")

    try:
        timeout = float(timeout_raw) if timeout_raw else DEFAULT_TIMEOUT
    except ValueError:
        timeout = DEFAULT_TIMEOUT

    try:
        max_retries = int(max_retries_raw) if max_retries_raw else DEFAULT_MAX_RETRIES
    except ValueError:
        max_retries = DEFAULT_MAX_RETRIES

    try:
        cache_ttl = int(cache_ttl_raw) if cache_ttl_raw else DEFAULT_CACHE_TTL
    except ValueError:
        cache_ttl = DEFAULT_CACHE_TTL

    namespace = _env("FISCAL_API_CACHE_NAMESPACE") or DEFAULT_NAMESPACE

    return FiscalAPIConfig(
        base_url=base_url,
        api_key=api_key,
        tenant_id=tenant,
        environment=environment,
        timeout_seconds=timeout,
        max_retries=max_retries,
        cache_ttl_seconds=cache_ttl,
        cache_namespace=namespace,
    )


def invalidate_cache() -> None:
    """Resetea cache interno (para pruebas)."""
    get_fiscal_api_config.cache_clear()
