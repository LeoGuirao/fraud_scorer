"""
Servicio de alto nivel para validar CFDIs usando FiscalAPI.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional, Union

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from fraud_scorer.config.feature_flags import get_fiscal_validation_stage, should_collect_fiscal_metrics
from fraud_scorer.config.fiscal_api_config import FiscalAPIConfig, get_fiscal_api_config
from fraud_scorer.models.fiscal_validation import (
    CFDIValidationRequest,
    FiscalValidationResult,
    FiscalValidationStatus,
)
from fraud_scorer.services.fiscal_api_cache import FiscalValidationCache
from fraud_scorer.services.fiscal_api_client import (
    FiscalAPIClient,
    FiscalAPIError,
    FiscalAPINotConfiguredError,
    FiscalAPINotFound,
    FiscalAPIUnavailable,
    FiscalAPIUnauthorized,
)

try:  # pragma: no cover - métricas opcionales
    from prometheus_client import Counter, Histogram

    VALIDATION_COUNTER = Counter(
        "fiscal_validation_events_total",
        "Eventos del servicio FiscalAPI",
        labelnames=("outcome", "environment", "stage"),
    )
    VALIDATION_LATENCY = Histogram(
        "fiscal_validation_latency_seconds",
        "Latencia de llamadas a FiscalAPI",
        labelnames=("environment", "stage"),
        buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13),
    )
except Exception:  # pragma: no cover - fallback silencioso

    class _Noop:  # pylint: disable=too-few-public-methods
        def labels(self, *args: Any, **kwargs: Any) -> "_Noop":
            return self

        def inc(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def observe(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    VALIDATION_COUNTER = VALIDATION_LATENCY = _Noop()

logger = logging.getLogger(__name__)

RawRequest = Union[CFDIValidationRequest, Dict[str, Any]]

_METADATA_ALIAS_MAP: Dict[str, tuple[str, ...]] = {
    "issuer_name": ("issuer_name", "issuerName", "emisor_nombre", "razon_social_emisor", "emisor"),
    "recipient_name": ("recipient_name", "recipientName", "receptor_nombre", "razon_social_receptor", "razón_social"),
    "invoice_effect": ("invoice_effect", "effect", "cfdi_effect", "tipo_comprobante", "cfdiType"),
    "issue_date": ("issue_date", "fecha_expedicion", "fecha_emision", "invoice_date", "date"),
    "sat_certification_date": ("sat_certification_date", "fecha_certificacion_sat", "satCertificationDate", "invoiceSignatureDate", "fecha_timbrado"),
    "cancellation_date": ("cancellation_date", "fecha_cancelacion", "cancellationDate"),
    "pac_certifier": ("pac_certifier", "pac", "rfc_pac", "pac_rfc", "provider_rfc"),
    "invoice_total": ("invoice_total", "total", "monto_total", "total_cfdi"),
    "source": ("source", "metadata_source"),
}
_ALLOWED_METADATA_KEYS = set(_METADATA_ALIAS_MAP.keys())


class FiscalAPIService:
    """Orquesta cache, cliente y feature flag para FiscalAPI."""

    def __init__(
        self,
        config: Optional[FiscalAPIConfig] = None,
        cache: Optional[FiscalValidationCache] = None,
        client: Optional[FiscalAPIClient] = None,
    ) -> None:
        self.config = config or get_fiscal_api_config()
        self.stage = get_fiscal_validation_stage()
        self.cache = cache or FiscalValidationCache(
            namespace=self.config.cache_namespace,
            ttl_seconds=self.config.cache_ttl_seconds,
        )

        self._client_owned = False
        self.client: Optional[FiscalAPIClient] = None
        if client is not None:
            self.client = client
        elif self.config.is_configured():
            try:
                self.client = FiscalAPIClient(self.config)
                self._client_owned = True
            except FiscalAPINotConfiguredError:
                logger.warning("FiscalAPIService: configuración incompleta, se operará en modo pendiente.")
                self.client = None
        else:
            logger.info("FiscalAPIService: FiscalAPI deshabilitado por falta de configuración.")

    def should_validate_document(self, document_type: str) -> bool:
        return document_type in self.config.document_types

    async def close(self) -> None:
        if self._client_owned and self.client:
            await self.client.aclose()

    async def validate_cfdi_status(self, request: RawRequest) -> FiscalValidationResult:
        request_obj = (
            request
            if isinstance(request, CFDIValidationRequest)
            else CFDIValidationRequest.model_validate(request)
        )
        base_metadata = self._sanitize_metadata(getattr(request_obj, "metadata", {}))

        if self.stage == "disabled":
            result = FiscalValidationResult.pending(request_obj, error="fiscal_validation_disabled")
            return self._attach_metadata(result, base_metadata)

        cache_key = request_obj.cache_key
        cached = self.cache.get(cache_key)
        if cached:
            self._record_event("cache_hit")
            return cached

        if not self.client:
            logger.warning("FiscalAPIService: cliente no disponible, devolviendo pendiente.")
            return FiscalValidationResult.pending(request_obj, error="client_not_available")

        start = time.perf_counter()
        outcome = "success"

        try:
            payload = await self._call_with_retry(request_obj)
            latency = time.perf_counter() - start
            metadata = await self._collect_metadata(request_obj, base_metadata)
            result = FiscalValidationResult.from_api_response(
                request=request_obj,
                response_payload=payload,
                latency_ms=int(latency * 1000),
                metadata=metadata,
            )
            result = self._attach_metadata(result, metadata)
            logger.debug(
                "FiscalAPIService: status=%s uuid=%s env=%s",
                result.status.value,
                request_obj.uuid,
                self.config.environment,
            )
            self.cache.set(cache_key, result)
            self._record_latency(latency)
            outcome = self._outcome_from_result(result)
            self._record_event(outcome)
            return result
        except FiscalAPINotFound as exc:
            latency = time.perf_counter() - start
            result = FiscalValidationResult(
                request=request_obj,
                status=FiscalValidationStatus.NO_ENCONTRADO,
                status_detail=str(exc),
                validation_timestamp=datetime.now(timezone.utc),
            )
            result = self._attach_metadata(result, base_metadata)
            self.cache.set(cache_key, result)
            self._record_latency(latency)
            self._record_event("not_found")
            return result
        except FiscalAPIUnauthorized as exc:
            logger.error("FiscalAPIService: credenciales inválidas (%s)", exc)
            self._record_event("unauthorized")
            result = FiscalValidationResult.pending(request_obj, error="unauthorized")
            return self._attach_metadata(result, base_metadata)
        except FiscalAPIUnavailable as exc:
            logger.warning("FiscalAPIService: indisponible (%s)", exc)
            self._record_event("unavailable")
            result = FiscalValidationResult.pending(request_obj, error=str(exc))
            return self._attach_metadata(result, base_metadata)
        except FiscalAPIError as exc:
            logger.error("FiscalAPIService: error genérico (%s)", exc)
            self._record_event("error")
            result = FiscalValidationResult.pending(request_obj, error=str(exc))
            return self._attach_metadata(result, base_metadata)

    async def _call_with_retry(self, request: CFDIValidationRequest) -> Dict[str, Any]:
        assert self.client is not None  # Protected by caller
        retrying = AsyncRetrying(
            stop=stop_after_attempt(max(1, self.config.max_retries)),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(FiscalAPIUnavailable),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self.client.validate_cfdi(request)
        raise FiscalAPIUnavailable("No fue posible validar CFDI tras reintentos")

    def _record_event(self, outcome: str) -> None:
        if not should_collect_fiscal_metrics():
            return
        VALIDATION_COUNTER.labels(
            outcome=outcome,
            environment=self.config.environment,
            stage=self.stage,
        ).inc()

    def _record_latency(self, latency_seconds: float) -> None:
        if not should_collect_fiscal_metrics():
            return
        VALIDATION_LATENCY.labels(
            environment=self.config.environment,
            stage=self.stage,
        ).observe(latency_seconds)

    async def _collect_metadata(
        self,
        request: CFDIValidationRequest,
        base_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(base_metadata)
        if not self.client:
            return merged
        try:
            api_details = await self.client.fetch_invoice_details(request)
        except (FiscalAPIUnavailable, FiscalAPIError, FiscalAPINotFound) as exc:  # pragma: no cover - resiliencia
            logger.debug("FiscalAPIService: metadata API fallback (%s)", exc)
            return merged
        if api_details:
            sanitized = self._sanitize_metadata(api_details)
            merged = self._merge_metadata(merged, sanitized)
        return merged

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not metadata:
            return {}
        sanitized: Dict[str, Any] = {}

        for canonical, aliases in _METADATA_ALIAS_MAP.items():
            value = None
            for alias in aliases:
                if alias in metadata:
                    value = metadata[alias]
                    break
            if value is None:
                continue

            if isinstance(value, (int, float, Decimal)):
                normalized = str(value)
            else:
                normalized = str(value).strip()

            if not normalized:
                continue

            sanitized[canonical] = normalized

        return sanitized

    @staticmethod
    def _merge_metadata(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        if not new:
            return base
        merged = dict(base)
        for key, value in new.items():
            if value and key not in merged:
                merged[key] = value
        return merged

    @staticmethod
    def _attach_metadata(result: FiscalValidationResult, metadata: Dict[str, Any]) -> FiscalValidationResult:
        if not metadata:
            return result
        merged_metadata: Dict[str, Any] = dict(result.metadata or {})
        for key, value in metadata.items():
            if value and key not in merged_metadata:
                merged_metadata[key] = value
        result.metadata = merged_metadata

        for field in (
            "issuer_name",
            "recipient_name",
            "invoice_effect",
            "issue_date",
            "sat_certification_date",
            "cancellation_date",
            "pac_certifier",
        ):
            current = getattr(result, field, None)
            if not current and merged_metadata.get(field):
                setattr(result, field, merged_metadata[field])
        return result

    @staticmethod
    def _outcome_from_result(result: FiscalValidationResult) -> str:
        if result.is_vigente():
            return "vigente"
        if result.is_cancelado():
            return "cancelado"
        if result.is_not_found():
            return "not_found"
        if result.is_efos():
            return "efos"
        return "success"
