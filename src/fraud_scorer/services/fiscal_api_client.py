"""
Cliente HTTP para FiscalAPI (validación de CFDIs).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

from fraud_scorer.config.fiscal_api_config import FiscalAPIConfig, get_fiscal_api_config
from fraud_scorer.models.fiscal_validation import CFDIValidationRequest

logger = logging.getLogger(__name__)


class FiscalAPIError(Exception):
    """Error genérico al comunicarse con FiscalAPI."""


class FiscalAPINotConfiguredError(FiscalAPIError):
    """Se intenta usar FiscalAPI sin credenciales base."""


class FiscalAPIUnauthorized(FiscalAPIError):
    """Credenciales inválidas o expiradas."""


class FiscalAPINotFound(FiscalAPIError):
    """El CFDI no existe en el SAT."""


class FiscalAPIUnavailable(FiscalAPIError):
    """Timeouts o indisponibilidad temporal."""


class FiscalAPIClient:
    """Pequeño wrapper sobre httpx.AsyncClient con instrumentación."""

    STATUS_ENDPOINT = "/api/v4/invoices/status"

    def __init__(
        self,
        config: Optional[FiscalAPIConfig] = None,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._config = config or get_fiscal_api_config()
        if not self._config.is_configured():
            raise FiscalAPINotConfiguredError("FiscalAPI no está configurado. Revisa variables de entorno.")

        self._client = client or httpx.AsyncClient(
            base_url=self._config.base_url.rstrip("/"),
            timeout=self._config.timeout_seconds,
            headers=self._config.headers(),
        )
        self._owns_client = client is None
        self._lock = asyncio.Lock()

    @property
    def config(self) -> FiscalAPIConfig:
        return self._config

    async def __aenter__(self) -> "FiscalAPIClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - sintaxis
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def validate_cfdi(self, request: CFDIValidationRequest) -> Dict[str, Any]:
        payload = self._build_payload(request)
        try:
            async with self._lock:  # serializa headers dinámicos si hiciera falta
                response = await self._client.post(self.STATUS_ENDPOINT, json=payload)
        except httpx.TimeoutException as exc:
            raise FiscalAPIUnavailable("FiscalAPI timeout") from exc
        except httpx.RequestError as exc:
            raise FiscalAPIUnavailable(f"FiscalAPI request error: {exc}") from exc
        return self._handle_response(response)

    async def fetch_invoice_details(self, request: CFDIValidationRequest) -> Optional[Dict[str, Any]]:
        """
        Intenta recuperar información adicional del CFDI usando el listado de facturas.
        No todos los tenants tienen visibilidad de CFDIs externos, por lo que este método
        puede devolver None sin considerar que sea un error.
        """
        params = {
            "invoiceUuid": request.uuid,
            "pageSize": 1,
        }
        # Aportar RFCs cuando están disponibles mejora la precisión si el tenant opera múltiples RFC.
        if request.issuer_rfc:
            params["issuerTin"] = request.issuer_rfc
        if request.recipient_rfc:
            params["recipientTin"] = request.recipient_rfc

        try:
            response = await self._client.get("/api/v4/invoices", params=params)
        except httpx.TimeoutException as exc:  # pragma: no cover - resiliencia
            raise FiscalAPIUnavailable("FiscalAPI timeout") from exc
        except httpx.RequestError as exc:  # pragma: no cover - resiliencia
            raise FiscalAPIUnavailable(f"FiscalAPI request error: {exc}") from exc

        payload = self._handle_response(response)
        items = []
        data_block = payload.get("data")
        if isinstance(data_block, dict):
            items = data_block.get("items") or []
        if not items:
            return None

        invoice = items[0] or {}
        details: Dict[str, Any] = {"source": "api_invoices"}

        issuer = invoice.get("issuer")
        if isinstance(issuer, dict):
            details["issuer_name"] = issuer.get("taxName") or issuer.get("name")

        receiver = invoice.get("recipient") or invoice.get("receiver")
        if isinstance(receiver, dict):
            details["recipient_name"] = receiver.get("taxName") or receiver.get("name")

        details["issue_date"] = invoice.get("date") or invoice.get("issueDate")
        details["invoice_effect"] = invoice.get("cfdiTypeDescription") or invoice.get("cfdiType")
        details["invoice_total"] = invoice.get("total")

        responses = invoice.get("responses")
        if isinstance(responses, list) and responses:
            response_entry = responses[0] or {}
            details.setdefault("sat_certification_date", response_entry.get("invoiceSignatureDate"))
            details.setdefault("pac_certifier", response_entry.get("provider") or response_entry.get("rfcPac"))
            details.setdefault("cancellation_date", response_entry.get("cancellationDate"))

        cancellation_info = invoice.get("cancellation") or {}
        if isinstance(cancellation_info, dict):
            details.setdefault("cancellation_date", cancellation_info.get("date"))

        return {k: v for k, v in details.items() if v}

    def _build_payload(self, request: CFDIValidationRequest) -> Dict[str, Any]:
        payload = {
            "issuerTin": request.issuer_rfc,
            "recipientTin": request.recipient_rfc,
            "invoiceTotal": format(request.total, ".6f"),
            "invoiceUuid": request.uuid,
            "last8DigitsIssuerSignature": request.signature_last_8,
        }
        if request.sello_digital:
            payload["digitalSeal"] = request.sello_digital
        return payload

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        status = response.status_code
        try:
            payload = response.json()
        except Exception:
            payload = {}

        if status == 401:
            raise FiscalAPIUnauthorized("FiscalAPI: 401 Unauthorized")
        if status == 404:
            raise FiscalAPINotFound("FiscalAPI: CFDI no encontrado")
        if status == 429:
            raise FiscalAPIUnavailable("FiscalAPI: rate limit alcanzado (429)")
        if status >= 500:
            raise FiscalAPIUnavailable(f"FiscalAPI: error {status}")
        if status >= 400:
            raise FiscalAPIError(f"FiscalAPI devolvió error {status}: {payload}")

        return payload
