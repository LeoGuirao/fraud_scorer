"""
Modelos para solicitudes y resultados de validación fiscal (CFDI).
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def _normalize_rfc(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return "".join(str(value).strip().upper().split())


def _normalize_uuid(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return str(value).strip().lower()


def _normalize_signature(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip()
    if len(normalized) < 8:
        return normalized.upper()
    return normalized


class FiscalValidationStatus(str, Enum):
    """Estados normalizados para un CFDI."""

    VIGENTE = "vigente"
    CANCELADO = "cancelado"
    NO_ENCONTRADO = "no_encontrado"
    PENDIENTE = "validacion_pendiente"
    ERROR = "error"

    @classmethod
    def from_raw(cls, value: Optional[str]) -> "FiscalValidationStatus":
        if not value:
            return cls.PENDIENTE
        lowered = str(value).strip().lower()
        if "vigen" in lowered or lowered == "active":
            return cls.VIGENTE
        if "cancel" in lowered or lowered.startswith("c"):
            return cls.CANCELADO
        if "no_encontrado" in lowered or "notfound" in lowered or lowered in {"404", "no encontrado"}:
            return cls.NO_ENCONTRADO
        if lowered in {"pendiente", "pending"}:
            return cls.PENDIENTE
        return cls.ERROR


class CFDIValidationRequest(BaseModel):
    """Datos imprescindibles para validar un CFDI."""

    issuer_rfc: str = Field(..., description="RFC del emisor (proveedor)")
    recipient_rfc: str = Field(..., description="RFC del receptor (aseguradora)")
    total: Decimal = Field(..., description="Total del CFDI con 6 decimales")
    uuid: str = Field(..., description="UUID del CFDI")
    signature_last_8: str = Field(..., description="Últimos 8 caracteres del sello digital")
    document_type: str = Field(..., description="Tipo de documento interno")
    sello_digital: Optional[str] = Field(default=None, description="Sello digital completo si está disponible")
    case_id: Optional[str] = Field(default=None)
    document_id: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Información auxiliar para enriquecer la validación")

    @field_validator("issuer_rfc", "recipient_rfc", mode="before")
    @classmethod
    def _upper_rfc(cls, value: Any) -> Any:
        normalized = _normalize_rfc(value)
        if not normalized:
            raise ValueError("RFC requerido para validación fiscal")
        return normalized

    @field_validator("uuid", mode="before")
    @classmethod
    def _validate_uuid(cls, value: Any) -> Any:
        normalized = _normalize_uuid(value)
        if not normalized:
            raise ValueError("UUID requerido para validación fiscal")
        return normalized

    @field_validator("signature_last_8", mode="before")
    @classmethod
    def _normalize_signature_last_8(cls, value: Any) -> Any:
        normalized = _normalize_signature(value)
        if not normalized or len(normalized) < 8:
            raise ValueError("Se requieren al menos 8 caracteres del sello digital")
        return normalized[-8:]

    @field_validator("document_type", mode="before")
    @classmethod
    def _normalize_doc_type(cls, value: Any) -> Any:
        if not value:
            raise ValueError("document_type es obligatorio")
        return str(value).strip()

    @field_validator("total", mode="before")
    @classmethod
    def _normalize_total(cls, value: Any) -> Any:
        if isinstance(value, Decimal):
            return value.quantize(Decimal("0.000001"))
        try:
            normalized = Decimal(str(value))
            return normalized.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        except (InvalidOperation, TypeError):
            raise ValueError("Total inválido para CFDI")

    @property
    def cache_key(self) -> str:
        total_str = format(self.total, "f")
        parts = (
            self.document_type,
            self.issuer_rfc,
            self.recipient_rfc,
            self.uuid,
            total_str,
            self.signature_last_8,
        )
        return ":".join(str(part) for part in parts if part is not None)

    def model_dump_safe(self) -> Dict[str, Any]:
        """Serialización para logs (oculta sello completo)."""
        payload = self.model_dump()
        if payload.get("sello_digital"):
            payload["sello_digital"] = f"{payload['sello_digital'][:10]}...<omitted>"
        if payload.get("metadata"):
            # Evitar volcar estructuras pesadas en logs protectivos
            payload["metadata"] = {k: payload["metadata"][k] for k in ("issuer_name", "recipient_name") if k in payload["metadata"]}
        return payload


class FiscalValidationResult(BaseModel):
    """Resultado estructurado de la validación contra FiscalAPI."""

    request: CFDIValidationRequest
    status: FiscalValidationStatus = FiscalValidationStatus.PENDIENTE
    status_code: Optional[str] = None
    status_detail: Optional[str] = None
    cancelable_status: Optional[str] = None
    cancelable_code: Optional[str] = None
    matches_total: Optional[bool] = None
    signature_valid: Optional[bool] = None
    efos_data: Dict[str, Any] = Field(default_factory=dict)
    validation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: Optional[int] = None
    issuer_name: Optional[str] = None
    recipient_name: Optional[str] = None
    invoice_effect: Optional[str] = None
    issue_date: Optional[str] = None
    sat_certification_date: Optional[str] = None
    cancellation_date: Optional[str] = None
    pac_certifier: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _inject_status(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        raw_status = values.get("status")
        if not isinstance(raw_status, FiscalValidationStatus):
            values["status"] = FiscalValidationStatus.from_raw(raw_status)
        return values

    @property
    def is_successful(self) -> bool:
        return self.status == FiscalValidationStatus.VIGENTE and not self.error

    def is_vigente(self) -> bool:
        return self.status == FiscalValidationStatus.VIGENTE

    def is_cancelado(self) -> bool:
        return self.status == FiscalValidationStatus.CANCELADO

    def is_not_found(self) -> bool:
        return self.status == FiscalValidationStatus.NO_ENCONTRADO

    def is_pending(self) -> bool:
        return self.status == FiscalValidationStatus.PENDIENTE

    def had_error(self) -> bool:
        return self.status == FiscalValidationStatus.ERROR or bool(self.error)

    def is_efos(self) -> bool:
        if not self.efos_data:
            return False
        if isinstance(self.efos_data, dict):
            listed = self.efos_data.get("listed")
            if isinstance(listed, bool):
                return listed
            status = str(self.efos_data.get("status") or "").lower()
            if status in {"listed", "efos", "positive"}:
                return True
            code = str(self.efos_data.get("code") or "").lower()
            if code in {"efos", "blacklist", "400"}:
                return True
        elif isinstance(self.efos_data, str):
            lowered = self.efos_data.lower()
            return "efos" in lowered or "blacklist" in lowered or lowered.startswith("4")
        return False

    def as_indicator_payload(self) -> Dict[str, Any]:
        """Payload base para generar FraudIndicator en FraudAnalyzer."""
        return {
            "status": self.status.value,
            "status_code": self.status_code,
            "status_detail": self.status_detail,
            "cancelable_status": self.cancelable_status,
            "cancelable_code": self.cancelable_code,
            "matches_total": self.matches_total,
            "signature_valid": self.signature_valid,
            "is_efos": self.is_efos(),
            "timestamp": self.validation_timestamp.isoformat(),
        }

    def get_fraud_flags(self) -> List[str]:
        flags: List[str] = []
        if self.is_cancelado():
            flags.append("cfdi_cancelado")
        if self.is_not_found():
            flags.append("uuid_no_encontrado")
        if self.is_efos():
            flags.append("emisor_efos")
        if self.matches_total is False:
            flags.append("totales_no_coinciden")
        if self.signature_valid is False:
            flags.append("sello_no_valido")
        if self.had_error():
            flags.append("validacion_error")
        return flags

    def to_case_index(self) -> Dict[str, Any]:
        payload = {
            "status": self.status.value,
            "status_code": self.status_code,
            "status_detail": self.status_detail,
            "cancelable_status": self.cancelable_status,
            "cancelable_code": self.cancelable_code,
            "matches_total": self.matches_total,
            "signature_valid": self.signature_valid,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "error": self.error,
            "latency_ms": self.latency_ms,
            "efos": self.efos_data,
            "issuer_name": self.issuer_name,
            "recipient_name": self.recipient_name,
            "invoice_effect": self.invoice_effect,
            "issue_date": self.issue_date,
            "sat_certification_date": self.sat_certification_date,
            "cancellation_date": self.cancellation_date,
            "pac_certifier": self.pac_certifier,
            "metadata": self.metadata or None,
        }
        if self.raw_response:
            payload["raw_response"] = self.raw_response
        return payload

    @classmethod
    def from_api_response(
        cls,
        request: CFDIValidationRequest,
        response_payload: Dict[str, Any],
        latency_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FiscalValidationResult":
        payload = response_payload.get("data") if "data" in response_payload else response_payload
        if not isinstance(payload, dict):
            payload = {}

        status = FiscalValidationStatus.from_raw(payload.get("status") or response_payload.get("status"))
        cancel_data = (
            payload.get("cancelable")
            or payload.get("cancelableStatus")
            or response_payload.get("cancelable")
            or {}
        )
        efos = (
            payload.get("efosValidation")
            or payload.get("efos_validation")
            or response_payload.get("efosValidation")
            or {}
        )
        matches_total = (
            payload.get("matchesTotal")
            or payload.get("matches_total")
            or response_payload.get("matchesTotal")
        )
        signature_valid = (
            payload.get("signatureValid")
            or payload.get("signature_valid")
            or response_payload.get("signatureValid")
        )
        normalized_metadata = cls._normalize_metadata(metadata or {}, request)

        return cls(
            request=request,
            status=status,
            status_code=payload.get("statusCode") or response_payload.get("statusCode"),
            status_detail=payload.get("statusDetail") or response_payload.get("statusDetail"),
            cancelable_status=cancel_data.get("status") if isinstance(cancel_data, dict) else cancel_data or None,
            cancelable_code=cancel_data.get("code") if isinstance(cancel_data, dict) else None,
            matches_total=bool(matches_total) if matches_total is not None else None,
            signature_valid=bool(signature_valid) if signature_valid is not None else None,
            efos_data=efos if isinstance(efos, dict) else {"status": efos} if efos else {},
            validation_timestamp=datetime.now(timezone.utc),
            raw_response=response_payload,
            latency_ms=latency_ms,
            issuer_name=normalized_metadata.get("issuer_name"),
            recipient_name=normalized_metadata.get("recipient_name"),
            invoice_effect=normalized_metadata.get("invoice_effect"),
            issue_date=normalized_metadata.get("issue_date"),
            sat_certification_date=normalized_metadata.get("sat_certification_date"),
            cancellation_date=normalized_metadata.get("cancellation_date"),
            pac_certifier=normalized_metadata.get("pac_certifier"),
            metadata=normalized_metadata,
        )

    @classmethod
    def pending(
        cls,
        request: CFDIValidationRequest,
        error: Optional[str] = None,
    ) -> "FiscalValidationResult":
        return cls(
            request=request,
            status=FiscalValidationStatus.PENDIENTE if not error else FiscalValidationStatus.ERROR,
            error=error,
        )

    @classmethod
    def _normalize_metadata(cls, metadata: Dict[str, Any], request: CFDIValidationRequest) -> Dict[str, Any]:
        if not metadata:
            # Aseguramos regresar el total aun si no hay datos adicionales
            return {"invoice_total": format(request.total, "f")}

        def _coerce(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, (int, float, Decimal)):
                return str(value)
            text = str(value).strip()
            return text or None

        def _first(*keys: str) -> Optional[str]:
            for key in keys:
                if key in metadata:
                    coerced = _coerce(metadata[key])
                    if coerced:
                        return coerced
            return None

        normalized: Dict[str, Any] = {
            "issuer_name": _first(
                "issuer_name",
                "issuerName",
                "issuer_tax_name",
                "emisor",
                "emisor_nombre",
            ),
            "recipient_name": _first(
                "recipient_name",
                "recipientName",
                "receptor",
                "receptor_nombre",
                "razón_social",
                "razon_social_receptor",
            ),
            "invoice_effect": _first(
                "invoice_effect",
                "effect",
                "cfdi_effect",
                "cfdiTypeDescription",
                "tipo_comprobante",
            ),
            "issue_date": _first(
                "issue_date",
                "fecha_expedicion",
                "fecha_emision",
                "invoice_date",
                "date",
            ),
            "sat_certification_date": _first(
                "sat_certification_date",
                "fecha_certificacion_sat",
                "satCertificationDate",
                "invoiceSignatureDate",
                "signature_date",
            ),
            "cancellation_date": _first(
                "cancellation_date",
                "fecha_cancelacion",
                "cancellationDate",
            ),
            "pac_certifier": _first(
                "pac_certifier",
                "pac",
                "pac_rfc",
                "rfc_pac",
                "provider_rfc",
            ),
            "invoice_total": _first(
                "invoice_total",
                "total",
                "monto_total",
            )
            or format(request.total, "f"),
        }

        # Elimina llaves con valores vacíos
        normalized = {key: value for key, value in normalized.items() if value}
        return normalized
