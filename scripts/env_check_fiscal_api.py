#!/usr/bin/env python3
"""
Verificador de configuración para FiscalAPI.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Dict, List

from dotenv import load_dotenv

from fraud_scorer.config import feature_flags
from fraud_scorer.config.fiscal_api_config import get_fiscal_api_config, invalidate_cache
from fraud_scorer.models.fiscal_validation import CFDIValidationRequest, FiscalValidationResult
from fraud_scorer.services.fiscal_api_service import FiscalAPIService


REQUIRED_ENV_VARS = [
    "FISCAL_API_KEY",
    "FISCAL_API_TENANT",
]

MISSING_HELP: Dict[str, str] = {
    "FISCAL_API_KEY": (
        "Genera una API Key desde el portal FiscalAPI: Developers » API Keys, "
        "asigna responsable y guarda el valor en un lugar seguro. "
        "Luego copia la clave en tu `.env` como FISCAL_API_KEY."
    ),
    "FISCAL_API_TENANT": (
        "Recupera tu Tenant Key (TID) desde el perfil de FiscalAPI; "
        "es el identificador único de la organización. "
        "Agrega ese valor en `.env` como FISCAL_API_TENANT."
    ),
}


def check_environment() -> Dict[str, bool]:
    status: Dict[str, bool] = {}
    for key in REQUIRED_ENV_VARS:
        value = os.getenv(key) or os.getenv(key.replace("FISCAL_API_", "FISCALAPI_"))
        status[key] = bool(value and value.strip())
    return status


async def probe_validation(service: FiscalAPIService) -> FiscalValidationResult:
    demo_request = CFDIValidationRequest(
        issuer_rfc=os.getenv("FISCAL_API_TEST_ISSUER", "AAA010101AAA"),
        recipient_rfc=os.getenv("FISCAL_API_TEST_RECIPIENT", "BBB020202BBB"),
        total=os.getenv("FISCAL_API_TEST_TOTAL", "1000.00"),
        uuid=os.getenv("FISCAL_API_TEST_UUID", "12345678-1234-1234-1234-123456789012"),
        signature_last_8=os.getenv("FISCAL_API_TEST_LAST8", "ABCDEF12"),
        document_type="cfdi_carta_porte",
        sello_digital=os.getenv("FISCAL_API_TEST_SEAL"),
        case_id="FISCAL_PROBE",
        document_id="TEST_CFDI",
    )
    return await service.validate_cfdi_status(demo_request)


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida configuración de FiscalAPI.")
    parser.add_argument("--environment", default=os.getenv("FISCAL_API_ENVIRONMENT", "test"), help="Ambiente objetivo (test/production).")
    parser.add_argument("--probe", action="store_true", help="Realiza una validación de prueba usando datos sandbox.")
    args = parser.parse_args()

    load_dotenv()
    os.environ["FISCAL_API_ENVIRONMENT"] = args.environment
    invalidate_cache()
    feature_flags.invalidate_caches()

    env_status = check_environment()
    missing = [key for key, ok in env_status.items() if not ok]

    config = get_fiscal_api_config()
    print("=== FiscalAPI Configuration ===")
    print(f"Environment   : {config.environment}")
    print(f"Base URL      : {config.base_url}")
    print(f"Cache TTL     : {config.cache_ttl_seconds}s")
    print(f"Document types: {', '.join(config.document_types)}")
    print()

    if missing:
        print("❌ Missing environment variables:")
        for key in missing:
            print(f"   - {key}")
            guidance = MISSING_HELP.get(key)
            if guidance:
                print(f"     → {guidance}")
        print("Consulta la guía CFDI_VALIDATION_INTEGRATION_GUIDE.md (sección Credenciales) "
              "para más detalles.")
        sys.exit(1)

    if not config.is_configured():
        print("❌ FiscalAPI configuration incomplete (revisa _config).")
        sys.exit(1)

    print("✅ Environment variables loaded correctly.")

    if args.probe:
        try:
            service = FiscalAPIService(config=config)
            result = asyncio.run(probe_validation(service))
            print("🔍 Probe result:")
            print(f"   Status       : {result.status.value}")
            print(f"   Status code  : {result.status_code or 'N/A'}")
            print(f"   Detail       : {result.status_detail or result.error or 'N/A'}")
            print(f"   Matches total: {result.matches_total}")
            print(f"   Signature ok : {result.signature_valid}")
            print(f"   Timestamp    : {result.validation_timestamp.isoformat()}")
            print(f"   Request      : {result.request.model_dump_safe()}")

            if result.is_pending():
                print(
                    "⚠️  La validación quedó pendiente. Revisa el flag FISCAL_VALIDATION_ROLLOUT "
                    "y la conectividad hacia FiscalAPI."
                )
            elif result.had_error():
                print(
                    "⚠️  FiscalAPI devolvió un error. Verifica que la API Key/Tenant sean válidos "
                    "y que el CFDI de prueba exista en el ambiente seleccionado."
                )
            elif result.is_not_found():
                print(
                    "ℹ️  CFDI no encontrado. Confirma que el UUID y totales configurados en las variables "
                    "FISCAL_API_TEST_* correspondan a un CFDI válido en el ambiente de pruebas."
                )
            else:
                print("✅ Validación completada correctamente.")
        except Exception as exc:  # pragma: no cover - CLI best effort
            print(f"⚠️  Probe failed: {exc}")
            sys.exit(2)
    else:
        print("ℹ️  Use --probe to perform a live validation against FiscalAPI sandbox.")

    print("OK")


if __name__ == "__main__":
    main()
