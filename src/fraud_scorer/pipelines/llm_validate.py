import os
from typing import Dict, Any
from .canonical import CanonicalFields
from .normalizers import norm_date, norm_money

def llm_structured_normalize(llm_client, bundle: Dict[str, Any]) -> CanonicalFields:
    # TODO: usar function calling/structured outputs con tu AsyncOpenAI
    raw = (bundle or {}).get("kv", {})
    data = {
        "numero_siniestro": raw.get("numero_siniestro"),
        "numero_poliza": raw.get("numero_poliza"),
        "fecha_siniestro": norm_date(raw.get("fecha_siniestro", "")),
        "vigencia_inicio": norm_date(raw.get("vigencia_inicio", "")),
        "vigencia_fin": norm_date(raw.get("vigencia_fin", "")),
        "claim_amount": norm_money(raw.get("claim_amount", "")),
    }
    return CanonicalFields(**data)