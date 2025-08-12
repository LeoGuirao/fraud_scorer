# src/fraud_scorer/pipelines/mapper.py
from __future__ import annotations
from typing import Dict, Any
import os
from .field_map import FieldMap
from .semantic_alias import SemanticAliasMatcher
from .canonical import CanonicalFields
from .normalizers import norm_date, norm_money

_FIELD_MAP = FieldMap(os.getenv("FIELD_MAP_PATH", "src/fraud_scorer/config/field_map.yaml"))
_SEM = SemanticAliasMatcher(_FIELD_MAP.vocab()) if os.getenv("ENABLE_SEMANTIC_ALIAS", "true").lower() == "true" else None

def _first_scalar(v) -> str:
    if isinstance(v, (list, tuple)) and v:
        return str(v[0])
    if isinstance(v, dict) and v:
        return str(v.get("value") or next(iter(v.values()), ""))
    return "" if v is None else str(v)

def map_kv_to_canonical(doc_type: str, kv: Dict[str, Any]) -> CanonicalFields:
    canon: Dict[str, Any] = {}
    for k, v in (kv or {}).items():
        # 1) determinístico por YAML
        c = _FIELD_MAP.lookup(doc_type, k)
        # 2) fallback semántico
        if not c and _SEM:
            c, _ = _SEM.best(k)
        if c:
            val = _first_scalar(v)
            if val:
                canon.setdefault(c, val)
    # Normalizaciones clave
    if "fecha_siniestro" in canon: canon["fecha_siniestro"] = norm_date(canon["fecha_siniestro"])
    if "fecha_reclamacion" in canon: canon["fecha_reclamacion"] = norm_date(canon["fecha_reclamacion"])
    if "vigencia_inicio" in canon: canon["vigencia_inicio"] = norm_date(canon["vigencia_inicio"])
    if "vigencia_fin" in canon: canon["vigencia_fin"] = norm_date(canon["vigencia_fin"])
    if "claim_amount" in canon: canon["claim_amount"] = norm_money(canon["claim_amount"])
    return CanonicalFields(**canon)

def apply_canonical_mapping(extracted: Dict[str, Any], raw_ocr: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Toma un 'extracted' (de UniversalDocumentExtractor) y devuelve uno
    con 'specific_fields' canónicos fusionados.
    """
    doc_type = extracted.get("document_type", "otro")
    kv = extracted.get("key_value_pairs") or {}
    # preferimos kv del extractor; si no, intenta con raw_ocr
    if not kv and raw_ocr:
        kv = raw_ocr.get("key_value_pairs") or {}

    canon = map_kv_to_canonical(doc_type, kv).model_dump(exclude_none=True)
    # Merge con specific_fields existente
    sf = extracted.get("specific_fields") or {}
    merged_sf = {**sf, **canon}
    out = dict(extracted)
    out["specific_fields"] = merged_sf
    return out
