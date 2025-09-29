"""Utilidades para dividir documentos en chunks."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import RickAgentConfig, load_config


DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


@lru_cache(maxsize=8)
def get_splitter_for_type(document_type: str | None = None, *, config: RickAgentConfig | None = None) -> RecursiveCharacterTextSplitter:
    """Retorna un splitter configurado según el tipo de documento."""

    cfg = config or load_config()
    type_profile = _TYPE_PROFILES.get((document_type or "").lower())

    chunk_size = type_profile.get("chunk_size", cfg.chunk_size_default) if type_profile else cfg.chunk_size_default
    overlap = type_profile.get("chunk_overlap", cfg.chunk_overlap_default) if type_profile else cfg.chunk_overlap_default

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=type_profile.get("separators", DEFAULT_SEPARATORS) if type_profile else DEFAULT_SEPARATORS,
        keep_separator=False,
    )


@lru_cache(maxsize=1)
def get_default_splitter(*, config: RickAgentConfig | None = None) -> RecursiveCharacterTextSplitter:
    cfg = config or load_config()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size_default,
        chunk_overlap=cfg.chunk_overlap_default,
        separators=DEFAULT_SEPARATORS,
        keep_separator=False,
    )


_TYPE_PROFILES: Dict[str, Dict[str, int | list[str]]] = {
    "poliza_de_la_aseguradora": {"chunk_size": 1400, "chunk_overlap": 200},
    "contrato_prestacion_servicio_transportista": {"chunk_size": 1400, "chunk_overlap": 200},
    "informe_preliminar_del_ajustador": {"chunk_size": 1000, "chunk_overlap": 180},
    "informe_final_del_ajustador": {"chunk_size": 1000, "chunk_overlap": 180},
    "reporte_gps": {"chunk_size": 800, "chunk_overlap": 150},
    "analisis_fraude": {"chunk_size": 900, "chunk_overlap": 150},
}


__all__ = ["get_splitter_for_type", "get_default_splitter"]
