"""Configuración centralizada para el Agente Rick."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional

# Cargar variables de entorno desde .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv no es crítico


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class RickAgentConfig:
    """Valores de configuración cargados desde el entorno."""

    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 256
    chunk_size_default: int = 800
    chunk_overlap_default: int = 200
    min_documents: int = 2
    similarity_threshold: float = 0.20
    max_results: int = 6
    cache_ttl_minutes: int = 30
    chroma_base_path: Path = Path("data/chroma")
    audit_log_path: Path = Path("data/logs/agent_rick_audit.jsonl")
    rate_limit: int = 30
    rate_window_minutes: int = 60
    openai_residency: Optional[str] = None

    @property
    def chroma_global_snapshot(self) -> Path:
        return self.chroma_base_path / "_global_snapshot"


def load_config(overrides: Optional[dict[str, object]] = None) -> RickAgentConfig:
    """Carga la configuración desde variables de entorno y overrides opcionales."""

    overrides = overrides or {}
    base = RickAgentConfig(
        llm_model=str(overrides.get("llm_model") or os.getenv("AGENTE_RICK_LLM_MODEL", RickAgentConfig.llm_model)),
        embedding_model=str(
            overrides.get("embedding_model")
            or os.getenv("AGENTE_RICK_EMBEDDING_MODEL", RickAgentConfig.embedding_model)
        ),
        embedding_batch_size=int(
            overrides.get("embedding_batch_size")
            or _get_env_int("AGENTE_RICK_EMBED_BATCH", RickAgentConfig.embedding_batch_size)
        ),
        chunk_size_default=int(
            overrides.get("chunk_size_default")
            or _get_env_int("AGENTE_RICK_CHUNK_SIZE_DEFAULT", RickAgentConfig.chunk_size_default)
        ),
        chunk_overlap_default=int(
            overrides.get("chunk_overlap_default")
            or _get_env_int("AGENTE_RICK_CHUNK_OVERLAP_DEFAULT", RickAgentConfig.chunk_overlap_default)
        ),
        min_documents=int(
            overrides.get("min_documents")
            or _get_env_int("AGENTE_RICK_MIN_DOCS", RickAgentConfig.min_documents)
        ),
        similarity_threshold=float(
            overrides.get("similarity_threshold")
            or _get_env_float("AGENTE_RICK_SIMILARITY_THRESHOLD", RickAgentConfig.similarity_threshold)
        ),
        max_results=int(
            overrides.get("max_results")
            or _get_env_int("AGENTE_RICK_MAX_RESULTS", RickAgentConfig.max_results)
        ),
        cache_ttl_minutes=int(
            overrides.get("cache_ttl_minutes")
            or _get_env_int("AGENTE_RICK_CACHE_TTL_MINUTES", RickAgentConfig.cache_ttl_minutes)
        ),
        chroma_base_path=Path(
            overrides.get("chroma_base_path")
            or os.getenv("AGENTE_RICK_CHROMA_PATH", str(RickAgentConfig.chroma_base_path))
        ),
        audit_log_path=Path(
            overrides.get("audit_log_path")
            or os.getenv("AGENTE_RICK_AUDIT_PATH", str(RickAgentConfig.audit_log_path))
        ),
        rate_limit=int(
            overrides.get("rate_limit")
            or _get_env_int("AGENTE_RICK_RATE_LIMIT", RickAgentConfig.rate_limit)
        ),
        rate_window_minutes=int(
            overrides.get("rate_window_minutes")
            or _get_env_int("AGENTE_RICK_RATE_WINDOW_MINUTES", RickAgentConfig.rate_window_minutes)
        ),
   
        openai_residency=str(
            overrides.get("openai_residency")
            or os.getenv("AGENTE_RICK_OPENAI_RESIDENCY", "")
        ).strip() or None,
    )

    # Garantizar que las rutas existan
    base.chroma_base_path.mkdir(parents=True, exist_ok=True)
    base.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    return base


__all__ = ["RickAgentConfig", "load_config"]
