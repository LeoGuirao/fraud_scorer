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


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RickAgentConfig:
    """Valores de configuración cargados desde el entorno."""

    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 256
    chunk_size_default: int = 800
    chunk_overlap_default: int = 200
    min_documents: int = 2
    similarity_threshold: float = 0.35
    max_results: int = 15
    search_type: str = "hybrid"
    hybrid_alpha: float = 0.65
    lexical_top_k: int = 20
    dense_candidate_multiplier: int = 4
    mmr_lambda: float = 0.5
    cache_ttl_minutes: int = 30
    chroma_base_path: Path = Path("data/chroma")
    audit_log_path: Path = Path("data/logs/agent_rick_audit.jsonl")
    rate_limit: int = 30
    rate_window_minutes: int = 60
    openai_residency: Optional[str] = None
    agent_mode_enabled: bool = False
    agent_max_iterations: int = 5
    agent_llm_model: Optional[str] = None
    agent_gps_preview_limit: int = 50
    agent_gps_history_limit: int = 3
    agent_verbose: bool = False

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
        search_type=(
            str(
                overrides.get("search_type")
                or os.getenv("AGENTE_RICK_SEARCH_TYPE", RickAgentConfig.search_type)
            )
            .strip()
            .lower()
            or RickAgentConfig.search_type
        ),
        hybrid_alpha=min(
            1.0,
            max(
                0.0,
                float(
                    overrides.get("hybrid_alpha")
                    or _get_env_float("AGENTE_RICK_HYBRID_ALPHA", RickAgentConfig.hybrid_alpha)
                ),
            ),
        ),
        lexical_top_k=int(
            overrides.get("lexical_top_k")
            or _get_env_int("AGENTE_RICK_LEXICAL_TOP_K", RickAgentConfig.lexical_top_k)
        ),
        dense_candidate_multiplier=int(
            overrides.get("dense_candidate_multiplier")
            or _get_env_int(
                "AGENTE_RICK_DENSE_CANDIDATE_MULTIPLIER",
                RickAgentConfig.dense_candidate_multiplier,
            )
        ),
        mmr_lambda=min(
            1.0,
            max(
                0.0,
                float(
                    overrides.get("mmr_lambda")
                    or _get_env_float("AGENTE_RICK_MMR_LAMBDA", RickAgentConfig.mmr_lambda)
                ),
            ),
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
        agent_mode_enabled=bool(
            overrides.get("agent_mode_enabled")
            if overrides.get("agent_mode_enabled") is not None
            else _get_env_bool("AGENTE_RICK_AGENT_MODE_ENABLED", RickAgentConfig.agent_mode_enabled)
        ),
        agent_max_iterations=int(
            overrides.get("agent_max_iterations")
            or _get_env_int("AGENTE_RICK_AGENT_MAX_ITERATIONS", RickAgentConfig.agent_max_iterations)
        ),
        agent_llm_model=str(
            overrides.get("agent_llm_model")
            or os.getenv("AGENTE_RICK_AGENT_LLM_MODEL", "")  # permite vacío
        ).strip()
        or None,
        agent_gps_preview_limit=int(
            overrides.get("agent_gps_preview_limit")
            or _get_env_int("AGENTE_RICK_AGENT_GPS_PREVIEW_LIMIT", RickAgentConfig.agent_gps_preview_limit)
        ),
        agent_gps_history_limit=int(
            overrides.get("agent_gps_history_limit")
            or _get_env_int("AGENTE_RICK_AGENT_GPS_HISTORY_LIMIT", RickAgentConfig.agent_gps_history_limit)
        ),
        agent_verbose=bool(
            overrides.get("agent_verbose")
            if overrides.get("agent_verbose") is not None
            else _get_env_bool("AGENTE_RICK_AGENT_VERBOSE", RickAgentConfig.agent_verbose)
        ),
    )

    # Garantizar que las rutas existan
    base.chroma_base_path.mkdir(parents=True, exist_ok=True)
    base.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    return base


__all__ = ["RickAgentConfig", "load_config"]
