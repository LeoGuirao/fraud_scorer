"""Servicio principal del Agente Rick con pipeline RAG."""

from __future__ import annotations

import json
import time
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING
import logging

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - hints only
    from langchain.agents import AgentExecutor

try:  # LangChain 0.1.x
    from langchain.agents import AgentExecutor as AgentExecutorRuntime, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:  # pragma: no cover - compatibilidad con versiones previas
    AgentExecutorRuntime = None  # type: ignore
    create_tool_calling_agent = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore
    MessagesPlaceholder = None  # type: ignore

from langchain_openai import ChatOpenAI

from ..config import RickAgentConfig, load_config
from ..ingestion import FraudCaseDocumentLoader
from ..prompting.templates import build_messages
from ..retrieval import RickRetriever, RetrievedDocument
from ..vector_store import RickVectorStoreManager
from ..tools import build_rag_search_tool, build_gps_query_tool
from fraud_scorer.utils.geo_reference import suggest_reference_point
from fraud_scorer.services.gps_query_service import GPSDirectQueryService

logger = logging.getLogger(__name__)


NO_CONTEXT_MESSAGE = "No encontré información suficiente en los documentos del caso."


@dataclass
class RickQueryResult:
    answer: str
    sources: list[dict[str, Any]]
    latency_ms: int = 0
    tokens_input: int | None = None
    tokens_output: int | None = None
    metadata: Dict[str, Any] | None = None


class AgenteRickService:
    """Servicio principal del Agente Rick con recuperación y generación reales."""

    def __init__(
        self,
        *,
        config: RickAgentConfig | None = None,
        llm: ChatOpenAI | None = None,
    ) -> None:
        self.config = config or load_config()
        self._loader = FraudCaseDocumentLoader(config=self.config)
        self._vector_manager = RickVectorStoreManager(config=self.config)
        self._retriever = RickRetriever(config=self.config, vector_manager=self._vector_manager)
        self._llm = llm or ChatOpenAI(model=self.config.llm_model, temperature=0)
        self._gps_service = GPSDirectQueryService()

        self._agent_enabled = bool(self.config.agent_mode_enabled)
        self._agent_executor: "AgentExecutor | None" = None
        self._agent_llm: ChatOpenAI | None = None
        self._agent_tools: list[Any] = []
        self._agent_last_documents: list[RetrievedDocument] = []
        self._agent_last_gps_queries: list[dict[str, Any]] = []
        self._reference_point_cache: dict[str, tuple[float, float]] = {}
        self._gps_window_cache: dict[str, tuple[str, str, int]] = {}
        self._gps_metrics_cache: dict[str, str] = {}
        self._gps_documents_cache: dict[str, List[Dict[str, Any]]] = {}
        self._case_index_cache: dict[str, Dict[str, Any]] = {}

        if self._agent_enabled:
            self._initialize_agent()

    def _initialize_agent(self) -> None:
        if not self._agent_enabled:
            return
        if not all([AgentExecutorRuntime, create_tool_calling_agent, ChatPromptTemplate, MessagesPlaceholder]):
            logger.warning("Modo agentic deshabilitado: dependencias de LangChain no disponibles.")
            self._agent_enabled = False
            return

        def _capture_docs(items: List[RetrievedDocument]) -> None:
            self._agent_last_documents = list(items or [])

        def _capture_gps(payload: Dict[str, Any]) -> None:
            self._agent_last_gps_queries.append(payload)
            limit = max(1, self.config.agent_gps_history_limit)
            if len(self._agent_last_gps_queries) > limit:
                self._agent_last_gps_queries = self._agent_last_gps_queries[-limit:]

        try:
            rag_tool = build_rag_search_tool(
                retriever=self._retriever,
                max_results=self.config.max_results,
                on_results=_capture_docs,
            )
            gps_tool = build_gps_query_tool(
                service=self._gps_service,
                default_limit=self.config.agent_gps_preview_limit,
                on_result=_capture_gps,
            )
            self._agent_tools = [rag_tool, gps_tool]

            model_name = self.config.agent_llm_model or self.config.llm_model
            self._agent_llm = ChatOpenAI(model=model_name, temperature=0)

            system_prompt = (
                "Eres Rick, un analista de fraude especializado. "
                "Usa search_case_documents para obtener contexto textual y query_gps_location "
                "para validar ubicaciones o rutas. "
                "Cuando la pregunta haga referencia a un tipo específico de documento (por ejemplo, 'Denuncia de los hechos', "
                "'Oficio de denuncia', 'Carpeta de investigación'), pasa el parámetro document_type con el valor canónico "
                "correspondiente: denuncia_de_los_hechos, oficio_denuncia, carpeta_investigacion, poliza, "
                "acta_hechos_transito o reporte_gps. "
                "Diferencia la fecha y hora del siniestro de la fecha de la denuncia; busca expresiones como "
                "\"siendo las … horas del día …\" o referencias a kilómetros específicos en la narrativa. "
                "Cuando necesites fecha u hora exactas, ejecuta al menos una búsqueda adicional con términos concretos "
                "('siendo las', 'kilómetro', 'martes', número de fecha) y prioriza fragmentos con metadata "
                "content_category=denuncia_narrative. "
                "\n\nCuándo usar cada herramienta:\n"
                "- search_case_documents → solo para recuperar narrativa textual (denuncias, informes, pólizas).\n"
                "- query_gps_location → siempre que el usuario pida validar ubicaciones, rutas o datos del monitoreo GPS. "
                "No intentes responder consultas de GPS únicamente con search_case_documents.\n"
                "\n\nPara query_gps_location:\n"
                "- El parámetro document_name debe ser el nombre exacto del archivo GPS mostrado en [DOCUMENTOS GPS DISPONIBLES]\n"
                "- Usa start_time/end_time para rangos O timestamp para un momento exacto (nunca ambos)\n"
                "- Formato de fechas: ISO 8601 con 'Z' (ej: '2024-02-13T19:00:00Z')\n"
                "- Si aparece [PUNTO DE REFERENCIA SUGERIDO: ...], úsalo directamente en reference_point={{'lat': ..., 'lon': ...}}\n"
                "- Si no hay sugerencia, estima coordenadas (por ejemplo, centro de la ciudad o tramo de carretera declarado) y utilízalas en reference_point "
                "para calcular distancias geográficas (se devolverán estadísticas en km)\n"
                "\nResponde siempre en español con el formato:\n"
                "Resumen:\n- ...\n\nDetalle:\n- ...\n\nFuentes:\n- Documento (fase, página si aplica)\n\n"
                "Cuando la información no esté disponible responde exactamente: "
                "\"No encontré información suficiente en los documentos del caso.\". "
                "Valida fechas y horas ambiguas solicitando precisión antes de concluir y menciona distancias cuando compares ubicaciones."
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            agent = create_tool_calling_agent(self._agent_llm, self._agent_tools, prompt)
            self._agent_executor = AgentExecutorRuntime(  # type: ignore[call-arg]
                agent=agent,
                tools=self._agent_tools,
                max_iterations=max(1, self.config.agent_max_iterations),
                verbose=self.config.agent_verbose,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
            )
        except Exception as exc:  # pragma: no cover - inicialización defensiva
            logger.error("No se pudo inicializar el modo agentic: %s", exc)
            self._agent_enabled = False
            self._agent_executor = None

    # ------------------------------------------------------------------
    # Indexación

    def index_case(self, case_id: str, *, rebuild: bool = False) -> None:
        documents = self._loader.load_case_documents(case_id)
        if rebuild:
            self._vector_manager.delete_case_index(case_id)
        self._vector_manager.upsert_documents(case_id, documents)
        self._reference_point_cache.pop(case_id, None)
        self._gps_window_cache.pop(case_id, None)
        self._gps_metrics_cache.pop(case_id, None)
        self._gps_documents_cache.pop(case_id, None)
        self._case_index_cache.pop(case_id, None)

    # ------------------------------------------------------------------
    # Consulta

    def query(
        self,
        *,
        case_id: str,
        question: str,
        user_id: Optional[str] = None,
        scope: str = "case",
        language: str = "es",
        module: Optional[str] = None,
    ) -> RickQueryResult:
        if not self._agent_enabled or self._agent_executor is None:
            result = self._query_rag_only(
                case_id=case_id,
                question=question,
                user_id=user_id,
                scope=scope,
                language=language,
                module=module,
            )
            result.metadata = result.metadata or {}
            result.metadata.setdefault("mode", "rag")
            return result

        return self._query_agentic(
            case_id=case_id,
            question=question,
            user_id=user_id,
            scope=scope,
            language=language,
            module=module,
        )

    def _query_rag_only(
        self,
        *,
        case_id: str,
        question: str,
        user_id: Optional[str],
        scope: str,
        language: str,
        module: Optional[str],
        status_override: Optional[str] = None,
    ) -> RickQueryResult:
        start = time.perf_counter()
        documents = self._retriever.retrieve(case_id, question, k=self.config.max_results)

        if not documents:
            answer = NO_CONTEXT_MESSAGE
            result = RickQueryResult(answer=answer, sources=[], latency_ms=_elapsed(start), metadata={"mode": "rag"})
            self._write_audit_entry(
                case_id=case_id,
                question=question,
                user_id=user_id,
                scope=scope,
                language=language,
                module=module,
                status=status_override or "no_context",
                answer=answer,
                retrieved_documents=[],
                latency_ms=result.latency_ms,
                context_quality="none",
            )
            return result

        similarities = [doc.similarity for doc in documents]
        best_similarity = max(similarities)
        adaptive_threshold = self._compute_adaptive_threshold(similarities)
        low_similarity = best_similarity < adaptive_threshold

        context_segments = [self._format_context_segment(item) for item in documents]
        if low_similarity:
            context_segments.insert(
                0,
                (
                    "⚠️ Contexto recuperado con baja similitud respecto a la pregunta. "
                    "Valida toda afirmación contra las citas disponibles y responde solo si encuentras evidencia explícita."
                ),
            )
        score_breakdown = self._build_score_breakdown(documents)
        similarity_histogram = self._build_similarity_histogram(similarities)
        messages = build_messages(context_segments, question)

        try:
            llm_response = self._llm.invoke(messages)
            metadata = getattr(llm_response, "response_metadata", {}) or {}
            token_usage = metadata.get("token_usage") or {}
        except Exception as exc:  # pragma: no cover - depende del proveedor
            logger.error("Error invocando LLM para caso %s: %s", case_id, exc)
            answer = "No se pudo generar una respuesta en este momento. Intenta nuevamente más tarde."
            latency_ms = _elapsed(start)
            fallback_result = RickQueryResult(
                answer=answer,
                sources=self._build_sources(documents),
                latency_ms=latency_ms,
                metadata={"mode": "rag", "llm_error": str(exc)},
            )
            self._write_audit_entry(
                case_id=case_id,
                question=question,
                user_id=user_id,
                scope=scope,
                language=language,
                module=module,
                status=status_override or "llm_error",
                answer=answer,
                retrieved_documents=documents,
                latency_ms=latency_ms,
                context_quality="low" if low_similarity else "normal",
                adaptive_threshold=adaptive_threshold,
            )
            return fallback_result

        latency_ms = _elapsed(start)
        sources = self._build_sources(documents)

        result = RickQueryResult(
            answer=llm_response.content.strip(),
            sources=sources,
            latency_ms=latency_ms,
            tokens_input=token_usage.get("prompt_tokens"),
            tokens_output=token_usage.get("completion_tokens"),
            metadata={
                "mode": "rag",
                "similarity_top": best_similarity,
                "retrieved": len(documents),
                "similarity_histogram": similarity_histogram,
                "score_breakdown": score_breakdown,
                "context_quality": "low" if low_similarity else "normal",
                "adaptive_threshold": adaptive_threshold,
            },
        )

        status_tag = status_override or ("answered_low_similarity" if low_similarity else "answered")
        self._write_audit_entry(
            case_id=case_id,
            question=question,
            user_id=user_id,
            scope=scope,
            language=language,
            module=module,
            status=status_tag,
            answer=result.answer,
            retrieved_documents=documents,
            latency_ms=latency_ms,
            token_usage=token_usage,
            similarity_histogram=similarity_histogram,
            score_breakdown=score_breakdown,
            context_quality="low" if low_similarity else "normal",
            adaptive_threshold=adaptive_threshold,
        )
        return result

    def _compute_adaptive_threshold(self, similarities: Sequence[float]) -> float:
        if not similarities:
            return self.config.similarity_threshold

        valid_scores = [score for score in similarities if isinstance(score, (int, float))]
        if not valid_scores:
            return self.config.similarity_threshold

        mean_value = sum(valid_scores) / len(valid_scores)
        variance = sum((score - mean_value) ** 2 for score in valid_scores) / len(valid_scores)
        std_dev = math.sqrt(max(variance, 0.0))

        candidate = mean_value + (std_dev * 0.5)
        lower_bound = self.config.similarity_threshold * 0.6
        return max(lower_bound, min(self.config.similarity_threshold, candidate))

    def _get_gps_documents_hint(self, case_id: str) -> str:
        """Genera un hint con los nombres de documentos GPS disponibles para el caso."""
        try:
            documents = self._gps_service.list_documents(case_id)
        except Exception:  # pragma: no cover - defensivo
            return ""

        self._gps_documents_cache[case_id] = documents

        if not documents:
            return ""

        gps_docs = [entry.get("document_name") for entry in documents if entry.get("document_name")]
        if not gps_docs:
            return ""

        docs_list = ", ".join(f"'{doc}'" for doc in gps_docs[:5])  # Limitar a 5 para no saturar
        return (
            f"\n[DOCUMENTOS GPS DISPONIBLES: {docs_list}]\n"
            "Consulta estos archivos únicamente con query_gps_location (no con search_case_documents)."
        )

    def _get_reference_point_hint(self, case_id: str) -> str:
        cached = self._reference_point_cache.get(case_id)
        if cached:
            lat, lon = cached
            return f"\n[PUNTO DE REFERENCIA SUGERIDO: lat={lat:.6f}, lon={lon:.6f}]"

        try:
            case_index = self._loader.cache_manager.get_case_index(case_id, auto_reconstruct=True) or {}
        except Exception:  # pragma: no cover - defensivo
            case_index = {}

        self._case_index_cache[case_id] = case_index

        consolidated = (case_index.get("consolidated_data") or {}).get("consolidated_fields") or {}
        location_text = consolidated.get("lugar_hechos")

        reference = suggest_reference_point(location_text) if location_text else None

        if not reference:
            return ""

        self._reference_point_cache[case_id] = reference
        lat, lon = reference
        return f"\n[PUNTO DE REFERENCIA SUGERIDO: lat={lat:.6f}, lon={lon:.6f}]"

    def _get_gps_window_hint(self, case_id: str, reference: Optional[tuple[float, float]]) -> str:
        if not reference:
            return ""

        cached = self._gps_window_cache.get(case_id)
        if cached:
            start_str, end_str, limit = cached
            return f"\n[VENTANA GPS SUGERIDA: start={start_str}, end={end_str}, limit={limit}]"

        documents = self._gps_documents_cache.get(case_id)
        if documents is None:
            try:
                documents = self._gps_service.list_documents(case_id)
            except Exception:  # pragma: no cover - defensivo
                documents = []
            self._gps_documents_cache[case_id] = documents

        lat_ref, lon_ref = reference
        tolerance = 0.015  # ≈1.5 km
        start_str = end_str = ""
        limit = 200

        case_index = self._case_index_cache.get(case_id)
        if case_index is None:
            try:
                case_index = self._loader.cache_manager.get_case_index(case_id, auto_reconstruct=True) or {}
            except Exception:
                case_index = {}
            self._case_index_cache[case_id] = case_index

        consolidated = (case_index.get("consolidated_data") or {}).get("consolidated_fields") or {}
        event_dt = _parse_event_datetime(
            consolidated.get("fecha_ocurrencia") or consolidated.get("fecha_siniestro"),
            consolidated.get("hora_siniestro") or consolidated.get("hora_ocurrencia"),
        )
        if event_dt:
            event_dt = event_dt.replace(microsecond=0)

        for entry in documents:
            doc_name = entry.get("document_name")
            if not doc_name:
                continue
            try:
                df = self._gps_service.cache.load_dataset(case_id, doc_name)
            except Exception:
                continue
            df = df.dropna(subset=["timestamp", "latitude", "longitude"])
            if df.empty:
                continue

            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            if df.empty:
                continue

            lat_series = pd.to_numeric(df["latitude"], errors="coerce")
            lon_series = pd.to_numeric(df["longitude"], errors="coerce")

            matches = pd.DataFrame()

            if event_dt is not None:
                window = timedelta(minutes=30)
                matches = df[
                    (df["timestamp"] >= event_dt - window) & (df["timestamp"] <= event_dt + window)
                ]

            if matches.empty:
                mask = (lat_series - lat_ref).abs() <= tolerance
                mask &= (lon_series - lon_ref).abs() <= tolerance
                matches = df[mask]

            if matches.empty:
                continue

            start_ts = matches["timestamp"].min()
            end_ts = matches["timestamp"].max()
            if not isinstance(start_ts, datetime) or not isinstance(end_ts, datetime):
                continue

            desired_span = timedelta(minutes=40)
            if end_ts - start_ts > desired_span:
                end_ts = start_ts + desired_span

            start_str = start_ts.replace(microsecond=0).isoformat()
            end_str = end_ts.replace(microsecond=0).isoformat()
            if not start_str.endswith("Z"):
                start_str += "Z"
            if not end_str.endswith("Z"):
                end_str += "Z"
            break

        if not start_str or not end_str:
            return ""

        self._gps_window_cache[case_id] = (start_str, end_str, limit)
        return f"\n[VENTANA GPS SUGERIDA: start={start_str}, end={end_str}, limit={limit}]"

    def _get_gps_metrics_hint(self, case_id: str) -> str:
        cached = self._gps_metrics_cache.get(case_id)
        if cached:
            return cached

        documents = self._gps_documents_cache.get(case_id)
        if documents is None:
            try:
                documents = self._gps_service.list_documents(case_id)
            except Exception:
                documents = []
            self._gps_documents_cache[case_id] = documents

        if not documents:
            return ""

        lines: List[str] = []
        for entry in documents:
            doc_name = entry.get("document_name")
            if not doc_name:
                continue
            try:
                df = self._gps_service.cache.load_dataset(case_id, doc_name)
            except Exception:
                continue
            df = df.dropna(subset=["timestamp", "latitude", "longitude"])
            if df.empty:
                continue

            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            df = df.dropna(subset=["timestamp", "latitude", "longitude"])
            if df.empty:
                continue

            total_distance = _compute_total_distance(df)
            stops = _find_long_stops(df)
            landmark_hits = _match_known_landmarks(df)
            checkpoints = _extract_checkpoints(df)

            line_parts = [f"- {doc_name}: distancia≈{total_distance:.1f} km"]
            if landmark_hits:
                casetas_text = "; ".join(
                    f"{hit.timestamp:%d/%m %H:%M} {hit.landmark.name}" for hit in landmark_hits[:4]
                )
                line_parts.append(f"casetas={casetas_text}")
                if checkpoints:
                    unmatched = [
                        cp
                        for cp in checkpoints
                        if all(
                            _haversine_km(cp[1], cp[2], hit.latitude, hit.longitude) > hit.landmark.radius_km
                            for hit in landmark_hits
                        )
                    ]
                else:
                    unmatched = []
            else:
                unmatched = checkpoints
            if unmatched:
                checkpoints_text = "; ".join(
                    f"{ts:%d/%m %H:%M} ({lat:.4f}, {lon:.4f})" for ts, lat, lon in unmatched
                )
                line_parts.append(f"checkpoints={checkpoints_text}")
            if stops:
                stops_text = "; ".join(
                    f"{start:%d/%m %H:%M}→{end:%H:%M} ({minutes} min)"
                    for start, end, minutes in stops[:2]
                )
                line_parts.append(f"detenciones={stops_text}")
            lines.append(" | ".join(line_parts))

        if not lines:
            return ""

        hint = "\n[RESUMEN GPS AGREGADO]\n" + "\n".join(lines)
        self._gps_metrics_cache[case_id] = hint
        return hint

    def _query_agentic(
        self,
        *,
        case_id: str,
        question: str,
        user_id: Optional[str],
        scope: str,
        language: str,
        module: Optional[str],
    ) -> RickQueryResult:
        if self._agent_executor is None:
            return self._query_rag_only(
                case_id=case_id,
                question=question,
                user_id=user_id,
                scope=scope,
                language=language,
                module=module,
            )

        self._agent_last_documents = []
        self._agent_last_gps_queries = []

        start = time.perf_counter()
        stripped_question = question.strip()
        cleaned_question = stripped_question if stripped_question else question

        # Obtener lista de documentos GPS disponibles para este caso
        gps_docs_hint = self._get_gps_documents_hint(case_id)
        reference_hint = self._get_reference_point_hint(case_id)
        reference_point = self._reference_point_cache.get(case_id)
        window_hint = self._get_gps_window_hint(case_id, reference_point)
        metrics_hint = self._get_gps_metrics_hint(case_id)

        payload = {
            "input": f"[CASO: {case_id}]{gps_docs_hint}{reference_hint}{window_hint}{metrics_hint}\n{cleaned_question}"
        }
        try:
            agent_response = self._agent_executor.invoke(payload)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - depende de LangChain/LLM
            logger.error("Error en modo agentic para caso %s: %s", case_id, exc)
            fallback = self._query_rag_only(
                case_id=case_id,
                question=question,
                user_id=user_id,
                scope=scope,
                language=language,
                module=module,
                status_override="agent_error_fallback",
            )
            fallback.metadata = fallback.metadata or {}
            fallback.metadata.update({"mode": "rag_fallback", "agent_error": str(exc)})
            return fallback

        latency_ms = _elapsed(start)
        output_text = ""
        intermediate_steps = []
        if isinstance(agent_response, dict):
            output_text = agent_response.get("output") or ""
            intermediate_steps = agent_response.get("intermediate_steps", [])
        else:
            output_text = str(agent_response or "")

        answer = output_text.strip() or NO_CONTEXT_MESSAGE
        sources = self._build_agent_sources()
        agent_steps = self._serialize_agent_steps(intermediate_steps)
        metadata = {
            "mode": "agent",
            "agent_steps": agent_steps,
            "gps_queries": list(self._agent_last_gps_queries),
            "retrieved": len(self._agent_last_documents),
        }

        result = RickQueryResult(
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            metadata=metadata,
        )

        status = "agent_answered" if answer != NO_CONTEXT_MESSAGE else "agent_no_context"
        self._write_audit_entry(
            case_id=case_id,
            question=question,
            user_id=user_id,
            scope=scope,
            language=language,
            module=module,
            status=status,
            answer=answer,
            retrieved_documents=self._agent_last_documents,
            latency_ms=latency_ms,
            agent_steps=agent_steps,
        )
        return result

    def _build_agent_sources(self) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        if self._agent_last_documents:
            sources.extend(self._build_sources(self._agent_last_documents))
        for payload in self._agent_last_gps_queries:
            entry = {
                "source_document": payload.get("document_name"),
                "source": "gps_direct",
                "document_type": "gps_dataset",
                "row_count": payload.get("row_count"),
                "filters": payload.get("filters"),
            }
            sources.append({key: value for key, value in entry.items() if value is not None})
        return sources

    def _serialize_agent_steps(self, steps: Any) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        if not steps:
            return serialized
        for item in steps:
            try:
                action, observation = item
            except (TypeError, ValueError):
                continue
            tool_name = getattr(action, "tool", None)
            tool_input = getattr(action, "tool_input", None)
            log = getattr(action, "log", None)
            observation_text = observation
            if not isinstance(observation_text, str):
                observation_text = repr(observation_text)
            if isinstance(observation_text, str) and len(observation_text) > 500:
                observation_text = observation_text[:500] + "...[truncado]"
            entry = {
                "tool": tool_name,
                "tool_input": str(tool_input)[:500] if tool_input is not None else None,
                "observation": observation_text,
                "log": log[:200] if isinstance(log, str) else None,
            }
            serialized.append({key: value for key, value in entry.items() if value})
        return serialized

    # ------------------------------------------------------------------
    # Utilidades internas

    def _build_sources(self, documents: list[RetrievedDocument]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for item in documents:
            meta = item.document.metadata if isinstance(item.document.metadata, dict) else {}
            entry = {
                "source_document": meta.get("source_document") or meta.get("case_path"),
                "source": meta.get("source"),
                "document_type": meta.get("document_type"),
                "chunk_index": meta.get("chunk_index"),
                "similarity": _safe_round(item.similarity),
                "dense_similarity": _safe_round(meta.get("dense_similarity")),
                "lexical_score": _safe_round(meta.get("lexical_score")),
                "lexical_score_normalized": _safe_round(meta.get("lexical_score_normalized")),
                "hybrid_score": _safe_round(meta.get("hybrid_score")),
                "retrieval_rank": meta.get("retrieval_rank"),
                "search_strategy": meta.get("search_strategy"),
            }
            sources.append({key: value for key, value in entry.items() if value is not None})
        return sources

    def _format_context_segment(self, item: RetrievedDocument) -> str:
        meta = item.document.metadata if isinstance(item.document.metadata, dict) else {}
        header_parts = [
            meta.get("source_document") or meta.get("case_path"),
            f"tipo: {meta.get('document_type')}" if meta.get("document_type") else None,
            f"fase: {meta.get('source')}" if meta.get("source") else None,
            f"similitud: {item.similarity:.2f}",
        ]
        dense_score = _safe_round(meta.get("dense_similarity"), digits=2)
        lexical_score = _safe_round(meta.get("lexical_score_normalized"), digits=2)
        if dense_score is not None:
            header_parts.append(f"denso: {dense_score:.2f}")
        if lexical_score is not None and lexical_score > 0:
            header_parts.append(f"lexical: {lexical_score:.2f}")
        header = " | ".join(part for part in header_parts if part)
        return f"[{header}]\n{item.document.page_content.strip()}"

    def _build_score_breakdown(self, documents: Sequence[RetrievedDocument]) -> list[dict[str, Any]]:
        breakdown: list[dict[str, Any]] = []
        for item in documents:
            meta = item.document.metadata if isinstance(item.document.metadata, dict) else {}
            entry = {
                "rank": meta.get("retrieval_rank"),
                "hybrid": _safe_round(meta.get("hybrid_score") or item.similarity),
                "dense": _safe_round(meta.get("dense_similarity")),
                "lexical": _safe_round(meta.get("lexical_score_normalized")),
                "lexical_raw": _safe_round(meta.get("lexical_score")),
            }
            breakdown.append({key: value for key, value in entry.items() if value is not None})
        return breakdown

    def _build_similarity_histogram(self, values: Sequence[float]) -> Dict[str, int]:
        bins = [round(step * 0.1, 1) for step in range(11)]
        histogram: Dict[str, int] = {f"{bins[idx]:.1f}-{bins[idx + 1]:.1f}": 0 for idx in range(len(bins) - 1)}
        histogram["1.0"] = 0

        for raw in values:
            try:
                score = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            score = max(0.0, min(1.0, score))
            placed = False
            for idx in range(len(bins) - 1):
                start, end = bins[idx], bins[idx + 1]
                if start <= score < end:
                    histogram[f"{start:.1f}-{end:.1f}"] += 1
                    placed = True
                    break
            if not placed:
                histogram["1.0"] += 1
        return histogram

    def _write_audit_entry(
        self,
        *,
        case_id: str,
        question: str,
        user_id: Optional[str],
        scope: str,
        language: str,
        module: Optional[str],
        status: str,
        answer: str,
        retrieved_documents: list[RetrievedDocument],
        latency_ms: Optional[int] = None,
        token_usage: Optional[dict] = None,
        similarity_histogram: Optional[Dict[str, int]] = None,
        score_breakdown: Optional[Sequence[dict[str, Any]]] = None,
        agent_steps: Optional[Sequence[dict[str, Any]]] = None,
        context_quality: Optional[str] = None,
        adaptive_threshold: Optional[float] = None,
    ) -> None:
        sources_payload = self._build_sources(retrieved_documents)
        histogram = similarity_histogram or self._build_similarity_histogram(
            [doc.similarity for doc in retrieved_documents]
        )
        breakdown = list(score_breakdown or self._build_score_breakdown(retrieved_documents))
        timestamp_str = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        entry = {
            "timestamp": timestamp_str,
            "case_id": case_id,
            "module": module or "general",
            "question": question,
            "user_id": user_id,
            "scope": scope,
            "language": language,
            "status": status,
            "answer_preview": answer[:200],
            "latency_ms": latency_ms,
            "sources": sources_payload,
            "token_usage": token_usage,
            "retrieved": sources_payload,
            "retrieved_count": len(retrieved_documents),
            "similarity_histogram": histogram,
            "score_breakdown": breakdown,
            "agent_steps": list(agent_steps or []),
            "context_quality": context_quality,
            "adaptive_threshold": adaptive_threshold,
        }
        try:
            self.config.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.audit_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover - defensivo
            logger.warning("No se pudo escribir auditoría para el Agente Rick: %s", exc)


def _parse_event_datetime(date_str: Optional[str], time_str: Optional[str]) -> Optional[datetime]:
    if not date_str or not time_str:
        return None
    date_str = date_str.strip()
    time_str = time_str.strip()
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            base_date = datetime.strptime(date_str, "%Y-%m-%d")
        elif re.match(r"^\d{2}/\d{2}/\d{4}$", date_str):
            base_date = datetime.strptime(date_str, "%d/%m/%Y")
        else:
            return None
    except ValueError:
        return None

    match = re.search(r"(\d{1,2})[:hH](\d{2})(?::(\d{2}))?\s*(am|pm|hrs|horas)?", time_str, re.IGNORECASE)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3) or 0)
        suffix = (match.group(4) or "").lower()
        if suffix in {"pm"} and hours < 12:
            hours += 12
        if suffix in {"am"} and hours == 12:
            hours = 0
    else:
        digits = re.findall(r"\d{2}", time_str)
        if len(digits) < 2:
            return None
        hours = int(digits[0])
        minutes = int(digits[1])
        seconds = int(digits[2]) if len(digits) > 2 else 0

    try:
        return base_date.replace(hour=hours, minute=minutes, second=seconds, microsecond=0)
    except ValueError:
        return None


def _compute_total_distance(df: pd.DataFrame) -> float:
    df = df.sort_values("timestamp")
    latitudes = pd.to_numeric(df["latitude"], errors="coerce")
    longitudes = pd.to_numeric(df["longitude"], errors="coerce")
    total = 0.0
    for lat1, lon1, lat2, lon2 in zip(
        latitudes[:-1], longitudes[:-1], latitudes[1:], longitudes[1:]
    ):
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            continue
        total += _haversine_km(float(lat1), float(lon1), float(lat2), float(lon2))
    return total


def _find_long_stops(df: pd.DataFrame, threshold: timedelta = timedelta(minutes=60)) -> List[tuple]:
    stops: List[tuple] = []
    timestamps = df["timestamp"].sort_values().reset_index(drop=True)
    diffs = timestamps.diff()
    for idx, delta in diffs.items():
        if pd.isna(delta) or delta < threshold:
            continue
        start = timestamps.iloc[idx - 1]
        end = timestamps.iloc[idx]
        minutes = int(delta.total_seconds() // 60)
        stops.append((start.to_pydatetime() if hasattr(start, "to_pydatetime") else start,
                      end.to_pydatetime() if hasattr(end, "to_pydatetime") else end,
                      minutes))
    return stops


def _extract_checkpoints(
    df: pd.DataFrame,
    max_points: int = 4,
    min_distance_km: float = 10.0,
) -> List[tuple]:
    checkpoints: List[tuple] = []
    df = df.sort_values("timestamp")
    latitudes = pd.to_numeric(df["latitude"], errors="coerce")
    longitudes = pd.to_numeric(df["longitude"], errors="coerce")
    timestamps = df["timestamp"]

    for ts, lat, lon in zip(timestamps, latitudes, longitudes):
        if pd.isna(lat) or pd.isna(lon):
            continue
        lat = float(lat)
        lon = float(lon)
        ts_value = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if not checkpoints:
            checkpoints.append((ts_value, lat, lon))
            continue
        distances = [_haversine_km(lat, lon, prev_lat, prev_lon) for _, prev_lat, prev_lon in checkpoints]
        if all(distance >= min_distance_km for distance in distances):
            checkpoints.append((ts_value, lat, lon))
        if len(checkpoints) >= max_points:
            break
    return checkpoints


@dataclass(frozen=True)
class GPSLandmark:
    name: str
    latitude: float
    longitude: float
    radius_km: float = 5.0


@dataclass(frozen=True)
class GPSLandmarkMatch:
    landmark: GPSLandmark
    timestamp: datetime
    latitude: float
    longitude: float
    distance_km: float


KNOWN_GPS_LANDMARKS: tuple[GPSLandmark, ...] = (
    GPSLandmark(name="Libramiento Poniente Tampico", latitude=22.2753, longitude=-97.8935, radius_km=4.0),
    GPSLandmark(name="Autopista Valles-Tamuín", latitude=21.9898, longitude=-99.1021, radius_km=4.0),
    GPSLandmark(name="Caseta Rayón", latitude=21.8644, longitude=-99.6182, radius_km=4.0),
    GPSLandmark(name="Cerritos-Rioverde", latitude=22.4321, longitude=-100.3241, radius_km=4.5),
)


def _match_known_landmarks(df: pd.DataFrame) -> List[GPSLandmarkMatch]:
    if df.empty:
        return []

    hits: List[GPSLandmarkMatch] = []
    coords = df.sort_values("timestamp")
    timestamps = coords["timestamp"]
    latitudes = coords["latitude"]
    longitudes = coords["longitude"]

    for landmark in KNOWN_GPS_LANDMARKS:
        best_entry: tuple[Any, float, float, float] | None = None
        for ts, lat, lon in zip(timestamps, latitudes, longitudes):
            if pd.isna(lat) or pd.isna(lon):
                continue
            lat_f = float(lat)
            lon_f = float(lon)
            distance = _haversine_km(lat_f, lon_f, landmark.latitude, landmark.longitude)
            if best_entry is None or distance < best_entry[3]:
                best_entry = (ts, lat_f, lon_f, distance)
        if not best_entry:
            continue
        ts, lat_f, lon_f, distance = best_entry
        if distance > landmark.radius_km:
            continue
        ts_value = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        hits.append(
            GPSLandmarkMatch(
                landmark=landmark,
                timestamp=ts_value,
                latitude=lat_f,
                longitude=lon_f,
                distance_km=distance,
            )
        )

    hits.sort(key=lambda item: item.timestamp)
    return hits


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def _elapsed(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _safe_round(value: Any, *, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


__all__ = ["AgenteRickService", "RickQueryResult"]
