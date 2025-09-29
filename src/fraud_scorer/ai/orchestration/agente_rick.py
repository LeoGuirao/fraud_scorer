"""Servicio principal del Agente Rick con pipeline RAG."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import time
from typing import Any, Dict, Optional
import logging

from langchain_openai import ChatOpenAI

from ..config import RickAgentConfig, load_config
from ..ingestion import FraudCaseDocumentLoader
from ..prompting.templates import build_messages
from ..retrieval import RickRetriever, RetrievedDocument
from ..vector_store import RickVectorStoreManager

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

    # ------------------------------------------------------------------
    # Indexación

    def index_case(self, case_id: str, *, rebuild: bool = False) -> None:
        documents = self._loader.load_case_documents(case_id)
        if rebuild:
            self._vector_manager.delete_case_index(case_id)
        self._vector_manager.upsert_documents(case_id, documents)

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
        start = time.perf_counter()
        documents = self._retriever.retrieve(case_id, question, k=self.config.max_results)

        if not documents:
            answer = NO_CONTEXT_MESSAGE
            result = RickQueryResult(answer=answer, sources=[], latency_ms=_elapsed(start))
            self._write_audit_entry(
                case_id=case_id,
                question=question,
                user_id=user_id,
                scope=scope,
                language=language,
                module=module,
                status="no_context",
                answer=answer,
                retrieved_documents=[],
                latency_ms=result.latency_ms,
            )
            return result

        best_similarity = max(doc.similarity for doc in documents)
        if best_similarity < self.config.similarity_threshold:
            answer = NO_CONTEXT_MESSAGE
            result = RickQueryResult(answer=answer, sources=[], latency_ms=_elapsed(start))
            self._write_audit_entry(
                case_id=case_id,
                question=question,
                user_id=user_id,
                scope=scope,
                language=language,
                module=module,
                status="low_similarity",
                answer=answer,
                retrieved_documents=documents,
                latency_ms=result.latency_ms,
            )
            return result

        context_segments = [self._format_context_segment(item) for item in documents]
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
            )
            self._write_audit_entry(
                case_id=case_id,
                question=question,
                user_id=user_id,
                scope=scope,
                language=language,
                module=module,
                status="llm_error",
                answer=answer,
                retrieved_documents=documents,
                latency_ms=latency_ms,
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
                "similarity_top": best_similarity,
                "retrieved": len(documents),
            },
        )

        self._write_audit_entry(
            case_id=case_id,
            question=question,
            user_id=user_id,
            scope=scope,
            language=language,
            module=module,
            status="answered",
            answer=result.answer,
            retrieved_documents=documents,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
        return result

    # ------------------------------------------------------------------
    # Utilidades internas

    def _build_sources(self, documents: list[RetrievedDocument]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for item in documents:
            meta = item.document.metadata
            sources.append(
                {
                    "source_document": meta.get("source_document") or meta.get("case_path"),
                    "source": meta.get("source"),
                    "document_type": meta.get("document_type"),
                    "chunk_index": meta.get("chunk_index"),
                    "similarity": round(item.similarity, 4),
                }
            )
        return sources

    def _format_context_segment(self, item: RetrievedDocument) -> str:
        meta = item.document.metadata
        header_parts = [
            meta.get("source_document") or meta.get("case_path"),
            f"tipo: {meta.get('document_type')}" if meta.get("document_type") else None,
            f"fase: {meta.get('source')}" if meta.get("source") else None,
            f"similitud: {item.similarity:.2f}",
        ]
        header = " | ".join(part for part in header_parts if part)
        return f"[{header}]\n{item.document.page_content.strip()}"

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
    ) -> None:
        sources_payload = [
            {
                "source_document": doc.document.metadata.get("source_document")
                or doc.document.metadata.get("case_path"),
                "document_type": doc.document.metadata.get("document_type"),
                "similarity": round(doc.similarity, 4),
            }
            for doc in retrieved_documents
        ]
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
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
        }
        try:
            self.config.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.audit_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover - defensivo
            logger.warning("No se pudo escribir auditoría para el Agente Rick: %s", exc)


def _elapsed(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


__all__ = ["AgenteRickService", "RickQueryResult"]
