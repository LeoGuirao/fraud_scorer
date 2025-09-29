"""Recuperación de documentos para el Agente Rick."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import logging

from langchain.schema import Document

from ..config import RickAgentConfig, load_config
from ..vector_store import RickVectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    document: Document
    similarity: float


class RickRetriever:
    """Expone utilidades de búsqueda semántica usando Chroma."""

    def __init__(
        self,
        *,
        config: RickAgentConfig | None = None,
        vector_manager: RickVectorStoreManager | None = None,
    ) -> None:
        self.config = config or load_config()
        self.vector_manager = vector_manager or RickVectorStoreManager(config=self.config)

    def retrieve(self, case_id: str, query: str, *, k: int | None = None) -> List[RetrievedDocument]:
        store = self.vector_manager.load_store(case_id)
        k = k or self.config.max_results
        try:
            results = store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.error("Error consultando índice Chroma para %s: %s", case_id, exc)
            return []

        retrieved: List[RetrievedDocument] = []
        for doc, score in results:
            # Chroma devuelve distancia, convertimos a similitud simple (1 - distancia)
            similarity = 1 - float(score)
            retrieved.append(RetrievedDocument(document=doc, similarity=similarity))
        return retrieved


__all__ = ["RickRetriever", "RetrievedDocument"]
