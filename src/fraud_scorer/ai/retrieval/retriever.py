"""Recuperación híbrida de documentos para el Agente Rick."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from langchain.schema import Document

from ..config import RickAgentConfig, load_config
from ..vector_store import RickVectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    document: Document
    similarity: float


@dataclass
class _DenseCandidate:
    index: int
    similarity: float
    vector_id: str


@dataclass
class _ScoreDetail:
    index: int
    dense: float
    lexical_raw: float
    lexical_norm: float
    combined: float


@dataclass
class _CaseCache:
    documents: List[Document]
    embeddings: np.ndarray
    normalized_embeddings: np.ndarray
    vector_ids: List[str]
    index_by_vector_id: Dict[str, int]
    bm25: Optional["_BM25Index"]
    vector_count: int


class RickRetriever:
    """Recuperador híbrido (BM25 + denso) con soporte para MMR."""

    def __init__(
        self,
        *,
        config: RickAgentConfig | None = None,
        vector_manager: RickVectorStoreManager | None = None,
    ) -> None:
        self.config = config or load_config()
        self.vector_manager = vector_manager or RickVectorStoreManager(config=self.config)
        self._case_cache: Dict[str, _CaseCache] = {}

    # ------------------------------------------------------------------
    # API pública

    def retrieve(self, case_id: str, query: str, *, k: int | None = None) -> List[RetrievedDocument]:
        store = self.vector_manager.load_store(case_id)
        limit = max(1, k or self.config.max_results)

        cache = self._ensure_case_cache(case_id, store)
        if cache.vector_count == 0:
            return []

        candidate_limit = max(limit * self.config.dense_candidate_multiplier, limit)
        dense_results = self._safe_similarity_search(store, query, candidate_limit)
        dense_candidates = self._prepare_dense_candidates(case_id, dense_results, cache)
        lexical_norm, lexical_raw = self._build_lexical_scores(cache, query)
        score_details = self._merge_scores(dense_candidates, lexical_norm, lexical_raw)
        if not score_details:
            return []

        ranked = self._select_candidates(score_details, cache, limit)
        retrieved: List[RetrievedDocument] = []
        for rank, detail in enumerate(ranked, start=1):
            doc = self._build_document(cache, detail, rank)
            retrieved.append(RetrievedDocument(document=doc, similarity=detail.combined))
        return retrieved

    # ------------------------------------------------------------------
    # Preparación de datos

    def _ensure_case_cache(self, case_id: str, store: "Chroma") -> _CaseCache:  # type: ignore[name-defined]
        try:
            vector_count = int(store._collection.count())  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - compatibilidad en tiempo de ejecución
            vector_count = None

        cache = self._case_cache.get(case_id)
        if cache and (vector_count is None or cache.vector_count == vector_count):
            return cache

        payload = store.get(include=["documents", "metadatas", "ids", "embeddings"])
        raw_documents = list(payload.get("documents") or [])
        metadatas = list(payload.get("metadatas") or [])
        vector_ids = list(payload.get("ids") or [])
        embeddings = payload.get("embeddings") or []

        count = min(len(raw_documents), len(metadatas), len(vector_ids))
        documents: List[Document] = []
        index_by_vector_id: Dict[str, int] = {}
        for idx in range(count):
            text = raw_documents[idx] or ""
            metadata = metadatas[idx] or {}
            if not isinstance(metadata, dict):
                metadata = dict(metadata)
            vector_id = vector_ids[idx]
            metadata.setdefault("vector_id", vector_id)
            documents.append(Document(page_content=text, metadata=metadata))
            index_by_vector_id[vector_id] = idx

        embedding_matrix = np.array(embeddings[:count], dtype=float) if embeddings else np.zeros((0, 0), dtype=float)
        if embedding_matrix.shape[0] != count:
            embedding_matrix = np.zeros((0, 0), dtype=float)
        normalized_embeddings = _normalize_embeddings(embedding_matrix)
        bm25_index = _BM25Index(documents) if documents else None

        cache = _CaseCache(
            documents=documents,
            embeddings=embedding_matrix,
            normalized_embeddings=normalized_embeddings,
            vector_ids=vector_ids[:count],
            index_by_vector_id=index_by_vector_id,
            bm25=bm25_index,
            vector_count=count,
        )
        self._case_cache[case_id] = cache
        return cache

    def _safe_similarity_search(self, store: "Chroma", query: str, k: int) -> List[tuple[Document, float]]:  # type: ignore[name-defined]
        try:
            return store.similarity_search_with_score(query, k=k)
        except Exception as exc:  # pragma: no cover - defensivo frente a chroma
            logger.error("Error consultando índice Chroma: %s", exc)
            return []

    def _prepare_dense_candidates(
        self,
        case_id: str,
        results: Sequence[tuple[Document, float]],
        cache: _CaseCache,
    ) -> List[_DenseCandidate]:
        candidates: List[_DenseCandidate] = []
        seen_indices: set[int] = set()
        for doc, distance in results:
            similarity = self._score_from_distance(distance)
            if similarity <= 0:
                continue
            vector_id = doc.metadata.get("vector_id") if isinstance(doc.metadata, dict) else None
            if not vector_id:
                vector_id = self._compute_document_id(case_id, doc)
            idx = cache.index_by_vector_id.get(vector_id)
            if idx is None or idx in seen_indices:
                continue
            seen_indices.add(idx)
            candidates.append(_DenseCandidate(index=idx, similarity=similarity, vector_id=vector_id))
        return candidates

    def _build_lexical_scores(
        self,
        cache: _CaseCache,
        query: str,
    ) -> tuple[Dict[int, float], Dict[int, float]]:
        if not cache.bm25:
            return {}, {}
        scores = cache.bm25.scores(query)
        if not scores:
            return {}, {}

        lexical_top_k = max(1, self.config.lexical_top_k)
        indices = np.argsort(np.array(scores, dtype=float))[::-1]
        top_indices = [int(idx) for idx in indices[:lexical_top_k] if scores[int(idx)] > 0]
        if not top_indices:
            return {}, {}

        raw_scores = {idx: float(scores[idx]) for idx in top_indices}
        max_score = max(raw_scores.values()) if raw_scores else 0.0
        normalized = {idx: (score / max_score) if max_score > 0 else 0.0 for idx, score in raw_scores.items()}
        return normalized, raw_scores

    def _merge_scores(
        self,
        dense_candidates: Sequence[_DenseCandidate],
        lexical_norm: Dict[int, float],
        lexical_raw: Dict[int, float],
    ) -> List[_ScoreDetail]:
        alpha = min(max(self.config.hybrid_alpha, 0.0), 1.0)
        score_map: Dict[int, _ScoreDetail] = {}

        for candidate in dense_candidates:
            score_map[candidate.index] = _ScoreDetail(
                index=candidate.index,
                dense=candidate.similarity,
                lexical_raw=0.0,
                lexical_norm=0.0,
                combined=candidate.similarity,
            )

        for idx, norm_score in lexical_norm.items():
            raw_score = lexical_raw.get(idx, 0.0)
            current = score_map.get(idx)
            if current:
                current.lexical_raw = raw_score
                current.lexical_norm = norm_score
            else:
                score_map[idx] = _ScoreDetail(
                    index=idx,
                    dense=0.0,
                    lexical_raw=raw_score,
                    lexical_norm=norm_score,
                    combined=0.0,
                )

        use_hybrid = lexical_norm and self.config.search_type.lower() in {"hybrid", "mmr"}
        for detail in score_map.values():
            if use_hybrid:
                detail.combined = alpha * detail.dense + (1 - alpha) * detail.lexical_norm
            else:
                detail.combined = detail.dense
        return [detail for detail in score_map.values() if detail.combined > 0 or detail.dense > 0]

    # ------------------------------------------------------------------
    # Selección y ensamblado

    def _select_candidates(self, details: List[_ScoreDetail], cache: _CaseCache, k: int) -> List[_ScoreDetail]:
        if not details:
            return []

        strategy = (self.config.search_type or "hybrid").lower()
        if strategy not in {"hybrid", "mmr", "similarity"}:
            strategy = "hybrid"

        details.sort(key=lambda item: item.combined, reverse=True)
        if strategy == "similarity" or cache.normalized_embeddings.size == 0:
            return details[:k]
        return self._apply_mmr(details, cache, k)

    def _apply_mmr(self, details: List[_ScoreDetail], cache: _CaseCache, k: int) -> List[_ScoreDetail]:
        if cache.normalized_embeddings.size == 0:
            return details[:k]

        mmr_lambda = min(max(self.config.mmr_lambda, 0.0), 1.0)
        selected: List[int] = []
        candidates = [detail.index for detail in details]
        score_lookup = {detail.index: detail.combined for detail in details}
        detail_lookup = {detail.index: detail for detail in details}
        embeddings = cache.normalized_embeddings

        while candidates and len(selected) < k:
            best_idx: Optional[int] = None
            best_score = -float("inf")
            for idx in candidates:
                relevance = score_lookup.get(idx, 0.0)
                if not selected:
                    mmr_score = relevance
                else:
                    diversity = max(float(np.dot(embeddings[idx], embeddings[other])) for other in selected)
                    mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * diversity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            candidates.remove(best_idx)

        return [detail_lookup[idx] for idx in selected]

    def _build_document(self, cache: _CaseCache, detail: _ScoreDetail, rank: int) -> Document:
        base = cache.documents[detail.index]
        metadata = dict(base.metadata or {})
        metadata.update(
            {
                "dense_similarity": round(detail.dense, 4),
                "lexical_score": round(detail.lexical_raw, 4),
                "lexical_score_normalized": round(detail.lexical_norm, 4),
                "hybrid_score": round(detail.combined, 4),
                "search_strategy": self.config.search_type,
                "retrieval_rank": rank,
            }
        )
        return Document(page_content=base.page_content, metadata=metadata)

    # ------------------------------------------------------------------
    # Utilidades internas

    @staticmethod
    def _score_from_distance(distance: float) -> float:
        try:
            similarity = 1.0 - float(distance)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(similarity):
            return 0.0
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def _compute_document_id(case_id: str, doc: Document) -> str:
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        source = str(metadata.get("source_document") or metadata.get("source") or "unknown")
        chunk_index = str(metadata.get("chunk_index") or 0)
        fingerprint = (
            str(metadata.get("hash"))
            or str(metadata.get("fingerprint"))
            or hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
        )
        raw = f"{case_id}:{source}:{chunk_index}:{fingerprint}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class _BM25Index:
    """Índice BM25 ligero para reponderar resultados densos."""

    def __init__(self, documents: Sequence[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tokenizer = _tokenize
        self.term_freqs: List[Counter[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.doc_lengths: List[int] = []

        for doc in documents:
            tokens = self._tokenizer(doc.page_content)
            counter = Counter(tokens)
            self.term_freqs.append(counter)
            self.doc_freq.update(counter.keys())
            self.doc_lengths.append(len(tokens))

        self.size = len(documents)
        self.avg_doc_len = sum(self.doc_lengths) / self.size if self.size else 0.0

    def scores(self, query: str) -> List[float]:
        tokens = self._tokenizer(query)
        if not tokens or not self.size:
            return [0.0] * self.size

        idf_cache: Dict[str, float] = {}
        denominator_cache: Dict[int, float] = {}
        results: List[float] = []

        for idx, term_freq in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_lengths[idx] or 1
            denominator_cache.setdefault(idx, self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1))))
            normalization = denominator_cache[idx]
            for token in tokens:
                freq = term_freq.get(token)
                if not freq:
                    continue
                idf = idf_cache.get(token)
                if idf is None:
                    df = self.doc_freq.get(token, 0)
                    idf = math.log(1 + (self.size - df + 0.5) / (df + 0.5)) if df else 0.0
                    idf_cache[token] = idf
                numerator = freq * (self.k1 + 1)
                score += idf * (numerator / (freq + normalization))
            results.append(score)
        return results


_TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    normalized = unicodedata.normalize("NFKD", text).lower()
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return _TOKEN_PATTERN.findall(ascii_text)


def _normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


__all__ = ["RickRetriever", "RetrievedDocument"]
