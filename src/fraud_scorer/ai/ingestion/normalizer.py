"""Normalización y deduplicación de documentos para el índice RAG."""

from __future__ import annotations

import hashlib
from typing import Iterable, List

from langchain.schema import Document


def _hash_content(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def deduplicate_documents(documents: Iterable[Document]) -> List[Document]:
    """Elimina documents con contenido repetido conservando el primero."""

    seen = set()
    unique_docs: List[Document] = []
    for doc in documents:
        fingerprint = doc.metadata.get("fingerprint")
        if not fingerprint:
            fingerprint = _hash_content(doc.page_content)
            doc.metadata["fingerprint"] = fingerprint
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_docs.append(doc)
    return unique_docs


__all__ = ["deduplicate_documents"]
