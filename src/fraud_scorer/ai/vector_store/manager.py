"""Gestión de la base vectorial del Agente Rick."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import openai
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient
from chromadb.config import Settings

from ..config import RickAgentConfig, load_config

logger = logging.getLogger(__name__)


class RickVectorStoreManager:
    """Encapsula la interacción con Chroma para casos individuales."""

    def __init__(
        self,
        *,
        config: RickAgentConfig | None = None,
        embeddings: OpenAIEmbeddings | None = None,
    ) -> None:
        self.config = config or load_config()
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            headers = None
            if self.config.openai_residency:
                headers = {"OpenAI-Compute-Residency": self.config.openai_residency}
            self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model, default_headers=headers)

    # ---------------------------- paths ---------------------------------

    def _case_path(self, case_id: str) -> Path:
        path = self.config.chroma_base_path / case_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _manifest_path(self, case_id: str) -> Path:
        return self._case_path(case_id) / "index_manifest.json"

    # ---------------------------- main API ------------------------------

    def upsert_documents(self, case_id: str, documents: Iterable[Document]) -> None:
        docs = list(documents)
        if not docs:
            return

        store = self._get_store(case_id)
        max_batch_size = max(1, min(self.config.embedding_batch_size, len(docs)))
        current_batch_size = max_batch_size
        all_ids: list[str] = []
        processed = 0
        start = 0
        success_streak = 0

        while start < len(docs):
            end = min(start + current_batch_size, len(docs))
            original_batch = docs[start:end]
            original_ids = [self._build_document_id(case_id, doc) for doc in original_batch]

            for doc, doc_id in zip(original_batch, original_ids):
                if not isinstance(doc.metadata, dict):
                    doc.metadata = {}
                else:
                    doc.metadata = dict(doc.metadata)
                doc.metadata.setdefault("vector_id", doc_id)

            batch, ids = self._filter_new_records(store, original_batch, original_ids)

            if not ids:
                logger.debug(
                    "Lote %s-%s ya estaba indexado para %s; se omite re-inserción.",
                    start,
                    end,
                    case_id,
                )
                start = end
                success_streak = min(success_streak + 1, 3)
                continue

            try:
                store.add_documents(documents=batch, ids=ids)
                processed += len(ids)
                all_ids.extend(ids)
                success_streak += 1
                logger.debug(
                    "Embeddings generados para lote %s-%s (%s/%s) con batch=%s",
                    start,
                    end,
                    processed,
                    len(docs),
                    current_batch_size,
                )
                start = end

                if current_batch_size < max_batch_size and success_streak >= 3:
                    new_size = min(max_batch_size, current_batch_size * 2)
                    if new_size != current_batch_size:
                        logger.debug(
                            "Incrementando tamaño de lote para %s de %s a %s tras %s lotes estables.",
                            case_id,
                            current_batch_size,
                            new_size,
                            success_streak,
                        )
                        current_batch_size = new_size
                    success_streak = 0
            except openai.RateLimitError as exc:  # pragma: no cover - depende del proveedor
                success_streak = 0
                wait_seconds = getattr(exc, "retry_after", None)
                if wait_seconds is None:
                    wait_seconds = min(10.0, max(1.5, current_batch_size / 16))

                if current_batch_size > 1:
                    new_size = max(1, current_batch_size // 2)
                    logger.warning(
                        "OpenAI rate limit al generar embeddings para %s (%s-%s). Reduciendo lote de %s a %s y reintentando en %.1fs.",
                        case_id,
                        start,
                        end,
                        current_batch_size,
                        new_size,
                        wait_seconds,
                    )
                    current_batch_size = new_size
                else:
                    logger.warning(
                        "OpenAI rate limit sostenido para %s. Esperando %.1fs antes de reintentar lote unitario.",
                        case_id,
                        wait_seconds,
                    )
                time.sleep(wait_seconds)
                continue
            except openai.OpenAIError as exc:  # pragma: no cover - depende del proveedor
                success_streak = 0
                error_code = getattr(exc, "status_code", None)
                if error_code == 451:
                    raise RuntimeError(
                        "OpenAI rechazó la generación de embeddings (451 residency). Configura residency adecuado o utiliza otro proveedor."
                    ) from exc
                raise RuntimeError(
                    "Error generando embeddings con OpenAI. Ajusta el tamaño de lote o revisa tu configuración."
                ) from exc
            except Exception as exc:  # pragma: no cover - depende del proveedor
                success_streak = 0
                raise RuntimeError(
                    "Error generando embeddings (posible límite de tokens). Reduce el tamaño de lote con AGENTE_RICK_EMBED_BATCH."
                ) from exc

        try:
            self._write_manifest(case_id, docs)
            self._verify_persistence(case_id, all_ids)
        except Exception as exc:  # pragma: no cover - depende del proveedor
            raise RuntimeError(
                "Embeddings generados, pero falló la persistencia del índice. Revisa permisos de disco."
            ) from exc

    def load_store(self, case_id: str) -> Chroma:
        return self._get_store(case_id)

    def delete_case_index(self, case_id: str) -> None:
        path = self._case_path(case_id)
        if path.exists():
            for item in path.iterdir():
                if item.is_file():
                    item.unlink()
            for item in sorted(path.iterdir(), reverse=True):
                if item.is_dir():
                    self._remove_dir(item)
            path.rmdir()

    # ---------------------------- helpers -------------------------------

    def _filter_new_records(
        self,
        store: Chroma,
        batch: List[Document],
        ids: List[str],
    ) -> Tuple[List[Document], List[str]]:
        """Elimina IDs duplicados en memoria y aquellos ya persistidos."""

        unique_docs: List[Document] = []
        unique_ids: List[str] = []
        seen = set()

        for doc, doc_id in zip(batch, ids):
            if doc_id in seen:
                continue
            unique_docs.append(doc)
            unique_ids.append(doc_id)
            seen.add(doc_id)

        if not unique_ids:
            return unique_docs, unique_ids

        existing_ids: set[str] = set()
        try:
            payload = store.get(ids=unique_ids, include=[])
            existing_ids.update(payload.get("ids") or [])
        except Exception as exc:  # pragma: no cover - defensivo
            logger.debug("No se pudo verificar IDs existentes en Chroma: %s", exc)

        if not existing_ids:
            return unique_docs, unique_ids

        filtered_docs: List[Document] = []
        filtered_ids: List[str] = []
        for doc, doc_id in zip(unique_docs, unique_ids):
            if doc_id in existing_ids:
                continue
            filtered_docs.append(doc)
            filtered_ids.append(doc_id)

        return filtered_docs, filtered_ids

    def _get_store(self, case_id: str) -> Chroma:
        case_path = self._case_path(case_id)
        settings = Settings(anonymized_telemetry=False)
        client = PersistentClient(path=str(case_path), settings=settings)

        return Chroma(
            collection_name=self._collection_name(case_id),
            embedding_function=self.embeddings,
            client=client,
        )

    def _build_document_id(self, case_id: str, document: Document) -> str:
        source = str(document.metadata.get("source_document") or document.metadata.get("source") or "unknown")
        chunk_index = str(document.metadata.get("chunk_index") or 0)
        fingerprint = (
            str(document.metadata.get("hash"))
            or str(document.metadata.get("fingerprint"))
            or hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()
        )
        raw = f"{case_id}:{source}:{chunk_index}:{fingerprint}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _write_manifest(self, case_id: str, documents: List[Document]) -> None:
        manifest = {
            "case_id": case_id,
            "vector_count": len(documents),
            "chunk_params": {
                "chunk_size": self.config.chunk_size_default,
                "chunk_overlap": self.config.chunk_overlap_default,
            },
            "sources": sorted(
                {doc.metadata.get("source") for doc in documents if doc.metadata.get("source")}
            ),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        with self._manifest_path(case_id).open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)

    def _remove_dir(self, path: Path) -> None:
        for item in path.iterdir():
            if item.is_dir():
                self._remove_dir(item)
            else:
                item.unlink()
        path.rmdir()

    def _collection_name(self, case_id: str) -> str:
        base = f"case_{case_id}" if not case_id.startswith("case_") else case_id
        sanitized = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
        sanitized = sanitized.strip("._-") or "case_index"
        return sanitized.lower()

    def _verify_persistence(self, case_id: str, ids: list[str]) -> None:
        if not ids:
            return

        try:
            reloaded_store = self._get_store(case_id)
            chunk = 50
            total_found = 0
            for start in range(0, len(ids), chunk):
                subset = ids[start : start + chunk]
                payload = reloaded_store.get(ids=subset, include=[])
                total_found += len(payload.get("ids") or [])
            if total_found < len(ids):
                raise RuntimeError(
                    f"Persistencia incompleta para {case_id}: {total_found}/{len(ids)} embeddings disponibles tras reload."
                )
        except Exception as exc:  # pragma: no cover - defensivo
            raise RuntimeError(
                "El índice se generó pero no quedó accesible tras recarga. Revisa la configuración de Chroma."
            ) from exc


__all__ = ["RickVectorStoreManager"]
