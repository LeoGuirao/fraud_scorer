"""CLI para construir índices RAG del Agente Rick."""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from langchain.schema import Document

from ..config import load_config
from ..ingestion import FraudCaseDocumentLoader
from ..vector_store import RickVectorStoreManager


logger = logging.getLogger(__name__)


def build_case_index(case_id: str, *, rebuild: bool = False) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY no está configurada. Configura la variable antes de construir el índice del Agente Rick."
        )
    config = load_config()
    loader = FraudCaseDocumentLoader(config=config)
    manager = RickVectorStoreManager(config=config)

    documents = loader.load_case_documents(case_id)
    resolved_case_id = loader.last_case_id or case_id

    if rebuild:
        manager.delete_case_index(resolved_case_id)

    try:
        manager.upsert_documents(resolved_case_id, documents)
    except Exception as exc:
        logger.error("Falló la generación del índice para %s: %s", resolved_case_id, exc)
        manager.delete_case_index(resolved_case_id)
        raise

    logger.info("Índice RAG actualizado para %s (%s documentos)", resolved_case_id, len(documents))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye el índice vectorial del Agente Rick")
    parser.add_argument("case_id", help="Identificador del caso a indexar")
    parser.add_argument("--rebuild", action="store_true", help="Recrea el índice desde cero")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    build_case_index(args.case_id, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
