# src/fraud_scorer/parsers/document_parser.py
"""
DocumentParser: Orquestador para procesar múltiples formatos de documentos.

Este módulo identifica el tipo de archivo y delega el procesamiento al
lector/procesador correspondiente, devolviendo SIEMPRE una salida unificada:
{
  "text": str,
  "tables": List[Table],
  "key_value_pairs": Dict[str, Any],
  "metadata": DocumentMetadata
}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Lectores y tipos unificados
from fraud_scorer.parsers.types import (
    DocumentReader,
    ParsedDocument,
    OCR_EXTENSIONS,
)
from fraud_scorer.parsers.readers.azure_reader import AzureOCRReader

# OCR de Azure (inyectado al adapter)
from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor

# Librerías para formatos nativos
import docx
import pandas as pd


logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Orquesta el parsing de diferentes tipos de documentos.

    - Para imágenes/PDF: usa un lector OCR (adapter) que normaliza la salida.
    - Para DOCX/XLSX/CSV: usa parsers nativos y normaliza aquí mismo.
    """

    def __init__(self, ocr_processor: AzureOCRProcessor):
        """
        Inicializa el parser con los procesadores necesarios.

        Args:
            ocr_processor: Instancia del procesador OCR (AzureOCRProcessor).
        """
        # Adapter/Reader que encapsula Azure y devuelve salida unificada
        self.ocr_reader: DocumentReader = AzureOCRReader(ocr_processor)
        logger.info("DocumentParser inicializado con todos los procesadores.")

    def parse_document(self, doc_path: Path) -> Optional[ParsedDocument]:
        """
        Parsea un documento, seleccionando el método apropiado según su extensión.

        Args:
            doc_path: Ruta al documento a procesar.

        Returns:
            ParsedDocument con salida unificada o None si no es soportado o hay error.
        """
        if not doc_path.exists():
            logger.error(f"El archivo no existe: {doc_path}")
            return None

        # Ignorar archivos ocultos temporales (p. ej., ._archivo.pdf en macOS)
        if doc_path.name.startswith("._"):
            logger.warning(f"Omitiendo archivo temporal/oculto: {doc_path.name}")
            return None

        ext = doc_path.suffix.lower()
        logger.info(f"Iniciando parsing para: {doc_path.name} (tipo: {ext})")

        try:
            # 1) OCR para imágenes y PDFs
            if ext in OCR_EXTENSIONS:
                return self.ocr_reader.read(doc_path)

            # 2) DOCX
            if ext == ".docx":
                return self._parse_docx(doc_path)

            # 3) XLSX (Excel)
            if ext == ".xlsx":
                return self._parse_excel(doc_path)

            # 4) CSV
            if ext == ".csv":
                return self._parse_csv(doc_path)

            # 5) No soportado
            logger.warning(f"Formato no soportado: {ext} → {doc_path.name}")
            return None

        except Exception as e:
            logger.error(f"Error al parsear {doc_path.name}: {e}", exc_info=True)
            return None

    # ==========================
    # Parsers nativos unificados
    # ==========================

    def _parse_docx(self, doc_path: Path) -> ParsedDocument:
        """Parsea un archivo .docx a la salida unificada."""
        document = docx.Document(doc_path)
        full_text = "\n".join(p.text for p in document.paragraphs if p.text is not None)

        tables = []
        for t in document.tables:
            # Headers seguros (si hay filas)
            headers = [c.text for c in t.rows[0].cells] if t.rows and len(t.rows) > 0 else []
            data_rows = []
            if t.rows and len(t.rows) > 1:
                for r in t.rows[1:]:
                    data_rows.append([c.text for c in r.cells])

            tables.append({
                "headers": headers,
                "data_rows": data_rows,
            })

        return {
            "text": full_text,
            "tables": tables,
            "key_value_pairs": {},  # K-V se puede inferir luego con IA
            "metadata": {
                "source_type": "docx",
                "file_name": doc_path.name,
            },
        }

    def _parse_excel(self, doc_path: Path) -> ParsedDocument:
        """Parses .xlsx a salida unificada (cada hoja → tabla)."""
        xls = pd.ExcelFile(doc_path)
        text_parts = []
        tables = []

        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            # Texto completo de la hoja
            text_parts.append(f"--- Hoja: {sheet} ---\n{df.to_string(index=False)}")
            # Tabla normalizada
            tables.append({
                "sheet_name": sheet,
                "headers": df.columns.tolist(),
                "data_rows": df.where(pd.notnull(df), None).values.tolist(),  # NaN → None
            })

        return {
            "text": "\n\n".join(text_parts),
            "tables": tables,
            "key_value_pairs": {},
            "metadata": {
                "source_type": "xlsx",
                "file_name": doc_path.name,
                "sheets": xls.sheet_names,
            },
        }

    def _parse_csv(self, doc_path: Path) -> ParsedDocument:
        """Parses .csv a salida unificada (texto + tabla)."""
        try:
            df = pd.read_csv(doc_path, sep=None, engine="python", encoding="utf-8")
        except (UnicodeDecodeError, pd.errors.ParserError):
            logger.warning(f"No se pudo decodificar {doc_path.name} con UTF-8; intentando con latin1.")
            df = pd.read_csv(doc_path, sep=None, engine="python", encoding="latin1")

        return {
            "text": df.to_string(index=False),
            "tables": [{
                "headers": df.columns.tolist(),
                "data_rows": df.where(pd.notnull(df), None).values.tolist(),
            }],
            "key_value_pairs": {},
            "metadata": {
                "source_type": "csv",
                "file_name": doc_path.name,
            },
        }
