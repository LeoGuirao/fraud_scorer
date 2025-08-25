"""
Tipos y contratos para el sistema de parsers.

Este módulo define:
- Estructuras de datos tipadas para el resultado unificado de parsing.
- El protocolo (interfaz) que deben implementar los lectores/adaptadores
  de documentos (p. ej., AzureOCRReader, GoogleVisionReader, etc.).
- Conjuntos de extensiones soportadas por tipo de fuente.

La idea es desacoplar la orquestación (DocumentParser) de los
proveedores/concretos, garantizando una salida homogénea:
{
  "text": str,
  "tables": List[Table],
  "key_value_pairs": Dict[str, Any],
  "metadata": DocumentMetadata
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Protocol, TypedDict, runtime_checkable


# ==========================
# Tipos de datos unificados
# ==========================

class Table(TypedDict, total=False):
    """
    Representación tabular unificada.
    Campos opcionales para adaptarse a diferentes fuentes (OCR, DOCX, XLSX, CSV).
    """
    # Identificadores / metadatos
    id: int
    sheet_name: str
    row_count: int
    column_count: int
    confidence: float

    # Vista "alta" nivel: headers + filas
    headers: List[str]
    data_rows: List[List[Any]]

    # Vista "baja" nivel (opcional) para OCR con celdas sueltas
    cells: List[Dict[str, Any]]


class DocumentMetadata(TypedDict, total=False):
    """
    Metadatos comunes de un documento parseado.
    """
    source_type: str             # "ocr" | "docx" | "xlsx" | "csv" | ...
    file_name: str
    file_size: int               # bytes (si está disponible)
    model_used: str              # para OCR (p. ej., "prebuilt-document")
    page_count: int              # para OCR/PDF
    language: str                # idioma detectado (si aplica)
    sheets: List[str]            # para XLSX
    engine: str                  # proveedor/subsistema (p. ej., "azure_document_intelligence")
    success: bool                # si el proveedor reporta éxito
    errors: List[str]            # lista de errores del proveedor
    confidence: Dict[str, float] # métricas de confianza (overall, etc.)


class ParsedDocument(TypedDict, total=False):
    """
    Salida unificada de cualquier lector/adaptador de documentos.
    """
    text: str
    tables: List[Table]
    key_value_pairs: Dict[str, Any]
    metadata: DocumentMetadata


# ==========================
# Protocolo de lectores
# ==========================

@runtime_checkable
class DocumentReader(Protocol):
    """
    Contrato que deben implementar los lectores/adaptadores de documentos.
    Deben devolver el resultado en el formato unificado `ParsedDocument`.
    """
    def read(self, path: Path) -> ParsedDocument:
        """
        Lee y normaliza un documento a la estructura unificada.

        Args:
            path: Ruta al archivo de entrada.

        Returns:
            ParsedDocument: diccionario con `text`, `tables`, `key_value_pairs`, `metadata`.
        """
        ...


# ==========================
# Extensiones soportadas
# ==========================

OCR_EXTENSIONS: set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
DOC_EXTENSIONS: set[str] = {".docx"}
SPREADSHEET_EXTENSIONS: set[str] = {".xlsx", ".csv"}

SUPPORTED_EXTENSIONS: set[str] = (
    OCR_EXTENSIONS | DOC_EXTENSIONS | SPREADSHEET_EXTENSIONS
)


__all__ = [
    "Table",
    "DocumentMetadata",
    "ParsedDocument",
    "DocumentReader",
    "OCR_EXTENSIONS",
    "DOC_EXTENSIONS",
    "SPREADSHEET_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
]
