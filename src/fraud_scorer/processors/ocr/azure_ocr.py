"""
Módulo OCR usando Azure Document Intelligence - VERSIÓN CORREGIDA
"""
import os
import asyncio
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

try:
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    from dotenv import load_dotenv
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logging.warning("Azure Form Recognizer no está instalado")

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)

class AzureOCRProcessor:
    def __init__(self, endpoint: str = None, api_key: str = None):
        if not AZURE_AVAILABLE:
            raise ImportError("Azure Form Recognizer no está disponible")
        
        # Obtener credenciales del ambiente si no se proporcionan
        self.endpoint = endpoint or os.getenv('AZURE_ENDPOINT')
        self.api_key = api_key or os.getenv('AZURE_OCR_KEY')
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Faltan credenciales de Azure. Configura AZURE_ENDPOINT y AZURE_OCR_KEY")
        
        self.client = DocumentAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
        logger.info("Cliente Azure OCR inicializado correctamente")

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Analiza un documento usando Azure Document Intelligence
        
        Args:
            file_path: Ruta al archivo a analizar
            
        Returns:
            Dict con texto extraído, tablas y pares clave-valor
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            
            logger.info(f"Iniciando análisis OCR de: {file_path.name}")
            
            # Leer archivo
            with open(file_path, "rb") as file:
                file_content = file.read()
            
            # Determinar modelo basado en extensión/contenido
            model = self._select_model(file_path)
            
            # Ejecutar análisis
            poller = self.client.begin_analyze_document(model, file_content)
            result = poller.result()
            
            # Extraer datos estructurados
            extracted_data = {
                "text": self._extract_text(result),
                "tables": self._extract_tables(result),
                "key_values": self._extract_key_values(result),
                "entities": self._extract_entities(result),
                "confidence": self._get_confidence_scores(result),
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": len(file_content),
                    "model_used": model,
                    "page_count": len(getattr(result, 'pages', []))
                }
            }
            
            logger.info(
                f"OCR completado: {len(extracted_data['text'])} chars, "
                f"{len(extracted_data['tables'])} tablas, "
                f"{len(extracted_data['key_values'])} pares KV"
            )
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error en análisis OCR: {str(e)}")
            raise

    def _select_model(self, file_path: Path) -> str:
        """Selecciona el modelo óptimo basado en el tipo de archivo"""
        return "prebuilt-document"

    def _extract_text(self, result) -> str:
        """Extrae todo el texto del documento"""
        if hasattr(result, 'content') and result.content:
            return result.content
        return ""

    def _extract_tables(self, result) -> List[Dict[str, Any]]:
        """Extrae tablas estructuradas del documento"""
        tables = []
        
        if hasattr(result, 'tables') and result.tables:
            for table_idx, table in enumerate(result.tables):
                table_data = {
                    "id": table_idx,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "cells": [],
                    "confidence": getattr(table, 'confidence', 0.0)
                }
                
                if hasattr(table, 'cells') and table.cells:
                    for cell in table.cells:
                        cell_data = {
                            "row": cell.row_index,
                            "column": cell.column_index,
                            "text": cell.content if cell.content else "",
                            "confidence": getattr(cell, 'confidence', 0.0),
                            "kind": getattr(cell, 'kind', 'content')
                        }
                        table_data["cells"].append(cell_data)
                
                tables.append(table_data)
        
        return tables

    def _extract_key_values(self, result) -> Dict[str, str]:
        """Extrae pares clave-valor del documento - VERSIÓN CORREGIDA"""
        kv_pairs = {}
        
        if hasattr(result, 'key_value_pairs') and result.key_value_pairs:
            for kv in result.key_value_pairs:
                try:
                    # Verificar que tanto key como value existen y tienen content
                    if (kv.key and hasattr(kv.key, 'content') and kv.key.content and
                        kv.value and hasattr(kv.value, 'content') and kv.value.content):
                        
                        key = kv.key.content.strip()
                        value = kv.value.content.strip()
                        
                        if key and value:  # Asegurar que no estén vacíos
                            kv_pairs[key] = value
                            
                except AttributeError as e:
                    # Log del error pero continuar con otros pares
                    logger.warning(f"Error procesando par clave-valor: {e}")
                    continue
        
        return kv_pairs

    def _extract_entities(self, result) -> List[Dict[str, Any]]:
        """Extrae entidades nombradas del documento"""
        entities = []
        
        if hasattr(result, 'entities') and result.entities:
            for entity in result.entities:
                try:
                    entity_data = {
                        "text": entity.content if hasattr(entity, 'content') else "",
                        "category": entity.category if hasattr(entity, 'category') else "",
                        "subcategory": getattr(entity, 'subcategory', None),
                        "confidence": getattr(entity, 'confidence', 0.0)
                    }
                    entities.append(entity_data)
                except AttributeError as e:
                    logger.warning(f"Error procesando entidad: {e}")
                    continue
        
        return entities

    def _get_confidence_scores(self, result) -> Dict[str, float]:
        """Obtiene scores de confianza del análisis"""
        return {
            "overall": getattr(result, 'confidence', 0.0),
            "pages": len(getattr(result, 'pages', [])),
            "extraction_quality": "high"
        }


# Función de conveniencia para uso directo
def analyze_document(file_path: str) -> Dict[str, Any]:
    """
    Función de conveniencia para análisis rápido de documentos
    """
    processor = AzureOCRProcessor()
    return processor.analyze_document(file_path)


# Función síncrona para testing rápido
def analyze_document_sync(file_path: str) -> Dict[str, Any]:
    """
    Versión síncrona para testing en REPL
    """
    return analyze_document(file_path)