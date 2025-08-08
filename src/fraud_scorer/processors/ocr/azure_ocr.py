"""
Módulo OCR usando Azure Document Intelligence - VERSIÓN MEJORADA
"""
import os
import asyncio
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError, ClientAuthenticationError
    from dotenv import load_dotenv
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logging.warning("Azure Form Recognizer no está instalado")

load_dotenv()
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Tipos de documento soportados"""
    FACTURA = "factura"
    DENUNCIA = "denuncia"
    IDENTIFICACION = "identificacion"
    LICENCIA = "licencia"
    POLIZA = "poliza"
    PERITAJE = "peritaje"
    CARTA_PORTE = "carta_porte"
    GENERAL = "general"

@dataclass
class OCRResult:
    """Resultado estructurado del OCR"""
    text: str
    tables: List[Dict[str, Any]]
    key_values: Dict[str, str]
    entities: List[Dict[str, Any]]
    confidence: Dict[str, float]
    metadata: Dict[str, Any]
    errors: List[str]
    success: bool

class AzureOCRProcessor:
    def __init__(self, endpoint: str = None, api_key: str = None):
        if not AZURE_AVAILABLE:
            raise ImportError("Azure Form Recognizer no está disponible. Instala: pip install azure-ai-formrecognizer")
        
        self.endpoint = endpoint or os.getenv('AZURE_ENDPOINT')
        self.api_key = api_key or os.getenv('AZURE_OCR_KEY')
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Faltan credenciales de Azure. Configura AZURE_ENDPOINT y AZURE_OCR_KEY en .env"
            )
        
        try:
            self.client = DocumentAnalysisClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key)
            )
            logger.info("Cliente Azure OCR inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando cliente Azure: {e}")
            raise

    async def analyze_document_async(self, file_path: str) -> OCRResult:
        """
        Versión asíncrona del análisis de documentos
        """
        return await asyncio.to_thread(self.analyze_document, file_path)

    def analyze_document(self, file_path: str) -> OCRResult:
        """
        Analiza un documento con manejo robusto de errores
        """
        errors = []
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                error_msg = f"Archivo no encontrado: {file_path}"
                logger.error(error_msg)
                return self._create_error_result(error_msg)
            
            # Validar tamaño del archivo
            file_size = file_path.stat().st_size
            if file_size > 50 * 1024 * 1024:  # 50MB límite
                error_msg = f"Archivo demasiado grande: {file_size / 1024 / 1024:.2f}MB (máx 50MB)"
                logger.error(error_msg)
                return self._create_error_result(error_msg)
            
            logger.info(f"Iniciando análisis OCR de: {file_path.name} ({file_size / 1024:.2f}KB)")
            
            with open(file_path, "rb") as file:
                file_content = file.read()
            
            model = self._select_model(file_path)
            
            # Análisis con timeout y reintentos
            try:
                poller = self.client.begin_analyze_document(
                    model, 
                    file_content,
                    locale="es-MX"  # Español de México para mejor precisión
                )
                result = poller.result()
                
            except ClientAuthenticationError as e:
                error_msg = "Error de autenticación con Azure. Verifica tus credenciales."
                logger.error(f"{error_msg}: {e}")
                return self._create_error_result(error_msg)
                
            except HttpResponseError as e:
                if e.status_code == 429:
                    error_msg = "Límite de rate excedido. Intenta más tarde."
                elif e.status_code == 413:
                    error_msg = "Documento demasiado grande para procesar."
                else:
                    error_msg = f"Error HTTP {e.status_code}: {e.message}"
                logger.error(error_msg)
                return self._create_error_result(error_msg)
            
            # Procesar resultado
            extracted_data = self._process_result(result, file_path, file_size, model)
            
            # Validar calidad del OCR
            if len(extracted_data.text) < 50:
                errors.append("Texto extraído muy corto, posible problema de calidad")
            
            if extracted_data.confidence["overall"] < 0.5:
                errors.append("Confianza baja en el reconocimiento")
            
            extracted_data.errors = errors
            extracted_data.success = len(errors) == 0
            
            logger.info(
                f"OCR completado: {len(extracted_data.text)} chars, "
                f"confianza: {extracted_data.confidence['overall']:.2%}"
            )
            
            return extracted_data
            
        except Exception as e:
            error_msg = f"Error inesperado en análisis OCR: {str(e)}"
            logger.exception(error_msg)
            return self._create_error_result(error_msg)

    def _process_result(self, result, file_path: Path, file_size: int, model: str) -> OCRResult:
        """Procesa el resultado del OCR de forma segura"""
        return OCRResult(
            text=self._extract_text(result),
            tables=self._extract_tables(result),
            key_values=self._extract_key_values(result),
            entities=self._extract_entities(result),
            confidence=self._get_confidence_scores(result),
            metadata={
                "file_name": file_path.name,
                "file_size": file_size,
                "model_used": model,
                "page_count": len(getattr(result, 'pages', [])),
                "language": getattr(result, 'language', 'es')
            },
            errors=[],
            success=True
        )

    def _create_error_result(self, error_msg: str) -> OCRResult:
        """Crea un resultado de error estructurado"""
        return OCRResult(
            text="",
            tables=[],
            key_values={},
            entities=[],
            confidence={"overall": 0.0},
            metadata={},
            errors=[error_msg],
            success=False
        )

    def _select_model(self, file_path: Path) -> str:
        """Selecciona el modelo óptimo basado en el nombre del archivo"""
        filename_lower = file_path.name.lower()
        
        # Mapeo de palabras clave a modelos especializados
        model_mappings = {
            "prebuilt-invoice": ["factura", "invoice", "cfdi", "recibo"],
            "prebuilt-receipt": ["ticket", "nota", "comprobante"],
            "prebuilt-idDocument": ["ine", "ife", "identificacion", "credencial", "pasaporte"],
            "prebuilt-businessCard": ["tarjeta", "card"],
        }
        
        for model, keywords in model_mappings.items():
            if any(keyword in filename_lower for keyword in keywords):
                logger.info(f"Usando modelo especializado: {model}")
                return model
        
        return "prebuilt-document"

    def _extract_text(self, result) -> str:
        """Extrae texto con manejo seguro"""
        try:
            if hasattr(result, 'content') and result.content:
                return result.content.strip()
        except Exception as e:
            logger.warning(f"Error extrayendo texto: {e}")
        return ""

    def _extract_tables(self, result) -> List[Dict[str, Any]]:
        """Extrae tablas con estructura mejorada"""
        tables = []
        
        try:
            if hasattr(result, 'tables') and result.tables:
                for table_idx, table in enumerate(result.tables):
                    table_data = {
                        "id": table_idx,
                        "row_count": getattr(table, 'row_count', 0),
                        "column_count": getattr(table, 'column_count', 0),
                        "cells": [],
                        "confidence": getattr(table, 'confidence', 0.0),
                        "headers": [],  # Intentar detectar headers
                        "data_rows": []  # Filas de datos estructuradas
                    }
                    
                    if hasattr(table, 'cells') and table.cells:
                        # Organizar celdas por filas
                        rows_dict = {}
                        for cell in table.cells:
                            row_idx = cell.row_index
                            col_idx = cell.column_index
                            
                            if row_idx not in rows_dict:
                                rows_dict[row_idx] = {}
                            
                            rows_dict[row_idx][col_idx] = {
                                "text": cell.content if cell.content else "",
                                "confidence": getattr(cell, 'confidence', 0.0),
                                "kind": getattr(cell, 'kind', 'content')
                            }
                        
                        # Detectar headers (primera fila)
                        if 0 in rows_dict:
                            table_data["headers"] = [
                                rows_dict[0].get(i, {}).get("text", "") 
                                for i in range(table_data["column_count"])
                            ]
                        
                        # Estructurar filas de datos
                        for row_idx in sorted(rows_dict.keys()):
                            if row_idx > 0:  # Saltar headers
                                row_data = [
                                    rows_dict[row_idx].get(i, {}).get("text", "")
                                    for i in range(table_data["column_count"])
                                ]
                                table_data["data_rows"].append(row_data)
                            
                            # También mantener formato original para compatibilidad
                            for col_idx in sorted(rows_dict[row_idx].keys()):
                                cell_info = rows_dict[row_idx][col_idx]
                                cell_info.update({
                                    "row": row_idx,
                                    "column": col_idx
                                })
                                table_data["cells"].append(cell_info)
                    
                    tables.append(table_data)
                    
        except Exception as e:
            logger.warning(f"Error procesando tablas: {e}")
        
        return tables

    def _extract_key_values(self, result) -> Dict[str, str]:
        """Extrae pares clave-valor con validación mejorada"""
        kv_pairs = {}
        
        try:
            if hasattr(result, 'key_value_pairs') and result.key_value_pairs:
                for kv in result.key_value_pairs:
                    try:
                        if self._is_valid_key_value(kv):
                            key = kv.key.content.strip()
                            value = kv.value.content.strip()
                            
                            # Normalizar y limpiar
                            key = self._normalize_key(key)
                            value = self._normalize_value(value)
                            
                            if key and value:
                                kv_pairs[key] = value
                                
                    except (AttributeError, TypeError) as e:
                        logger.debug(f"Par KV inválido ignorado: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Error extrayendo pares clave-valor: {e}")
        
        return kv_pairs

    def _is_valid_key_value(self, kv) -> bool:
        """Valida que un par clave-valor sea válido"""
        return (
            kv and
            hasattr(kv, 'key') and kv.key and
            hasattr(kv.key, 'content') and kv.key.content and
            hasattr(kv, 'value') and kv.value and
            hasattr(kv.value, 'content') and kv.value.content
        )

    def _normalize_key(self, key: str) -> str:
        """Normaliza las claves (quita : y espacios extras)"""
        return key.replace(":", "").strip().lower().replace(" ", "_")

    def _normalize_value(self, value: str) -> str:
        """Normaliza los valores"""
        return " ".join(value.split())  # Elimina espacios múltiples

    def _extract_entities(self, result) -> List[Dict[str, Any]]:
        """Extrae entidades con categorización mejorada"""
        entities = []
        seen_entities = set()  # Para evitar duplicados
        
        try:
            if hasattr(result, 'entities') and result.entities:
                for entity in result.entities:
                    try:
                        content = getattr(entity, 'content', "").strip()
                        
                        if content and content not in seen_entities:
                            entity_data = {
                                "text": content,
                                "category": getattr(entity, 'category', ""),
                                "subcategory": getattr(entity, 'subcategory', None),
                                "confidence": getattr(entity, 'confidence', 0.0),
                                "normalized": self._normalize_entity(content, getattr(entity, 'category', ""))
                            }
                            entities.append(entity_data)
                            seen_entities.add(content)
                            
                    except Exception as e:
                        logger.debug(f"Error procesando entidad individual: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Error extrayendo entidades: {e}")
        
        return entities

    def _normalize_entity(self, text: str, category: str) -> str:
        """Normaliza entidades según su categoría"""
        if category == "DateTime":
            # Aquí podrías parsear fechas a formato ISO
            return text
        elif category == "Currency":
            # Eliminar símbolos de moneda y normalizar
            return text.replace("$", "").replace(",", "").strip()
        elif category == "PhoneNumber":
            # Normalizar teléfonos
            return "".join(filter(str.isdigit, text))
        else:
            return text

    def _get_confidence_scores(self, result) -> Dict[str, float]:
        """Calcula scores de confianza detallados"""
        scores = {
            "overall": 0.0,
            "text_extraction": 0.0,
            "table_detection": 0.0,
            "entity_recognition": 0.0,
            "quality_assessment": "unknown"
        }
        
        try:
            # Confianza general del documento
            if hasattr(result, 'confidence'):
                scores["overall"] = result.confidence
            elif hasattr(result, 'pages') and result.pages:
                # Promedio de confianza de páginas
                page_confidences = [
                    getattr(page, 'confidence', 0.0) 
                    for page in result.pages 
                    if hasattr(page, 'confidence')
                ]
                if page_confidences:
                    scores["overall"] = sum(page_confidences) / len(page_confidences)
            
            # Estimación de calidad basada en cantidad de contenido extraído
            if hasattr(result, 'content'):
                content_length = len(result.content)
                if content_length > 1000:
                    scores["text_extraction"] = 0.9
                elif content_length > 500:
                    scores["text_extraction"] = 0.7
                elif content_length > 100:
                    scores["text_extraction"] = 0.5
                else:
                    scores["text_extraction"] = 0.3
            
            # Confianza en tablas
            if hasattr(result, 'tables') and result.tables:
                table_confidences = [
                    getattr(table, 'confidence', 0.0) 
                    for table in result.tables 
                    if hasattr(table, 'confidence')
                ]
                if table_confidences:
                    scores["table_detection"] = sum(table_confidences) / len(table_confidences)
            
            # Confianza en entidades
            if hasattr(result, 'entities') and result.entities:
                entity_confidences = [
                    getattr(entity, 'confidence', 0.0) 
                    for entity in result.entities 
                    if hasattr(entity, 'confidence')
                ]
                if entity_confidences:
                    scores["entity_recognition"] = sum(entity_confidences) / len(entity_confidences)
            
            # Evaluación de calidad general
            avg_confidence = scores["overall"]
            if avg_confidence >= 0.9:
                scores["quality_assessment"] = "excellent"
            elif avg_confidence >= 0.7:
                scores["quality_assessment"] = "good"
            elif avg_confidence >= 0.5:
                scores["quality_assessment"] = "fair"
            else:
                scores["quality_assessment"] = "poor"
                
        except Exception as e:
            logger.warning(f"Error calculando scores de confianza: {e}")
        
        return scores


# Funciones de conveniencia
def analyze_document(file_path: str) -> OCRResult:
    """Función de conveniencia para análisis rápido"""
    processor = AzureOCRProcessor()
    return processor.analyze_document(file_path)

async def analyze_document_async(file_path: str) -> OCRResult:
    """Función asíncrona de conveniencia"""
    processor = AzureOCRProcessor()
    return await processor.analyze_document_async(file_path)