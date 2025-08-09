from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re
import logging

logger = logging.getLogger(__name__)

class UniversalDocumentExtractor:
    """
    Extractor universal que procesa datos OCR y los estructura
    para cualquier tipo de documento
    """
    
    def __init__(self):
        # Patrones comunes en documentos mexicanos
        self.patterns = {
            'rfc': r'[A-Z&Ñ]{3,4}[0-9]{6}[A-Z0-9]{3}',
            'curp': r'[A-Z]{4}[0-9]{6}[HM][A-Z]{5}[A-Z0-9]{2}',
            'fecha': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'moneda': r'\$?\s*[\d,]+\.?\d*',
            'placa': r'[A-Z]{3}-?\d{2,4}-?[A-Z]?',
            'serie_vehicular': r'[A-Z0-9]{17}',
            'telefono': r'[\d\s\-\(\)]{10,}',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'codigo_qr': r'(https?://[^\s]+)',
            'folio': r'[Ff]olio:?\s*([A-Z0-9\-]+)',
        }
        
        # Palabras clave por tipo de documento
        self.doc_keywords = {
            'factura': ['factura', 'cfdi', 'comprobante fiscal', 'subtotal', 'iva', 'total'],
            'denuncia': ['denuncia', 'ministerio público', 'declaración', 'hechos', 'carpeta'],
            'identificacion': ['ine', 'ife', 'identificación', 'credencial', 'elector'],
            'licencia': ['licencia', 'conducir', 'automovilista', 'vigencia'],
            'poliza': ['póliza', 'asegurado', 'cobertura', 'prima', 'vigencia'],
            'peritaje': ['peritaje', 'dictamen', 'evaluación', 'daños', 'reparación'],
            'carta_porte': ['carta porte', 'transportista', 'origen', 'destino', 'mercancía'],
            'bitacora_gps': ['gps', 'coordenadas', 'velocidad', 'ruta', 'parada'],
            'estado_cuenta': ['estado de cuenta', 'banco', 'movimientos', 'saldo'],
        }
    
    def extract_structured_data(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae y estructura todos los datos del resultado OCR
        (tolerante a distintas llaves: key_values vs key_value_pairs, metadatos en metadata, etc.)
        """
        logger.info("Iniciando extracción estructurada de datos")
        
        # Texto principal
        text = ocr_result.get('text') or ""
        if not isinstance(text, str):
            text = str(text)

        # Detectar tipo de documento
        doc_type = self._detect_document_type(text)
        
        # Entidades comunes
        entities = self._extract_common_entities(text)
        
        # Tablas (tolerante a distintos formatos)
        table_data = self._process_tables(ocr_result.get('tables', []) or [])
        
        # Pares clave-valor (acepta ambas llaves)
        key_values = (
            ocr_result.get('key_value_pairs')
            or ocr_result.get('key_values')
            or {}
        )

        # Campos específicos por tipo
        specific_fields = self._extract_specific_fields(text, doc_type)
        
        # Códigos QR y URLs
        qr_codes = self._extract_qr_codes(text)

        # Metadata OCR tolerante
        meta = (ocr_result.get('metadata') or {})
        confidence_scores = (
            ocr_result.get('confidence_scores')
            or ocr_result.get('confidence')
            or {}
        )
        page_count = (
            ocr_result.get('page_count')
            or meta.get('page_count')
            or 1
        )
        language = (
            ocr_result.get('language')
            or meta.get('language')
            or 'es'
        )
        
        structured_data = {
            "document_type": doc_type,
            "extracted_at": datetime.now().isoformat(),
            "raw_text": text,
            "text_length": len(text),
            
            # Entidades comunes
            "entities": entities,
            
            # Datos de tablas
            "tables": table_data,
            
            # Campos clave-valor
            "key_value_pairs": key_values,
            
            # Campos específicos del tipo de documento
            "specific_fields": specific_fields,
            
            # QR y URLs
            "qr_codes_urls": qr_codes,
            
            # Metadata del OCR (normalizada)
            "ocr_metadata": {
                "confidence": confidence_scores,
                "page_count": page_count,
                "language": language
            },
            
            # Secciones del texto (para análisis por partes)
            "text_sections": self._segment_text(text)
        }
        
        logger.info(f"Extracción completada. Tipo: {doc_type}, Entidades: {len(entities)}")
        return structured_data
    
    def _detect_document_type(self, text: str) -> str:
        """Detecta el tipo de documento basado en palabras clave"""
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        
        for doc_type, keywords in self.doc_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[doc_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "otro"
    
    def _extract_common_entities(self, text: str) -> Dict[str, List[str]]:
        """Extrae entidades comunes usando expresiones regulares"""
        entities: Dict[str, List[str]] = {}
        
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Limpiar y eliminar duplicados
                clean = list({m.strip() for m in matches if str(m).strip()})
                if clean:
                    entities[entity_type] = clean
        
        return entities
    
    def _process_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Procesa y estructura las tablas detectadas.
        Soporta:
          - celdas en formato {"cells": [{"row":..,"column":..,"text":..}, ...]}
          - filas directas en "data_rows" con "headers"
        """
        processed_tables: List[Dict[str, Any]] = []
        
        for i, table in enumerate(tables):
            # Si vienen data_rows/headers, úsalo directamente
            if table.get("data_rows"):
                processed_tables.append({
                    "table_index": i,
                    "rows": table.get("row_count", len(table.get("data_rows", []))),
                    "columns": table.get("column_count", len(table.get("headers", [])) or (len(table["data_rows"][0]) if table["data_rows"] else 0)),
                    "headers": table.get("headers", []),
                    "data": table.get("data_rows", [])
                })
                continue

            # Caso clásico con "cells"
            processed_table = {
                "table_index": i,
                "rows": table.get('row_count', 0),
                "columns": table.get('column_count', 0),
                "data": []
            }
            
            cells = table.get('cells', []) or []
            if cells:
                # Crear matriz
                matrix: Dict[int, Dict[int, str]] = {}
                for cell in cells:
                    row = int(cell.get('row', cell.get('row_index', 0)) or 0)
                    col = int(cell.get('column', cell.get('column_index', 0)) or 0)
                    if row not in matrix:
                        matrix[row] = {}
                    matrix[row][col] = cell.get('text') or cell.get('content') or ''
                
                # Convertir a lista de filas
                for row_idx in sorted(matrix.keys()):
                    row_data = []
                    for col_idx in sorted(matrix.get(row_idx, {}).keys()):
                        row_data.append(matrix[row_idx].get(col_idx, ''))
                    processed_table["data"].append(row_data)
            
            processed_tables.append(processed_table)
        
        return processed_tables
    
    def _extract_specific_fields(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Extrae campos específicos según el tipo de documento"""
        fields: Dict[str, Any] = {}
        
        if doc_type == "factura":
            # Buscar número de factura
            folio_match = re.search(r'[Ff]olio:?\s*([A-Z0-9\-]+)', text)
            if folio_match:
                fields['numero_factura'] = folio_match.group(1)
            
            # Buscar totales
            total_match = re.search(r'[Tt]otal:?\s*\$?\s*([\d,]+\.?\d*)', text)
            if total_match:
                fields['total'] = total_match.group(1).replace(',', '')
                
        elif doc_type == "denuncia":
            # Buscar número de carpeta
            carpeta_match = re.search(r'[Cc]arpeta:?\s*([A-Z0-9\-/]+)', text)
            if carpeta_match:
                fields['numero_carpeta'] = carpeta_match.group(1)
            
            # Buscar ministerio público
            mp_match = re.search(r'[Mm]inisterio\s+[Pp]úblico:?\s*([^\n]+)', text)
            if mp_match:
                fields['ministerio_publico'] = mp_match.group(1).strip()
                
        elif doc_type == "identificacion":
            # Buscar número de credencial
            cred_match = re.search(r'[Cc]redencial:?\s*([0-9]+)', text)
            if cred_match:
                fields['numero_credencial'] = cred_match.group(1)
                
        # Agregar más tipos según necesites
        
        return fields
    
    def _extract_qr_codes(self, text: str) -> List[Dict[str, str]]:
        """Extrae códigos QR y URLs para verificación posterior"""
        qr_data: List[Dict[str, str]] = []
        
        # Buscar URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        
        for url in urls:
            qr_data.append({
                "type": "url",
                "value": url,
                "needs_verification": True
            })
        
        # Buscar referencias a códigos QR
        tl = text.lower()
        if 'código qr' in tl or 'codigo qr' in tl or 'qr code' in tl:
            qr_data.append({
                "type": "qr_reference",
                "value": "Documento contiene código QR",
                "needs_visual_extraction": True
            })
        
        return qr_data
    
    def _segment_text(self, text: str) -> List[Dict[str, str]]:
        """Segmenta el texto en secciones lógicas"""
        sections: List[Dict[str, str]] = []
        
        # Dividir por párrafos significativos
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 50:  # Solo párrafos con contenido sustancial
                sections.append({
                    "section_index": i,
                    "content": paragraph,
                    "word_count": len(paragraph.split()),
                    "char_count": len(paragraph)
                })
        
        return sections
