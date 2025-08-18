# src/fraud_scorer/extractors/intelligent_extractor.py

from typing import Dict, Any, List, Optional, Tuple, Union
import re
import logging
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ExtractionStrategy(Enum):
    """Estrategias de extracción en orden de prioridad"""
    KEY_VALUE = "key_value"          # Buscar en pares clave-valor del OCR
    REGEX_STRICT = "regex_strict"    # Regex exacto
    REGEX_FUZZY = "regex_fuzzy"      # Regex flexible
    PROXIMITY = "proximity"          # Buscar cerca de palabras clave
    TABLE_SCAN = "table_scan"        # Buscar en tablas del OCR
    COMPOSITE = "composite"          # Combinar múltiples campos
    ML_CONTEXT = "ml_context"        # Usar ML para entender contexto
    LLM_FALLBACK = "llm_fallback"    # Último recurso: GPT-4

@dataclass
class ExtractionResult:
    """Resultado de una extracción"""
    value: Any
    confidence: float
    strategy: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FieldDefinition:
    """Define cómo extraer un campo específico"""
    name: str
    display_name: str
    strategies: List[ExtractionStrategy]
    patterns: List[str]
    validators: List[callable]
    normalizer: Optional[callable] = None
    confidence_threshold: float = 0.6
    anchor_words: List[str] = field(default_factory=list)
    value_type: str = "string"  # string, number, date, amount

class IntelligentFieldExtractor:
    """Extractor inteligente multi-estrategia para campos de documentos"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.field_definitions = self._initialize_field_definitions()
        self.extraction_stats = {}
        self.cache = {}
        
        # Cargar configuración personalizada si existe
        if config_path and config_path.exists():
            self._load_custom_config(config_path)
    
    def extract_all_fields(
        self,
        document: Dict[str, Any],
        fields_to_extract: Optional[List[str]] = None,
        debug: bool = False
    ) -> Dict[str, ExtractionResult]:
        """
        Extrae todos los campos solicitados del documento
        
        Args:
            document: Documento del que extraer
            fields_to_extract: Lista de campos a extraer (None = todos)
            debug: Si mostrar información de debug
            
        Returns:
            Dict con los resultados de extracción
        """
        results = {}
        
        # Determinar qué campos extraer
        if fields_to_extract is None:
            fields_to_extract = list(self.field_definitions.keys())
        
        # Cachear el documento procesado
        doc_hash = self._hash_document(document)
        
        for field_name in fields_to_extract:
            if field_name not in self.field_definitions:
                logger.warning(f"Campo '{field_name}' no definido")
                continue
            
            # Verificar cache
            cache_key = f"{doc_hash}:{field_name}"
            if cache_key in self.cache:
                results[field_name] = self.cache[cache_key]
                continue
            
            # Extraer campo
            result = self.extract_field(field_name, document, debug)
            
            if result.value is not None:
                results[field_name] = result
                self.cache[cache_key] = result
                
                if debug:
                    logger.info(
                        f"✓ {field_name}: '{result.value}' "
                        f"(estrategia: {result.strategy}, confianza: {result.confidence:.2f})"
                    )
            elif debug:
                logger.warning(f"✗ {field_name}: No extraído")
        
        return results
    
    def extract_field(
        self, 
        field_name: str, 
        document: Dict[str, Any],
        debug: bool = False
    ) -> ExtractionResult:
        """
        Extrae un campo específico usando múltiples estrategias
        
        Returns:
            ExtractionResult con el valor extraído
        """
        if field_name not in self.field_definitions:
            return ExtractionResult(None, 0.0, "undefined")
        
        field_def = self.field_definitions[field_name]
        
        # Probar cada estrategia en orden
        for strategy in field_def.strategies:
            try:
                if debug:
                    logger.debug(f"Probando {strategy.value} para {field_name}")
                
                result = self._apply_strategy(strategy, field_def, document)
                
                if result.value is not None and result.confidence >= field_def.confidence_threshold:
                    # Validar el valor extraído
                    if self._validate_value(result.value, field_def.validators):
                        # Normalizar si hay normalizador definido
                        if field_def.normalizer:
                            result.value = field_def.normalizer(result.value)
                        
                        # Actualizar estadísticas
                        self._update_stats(field_name, strategy, True)
                        
                        return result
                        
            except Exception as e:
                logger.error(f"Error en estrategia {strategy.value} para {field_name}: {e}")
                self._update_stats(field_name, strategy, False)
                continue
        
        return ExtractionResult(None, 0.0, "failed")
    
    def _apply_strategy(
        self, 
        strategy: ExtractionStrategy,
        field_def: FieldDefinition,
        document: Dict[str, Any]
    ) -> ExtractionResult:
        """Aplica una estrategia específica de extracción"""
        
        if strategy == ExtractionStrategy.KEY_VALUE:
            return self._extract_from_key_value(field_def, document)
        elif strategy == ExtractionStrategy.REGEX_STRICT:
            return self._extract_with_strict_regex(field_def, document)
        elif strategy == ExtractionStrategy.REGEX_FUZZY:
            return self._extract_with_fuzzy_regex(field_def, document)
        elif strategy == ExtractionStrategy.PROXIMITY:
            return self._extract_by_proximity(field_def, document)
        elif strategy == ExtractionStrategy.TABLE_SCAN:
            return self._extract_from_tables(field_def, document)
        elif strategy == ExtractionStrategy.COMPOSITE:
            return self._extract_composite(field_def, document)
        elif strategy == ExtractionStrategy.ML_CONTEXT:
            return self._extract_with_ml(field_def, document)
        elif strategy == ExtractionStrategy.LLM_FALLBACK:
            return self._extract_with_llm(field_def, document)
        
        return ExtractionResult(None, 0.0, strategy.value)
    
    # ==================== ESTRATEGIAS DE EXTRACCIÓN ====================
    
    def _extract_from_key_value(
        self, 
        field_def: FieldDefinition,
        document: Dict[str, Any]
    ) -> ExtractionResult:
        """Extrae de pares clave-valor del OCR"""
        
        kv_pairs = document.get('key_value_pairs', {})
        if not kv_pairs:
            return ExtractionResult(None, 0.0, "key_value")
        
        # Buscar variaciones del nombre del campo
        possible_keys = self._generate_key_variations(field_def.name)
        
        # Búsqueda exacta
        for key in possible_keys:
            if key in kv_pairs and kv_pairs[key]:
                value = str(kv_pairs[key]).strip()
                if value:
                    return ExtractionResult(
                        value=value,
                        confidence=0.95,
                        strategy="key_value",
                        source=f"key={key}"
                    )
        
        # Búsqueda fuzzy en las claves
        for ocr_key, value in kv_pairs.items():
            if not value:
                continue
                
            # Calcular similitud
            similarity = self._calculate_similarity(field_def.name, ocr_key)
            if similarity > 0.7:
                return ExtractionResult(
                    value=str(value).strip(),
                    confidence=similarity,
                    strategy="key_value",
                    source=f"fuzzy_key={ocr_key}"
                )
            
            # Buscar el campo dentro del valor (caso común en OCR)
            for anchor in field_def.anchor_words[:3]:
                if anchor.lower() in str(value).lower():
                    extracted = self._extract_value_from_text(str(value), field_def)
                    if extracted:
                        return ExtractionResult(
                            value=extracted,
                            confidence=0.75,
                            strategy="key_value",
                            source=f"embedded_in={ocr_key}"
                        )
        
        return ExtractionResult(None, 0.0, "key_value")
    
    def _extract_with_strict_regex(
        self,
        field_def: FieldDefinition,
        document: Dict[str, Any]
    ) -> ExtractionResult:
        """Extrae con regex estricto"""
        
        text = document.get('raw_text', '')
        if not text:
            return ExtractionResult(None, 0.0, "regex_strict")
        
        # Probar los primeros patrones (más específicos)
        for i, pattern in enumerate(field_def.patterns[:5]):
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if match.groups():
                        value = match.group(1).strip()
                        if value:
                            return ExtractionResult(
                                value=value,
                                confidence=0.9 - (i * 0.05),  # Menor confianza para patrones posteriores
                                strategy="regex_strict",
                                source=f"pattern_{i}",
                                metadata={"pattern": pattern}
                            )
            except Exception as e:
                logger.debug(f"Error en patrón regex: {e}")
                continue
        
        return ExtractionResult(None, 0.0, "regex_strict")
    
    def _extract_with_fuzzy_regex(
        self,
        field_def: FieldDefinition,
        document: Dict[str, Any]
    ) -> ExtractionResult:
        """Extrae con regex flexible (permite variaciones)"""
        
        text = document.get('raw_text', '')
        if not text:
            return ExtractionResult(None, 0.0, "regex_fuzzy")
        
        # Normalizar texto
        normalized_text = self._normalize_text(text)
        
        for i, pattern in enumerate(field_def.patterns):
            try:
                # Hacer el patrón más flexible
                fuzzy_pattern = self._make_pattern_fuzzy(pattern)
                
                matches = re.finditer(fuzzy_pattern, normalized_text, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        # Extraer del texto original
                        start, end = match.span(1)
                        value = self._extract_from_original_position(text, normalized_text, start, end)
                        
                        if value:
                            return ExtractionResult(
                                value=value,
                                confidence=0.75 - (i * 0.05),
                                strategy="regex_fuzzy",
                                source=f"fuzzy_pattern_{i}"
                            )
            except Exception as e:
                logger.debug(f"Error en fuzzy regex: {e}")
                continue
        
        return ExtractionResult(None, 0.0, "regex_fuzzy")
    
    def _extract_by_proximity(
        self,
        field_def: FieldDefinition,
        document: Dict[str, Any]
    ) -> ExtractionResult:
        """Extrae valores que están cerca de palabras clave"""
        
        text = document.get('raw_text', '')
        if not text:
            return ExtractionResult(None, 0.0, "proximity")
        
        for anchor in field_def.anchor_words:
            # Buscar el ancla
            anchor_pattern = re.escape(anchor)
            anchor_matches = re.finditer(anchor_pattern, text, re.IGNORECASE)
            
            for anchor_match in anchor_matches:
                # Extraer texto después del ancla (hasta 100 caracteres)
                start = anchor_match.end()
                context = text[start:start + 100]
                
                # Buscar un valor válido en el contexto
                value = self._extract_value_from_context(context, field_def)
                
                if value:
                    return ExtractionResult(
                        value=value,
                        confidence=0.65,
                        strategy="proximity",
                        source=f"near_{anchor}"
                    )
        
        return ExtractionResult(None, 0.0, "proximity")
    
    def _extract_from_tables(
        self,
        field_def: FieldDefinition,
        document: Dict[str, Any]
    ) -> ExtractionResult:
        """Extrae valores de las tablas del documento"""
        
        tables = document.get('tables', [])
        if not tables:
            return ExtractionResult(None, 0.0, "table_scan")
        
        for table_idx, table in enumerate(tables):
            # Buscar en headers
            headers = table.get('headers', [])
            for header_idx, header in enumerate(headers):
                if self._matches_field(header, field_def):
                    # Buscar valor en la misma columna
                    for row in table.get('data_rows', []):
                        if header_idx < len(row) and row[header_idx]:
                            value = str(row[header_idx]).strip()
                            if value:
                                return ExtractionResult(
                                    value=value,
                                    confidence=0.8,
                                    strategy="table_scan",
                                    source=f"table_{table_idx}_col_{header_idx}"
                                )
            
            # Buscar en celdas
            cells = table.get('cells', [])
            for cell in cells:
                cell_text = cell.get('text', '')
                if self._matches_field(cell_text, field_def):
                    # Buscar valor en celdas adyacentes
                    row = cell.get('row', 0)
                    col = cell.get('column', 0)
                    
                    # Buscar a la derecha
                    for next_cell in cells:
                        if (next_cell.get('row') == row and 
                            next_cell.get('column') == col + 1):
                            value = next_cell.get('text', '').strip()
                            if value:
                                return ExtractionResult(
                                    value=value,
                                    confidence=0.75,
                                    strategy="table_scan",
                                    source=f"table_{table_idx}_cell"
                                )
        
        return ExtractionResult(None, 0.0, "table_scan")
    
    def _extract_composite(
        self,
        field_def: FieldDefinition,
        document: Dict[str, Any]
    ) -> ExtractionResult:
        """Extrae valores combinando múltiples fuentes"""
        
        # Esta estrategia es útil para campos que pueden estar divididos
        # Por ejemplo: vigencia que viene en dos campos separados
        
        if field_def.name == "vigencia_completa":
            # Buscar inicio y fin por separado
            inicio = self.extract_field("vigencia_inicio", document)
            fin = self.extract_field("vigencia_fin", document)
            
            if inicio.value and fin.value:
                value = f"{inicio.value} al {fin.value}"
                confidence = min(inicio.confidence, fin.confidence)
                return ExtractionResult(
                    value=value,
                    confidence=confidence,
                    strategy="composite",
                    source="vigencia_inicio+vigencia_fin"
                )
        
        return ExtractionResult(None, 0.0, "composite")
    
    def _extract_with_ml(
        self,
        field_def: FieldDefinition,
        document: Dict[str, Any]
    ) -> ExtractionResult:
        """Usa ML para extraer campos (placeholder para integración futura)"""
        
        # Aquí integrarías modelos de ML como:
        # - spaCy para NER
        # - Transformers para clasificación
        # - LayoutLM para comprensión de layout
        
        text = document.get('raw_text', '')
        if not text:
            return ExtractionResult(None, 0.0, "ml_context")
        
        # Ejemplo simplificado con NER para nombres de empresas
        if field_def.name == "nombre_asegurado":
            # Buscar patrones de empresas mexicanas
            empresa_patterns = [
                r'([A-Z][A-Z\s]+,?\s*S\.?\s*[AC]\.?\s*(?:de\s+)?C\.?\s*V\.?)',
                r'([A-Z][A-Z\s]+,?\s*S\.?\s*[AC]\.?)',
            ]
            
            for pattern in empresa_patterns:
                match = re.search(pattern, text)
                if match:
                    return ExtractionResult(
                        value=match.group(1).strip(),
                        confidence=0.7,
                        strategy="ml_context",
                        source="ner_pattern"
                    )
        
        return ExtractionResult(None, 0.0, "ml_context")
    
    def _extract_with_llm(
        self,
        field_def: FieldDefinition,
        document: Dict[str, Any]
    ) -> ExtractionResult:
        """Usa LLM como último recurso (placeholder)"""
        
        # Este sería tu último recurso - costoso pero efectivo
        # Deberías implementarlo solo si tienes acceso a GPT-4
        
        try:
            # Limitar contexto para reducir costos
            text = document.get('raw_text', '')[:1500]
            
            if not text:
                return ExtractionResult(None, 0.0, "llm_fallback")
            
            # Aquí llamarías a tu API de OpenAI
            # Por ahora, retornamos un placeholder
            
            # prompt = f"""
            # Extrae únicamente el valor de '{field_def.display_name}' del siguiente texto.
            # Si no lo encuentras, responde 'NO_ENCONTRADO'.
            # 
            # Texto: {text}
            # 
            # {field_def.display_name}:
            # """
            # 
            # response = await openai_client.complete(prompt)
            # value = response.strip()
            # 
            # if value and value != 'NO_ENCONTRADO':
            #     return ExtractionResult(
            #         value=value,
            #         confidence=0.6,
            #         strategy="llm_fallback",
            #         source="gpt-4"
            #     )
            
        except Exception as e:
            logger.error(f"Error en LLM fallback: {e}")
        
        return ExtractionResult(None, 0.0, "llm_fallback")
    
    # ==================== MÉTODOS AUXILIARES ====================
    
    def _initialize_field_definitions(self) -> Dict[str, FieldDefinition]:
        """Define cómo extraer cada campo importante"""
        
        return {
            "numero_poliza": FieldDefinition(
                name="numero_poliza",
                display_name="Número de Póliza",
                strategies=[
                    ExtractionStrategy.KEY_VALUE,
                    ExtractionStrategy.REGEX_STRICT,
                    ExtractionStrategy.REGEX_FUZZY,
                    ExtractionStrategy.PROXIMITY,
                    ExtractionStrategy.TABLE_SCAN,
                ],
                patterns=[
                    # Patrones más específicos primero
                    r"P[óo]liza\s*No\s*\.?\s*:?\s*([\d\s\-]+)",
                    r"N[úu]mero\s+de\s+P[óo]liza\s*:?\s*([\d\s\-]+)",
                    r"P[óo]liza\s*#?\s*:?\s*([\d\s\-]+)",
                    # Patrones con contexto
                    r"Reclamaci[óo]n.*?P[óo]liza\s*(?:No\.?|#)?\s*:?\s*([\d\s\-]+)",
                    r"Asunto.*?P[óo]liza.*?([\d]{3,}\s*-\s*[\d]{4,})",
                    # Patrones más flexibles
                    r"(?:Policy|Poliza)[^\d]*?([\d\-\s]+\d)",
                    r"Contrato\s*(?:de\s+)?[Ss]eguro\s*(?:No\.?|#)?\s*:?\s*([\d\s\-]+)",
                ],
                validators=[
                    lambda x: bool(re.search(r'\d', x)),  # Debe tener números
                    lambda x: len(x.replace(" ", "").replace("-", "")) >= 3,  # Mínimo 3 caracteres
                    lambda x: len(x) < 50,  # No puede ser muy largo
                ],
                normalizer=lambda x: re.sub(r'\s+', ' ', x.strip()),
                anchor_words=["póliza", "policy", "número de póliza", "no. póliza", "contrato"],
                value_type="string"
            ),
            
            "nombre_asegurado": FieldDefinition(
                name="nombre_asegurado",
                display_name="Nombre del Asegurado",
                strategies=[
                    ExtractionStrategy.KEY_VALUE,
                    ExtractionStrategy.REGEX_STRICT,
                    ExtractionStrategy.PROXIMITY,
                    ExtractionStrategy.TABLE_SCAN,
                    ExtractionStrategy.ML_CONTEXT,
                ],
                patterns=[
                    # Patrones para empresas mexicanas
                    r"(?:Nombre|Asegurado|Contratante)\s*:?\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s,\.]+(?:S\.A\.|S\.C\.|C\.V\.)?)",
                    r"Raz[óo]n\s+Social\s*:?\s*([^\n\r]+)",
                    # Patrón específico para MODA YKT
                    r"(MODA\s+YKT[^,\n]*(?:,?\s*S\.?\s*A\.?\s*(?:de|DE)?\s*C\.?\s*V\.?)?)",
                    # Patrones generales
                    r"Cliente\s*:?\s*([A-ZÁÉÍÓÚÑ][^\n\r]+)",
                    r"Nombre\s*:?\s*([A-ZÁÉÍÓÚÑ][^\n\r]+)",
                ],
                validators=[
                    lambda x: len(x) > 3,
                    lambda x: not x.replace(" ", "").replace(".", "").isdigit(),
                    lambda x: not any(word in x.lower() for word in ["fecha", "domicilio", "rfc", "póliza"]),
                ],
                normalizer=lambda x: ' '.join(x.split()),
                anchor_words=["asegurado", "contratante", "nombre", "razón social", "cliente"],
                value_type="string"
            ),
            
            "rfc": FieldDefinition(
                name="rfc",
                display_name="RFC",
                strategies=[
                    ExtractionStrategy.KEY_VALUE,
                    ExtractionStrategy.REGEX_STRICT,
                    ExtractionStrategy.PROXIMITY,
                    ExtractionStrategy.TABLE_SCAN,
                ],
                patterns=[
                    r"RFC\s*:?\s*([A-ZÑ&]{3,4}\d{6}[A-Z\d]{3})",
                    r"R\.F\.C\.\s*:?\s*([A-ZÑ&]{3,4}\d{6}[A-Z\d]{3})",
                    r"Registro\s+Federal.*?:?\s*([A-ZÑ&]{3,4}\d{6}[A-Z\d]{3})",
                    r"([A-ZÑ&]{3,4}\d{6}[A-Z\d]{3})",  # RFC solo
                ],
                validators=[
                    lambda x: bool(re.match(r'^[A-ZÑ&]{3,4}\d{6}[A-Z\d]{3}$', x.upper())),
                ],
                normalizer=lambda x: x.upper().strip(),
                anchor_words=["rfc", "r.f.c.", "registro federal"],
                value_type="string"
            ),
            
            "monto_reclamacion": FieldDefinition(
                name="monto_reclamacion",
                display_name="Monto de Reclamación",
                strategies=[
                    ExtractionStrategy.KEY_VALUE,
                    ExtractionStrategy.REGEX_STRICT,
                    ExtractionStrategy.REGEX_FUZZY,
                    ExtractionStrategy.TABLE_SCAN,
                ],
                patterns=[
                    # Patrones específicos de reclamación
                    r"(?:Monto|Importe)\s+(?:de\s+)?(?:la\s+)?Reclamaci[óo]n\s*:?\s*\$?\s*([\d,]+(?:\.\d{2})?)",
                    r"Reclamaci[óo]n\s+por\s+(?:la\s+cantidad\s+de\s+)?\$?\s*([\d,]+(?:\.\d{2})?)",
                    r"Cantidad\s+Reclamada\s*:?\s*\$?\s*([\d,]+(?:\.\d{2})?)",
                    # Patrones de total
                    r"Total\s*:?\s*\$\s*([\d,]+(?:\.\d{2})?)",
                    r"Importe\s+Total\s*:?\s*\$?\s*([\d,]+(?:\.\d{2})?)",
                    # Patrón para cualquier monto grande
                    r"\$\s*((?:[1-9]\d{0,2},)?(?:\d{3},)*\d{3}(?:\.\d{2})?)",
                ],
                validators=[
                    lambda x: self._parse_amount(x) > 0,
                    lambda x: self._parse_amount(x) < 1000000000,  # Menos de mil millones
                ],
                normalizer=lambda x: f"{self._parse_amount(x):,.2f}",
                anchor_words=["monto", "importe", "total", "cantidad", "reclamación", "valor"],
                value_type="amount"
            ),
            
            "fecha_siniestro": FieldDefinition(
                name="fecha_siniestro",
                display_name="Fecha del Siniestro",
                strategies=[
                    ExtractionStrategy.KEY_VALUE,
                    ExtractionStrategy.REGEX_STRICT,
                    ExtractionStrategy.PROXIMITY,
                    ExtractionStrategy.TABLE_SCAN,
                ],
                patterns=[
                    r"Fecha\s+del?\s+Siniestro\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"Fecha\s+de\s+Ocurrencia\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"Ocurri[óo]\s+el\s+(?:d[íi]a\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    # Formato con nombre de mes
                    r"Fecha.*?Siniestro.*?(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})",
                    # Cualquier fecha después de "siniestro"
                    r"[Ss]iniestro.*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                ],
                validators=[
                    lambda x: self._validate_date(x),
                ],
                normalizer=lambda x: self._normalize_date(x),
                anchor_words=["fecha del siniestro", "fecha de ocurrencia", "ocurrió", "siniestro"],
                value_type="date"
            ),
            
            "vigencia_inicio": FieldDefinition(
                name="vigencia_inicio",
                display_name="Vigencia Desde",
                strategies=[
                    ExtractionStrategy.KEY_VALUE,
                    ExtractionStrategy.REGEX_STRICT,
                    ExtractionStrategy.PROXIMITY,
                    ExtractionStrategy.TABLE_SCAN,
                ],
                patterns=[
                    r"Vigencia\s+(?:desde|del)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"Inicio\s+de\s+Vigencia\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"Desde\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    # Patrón con mes en texto
                    r"Vigencia.*?(\d{1,2}/[A-Z]{3}/\d{4})\s*(?:al|hasta)",
                    # Vigencia completa (tomar primera fecha)
                    r"Vigencia\s*:?\s*(?:del\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+al",
                ],
                validators=[
                    lambda x: self._validate_date(x),
                ],
                normalizer=lambda x: self._normalize_date(x),
                anchor_words=["vigencia desde", "inicio de vigencia", "desde", "del"],
                value_type="date"
            ),
            
            "vigencia_fin": FieldDefinition(
                name="vigencia_fin",
                display_name="Vigencia Hasta",
                strategies=[
                    ExtractionStrategy.KEY_VALUE,
                    ExtractionStrategy.REGEX_STRICT,
                    ExtractionStrategy.PROXIMITY,
                    ExtractionStrategy.TABLE_SCAN,
                ],
                patterns=[
                    r"Vigencia\s+hasta\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"Fin\s+de\s+Vigencia\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"Hasta\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    # Patrón con mes en texto
                    r"(?:al|hasta)\s*(\d{1,2}/[A-Z]{3}/\d{4})",
                    # Vigencia completa (tomar segunda fecha)
                    r"Vigencia.*?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+al\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                ],
                validators=[
                    lambda x: self._validate_date(x),
                ],
                normalizer=lambda x: self._normalize_date(x),
                anchor_words=["vigencia hasta", "fin de vigencia", "hasta", "al"],
                value_type="date"
            ),
            
            "domicilio_poliza": FieldDefinition(
                name="domicilio_poliza",
                display_name="Domicilio de la Póliza",
                strategies=[
                    ExtractionStrategy.KEY_VALUE,
                    ExtractionStrategy.REGEX_STRICT,
                    ExtractionStrategy.PROXIMITY,
                    ExtractionStrategy.TABLE_SCAN,
                ],
                patterns=[
                    r"Domicilio\s*(?:Fiscal)?\s*:?\s*([^,\n]+(?:,\s*[^,\n]+)*)",
                    r"Direcci[óo]n\s*:?\s*([^,\n]+(?:,\s*[^,\n]+)*)",
                    r"Ubicaci[óo]n\s*:?\s*([^,\n]+(?:,\s*[^,\n]+)*)",
                    # Patrón específico para direcciones mexicanas
                    r"((?:Calle|Av\.|Avenida|Blvd\.?)[^,\n]+(?:,\s*[^,\n]+)*(?:C\.P\.\s*\d{5})?)",
                ],
                validators=[
                    lambda x: len(x) > 10,
                    lambda x: not x.replace(" ", "").isdigit(),
                ],
                normalizer=lambda x: ' '.join(x.split()),
                anchor_words=["domicilio", "dirección", "ubicación", "domicilio fiscal"],
                value_type="string"
            ),
            
            "lugar_hechos": FieldDefinition(
                name="lugar_hechos",
                display_name="Lugar de los Hechos",
                strategies=[
                    ExtractionStrategy.KEY_VALUE,
                    ExtractionStrategy.REGEX_STRICT,
                    ExtractionStrategy.PROXIMITY,
                    ExtractionStrategy.TABLE_SCAN,
                ],
                patterns=[
                    r"Lugar\s+de\s+los\s+Hechos\s*:?\s*([^\n]+)",
                    r"Ubicaci[óo]n\s+del\s+(?:Siniestro|Incidente)\s*:?\s*([^\n]+)",
                    r"Ocurri[óo]\s+en\s*:?\s*([^\n]+)",
                    # Patrón para carreteras
                    r"((?:Autopista|Carretera|Km\.?\s*\d+)[^\n]+)",
                ],
                validators=[
                    lambda x: len(x) > 5,
                ],
                normalizer=lambda x: ' '.join(x.split()),
                anchor_words=["lugar de los hechos", "ubicación del siniestro", "ocurrió en"],
                value_type="string"
            ),
            
            "tipo_siniestro": FieldDefinition(
                name="tipo_siniestro",
                display_name="Tipo de Siniestro",
                strategies=[
                    ExtractionStrategy.KEY_VALUE,
                    ExtractionStrategy.REGEX_STRICT,
                    ExtractionStrategy.PROXIMITY,
                ],
                patterns=[
                    r"Tipo\s+de\s+Siniestro\s*:?\s*([^\n]+)",
                    r"Clase\s+de\s+Siniestro\s*:?\s*([^\n]+)",
                    r"Siniestro\s+por\s+([^\n]+)",
                    # Patrones específicos
                    r"(Robo\s+(?:de\s+)?(?:Mercanc[íi]a|Bulto|Total))",
                    r"(Colisi[óo]n|Volcadura|Incendio)",
                ],
                validators=[
                    lambda x: len(x) > 3,
                    lambda x: len(x) < 100,
                ],
                normalizer=lambda x: x.strip().upper(),
                anchor_words=["tipo de siniestro", "clase de siniestro", "robo", "colisión"],
                value_type="string"
            ),
        }
    
    def _generate_key_variations(self, field_name: str) -> List[str]:
        """Genera variaciones posibles del nombre del campo"""
        
        variations = [field_name]
        
        # Mapeo exhaustivo de variaciones
        variations_map = {
            "numero_poliza": [
                "poliza", "no_poliza", "num_poliza", "numero_poliza",
                "policy_number", "policy", "no. póliza", "número de póliza",
                "póliza no", "contrato", "numero_contrato"
            ],
            "nombre_asegurado": [
                "asegurado", "contratante", "cliente", "insured", 
                "nombre", "razón social", "razon_social", "nombre_completo",
                "titular", "beneficiario"
            ],
            "rfc": [
                "rfc", "r.f.c.", "registro_federal", "clave_fiscal",
                "registro federal de contribuyentes"
            ],
            "monto_reclamacion": [
                "monto", "importe", "total", "cantidad", "valor", 
                "claim_amount", "monto_reclamado", "suma_reclamada",
                "cantidad_reclamada", "importe_total"
            ],
            "fecha_siniestro": [
                "fecha_siniestro", "fecha_ocurrencia", "fecha_del_siniestro",
                "fecha de ocurrencia", "occurrence_date", "fecha_evento"
            ],
            "vigencia_inicio": [
                "vigencia_desde", "desde", "inicio_vigencia", "from",
                "vigencia_inicial", "fecha_inicio", "inicio"
            ],
            "vigencia_fin": [
                "vigencia_hasta", "hasta", "fin_vigencia", "to",
                "vigencia_final", "fecha_fin", "termino"
            ],
        }
        
        if field_name in variations_map:
            variations.extend(variations_map[field_name])
        
        # Agregar variaciones con y sin guiones bajos
        variations_with_spaces = []
        for var in variations:
            variations_with_spaces.append(var.replace("_", " "))
            variations_with_spaces.append(var.replace("_", ""))
        
        variations.extend(variations_with_spaces)
        
        return list(set(variations))  # Eliminar duplicados
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calcula similitud entre dos strings usando distancia de Levenshtein"""
        
        # Normalizar strings
        s1 = str1.lower().replace("_", " ").replace("-", " ").strip()
        s2 = str2.lower().replace("_", " ").replace("-", " ").strip()
        
        if s1 == s2:
            return 1.0
        
        # Si uno contiene al otro
        if s1 in s2 or s2 in s1:
            return 0.8
        
        # Calcular distancia de Levenshtein normalizada
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, s1, s2).ratio()
        except:
            return 0.0
    
    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para búsqueda fuzzy"""
        
        # Quitar acentos
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        
        # Normalizar espacios y puntuación
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.lower().strip()
    
    def _make_pattern_fuzzy(self, pattern: str) -> str:
        """Hace un patrón regex más flexible"""
        
        # Permitir espacios variables
        pattern = re.sub(r'\\s\+', r'\\s*', pattern)
        pattern = re.sub(r' ', r'\\s*', pattern)
        
        # Permitir acentos opcionales
        replacements = {
            'a': '[aáÁ]', 'e': '[eéÉ]', 'i': '[iíÍ]',
            'o': '[oóÓ]', 'u': '[uúÚ]', 'n': '[nñÑ]'
        }
        
        for char, replacement in replacements.items():
            pattern = pattern.replace(char, replacement)
        
        return pattern
    
    def _extract_from_original_position(
        self,
        original: str,
        normalized: str,
        start: int,
        end: int
    ) -> str:
        """Mapea posiciones del texto normalizado al original"""
        
        # Implementación simplificada - en producción sería más sofisticado
        # Por ahora, buscar el texto extraído en el original
        extracted_normalized = normalized[start:end]
        
        # Buscar una coincidencia aproximada en el original
        pattern = re.escape(extracted_normalized)
        pattern = self._make_pattern_fuzzy(pattern)
        
        match = re.search(pattern, original, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        
        # Fallback: retornar el texto normalizado
        return extracted_normalized.strip()
    
    def _extract_value_from_text(self, text: str, field_def: FieldDefinition) -> Optional[str]:
        """Extrae un valor de un texto usando los patrones del campo"""
        
        for pattern in field_def.patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.groups():
                return match.group(1).strip()
        
        return None
    
    def _extract_value_from_context(self, context: str, field_def: FieldDefinition) -> Optional[str]:
        """Extrae un valor válido del contexto cercano a una palabra ancla"""
        
        # Limpiar contexto
        context = context.strip()
        if not context:
            return None
        
        # Buscar según el tipo de valor esperado
        if field_def.value_type == "amount":
            # Buscar montos
            match = re.search(r'\$?\s*([\d,]+(?:\.\d{2})?)', context)
            if match:
                return match.group(1)
        
        elif field_def.value_type == "date":
            # Buscar fechas
            match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', context)
            if match:
                return match.group(1)
        
        elif field_def.value_type == "number":
            # Buscar números
            match = re.search(r'([\d\-]+)', context)
            if match:
                return match.group(1)
        
        else:  # string
            # Tomar hasta el siguiente salto de línea o puntuación
            match = re.match(r'[:\s]*([^,\n;]+)', context)
            if match:
                value = match.group(1).strip()
                if self._looks_like_field_value(field_def.name, value):
                    return value
        
        return None
    
    def _matches_field(self, text: str, field_def: FieldDefinition) -> bool:
        """Verifica si un texto corresponde al campo buscado"""
        
        text_lower = text.lower().strip()
        
        # Verificar anclas
        for anchor in field_def.anchor_words:
            if anchor.lower() in text_lower:
                return True
        
        # Verificar similitud con el nombre del campo
        similarity = self._calculate_similarity(field_def.name, text)
        return similarity > 0.7
    
    def _looks_like_field_value(self, field_name: str, value: str) -> bool:
        """Valida si un valor parece correcto para el campo"""
        
        if not value or len(value.strip()) < 2:
            return False
        
        # Validaciones específicas por campo
        if field_name == "numero_poliza":
            return bool(re.search(r'\d', value)) and len(value) < 50
        
        elif field_name == "nombre_asegurado":
            return len(value) > 3 and not value.replace(" ", "").isdigit()
        
        elif field_name in ["monto_reclamacion", "monto", "total"]:
            return bool(re.search(r'[\d,]+', value))
        
        elif field_name in ["fecha_siniestro", "vigencia_inicio", "vigencia_fin"]:
            return bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', value))
        
        elif field_name == "rfc":
            return bool(re.match(r'^[A-ZÑ&]{3,4}\d{6}[A-Z\d]{3}$', value.upper()))
        
        # Por defecto, aceptar si no es muy largo
        return len(value) < 500
    
    def _validate_value(self, value: Any, validators: List[callable]) -> bool:
        """Valida un valor usando los validadores definidos"""
        
        if not validators:
            return True
        
        for validator in validators:
            try:
                if not validator(value):
                    return False
            except Exception as e:
                logger.debug(f"Error en validador: {e}")
                return False
        
        return True
    
    def _parse_amount(self, amount_str: str) -> float:
        """Parsea un string de monto a float"""
        
        if not amount_str:
            return 0.0
        
        try:
            # Limpiar el string
            clean = str(amount_str)
            clean = clean.replace("$", "").replace(",", "")
            clean = clean.replace("MXN", "").replace("MN", "")
            clean = clean.replace("USD", "").strip()
            
            return float(clean)
        except:
            return 0.0
    
    def _validate_date(self, date_str: str) -> bool:
        """Valida que una fecha sea válida"""
        
        if not date_str:
            return False
        
        # Verificar formato básico
        if not re.search(r'\d', date_str):
            return False
        
        # Intentar parsear
        try:
            parsed = self._normalize_date(date_str)
            return parsed is not None
        except:
            return False
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normaliza una fecha a formato ISO"""
        
        if not date_str:
            return None
        
        # Limpiar string
        date_str = str(date_str).strip()
        
        # Mapeo de meses en español
        month_map = {
            'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
            'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
            'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12',
            'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'ago': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12',
            'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12',
        }
        
        # Reemplazar nombres de meses
        for month_name, month_num in month_map.items():
            date_str = date_str.replace(month_name, month_num)
        
        # Formatos a probar
        formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
            "%d/%m/%y", "%d-%m-%y", "%y-%m-%d",
            "%m/%d/%Y", "%m-%d-%Y",
            "%d de %m de %Y", "%d %m %Y",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                # Validar que el año sea razonable
                if 1900 <= dt.year <= 2100:
                    return dt.strftime("%Y-%m-%d")
            except:
                continue
        
        return None
    
    def _hash_document(self, document: Dict[str, Any]) -> str:
        """Genera un hash único para un documento"""
        
        # Usar solo campos clave para el hash
        key_content = {
            'text': document.get('raw_text', '')[:500],
            'type': document.get('document_type', ''),
        }
        
        content = json.dumps(key_content, sort_keys=True)
        return str(hash(content))
    
    def _update_stats(self, field_name: str, strategy: ExtractionStrategy, success: bool):
        """Actualiza estadísticas de extracción"""
        
        if field_name not in self.extraction_stats:
            self.extraction_stats[field_name] = {}
        
        strategy_name = strategy.value
        if strategy_name not in self.extraction_stats[field_name]:
            self.extraction_stats[field_name][strategy_name] = {
                'attempts': 0,
                'successes': 0
            }
        
        self.extraction_stats[field_name][strategy_name]['attempts'] += 1
        if success:
            self.extraction_stats[field_name][strategy_name]['successes'] += 1
    
    def get_stats_report(self) -> Dict[str, Any]:
        """Genera reporte de estadísticas de extracción"""
        
        report = {}
        
        for field, strategies in self.extraction_stats.items():
            field_report = {}
            total_attempts = 0
            total_successes = 0
            
            for strategy, stats in strategies.items():
                attempts = stats['attempts']
                successes = stats['successes']
                
                total_attempts += attempts
                total_successes += successes
                
                success_rate = (successes / attempts * 100) if attempts > 0 else 0
                
                field_report[strategy] = {
                    'attempts': attempts,
                    'successes': successes,
                    'success_rate': f"{success_rate:.1f}%"
                }
            
            overall_rate = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
            
            report[field] = {
                'strategies': field_report,
                'overall': {
                    'attempts': total_attempts,
                    'successes': total_successes,
                    'success_rate': f"{overall_rate:.1f}%"
                }
            }
        
        return report
    
    def _load_custom_config(self, config_path: Path):
        """Carga configuración personalizada de campos"""
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Actualizar definiciones de campos
            for field_name, field_config in config.get('fields', {}).items():
                if field_name in self.field_definitions:
                    # Actualizar campo existente
                    field_def = self.field_definitions[field_name]
                    
                    if 'patterns' in field_config:
                        field_def.patterns.extend(field_config['patterns'])
                    
                    if 'anchor_words' in field_config:
                        field_def.anchor_words.extend(field_config['anchor_words'])
                else:
                    # Crear nuevo campo
                    # TODO: Implementar creación de campos personalizados
                    pass
            
            logger.info(f"Configuración personalizada cargada desde {config_path}")
            
        except Exception as e:
            logger.error(f"Error cargando configuración personalizada: {e}")