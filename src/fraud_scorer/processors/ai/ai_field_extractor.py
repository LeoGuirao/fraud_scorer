# src/fraud_scorer/ai_extractors/ai_field_extractor.py
"""
AIFieldExtractor: Extrae campos de documentos individuales usando IA
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import asyncio
from datetime import datetime

from openai import AsyncOpenAI
import instructor
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from fraud_scorer.settings import ExtractionConfig, ExtractionRoute
from fraud_scorer.models.extraction import DocumentExtraction
from fraud_scorer.prompts.extraction_prompts import ExtractionPromptBuilder
from fraud_scorer.utils.validators import FieldValidator

logger = logging.getLogger(__name__)


class AIFieldExtractor:
    """
    Extractor de campos usando IA para documentos individuales.
    Procesa documento por documento de forma eficiente.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el extractor con cliente de OpenAI

        Args:
            api_key: API key de OpenAI (opcional, usa env var por defecto)
        """
        raw_client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        # Instructor "parchea" el cliente (opcional para otros métodos)
        self.client = instructor.patch(raw_client)

        self.config = ExtractionConfig()
        self.prompt_builder = ExtractionPromptBuilder()
        self.validator = FieldValidator()  # Nuevo validador
        self.extraction_cache: Dict[str, DocumentExtraction] = {}
        
        # Configuración de rutas
        self.route_config = self.config.ROUTE_CONFIG
        self.field_mapping = self.config.DOCUMENT_FIELD_MAPPING
        self.validation_rules = self.config.FIELD_VALIDATION_RULES

        logger.info("AIFieldExtractor inicializado con Sistema de Extracción Guiada")

    # =============================================================================
    #   MÉTODO ACTUALIZADO: maneja OCRResult (objeto) y dict + prompt mejorado
    # =============================================================================
    async def extract_from_document(
        self,
        ocr_result: Dict[str, Any],
        document_name: str,
        document_type: Optional[str] = None,
        use_cache: bool = True,
    ) -> DocumentExtraction:
        """
        Extrae campos de un documento individual
        """
        # Cache por contenido + nombre
        cache_key = self._generate_cache_key(document_name, ocr_result)
        if use_cache and cache_key in self.extraction_cache:
            logger.info(f"Usando cache de extracción para {document_name}")
            return self.extraction_cache[cache_key]

        # Detectar tipo si no viene
        if not document_type:
            document_type = self._detect_document_type(
                self._ocr_to_dict_safe(ocr_result),  # normalizamos para detección
                document_name
            )

        logger.info(f"Extrayendo campos de {document_name} (tipo: {document_type})")

        # IMPORTANTE: Manejar diferentes formatos de OCR
        # El OCR puede venir como dict directo o como un objeto con .text/.key_values/.tables
        if hasattr(ocr_result, "text"):
            # Objeto OCRResult-like
            prepared_content = {
                "text": getattr(ocr_result, "text", "") or "",
                "key_value_pairs": getattr(ocr_result, "key_values", {}) or {},
                "tables": getattr(ocr_result, "tables", []) or [],
            }
        else:
            # Dict ya normalizado
            prepared_content = self._prepare_ocr_content(ocr_result)

        # Si no hay contenido, retornar vacío
        if not prepared_content.get("text") and not prepared_content.get("key_value_pairs"):
            logger.warning(f"No hay contenido para extraer en {document_name}")
            empty = DocumentExtraction(
                source_document=document_name,
                document_type=document_type or "otro",
                extracted_fields={field: None for field in self.config.REQUIRED_FIELDS},
                extraction_metadata={"warning": "no_content"},
            )
            if use_cache:
                self.extraction_cache[cache_key] = empty
            self._log_extraction_metrics(empty)
            return empty

        # Determinar la ruta de procesamiento
        route = self._determine_route(document_name, prepared_content)
        
        # Construir prompt con guías si el tipo es conocido
        if document_type in self.field_mapping:
            # Usar prompt guiado
            prompt = self.prompt_builder.build_guided_extraction_prompt(
                document_name=document_name,
                document_type=document_type,
                content=prepared_content,
                route=route.value if isinstance(route, ExtractionRoute) else route
            )
        else:
            # Usar prompt mejorado legacy
            prompt = self._build_enhanced_prompt(
                document_name=document_name,
                document_type=document_type or "otro",
                ocr_content=prepared_content,
            )

        # Llamar IA con reintentos (ahora acepta que el LLM responda solo con el JSON de campos)
        extraction = await self._call_ai_with_retry(
            prompt=prompt,
            document_name=document_name,
            document_type=document_type or "otro",
            route=route
        )
        
        # Aplicar máscara de campos permitidos
        extraction = self._apply_field_mask(extraction, document_type)

        # Post-proceso
        extraction = self._post_process_extraction(extraction)

        # Guardar en cache
        if use_cache:
            self.extraction_cache[cache_key] = extraction

        # Métricas
        self._log_extraction_metrics(extraction)

        return extraction

    async def extract_from_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        parallel_limit: int = 3,
    ) -> List[DocumentExtraction]:
        """
        Extrae campos de múltiples documentos en paralelo controlado

        Args:
            documents: Lista de dicts con {filename, ocr_result, document_type?}
            parallel_limit: Nº máx de extracciones paralelas

        Returns:
            Lista de DocumentExtraction exitosas
        """
        logger.info(f"Procesando batch de {len(documents)} documentos")

        sem = asyncio.Semaphore(parallel_limit)

        async def _extract(doc: Dict[str, Any]) -> Union[DocumentExtraction, Exception]:
            async with sem:
                try:
                    return await self.extract_from_document(
                        ocr_result=doc["ocr_result"],
                        document_name=doc["filename"],
                        document_type=doc.get("document_type"),
                    )
                except Exception as e:
                    logger.error(f"Error procesando {doc.get('filename', 'desconocido')}: {e}")
                    return e

        tasks = [_extract(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        extractions: List[DocumentExtraction] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Falló {documents[i].get('filename', 'desconocido')}: {r}")
            else:
                extractions.append(r)

        logger.info(f"Extracciones exitosas: {len(extractions)}/{len(documents)}")
        return extractions

    # =============================================================================
    #   REINTENTO + PARSEO ROBUSTO DE JSON DE CAMPOS
    # =============================================================================
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_ai_with_retry(
        self,
        prompt: str,
        document_name: str,
        document_type: str,
        route: Optional[str] = None,
    ) -> DocumentExtraction:
        """
        Llama a la API de OpenAI con reintentos automáticos.
        Acepta que el LLM responda SOLO con el JSON de campos extraídos.
        Construye un DocumentExtraction válido con esos campos.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.get_model_for_task("extraction"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un experto en extracción de datos de documentos de seguros. "
                            "Responde SOLO con un JSON válido de campos solicitados; no agregues texto extra."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.OPENAI_CONFIG.get("temperature", 0.1),
                max_tokens=self.config.OPENAI_CONFIG.get("max_tokens", 1200),
            )

            content = ""
            if response and response.choices:
                content = response.choices[0].message.content or ""

            # Quitar fences si vienen con ```json
            content = content.strip()
            if content.startswith("```"):
                # elimina fences tipo ```json ... ```
                content = content.strip("`")
                # si quedó con "json\n{...}", corta a partir de la primera "{"
                brace_pos = content.find("{")
                if brace_pos != -1:
                    content = content[brace_pos:]

            # Intentar parsear JSON
            fields_dict: Dict[str, Any] = {}
            try:
                fields_dict = json.loads(content)
                if not isinstance(fields_dict, dict):
                    raise ValueError("La respuesta no es un objeto JSON")
            except Exception as pe:
                logger.error(f"No se pudo parsear JSON para {document_name}: {pe}")
                fields_dict = {}

            # Normalizar: solo mantener campos requeridos y asegurar presencia
            normalized_fields: Dict[str, Optional[Any]] = {
                field: fields_dict.get(field, None) for field in self.config.REQUIRED_FIELDS
            }

            return DocumentExtraction(
                source_document=document_name,
                document_type=document_type or "otro",
                extracted_fields=normalized_fields,
                extraction_metadata={
                    "raw_response_len": len(content),
                    "parsed_ok": bool(fields_dict),
                },
            )

        except Exception as e:
            # Cualquier error → se reintenta por tenacity
            logger.error(f"Error llamando a OpenAI para {document_name}: {e}")
            # En el último reintento fallido, tenacity propagará; captúalo arriba si quieres
            raise

    # -------------------------------
    # Helpers de pre/post-procesado
    # -------------------------------

    def _prepare_ocr_content(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepara y limpia el contenido del OCR/Parser para la IA.
        """
        prepared = {
            "text": (ocr_result.get("text") or "")[:10000],  # cota dura de texto
            "key_value_pairs": ocr_result.get("key_value_pairs") or {},
            "tables": self._simplify_tables(ocr_result.get("tables") or []),
        }

        if len(prepared["text"]) > 5000:
            prepared["text"] = self._extract_relevant_sections(prepared["text"])

        return prepared

    def _simplify_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simplifica tablas para reducir tokens.
        """
        simplified: List[Dict[str, Any]] = []
        for t in tables[:5]:  # máx 5 tablas
            simplified.append(
                {
                    "headers": t.get("headers", []),
                    "rows": (t.get("data_rows") or [])[:10],  # máx 10 filas
                }
            )
        return simplified

    def _extract_relevant_sections(self, text: str) -> str:
        """
        Extrae secciones relevantes del texto basándose en palabras clave.
        """
        keywords = [
            "póliza",
            "poliza",
            "asegurado",
            "vigencia",
            "siniestro",
            "monto",
            "fecha",
            "reclamación",
            "reclamacion",
            "cobertura",
        ]
        lines = text.split("\n")
        relevant = [ln for ln in lines if any(kw in ln.lower() for kw in keywords)]
        if len(relevant) > 10:
            return "\n".join(relevant[:100])
        return "\n".join(lines[:100])

    def _detect_document_type(self, ocr_result: Dict[str, Any], filename: str) -> str:
        """
        Detecta el tipo de documento basándose en el contenido y nombre.
        """
        text_lower = (ocr_result.get("text") or "").lower()
        filename_lower = (filename or "").lower()

        type_indicators = {
            "poliza": ["póliza", "poliza", "cobertura", "vigencia", "prima", "asegurado"],
            "factura": ["factura", "cfdi", "subtotal", "iva", "total"],
            "denuncia": ["denuncia", "ministerio", "querella", "declaración", "declaracion"],
            "peritaje": ["peritaje", "dictamen", "evaluación", "evaluacion", "daños", "danos"],
            "carta_porte": ["carta porte", "transportista", "remitente", "destinatario"],
        }

        scores: Dict[str, int] = {}
        for doc_type, kws in type_indicators.items():
            score = sum(1 for kw in kws if kw in text_lower or kw in filename_lower)
            if score:
                scores[doc_type] = score

        return max(scores, key=scores.get) if scores else "otro"

    def _post_process_extraction(self, extraction: DocumentExtraction) -> DocumentExtraction:
        """
        Post-procesa la extracción para limpiar y validar datos.
        """
        fields = dict(extraction.extracted_fields or {})

        # Normalizar nulos / vacíos
        for k, v in list(fields.items()):
            if v in ("", "null", "None", "N/A", "NO ESPECIFICADO"):
                fields[k] = None

        # Fechas conocidas
        date_fields = ["fecha_ocurrencia", "fecha_reclamacion", "vigencia_inicio", "vigencia_fin"]
        for f in date_fields:
            if fields.get(f):
                fields[f] = self._format_date(fields[f])

        # Monto
        if fields.get("monto_reclamacion") is not None:
            fields["monto_reclamacion"] = self._format_amount(fields["monto_reclamacion"])

        extraction.extracted_fields = fields
        return extraction

    def _format_date(self, date_str: Any) -> Optional[str]:
        """Formatea una fecha a YYYY-MM-DD (placeholder simple)."""
        if not date_str:
            return None
        # Aquí podrías usar dateparser o reglas específicas.
        return str(date_str)

    def _format_amount(self, amount: Any) -> Optional[float]:
        """Formatea un monto a float robustamente."""
        if amount is None:
            return None
        if isinstance(amount, (int, float)):
            return float(amount)
        if isinstance(amount, str):
            # Quitar todo lo que no sea número/.,,
            clean = re.sub(r"[^\d.,-]", "", amount)
            # Normalizar: si hay ambas , y ., asume formato latino "1.234,56"
            if "," in clean and "." in clean:
                clean = clean.replace(".", "").replace(",", ".")
            else:
                clean = clean.replace(",", "")
            try:
                return float(clean)
            except Exception:
                return None
        return None

    def _generate_cache_key(self, document_name: str, ocr_result: Any) -> str:
        """Genera una clave única para el cache a partir de nombre+texto."""
        if hasattr(ocr_result, "text"):
            text_snip = (getattr(ocr_result, "text", "") or "")[:1000]
        else:
            try:
                text_snip = (ocr_result.get("text") or "")[:1000]
            except Exception:
                text_snip = ""
        try:
            payload = json.dumps({"name": document_name, "text": text_snip}, ensure_ascii=False)
        except Exception:
            payload = f"{document_name}:{text_snip}"
        import hashlib

        return hashlib.md5(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _ocr_to_dict_safe(self, ocr_result: Any) -> Dict[str, Any]:
        """Convierte OCRResult-like a dict mínimo para funciones que esperan dict."""
        if hasattr(ocr_result, "text"):
            return {
                "text": getattr(ocr_result, "text", "") or "",
                "key_value_pairs": getattr(ocr_result, "key_values", {}) or {},
                "tables": getattr(ocr_result, "tables", []) or [],
            }
        return ocr_result or {}

    def _log_extraction_metrics(self, extraction: DocumentExtraction) -> None:
        """Registra métricas de la extracción."""
        total = len(extraction.extracted_fields or {})
        filled = sum(1 for v in (extraction.extracted_fields or {}).values() if v is not None)
        logger.info(
            f"Documento: {extraction.source_document} | Campos extraídos: {filled}/{total} | Tipo: {extraction.document_type}"
        )

    # =============================================================================
    #   NUEVO: Prompt de extracción mejorado
    # =============================================================================
    def _build_enhanced_prompt(
        self,
        document_name: str,
        document_type: str,
        ocr_content: Dict[str, Any],
    ) -> str:
        """
        Construye un prompt mejorado para extracción.
        El LLM debe responder SOLO con el JSON de los campos requeridos.
        """
        # Campos específicos por tipo de documento
        field_focus = {
            "poliza": ["numero_poliza", "nombre_asegurado", "vigencia_inicio", "vigencia_fin"],
            "factura": ["rfc", "monto_reclamacion", "fecha_ocurrencia"],
            "denuncia": ["fecha_ocurrencia", "lugar_hechos", "tipo_siniestro"],
        }
        priority_fields = field_focus.get(document_type, list(self.config.REQUIRED_FIELDS)[:5])

        # Para evitar respuestas enormes, recortamos texto si viene muy largo
        text_preview = (ocr_content.get("text") or "")[:3000]

        prompt = f"""
Analiza este documento tipo '{document_type}' y extrae la siguiente información.

DOCUMENTO: {document_name}

CONTENIDO OCR:
{text_preview}

CAMPOS CLAVE-VALOR DETECTADOS:
{json.dumps(ocr_content.get('key_value_pairs', {}), ensure_ascii=False, indent=2)}

INSTRUCCIONES:
1. Extrae ÚNICAMENTE los valores que encuentres en el documento.
2. Si no encuentras un campo, déjalo como null.
3. NO inventes información.
4. Presta especial atención a estos campos: {', '.join(priority_fields)}.

CAMPOS REQUERIDOS:
{json.dumps({field: "valor extraído o null" for field in self.config.REQUIRED_FIELDS}, ensure_ascii=False, indent=2)}

Responde SOLO con el JSON de los campos extraídos.
"""
        return prompt
    
    def _determine_route(self, document_name: str, content: Dict[str, Any]) -> str:
        """
        Determina la ruta de procesamiento según el tipo de archivo y contenido
        """
        # Obtener extensión del archivo
        ext = Path(document_name).suffix.lower()
        
        # Verificar configuración de ruta
        if ext in self.route_config:
            route = self.route_config[ext]
            if isinstance(route, ExtractionRoute):
                return route.value
            elif route == "auto":
                # Decidir según contenido
                if self._is_scanned_document(content):
                    return ExtractionRoute.DIRECT_AI.value
                else:
                    return ExtractionRoute.OCR_TEXT.value
            return route
        
        # Por defecto, usar OCR + texto
        return ExtractionRoute.OCR_TEXT.value
    
    def _is_scanned_document(self, content: Dict[str, Any]) -> bool:
        """
        Detecta si es un documento escaneado (principalmente imagen)
        """
        text = content.get("text", "")
        tables = content.get("tables", [])
        
        # Si hay muy poco texto pero hay contenido, probablemente es escaneado
        if len(text) < 100 and not tables:
            return True
        
        # Ratio de caracteres especiales (documentos escaneados mal OCR tienen muchos)
        if text:
            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
            ratio = special_chars / len(text)
            if ratio > 0.3:  # Más del 30% caracteres raros
                return True
        
        return False
    
    def _apply_field_mask(self, extraction: DocumentExtraction, document_type: str) -> DocumentExtraction:
        """
        Aplica máscara de campos permitidos según el tipo de documento
        """
        # Si no hay mapeo para este tipo, retornar sin cambios
        if document_type not in self.field_mapping:
            logger.info(f"No hay restricciones de campos para tipo: {document_type}")
            return extraction
        
        # Obtener campos permitidos
        allowed_fields = set(self.field_mapping[document_type])
        
        # Aplicar máscara
        masked_fields = {}
        for field, value in extraction.extracted_fields.items():
            if field in allowed_fields:
                masked_fields[field] = value
            else:
                # Campo no permitido → null
                masked_fields[field] = None
                if value is not None:
                    logger.debug(f"Campo '{field}' anulado para documento tipo '{document_type}' (valor era: {value})")
        
        # Actualizar extracción
        extraction.extracted_fields = masked_fields
        
        # Agregar metadata sobre la máscara aplicada
        if not extraction.extraction_metadata:
            extraction.extraction_metadata = {}
        extraction.extraction_metadata["field_mask_applied"] = True
        extraction.extraction_metadata["allowed_fields"] = list(allowed_fields)
        
        return extraction
    
    async def extract_from_document_guided(
        self,
        content: Union[Dict[str, Any], bytes, Path],
        document_name: str,
        document_type: str,
        route: str = "ocr_text",
        model: str = None,
        use_cache: bool = True
    ) -> DocumentExtraction:
        """
        Extracción guiada con doble ruta y validación estricta
        
        Args:
            content: OCR dict, bytes de imagen, o Path al archivo
            document_name: Nombre del documento
            document_type: Tipo detectado (informe_preliminar_del_ajustador, etc.)
            route: "direct_ai" o "ocr_text"
            model: Modelo a usar (si None, usa el default)
        """
        from pathlib import Path
        
        # Usar modelo default si no se especifica
        if model is None:
            model = self.config.get_model_for_task("extraction")
        
        # Cache key considerando la ruta y tipo
        cache_key = f"{document_name}_{document_type}_{route}"
        if use_cache and cache_key in self.extraction_cache:
            logger.info(f"Usando cache de extracción guiada para {document_name}")
            return self.extraction_cache[cache_key]
        
        logger.info(
            f"Extracción guiada iniciada:\n"
            f"  Documento: {document_name}\n"
            f"  Tipo: {document_type}\n"
            f"  Ruta: {route}\n"
            f"  Modelo: {model}"
        )
        
        # 1. Obtener campos permitidos para este documento
        allowed_fields = self.field_mapping.get(document_type, [])
        
        if not allowed_fields:
            logger.warning(
                f"Documento tipo '{document_type}' no tiene campos permitidos. "
                f"Retornando todos los campos como null."
            )
            return self._create_null_extraction(document_name, document_type)
        
        # 2. Construir prompt con guía
        prompt = self.prompt_builder.build_guided_extraction_prompt(
            document_name=document_name,
            document_type=document_type,
            content=content if route == "ocr_text" else None,
            route=route
        )
        
        # 3. Ejecutar extracción según ruta
        try:
            if route == "direct_ai":
                raw_extraction = await self._extract_direct_ai(
                    content=content,
                    prompt=prompt,
                    model=model
                )
            else:
                raw_extraction = await self._extract_ocr_text(
                    prompt=prompt,
                    model=model,
                    ocr_content=content
                )
        except Exception as e:
            logger.error(f"Error en extracción: {e}")
            raw_extraction = {}
        
        # 4. Aplicar máscara y validaciones
        extraction = self._apply_field_mask_dict(raw_extraction, allowed_fields)
        extraction = self._validate_and_transform_dict(extraction, document_type)
        
        # 5. Crear resultado
        result = DocumentExtraction(
            source_document=document_name,
            document_type=document_type,
            extracted_fields=extraction,
            extraction_metadata={
                "route": route,
                "model_used": model,
                "guide_applied": True,
                "allowed_fields": allowed_fields,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Guardar en cache
        if use_cache:
            self.extraction_cache[cache_key] = result
        
        return result
    
    async def _extract_direct_ai(
        self, 
        content: Union[bytes, Path],
        prompt: str,
        model: str
    ) -> Dict[str, Any]:
        """
        Extracción usando visión directa con GPT-4V
        """
        from pathlib import Path
        logger.info(f"Iniciando extracción Direct AI con modelo {model}")
        
        # Preparar imagen
        if isinstance(content, Path):
            with open(content, 'rb') as f:
                image_bytes = f.read()
        else:
            image_bytes = content
        
        # Codificar en base64
        import base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        try:
            # Llamada a la API con visión
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parsear respuesta
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            logger.info(f"Direct AI extracción exitosa")
            return result
            
        except Exception as e:
            logger.error(f"Error en Direct AI: {e}")
            return {}
    
    async def _extract_ocr_text(
        self, 
        prompt: str,
        model: str,
        ocr_content: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extracción usando OCR + texto con IA
        """
        logger.info(f"Iniciando extracción OCR + texto con modelo {model}")
        
        try:
            # Llamada a la API
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parsear respuesta
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            logger.info(f"OCR + texto extracción exitosa")
            return result
            
        except Exception as e:
            logger.error(f"Error en OCR + texto: {e}")
            return {}
    
    def _apply_field_mask_dict(
        self, 
        extraction: Dict[str, Any],
        allowed_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Aplica máscara de seguridad a un diccionario
        """
        masked = {}
        
        for field in self.config.REQUIRED_FIELDS:
            if field in allowed_fields:
                masked[field] = extraction.get(field, None)
            else:
                masked[field] = None
                if field in extraction and extraction[field] is not None:
                    logger.warning(f"Campo '{field}' bloqueado (no permitido)")
        
        return masked
    
    def _validate_and_transform_dict(
        self,
        extraction: Dict[str, Any],
        document_type: str
    ) -> Dict[str, Any]:
        """
        Aplica validaciones y transformaciones a un diccionario
        """
        validated = extraction.copy()
        
        for field, value in extraction.items():
            if value is None:
                continue
            
            rules = self.validation_rules.get(field, {})
            
            # Aplicar transformaciones
            if 'transform' in rules and callable(rules['transform']):
                try:
                    validated[field] = rules['transform'](value)
                except Exception as e:
                    logger.error(f"Error transformando {field}: {e}")
            
            # Validar formato de fecha
            if rules.get('type') == 'date' and value:
                validated[field] = self._normalize_date(value)
            
            # Validar tipo numérico
            if rules.get('type') == 'float' and value:
                try:
                    validated[field] = float(str(value).replace(',', '').replace('$', ''))
                except:
                    logger.error(f"No se pudo convertir {field} a float: {value}")
        
        return validated
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """
        Normaliza fechas al formato DD/MM/YYYY
        """
        if not date_str:
            return None
        
        # Mapeo de meses abreviados
        month_map = {
            'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
        }
        
        # Reemplazar meses abreviados
        for abbr, num in month_map.items():
            date_str = date_str.replace(abbr, num)
        
        # Intentar diferentes formatos
        import re
        
        # Formato YYYY-MM-DD
        if match := re.match(r'(\d{4})-(\d{2})-(\d{2})', date_str):
            return f"{match.group(3)}/{match.group(2)}/{match.group(1)}"
        
        # Formato DD/MM/YYYY
        if match := re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str):
            return f"{match.group(1).zfill(2)}/{match.group(2).zfill(2)}/{match.group(3)}"
        
        return date_str
    
    def _create_null_extraction(self, document_name: str, document_type: str) -> DocumentExtraction:
        """
        Crea una extracción con todos los campos en null
        """
        null_fields = {field: None for field in self.config.REQUIRED_FIELDS}
        
        return DocumentExtraction(
            source_document=document_name,
            document_type=document_type,
            extracted_fields=null_fields,
            extraction_metadata={
                "route": "null",
                "guide_applied": True,
                "reason": "document_type_not_recognized",
                "timestamp": datetime.now().isoformat()
            }
        )
