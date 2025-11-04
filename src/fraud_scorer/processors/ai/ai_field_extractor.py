# src/fraud_scorer/ai_extractors/ai_field_extractor.py
"""
AIFieldExtractor: Extrae campos de documentos individuales usando IA
"""

from __future__ import annotations

import os
import re
import json
import logging
import base64
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import asyncio
from datetime import datetime

import httpx
from openai import AsyncOpenAI
import instructor
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
import random
import time

from fraud_scorer.settings import (
    ExtractionConfig,
    ExtractionRoute,
    get_model_for_task,
    MODEL_SAMPLING_CONFIG,
)
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
        # Contexto de póliza (p. ej., HDI_EN_MI_CASA)
        self.policy_context: Optional[str] = None
        self.policy_type_cache: Dict[str, str] = {}
        
        # Configuración de rutas
        self.route_config = self.config.ROUTE_CONFIG
        self.document_extraction_routes = getattr(self.config, 'DOCUMENT_EXTRACTION_ROUTES', {})
        self.field_mapping = self.config.DOCUMENT_FIELD_MAPPING
        self.validation_rules = self.config.FIELD_VALIDATION_RULES

        logger.info("AIFieldExtractor inicializado con Sistema de Extracción Guiada")

    def set_policy_context(self, policy_type: str, case_id: Optional[str] = None) -> None:
        """Establece el contexto de tipo de póliza para todo el proceso de extracción."""
        self.policy_context = policy_type
        if case_id:
            self.policy_type_cache[case_id] = policy_type
        logger.info(f"Contexto de póliza establecido: {policy_type}")

    def _get_text_content(self, ocr_result: Dict[str, Any]) -> str:
        """Extrae texto del resultado OCR de manera segura (acepta dict u objeto)."""
        if isinstance(ocr_result, dict):
            return ocr_result.get("text", "") or ""
        if hasattr(ocr_result, "text"):
            return getattr(ocr_result, "text", "") or ""
        return ""

    async def detect_policy_type(
        self,
        ocr_result: Dict[str, Any],
        document_type: str
    ) -> Optional[str]:
        """Detecta el tipo de póliza. Retorna 'HDI_EN_MI_CASA' o None."""
        try:
            if document_type != "poliza_de_la_aseguradora":
                return None
            # Normalizar OCR a dict seguro
            ocr_dict = self._ocr_to_dict_safe(ocr_result)
            text = (ocr_dict.get("text") or "").upper()
            kv_pairs = ocr_dict.get("key_value_pairs") or {}

            # Buscar en pares clave-valor
            for key, value in kv_pairs.items():
                try:
                    k = str(key).lower()
                    v = str(value).upper()
                except Exception:
                    continue
                if ("tipo" in k and "poliza" in k) or ("tipo" in k and "póliza" in k):
                    if "HDI EN MI CASA" in v:
                        logger.info(f"Detectado HDI EN MI CASA en KV '{key}'")
                        return "HDI_EN_MI_CASA"

            # Búsqueda en texto completo con patrones de contexto
            if "HDI EN MI CASA" in text:
                patterns = [
                    r"TIPO\s+DE\s+P[OÓ]LIZA[:\s]+HDI\s+EN\s+MI\s+CASA",
                    r"P[OÓ]LIZA\s+HDI\s+EN\s+MI\s+CASA",
                ]
                for pat in patterns:
                    if re.search(pat, text, re.IGNORECASE):
                        logger.info(f"Detectado HDI EN MI CASA por patrón: {pat}")
                        return "HDI_EN_MI_CASA"

            return None
        except Exception as e:
            logger.error(f"Error detectando tipo de póliza: {e}")
            return None

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
        route = self._determine_route(document_name, prepared_content, document_type)
        allowed_fields_for_mask = self._get_allowed_fields(document_type)
        if allowed_fields_for_mask:
            expected_fields = sorted(allowed_fields_for_mask)
        else:
            expected_fields = sorted(self.config.REQUIRED_FIELDS)

        # Construir prompt con guías si el tipo es conocido
        if document_type in self.field_mapping:
            # Usar prompt guiado
            prompt = self.prompt_builder.build_guided_extraction_prompt(
                document_name=document_name,
                document_type=document_type,
                content=prepared_content,
                route=route.value if isinstance(route, ExtractionRoute) else route,
                policy_type=self.policy_context
            )
        else:
            # Usar prompt mejorado legacy
            prompt = self._build_enhanced_prompt(
                document_name=document_name,
                document_type=document_type or "otro",
                ocr_content=prepared_content,
            )

        # Definir modelo específico para carpeta de investigación (requiere GPT-5)
        doc_type_normalized = (document_type or "otro").lower()
        model_override = "gpt-5" if doc_type_normalized == "carpeta_de_investigacion" else None

        # Llamar IA con reintentos (ahora acepta que el LLM responda solo con el JSON de campos)
        text_snippet = (prepared_content.get("text") or "")[:8000]
        extraction = await self._call_ai_with_retry(
            prompt=prompt,
            document_name=document_name,
            document_type=document_type or "otro",
            route=route,
            ocr_text_snippet=text_snippet,
            model_override=model_override,
            expected_fields=expected_fields,
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
                    document_type = doc.get("document_type")
                    if not document_type:
                        detected = self._detect_document_type(
                            self._ocr_to_dict_safe(doc.get("ocr_result", {})),
                            doc.get("filename", "desconocido"),
                        )
                        document_type = detected

                    if document_type and document_type in self.field_mapping:
                        route_value = doc.get("route") or "ocr_text"
                        return await self.extract_from_document_guided(
                            content=doc["ocr_result"],
                            document_name=doc["filename"],
                            document_type=document_type,
                            route=route_value,
                        )

                    # Fallback al flujo legacy si no hay guías disponibles
                    return await self.extract_from_document(
                        ocr_result=doc["ocr_result"],
                        document_name=doc["filename"],
                        document_type=document_type,
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
    async def _call_ai_with_retry(
        self,
        prompt: str,
        document_name: str,
        document_type: str,
        route: Optional[str] = None,
        ocr_text_snippet: Optional[str] = None,
        *,
        complexity: str = "normal",
        model_override: Optional[str] = None,
        expected_fields: Optional[List[str]] = None,
    ) -> DocumentExtraction:
        """
        Llama a la API de OpenAI con reintentos inteligentes.
        - No reintenta errores 400 (parámetros inválidos)
        - Reintenta 429 con backoff exponencial + jitter
        - Reintenta 500+ con backoff
        """
        escalated = (complexity or "normal").lower() == "high"
        openai_params = self.config.get_openai_params("extraction", escalated=escalated)
        max_retries = int(openai_params.get("max_retries", 3))
        base_delay = 2  # segundos

        if expected_fields is not None:
            allowed_fields: List[str] = sorted(set(expected_fields))
        else:
            allowed_fields_set = set(self.config.REQUIRED_FIELDS)
            if document_type in self.field_mapping:
                allowed_fields_set.update(self.field_mapping.get(document_type, []) or [])
            allowed_fields = sorted(allowed_fields_set)

        for attempt in range(max_retries):
            try:
                model_name = model_override or get_model_for_task(
                    "extraction", route or "ocr_text", complexity=complexity
                )
                supports_sampling = self._supports_custom_sampling(model_name)
                request_kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Eres un experto en extracción de datos de documentos de seguros. "
                                "Responde SOLO con un JSON válido de campos solicitados; no agregues texto extra."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_completion_tokens": openai_params.get("max_completion_tokens", 1600),
                }
                if supports_sampling and "temperature" in openai_params:
                    request_kwargs["temperature"] = openai_params["temperature"]
                if supports_sampling and "top_p" in openai_params:
                    request_kwargs["top_p"] = openai_params["top_p"]
                if "timeout" in openai_params:
                    request_kwargs["timeout"] = openai_params["timeout"]

                response = await self.client.chat.completions.create(**request_kwargs)

                # Éxito - procesar respuesta
                message = response.choices[0].message if response and response.choices else None
                raw_content = ""
                fields_dict: Dict[str, Any] = {}
                if message is not None:
                    raw_content = self._stringify_message_content(message)
                    fields_dict = self._parse_json_payload(
                        raw_content,
                        document_name=document_name,
                        model_name=model_name,
                        route=route,
                    )

                if not raw_content:
                    raw_content = "{}"

                # Normalizar campos alineados con el tipo de documento
                normalized_fields: Dict[str, Optional[Any]] = {
                    field: fields_dict.get(field, None) for field in allowed_fields
                }

                extraction = DocumentExtraction(
                    source_document=document_name,
                    document_type=document_type or "otro",
                    extracted_fields=normalized_fields,
                    extraction_metadata={
                        "raw_response_len": len(raw_content),
                        "parsed_ok": bool(fields_dict),
                        "route_used": route or "ocr_text",
                        "model_used": model_name,
                    },
                )
                self._record_model_attempt(extraction, model_name)
                if ocr_text_snippet:
                    try:
                        snippet = (ocr_text_snippet or "")[:4000]
                        metadata = extraction.extraction_metadata
                        metadata.setdefault("raw_text_snippet", snippet)
                        metadata.setdefault("raw_text", snippet)
                    except Exception:
                        logger.debug("No se pudo adjuntar raw_text_snippet", exc_info=True)
                return extraction
                
            except Exception as e:
                error_code = getattr(e, 'status_code', None) or getattr(e, 'http_status', None)
                error_type = getattr(e, 'error_type', None) or type(e).__name__

                # No reintentar errores 400 (parámetros inválidos)
                if error_code == 400 or "invalid_request_error" in str(e):
                    logger.error(f"Error 400 - Parámetros inválidos para {document_name}: {e}")
                    # Retornar extracción vacía sin reintentar
                    extraction = DocumentExtraction(
                        source_document=document_name,
                        document_type=document_type or "otro",
                        extracted_fields={field: None for field in self.config.REQUIRED_FIELDS},
                        extraction_metadata={
                            "error": str(e),
                            "error_type": "invalid_request",
                            "route_used": route or "ocr_text",
                        },
                    )
                    if ocr_text_snippet:
                        try:
                            snippet = (ocr_text_snippet or "")[:8000]
                            metadata = extraction.extraction_metadata
                            metadata.setdefault("raw_text_snippet", snippet)
                            metadata.setdefault("raw_text", snippet)
                        except Exception:
                            logger.debug("No se pudo adjuntar raw_text_snippet", exc_info=True)
                    return extraction

                # Para 429 (rate limit), esperar con backoff + jitter
                if error_code == 429 or "rate_limit" in str(e).lower():
                    retry_after = getattr(e, 'retry_after', None)
                    if retry_after:
                        delay = float(retry_after)
                    else:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)

                    logger.warning(f"Rate limit hit para {document_name}. Esperando {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue  # Reintentar

                # Para errores de conexión/timeouts, reintentar con backoff moderado
                if isinstance(e, (httpx.RequestError, asyncio.TimeoutError)) or "connection error" in str(e).lower():
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Connection issue para {document_name}. Reintentando en {delay:.1f}s... ({e})")
                    await asyncio.sleep(delay)
                    continue

                # Para errores 500+, reintentar con backoff
                if error_code and error_code >= 500:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Error {error_code} para {document_name}. Reintentando en {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                
                # Para otros errores en el último intento
                if attempt == max_retries - 1:
                    logger.error(f"Fallo tras {max_retries} intentos para {document_name}: {e}")
                    extraction = DocumentExtraction(
                        source_document=document_name,
                        document_type=document_type or "otro",
                        extracted_fields={field: None for field in self.config.REQUIRED_FIELDS},
                        extraction_metadata={
                            "error": str(e),
                            "attempts": attempt + 1,
                            "route_used": route or "ocr_text",
                        },
                    )
                    if ocr_text_snippet:
                        try:
                            snippet = (ocr_text_snippet or "")[:8000]
                            metadata = extraction.extraction_metadata
                            metadata.setdefault("raw_text_snippet", snippet)
                            metadata.setdefault("raw_text", snippet)
                        except Exception:
                            logger.debug("No se pudo adjuntar raw_text_snippet", exc_info=True)
                    return extraction
                
                # Reintentar otros errores
                await asyncio.sleep(base_delay * (2 ** attempt))
        
        # Si llegamos aquí, fallaron todos los intentos
        logger.error(f"No se pudo procesar {document_name} tras {max_retries} intentos")
        extraction = DocumentExtraction(
            source_document=document_name,
            document_type=document_type or "otro",
            extracted_fields={field: None for field in self.config.REQUIRED_FIELDS},
            extraction_metadata={
                "error": "Max retries exceeded",
                "route_used": route or "ocr_text",
            },
        )
        if ocr_text_snippet:
            try:
                snippet = (ocr_text_snippet or "")[:8000]
                metadata = extraction.extraction_metadata
                metadata.setdefault("raw_text_snippet", snippet)
                metadata.setdefault("raw_text", snippet)
            except Exception:
                logger.debug("No se pudo adjuntar raw_text_snippet", exc_info=True)
        return extraction

    # -------------------------------
    # Helpers de pre/post-procesado
    # -------------------------------

    def _prepare_ocr_content(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepara y limpia el contenido del OCR/Parser para la IA.
        """
        raw_text = ocr_result.get("text") or ""
        max_length = 6000

        prepared = {
            "text": raw_text[:max_length],
            "key_value_pairs": ocr_result.get("key_value_pairs") or {},
            "tables": self._simplify_tables(ocr_result.get("tables") or []),
        }

        if len(raw_text) > max_length:
            prepared["text"] = self._extract_relevant_sections(raw_text, max_items=max_length)

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

    def _extract_relevant_sections(self, text: str, *, max_items: int = 6000) -> str:
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

        selected_lines: List[str]
        if relevant:
            selected_lines = relevant[:200]
        else:
            selected_lines = lines[:200]

        snippet = "\n".join(selected_lines)
        return snippet[:max_items]

    async def _chat_with_retry(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2000,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_retries: int = 3,
        timeout: Optional[int] = None,
    ) -> Any:
        """
        Llamada a chat.completions con backoff y compatibilidad de parámetros.
        - Reintenta 429/5xx con backoff exponencial + jitter
        - No reintenta 400 invalid_request
        """
        base_delay = 2
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                }
                sampling_overrides = MODEL_SAMPLING_CONFIG.get(model, {})
                temp_override = sampling_overrides.get("temperature", temperature)
                top_p_override = sampling_overrides.get("top_p", top_p)
                if temp_override is not None and self._supports_custom_sampling(model):
                    kwargs["temperature"] = temp_override
                if top_p_override is not None and self._supports_custom_sampling(model):
                    kwargs["top_p"] = top_p_override
                if timeout is not None:
                    kwargs["timeout"] = timeout
                # Llamada
                return await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                code = getattr(e, 'status_code', None) or getattr(e, 'http_status', None)
                msg = str(e).lower()
                # No reintentar 400 invalid_request
                if code == 400 or "invalid_request_error" in msg:
                    logger.error(f"Error 400 (no recuperable): {e}")
                    raise
                # 429 backoff
                if code == 429 or "rate limit" in msg:
                    delay = base_delay * (2 ** attempt) + (0.5)
                    logger.warning(f"Rate limit 429, esperando {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                # Errores de conexión/timeouts → reintentar
                if isinstance(e, (httpx.RequestError, asyncio.TimeoutError)) or "connection error" in msg:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Connection error en chat, reintento en {delay:.1f}s ({e})")
                    await asyncio.sleep(delay)
                    continue
                # 5xx backoff
                if code and code >= 500:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Error {code}, reintento en {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                # Otros errores → no reintentar
                logger.error(f"Error no recuperable: {e}")
                raise

    def _detect_document_type(self, ocr_result: Dict[str, Any], filename: str) -> str:
        """
        Detecta el tipo de documento basándose en el contenido y nombre.
        Devuelve SIEMPRE un tipo canónico compatible con DOCUMENT_FIELD_MAPPING.
        """
        text_lower = (ocr_result.get("text") or "").lower()
        filename_lower = (filename or "").lower()

        # PRIORIDAD: expediente_de_cobranza por contenido (evitar depender del nombre)
        cobranza_cues = [
            "cobranza", "cumplimiento", "recibo", "recibos", "pagado", "pagos",
            "vigencia", "validación de cobertura", "validez cobertura", "estado de cuenta", "primas"
        ]
        if any(cue in text_lower for cue in cobranza_cues):
            return "expediente_de_cobranza"

        # Regla de identificación oficial (más estricta para evitar falsos positivos)
        id_core = ["instituto nacional electoral", "credencial para votar", "clave de elector"]
        id_signals = 0
        if any(k in text_lower for k in id_core):
            id_signals += 2  # señales fuertes
        if any(k in text_lower for k in ["ine", "ife", "pasaporte", "cédula", "cedula", "licencia"]):
            id_signals += 1
        if any(k in text_lower for k in ["curp", "vigencia", "sexo", "nacionalidad"]):
            id_signals += 1
        # También considerar el nombre del archivo, pero no como único criterio
        if any(k in filename_lower for k in ["identificacion", "identificación", "ine", "ife", "pasaporte", "cedula", "cédula"]):
            id_signals += 1
        if id_signals >= 2:
            return "identificacion_oficial"

        # Reglas prioritarias por nombre de archivo
        if "informe preliminar" in filename_lower or "informe_preliminar" in filename_lower:
            return "informe_preliminar_del_ajustador"
        if "informe final" in filename_lower or "informe_final" in filename_lower:
            return "informe_final_del_ajustador"
        if "poliza" in filename_lower or "póliza" in filename_lower:
            return "poliza_de_la_aseguradora"
        if "cobranza" in filename_lower:
            return "expediente_de_cobranza"
        if "acreditacion" in filename_lower or "acreditación" in filename_lower:
            return "acreditacion_de_propiedad_y_representacion"
        if "carta" in filename_lower and "reclamacion" in filename_lower:
            # Detectar aseguradora vs transportista
            if any(k in filename_lower for k in ["transportista", "transporte", "paqueteria", "paquetería"]):
                return "carta_de_reclamacion_formal_al_transportista"
            # Señales de aseguradora (nombres comunes o palabra aseguradora)
            if any(k in filename_lower for k in ["aseguradora", "hdi", "axa", "qualitas", "gnp", "mapfre", "zurich"]):
                return "carta_de_reclamacion_formal_a_la_aseguradora"
            # Por defecto, asumir aseguradora
            return "carta_de_reclamacion_formal_a_la_aseguradora"
        if "denuncia" in filename_lower:
            return "denuncia_de_los_hechos"
        if "aviso de siniestro" in filename_lower or "aviso_de_siniestro" in filename_lower:
            return "aviso_de_siniestro_transportista"
        if "checklist" in filename_lower and "antifraude" in filename_lower:
            return "checklist_antifraude"
        if "reporte gps" in filename_lower or "gps" in filename_lower:
            return "reporte_gps"
        if "salida de almacen" in filename_lower or "salida_de_almacen" in filename_lower:
            return "salida_de_almacen"
        if any(x in filename_lower for x in ["guia", "guías", "guias", "factura"]):
            return "guias_y_facturas"

        # Indicadores básicos (no canónicos)
        type_indicators = {
            "poliza": ["póliza", "poliza", "cobertura", "vigencia", "prima", "asegurado"],
            "factura": ["factura", "cfdi", "subtotal", "iva", "total"],
            "denuncia": ["denuncia", "ministerio", "querella", "declaración", "declaracion"],
            "informe_preliminar": ["informe preliminar", "estimación inicial", "informe_ajustador"],
            "informe_final": ["informe final", "conclusión", "recomendación", "ajustador"],
            "salida_de_almacen": ["salida de almacén", "oc", "orden de salida"],
            "licencia": ["licencia", "conductor", "operador"],
            "tarjeta_circulacion": ["tarjeta de circulación", "placas", "niv"],
            "reporte_gps": ["gps", "reporte gps"],
            "aviso_siniestro": ["aviso de siniestro", "reporte siniestro", "transportista"],
        }

        scores: Dict[str, int] = {}
        for doc_type, kws in type_indicators.items():
            score = sum(1 for kw in kws if kw in text_lower or kw in filename_lower)
            if score:
                scores[doc_type] = score

        detected = max(scores, key=scores.get) if scores else "otro"

        # Mapear a tipos canónicos
        synonyms_to_canonical = {
            "poliza": "poliza_de_la_aseguradora",
            "factura": "guias_y_facturas",
            "denuncia": "denuncia_de_los_hechos",
            "informe_preliminar": "informe_preliminar_del_ajustador",
            "informe_final": "informe_final_del_ajustador",
            "salida_de_almacen": "salida_de_almacen",
            "licencia": "licencia_del_operador",
            "tarjeta_circulacion": "tarjeta_de_circulacion_vehiculo",
            "reporte_gps": "reporte_gps",
            "aviso_siniestro": "aviso_de_siniestro_transportista",
            "otro": "otro",
        }

        return synonyms_to_canonical.get(detected, "otro")

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

        for rfc_field in ("issuer_rfc", "recipient_rfc"):
            if fields.get(rfc_field):
                fields[rfc_field] = self._normalize_rfc(fields[rfc_field])

        sello = fields.get("sello_digital_sat")
        if sello:
            sanitized = self._sanitize_sello_digital(sello)
            if sanitized:
                fields["sello_digital_sat"] = sanitized
                fields["signature_last_8"] = sanitized[-8:]
            else:
                logger.debug("Sello digital inválido, descartando valor")
                fields["sello_digital_sat"] = None
                fields.pop("signature_last_8", None)

        if not fields.get("signature_last_8") and fields.get("sello_digital_sat"):
            sello_clean = fields["sello_digital_sat"]
            if sello_clean and len(sello_clean) >= 8:
                fields["signature_last_8"] = sello_clean[-8:]

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

    def _normalize_rfc(self, value: Any) -> Optional[str]:
        if not value:
            return None
        candidate = "".join(str(value).strip().upper().split())
        return candidate or None

    def _sanitize_sello_digital(self, value: Any) -> Optional[str]:
        if not value:
            return None
        candidate = "".join(str(value).strip().split())
        if len(candidate) < 8:
            return None
        # Rellenar padding si hace falta
        padding = len(candidate) % 4
        if padding:
            candidate += "=" * (4 - padding)
        try:
            base64.b64decode(candidate, validate=True)
        except Exception:
            return None
        return candidate

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
            "denuncia_de_los_hechos": ["fecha_ocurrencia", "lugar_hechos", "tipo_siniestro"],
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
    
    def _determine_route(self, document_name: str, content: Dict[str, Any], document_type: Optional[str] = None) -> str:
        """
        Determina la ruta de procesamiento según el tipo de documento y contenido
        Prioridad: 1) Tipo de documento (parámetro) 2) Extensión 3) Default
        """
        # 1. Usar tipo de documento si se proporciona como parámetro
        if document_type and document_type in self.document_extraction_routes:
            route = self.document_extraction_routes[document_type]
            if isinstance(route, ExtractionRoute):
                return route.value
            return route
        
        # 2. Fallback a extensión del archivo
        ext = Path(document_name).suffix.lower()
        
        # Verificar configuración de ruta por extensión
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
        
        # 3. Por defecto, usar OCR + texto
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

    def _get_allowed_fields(self, document_type: Optional[str]) -> Optional[set[str]]:
        if not document_type:
            return None
        allowed = self.field_mapping.get(document_type)
        return set(allowed or []) if allowed is not None else None
    
    async def extract_from_document_guided(
        self,
        content: Union[Dict[str, Any], bytes, Path],
        document_name: str,
        document_type: str,
        route: str = "ocr_text",
        model: Optional[str] = None,
        use_cache: bool = True,
        allow_escalation: bool = True,
    ) -> DocumentExtraction:
        """Extracción guiada con validaciones y escalación opcional."""

        route_value = route.value if hasattr(route, "value") else (route or "ocr_text")
        selected_model = model or get_model_for_task("extraction", route_value, complexity="normal")
        cache_key = f"{document_name}_{document_type}_{route_value}_{selected_model}"

        result: Optional[DocumentExtraction] = None
        if use_cache and cache_key in self.extraction_cache:
            logger.info(f"Usando cache de extracción guiada para {document_name} ({selected_model})")
            result = self.extraction_cache[cache_key]

        if result is None:
            logger.info(
                "Extracción guiada iniciada:\n"
                f"  Documento: {document_name}\n"
                f"  Tipo: {document_type}\n"
                f"  Ruta: {route_value}\n"
                f"  Modelo: {selected_model}"
            )
            if (self.policy_context or "") == "HDI_EN_MI_CASA":
                logger.info("HDI_EXTRACTION: contexto activo para extracción guiada")

            allowed_fields = list(self.field_mapping.get(document_type, []) or [])
            if (self.policy_context or "") == "HDI_EN_MI_CASA" and document_type == "poliza_de_la_aseguradora":
                if "lugar_hechos" not in allowed_fields:
                    allowed_fields.append("lugar_hechos")

            if not allowed_fields:
                logger.warning(
                    "Documento tipo '%s' no tiene campos permitidos; devolviendo extracción nula",
                    document_type,
                )
                return self._create_null_extraction(document_name, document_type)

            prompt = self.prompt_builder.build_guided_extraction_prompt(
                document_name=document_name,
                document_type=document_type,
                content=content if route_value == "ocr_text" else None,
                route=route_value,
                policy_type=self.policy_context,
            )

            try:
                if route_value == "direct_ai":
                    raw_extraction = await self._extract_direct_ai(
                        content=content,
                        prompt=prompt,
                        model=selected_model,
                        document_name=document_name,
                        route=route_value,
                    )
                else:
                    raw_extraction = await self._extract_ocr_text(
                        prompt=prompt,
                        model=selected_model,
                        document_name=document_name,
                        route=route_value,
                        ocr_content=content,
                    )
            except Exception as exc:
                logger.error("Error en extracción: %s", exc)
                raw_extraction = {}

            sanitized = self._apply_field_mask_dict(raw_extraction, allowed_fields)
            if self.policy_context:
                try:
                    self.validator.set_policy_type(self.policy_context)
                except Exception:
                    pass
            sanitized = self._validate_and_transform_dict(sanitized, document_type)

            result = DocumentExtraction(
                source_document=document_name,
                document_type=document_type,
                extracted_fields=sanitized,
                extraction_metadata={
                    "route": route_value,
                    "model_used": selected_model,
                    "guide_applied": True,
                    "allowed_fields": allowed_fields,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            if isinstance(content, dict):
                try:
                    snippet = (content.get("text") or "")[:4000]
                    if snippet:
                        meta = result.extraction_metadata
                        meta.setdefault("raw_text_snippet", snippet)
                        meta.setdefault("raw_text", snippet)
                except Exception:
                    logger.debug("No se pudo adjuntar raw_text_snippet en modo guiado", exc_info=True)

            self._record_model_attempt(result, selected_model)

            try:
                if (
                    (self.policy_context or "") == "HDI_EN_MI_CASA"
                    and document_type == "informe_final_del_ajustador"
                    and (result.extracted_fields.get("tipo_siniestro") in (None, "", "N/A"))
                    and isinstance(content, dict)
                ):
                    text = (content.get("text") or "")
                    inferred = self._fallback_tipo_siniestro(text)
                    if inferred:
                        result.extracted_fields["tipo_siniestro"] = inferred
                        result.extraction_metadata["fallback_tipo_siniestro"] = True
                        logger.info("TIPO_SINIESTRO_FALLBACK: '%s'", inferred)
            except Exception as exc:
                logger.warning("Fallback tipo_siniestro falló: %s", exc)

        final_result = result

        allowed_field_set = set(allowed_fields or []) if allowed_fields else None

        if final_result and use_cache:
            self.extraction_cache[cache_key] = final_result

        return final_result

    @staticmethod
    def _supports_custom_sampling(model_name: Optional[str]) -> bool:
        """Controla si se envían parámetros de sampling."""
        if not model_name:
            return True
        lowered = model_name.lower()
        if lowered.startswith("gpt-5"):
            return False
        return True

    @staticmethod
    def _stringify_message_content(message: Any) -> str:
        """Devuelve contenido textual utilizable desde un mensaje de OpenAI."""
        raw_content = (getattr(message, "content", None) or "").strip()
        if raw_content:
            return raw_content
        if hasattr(message, "parsed") and message.parsed is not None:
            try:
                return json.dumps(message.parsed, ensure_ascii=False)
            except Exception:
                return ""
        return ""

    def _parse_json_payload(
        self,
        content: str,
        *,
        document_name: str,
        model_name: str,
        route: Optional[str],
    ) -> Dict[str, Any]:
        """Intenta parsear JSON robustamente y reporta fallos con contexto útil."""
        cleaned = (content or "").strip()
        if not cleaned:
            logger.error(
                "❌ Respuesta vacía del modelo %s para %s (ruta %s)",
                model_name,
                document_name,
                route or "ocr_text",
            )
            return {}

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            newline_pos = cleaned.find("\n")
            if newline_pos != -1:
                fence_header = cleaned[:newline_pos].lower()
                if fence_header.startswith("json"):
                    cleaned = cleaned[newline_pos + 1 :]
            brace_pos = cleaned.find("{")
            if brace_pos != -1:
                cleaned = cleaned[brace_pos:]
            cleaned = cleaned.rstrip("`")

        try:
            parsed = json.loads(cleaned)
        except Exception as exc:  # noqa: BLE001 - queremos detalles
            logger.error(
                "❌ No se pudo parsear JSON para %s: %s",
                document_name,
                exc,
            )
            logger.error("📄 Contenido (primeros 500 chars): %s", cleaned[:500])
            logger.error("📄 Contenido (últimos 200 chars): %s", cleaned[-200:])
            logger.error("🔧 Modelo: %s | Ruta: %s", model_name, route or "ocr_text")
            return {}

        if not isinstance(parsed, dict):
            logger.error(
                "❌ Respuesta del modelo %s no es un objeto JSON para %s (ruta %s)",
                model_name,
                document_name,
                route or "ocr_text",
            )
            return {}

        return parsed

    def _record_model_attempt(
        self,
        extraction: DocumentExtraction,
        model_name: str,
    ) -> None:
        """Mantiene trazabilidad de modelos utilizados o intentados."""
        if not extraction.extraction_metadata:
            extraction.extraction_metadata = {}
        attempted: List[str] = list(extraction.extraction_metadata.get("models_attempted") or [])
        if model_name and model_name not in attempted:
            attempted.append(model_name)
        extraction.extraction_metadata["models_attempted"] = attempted

    def _fallback_tipo_siniestro(self, text: str) -> Optional[str]:
        """
        Heurística ligera para extraer 'tipo_siniestro' del texto del Informe Final.
        Busca patrones comunes (tipo/naturaleza del siniestro) y frases "Daños a ... por ...".
        """
        if not text:
            return None
        import re as _re
        sample = text[:6000]  # primeras ~6000 chars para contexto
        lines = [ln.strip() for ln in sample.split('\n') if ln.strip()]

        # 1) Patrones explícitos
        patterns = [
            r"(?i)tipo\s+de\s+siniestro\s*[:\-]\s*(.+)",
            r"(?i)naturaleza\s+del\s+siniestro\s*[:\-]\s*(.+)",
        ]
        for pat in patterns:
            m = _re.search(pat, sample)
            if m:
                val = m.group(1).strip()
                # Limpiar hasta fin de línea
                val = val.split('\n')[0].strip()
                return val[:120]

        # 2) Frases "Daños a ... por ..."
        for ln in lines[:200]:  # primeras ~200 líneas
            if _re.search(r"(?i)daños?\s+a\s+.+\s+por\s+.+", ln):
                return ln[:160]

        # 3) Palabra clave seguida de causa
        for ln in lines[:200]:
            if _re.search(r"(?i)(variaci[oó]n|sobre\-?voltaje|inundaci[oó]n|incendio|robo|daños\s+el[eé]ctricos)", ln):
                return ln[:160]

        return None
    
    async def _extract_direct_ai(
        self,
        *,
        content: Union[bytes, Path],
        prompt: str,
        model: str,
        document_name: str,
        route: str,
    ) -> Dict[str, Any]:
        """
        Extracción usando visión directa con modelos multimodales GPT-5.
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
            # Llamada a la API con visión (usa helper con compatibilidad de temperature)
            messages = [
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
            ]
            params = self.config.get_openai_params("extraction", escalated=model.endswith("thinking"))
            supports_sampling = self._supports_custom_sampling(model)
            response = await self._chat_with_retry(
                model=model,
                messages=messages,
                max_tokens=params.get("max_completion_tokens", 2000),
                temperature=params.get("temperature") if supports_sampling else None,
                top_p=params.get("top_p") if supports_sampling else None,
                max_retries=int(params.get("max_retries", 3)),
                timeout=params.get("timeout"),
            )

            # Parsear respuesta manualmente
            message = response.choices[0].message if response and response.choices else None
            raw_content = self._stringify_message_content(message) if message else ""
            result = self._parse_json_payload(
                raw_content,
                document_name=document_name,
                model_name=model,
                route=route,
            )
            logger.info(f"Direct AI extracción exitosa")
            return result
            
        except Exception as e:
            logger.error(f"Error en Direct AI: {e}")
            return {}
    
    async def _extract_ocr_text(
        self,
        *,
        prompt: str,
        model: str,
        document_name: str,
        route: str,
        ocr_content: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Extracción usando OCR + texto con IA
        """
        logger.info(f"Iniciando extracción OCR + texto con modelo {model}")

        try:
            # Llamada a la API (usa helper con compatibilidad de temperature)
            messages = [{"role": "user", "content": prompt}]
            params = self.config.get_openai_params("extraction", escalated=model.endswith("thinking"))
            supports_sampling = self._supports_custom_sampling(model)
            response = await self._chat_with_retry(
                model=model,
                messages=messages,
                max_tokens=params.get("max_completion_tokens", 2000),
                temperature=params.get("temperature") if supports_sampling else None,
                top_p=params.get("top_p") if supports_sampling else None,
                max_retries=int(params.get("max_retries", 3)),
                timeout=params.get("timeout"),
            )

            # Parsear respuesta manualmente
            message = response.choices[0].message if response and response.choices else None
            raw_content = self._stringify_message_content(message) if message else ""
            result = self._parse_json_payload(
                raw_content,
                document_name=document_name,
                model_name=model,
                route=route,
            )
            logger.info(f"OCR + texto extracción exitosa")
            return result
            
        except Exception as e:
            logger.error(f"Error en OCR + texto: {e}")
            return {}
    
    def _apply_field_mask_dict(
        self,
        extraction: Dict[str, Any],
        allowed_fields: List[str],
    ) -> Dict[str, Any]:
        """
        Aplica máscara de seguridad a un diccionario manteniendo sólo campos permitidos.
        Además garantiza que los campos requeridos existan (con valor None cuando no se permitan).
        """
        if not isinstance(extraction, dict):
            return {field: None for field in self.config.REQUIRED_FIELDS}

        allowed_set = set(allowed_fields or [])
        required_set = set(self.config.REQUIRED_FIELDS)
        masked: Dict[str, Any] = {}

        # Conservar valores permitidos por la guía
        for field in allowed_set:
            masked[field] = extraction.get(field, None)

        # Asegurar presencia de campos requeridos con fallback a None cuando no están autorizados
        for field in required_set:
            if field in allowed_set:
                masked[field] = extraction.get(field, None)
            else:
                if field in extraction and extraction[field] is not None:
                    logger.warning("Campo '%s' bloqueado (no permitido)", field)
                masked.setdefault(field, None)

        # Registrar campos no autorizados presentes en la extracción original para diagnóstico
        for field in extraction.keys():
            if field not in allowed_set and field not in required_set:
                logger.debug("Descartando campo no permitido '%s' en extracción", field)

        return masked
    
    def _validate_and_transform_dict(
        self,
        extraction: Dict[str, Any],
        document_type: str
    ) -> Dict[str, Any]:
        """
        Aplica validaciones y transformaciones a un diccionario
        """
        # Si tenemos validador disponible, usarlo para cada campo (respeta contexto de póliza)
        validated = extraction.copy()
        try:
            for field, value in extraction.items():
                if value is None:
                    continue
                ok, new_value, err = self.validator.validate_field(field, value)
                if ok:
                    validated[field] = new_value
                else:
                    # Si no es válido, mantener original pero registrar error
                    logger.debug(f"Validación fallida {field}: {err}")
                    validated[field] = value
            return validated
        except Exception as e:
            logger.warning(f"Fallo validando con FieldValidator, usando reglas locales: {e}")
            # Fallback a reglas locales
            validated = extraction.copy()
            for field, value in extraction.items():
                if value is None:
                    continue
                rules = self.validation_rules.get(field, {})
                if 'transform' in rules and callable(rules['transform']):
                    try:
                        validated[field] = rules['transform'](value)
                    except Exception as ex:
                        logger.error(f"Error transformando {field}: {ex}")
                if rules.get('type') == 'date' and value:
                    validated[field] = self._normalize_date(value)
                if rules.get('type') == 'float' and value:
                    try:
                        validated[field] = float(str(value).replace(',', '').replace('$', ''))
                    except Exception:
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
    
    def _create_null_extraction(self, document_name: str, document_type: Optional[str]) -> DocumentExtraction:
        """
        Crea una extracción con todos los campos en null
        """
        null_fields = {field: None for field in self.config.REQUIRED_FIELDS}
        
        return DocumentExtraction(
            source_document=document_name,
            document_type=document_type or "otro",
            extracted_fields=null_fields,
            extraction_metadata={
                "route": "null",
                "guide_applied": True,
                "reason": "document_type_not_recognized",
                "timestamp": datetime.now().isoformat()
            }
        )
