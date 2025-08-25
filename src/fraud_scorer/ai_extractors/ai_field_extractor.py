# src/fraud_scorer/ai_extractors/ai_field_extractor.py
"""
AIFieldExtractor: Extrae campos de documentos individuales usando IA
"""

from __future__ import annotations

import os  # ← agregado
import re  # ← agregado
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

from .config import ExtractionConfig
from .models.extraction_models import DocumentExtraction
from .prompts.extraction_prompts import ExtractionPromptBuilder

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
        # Instructor "parchea" el cliente para soportar response_model=...
        self.client = instructor.patch(raw_client)

        self.config = ExtractionConfig()
        self.prompt_builder = ExtractionPromptBuilder()
        self.extraction_cache: Dict[str, DocumentExtraction] = {}

        logger.info("AIFieldExtractor inicializado")

    async def extract_from_document(
        self,
        ocr_result: Dict[str, Any],
        document_name: str,
        document_type: Optional[str] = None,
        use_cache: bool = True,
    ) -> DocumentExtraction:
        """
        Extrae campos de un documento individual

        Args:
            ocr_result: Resultado del OCR/Parser (JSON unificado)
            document_name: Nombre del archivo
            document_type: Tipo de documento (opcional)
            use_cache: Si usar cache de extracciones

        Returns:
            DocumentExtraction con los campos extraídos
        """
        # Cache por contenido + nombre
        cache_key = self._generate_cache_key(document_name, ocr_result)
        if use_cache and cache_key in self.extraction_cache:
            logger.info(f"Usando cache de extracción para {document_name}")
            return self.extraction_cache[cache_key]

        # Detectar tipo si no viene
        if not document_type:
            document_type = self._detect_document_type(ocr_result, document_name)

        logger.info(f"Extrayendo campos de {document_name} (tipo: {document_type})")

        # Preparar contenido para la IA (token-friendly)
        prepared_content = self._prepare_ocr_content(ocr_result)

        # Construir prompt
        prompt = self.prompt_builder.build_extraction_prompt(
            document_name=document_name,
            document_type=document_type,
            ocr_content=prepared_content,
            required_fields=self.config.REQUIRED_FIELDS,
        )

        # Llamar IA con reintentos
        extraction = await self._call_ai_with_retry(
            prompt=prompt,
            document_name=document_name,
            document_type=document_type,
        )

        # Post-proceso (normalización y formatos)
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_ai_with_retry(
        self,
        prompt: str,
        document_name: str,
        document_type: str,
    ) -> DocumentExtraction:
        """
        Llama a la API de OpenAI con reintentos automáticos.
        Intenta siempre devolver un DocumentExtraction válido.
        """
        try:
            # Con instructor, esto devuelve directamente un `DocumentExtraction`
            response = await self.client.chat.completions.create(
                model=self.config.get_model_for_task("extraction"),
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un experto en extracción de datos de documentos de seguros. Responde estrictamente con el esquema solicitado.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_model=DocumentExtraction,  # ← validación pydantic
                temperature=self.config.OPENAI_CONFIG["temperature"],
                max_tokens=self.config.OPENAI_CONFIG["max_tokens"],
            )

            # Aseguramos attrs mínimos
            response.source_document = document_name
            response.document_type = document_type or "otro"
            # Aseguramos que exista extracted_fields (por seguridad)
            if not getattr(response, "extracted_fields", None):
                response.extracted_fields = {
                    field: None for field in self.config.REQUIRED_FIELDS
                }
            return response

        except ValidationError as e:
            # El LLM no cumplió el esquema → devolvemos objeto vacío seguro (NO reintenta)
            logger.error(f"Error de validación en {document_name}: {e}")
            return DocumentExtraction(
                source_document=document_name,
                document_type=document_type or "otro",
                extracted_fields={field: None for field in self.config.REQUIRED_FIELDS},
                extraction_metadata={"error": "validation_error", "details": str(e)},
            )

        except Exception as e:
            # Cualquier otro error → se reintenta por tenacity
            logger.error(f"Error llamando a OpenAI para {document_name}: {e}")
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

        # Fechas conocidas (ajusta a tus nombres reales)
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
        # Aquí puedes usar dateparser o reglas específicas.
        # Por ahora, devolvemos string plano:
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

    def _generate_cache_key(self, document_name: str, ocr_result: Dict[str, Any]) -> str:
        """Genera una clave única para el cache a partir de nombre+texto."""
        text_snip = (ocr_result.get("text") or "")[:1000]
        try:
            payload = json.dumps({"name": document_name, "text": text_snip}, ensure_ascii=False)
        except Exception:
            payload = f"{document_name}:{text_snip}"
        import hashlib

        return hashlib.md5(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _log_extraction_metrics(self, extraction: DocumentExtraction) -> None:
        """Registra métricas de la extracción."""
        total = len(extraction.extracted_fields or {})
        filled = sum(1 for v in (extraction.extracted_fields or {}).values() if v is not None)
        logger.info(
            f"Documento: {extraction.source_document} | Campos extraídos: {filled}/{total} | Tipo: {extraction.document_type}"
        )
