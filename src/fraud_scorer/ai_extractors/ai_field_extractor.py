# src/fraud_scorer/ai_extractors/ai_field_extractor.py

"""
AIFieldExtractor: Extrae campos de documentos individuales usando IA
"""
import os  # ← Agregar este import
import re  # ← Agregar este import
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
from datetime import datetime

from openai import AsyncOpenAI
import instructor
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import ExtractionConfig
from .models.extraction_models import DocumentExtraction, ExtractedField
from .prompts.extraction_prompts import ExtractionPromptBuilder

logger = logging.getLogger(__name__)

class AIFieldExtractor:
    """
    Extractor de campos usando IA para documentos individuales
    Procesa documento por documento de forma eficiente
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el extractor con cliente de OpenAI
        
        Args:
            api_key: API key de OpenAI (opcional, usa env var por defecto)
        """
        self.client = instructor.patch(
            AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        )
        self.config = ExtractionConfig()
        self.prompt_builder = ExtractionPromptBuilder()
        self.extraction_cache = {}
        
        logger.info("AIFieldExtractor inicializado")
    
    async def extract_from_document(
        self, 
        ocr_result: Dict[str, Any],
        document_name: str,
        document_type: Optional[str] = None,
        use_cache: bool = True
    ) -> DocumentExtraction:
        """
        Extrae campos de un documento individual
        
        Args:
            ocr_result: Resultado del OCR (JSON de Azure)
            document_name: Nombre del archivo
            document_type: Tipo de documento (opcional)
            use_cache: Si usar cache de extracciones
            
        Returns:
            DocumentExtraction con los campos extraídos
        """
        # Check cache
        cache_key = self._generate_cache_key(document_name, ocr_result)
        if use_cache and cache_key in self.extraction_cache:
            logger.info(f"Usando cache para {document_name}")
            return self.extraction_cache[cache_key]
        
        # Detectar tipo de documento si no se proporciona
        if not document_type:
            document_type = self._detect_document_type(ocr_result, document_name)
        
        logger.info(f"Extrayendo campos de {document_name} (tipo: {document_type})")
        
        # Preparar el contenido para la IA
        prepared_content = self._prepare_ocr_content(ocr_result)
        
        # Construir el prompt
        prompt = self.prompt_builder.build_extraction_prompt(
            document_name=document_name,
            document_type=document_type,
            ocr_content=prepared_content,
            required_fields=self.config.REQUIRED_FIELDS
        )
        
        # Llamar a la IA con reintentos
        extraction = await self._call_ai_with_retry(
            prompt=prompt,
            document_name=document_name,
            document_type=document_type
        )
        
        # Validar y post-procesar
        extraction = self._post_process_extraction(extraction)
        
        # Guardar en cache
        if use_cache:
            self.extraction_cache[cache_key] = extraction
        
        # Log de métricas
        self._log_extraction_metrics(extraction)
        
        return extraction
    
    async def extract_from_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        parallel_limit: int = 3
    ) -> List[DocumentExtraction]:
        """
        Extrae campos de múltiples documentos en paralelo controlado
        
        Args:
            documents: Lista de documentos con OCR
            parallel_limit: Número máximo de extracciones paralelas
            
        Returns:
            Lista de DocumentExtraction
        """
        logger.info(f"Procesando batch de {len(documents)} documentos")
        
        # Crear semáforo para limitar paralelismo
        semaphore = asyncio.Semaphore(parallel_limit)
        
        async def extract_with_semaphore(doc):
            async with semaphore:
                return await self.extract_from_document(
                    ocr_result=doc['ocr_result'],
                    document_name=doc['filename'],
                    document_type=doc.get('document_type')
                )
        
        # Procesar en paralelo controlado
        tasks = [extract_with_semaphore(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar errores
        successful_extractions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error procesando {documents[i]['filename']}: {result}")
            else:
                successful_extractions.append(result)
        
        logger.info(f"Extracciones exitosas: {len(successful_extractions)}/{len(documents)}")
        return successful_extractions
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _call_ai_with_retry(
        self, 
        prompt: str,
        document_name: str,
        document_type: str
    ) -> DocumentExtraction:
        """
        Llama a la API de OpenAI con reintentos automáticos
        """
        try:
            # Usar instructor para obtener respuesta estructurada
            response = await self.client.chat.completions.create(
                model=self.config.get_model_for_task("extraction"),
                messages=[
                    {"role": "system", "content": "Eres un experto en extracción de datos de documentos de seguros."},
                    {"role": "user", "content": prompt}
                ],
                response_model=DocumentExtraction,
                temperature=self.config.OPENAI_CONFIG["temperature"],
                max_tokens=self.config.OPENAI_CONFIG["max_tokens"]
            )
            
            # Asegurar que tenemos los campos correctos
            response.source_document = document_name
            response.document_type = document_type
            
            return response
            
        except ValidationError as e:
            logger.error(f"Error de validación en {document_name}: {e}")
            # Retornar extracción vacía en caso de error
            return DocumentExtraction(
                source_document=document_name,
                document_type=document_type,
                extracted_fields={field: None for field in self.config.REQUIRED_FIELDS},
                extraction_metadata={"error": str(e)}
            )
        except Exception as e:
            logger.error(f"Error llamando a OpenAI para {document_name}: {e}")
            raise
    
    def _prepare_ocr_content(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepara y limpia el contenido del OCR para la IA
        """
        # Extraer solo las partes relevantes
        prepared = {
            "text": ocr_result.get("text", "")[:10000],  # Limitar texto
            "key_value_pairs": ocr_result.get("key_value_pairs", {}),
            "tables": self._simplify_tables(ocr_result.get("tables", [])),
        }
        
        # Si el texto es muy largo, intentar extraer secciones relevantes
        if len(prepared["text"]) > 5000:
            prepared["text"] = self._extract_relevant_sections(prepared["text"])
        
        return prepared
    
    def _simplify_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Simplifica las tablas para reducir tokens
        """
        simplified = []
        for table in tables[:5]:  # Máximo 5 tablas
            simplified.append({
                "headers": table.get("headers", []),
                "rows": table.get("data_rows", [])[:10]  # Máximo 10 filas
            })
        return simplified
    
    def _extract_relevant_sections(self, text: str) -> str:
        """
        Extrae secciones relevantes del texto basándose en palabras clave
        """
        keywords = [
            "póliza", "asegurado", "vigencia", "siniestro", 
            "monto", "fecha", "reclamación", "cobertura"
        ]
        
        lines = text.split('\n')
        relevant_lines = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                relevant_lines.append(line)
        
        # Si tenemos suficientes líneas relevantes, usar esas
        if len(relevant_lines) > 10:
            return '\n'.join(relevant_lines[:100])
        
        # Si no, retornar las primeras líneas
        return '\n'.join(lines[:100])
    
    def _detect_document_type(self, ocr_result: Dict, filename: str) -> str:
        """
        Detecta el tipo de documento basándose en el contenido y nombre
        """
        text_lower = ocr_result.get("text", "").lower()
        filename_lower = filename.lower()
        
        # Mapeo de palabras clave a tipos
        type_indicators = {
            "poliza": ["póliza", "cobertura", "vigencia", "prima", "asegurado"],
            "factura": ["factura", "cfdi", "subtotal", "iva", "total"],
            "denuncia": ["denuncia", "ministerio", "querella", "declaración"],
            "peritaje": ["peritaje", "dictamen", "evaluación", "daños"],
            "carta_porte": ["carta porte", "transportista", "remitente", "destinatario"]
        }
        
        scores = {}
        for doc_type, keywords in type_indicators.items():
            score = sum(1 for kw in keywords if kw in text_lower or kw in filename_lower)
            if score > 0:
                scores[doc_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "otro"
    
    def _post_process_extraction(self, extraction: DocumentExtraction) -> DocumentExtraction:
        """
        Post-procesa la extracción para limpiar y validar datos
        """
        fields = extraction.extracted_fields
        
        # Limpiar campos vacíos
        for field in fields:
            if fields[field] in ["", "null", "None", "N/A", "NO ESPECIFICADO"]:
                fields[field] = None
        
        # Formatear fechas
        date_fields = ['fecha_ocurrencia', 'fecha_reclamacion', 'vigencia_inicio', 'vigencia_fin']
        for field in date_fields:
            if fields.get(field):
                fields[field] = self._format_date(fields[field])
        
        # Formatear montos
        if fields.get('monto_reclamacion'):
            fields['monto_reclamacion'] = self._format_amount(fields['monto_reclamacion'])
        
        extraction.extracted_fields = fields
        return extraction
    
    def _format_date(self, date_str: Any) -> Optional[str]:
        """Formatea una fecha a YYYY-MM-DD"""
        if not date_str:
            return None
        
        # Aquí implementarías la lógica de formateo de fechas
        # Por ahora retornamos como string
        return str(date_str)
    
    def _format_amount(self, amount: Any) -> Optional[float]:
        """Formatea un monto a float"""
        if not amount:
            return None
        
        if isinstance(amount, (int, float)):
            return float(amount)
        
        # Si es string, limpiar y convertir
        if isinstance(amount, str):
            clean = re.sub(r'[^\d.,]', '', amount)
            clean = clean.replace(',', '')
            try:
                return float(clean)
            except:
                return None
        
        return None
    
    def _generate_cache_key(self, document_name: str, ocr_result: Dict) -> str:
        """Genera una clave única para el cache"""
        # Usar hash del contenido + nombre
        content_str = json.dumps(ocr_result.get("text", "")[:1000], sort_keys=True)
        import hashlib
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        return f"{document_name}_{content_hash}"
    
    def _log_extraction_metrics(self, extraction: DocumentExtraction):
        """Registra métricas de la extracción"""
        fields_extracted = sum(
            1 for v in extraction.extracted_fields.values() 
            if v is not None
        )
        total_fields = len(extraction.extracted_fields)
        
        logger.info(
            f"Documento: {extraction.source_document} | "
            f"Campos extraídos: {fields_extracted}/{total_fields} | "
            f"Tipo: {extraction.document_type}"
        )