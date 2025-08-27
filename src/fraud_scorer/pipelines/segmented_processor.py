# src/fraud_scorer/pipelines/segmented_processor.py

"""
Pipeline segmentado con aislamiento por documento + consolidación inteligente.
Versión FINAL con todas las correcciones aplicadas.
"""

from __future__ import annotations
from fraud_scorer.processors.ai.ai_field_extractor import AIFieldExtractor

import json
import hashlib
import gc
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# IMPORTANTE: traemos tus clases del template processor
from fraud_scorer.templates.ai_report_generator import (
    AIReportGenerator,
)

logger = logging.getLogger(__name__)

# =====================================================
# CARGA DE CONFIGURACIÓN (YAML opcional)
# =====================================================
import os
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None  # si no está, usamos defaults

DEFAULT_CONFIG = {
    "pipeline": {
        "isolation_level": "strict",
        "batch_size": 3,
        "parallel_workers": 2,
        "memory_limits": {
            "factura": 50,
            "poliza": 30,
            "denuncia": 100,
            "bitacora_gps": 200,
            "default": 50,
        },
    },
    "consolidation": {
        "strategy": "priority",
        "source_priority": {
            "poliza": 1,
            "poliza_seguro": 1,
            "denuncia": 2,
            "identificacion": 2,
            "carta_porte": 3,
            "factura": 4,
            "factura_compra": 4,
            "bitacora_gps": 5,
            "peritaje": 6,
            "otro": 99
        },
        "field_priorities": {
            "numero_poliza": ["poliza", "poliza_seguro", "carta_reclamacion"],
            "nombre_asegurado": ["poliza", "poliza_seguro", "denuncia"],
            "vigencia_inicio": ["poliza", "poliza_seguro"],
            "vigencia_fin": ["poliza", "poliza_seguro"],
            "domicilio_poliza": ["poliza", "poliza_seguro"],
            "rfc": ["factura", "factura_compra"],
            "total": ["factura", "factura_compra"],
            "fecha_siniestro": ["denuncia", "bitacora_gps"],
            "lugar_hechos": ["denuncia", "bitacora_gps"],
        },
    },
}

def _deep_update(base: dict, patch: dict) -> dict:
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_pipeline_config(path: str | None = None) -> dict:
    """
    Carga config del pipeline desde YAML.
    Precedencia:
      1) path explícito
      2) ENV FRAUD_PIPELINE_CONFIG
      3) DEFAULT_CONFIG
    """
    cfg = dict(DEFAULT_CONFIG)
    candidate = path or os.getenv("FRAUD_PIPELINE_CONFIG")
    if candidate and yaml:
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            _deep_update(cfg, y)
            logger.info(f"Config pipeline cargada: {candidate}")
        except Exception as e:
            logger.warning(f"No se pudo leer config {candidate}: {e}. Usando defaults.")
    elif candidate and not yaml:
        logger.warning("PyYAML no instalado; ignoro FRAUD_PIPELINE_CONFIG. Usando defaults.")
    return cfg


# =====================================================
# CAPA 1: PROCESAMIENTO INDIVIDUAL AISLADO
# =====================================================

class DocumentIsolationLevel(Enum):
    STRICT = "strict"      # Cada documento en proceso separado
    MODERATE = "moderate"  # Batches de 3-5 documentos
    RELAXED = "relaxed"    # Batches de 10 documentos

@dataclass
class IsolatedDocumentContext:
    document_id: str
    document_hash: str
    document_type: str
    source_file: str
    processing_timestamp: datetime

    # Datos aislados
    raw_data: Dict[str, Any] = field(default_factory=dict)
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, bool] = field(default_factory=dict)

    # Control de memoria
    memory_usage_mb: float = 0.0
    processing_time_ms: float = 0.0

    def __post_init__(self):
        if not self.document_hash:
            content = json.dumps(self.raw_data, sort_keys=True, ensure_ascii=False, default=str)
            self.document_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def to_cache(self, cache_dir: Path) -> Path:
        cache_file = cache_dir / f"{self.document_id}_{self.document_hash}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2, default=str)
        return cache_file

    @classmethod
    def from_cache(cls, cache_file: Path) -> "IsolatedDocumentContext":
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convertir la fecha string de vuelta a datetime
        if 'processing_timestamp' in data and isinstance(data['processing_timestamp'], str):
            data['processing_timestamp'] = datetime.fromisoformat(data['processing_timestamp'])
        return cls(**data)


class IsolatedDocumentProcessor:
    """
    Procesa documentos de forma aislada usando IntelligentFieldExtractor
    """
    def __init__(
        self,
        isolation_level: DocumentIsolationLevel = DocumentIsolationLevel.STRICT,
        cache_dir: Optional[Path] = None,
        config: Optional[dict] = None,
    ):
        self.isolation_level = isolation_level
        self.cache_dir = cache_dir or Path("data/temp/isolated_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 🔧 CORRECCIÓN 1: Inicializar IntelligentFieldExtractor
        self.intelligent_extractor = AIReportGenerator()
        logger.info("✓ IntelligentFieldExtractor inicializado")

        cfg = (config or {}).get("pipeline", {})
        self.memory_limits = cfg.get("memory_limits") or DEFAULT_CONFIG["pipeline"]["memory_limits"]
        self.config = config or {}

    def process_document_isolated(
        self,
        document: Dict[str, Any],
        document_index: int
    ) -> IsolatedDocumentContext:
        """Procesa un documento de forma aislada"""
        
        # Medición inicial de memoria
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
        except Exception:
            process = None
            mem_before = 0.0
        
        start_time = datetime.now()

        try:
            ctx = self._create_isolated_context(document, document_index)
            ctx.extracted_fields = self._extract_fields_isolated(ctx)
            
            # 🔧 CORRECCIÓN 2: No llamar a _validate_fields_isolated
            ctx.validation_results = {}  # Simplemente asignar dict vacío
            
            ctx = self._cleanup_context(ctx)

            # Medición final
            try:
                if process is not None:
                    mem_after = process.memory_info().rss / 1024 / 1024
                else:
                    mem_after = mem_before
            except Exception:
                mem_after = mem_before
            
            ctx.memory_usage_mb = max(0.0, mem_after - mem_before)
            ctx.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            ctx.to_cache(self.cache_dir)
            
            if self.isolation_level == DocumentIsolationLevel.STRICT:
                gc.collect()
            
            return ctx

        except Exception as e:
            logger.error(f"Error procesando documento {document_index}: {e}", exc_info=True)
            return IsolatedDocumentContext(
                document_id=f"doc_{document_index:03d}",
                document_hash="error",
                document_type="error",
                source_file="unknown",
                processing_timestamp=datetime.now(),
            )

    def _extract_fields_isolated(self, context: IsolatedDocumentContext) -> Dict[str, Any]:
        """USA INTELLIGENTFIELDEXTRACTOR"""
        
        # Preparar documento para el extractor
        document = {
            'raw_text': (context.raw_data.get('text', '') or '')[:50000],
            'key_value_pairs': context.raw_data.get('key_value_pairs', {}) or {},
            'specific_fields': context.raw_data.get('specific_fields', {}) or {},
            'tables': context.raw_data.get('tables', []) or [],
            'document_type': context.document_type
        }
        
        # Mapeo de campos por tipo de documento
        fields_mapping = {
            'poliza': ['numero_poliza', 'nombre_asegurado', 'rfc', 'vigencia_inicio', 'vigencia_fin', 'domicilio_poliza'],
            'poliza_seguro': ['numero_poliza', 'nombre_asegurado', 'rfc', 'vigencia_inicio', 'vigencia_fin', 'domicilio_poliza'],
            'factura': ['rfc', 'fecha_siniestro', 'monto_reclamacion', 'total'],
            'factura_compra': ['rfc', 'fecha_siniestro', 'monto_reclamacion', 'total'],
            'denuncia': ['fecha_siniestro', 'lugar_hechos', 'tipo_siniestro'],
            'carta_porte': ['lugar_hechos', 'fecha_siniestro'],
            'bitacora_gps': ['lugar_hechos']
        }
        
        fields_to_extract = fields_mapping.get(
            context.document_type,
            ['numero_poliza', 'nombre_asegurado', 'monto_reclamacion']
        )
        
        # Extraer con el sistema inteligente
        extraction_results = self.intelligent_extractor.extract_all_fields(
            document=document,
            fields_to_extract=fields_to_extract,
            debug=False
        )
        
        # Convertir al formato esperado
        fields = {}
        for field_name, result in extraction_results.items():
            if result.value is not None:
                fields[field_name] = result.value
                
        if fields:
            logger.info(f"✓ {context.document_id}: {len(fields)} campos extraídos con IntelligentExtractor")
        else:
            logger.warning(f"⚠ {context.document_id}: Sin campos extraídos")
            
        return fields

    def _cleanup_context(self, ctx: IsolatedDocumentContext) -> IsolatedDocumentContext:
        """Limpia el contexto para reducir memoria"""
        if "text" in ctx.raw_data:
            ctx.raw_data["text"] = (ctx.raw_data["text"] or "")[:1000] + "..."
        if "entities" in ctx.raw_data:
            ctx.raw_data["entities"] = (ctx.raw_data["entities"] or [])[:10]
        return ctx

    def _create_isolated_context(self, document: Dict[str, Any], index: int) -> IsolatedDocumentContext:
        """Crea contexto aislado para el documento"""
        import copy
        doc_copy = copy.deepcopy(document)
        meta = doc_copy.get("ocr_metadata", {}) or {}
        source_file = meta.get("source_name") or meta.get("file_name") or f"document_{index}.pdf"

        return IsolatedDocumentContext(
            document_id=f"doc_{index:03d}",
            document_hash="",
            document_type=doc_copy.get("document_type", "unknown"),
            source_file=source_file,
            processing_timestamp=datetime.now(),
            raw_data={
                "text": (doc_copy.get("raw_text") or "")[:50000],
                "entities": (doc_copy.get("entities") or [])[:100],
                "key_value_pairs": doc_copy.get("key_value_pairs") or {},
                "specific_fields": doc_copy.get("specific_fields") or {},
                "tables": doc_copy.get("tables") or [],
            },
        )


# =====================================================
# CAPA 2: CONSOLIDACIÓN INTELIGENTE
# =====================================================

class ConsolidationStrategy(Enum):
    PRIORITY = "priority"
    VOTING = "voting"
    CONFIDENCE = "confidence"
    LATEST = "latest"


class IntelligentConsolidator:
    """
    🔧 CORRECCIÓN 3: Constructor simplificado
    """
    def __init__(self, config: Optional[dict] = None):
        """
        Inicializa el consolidador con configuración opcional
        """
        cfg = config or DEFAULT_CONFIG
        self.config = cfg.get("consolidation", {})
        self.SOURCE_PRIORITY = self.config.get("source_priority", DEFAULT_CONFIG["consolidation"]["source_priority"])
        self.field_priorities = self.config.get("field_priorities", DEFAULT_CONFIG["consolidation"]["field_priorities"])
        self.strategy = ConsolidationStrategy(self.config.get("strategy", "priority"))
        logger.info(f"Consolidador inicializado con estrategia: {self.strategy.value}")

    def consolidate(
        self,
        contexts: List[IsolatedDocumentContext],
        strategy: Optional[ConsolidationStrategy] = None
    ) -> Dict[str, Any]:
        """Consolida múltiples contextos en un resultado único"""
        
        use_strategy = strategy or self.strategy
        logger.info(f"Consolidando {len(contexts)} contextos con estrategia {use_strategy.value}")
        
        if use_strategy == ConsolidationStrategy.PRIORITY:
            return self._consolidate_by_priority(contexts)
        elif use_strategy == ConsolidationStrategy.VOTING:
            return self._consolidate_by_voting(contexts)
        elif use_strategy == ConsolidationStrategy.CONFIDENCE:
            return self._consolidate_by_confidence(contexts)
        else:
            return self._consolidate_by_latest(contexts)

    def _consolidate_by_priority(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, Any]:
        """Consolida usando prioridad de tipos de documento"""
        
        # Agrupar datos por tipo y campo
        aggregated = defaultdict(lambda: defaultdict(list))
        
        for ctx in contexts:
            doc_type = ctx.document_type
            for field_name, value in ctx.extracted_fields.items():
                if value is not None and value != "":
                    aggregated[field_name][doc_type].append(value)
        
        # Consolidar cada campo según prioridad
        case_info = {}
        for field_name, sources in aggregated.items():
            # Obtener prioridades para este campo
            field_priority = self.field_priorities.get(field_name, list(self.SOURCE_PRIORITY.keys()))
            
            # Buscar el valor del tipo de documento con mayor prioridad
            for doc_type in field_priority:
                if doc_type in sources and sources[doc_type]:
                    # Tomar el primer valor disponible de este tipo
                    case_info[field_name] = sources[doc_type][0]
                    break
            
            # Si no se encontró en los prioritarios, usar prioridad general
            if field_name not in case_info:
                sorted_types = sorted(
                    sources.keys(),
                    key=lambda x: self.SOURCE_PRIORITY.get(x, 99)
                )
                if sorted_types and sources[sorted_types[0]]:
                    case_info[field_name] = sources[sorted_types[0]][0]
        
        # Construir resultado final
        result = {
            "case_info": case_info,
            "aggregated_data": dict(aggregated),
            "documents_summary": [
                {
                    "source": ctx.source_file,
                    "type": ctx.document_type,
                    "fields_extracted": len(ctx.extracted_fields),
                    "validations_passed": sum(1 for v in ctx.validation_results.values() if v),
                    "validations_total": len(ctx.validation_results)
                }
                for ctx in contexts
            ],
            "validation_summary": {
                "total": sum(len(ctx.validation_results) for ctx in contexts),
                "passed": sum(sum(1 for v in ctx.validation_results.values() if v) for ctx in contexts),
                "success_rate": 0.0,
                "failures_by_type": {}
            },
            "processing_stats": {
                "total_documents": len(contexts),
                "total_memory_mb": sum(ctx.memory_usage_mb for ctx in contexts),
                "avg_memory_mb": sum(ctx.memory_usage_mb for ctx in contexts) / len(contexts) if contexts else 0,
                "max_memory_mb": max((ctx.memory_usage_mb for ctx in contexts), default=0),
                "total_time_ms": sum(ctx.processing_time_ms for ctx in contexts),
                "avg_time_ms": sum(ctx.processing_time_ms for ctx in contexts) / len(contexts) if contexts else 0,
                "max_time_ms": max((ctx.processing_time_ms for ctx in contexts), default=0),
            }
        }
        
        # Calcular tasa de éxito
        if result["validation_summary"]["total"] > 0:
            result["validation_summary"]["success_rate"] = (
                result["validation_summary"]["passed"] / result["validation_summary"]["total"]
            )
        
        logger.info(f"Consolidación completada: {len(case_info)} campos únicos consolidados")
        return result

    def _consolidate_by_voting(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, Any]:
        """Consolida usando votación (mayoría)"""
        # Por implementar si es necesario
        return self._consolidate_by_priority(contexts)

    def _consolidate_by_confidence(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, Any]:
        """Consolida usando confianza de extracción"""
        # Por implementar si es necesario
        return self._consolidate_by_priority(contexts)

    def _consolidate_by_latest(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, Any]:
        """Consolida usando el documento más reciente"""
        # Por implementar si es necesario
        return self._consolidate_by_priority(contexts)


# =====================================================
# CAPA 3: ORQUESTADOR PRINCIPAL
# =====================================================

class SegmentedPipelineOrchestrator:
    """
    🔧 CORRECCIÓN 4: Inicialización correcta del consolidador
    """
    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_pipeline_config()
        pcfg = cfg.get("pipeline", {})
        
        self.batch_size = int(pcfg.get("batch_size", 3))
        self.parallel_workers = int(pcfg.get("parallel_workers", 2))
        
        # Inicializar procesador con configuración
        self.processor = IsolatedDocumentProcessor(
            isolation_level=DocumentIsolationLevel.STRICT,
            config=cfg
        )
        
        # 🔧 CORRECCIÓN: Inicializar consolidador con configuración
        self.consolidator = IntelligentConsolidator(config=cfg)
        
        self.cache_dir = Path("data/temp/pipeline_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {"documents_processed": 0, "batches_completed": 0, "errors": [], "warnings": []}

    async def process_documents(
        self,
        documents: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Procesa documentos de forma asíncrona"""
        
        logger.info(f"Procesamiento segmentado de {len(documents)} docs (batch={self.batch_size}, workers={self.parallel_workers})")
        
        all_ctx: List[IsolatedDocumentContext] = []
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(documents), self.batch_size):
            batch = documents[batch_idx:batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1
            logger.info(f"Batch {batch_num}/{total_batches}")
            
            batch_ctx = await self._process_batch_parallel(batch, batch_idx)
            all_ctx.extend(batch_ctx)
            
            if progress_callback:
                progress = (batch_num / total_batches) * 100.0
                progress_callback(progress, f"Batch {batch_num}/{total_batches}")
            
            self._cleanup_batch()
            self.metrics["batches_completed"] += 1
            self.metrics["documents_processed"] += len(batch)
        
        logger.info("Consolidando resultados…")
        consolidated = self.consolidator.consolidate(all_ctx, ConsolidationStrategy.PRIORITY)
        
        consolidated["pipeline_metrics"] = self.metrics
        self._save_consolidated_result(consolidated)
        
        return consolidated

    async def _process_batch_parallel(
        self,
        batch: List[Dict[str, Any]],
        start_index: int
    ) -> List[IsolatedDocumentContext]:
        """Procesa un batch en paralelo"""
        
        ctxs: List[IsolatedDocumentContext] = []
        
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = [
                executor.submit(self.processor.process_document_isolated, doc, start_index + i)
                for i, doc in enumerate(batch)
            ]
            
            for future in futures:
                try:
                    ctx = future.result(timeout=60)
                    ctxs.append(ctx)
                except Exception as e:
                    logger.error(f"Error en procesamiento paralelo: {e}", exc_info=True)
                    self.metrics["errors"].append(str(e))
                    # Crear contexto de error
                    ctxs.append(IsolatedDocumentContext(
                        document_id="error",
                        document_hash="error",
                        document_type="error",
                        source_file="error",
                        processing_timestamp=datetime.now(),
                    ))
        
        return ctxs

    def _cleanup_batch(self):
        """Limpia memoria después de cada batch"""
        gc.collect()
        self._cleanup_old_cache()

    def _cleanup_old_cache(self):
        """Limpia archivos de cache antiguos"""
        from datetime import timedelta
        now = datetime.now()
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                age = now - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if age > timedelta(hours=1):
                    cache_file.unlink()
            except Exception:
                pass

    def _save_consolidated_result(self, consolidated: Dict[str, Any]):
        """Guarda el resultado consolidado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.cache_dir / f"consolidated_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Resultado consolidado guardado en: {output_file}")


# =====================================================
# INTEGRACIÓN CON TEMPLATE PROCESSOR
# =====================================================

class SegmentedTemplateProcessor(AIReportGenerator):
    """Template processor que usa el pipeline segmentado"""

    def __init__(
        self,
        template_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        config: Optional[dict] = None
    ):
        super().__init__(template_dir)
        cfg = config or load_pipeline_config(config_path)
        self.orchestrator = SegmentedPipelineOrchestrator(config=cfg)

    async def extract_from_documents_async(
        self,
        documents: List[Dict[str, Any]],
        ai_analysis: Dict[str, Any]
    ) -> AIReportGenerator:
        """Extracción asíncrona desde documentos"""
        
        logger.info("Iniciando extracción con pipeline segmentado")
        consolidated = await self.orchestrator.process_documents(
            documents,
            progress_callback=self._log_progress
        )
        
        informe = self._build_informe_from_consolidated(consolidated, ai_analysis)
        return informe

    def extract_from_documents(
        self,
        documents: List[Dict[str, Any]],
        ai_analysis: Dict[str, Any]
    ) -> AIReportGenerator:
        """Extracción síncrona desde documentos"""
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.extract_from_documents_async(documents, ai_analysis)
            )
        finally:
            loop.close()

    def _build_informe_from_consolidated(
        self,
        consolidated: Dict[str, Any],
        ai: Dict[str, Any]
    ) -> AIReportGenerator:
        """Construye el informe desde datos consolidados"""
        
        case_info = consolidated.get("case_info", {})
        
        # Obtener el análisis de fraude desde AI
        fraud_analysis = ai.get("fraud_analysis") or ai.get("ai_analysis_raw") or {}
        
        # Normalización del nombre del asegurado
        nombre_asegurado = case_info.get("nombre_asegurado", "") or ""
        if nombre_asegurado and "S.A. DE" in nombre_asegurado and "C.V." not in nombre_asegurado:
            nombre_asegurado = nombre_asegurado.replace("S.A. DE", "S.A. DE C.V.")
        
        # Obtener case_id desde AI o generar uno
        numero_siniestro = ai.get("case_id", "CASE-2025-0001")
        
        informe = AIReportGenerator(
            numero_siniestro=numero_siniestro,
            nombre_asegurado=nombre_asegurado or "NO IDENTIFICADO",
            numero_poliza=case_info.get("numero_poliza") or "SIN PÓLIZA",
            vigencia_desde=self._format_date(case_info.get("vigencia_inicio", "")),
            vigencia_hasta=self._format_date(case_info.get("vigencia_fin", "")),
            domicilio_poliza=case_info.get("domicilio_poliza") or "NO ESPECIFICADO",
            bien_reclamado=self._extract_bien_from_aggregated(consolidated.get("aggregated_data", {})),
            monto_reclamacion=self._calculate_total_from_aggregated(consolidated.get("aggregated_data", {})),
            tipo_siniestro=self._determine_claim_type_from_aggregated(
                consolidated.get("aggregated_data", {}),
                ai
            ),
            fecha_ocurrencia=self._format_date(case_info.get("fecha_siniestro", "")),
            fecha_reclamacion=self._format_date(case_info.get("fecha_reclamacion", "")),
            lugar_hechos=case_info.get("lugar_hechos") or "NO ESPECIFICADO",
        )
        
        # Secciones analíticas
        informe.analisis_turno = self._generate_analisis_turno(ai)
        informe.planteamiento_problema = self._generate_planteamiento(ai)
        
        # Documentos analizados
        informe.documentos_analizados = []
        for summary in consolidated.get("documents_summary", []):
            descripcion = f"{summary.get('source', '')} · extraídos: {summary.get('fields_extracted', 0)}"
            if summary.get('validations_total', 0) > 0:
                descripcion += f" · validaciones ok: {summary.get('validations_passed', 0)}"
            
            informe.documentos_analizados.append(
                AIReportGenerator(
                    tipo_documento=str(summary.get("type", "")).upper().replace("_", " "),
                    descripcion=descripcion,
                    hallazgos=[],
                    nivel_alerta=AIReportGenerator.INFO.value,
                    imagen=None,
                    metadata=summary,
                )
            )
        
        # Inconsistencias desde AI
        informe.inconsistencias = self._build_inconsistencies(fraud_analysis)
        
        # Conclusiones
        fraud_score = float(fraud_analysis.get("fraud_score", 0.0))
        informe.consideraciones = self._generate_considerations_enhanced(informe, fraud_analysis)
        informe.conclusion_texto, informe.conclusion_veredicto, informe.conclusion_tipo = \
            self._generate_conclusion_enhanced(fraud_score, informe, fraud_analysis)
        
        if informe.conclusion_tipo == "tentativa":
            informe.soporte_legal = self.articulos_legales
        
        return informe

    def _log_progress(self, progress: float, message: str):
        """Log de progreso"""
        logger.info(f"Progreso: {progress:.1f}% - {message}")

    def _extract_bien_from_aggregated(self, aggregated: Dict[str, Any]) -> str:
        """Extrae el bien reclamado desde datos agregados"""
        for key in ["factura", "factura_compra"]:
            if key in aggregated and aggregated[key]:
                for values in aggregated[key].values():
                    if values and isinstance(values, list) and values[0]:
                        # Buscar algo que parezca una descripción
                        val = str(values[0])
                        if len(val) > 10 and not val.replace(".", "").replace(",", "").isdigit():
                            return val
        return "MERCANCÍA DIVERSA"

    def _calculate_total_from_aggregated(self, aggregated: Dict[str, Any]) -> str:
        """Calcula el total desde datos agregados"""
        total = 0.0
        
        # Buscar en campos de total o monto_reclamacion
        for field_name in ["total", "monto_reclamacion"]:
            if field_name in aggregated:
                for doc_type, values in aggregated[field_name].items():
                    if values and isinstance(values, list):
                        for val in values:
                            try:
                                # Limpiar y convertir
                                clean = str(val).replace(",", "").replace("$", "").replace("MXN", "").replace("MN", "").strip()
                                amount = float(clean)
                                if 0 < amount < 100_000_000:  # Evitar outliers
                                    total = max(total, amount)  # Tomar el mayor
                            except Exception:
                                continue
        
        return f"{total:,.2f}" if total > 0 else "0.00"

    def _determine_claim_type_from_aggregated(self, aggregated: Dict[str, Any], ai: Dict[str, Any]) -> str:
        """Determina el tipo de siniestro"""
        # Buscar en tipo_siniestro primero
        if "tipo_siniestro" in aggregated:
            for values in aggregated["tipo_siniestro"].values():
                if values and isinstance(values, list) and values[0]:
                    return str(values[0]).upper()
        
        # Heurística basada en contenido
        mapping = {
            "robo": "ROBO",
            "colision": "COLISIÓN",
            "colisión": "COLISIÓN",
            "incendio": "INCENDIO",
            "daño": "DAÑOS",
            "daños": "DAÑOS",
            "mercancia": "ROBO DE MERCANCÍA",
            "mercancía": "ROBO DE MERCANCÍA",
        }
        
        content = json.dumps(aggregated, ensure_ascii=False).lower()
        for keyword, claim_type in mapping.items():
            if keyword in content:
                return claim_type
        
        # Fallback a AI
        return ai.get("claim_type", "NO ESPECIFICADO")

    def _format_date(self, date_str: str) -> str:
        """Formatea fechas a ISO (YYYY-MM-DD)"""
        if not date_str:
            return ""
        
        # Si ya está en formato ISO
        if re.match(r"^\d{4}-\d{2}-\d{2}$", str(date_str).strip()):
            return str(date_str).strip()
        
        # Intentar parsear
        from datetime import datetime as dt
        formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
            "%d/%m/%y", "%Y/%m/%d", "%m/%d/%Y"
        ]
        
        s = str(date_str).strip()
        for fmt in formats:
            try:
                parsed = dt.strptime(s, fmt)
                return parsed.strftime("%Y-%m-%d")
            except Exception:
                continue
        
        return s