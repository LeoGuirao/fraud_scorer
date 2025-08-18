# src/fraud_scorer/pipelines/segmented_processor.py

"""
Pipeline segmentado con aislamiento por documento + consolidación inteligente.
- Lee configuración externa opcional (YAML) vía FRAUD_PIPELINE_CONFIG o parámetro.
- Reduce tamaño de contexto y evita contaminación entre documentos.
"""

from __future__ import annotations
from fraud_scorer.extractors.intelligent_extractor import IntelligentFieldExtractor

import json
import hashlib
import gc
import re
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# IMPORTANTE: traemos tus clases del template processor
from fraud_scorer.templates.template_processor import (
    TemplateProcessor,
    InformeSiniestro,
    DocumentoAnalizado,
    NivelAlerta,
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
        "field_priorities": {
            # Estos defaults serán sobreescritos abajo por los corregidos,
            # pero se mantienen por compatibilidad si no hay YAML.
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
    Si PyYAML no está disponible o hay error, usa defaults.
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
        return cls(**data)

class IsolatedDocumentProcessor:
    def __init__(
        self,
        isolation_level: DocumentIsolationLevel = DocumentIsolationLevel.STRICT,
        cache_dir: Optional[Path] = None,
        config: Optional[dict] = None,
    ):
        self.isolation_level = isolation_level
        self.cache_dir = cache_dir or Path("data/temp/isolated_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # conservar config para el flag use_intelligent_extractor
        self.config = config or {}

        cfg = (self.config or {}).get("pipeline", {})
        self.memory_limits = cfg.get("memory_limits") or {
            "factura": 50, "poliza": 30, "denuncia": 100, "bitacora_gps": 200, "default": 50
        }

        # AGREGAR ESTA LÍNEA
        self.intelligent_extractor = IntelligentFieldExtractor()
        logger.info("✓ IntelligentFieldExtractor inicializado")

    def process_document_isolated(
        self,
        document: Dict[str, Any],
        document_index: int
    ) -> IsolatedDocumentContext:
        """Procesa un documento de forma aislada"""
        
        # Medición inicial de memoria
        try:
            import psutil, os
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
        except Exception:
            process = None
            mem_before = 0.0
        
        start_time = datetime.now()

        try:
            ctx = self._create_isolated_context(document, document_index)
            ctx.extracted_fields = self._extract_fields_isolated(ctx)
            
            # 🔧 FIX: Simplemente asignar un diccionario vacío si no existe el método
            ctx.validation_results = {}  # ← ESTE ES EL FIX
            
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

            # IMPORTANTE: Guardar en cache SOLO si hay campos extraídos
            if ctx.extracted_fields:  # ← AGREGAR ESTA VALIDACIÓN
                ctx.to_cache(self.cache_dir)
            
            if self.isolation_level == DocumentIsolationLevel.STRICT:
                gc.collect()
            
            return ctx

        except Exception as e:
            logger.error(f"Error procesando documento {document_index}: {e}", exc_info=True)
            # En caso de error, retornar contexto con campos vacíos pero válido
            return IsolatedDocumentContext(
                document_id=f"doc_{document_index:03d}",
                document_hash="",
                document_type=document.get("document_type", "unknown"),
                source_file=document.get("ocr_metadata", {}).get("source_name", "unknown"),
                processing_timestamp=datetime.now(),
                extracted_fields={},  # ← Asegurar que tiene campos vacíos
                validation_results={}  # ← Asegurar que tiene validación vacía
            )

    def _create_isolated_context(self, document: Dict[str, Any], index: int) -> IsolatedDocumentContext:
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
                "text": (doc_copy.get("raw_text") or "")[:10000],
                "entities": (doc_copy.get("entities") or [])[:100],
                "key_value_pairs": doc_copy.get("key_value_pairs") or {},
                "specific_fields": doc_copy.get("specific_fields") or {},
            },
        )

    def _extract_fields_isolated(self, context: IsolatedDocumentContext) -> Dict[str, Any]:
        """REEMPLAZAR COMPLETAMENTE este método"""
        
        # Verificar si usar el extractor inteligente
        use_intelligent = (self.config.get('pipeline', {}) or {}).get('use_intelligent_extractor', True)
        
        if not use_intelligent:
            # Usar el método antiguo si está deshabilitado
            return self._extract_fields_isolated_legacy(context)
        
        # USAR EL EXTRACTOR INTELIGENTE
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
        fields: Dict[str, Any] = {}
        for field_name, result in extraction_results.items():
            if result.value is not None:
                fields[field_name] = result.value
                
        if fields:
            logger.info(f"✓ {context.document_id}: {len(fields)} campos extraídos con IntelligentExtractor")
        else:
            logger.warning(f"⚠ {context.document_id}: Sin campos extraídos")
            
        return fields
    
    def _extract_fields_isolated_legacy(self, context: IsolatedDocumentContext) -> Dict[str, Any]:
        """Método antiguo como respaldo (tu código actual)"""
        extractors: Dict[str, Callable[[IsolatedDocumentContext], Dict[str, Any]]] = {
            "factura": self._extract_factura_isolated,
            "factura_compra": self._extract_factura_isolated,
            "poliza": self._extract_poliza_isolated,
            "poliza_seguro": self._extract_poliza_isolated,
            "denuncia": self._extract_denuncia_isolated,
            "carta_porte": self._extract_carta_porte_isolated,
            "bitacora_gps": self._extract_gps_isolated,
        }
        extractor = extractors.get(context.document_type, self._extract_generic_isolated)
        return extractor(context)

    # ---- extractores concretos (legacy) y utilidades debajo (sin cambios) ----

# =====================================================
# CAPA 2: CONSOLIDACIÓN INTELIGENTE
# =====================================================

class ConsolidationStrategy(Enum):
    PRIORITY = "priority"
    VOTING = "voting"
    CONFIDENCE = "confidence"
    LATEST = "latest"

class IntelligentConsolidator:
    def __init__(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.PRIORITY,
        field_priorities: Optional[dict] = None,
        config: Optional[dict] = None,
    ):
        if config:
            scfg = (config.get("consolidation") or {})
            strategy = ConsolidationStrategy(scfg.get("strategy", strategy.value))
            # Permite override desde YAML
            field_priorities = scfg.get("field_priorities") or field_priorities

        self.strategy = strategy

        # CORRECCIÓN 1: field_priorities con nombres correctos
        self.field_priorities = field_priorities or {
            "numero_poliza": ["poliza", "poliza_seguro", "carta_reclamacion"],
            "nombre_asegurado": ["poliza", "poliza_seguro", "denuncia"],
            "vigencia_inicio": ["poliza", "poliza_seguro"],    # antes: vigencia_desde
            "vigencia_fin": ["poliza", "poliza_seguro"],       # antes: vigencia_hasta
            "domicilio_poliza": ["poliza", "poliza_seguro"],   # agregado
            "rfc": ["factura", "factura_compra"],
            "total": ["factura", "factura_compra"],            # antes: monto_total
            "fecha_siniestro": ["denuncia", "bitacora_gps"],
            "lugar_hechos": ["denuncia", "bitacora_gps"],
        }

    def consolidate_contexts(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, Any]:
        logger.info(f"Consolidando {len(contexts)} contextos con estrategia {self.strategy}")
        consolidated: Dict[str, Any] = {
            "case_info": {},
            "documents_summary": [],
            "aggregated_data": {},
            "validation_summary": {},
            "processing_stats": {}
        }
        consolidated["case_info"] = self._consolidate_case_info(contexts)
        consolidated["documents_summary"] = self._create_documents_summary(contexts)
        consolidated["aggregated_data"] = self._aggregate_by_type(contexts)
        consolidated["validation_summary"] = self._consolidate_validations(contexts)
        consolidated["processing_stats"] = self._calculate_stats(contexts)
        return consolidated

    def _consolidate_case_info(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, Any]:
        if self.strategy == ConsolidationStrategy.PRIORITY:
            return self._consolidate_by_priority(contexts)
        elif self.strategy == ConsolidationStrategy.VOTING:
            return self._consolidate_by_voting(contexts)
        elif self.strategy == ConsolidationStrategy.CONFIDENCE:
            # placeholder: no tenemos confidences numéricas todavía
            return self._consolidate_by_priority(contexts)
        else:
            return self._consolidate_by_latest(contexts)

    def _consolidate_by_priority(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        by_type: Dict[str, List[IsolatedDocumentContext]] = defaultdict(list)
        for c in contexts:
            by_type[c.document_type].append(c)

        for field, prio in self.field_priorities.items():
            for t in prio:
                if t in by_type:
                    for c in by_type[t]:
                        v = c.extracted_fields.get(field)
                        if v:
                            result[field] = v
                            break
                    if field in result:
                        break
        return result

    def _consolidate_by_voting(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, Any]:
        from collections import Counter
        vals: Dict[str, List[str]] = defaultdict(list)
        for c in contexts:
            for k, v in c.extracted_fields.items():
                if v:
                    vals[k].append(str(v))
        out = {}
        for k, arr in vals.items():
            if arr:
                out[k] = Counter(arr).most_common(1)[0][0]
        return out

    def _consolidate_by_latest(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, Any]:
        # simple: usa el último que tenga el campo
        out: Dict[str, Any] = {}
        for c in sorted(contexts, key=lambda x: x.processing_timestamp):
            for k, v in c.extracted_fields.items():
                if v:
                    out[k] = v
        return out

    def _create_documents_summary(self, contexts: List[IsolatedDocumentContext]) -> List[Dict[str, Any]]:
        summary = []
        for c in contexts:
            summary.append({
                "document_id": c.document_id,
                "type": c.document_type,
                "source": c.source_file,
                "fields_extracted": len(c.extracted_fields),
                "validations_passed": sum(c.validation_results.values()),
                "memory_mb": round(c.memory_usage_mb, 2),
                "processing_ms": round(c.processing_time_ms, 2)
            })
        return summary

    def _aggregate_by_type(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, Any]:
        agg: Dict[str, Any] = defaultdict(lambda: {"count": 0, "documents": [], "common_fields": {}})
        for c in contexts:
            t = c.document_type
            agg[t]["count"] += 1
            agg[t]["documents"].append(c.source_file)
            for k, v in c.extracted_fields.items():
                agg[t]["common_fields"].setdefault(k, []).append(v)
        return dict(agg)

    def _consolidate_validations(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, Any]:
        total = 0
        passed = 0
        failed_by_type: Dict[str, List[str]] = defaultdict(list)
        for c in contexts:
            for name, ok in c.validation_results.items():
                total += 1
                if ok: passed += 1
                else: failed_by_type[c.document_type].append(name)
        return {
            "total": total,
            "passed": passed,
            "success_rate": (passed / total) if total else 0.0,
            "failures_by_type": dict(failed_by_type),
        }

    def _calculate_stats(self, contexts: List[IsolatedDocumentContext]) -> Dict[str, float]:
        if not contexts:
            return {}
        mem = [c.memory_usage_mb for c in contexts]
        tms = [c.processing_time_ms for c in contexts]
        return {
            "total_documents": len(contexts),
            "total_memory_mb": float(sum(mem)),
            "avg_memory_mb": float(sum(mem) / len(mem)) if mem else 0.0,
            "max_memory_mb": float(max(mem)) if mem else 0.0,
            "total_time_ms": float(sum(tms)),
            "avg_time_ms": float(sum(tms) / len(tms)) if tms else 0.0,
            "max_time_ms": float(max(tms)) if tms else 0.0,
        }

# =====================================================
# CAPA 3: ORQUESTADOR PRINCIPAL
# =====================================================

class SegmentedPipelineOrchestrator:
    def __init__(
        self,
        isolation_level: DocumentIsolationLevel = DocumentIsolationLevel.STRICT,
        consolidation_strategy: ConsolidationStrategy = ConsolidationStrategy.PRIORITY,
        batch_size: int = 3,
        parallel_workers: int = 2,
        config: Optional[dict] = None,
    ):
        cfg = config or {}
        pcfg = cfg.get("pipeline", {})
        isolation_str = (pcfg.get("isolation_level") or isolation_level.value).lower()
        isolation_level = DocumentIsolationLevel(isolation_str)
        batch_size = int(pcfg.get("batch_size", batch_size))
        parallel_workers = int(pcfg.get("parallel_workers", parallel_workers))

        self.processor = IsolatedDocumentProcessor(isolation_level, config=cfg)
        self.consolidator = IntelligentConsolidator(
            strategy=consolidation_strategy,
            config=cfg,
        )
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers
        self.cache_dir = Path("data/temp/pipeline_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {"documents_processed": 0, "batches_completed": 0, "errors": [], "warnings": []}

    async def process_documents(
        self,
        documents: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
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
                progress_callback((batch_num / total_batches) * 100.0, f"Batch {batch_num}/{total_batches}")

            self._cleanup_batch()
            self.metrics["batches_completed"] += 1
            self.metrics["documents_processed"] += len(batch)

        logger.info("Consolidando resultados…")
        consolidated = self.consolidator.consolidate_contexts(all_ctx)
        consolidated["pipeline_metrics"] = self.metrics
        self._save_consolidated_result(consolidated)
        return consolidated

    async def _process_batch_parallel(self, batch: List[Dict[str, Any]], start_index: int) -> List[IsolatedDocumentContext]:
        ctxs: List[IsolatedDocumentContext] = []
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as ex:
            futures = [ex.submit(self.processor.process_document_isolated, doc, start_index + i) for i, doc in enumerate(batch)]
            for fut in futures:
                try:
                    ctxs.append(fut.result(timeout=60))
                except Exception as e:
                    logger.error(f"Error en procesamiento paralelo: {e}", exc_info=True)
                    ctxs.append(IsolatedDocumentContext(
                        document_id="error",
                        document_hash="error",
                        document_type="error",
                        source_file="error",
                        processing_timestamp=datetime.now(),
                    ))
        return ctxs

    def _cleanup_batch(self):
        gc.collect()
        self._cleanup_old_cache()

    def _cleanup_old_cache(self):
        from datetime import timedelta
        now = datetime.now()
        for cf in self.cache_dir.glob("*.json"):
            age = now - datetime.fromtimestamp(cf.stat().st_mtime)
            if age > timedelta(hours=1):
                try: cf.unlink()
                except Exception: pass

    def _save_consolidated_result(self, consolidated: Dict[str, Any]):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.cache_dir / f"consolidated_{ts}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Resultado consolidado guardado en: {out}")

# =====================================================
# INTEGRACIÓN CON TEMPLATE PROCESSOR
# =====================================================

class SegmentedTemplateProcessor(TemplateProcessor):
    """Template processor que usa el pipeline segmentado (aislado) con config externa opcional."""

    def __init__(self, template_dir: Optional[str] = None, config_path: Optional[str] = None, config: Optional[dict] = None):
        super().__init__(template_dir)
        cfg = config or load_pipeline_config(config_path)
        self.orchestrator = SegmentedPipelineOrchestrator(config=cfg)

    async def extract_from_documents_async(
        self,
        documents: List[Dict[str, Any]],
        ai_analysis: Dict[str, Any]
    ) -> InformeSiniestro:
        logger.info("Iniciando extracción con pipeline segmentado")
        consolidated = await self.orchestrator.process_documents(documents, progress_callback=self._log_progress)
        informe = self._build_informe_from_consolidated(consolidated, ai_analysis)
        return informe

    def extract_from_documents(self, documents: List[Dict[str, Any]], ai_analysis: Dict[str, Any]) -> InformeSiniestro:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.extract_from_documents_async(documents, ai_analysis))
        finally:
            loop.close()

    # ------- helpers específicos de este TP -------
    # CORRECCIÓN 3: _build_informe_from_consolidated mejorado
    def _build_informe_from_consolidated(self, consolidated: Dict[str, Any], ai: Dict[str, Any]) -> InformeSiniestro:
        case_info = consolidated.get("case_info", {})

        # --- INICIO DE LA CORRECCIÓN ---
        # La IA devuelve resultados bajo "fraud_analysis" o "ai_analysis_raw".
        # Buscamos de forma robusta en ambos.
        fraud_analysis_data = ai.get("fraud_analysis") or ai.get("ai_analysis_raw") or {}
        # --- FIN DE LA CORRECCIÓN ---

        # Normalización del nombre del asegurado
        nombre_asegurado = case_info.get("nombre_asegurado", "") or ""
        if nombre_asegurado and "S.A. DE" in nombre_asegurado and "C.V." not in nombre_asegurado:
            nombre_asegurado = nombre_asegurado.replace("S.A. DE", "S.A. DE C.V.")

        informe = InformeSiniestro(
            numero_siniestro="",  # ⚠️ dejar vacío (ya no "1")
            nombre_asegurado=nombre_asegurado or "NO IDENTIFICADO",
            numero_poliza=case_info.get("numero_poliza") or "SIN PÓLIZA",
            vigencia_desde=self._format_date(case_info.get("vigencia_inicio", "")),
            vigencia_hasta=self._format_date(case_info.get("vigencia_fin", "")),
            domicilio_poliza=case_info.get("domicilio_poliza") or "NO ESPECIFICADO",
            bien_reclamado=self._extract_bien_from_aggregated(consolidated.get("aggregated_data", {})),
            monto_reclamacion=self._calculate_total_from_aggregated(consolidated.get("aggregated_data", {})),
            tipo_siniestro=self._determine_claim_type_from_aggregated(
                consolidated.get("aggregated_data", {}),  # usa aggregated_data
                ai  # si tu _determine_claim_type_from_aggregated requiere hints de IA, se conserva
            ),
            fecha_ocurrencia=self._format_date(case_info.get("fecha_siniestro", "")),
            fecha_reclamacion=self._format_date(case_info.get("fecha_reclamacion", "")),
            lugar_hechos=case_info.get("lugar_hechos") or "NO ESPECIFICADO",
        )

        # Secciones analíticas (si estas funciones consumen IA cruda, puedes adaptar internamente)
        informe.analisis_turno = self._generate_analisis_turno(ai)
        informe.planteamiento_problema = self._generate_planteamiento(ai)

        # Documentos analizados a partir del summary
        informe.documentos_analizados = []
        for s in consolidated.get("documents_summary", []):
            desc = f"{s.get('source','')} · extraídos:{s.get('fields_extracted',0)} · ok:{s.get('validations_passed',0)}"
            informe.documentos_analizados.append(
                DocumentoAnalizado(
                    tipo_documento=str(s.get("type","")).upper().replace("_"," "),
                    descripcion=desc,
                    hallazgos=[],
                    nivel_alerta=NivelAlerta.INFO.value,
                    imagen=None,
                    metadata=s,
                )
            )

        # Inconsistencias (ahora leyendo desde la estructura correcta)
        informe.inconsistencias = self._build_inconsistencies(fraud_analysis_data)

        # Conclusiones (también desde la estructura correcta)
        fraud_score = float(fraud_analysis_data.get("fraud_score", 0.0))
        informe.consideraciones = self._generate_considerations_enhanced(informe, fraud_analysis_data)
        informe.conclusion_texto, informe.conclusion_veredicto, informe.conclusion_tipo = \
            self._generate_conclusion_enhanced(fraud_score, informe, fraud_analysis_data)

        if informe.conclusion_tipo == "tentativa":
            informe.soporte_legal = self.articulos_legales

        return informe

    def _log_progress(self, progress: float, message: str):
        logger.info(f"Progreso: {progress:.1f}% - {message}")

    def _extract_bien_from_aggregated(self, aggregated: Dict[str, Any]) -> str:
        factura = aggregated.get("factura") or aggregated.get("factura_compra") or {}
        cf = factura.get("common_fields", {}) if factura else {}
        desc = cf.get("descripcion") or []
        return (desc[0] if isinstance(desc, list) and desc else "MERCANCÍA DIVERSA")

    # CORRECCIÓN 5: _calculate_total_from_aggregated más robusto
    def _calculate_total_from_aggregated(self, aggregated: Dict[str, Any]) -> str:
        total = 0.0
        for key in ["factura", "factura_compra"]:
            doc_data = aggregated.get(key, {}) or {}
            common_fields = doc_data.get("common_fields", {}) or {}
            totales = common_fields.get("total", []) or []
            if isinstance(totales, list):
                for t in totales:
                    try:
                        clean = (
                            str(t)
                            .replace(",", "")
                            .replace("$", "")
                            .replace("MXN", "")
                            .replace("MN", "")
                            .strip()
                        )
                        val = float(clean)
                        if 0 < val < 100_000_000:  # evitar outliers absurdos
                            total += val
                    except Exception:
                        continue
        return f"{total:,.2f}" if total > 0 else "0.00"

    def _determine_claim_type_from_aggregated(self, aggregated: Dict[str, Any], ai: Dict[str, Any]) -> str:
        # heurística rápida: mira textos/llaves y decide (o usa IA como último recurso)
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
        s = json.dumps(aggregated, ensure_ascii=False).lower()
        for k, v in mapping.items():
            if k in s:
                return v
        return ai.get("claim_type", "NO ESPECIFICADO")

    # CORRECCIÓN 4: agregar _format_date
    def _format_date(self, date_str: str) -> str:
        """Formatea fechas a ISO (YYYY-MM-DD) cuando es posible."""
        if not date_str:
            return ""
        if re.match(r"^\d{4}-\d{2}-\d{2}$", str(date_str).strip()):
            return str(date_str).strip()
        from datetime import datetime as _dt
        formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
            "%d/%m/%y", "%Y/%m/%d", "%m/%d/%Y"
        ]
        s = str(date_str).strip()
        for fmt in formats:
            try:
                dt = _dt.strptime(s, fmt)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                continue
        return s
