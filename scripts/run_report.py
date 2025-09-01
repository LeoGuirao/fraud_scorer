#!/usr/bin/env python3
"""
Fraud Scorer v2.0 - Sistema de análisis con IA (solo v2, sin legacy)
"""

import sys
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import json
import re
from datetime import datetime
import threading
import signal
import shutil

# Añadir la raíz del proyecto al path de Python
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fraud_scorer.run_report")

# ==== Componentes del sistema v2 ====
from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
from fraud_scorer.parsers.document_parser import DocumentParser
from fraud_scorer.storage.ocr_cache import OCRCacheManager
from fraud_scorer.storage.cases import create_case

from fraud_scorer.processors.ai.ai_field_extractor import AIFieldExtractor
from fraud_scorer.processors.ai.ai_consolidator import AIConsolidator
from fraud_scorer.templates.ai_report_generator import AIReportGenerator
from fraud_scorer.models.extraction import (
    DocumentExtraction,
    ConsolidatedExtraction,
    ProgressEvent,
)
import time
import os


class _ProgressEmitter:
    """Emisor de eventos de progreso a archivos JSONL para seguimiento en tiempo real"""
    
    def __init__(self, case_id: str):
        self.case_id = case_id
        self.start_time = time.time()
        self.stage_times = {}  # Rastrea tiempos para EWMA
        self.ewma_alpha = 0.25  # Factor de suavizado EWMA
        self.avg_stage_times = {}  # Promedios móviles
        
        # Configurar ruta de salida
        base = os.getenv("FS_DATA_DIR", "data")
        self.cache_dir = Path(base) / "temp" / "pipeline_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.cache_dir / f"{case_id}.status.jsonl"
        
    def emit(self, stage: str, status: str, doc_index: int = None, 
             doc_total: int = None, message: str = None):
        """Emite un evento de progreso"""
        current_time = time.time()
        elapsed_ms = int((current_time - self.start_time) * 1000)
        
        # Actualizar EWMA para timing de etapa
        if status == "done" and stage in self.stage_times:
            stage_duration = current_time - self.stage_times[stage]
            if stage in self.avg_stage_times:
                # Actualización EWMA
                self.avg_stage_times[stage] = (
                    self.ewma_alpha * stage_duration + 
                    (1 - self.ewma_alpha) * self.avg_stage_times[stage]
                )
            else:
                self.avg_stage_times[stage] = stage_duration
        elif status == "started":
            self.stage_times[stage] = current_time
            
        # Calcular ETA basado en etapas restantes
        eta_ms = self._calculate_eta(stage, status)
        
        event = ProgressEvent(
            timestamp=current_time,
            case_id=self.case_id,
            stage=stage,
            doc_index=doc_index,
            doc_total=doc_total,
            status=status,
            elapsed_ms=elapsed_ms,
            avg_stage_ms=int(self.avg_stage_times.get(stage, 0) * 1000) if stage in self.avg_stage_times else None,
            eta_ms=eta_ms,
            message=message
        )
        
        # Escribir a JSONL
        try:
            with open(self.status_file, "a") as f:
                f.write(json.dumps(event.model_dump()) + "\n")
        except Exception as e:
            logger.warning(f"No se pudo escribir evento de progreso: {e}")
            
    def _calculate_eta(self, current_stage: str, status: str) -> Optional[int]:
        """Calcula tiempo estimado restante basado en EWMA"""
        stages = ["upload", "ocr", "extract", "consolidate", "analyze", "report"]
        
        if status == "done":
            # Encontrar etapas restantes
            try:
                current_idx = stages.index(current_stage)
                remaining_stages = stages[current_idx + 1:]
            except ValueError:
                return None
        else:
            # Incluir etapa actual si no está terminada
            try:
                current_idx = stages.index(current_stage)
                remaining_stages = stages[current_idx:]
            except ValueError:
                return None
                
        # Sumar tiempos estimados para etapas restantes
        total_remaining = 0
        for stage in remaining_stages:
            if stage in self.avg_stage_times:
                total_remaining += self.avg_stage_times[stage]
            else:
                # Estimaciones por defecto en segundos
                defaults = {
                    "upload": 2, "ocr": 15, "extract": 10,
                    "consolidate": 5, "analyze": 8, "report": 5
                }
                total_remaining += defaults.get(stage, 5)
                
        return int(total_remaining * 1000)  # Convertir a ms


class FraudAnalysisSystemV2:
    """
    Sistema de análisis de fraude v2.0 con IA y Cache OCR (sin legacy).
    """

    def __init__(self):
        # OCR + Parser
        self.ocr_processor = AzureOCRProcessor()
        self.document_parser = DocumentParser(self.ocr_processor)

        # Cache OCR
        self.cache_manager = OCRCacheManager()

        # IA v2
        self.extractor = AIFieldExtractor()
        self.consolidator = AIConsolidator()
        template_path = project_root / "src" / "fraud_scorer" / "templates"
        self.report_generator = AIReportGenerator(template_dir=template_path)

        logger.info("Sistema v2.0 inicializado con componentes de IA")
        
        # Control de cancelación
        self._cancelled = False
        self._cancel_lock = threading.Lock()
        self._cleanup_paths = []
        self.cancellation_check = None
        self.progress_emitter = None  # Se inicializa por caso

    def cancel(self):
        """Señala que el proceso debe cancelarse"""
        with self._cancel_lock:
            self._cancelled = True
            logger.info("🛑 Proceso marcado para cancelación")
    
    def is_cancelled(self) -> bool:
        """Verifica si el proceso fue cancelado"""
        with self._cancel_lock:
            return self._cancelled
    
    def reset_cancellation(self):
        """Resetea el estado de cancelación"""
        with self._cancel_lock:
            self._cancelled = False
            self._cleanup_paths = []
    
    async def cleanup_on_cancel(self):
        """Limpia archivos temporales al cancelar"""
        logger.info("🧹 Limpiando archivos temporales...")
        for path in self._cleanup_paths:
            try:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                        logger.debug(f"  ✓ Eliminado directorio: {path}")
                    else:
                        path.unlink()
                        logger.debug(f"  ✓ Eliminado archivo: {path}")
            except Exception as e:
                logger.warning(f"  ✗ No se pudo eliminar {path}: {e}")
        self._cleanup_paths = []
    
    def _clean_previous_case_files(
        self,
        output_path: Path,
        case_id: str,
        insured_name: str,
        claim_number: str
    ) -> int:
        """
        Limpia archivos anteriores relacionados con un caso.
        Busca y elimina archivos HTML, PDF y JSON antiguos antes de generar nuevos.
        
        Returns:
            Número de archivos eliminados
        """
        files_cleaned = 0
        
        # Patrones de archivos a limpiar
        patterns_to_clean = [
            # Formato nuevo
            f"{insured_name}_{claim_number}_*.html",
            f"{insured_name}_{claim_number}_*.pdf",
            f"{insured_name}_{claim_number}_*.json",
            f"INF-{insured_name}-{claim_number}*.html",
            f"INF-{insured_name}-{claim_number}*.pdf",
            # Formato antiguo por case_id
            f"*{case_id}*.html",
            f"*{case_id}*.pdf",
            f"*CASE-2025-*.json",
            # Archivos de replay
            f"replay_{case_id}_*.json",
        ]
        
        for pattern in patterns_to_clean:
            for file_path in output_path.glob(pattern):
                try:
                    file_path.unlink()
                    logger.debug(f"  ✓ Eliminado: {file_path.name}")
                    files_cleaned += 1
                except Exception as e:
                    logger.warning(f"  ✗ No se pudo eliminar {file_path.name}: {e}")
        
        return files_cleaned

    async def process_case(
        self,
        folder_path: Path,
        output_path: Path,
        case_title: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        cancellation_check: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Procesa un caso completo con el flujo v2 (solo IA).
        """
        logger.info("=" * 60)
        logger.info(f"📁 Procesando caso: {folder_path.name}")
        logger.info("🤖 Modo: IA Avanzada v2.0")
        logger.info("=" * 60)

        # Buscar documentos soportados (ignorar archivos '._' de macOS)
        supported_extensions = {
            ".pdf", ".png", ".jpg", ".jpeg", ".tiff",
            ".docx", ".xlsx", ".csv"
        }
        documents = [
            p for p in folder_path.glob("*")
            if p.suffix.lower() in supported_extensions and not p.name.startswith("._")
        ]
        if not documents:
            raise RuntimeError("No se encontraron documentos para procesar")

        logger.info(f"✓ Encontrados {len(documents)} documentos")

        # Verificar si ya existe un caso para esta ruta o título
        from fraud_scorer.storage.cases import get_case_by_path, get_case_by_title
        
        # Primero intentar por ruta
        existing_case = get_case_by_path(str(folder_path))
        
        if existing_case:
            case_id = existing_case['case_id']
            logger.info(f"✓ Usando caso existente por ruta: {case_id}")
        else:
            # Si no hay caso por ruta, intentar por título
            title = case_title or folder_path.name
            existing_case = get_case_by_title(title)
            
            if existing_case:
                case_id = existing_case['case_id']
                logger.info(f"✓ Usando caso existente por título: {case_id}")
            else:
                # Solo crear nuevo caso si no existe ninguno
                case_id = create_case(
                    title=title,
                    base_path=str(folder_path)
                )
                logger.info(f"✓ Nuevo caso creado: {case_id}")
        
        logger.info(f"✓ Case ID final: {case_id}")
        
        # Guardar callbacks para usar en los métodos internos
        self.progress_callback = progress_callback
        self.cancellation_check = cancellation_check
        
        # Reset estado de cancelación
        self.reset_cancellation()
        
        # Registrar carpeta para limpieza en caso de cancelación
        if folder_path not in self._cleanup_paths:
            self._cleanup_paths.append(folder_path)

        # Inicializar emisor de progreso para este caso
        self.progress_emitter = _ProgressEmitter(case_id)

        # Ejecutar pipeline v2
        return await self._process_with_ai(documents, case_id, output_path)

    async def _process_with_ai(
        self,
        documents: List[Path],
        case_id: str,
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Procesamiento con el sistema de IA y cache.
        """
        # ============================================
        # FASE 1: OCR/Parsing
        # ============================================
        logger.info("\n📖 FASE 1: Procesamiento de Documentos")
        logger.info("-" * 40)
        
        # Notificar inicio de procesamiento
        if self.progress_callback:
            self.progress_callback("Iniciando procesamiento de documentos...", 5)
        
        # Emitir evento de inicio de OCR
        if self.progress_emitter:
            self.progress_emitter.emit("ocr", "started", message="Iniciando procesamiento de documentos")
        
        # Verificar cancelación
        if self.cancellation_check and await self.cancellation_check():
            await self.cleanup_on_cancel()
            raise asyncio.CancelledError("Proceso cancelado por el usuario")

        ocr_results: List[Dict[str, Any]] = []
        cache_files: List[str] = []

        # Paso 3.2: Verificar si todos los documentos ya están en cache
        all_cached = True
        cached_count = 0
        if self.cache_manager:
            for doc_path in documents:
                if self.cache_manager.has_cache(doc_path):
                    cached_count += 1
                else:
                    all_cached = False
            
            logger.info(f"📊 Estado del cache: {cached_count}/{len(documents)} documentos en cache")
            
            # Notificar estado del cache
            if self.progress_callback:
                cache_msg = f"Cache: {cached_count}/{len(documents)} documentos disponibles"
                self.progress_callback(cache_msg, 10)
            
            if all_cached:
                logger.info("✨ Todos los documentos ya están en el cache. Omitiendo fase de OCR.")
                logger.info("⚡ SALTO DIRECTO A ANÁLISIS IA - Optimización activada")
                logger.info("-" * 40)
                
                # Cargar todos los documentos desde el cache
                for idx, doc_path in enumerate(documents, 1):
                    # Verificar cancelación antes de cada documento
                    if self.cancellation_check and await self.cancellation_check():
                        await self.cleanup_on_cancel()
                        raise asyncio.CancelledError(f"Proceso cancelado en documento {idx}/{len(documents)}")
                    
                    logger.info(f"  ⚡ Cargando desde cache: {doc_path.name}")
                    
                    # Notificar progreso por documento
                    if self.progress_callback:
                        progress = 10 + (idx * 20 // len(documents))
                        self.progress_callback(
                            f"Cargando desde cache: {doc_path.name}",
                            progress,
                            doc_path.name
                        )
                    
                    # Emitir evento de progreso OCR
                    if self.progress_emitter:
                        self.progress_emitter.emit(
                            "ocr", "running", 
                            doc_index=idx, doc_total=len(documents),
                            message=f"Cargando desde cache: {doc_path.name}"
                        )
                    
                    ocr_result = self.cache_manager.get_cache(doc_path, case_id)
                    if ocr_result:
                        ocr_results.append({
                            "filename": doc_path.name,
                            "ocr_result": ocr_result,
                            "document_type": None,  # se detectará dentro del extractor
                        })
                        cache_files.append(str(doc_path))
                
                logger.info(f"✓ Carga desde cache completada: {len(ocr_results)}/{len(documents)} documentos")
            else:
                # Procesar documentos normalmente (algunos pueden estar en cache, otros no)
                for idx, doc_path in enumerate(documents, 1):
                    # Verificar cancelación antes de cada documento
                    if self.cancellation_check and await self.cancellation_check():
                        await self.cleanup_on_cancel()
                        raise asyncio.CancelledError(f"Proceso cancelado en documento {idx}/{len(documents)}")
                    
                    logger.info(f"  Procesando: {doc_path.name}")
                    
                    # Notificar progreso por documento
                    if self.progress_callback:
                        progress = 10 + (idx * 20 // len(documents))
                        self.progress_callback(
                            f"Procesando documento {idx}/{len(documents)}: {doc_path.name}",
                            progress,
                            doc_path.name
                        )
                    
                    # Emitir evento de progreso OCR
                    if self.progress_emitter:
                        self.progress_emitter.emit(
                            "ocr", "running",
                            doc_index=idx, doc_total=len(documents),
                            message=f"Procesando: {doc_path.name}"
                        )

                    # Usar cache si existe
                    if self.cache_manager.has_cache(doc_path):
                        logger.info(f"  ⚡ Usando cache para: {doc_path.name}")
                        ocr_result = self.cache_manager.get_cache(doc_path, case_id)
                        if ocr_result:
                            cache_files.append(str(doc_path))
                    else:
                        # OCR/Parser tolerante a fallos
                        logger.info(f"  🔄 Procesando con OCR/Parser: {doc_path.name}")
                        try:
                            ocr_result = self.document_parser.parse_document(doc_path)
                            if self.cache_manager and ocr_result:
                                self.cache_manager.save_cache(doc_path, ocr_result)
                        except Exception as e:
                            logger.error(f"  ❌ Error procesando {doc_path.name}: {e}", exc_info=True)
                            continue

                    if ocr_result:
                        ocr_results.append({
                            "filename": doc_path.name,
                            "ocr_result": ocr_result,
                            "document_type": None,  # se detectará dentro del extractor
                        })

                logger.info(f"✓ Procesamiento completado: {len(ocr_results)}/{len(documents)} exitosos")
        else:
            # Si no hay cache manager, procesar todos los documentos normalmente
            for doc_path in documents:
                logger.info(f"  Procesando: {doc_path.name}")
                logger.info(f"  🔄 Procesando con OCR/Parser: {doc_path.name}")
                try:
                    ocr_result = self.document_parser.parse_document(doc_path)
                except Exception as e:
                    logger.error(f"  ❌ Error procesando {doc_path.name}: {e}", exc_info=True)
                    continue

                if ocr_result:
                    ocr_results.append({
                        "filename": doc_path.name,
                        "ocr_result": ocr_result,
                        "document_type": None,  # se detectará dentro del extractor
                    })

            logger.info(f"✓ Procesamiento completado: {len(ocr_results)}/{len(documents)} exitosos")

        # Guardar índice del caso para replay
        if self.cache_manager:
            # Extraer información del título del caso
            # Obtener el nombre de la carpeta desde el primer documento
            folder_name = documents[0].parent.name if documents else "UNKNOWN"
            case_title = folder_name
            parts = case_title.split(' - ', 1)
            if len(parts) == 2:
                claim_number = parts[0].strip()
                insured_name = parts[1].strip()
            else:
                claim_number = ""
                insured_name = case_title
            
            case_data = {
                "case_id": case_id,
                "case_title": case_title,
                "insured_name": insured_name,
                "claim_number": claim_number,
                "total_documents": len(documents),
                "documents": [str(d) for d in documents],
                "cache_files": cache_files,
                "folder_path": str(documents[0].parent) if documents else "",
                "processed_at": datetime.now().isoformat(),
                "status": "processed"
            }
            self.cache_manager.save_case_index(case_id, case_data)
        
        # Emitir evento de finalización de OCR (si no se hizo antes)
        if self.progress_emitter and not all_cached:
            self.progress_emitter.emit("ocr", "done", message="Procesamiento de documentos completado")

        # ============================================
        # FASE 2: Extracción con IA
        # ============================================
        logger.info("\n🔍 FASE 2: Extracción de campos con IA")
        logger.info("-" * 40)
        
        # Verificar cancelación antes de fase 2
        if self.cancellation_check and await self.cancellation_check():
            await self.cleanup_on_cancel()
            raise asyncio.CancelledError("Proceso cancelado durante fase 2")
        
        # Notificar inicio de extracción
        if self.progress_callback:
            self.progress_callback("Extrayendo campos con inteligencia artificial...", 35)
        
        # Emitir evento de inicio de extracción
        if self.progress_emitter:
            self.progress_emitter.emit("extract", "started", message="Extrayendo campos con IA")

        extractions: List[DocumentExtraction] = await self.extractor.extract_from_documents_batch(
            documents=ocr_results,
            parallel_limit=3,
        )

        for extraction in extractions:
            fields_found = sum(1 for v in extraction.extracted_fields.values() if v is not None)
            logger.info(f"  ✓ {extraction.source_document}: {fields_found} campos extraídos")

        logger.info(f"✓ Extracción completada: {len(extractions)} documentos procesados")
        
        # Emitir evento de finalización de extracción
        if self.progress_emitter:
            self.progress_emitter.emit("extract", "done", message="Extracción completada")

        # ============================================
        # FASE 3: Consolidación con IA
        # ============================================
        logger.info("\n🧠 FASE 3: Consolidación inteligente")
        logger.info("-" * 40)
        
        # Verificar cancelación antes de fase 3
        if self.cancellation_check and await self.cancellation_check():
            await self.cleanup_on_cancel()
            raise asyncio.CancelledError("Proceso cancelado durante fase 3")
        
        # Notificar inicio de consolidación
        if self.progress_callback:
            self.progress_callback("Consolidando información con IA...", 55)
        
        # Emitir evento de inicio de consolidación
        if self.progress_emitter:
            self.progress_emitter.emit("consolidate", "started", message="Consolidando información")

        consolidated: ConsolidatedExtraction = await self.consolidator.consolidate_extractions(
            extractions=extractions,
            case_id=case_id,
            use_advanced_reasoning=True,
        )

        # Conteo robusto (Pydantic v2/dict)
        fields_obj = getattr(consolidated, "consolidated_fields", {}) or {}
        if hasattr(fields_obj, "model_dump"):
            fields_dict = fields_obj.model_dump()
        elif hasattr(fields_obj, "dict"):
            fields_dict = fields_obj.dict()
        else:
            fields_dict = dict(fields_obj)

        fields_filled = sum(1 for v in fields_dict.values() if v is not None)
        total_fields = len(fields_dict)
        logger.info(f"✓ Campos consolidados: {fields_filled}/{total_fields}")

        if consolidated.conflicts_resolved:
            logger.info(f"✓ Conflictos resueltos: {len(consolidated.conflicts_resolved)}")
            for conflict in consolidated.conflicts_resolved[:3]:
                logger.info(
                    f"  - {conflict.get('field', 'N/A')}: {str(conflict.get('reasoning', ''))[:80]}..."
                )
        
        # --- OBTENER DATOS PARA NOMBRAR ARCHIVOS ---
        # Extraemos los datos del objeto `consolidated`. 
        # Asegúrate de que los nombres de los campos coincidan con los de tu modelo `ConsolidatedFields`
        insured_name_from_data = fields_dict.get("nombre_asegurado", "Desconocido")
        claim_number_from_data = fields_dict.get("numero_siniestro", f"SINIESTRO_{case_id}")
        logger.info(f"✓ Datos para organización: {insured_name_from_data} - {claim_number_from_data}")
        
        # Actualizar el índice del caso con los datos reales extraídos
        case_index_path = self.cache_manager.index_dir / f"{case_id}.json"
        if case_index_path.exists():
            try:
                with open(case_index_path, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                # Actualizar con datos reales
                case_data["insured_name"] = insured_name_from_data
                case_data["claim_number"] = claim_number_from_data
                case_data["status"] = "processed"
                self.cache_manager.save_case_index(case_id, case_data)
                logger.info("✓ Índice del caso actualizado con datos reales")
            except Exception as e:
                logger.error(f"Error actualizando índice del caso: {e}")
        
        # Reorganizar cache con estructura [ASEGURADO] - [SINIESTRO]
        logger.info("📁 Reorganizando estructura de cache...")
        self.cache_manager.reorganize_cache_for_case(case_id, insured_name_from_data, claim_number_from_data)
        logger.info("✓ Cache reorganizado con nomenclatura consistente")
        
        # Emitir evento de finalización de consolidación
        if self.progress_emitter:
            self.progress_emitter.emit("consolidate", "done", message="Consolidación completada")

        # ============================================
        # FASE 4: Análisis de fraude (IA)
        # ============================================
        logger.info("\n🔎 FASE 4: Análisis de fraude")
        logger.info("-" * 40)
        
        # Verificar cancelación antes de fase 4
        if self.cancellation_check and await self.cancellation_check():
            await self.cleanup_on_cancel()
            raise asyncio.CancelledError("Proceso cancelado durante fase 4")
        
        # Notificar análisis de fraude
        if self.progress_callback:
            self.progress_callback("Analizando indicadores de fraude...", 70)
        
        # Emitir evento de inicio de análisis
        if self.progress_emitter:
            self.progress_emitter.emit("analyze", "started", message="Analizando indicadores de fraude")
        
        ai_analysis = await self._analyze_fraud(consolidated, extractions)
        fraud_score = ai_analysis.get("fraud_score", 0)
        risk_level = "BAJO" if fraud_score < 0.3 else ("MEDIO" if fraud_score < 0.6 else "ALTO")
        logger.info(f"✓ Fraud Score: {fraud_score:.2%}")
        logger.info(f"✓ Nivel de Riesgo: {risk_level}")
        
        # Emitir evento de finalización de análisis
        if self.progress_emitter:
            self.progress_emitter.emit("analyze", "done", message=f"Análisis completado - Riesgo: {risk_level}")

        # ============================================
        # FASE 5: Generación del reporte
        # ============================================
        logger.info("\n📝 FASE 5: Generación del reporte")
        logger.info("-" * 40)
        
        # Verificar cancelación antes de fase 5
        if self.cancellation_check and await self.cancellation_check():
            await self.cleanup_on_cancel()
            raise asyncio.CancelledError("Proceso cancelado durante fase 5")
        
        # Notificar generación de reporte
        if self.progress_callback:
            self.progress_callback("Generando reporte HTML y PDF...", 85)
        
        # Emitir evento de inicio de reporte
        if self.progress_emitter:
            self.progress_emitter.emit("report", "started", message="Generando reporte")

        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generar nombres de archivo con nomenclatura dinámica
        def sanitize_filename(name: str) -> str:
            if not name: 
                return "SIN_NOMBRE"
            return re.sub(r'[^a-zA-Z0-9_.-]+', '_', name).strip('_')
        
        s_insured = sanitize_filename(insured_name_from_data)
        s_claim = sanitize_filename(claim_number_from_data)
        
        # Limpieza completa de archivos anteriores para este caso
        logger.info("🧹 Limpiando archivos anteriores del caso...")
        old_files_cleaned = self._clean_previous_case_files(output_path, case_id, s_insured, s_claim)
        if old_files_cleaned > 0:
            logger.info(f"✓ Limpiados {old_files_cleaned} archivos anteriores")
        else:
            logger.info("✓ No se encontraron archivos anteriores para limpiar")
        
        # HTML - con nomenclatura dinámica
        html_filename = f"{s_insured}_{s_claim}_INFORME.html"
        html_path = output_path / html_filename
        
        # Ya no necesitamos eliminar individualmente porque la limpieza completa ya lo hizo
        if html_path.exists():
            logger.info(f"  ⚠️ Reemplazando archivo existente: {html_filename}")
            html_path.unlink()
        
        html_content = self.report_generator.generate_report(
            consolidated_data=consolidated,
            ai_analysis=ai_analysis,
            output_path=html_path,
            insured_name=insured_name_from_data,
            claim_number=claim_number_from_data
        )
        logger.info(f"✓ HTML generado: {html_path}")

        # PDF - con nomenclatura dinámica y reemplazo
        pdf_filename = f"{s_insured}_{s_claim}_INFORME.pdf"
        pdf_path = output_path / pdf_filename
        
        # Eliminar archivo existente si existe
        if pdf_path.exists():
            logger.info(f"  ⚠️ Reemplazando archivo existente: {pdf_filename}")
            pdf_path.unlink()
            
        if self.report_generator.generate_pdf(html_content, pdf_path):
            logger.info(f"✓ PDF generado: {pdf_path}")
        
        # Notificar finalización
        if self.progress_callback:
            self.progress_callback("Finalizando procesamiento...", 95)
        
        # Emitir evento de finalización de reporte
        if self.progress_emitter:
            self.progress_emitter.emit("report", "done", message="Reporte generado")

        # ============================================
        # FASE 6: Guardar resultados y Organizar archivos
        # ============================================
        logger.info("\n💾 FASE 6: Guardar resultados y Organizar archivos")
        logger.info("-" * 40)
        ocr_total = len(documents)
        ocr_success = len(ocr_results)
        extraction_total = ocr_success
        extraction_success = len(extractions)

        ocr_rate = (ocr_success / ocr_total) if ocr_total > 0 else 0
        extraction_rate = (extraction_success / extraction_total) if extraction_total > 0 else 0
        completion_rate = (fields_filled / total_fields) if total_fields > 0 else 0
        avg_confidence = (
            sum(consolidated.confidence_scores.values()) / len(consolidated.confidence_scores)
            if consolidated.confidence_scores else 0
        )

        # --- GUARDAR ARCHIVO CONSOLIDADO CON NOMENCLATURA DINÁMICA ---
        consolidated_filename = f"{s_insured}_{s_claim}_CONSOLIDADO.json"
        
        # GUARDAR el archivo consolidado en data/temp/pipeline_cache (usando ruta absoluta)
        pipeline_cache_dir = project_root / "data" / "temp" / "pipeline_cache"
        pipeline_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directorio pipeline_cache creado/verificado: {pipeline_cache_dir}")
        
        consolidated_json_path = pipeline_cache_dir / consolidated_filename
        
        # Eliminar archivo existente si existe
        if consolidated_json_path.exists():
            logger.info(f"  ⚠️ Reemplazando archivo consolidado existente: {consolidated_filename}")
            consolidated_json_path.unlink()
        
        logger.info(f"✓ Guardando archivo consolidado como: {consolidated_filename}")

        try:
            with open(consolidated_json_path, "w", encoding="utf-8") as f:
                # Guardamos solo los datos consolidados aquí
                json.dump(consolidated.model_dump(), f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"✓ JSON consolidado guardado exitosamente en: {consolidated_json_path}")
            logger.info(f"✓ Tamaño del archivo: {consolidated_json_path.stat().st_size} bytes")
        except Exception as e:
            logger.error(f"❌ Error guardando archivo consolidado: {e}")
            raise
        
        # --- LLAMAR A LA REORGANIZACIÓN DEL CACHÉ ---
        if self.cache_manager:
            self.cache_manager.reorganize_cache_for_case(
                case_id=case_id,
                insured_name=insured_name_from_data,
                claim_number=claim_number_from_data
            )

        results = {
            "case_id": case_id,
            "processing_date": datetime.now().isoformat(),
            "documents_processed": len(documents),
            "extraction_results": [e.model_dump() for e in extractions],
            "consolidated_data": consolidated.model_dump(),
            "fraud_analysis": ai_analysis,
            "processing_metrics": {
                "ocr_success_rate": f"{ocr_rate:.1%}",
                "extraction_success_rate": f"{extraction_rate:.1%}",
                "fields_completion_rate": f"{completion_rate:.1%}",
                "conflicts_resolved": len(consolidated.conflicts_resolved),
                "average_confidence": avg_confidence,
            },
        }

        # Guardamos el reporte completo de resultados (que incluye métricas, etc.) con nombre mejorado
        results_filename = f"{s_insured}_{s_claim}_RESULTADOS.json"
        json_path = output_path / results_filename
        
        # Eliminar archivo existente si existe
        if json_path.exists():
            logger.info(f"  ⚠️ Reemplazando archivo de resultados existente: {results_filename}")
            json_path.unlink()
            
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"✓ JSON de resultados guardado: {json_path}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)
        
        # Emitir evento final de completado
        if self.progress_emitter:
            self.progress_emitter.emit("report", "done", message="Procesamiento completado exitosamente")

        return results

    async def _analyze_fraud(
        self,
        consolidated: ConsolidatedExtraction,
        extractions: List[DocumentExtraction],
    ) -> Dict[str, Any]:
        """
        Análisis de fraude usando IA.
        """
        from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer

        analyzer = AIDocumentAnalyzer()

        # Payload ligero para IA
        docs_for_analysis = [
            {
                "document_type": extr.document_type,
                "key_value_pairs": extr.extracted_fields,
            }
            for extr in extractions
        ]

        analysis = await analyzer.analyze_claim_documents(docs_for_analysis)
        return analysis


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parser de argumentos (sin --use-legacy)."""
    p = argparse.ArgumentParser(description="Fraud Scorer v2.0 - Sistema de Análisis con IA")
    p.add_argument("folder", type=Path, nargs='?', help="Carpeta con documentos del caso")
    p.add_argument("--out", type=Path, default=Path("data/reports"), help="Carpeta de salida")
    p.add_argument("--title", help="Título del caso")
    p.add_argument("--debug", action="store_true", help="Modo debug con más logging")
    p.add_argument("--purge-case", metavar="CASE_ID", help="Limpia artefactos del caso especificado")
    p.add_argument("--purge-orphans", action="store_true", help="Elimina archivos sin entrada en DB")
    return p.parse_args(argv)


async def main(argv: List[str]) -> None:
    """Función principal."""
    args = parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Manejar operaciones de purga
    if args.purge_case or args.purge_orphans:
        from fraud_scorer.services.replay_service import ReplayService
        replay_service = ReplayService()
        
        if args.purge_case:
            logger.info(f"🧹 Limpiando artefactos del caso {args.purge_case}...")
            success = await replay_service.purge_case(args.purge_case)
            if success:
                logger.info(f"✅ Caso {args.purge_case} limpiado exitosamente")
            else:
                logger.error(f"❌ Error limpiando caso {args.purge_case}")
                sys.exit(1)
        
        if args.purge_orphans:
            logger.info("🧹 Eliminando archivos huérfanos...")
            count = await replay_service.purge_orphans()
            logger.info(f"✅ {count} archivos huérfanos eliminados")
        
        # Si solo se ejecutaron operaciones de purga, salir
        if not args.folder:
            sys.exit(0)

    # Validar que se proporcione una carpeta para procesamiento normal
    if not args.folder:
        print("❌ Error: Debe proporcionar una carpeta con documentos o usar --purge-case/--purge-orphans")
        sys.exit(1)

    if not args.folder.is_dir():
        print(f"❌ Error: La carpeta {args.folder} no existe o no es un directorio.")
        sys.exit(1)
    
    # Sistema de análisis
    system = None
    
    # Configurar manejador de señales para cancelación graceful
    def signal_handler(signum, frame):
        """Maneja señales de interrupción (Ctrl+C)"""
        logger.info(f"\n🛑 Señal {signum} recibida. Cancelando proceso...")
        if system:
            system.cancel()
        logger.info("🧹 Limpiando y saliendo...")
        sys.exit(0)
    
    # Registrar manejadores de señales
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args.out.mkdir(parents=True, exist_ok=True)

    system = FraudAnalysisSystemV2()

    try:
        result = await system.process_case(
            folder_path=args.folder,
            output_path=args.out,
            case_title=args.title,
        )
        
        # Mostrar resumen del resultado
        print("\n" + "=" * 60)
        print("✅ PROCESAMIENTO COMPLETADO")
        print("=" * 60)
        if result:
            print(f"📄 Reporte generado: {result.get('report_path', 'N/A')}")
            print(f"🆔 Case ID: {result.get('case_id', 'N/A')}")
            print(f"⚠️  Fraud Score: {result.get('fraud_score', 0):.2%}")
            print(f"🔍 Nivel de Riesgo: {result.get('risk_level', 'N/A')}")
        
    except asyncio.CancelledError:
        logger.info("\n⚠️ Proceso cancelado por el usuario")
        if system:
            await system.cleanup_on_cancel()
        sys.exit(130)  # Código estándar para interrupción por señal
        
    except Exception as e:
        logger.error(f"Error procesando caso: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Refuerzo: asegurar que 'src' esté en sys.path
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    asyncio.run(main(sys.argv[1:]))
