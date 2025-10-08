#!/usr/bin/env python3
"""
Fraud Scorer v2.0 - Sistema de análisis con IA (solo v2, sin legacy)
"""

import sys
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING
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
from fraud_scorer.parsers.processing_hint import ProcessingHint
from fraud_scorer.storage.ocr_cache import OCRCacheManager
from fraud_scorer.storage.post_process_verifier import verify_case_artifacts
from fraud_scorer.storage.db import sha256_of_file, get_conn, save_correlation_findings
from fraud_scorer.storage.cases import create_case

from fraud_scorer.processors.ai.ai_field_extractor import AIFieldExtractor
from fraud_scorer.processors.ai.ai_consolidator import AIConsolidator
from fraud_scorer.templates.ai_report_generator import AIReportGenerator
from fraud_scorer.analyzers.fraud_analyzer import FraudAnalyzer
from fraud_scorer.analyzers.unified_data_layer import UnifiedDataLayer
from fraud_scorer.analyzers.correlation import CorrelationEngine
from fraud_scorer.analyzers.fraud_guide_manager import FraudGuideManager
from fraud_scorer.templates.fraud_report_generator import FraudReportGenerator
from fraud_scorer.models.extraction import (
    DocumentExtraction,
    ConsolidatedExtraction,
    ProgressEvent,
)
from fraud_scorer.models.fraud_analysis import FraudAnalysisResult
from fraud_scorer.processors.document_classifier import DocumentClassifier
from fraud_scorer.processors.document_organizer import DocumentOrganizer
import time
import os

if TYPE_CHECKING:
    from fraud_scorer.analyzers.correlation.models import CorrelationReport


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
        stages = ["upload", "ocr", "extract", "consolidate", "report"]
        
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
                    "consolidate": 5, "report": 5
                }
                total_remaining += defaults.get(stage, 5)
                
        return int(total_remaining * 1000)  # Convertir a ms


class FraudAnalysisSystemV2:
    """
    Sistema de análisis de siniestros v2.0 con IA y Cache OCR (sin legacy).
    """

    def __init__(self, guided_mode: bool = True, extraction_mode: str = "auto", enable_fraud: bool = True):
        # OCR + Parser
        self.ocr_processor = AzureOCRProcessor()
        self.document_parser = DocumentParser(self.ocr_processor)

        # Cache OCR
        self.cache_manager = OCRCacheManager()

        # IA v2 con modo guiado
        self.guided_mode = guided_mode
        self.extraction_mode = extraction_mode
        self.extractor = AIFieldExtractor()
        self.consolidator = AIConsolidator()
        template_path = project_root / "src" / "fraud_scorer" / "templates"
        self.report_generator = AIReportGenerator(template_dir=template_path)
        # Control de análisis de fraude
        self.enable_fraud = bool(enable_fraud)
        self.correlation_engine = CorrelationEngine() if self.enable_fraud else None

        mode_desc = "Guiado" if self.guided_mode else "Estándar"
        logger.info(f"Sistema v2.0 inicializado - Modo: {mode_desc}, Extracción: {self.extraction_mode}")
        
        # Control de cancelación
        self._cancelled = False
        self._cancel_lock = threading.Lock()
        self._cleanup_paths = []
        self.cancellation_check = None
        self.progress_emitter = None  # Se inicializa por caso
        self.reprocess_mode = False
        self.reprocess_options: Dict[str, Any] = {}

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
    
    async def _pause_for_manual_review(self, case_id: str) -> None:
        """
        Fase 1.4.1: Pausa controlada para revisión manual de clasificación.
        Crea un archivo marcador '<case_id>.awaiting_review' y espera hasta
        que exista '<case_id>.resume'. No separa procesos ni fases públicas.
        """
        base = os.getenv("FS_DATA_DIR", "data")
        pc_dir = Path(base) / "temp" / "pipeline_cache"
        pc_dir.mkdir(parents=True, exist_ok=True)
        await_marker = pc_dir / f"{case_id}.awaiting_review"
        resume_marker = pc_dir / f"{case_id}.resume"

        # Crear marcador de espera
        try:
            with open(await_marker, "w", encoding="utf-8") as f:
                f.write(json.dumps({"case_id": case_id, "ts": time.time()}))
        except Exception:
            # Crear vacío como fallback
            try:
                await_marker.touch(exist_ok=True)
            except Exception:
                pass

        # Emitir evento de progreso si está disponible
        if self.progress_emitter:
            self.progress_emitter.emit("review", "started", message="Esperando revisión manual de clasificación")

        # Espera activa no bloqueante con chequeo de cancelación
        logger.info("⏸️  Pausa 1.4.1: esperando confirmación de revisión...")
        import asyncio as _async
        while not resume_marker.exists():
            # Permitir cancelación cooperativa si el caller la provee
            if self.cancellation_check and await self.cancellation_check():
                await self.cleanup_on_cancel()
                raise _async.CancelledError("Proceso cancelado durante revisión 1.4.1")
            await _async.sleep(0.5)

        # Limpiar marcadores y continuar
        try:
            resume_marker.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            await_marker.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

        if self.progress_emitter:
            self.progress_emitter.emit("review", "done", message="Revisión confirmada; continuando")

    def _extract_original_filename(self, ocr_payload: Any, cache_path: Path) -> Optional[str]:
        """Intenta recuperar el nombre original del documento desde el payload de OCR."""
        if isinstance(ocr_payload, dict):
            metadata = ocr_payload.get("metadata") if isinstance(ocr_payload.get("metadata"), dict) else None
            if metadata:
                for key in ("file_name", "filename", "original_filename"):
                    value = metadata.get(key)
                    if value:
                        return str(value)

        try:
            stem = cache_path.stem
            if stem.startswith("ocr_results_for_"):
                stem = stem[len("ocr_results_for_"):]
            suffix = cache_path.suffix
            if suffix.lower() == ".json":
                return stem
            return f"{stem}{suffix}" if suffix else stem
        except Exception:
            return None

    def _rehydrate_processing_hint(self, payload: Any) -> Optional[ProcessingHint]:
        """Reconstruye un ``ProcessingHint`` serializado si es posible."""

        if isinstance(payload, ProcessingHint):
            return payload
        if not isinstance(payload, dict):
            return None

        data = dict(payload)
        data.pop("ui_metadata", None)

        allowed_keys = {
            "file_name",
            "file_extension",
            "mime_type",
            "file_size_bytes",
            "manual_override",
            "is_gps_candidate",
            "confidence",
            "detector_version",
            "vector_ratio",
            "reason",
        }

        filtered = {key: data[key] for key in allowed_keys if key in data}
        required = {"file_name", "file_extension", "mime_type", "file_size_bytes"}
        if not required.issubset(filtered):
            logger.debug("ProcessingHint incompleto: faltan %s", required - set(filtered))
            return None

        try:
            return ProcessingHint(**filtered)
        except Exception as exc:
            logger.debug(
                "No se pudo reconstruir ProcessingHint para %s: %s",
                filtered.get("file_name"),
                exc,
            )
            return None

    def _prepare_docless_ocr(
        self,
        case_id: str,
        case_data: Dict[str, Any],
        base_folder: Path,
    ) -> Dict[str, Any]:
        """
        Reconstruye los resultados de OCR únicamente desde el cache JSON reorganizado.
        Devuelve un diccionario con `ocr_results`, `cache_files` y `doc_names`.
        """
        if not self.cache_manager:
            raise RuntimeError("El modo sin documentos originales requiere un administrador de cache activo")

        classified_types = case_data.get("classified_types") or []
        manual_overrides = case_data.get("manual_classifications") or {}
        extraction_results = case_data.get("extraction_results") or []

        doc_type_by_name: Dict[str, Any] = {}
        for item in classified_types:
            if not item:
                continue
            name = str(item.get("filename") or "").strip()
            if not name:
                continue
            doc_type_by_name.setdefault(name, item.get("document_type"))

        for name, override in manual_overrides.items():
            if override:
                doc_type_by_name[str(name)] = override

        for item in extraction_results:
            if not item:
                continue
            if isinstance(item, dict):
                name = item.get("source_document")
                doc_type = item.get("document_type")
            else:
                name = getattr(item, "source_document", None)
                doc_type = getattr(item, "document_type", None)
            if name and doc_type and name not in doc_type_by_name:
                doc_type_by_name[str(name)] = doc_type

        ocr_results: List[Dict[str, Any]] = []
        cache_files: List[str] = []
        doc_names: List[str] = []
        seen: set[str] = set()

        manifest_paths = case_data.get("documents") or []
        for path_str in manifest_paths:
            if not path_str:
                continue
            path = Path(path_str)
            try:
                ocr_payload = self.cache_manager.get_cache(path, case_id)
            except Exception as exc:
                logger.warning(f"No se pudo cargar OCR desde {path}: {exc}")
                continue

            if not ocr_payload:
                logger.warning(f"No se encontró contenido OCR en cache para {path}")
                continue

            original_name = self._extract_original_filename(ocr_payload, path)
            if not original_name:
                logger.warning(f"No se pudo inferir el nombre original para {path}")
                continue

            if original_name in seen:
                continue

            doc_names.append(original_name)
            seen.add(original_name)
            cache_files.append(str(path))
            ocr_results.append(
                {
                    "filename": original_name,
                    "ocr_result": ocr_payload,
                    "document_type": doc_type_by_name.get(original_name),
                }
            )

        if not doc_names:
            candidate_names: List[str] = []
            for item in extraction_results:
                if isinstance(item, dict):
                    cand = item.get("source_document")
                else:
                    cand = getattr(item, "source_document", None)
                if cand:
                    candidate_names.append(str(cand))

            if not candidate_names:
                candidate_names = [
                    str(item.get("filename"))
                    for item in classified_types
                    if item and item.get("filename")
                ]

            candidate_names = [name for name in candidate_names if name and name not in seen]

            for name in candidate_names:
                try:
                    doc_folder = self.cache_manager._sanitize_filename(Path(name).stem)
                except Exception:
                    doc_folder = Path(name).stem

                doc_path = base_folder / doc_folder / name

                try:
                    ocr_payload = self.cache_manager.get_cache(doc_path, case_id)
                except Exception as exc:
                    logger.warning(f"No se pudo cargar OCR desde cache para {name}: {exc}")
                    continue

                if not ocr_payload:
                    logger.warning(f"No existe cache OCR para {name}; omitiendo")
                    continue

                original_name = name
                if original_name in seen:
                    continue

                seen.add(original_name)
                doc_names.append(original_name)
                reorganized_path = None
                try:
                    reorganized_path = self.cache_manager._find_cache_in_reorganized_structure(
                        doc_path, case_id=case_id
                    )
                except Exception:
                    reorganized_path = None
                cache_files.append(str(reorganized_path or doc_path))
                ocr_results.append(
                    {
                        "filename": original_name,
                        "ocr_result": ocr_payload,
                        "document_type": doc_type_by_name.get(original_name),
                    }
                )

        if not ocr_results:
            raise RuntimeError(
                "No se pudo reconstruir OCR desde el cache JSON; ejecute re-OCR o revise la carpeta del caso"
            )

        return {
            "ocr_results": ocr_results,
            "cache_files": cache_files,
            "doc_names": doc_names,
        }

    def _hydrate_fraud_results(self, items: List[Any]) -> List[Any]:
        """Convierte dicts de fraude guardados en el índice a modelos FraudAnalysisResult."""
        hydrated: List[Any] = []
        if not items:
            return hydrated

        try:
            from fraud_scorer.models.fraud_analysis import FraudAnalysisResult
        except Exception as exc:
            logger.warning(f"No se pudo importar FraudAnalysisResult para rehidratar fraude: {exc}")
            return hydrated

        for item in items:
            if isinstance(item, FraudAnalysisResult):
                hydrated.append(item)
                continue
            if isinstance(item, dict):
                try:
                    payload = dict(item)
                    payload.setdefault("include_in_report", True)
                    hydrated.append(FraudAnalysisResult.model_validate(payload))
                except Exception as exc:
                    logger.warning(f"No se pudo reconstruir resultado de fraude: {exc}")
        return hydrated

    def _run_correlation_analysis(
        self,
        *,
        case_id: str,
        consolidated: ConsolidatedExtraction,
        extractions: List[DocumentExtraction],
        fraud_analyses: List[FraudAnalysisResult],
        case_data: Dict[str, Any],
    ) -> Optional["CorrelationReport"]:
        if not self.correlation_engine or not fraud_analyses:
            return None
        try:
            report = self.correlation_engine.run(
                case_id=case_id,
                consolidated=consolidated,
                extractions=extractions,
                fraud_results=fraud_analyses,
                case_index=case_data,
                cache_manager=self.cache_manager,
            )
            logger.info(
                "✓ Motor de correlación ejecutado: %s hallazgos",
                len(report.findings),
            )
            return report
        except Exception as exc:  # pragma: no cover - defensivo ante sandbox
            logger.warning(
                "⚠️ Motor de correlación degradado a revisión manual para %s: %s",
                case_id,
                exc,
            )
            return None
    
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
        reprocess_mode: bool = False,
        reprocess_options: Optional[Dict[str, Any]] = None,
        existing_case_id: Optional[str] = None,
        processing_hints: Optional[Dict[str, Dict[str, Any]]] = None,
        gps_manual_flags: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Procesa un caso completo con el flujo v2 (solo IA).
        """
        logger.info("=" * 60)
        logger.info(f"📁 Procesando caso: {folder_path.name}")
        logger.info("🤖 Modo: IA Avanzada v2.0")
        logger.info("=" * 60)

        # Log de depuración: verificar qué caso estamos procesando
        logger.info(f"🔍 DEPURACIÓN: Procesando con case_id={existing_case_id or 'AUTO-DETECTAR'}")
        logger.info(f"  Carpeta: {folder_path}")
        logger.info(f"  Título: {case_title or 'N/A'}")
        logger.info(f"  Modo reproceso: {reprocess_mode}")
        if reprocess_options:
            logger.info(f"  Opciones reproceso: {reprocess_options}")

        # Buscar documentos soportados (ignorar archivos '._' de macOS)
        supported_extensions = {
            ".pdf", ".png", ".jpg", ".jpeg", ".tiff",
            ".docx", ".xlsx", ".csv"
        }

        def _is_supported(path: Path) -> bool:
            return (
                path.is_file()
                and path.suffix.lower() in supported_extensions
                and not path.name.startswith("._")
            )

        documents: List[Path]

        case_index_for_reprocess: Dict[str, Any] = {}
        if self.reprocess_mode:
            try:
                if self.cache_manager and existing_case_id:
                    # Intentar cargar índice del caso existente para obtener documentos
                    case_index_for_reprocess = self.cache_manager.get_case_index(existing_case_id, auto_reconstruct=True) or {}
            except Exception:
                case_index_for_reprocess = {}

            stored_docs = case_index_for_reprocess.get("documents") or []
            if stored_docs:
                documents = [Path(p) for p in stored_docs]
                missing = [str(p) for p in documents if not p.exists()]
                if missing:
                    logger.info("Algunos documentos no existen físicamente; se usará el cache JSON: %s", ", ".join(missing))
            else:
                documents = []
        else:
            documents = []

        if not documents:
            if folder_path.exists() and folder_path.is_dir():
                documents = [p for p in folder_path.iterdir() if _is_supported(p)]
            else:
                documents = []

            if not documents and folder_path.exists():
                # Estructura reorganizada: buscar archivos soportados en subcarpetas
                nested_candidates = [p for p in folder_path.rglob("*") if _is_supported(p)]
                if nested_candidates:
                    documents = sorted(nested_candidates, key=lambda p: str(p))
                if documents:
                    logger.info(
                        "Carpeta %s no contiene documentos en el nivel raíz; usando %d archivos encontrados en subcarpetas",
                        folder_path,
                        len(documents)
                    )

        if not documents:
            allow_docless = False
            try:
                opts = dict(reprocess_options or {})
                if reprocess_mode and not opts.get("reprocess_ocr"):
                    # Permitimos modo sin documentos siempre que exista un case_id previo
                    # (cargado desde el índice de cache) y se solicite reprocesar
                    # al menos una fase distinta a OCR.
                    docless_flags = (
                        "reprocess_classification",
                        "reprocess_policy_detection",
                        "reprocess_extraction",
                        "reprocess_consolidation",
                        "reprocess_fraud",
                    )
                    wants_any = any(bool(opts.get(flag)) for flag in docless_flags)
                    has_existing_case = bool(existing_case_id)
                    allow_docless = wants_any and has_existing_case
            except Exception:
                allow_docless = False

            if not allow_docless:
                raise RuntimeError("No se encontraron documentos para procesar")
            else:
                logger.info(
                    "ℹ️ Reproceso sin archivos originales; se utilizarán los resultados de OCR almacenados"
                )

        logger.info(f"✓ Encontrados {len(documents)} documentos")

        # Verificar si ya existe un caso para esta ruta o título
        from fraud_scorer.storage.cases import get_case_by_path, get_case_by_title, get_conn
        import hashlib

        case_id = existing_case_id

        if case_id:
            logger.info(f"✓ Usando case_id proporcionado: {case_id}")
        else:
            # DETECCIÓN INTELIGENTE DE CASO EXISTENTE
            # 1. Primero intentar por ruta
            existing_case = get_case_by_path(str(folder_path))

            if existing_case:
                case_id = existing_case['case_id']
                logger.info(f"✓ Usando caso existente por ruta: {case_id}")
            else:
                # 2. Buscar por hash de documentos (primeros N) para detectar casos duplicados
                if documents:
                    try:
                        max_samples = int(os.getenv("FS_DETECT_HASH_SAMPLES", "5"))
                    except Exception:
                        max_samples = 5
                    for sample_doc in documents[:max(1, max_samples)]:
                        if not sample_doc.exists():
                            continue
                        try:
                            doc_hash = sha256_of_file(sample_doc)
                            # Buscar en BD si este documento ya existe
                            with get_conn() as conn:
                                row = conn.execute(
                                    """SELECT c.case_id, c.name
                                       FROM documents d
                                       JOIN cases c ON d.case_id = c.case_id
                                       WHERE d.file_hash = ?
                                       LIMIT 1""",
                                    (doc_hash,)
                                ).fetchone()
                            if row:
                                case_id = row['case_id']
                                logger.info(
                                    f"✓ Caso detectado por contenido de documentos: {case_id} ({row['name']})"
                                )
                                logger.info("  → Los documentos ya fueron procesados anteriormente")
                                logger.info(f"  → Hash del documento: {doc_hash[:8]}... ({sample_doc.name})")
                                logger.info("  → REUTILIZANDO caso existente en lugar de crear uno nuevo")
                                break
                        except Exception as e:
                            logger.debug(f"No se pudo calcular hash para detección en {sample_doc}: {e}")

                # 3. Si es carpeta reorganizada, buscar por patrón genérico de claim_number o carpeta
                if not case_id:
                    # Extraer claim_number del nombre de carpeta (patrón genérico: sufijo de 6+ dígitos)
                    import re
                    match = re.search(r'(?:^| - )(\d{6,})$', folder_path.name)
                    if not match:
                        match = re.search(r'(\d{6,})', folder_path.name)
                    claim_num = match.group(1) if match else None

                    # Buscar en índices de cache por claim_number o por coincidencia exacta de carpeta
                    if self.cache_manager:
                        for index_file in self.cache_manager.index_dir.glob("*.json"):
                            try:
                                with open(index_file, 'r', encoding='utf-8') as f:
                                    idx_data = json.load(f)
                                if claim_num and idx_data.get('claim_number') == claim_num:
                                    case_id = index_file.stem
                                    logger.info(f"✓ Caso detectado por claim_number {claim_num}: {case_id}")
                                    break
                                case_folder = idx_data.get('case_folder') or ""
                                if case_folder and case_folder == folder_path.name:
                                    case_id = index_file.stem
                                    logger.info(f"✓ Caso detectado por carpeta reorganizada: {case_id}")
                                    break
                            except Exception:
                                continue

                # 4. Si no se encontró caso existente, intentar por título
                if not case_id:
                    title = case_title or folder_path.name
                    existing_case = get_case_by_title(title)

                    if existing_case:
                        case_id = existing_case['case_id']
                        logger.info(f"✓ Usando caso existente por título: {case_id}")
                    else:
                        # Solo crear nuevo caso si no existe ninguno
                        logger.info("⚠️ No se encontró ningún caso existente que coincida")
                        logger.info("  → Creando un NUEVO caso en la BD...")
                        case_id = create_case(
                            title=title,
                            base_path=str(folder_path)
                        )
                        logger.info(f"✓ Nuevo caso creado: {case_id}")
                        logger.info(f"  → Este es un caso COMPLETAMENTE NUEVO")
        
        logger.info(f"✓ Case ID final: {case_id}")
        
        # Guardar callbacks para usar en los métodos internos
        self.progress_callback = progress_callback
        self.cancellation_check = cancellation_check

        # Reset estado de cancelación
        self.reset_cancellation()

        # Configurar opciones de reprocesamiento
        self.reprocess_mode = bool(reprocess_mode)
        self.reprocess_options = dict(reprocess_options or {})

        # Registrar carpeta para limpieza en caso de cancelación
        if folder_path not in self._cleanup_paths:
            self._cleanup_paths.append(folder_path)

        # Inicializar emisor de progreso para este caso
        self.progress_emitter = _ProgressEmitter(case_id)

        # Ejecutar pipeline v2
        return await self._process_with_ai(
            documents,
            case_id,
            output_path,
            folder_path,
            processing_hints=processing_hints,
            gps_manual_flags=gps_manual_flags,
        )

    async def _process_with_ai(
        self,
        documents: List[Path],
        case_id: str,
        output_path: Path,
        base_folder: Path,
        *,
        processing_hints: Optional[Dict[str, Dict[str, Any]]] = None,
        gps_manual_flags: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Procesamiento con el sistema de IA y cache.
        """
        options = dict(self.reprocess_options or {})
        if self.reprocess_mode and options.get("reprocess_extraction"):
            options["reprocess_consolidation"] = True

        def wants(flag: str) -> bool:
            if not self.reprocess_mode:
                return True
            return bool(options.get(flag))

        case_data: Dict[str, Any] = {}
        if self.cache_manager:
            try:
                case_data = self.cache_manager.get_case_index(case_id) or {}
            except Exception:
                case_data = {}
        case_data.setdefault("case_id", case_id)

        if processing_hints:
            existing_hints = case_data.get("gps_processing_hints")
            if isinstance(existing_hints, dict):
                existing_hints.update(processing_hints)
            else:
                case_data["gps_processing_hints"] = dict(processing_hints)

        if gps_manual_flags:
            existing_flags = case_data.get("gps_manual_flags")
            if isinstance(existing_flags, dict):
                existing_flags.update(gps_manual_flags)
            else:
                case_data["gps_manual_flags"] = dict(gps_manual_flags)

        hint_objects: Dict[str, ProcessingHint] = {}
        if processing_hints:
            for name, payload in processing_hints.items():
                hint = self._rehydrate_processing_hint(payload)
                if hint:
                    hint_objects[name] = hint

        def _hint_for_path(doc_path: Path) -> Optional[ProcessingHint]:
            if not hint_objects:
                return None
            hint = hint_objects.get(doc_path.name)
            if hint:
                return hint
            return hint_objects.get(str(doc_path.name))

        doc_ids_by_name: Dict[str, str] = {}
        try:
            with get_conn() as conn:
                rows = conn.execute(
                    "SELECT id, filename FROM documents WHERE case_id=?",
                    (case_id,)
                ).fetchall()
                for row in rows:
                    name = row["filename"]
                    if name and name not in doc_ids_by_name:
                        doc_ids_by_name[name] = row["id"]
        except Exception as db_exc:
            logger.debug(f"No se pudieron cargar IDs de documentos para {case_id}: {db_exc}")

        # Persistir hashes conocidos de documentos para reorganizaciones futuras
        document_hashes: Dict[str, str] = {}
        existing_hashes = case_data.get("document_hashes")
        if isinstance(existing_hashes, dict):
            document_hashes.update(existing_hashes)

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

        reuse_existing_ocr = self.reprocess_mode and not wants("reprocess_ocr")
        docless_mode = self.reprocess_mode and not wants("reprocess_ocr") and not documents
        docless_doc_names: List[str] = []
        all_cached = False

        # Asegurar que conocemos el hash de cada documento mientras aún está disponible
        for doc_path in documents:
            path_str = str(doc_path)
            if path_str in document_hashes:
                continue
            # No calcular hash sobre JSON de OCR; sólo sobre originales
            if Path(path_str).suffix.lower() == ".json":
                continue
            try:
                document_hashes[path_str] = sha256_of_file(doc_path)
            except FileNotFoundError:
                logger.warning(
                    "No se pudo calcular hash para %s porque el archivo no existe",
                    doc_path
                )
            except Exception as hash_exc:
                logger.warning(
                    "No se pudo calcular hash para %s: %s",
                    doc_path,
                    hash_exc
                )

        if docless_mode:
            docless_payload = self._prepare_docless_ocr(case_id, case_data, base_folder)
            ocr_results = docless_payload["ocr_results"]
            cache_files.extend(docless_payload["cache_files"])
            docless_doc_names = docless_payload["doc_names"]
            logger.info(f"✓ Documentos listos desde OCR reorganizado (JSON): {len(ocr_results)}")
            all_cached = True
            if self.progress_emitter:
                self.progress_emitter.emit("ocr", "done", message="OCR desde cache JSON listo")

        elif reuse_existing_ocr and self.cache_manager:
            logger.info("♻️ Reprocesamiento: se conservarán los resultados de OCR previos.")
            all_cached = True
            for idx, doc_path in enumerate(documents, 1):
                if self.cancellation_check and await self.cancellation_check():
                    await self.cleanup_on_cancel()
                    raise asyncio.CancelledError(f"Proceso cancelado en documento {idx}/{len(documents)}")

                from_cache = False
                ocr_result = self.cache_manager.get_cache(doc_path, case_id)
                if ocr_result:
                    from_cache = True
                else:
                    logger.warning(
                        "No se encontró OCR previo para %s; reprocesando solo este documento.",
                        doc_path.name,
                    )
                    try:
                        ocr_result = self.document_parser.parse_document(
                            doc_path,
                            hint=_hint_for_path(doc_path),
                        )
                        if ocr_result and self.cache_manager:
                            self.cache_manager.save_cache(doc_path, ocr_result, case_id)
                        all_cached = False
                    except Exception as e:
                        logger.error(f"  ❌ Error procesando {doc_path.name}: {e}", exc_info=True)
                        all_cached = False
                        continue

                if ocr_result:
                    ocr_results.append(
                        {
                            "filename": doc_path.name,
                            "ocr_result": ocr_result,
                            "document_type": None,
                        }
                    )
                    if from_cache:
                        cache_files.append(str(doc_path))

            logger.info(f"✓ Documentos listos desde OCR previo: {len(ocr_results)}/{len(documents)}")
        else:
            # Paso 3.2: Verificar si todos los documentos ya están en cache
            all_cached = True
            cached_count = 0
            if self.cache_manager:
                for doc_path in documents:
                    if self.cache_manager.has_cache(doc_path, case_id=case_id):
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
                                doc_path.name,
                            )

                        # Emitir evento de progreso OCR
                        if self.progress_emitter:
                            self.progress_emitter.emit(
                                "ocr",
                                "running",
                                doc_index=idx,
                                doc_total=len(documents),
                                message=f"Cargando desde cache: {doc_path.name}",
                            )

                        ocr_result = self.cache_manager.get_cache(doc_path, case_id)
                        if ocr_result:
                            ocr_results.append(
                                {
                                    "filename": doc_path.name,
                                    "ocr_result": ocr_result,
                                    "document_type": None,
                                }
                            )
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
                                doc_path.name,
                            )

                        # Emitir evento de progreso OCR
                        if self.progress_emitter:
                            self.progress_emitter.emit(
                                "ocr",
                                "running",
                                doc_index=idx,
                                doc_total=len(documents),
                                message=f"Procesando: {doc_path.name}",
                            )

                        # Usar cache si existe
                        if self.cache_manager.has_cache(doc_path, case_id=case_id):
                            logger.info(f"  ⚡ Usando cache para: {doc_path.name}")
                            ocr_result = self.cache_manager.get_cache(doc_path, case_id)
                            if ocr_result:
                                cache_files.append(str(doc_path))
                        else:
                            # OCR/Parser tolerante a fallos
                            logger.info(f"  🔄 Procesando con OCR/Parser: {doc_path.name}")
                            try:
                                ocr_result = self.document_parser.parse_document(
                                    doc_path,
                                    hint=_hint_for_path(doc_path),
                                )
                                if self.cache_manager and ocr_result:
                                    self.cache_manager.save_cache(doc_path, ocr_result, case_id)
                            except Exception as e:
                                logger.error(f"  ❌ Error procesando {doc_path.name}: {e}", exc_info=True)
                                continue

                        if ocr_result:
                            ocr_results.append(
                                {
                                    "filename": doc_path.name,
                                    "ocr_result": ocr_result,
                                    "document_type": None,
                                }
                            )

                    logger.info(f"✓ Procesamiento completado: {len(ocr_results)}/{len(documents)} exitosos")
            else:
                # Si no hay cache manager, procesar todos los documentos normalmente
                for doc_path in documents:
                    logger.info(f"  Procesando: {doc_path.name}")
                    logger.info(f"  🔄 Procesando con OCR/Parser: {doc_path.name}")
                    try:
                        ocr_result = self.document_parser.parse_document(
                            doc_path,
                            hint=_hint_for_path(doc_path),
                        )
                    except Exception as e:
                        logger.error(f"  ❌ Error procesando {doc_path.name}: {e}", exc_info=True)
                        continue

                    if ocr_result:
                        ocr_results.append(
                            {
                                "filename": doc_path.name,
                                "ocr_result": ocr_result,
                                "document_type": None,
                            }
                        )

                logger.info(f"✓ Procesamiento completado: {len(ocr_results)}/{len(documents)} exitosos")

        # Guardar índice del caso para replay
        if self.cache_manager:
            folder_name = base_folder.name if base_folder.name else case_data.get("case_title", "UNKNOWN")
            parts = folder_name.split(' - ', 1)
            fallback_insured_name = parts[0].strip() if len(parts) == 2 else folder_name
            fallback_claim_number = parts[1].strip() if len(parts) == 2 else ""

            case_data.setdefault("case_title", folder_name)
            if not case_data.get("insured_name"):
                case_data["insured_name"] = fallback_insured_name
            if not case_data.get("claim_number"):
                case_data["claim_number"] = fallback_claim_number

            existing_docs = set(case_data.get("documents") or [])
            merged_docs = list(existing_docs.union({str(d) for d in documents}))
            existing_cache_files = set(case_data.get("cache_files") or [])
            merged_cache = list(existing_cache_files.union(set(cache_files)))

            if merged_docs:
                case_data["documents"] = merged_docs
            elif documents:
                case_data["documents"] = [str(d) for d in documents]
            else:
                json_candidates = [path for path in cache_files if str(path).lower().endswith(".json")]
                if json_candidates:
                    case_data["documents"] = json_candidates

            case_data["cache_files"] = merged_cache
            doc_count_for_index = len(case_data.get("documents", []))
            if not doc_count_for_index:
                doc_count_for_index = len(documents) or len(docless_doc_names)
            case_data["total_documents"] = doc_count_for_index
            if not case_data.get("folder_path"):
                case_data["folder_path"] = str(base_folder)
            case_data["processed_at"] = datetime.now().isoformat()
            case_data["status"] = "processed"
            if document_hashes:
                case_data["document_hashes"] = document_hashes

            self.cache_manager.save_case_index(case_id, case_data)
        
        # Emitir evento de finalización de OCR (si no se hizo antes)
        if self.progress_emitter and not all_cached:
            self.progress_emitter.emit("ocr", "done", message="Procesamiento de documentos completado")

        # ============================================
        # FASE 1.4: CLASIFICACIÓN DE DOCUMENTOS (LLM + heurística)
        # ============================================
        logger.info("\n🧾 FASE 1.4: Clasificación de documentos")
        logger.info("-" * 40)

        use_llm_cls = os.getenv("USE_LLM_DOC_CLASSIFIER", "true").lower() == "true"
        from fraud_scorer.settings import CLASSIFICATION_CONFIG
        min_conf = CLASSIFICATION_CONFIG.get("min_confidence_threshold", 0.6)
        sample_len = CLASSIFICATION_CONFIG.get("sample_text_length", 1500)

        previous_classifications = case_data.get("classified_types") or []
        previous_types = {
            item.get("filename"): item.get("document_type")
            for item in previous_classifications
            if item and item.get("filename")
        }
        previous_confidence = {
            item.get("filename"): item.get("confidence")
            for item in previous_classifications
            if item and item.get("filename")
        }
        previous_reasons = {
            item.get("filename"): item.get("reasons") or []
            for item in previous_classifications
            if item and item.get("filename")
        }
        manual_overrides = case_data.get("manual_classifications") or {}

        run_classification = wants("reprocess_classification")

        ai_predictions_map: Dict[str, str] = {}
        ai_prediction_details: Dict[str, Dict[str, Any]] = {}

        if run_classification:
            classifier = DocumentClassifier()

            import asyncio as _asyncio
            sem = _asyncio.Semaphore(4)

            async def _classify_doc(doc_data: Dict[str, Any]) -> None:
                async with sem:
                    try:
                        ocr = doc_data.get("ocr_result") or {}
                        text = ""
                        if isinstance(ocr, dict):
                            text = (ocr.get("text") or "")[:sample_len]
                        else:
                            text = (getattr(ocr, "text", "") or "")[:sample_len]

                        doc_type, conf, reasons = await classifier.classify(
                            sample_text=text,
                            filename=doc_data.get("filename", ""),
                            use_llm_fallback=use_llm_cls,
                        )

                        if conf < min_conf:
                            try:
                                detected = self.extractor._detect_document_type(
                                    self.extractor._ocr_to_dict_safe(ocr),
                                    doc_data.get("filename", ""),
                                )
                                logger.info(
                                    f"  ⚠️ Baja confianza LLM ({conf:.2f}) para {doc_data.get('filename')}. "
                                    f"Usando heurística: {detected}"
                                )
                                doc_data["document_type"] = detected
                            except Exception:
                                doc_data["document_type"] = doc_type
                        else:
                            doc_data["document_type"] = doc_type

                        try:
                            doc_data["classification_confidence"] = float(conf)
                        except Exception:
                            doc_data["classification_confidence"] = None
                        doc_data["classification_reasons"] = reasons or []

                        logger.info(
                            f"  📄 {doc_data.get('filename')}: tipo={doc_data.get('document_type')} "
                            f"(conf={conf:.2f})"
                        )
                    except Exception as e:
                        logger.warning(f"  ❌ Error clasificando {doc_data.get('filename')}: {e}")
                        try:
                            ocr = doc_data.get("ocr_result") or {}
                            detected = self.extractor._detect_document_type(
                                self.extractor._ocr_to_dict_safe(ocr),
                                doc_data.get("filename", ""),
                            )
                            doc_data["document_type"] = detected
                        except Exception:
                            doc_data["document_type"] = "otro"
                        doc_data["classification_confidence"] = None
                        doc_data["classification_reasons"] = []
                    finally:
                        doc_data["_ai_document_type"] = doc_data.get("document_type")
                        doc_data["_ai_confidence"] = doc_data.get("classification_confidence")
                        doc_data["_ai_reasons"] = doc_data.get("classification_reasons") or []

            await _asyncio.gather(*[_classify_doc(d) for d in ocr_results])
        else:
            logger.info("♻️ Reprocesamiento: conservando clasificaciones previas.")
            baseline_map = {}
            raw_ai = case_data.get("ai_classifications")
            if isinstance(raw_ai, dict):
                baseline_map = dict(raw_ai)
            for doc in ocr_results:
                fname = doc.get("filename")
                baseline_type = baseline_map.get(fname) or previous_types.get(fname)

                doc["_ai_document_type"] = baseline_type or previous_types.get(fname) or doc.get("document_type")
                doc["_ai_confidence"] = previous_confidence.get(fname)
                doc["_ai_reasons"] = previous_reasons.get(fname, [])

                doc["classification_confidence"] = previous_confidence.get(fname)
                doc["classification_reasons"] = previous_reasons.get(fname, [])

                if baseline_type:
                    doc["document_type"] = baseline_type
                elif not doc.get("document_type"):
                    doc["document_type"] = previous_types.get(fname, "otro")

        if manual_overrides:
            for doc in ocr_results:
                fname = doc.get("filename")
                override = manual_overrides.get(fname)
                if override:
                    doc["document_type"] = override

        for doc in ocr_results:
            fname = doc.get("filename")
            if not fname:
                continue
            ai_type = doc.get("_ai_document_type") or doc.get("document_type")
            ai_predictions_map[fname] = ai_type
            ai_prediction_details[fname] = {
                "document_type": ai_type,
                "confidence": doc.get("_ai_confidence"),
                "reasons": doc.get("_ai_reasons") or [],
            }

        if self.cache_manager:
            try:
                case_data["classified_types"] = [
                    {
                        "filename": d.get("filename"),
                        "document_type": d.get("document_type"),
                        "confidence": d.get("classification_confidence"),
                        "reasons": d.get("classification_reasons"),
                        "ai_document_type": d.get("_ai_document_type"),
                        "ai_confidence": d.get("_ai_confidence"),
                        "ai_reasons": d.get("_ai_reasons"),
                    }
                    for d in ocr_results
                ]
                case_data["ai_classifications"] = dict(ai_predictions_map)
                case_data["ai_prediction_details"] = dict(ai_prediction_details)

                history_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "mode": "reprocess" if self.reprocess_mode else "initial",
                    "options": dict(self.reprocess_options or {}) if self.reprocess_mode else None,
                    "predictions": ai_prediction_details,
                }

                history = case_data.get("ai_predictions_history")
                if not isinstance(history, list):
                    history = []
                history.insert(0, history_entry)
                case_data["ai_predictions_history"] = history[:10]

                self.cache_manager.save_case_index(case_id, case_data)
            except Exception as e:
                logger.warning(f"No se pudo persistir mapping de tipos: {e}")

        # ============================================
        # FASE 1.4.1: Pausa para revisión manual (controlada por env)
        # ============================================
        try:
            if os.getenv("ENABLE_CLASSIFICATION_REVIEW", "false").lower() == "true":
                await self._pause_for_manual_review(case_id)
                # Tras la revisión, recargar tipos corregidos desde el índice
                try:
                    case_data = self.cache_manager.get_case_index(case_id) if self.cache_manager else None
                    if case_data and isinstance(case_data.get("classified_types"), list):
                        type_by_file = {
                            str(item.get("filename")): item.get("document_type")
                            for item in case_data["classified_types"]
                            if item and item.get("filename") and item.get("document_type")
                        }
                        fixes = 0
                        for d in ocr_results:
                            fname = d.get("filename")
                            if not fname:
                                continue
                            new_t = type_by_file.get(fname)
                            if new_t and new_t != d.get("document_type"):
                                prev = d.get("document_type")
                                d["document_type"] = new_t
                                fixes += 1
                                logger.info(f"  🔧 Tipo corregido por revisión: '{fname}': {prev} -> {new_t}")
                        if fixes:
                            logger.info(f"✓ Tipos actualizados desde revisión: {fixes} cambios aplicados")
                except Exception as _e:
                    logger.warning(f"No se pudieron aplicar tipos corregidos: {_e}")
        except Exception as e:
            logger.warning(f"No se pudo realizar pausa de revisión: {e}")

        # ============================================
        # FASE 1.5: DETECCIÓN DE TIPO DE PÓLIZA (HDI)
        # ============================================
        logger.info("\n🔎 FASE 1.5: Detección de tipo de póliza")
        logger.info("-" * 40)

        policy_type = case_data.get("policy_type") if not wants("reprocess_policy_detection") else None
        policy_document = None

        run_policy_detection = wants("reprocess_policy_detection")

        if run_policy_detection and os.getenv("ENABLE_HDI_SPECIAL_RULES", "true").lower() == "true":
            import unicodedata

            def _normalize(s: str) -> str:
                try:
                    s = unicodedata.normalize('NFKD', s)
                    return ''.join(c for c in s if not unicodedata.combining(c)).lower()
                except Exception:
                    return (s or "").lower()

            for doc_data in ocr_results:
                filename = doc_data.get("filename", "")
                filename_norm = _normalize(filename)

                # Determinar tipo canónico usando heurísticas existentes
                try:
                    ocr_dict = self.extractor._ocr_to_dict_safe(doc_data.get("ocr_result", {}))
                    detected_type = self.extractor._detect_document_type(ocr_dict, filename)
                except Exception:
                    detected_type = None

                is_policy_candidate = (
                    (detected_type == "poliza_de_la_aseguradora") or ("poliza" in filename_norm) or ("policy" in filename_norm)
                )
                if not is_policy_candidate:
                    continue

                logger.info(f"  📋 Analizando póliza: {filename}")
                try:
                    policy_type = await self.extractor.detect_policy_type(
                        doc_data.get("ocr_result", {}),
                        "poliza_de_la_aseguradora"
                    )
                    if policy_type == "HDI_EN_MI_CASA":
                        logger.info("  🏠 HDI EN MI CASA detectado")
                        policy_document = filename
                        break
                except Exception as e:
                    logger.error(f"  ❌ Error detectando tipo de póliza: {e}")
                    policy_type = None

            if policy_type == "HDI_EN_MI_CASA":
                logger.info("  ⚡ Configurando reglas especiales para HDI EN MI CASA")
                if hasattr(self.extractor, "set_policy_context"):
                    self.extractor.set_policy_context(policy_type, case_id=case_id)
                if hasattr(self.consolidator, "set_policy_context"):
                    self.consolidator.set_policy_context(policy_type)
                # Forzar tipo de documento 'poliza_de_la_aseguradora' al archivo detectado para no omitirlo en FASE 2
                try:
                    if policy_document:
                        for d in ocr_results:
                            if d.get("filename") == policy_document:
                                prev = d.get("document_type")
                                d["document_type"] = "poliza_de_la_aseguradora"
                                logger.info(f"  🔧 Ajuste de tipo: '{policy_document}': {prev} -> poliza_de_la_aseguradora")
                                break
                except Exception:
                    pass
                logger.info("HDI_DETECTION: Policy type detected: HDI_EN_MI_CASA")
            else:
                logger.info("  ✓ Póliza estándar o no detectada; reglas estándar")
        elif not run_policy_detection and policy_type:
            logger.info("♻️ Reprocesamiento: conservando tipo de póliza detectado previamente")
            if hasattr(self.extractor, "set_policy_context"):
                self.extractor.set_policy_context(policy_type, case_id=case_id)
            if hasattr(self.consolidator, "set_policy_context"):
                self.consolidator.set_policy_context(policy_type)
        elif not run_policy_detection:
            logger.info("♻️ Reprocesamiento: no se solicitaron cambios en la detección de póliza")
        else:
            logger.info("  ℹ️ Detección HDI deshabilitada por feature flag")

        if self.cache_manager and policy_type:
            case_data["policy_type"] = policy_type
            self.cache_manager.save_case_index(case_id, case_data)

        # ============================================
        # FASE 2: Extracción con IA
        # ============================================
        logger.info("\n🔍 FASE 2: Extracción de campos con IA")
        logger.info("-" * 40)
        previous_extractions = case_data.get("extraction_results") or []
        run_extraction = wants("reprocess_extraction") or not previous_extractions
        extractions: List[DocumentExtraction] = []

        if not run_extraction:
            try:
                extractions = [DocumentExtraction.model_validate(item) for item in previous_extractions]
            except Exception as e:
                logger.warning(f"No se pudieron reconstruir las extracciones previas ({e}); se reprocesarán")
                run_extraction = True

        if run_extraction:
            if self.cancellation_check and await self.cancellation_check():
                await self.cleanup_on_cancel()
                raise asyncio.CancelledError("Proceso cancelado durante fase 2")

            if self.progress_callback:
                self.progress_callback("Extrayendo campos con inteligencia artificial...", 35)

            if self.progress_emitter:
                self.progress_emitter.emit("extract", "started", message="Extrayendo campos con IA")

            if self.guided_mode:
                logger.info("  🛡️ Usando extracción guiada con restricciones documento-campo")
                extractions = []
                from fraud_scorer.settings import ExtractionConfig
                target_types = set(ExtractionConfig.EXTRACTION_TARGET_TYPES)

                for doc_data in ocr_results:
                    doc_type = doc_data.get("document_type")
                    if not doc_type and hasattr(self.extractor, '_detect_document_type'):
                        doc_type = self.extractor._detect_document_type(
                            doc_data.get("ocr_result", {}),
                            doc_data["filename"],
                        )
                        doc_data["document_type"] = doc_type

                    canonical_type = doc_type or "otro"
                    if canonical_type not in target_types:
                        logger.info(
                            f"  ⏭️  Omitido (no objetivo): {doc_data['filename']} (tipo: {canonical_type})"
                        )
                        continue

                    extraction = await self.extractor.extract_from_document_guided(
                        content=doc_data["ocr_result"],
                        document_name=doc_data["filename"],
                        document_type=doc_type or "otro",
                        route=self.extraction_mode if self.extraction_mode != "auto" else "ocr_text",
                    )
                    if extraction:
                        extractions.append(extraction)
            else:
                extractions = await self.extractor.extract_from_documents_batch(
                    documents=ocr_results,
                    parallel_limit=3,
                )

            if self.cache_manager:
                try:
                    case_data["extraction_results"] = [ex.model_dump() for ex in extractions]
                    self.cache_manager.save_case_index(case_id, case_data)
                except Exception as e:
                    logger.warning(f"No se pudieron guardar extracciones previas: {e}")
        else:
            logger.info("♻️ Reprocesamiento: reutilizando extracciones previas")
            if self.progress_callback:
                self.progress_callback("Extracciones previas reutilizadas", 45)
            if self.progress_emitter:
                self.progress_emitter.emit("extract", "done", message="Extracciones reutilizadas")

        for extraction in extractions:
            fields_found = sum(1 for v in extraction.extracted_fields.values() if v is not None)
            logger.info(f"  ✓ {extraction.source_document}: {fields_found} campos extraídos")

        logger.info(f"✓ Extracción lista: {len(extractions)} documentos")

        if run_extraction and self.progress_emitter:
            self.progress_emitter.emit("extract", "done", message="Extracción completada")

        # ============================================
        # FASE 3: Consolidación con IA
        # ============================================
        logger.info("\n🧠 FASE 3: Consolidación inteligente")
        logger.info("-" * 40)
        
        previous_consolidated = case_data.get("consolidated_data")
        run_consolidation = wants("reprocess_consolidation") or not previous_consolidated
        consolidated: ConsolidatedExtraction

        if not run_consolidation and previous_consolidated:
            try:
                consolidated = ConsolidatedExtraction.model_validate(previous_consolidated)
            except Exception as e:
                logger.warning(f"No se pudo reconstruir la consolidación previa ({e}); se reprocesará")
                run_consolidation = True

        if run_consolidation:
            if self.cancellation_check and await self.cancellation_check():
                await self.cleanup_on_cancel()
                raise asyncio.CancelledError("Proceso cancelado durante fase 3")

            if self.progress_callback:
                self.progress_callback("Consolidando información con IA...", 55)

            if self.progress_emitter:
                self.progress_emitter.emit("consolidate", "started", message="Consolidando información")

            consolidated = await self.consolidator.consolidate_extractions(
                extractions=extractions,
                case_id=case_id,
                use_advanced_reasoning=True,
                guided_mode=self.guided_mode,
            )

            if self.cache_manager:
                try:
                    case_data["consolidated_data"] = consolidated.model_dump()
                    self.cache_manager.save_case_index(case_id, case_data)
                except Exception as e:
                    logger.warning(f"No se pudo guardar consolidación previa: {e}")
        else:
            logger.info("♻️ Reprocesamiento: reutilizando consolidación previa")

            if self.progress_callback:
                self.progress_callback("Consolidación previa reutilizada", 65)
            if self.progress_emitter:
                self.progress_emitter.emit("consolidate", "done", message="Consolidación reutilizada")

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
        insured_name_from_data = fields_dict.get("nombre_asegurado") or "Desconocido"
        claim_number_from_data = fields_dict.get("numero_siniestro") or f"SINIESTRO_{case_id}"
        logger.info(f"✓ Datos para organización: {insured_name_from_data} - {claim_number_from_data}")
        
        case_data["insured_name"] = insured_name_from_data
        case_data["claim_number"] = claim_number_from_data
        case_data["status"] = "processed"
        if self.cache_manager:
            self.cache_manager.save_case_index(case_id, case_data)

        if run_consolidation and self.cache_manager:
            logger.info("📁 Reorganizando estructura de cache...")
            self.cache_manager.reorganize_cache_for_case(case_id, insured_name_from_data, claim_number_from_data)
            logger.info("✓ Cache reorganizado con nomenclatura consistente")
            try:
                self.cache_manager.cleanup_shards()
            except Exception as e:
                logger.warning(f"cleanup_shards falló: {e}")

        if run_consolidation and self.progress_emitter:
            self.progress_emitter.emit("consolidate", "done", message="Consolidación completada")

        # ============================================
        # FASE 3.5: Análisis de Fraude por Documento (opcional)
        # ============================================
        previous_fraud_raw = case_data.get("fraud_analyses") or []
        previous_fraud = self._hydrate_fraud_results(previous_fraud_raw)
        excluded_fraud_types = {"poliza_de_la_aseguradora", "reporte_gps"}
        if previous_fraud:
            filtered_prev = [f for f in previous_fraud if f.document_type not in excluded_fraud_types]
            if len(filtered_prev) != len(previous_fraud):
                logger.info(
                    "♻️ Se removieron %d análisis previamente almacenados por pertenecer a tipos excluidos",
                    len(previous_fraud) - len(filtered_prev),
                )
            previous_fraud = filtered_prev
        run_fraud = self.enable_fraud and (wants("reprocess_fraud") or not previous_fraud)
        fraud_analyses: List[FraudAnalysisResult] = []
        if self.enable_fraud and run_fraud:
            logger.info("\n🔎 FASE 3.5: Análisis de fraude por documento")
            logger.info("-" * 40)
            try:
                # Emitir evento de inicio (no afecta ETA)
                if self.progress_emitter:
                    self.progress_emitter.emit("analyze", "started", message="Analizando documentos para fraude")

                analyzer = FraudAnalyzer()
                guide_manager = FraudGuideManager()
                # Mapear OCR por filename
                ocr_map = {d.get("filename"): (d.get("ocr_result") or {}) for d in ocr_results}
                extractions_by_name = {ex.source_document: ex for ex in extractions}
                docs_for_analysis: List[Dict[str, Any]] = []
                eligible_count = 0
                skipped_no_guide = 0
                skipped_excluded = 0
                skipped_name_collision = 0
                for doc_data in ocr_results:
                    try:
                        name = doc_data.get("filename")
                        if not name:
                            continue

                        ocr_dict = ocr_map.get(name, {})
                        doc_type = (doc_data.get("document_type") or "").strip() or "otro"

                        guide = guide_manager.get_guide(doc_type)
                        if not guide:
                            skipped_no_guide += 1
                            continue

                        canonical_type = guide.document_type or doc_type
                        if canonical_type in excluded_fraud_types:
                            skipped_excluded += 1
                            logger.info(
                                "⏭️ Documento excluido del análisis de fraude: %s (tipo=%s)",
                                name,
                                canonical_type,
                            )
                            continue
                        if canonical_type and canonical_type != doc_type:
                            doc_type = canonical_type
                            doc_data["document_type"] = canonical_type

                        eligible_count += 1

                        extraction_obj = extractions_by_name.get(name)
                        if not extraction_obj:
                            extraction_obj = DocumentExtraction(
                                source_document=name,
                                document_type=doc_type,
                                extracted_fields={},
                                extraction_metadata={},
                            )
                        elif extraction_obj.document_type != doc_type:
                            extraction_obj = extraction_obj.copy(update={"document_type": doc_type})
                        extractions_by_name[name] = extraction_obj

                        doc_id = doc_ids_by_name.get(name)

                        # Si no se encuentra por nombre, buscar por hash o en otros casos
                        if not doc_id:
                            logger.debug(f"Buscando document_id alternativo para {name}")

                            # Buscar en caché de hashes
                            doc_hash = None
                            if self.cache_manager:
                                try:
                                    case_index = self.cache_manager.get_case_index(case_id)
                                    if case_index and 'document_hashes' in case_index:
                                        for doc_path, hash_val in case_index['document_hashes'].items():
                                            if name in doc_path or Path(doc_path).name == name:
                                                doc_hash = hash_val
                                                break
                                except Exception:
                                    pass

                            # Buscar por hash en BD
                            if doc_hash:
                                with get_conn() as conn:
                                    row = conn.execute(
                                        "SELECT id FROM documents WHERE file_hash = ? LIMIT 1",
                                        (doc_hash,)
                                    ).fetchone()
                                    if row:
                                        doc_id = row['id']
                                        logger.info(f"Document_id encontrado por hash para {name}: {doc_id}")

                            # Búsqueda ampliada: buscar en cualquier caso con el mismo nombre
                            if not doc_id:
                                with get_conn() as conn:
                                    row = conn.execute(
                                        "SELECT id, case_id, created_at FROM documents WHERE filename = ? ORDER BY created_at DESC LIMIT 1",
                                        (name,)
                                    ).fetchone()
                                if row:
                                    if row['case_id'] == case_id:
                                        doc_id = row['id']
                                        logger.info(
                                            "Document_id encontrado por nombre en el mismo caso para %s: %s",
                                            name,
                                            doc_id,
                                        )
                                    else:
                                        skipped_name_collision += 1
                                        logger.error(
                                            "❌ Nombre de documento %s coincide con caso %s. Se omite para evitar colisión (document_id=%s)",
                                            name,
                                            row['case_id'],
                                            row['id'],
                                        )
                                        continue

                        if not doc_id:
                            logger.warning(
                                "No se encontró document_id en DB para %s (%s); se omite del análisis de fraude",
                                name,
                                case_id,
                            )
                            continue
                        docs_for_analysis.append({
                            "id": doc_id,
                            "name": name,
                            "type": doc_type,
                            "ocr": ocr_dict,
                            "extraction": extraction_obj,
                        })
                    except Exception:
                        continue

                if skipped_no_guide:
                    logger.info(f"ℹ️ Documentos omitidos por no tener guía: {skipped_no_guide}")
                if skipped_excluded:
                    logger.info(f"ℹ️ Documentos omitidos por regla de exclusión: {skipped_excluded}")
                if skipped_name_collision:
                    logger.warning(f"⚠️ Documentos omitidos por colisión de nombre en otros casos: {skipped_name_collision}")

                if docs_for_analysis:
                    # Preparar capa de datos unificada con la información más reciente
                    snapshot = dict(case_data)
                    snapshot.setdefault("case_id", case_id)
                    snapshot.setdefault("claim_number", claim_number_from_data)
                    snapshot.setdefault("insured_name", insured_name_from_data)
                    # Preferimos `consolidated_fields` (ver BETTER_PRACTICES §13); tolera `fields` en datos antiguos.
                    field_obj = getattr(consolidated, "consolidated_fields", None) or getattr(consolidated, "fields", None)
                    if field_obj:
                        if hasattr(field_obj, "model_dump"):
                            consolidated_dump = field_obj.model_dump()
                        elif hasattr(field_obj, "dict"):
                            consolidated_dump = field_obj.dict()
                        else:
                            consolidated_dump = getattr(field_obj, "__dict__", {})
                    else:
                        consolidated_dump = {}
                        logger.warning("No se encontraron campos consolidados en el objeto ConsolidatedExtraction; se usará un dict vacío.")
                    snapshot["consolidated_data"] = {"consolidated_fields": consolidated_dump}
                    snapshot["extraction_results"] = [
                        ext.model_dump() if hasattr(ext, "model_dump") else (
                            ext.dict() if hasattr(ext, "dict") else {
                                "source_document": getattr(ext, "source_document", None),
                                "document_type": getattr(ext, "document_type", None),
                                "extracted_fields": getattr(ext, "extracted_fields", {}),
                            }
                        )
                        for ext in extractions
                    ]
                    data_layer = UnifiedDataLayer(snapshot, extractions=extractions)
                    fraud_analyses = await analyzer.analyze_batch(
                        documents=docs_for_analysis,
                        case_id=case_id,
                        parallel_limit=3,
                        context=data_layer.build_case_context(),
                        data_layer=data_layer,
                    )
                    logger.info(f"✓ Análisis de fraude completado: {len(fraud_analyses)} documentos analizados (elegibles: {eligible_count})")
                else:
                    logger.info("ℹ️ No hay documentos elegibles para análisis de fraude")

                if self.progress_emitter:
                    self.progress_emitter.emit("analyze", "done", message="Análisis de fraude completado")
            except Exception as e:
                logger.warning(f"⚠️ Fase de fraude falló o fue omitida: {e}")
            if self.cache_manager:
                try:
                    case_data["fraud_analyses"] = [
                        getattr(a, "model_dump", lambda: a)() for a in (fraud_analyses or [])
                    ]
                    self.cache_manager.save_case_index(case_id, case_data)
                except Exception as e:
                    logger.warning(f"No se pudieron guardar análisis de fraude: {e}")
        elif self.enable_fraud:
            logger.info("♻️ Reprocesamiento: reutilizando análisis de fraude previos")
            fraud_analyses = previous_fraud
            if self.progress_emitter:
                self.progress_emitter.emit("analyze", "done", message="Análisis de fraude reutilizado")

        correlation_report: Optional["CorrelationReport"] = None
        if self.enable_fraud and fraud_analyses:
            correlation_report = self._run_correlation_analysis(
                case_id=case_id,
                consolidated=consolidated,
                extractions=extractions,
                fraud_analyses=fraud_analyses,
                case_data=case_data,
            )
            if correlation_report and self.cache_manager:
                try:
                    case_data["fraud_correlations"] = correlation_report.model_dump()
                    self.cache_manager.save_case_index(case_id, case_data)
                except Exception as exc:
                    logger.warning(f"No se pudieron guardar correlaciones en el índice: {exc}")
            if correlation_report:
                try:
                    save_correlation_findings(case_id, correlation_report.findings)
                except Exception as exc:
                    logger.warning(f"No se pudieron persistir correlaciones en DB: {exc}")

        # ============================================
        # FASE 4: Generación del reporte
        # ============================================
        logger.info("\n📝 FASE 4: Generación del reporte")
        logger.info("-" * 40)
        
        # Verificar cancelación antes de fase 4
        if self.cancellation_check and await self.cancellation_check():
            await self.cleanup_on_cancel()
            raise asyncio.CancelledError("Proceso cancelado durante fase 4")
        
        # Notificar generación de reporte
        if self.progress_callback:
            self.progress_callback("Generando reporte HTML y PDF...", 75)
        
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
        
        html_content = None
        if self.enable_fraud and fraud_analyses:
            try:
                fraud_gen = FraudReportGenerator(template_dir=self.report_generator.template_dir)
                # Metadata simple por documento para trazabilidad en template
                analyzed_names = {a.document_name for a in fraud_analyses if hasattr(a, 'document_name')}
                docs_meta = [
                    {"name": d.get("filename"), "type": d.get("document_type")}
                    for d in ocr_results if d.get("filename") in analyzed_names
                ]
                report_data = fraud_gen.prepare_fraud_report_data(
                    consolidated_data=consolidated,
                    fraud_analyses=fraud_analyses,
                    documents_metadata=docs_meta,
                    correlation_report=correlation_report,
                )
                auto_html = fraud_gen.render_html_template("report_template.html", report_data)
                manual_html = None
                if isinstance(case_data, dict):
                    manual_html = case_data.get("report_override_html")
                html_content = manual_html if isinstance(manual_html, str) and manual_html.strip() else auto_html
                # Guardar HTML
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                logger.info(f"✓ HTML generado (con fraude): {html_path}")
            except Exception as e:
                logger.warning(f"Fallo generando reporte con fraude: {e}. Usando plantilla estándar.")
                html_content = self.report_generator.generate_report(
                    consolidated_data=consolidated,
                    output_path=html_path,
                    insured_name=insured_name_from_data,
                    claim_number=claim_number_from_data
                )
        else:
            html_content = self.report_generator.generate_report(
                consolidated_data=consolidated,
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
            self.progress_callback("Finalizando procesamiento...", 90)
        
        # Emitir evento de finalización de reporte
        if self.progress_emitter:
            self.progress_emitter.emit("report", "done", message="Reporte generado")

        # ============================================
        # FASE 5: Guardar resultados y Organizar archivos
        # ============================================
        logger.info("\n💾 FASE 5: Guardar resultados y Organizar archivos")
        logger.info("-" * 40)
        processed_input_count = len(documents)
        if not processed_input_count:
            processed_input_count = len(docless_doc_names) or len(case_data.get("documents") or [])

        ocr_total = processed_input_count
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
            "documents_processed": processed_input_count,
            "policy_type": getattr(self.extractor, "policy_context", None),
            "extraction_results": [e.model_dump() for e in extractions],
            "consolidated_data": consolidated.model_dump(),
            "fraud_enabled": bool(self.enable_fraud),
            "fraud_analyses": [getattr(a, "model_dump", lambda: a)() for a in (fraud_analyses or [])],
            "processing_metrics": {
                "ocr_success_rate": f"{ocr_rate:.1%}",
                "extraction_success_rate": f"{extraction_rate:.1%}",
                "fields_completion_rate": f"{completion_rate:.1%}",
                "conflicts_resolved": len(consolidated.conflicts_resolved),
                "average_confidence": avg_confidence,
            },
            "report_path": str(html_path),  # Ruta del reporte HTML
            "pdf_path": str(pdf_path),      # Ruta del PDF
        }

        if correlation_report:
            results["correlation_report"] = correlation_report.model_dump()

        verification_payload = verify_case_artifacts(case_id)
        results["post_process_verification"] = verification_payload

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

    # (Método _analyze_fraud eliminado)


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parser de argumentos con soporte para modo guiado y organización de documentos."""
    p = argparse.ArgumentParser(description="Fraud Scorer v2.0 - Sistema de Análisis con IA")
    p.add_argument("folder", type=Path, nargs='?', help="Carpeta con documentos del caso")
    p.add_argument("--out", type=Path, default=Path("data/reports"), help="Carpeta de salida")
    p.add_argument("--title", help="Título del caso")
    p.add_argument("--debug", action="store_true", help="Modo debug con más logging")
    p.add_argument("--guided", action="store_true", default=True, help="Activar modo guiado con restricciones documento-campo (activado por defecto)")
    p.add_argument("--mode", choices=["direct_ai", "ocr", "auto"], default="auto",
                  help="Modo de extracción: direct_ai (IA directa), ocr (OCR primero), auto (automático)")
    p.add_argument("--purge-case", metavar="CASE_ID", help="Limpia artefactos del caso especificado")
    p.add_argument("--purge-orphans", action="store_true", help="Elimina archivos sin entrada en DB")
    
    # Argumentos de organización de documentos
    p.add_argument("--organize-only", action="store_true", 
                  help="Solo ejecuta Fase A de organización (clasificación y staging)")
    p.add_argument("--organize-first", action="store_true",
                  help="Ejecuta organización completa (Fase A + B) antes del pipeline normal")
    p.add_argument("--skip-llm-classification", action="store_true",
                  help="Solo usar heurísticas para clasificación, sin LLM")
    p.add_argument("--extract-all-fields", action="store_true",
                  help="Extraer todos los campos en Fase B (sin restricciones)")
    # Opciones de fraude
    p.add_argument("--no-fraud", action="store_true", help="Desactiva el análisis de fraude por documento para esta ejecución")
    
    return p.parse_args(argv)


async def main(argv: List[str]) -> None:
    """Función principal con soporte para organización de documentos."""
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
    
    # ==== MANEJAR MODO DE ORGANIZACIÓN ====
    if args.organize_only or args.organize_first:
        logger.info("\n" + "=" * 60)
        logger.info("📂 MODO DE ORGANIZACIÓN DE DOCUMENTOS")
        logger.info("=" * 60)
        
        organizer = DocumentOrganizer()
        
        try:
            # Ejecutar Fase A (clasificación y staging)
            logger.info("\n🔍 FASE A: Clasificación y Staging")
            logger.info("-" * 40)
            staging_result = await organizer.organize_documents_phase_a(
                input_folder=args.folder,
                use_llm_fallback=not args.skip_llm_classification
            )
            
            if not staging_result["success"]:
                logger.error(f"❌ Error en Fase A: {staging_result.get('error')}")
                sys.exit(1)
            
            logger.info(f"✅ Fase A completada: {staging_result['metrics']['total_files']} archivos procesados")
            logger.info(f"📁 Carpeta de staging: {staging_result['staging_path']}")
            
            # Si es --organize-only, terminar aquí
            if args.organize_only:
                logger.info("\n" + "=" * 60)
                logger.info("✅ ORGANIZACIÓN FASE A COMPLETADA")
                logger.info("=" * 60)
                logger.info(f"📊 Resumen:")
                logger.info(f"  - Archivos procesados: {staging_result['metrics']['total_files']}")
                logger.info(f"  - Archivos clasificados: {staging_result['metrics']['classified']}")
                logger.info(f"  - Archivos no soportados: {staging_result['metrics']['unsupported']}")
                logger.info(f"  - Documentos únicos: {len(staging_result['metrics']['documents_by_type'])}")
                logger.info(f"  - Staging path: {staging_result['staging_path']}")
                return
            
            # Si es --organize-first, continuar con Fase B
            if args.organize_first:
                logger.info("\n🧠 FASE B: Extracción y Renombrado Final")
                logger.info("-" * 40)
                
                # Usar modo de extracción según argumentos
                extraction_mode = "all" if args.extract_all_fields else "key_fields_only"
                
                final_result = await organizer.organize_documents_phase_b(
                    staging_folder=Path(staging_result['staging_path']),
                    extraction_mode=extraction_mode
                )
                
                if not final_result["success"]:
                    logger.error(f"❌ Error en Fase B: {final_result.get('error')}")
                    # Aún así continuar con el pipeline si es posible
                else:
                    logger.info(f"✅ Fase B completada")
                    logger.info(f"📁 Carpeta final: {final_result['final_path']}")
                    logger.info(f"📄 Asegurado: {final_result.get('insured_name', 'N/A')}")
                    logger.info(f"📋 Siniestro: {final_result.get('claim_number', 'N/A')}")
                    
                    # Actualizar la carpeta a procesar para el pipeline normal
                    args.folder = Path(final_result['final_path'])
                    logger.info(f"\n🔄 Continuando con pipeline normal usando: {args.folder}")
        
        except Exception as e:
            logger.error(f"❌ Error en organización: {e}", exc_info=True)
            if args.organize_only:
                sys.exit(1)
            # Si es --organize-first, continuar con la carpeta original
            logger.warning("⚠️ Continuando con pipeline normal usando carpeta original")
    
    # ==== PIPELINE NORMAL DE ANÁLISIS ====
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

    system = FraudAnalysisSystemV2(
        guided_mode=args.guided,  # Ya tiene default=True en argparse
        extraction_mode=args.mode if hasattr(args, 'mode') else "auto",
        enable_fraud=not getattr(args, 'no_fraud', False),
    )

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
