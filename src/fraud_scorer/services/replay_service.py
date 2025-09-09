# src/fraud_scorer/services/replay_service.py

import logging
import shutil
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from ..storage.ocr_cache import OCRCacheManager
from ..storage.db import get_conn
from ..pipelines.data_flow import build_docs_for_template_from_db
from ..processors.ai.ai_field_extractor import AIFieldExtractor
from ..processors.ai.ai_consolidator import AIConsolidator
from ..templates.ai_report_generator import AIReportGenerator

logger = logging.getLogger(__name__)

class ReplayService:
    """
    Contiene toda la lógica de negocio para el sistema de Replay.
    Es utilizado tanto por la API web como podría serlo por la CLI.
    """
    def __init__(self):
        self.cache_manager = OCRCacheManager()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché (FS) y métricas DB complementarias.
        - FS: refleja borrados manuales (usa OCRCacheManager.get_cache_stats())
        - DB: agrega conteos de tablas para diagnóstico (no rompe UI)
        """
        try:
            fs_stats = self.cache_manager.get_cache_stats()  # {'total_cases','total_cached_files','cache_size_mb', ...}
            # Mapear a los nombres usados por la UI para no romper nada
            out = {
                "case_count": fs_stats.get("total_cases", 0),
                "file_count": fs_stats.get("total_cached_files", 0),
                "total_size_mb": fs_stats.get("cache_size_mb", 0.0),
                "cache_directory": fs_stats.get("cache_directory", str(self.cache_manager.cache_dir))
            }

            # Métricas DB (complemento)
            try:
                with get_conn() as conn:
                    out["db_cases"] = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                    out["db_documents"] = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                    out["db_ocr_results"] = conn.execute("SELECT COUNT(*) FROM ocr_results").fetchone()[0]
                    out["db_extracted_data"] = conn.execute("SELECT COUNT(*) FROM extracted_data").fetchone()[0]
                    # Agregar métricas globales de cache_stats si existen
                    row = conn.execute("SELECT ocr_hits, ocr_misses, bytes_saved, ms_saved FROM cache_stats WHERE scope='global'").fetchone()
                    if row:
                        out["ocr_hits"] = row["ocr_hits"]
                        out["ocr_misses"] = row["ocr_misses"]
                        out["bytes_saved"] = row["bytes_saved"]
                        out["ms_saved"] = row["ms_saved"]
            except Exception:
                pass
            return out
        except Exception as e:
            logger.error(f"Error al calcular estadísticas del caché: {e}")
            return {"case_count": 0, "file_count": 0, "total_size_mb": 0}

    def list_available_cases(self) -> List[Dict[str, Any]]:
        """Lista los casos disponibles desde la base de datos o el índice de caché."""
        # Usar el método existente del cache_manager que ya funciona
        cases = self.cache_manager.list_cached_cases()
        result = []
        
        for case in cases:
            # Verificar si el caso fue procesado (tiene fecha de procesamiento)
            is_processed = bool(case.get("processed_at", ""))
            
            result.append({
                "case_id": case["case_id"],
                "title": case.get("case_title", case["case_id"]),
                "created_at": case.get("processed_at", ""),
                "document_count": case.get("total_documents", 0),
                "is_processed": is_processed,
                "processed_at": case.get("processed_at", "N/A")
            })
        
        return result

    async def process_replay(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Función principal que ejecuta el pipeline de replay.
        Delega a _core_replay_processing para la lógica centralizada.
        """
        return await self._core_replay_processing(config)

    def cleanup_db_orphans(self) -> dict:
        """Elimina orfandad en DB para mantener consistencia tras cambios manuales en FS."""
        from ..storage.db import get_conn
        stats = {}
        with get_conn() as conn:
            stats['orphan_extracted_before'] = conn.execute(
                "SELECT COUNT(*) FROM extracted_data WHERE document_id NOT IN (SELECT id FROM documents)"
            ).fetchone()[0]
            stats['orphan_runs_before'] = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE case_id NOT IN (SELECT case_id FROM cases)"
            ).fetchone()[0]
            stats['orphan_ocr_before'] = conn.execute(
                "SELECT COUNT(*) FROM ocr_results WHERE document_id NOT IN (SELECT id FROM documents)"
            ).fetchone()[0]

            conn.execute("DELETE FROM extracted_data WHERE document_id NOT IN (SELECT id FROM documents)")
            conn.execute("DELETE FROM runs WHERE case_id NOT IN (SELECT case_id FROM cases)")
            conn.execute("DELETE FROM ocr_results WHERE document_id NOT IN (SELECT id FROM documents)")

            stats['orphan_extracted_after'] = conn.execute(
                "SELECT COUNT(*) FROM extracted_data WHERE document_id NOT IN (SELECT id FROM documents)"
            ).fetchone()[0]
            stats['orphan_runs_after'] = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE case_id NOT IN (SELECT case_id FROM cases)"
            ).fetchone()[0]
            stats['orphan_ocr_after'] = conn.execute(
                "SELECT COUNT(*) FROM ocr_results WHERE document_id NOT IN (SELECT id FROM documents)"
            ).fetchone()[0]

        return stats

    async def deep_purge_case(self, case_id: str) -> bool:
        """
        Limpieza profunda de un caso: artefactos de FS + filas en DB (cases y cascada).
        """
        try:
            # 1) Limpiar artefactos de FS
            await self.purge_case(case_id)

            # 2) Eliminar caso en DB (cascada elimina documentos, ocr_results, extracted_data, runs)
            from ..storage.db import get_conn
            with get_conn() as conn:
                conn.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))

            # 3) Limpiar métricas de cache del caso (ya lo hace purge_case, pero reforzamos)
            from ..storage.db import reset_cache_stats
            reset_cache_stats(case_id)

            logger.info(f"✅ Deep purge completado para {case_id}")
            return True
        except Exception as e:
            logger.error(f"Error en deep purge {case_id}: {e}")
            return False
    
    async def _core_replay_processing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lógica centralizada del sistema de replay.
        Esta función es usada tanto por la interfaz web como por la terminal.
        
        Args:
            config: Diccionario con las opciones del replay incluyendo:
                - case_id: ID del caso a procesar
                - use_ai: Boolean para usar AI o sistema legacy
                - output_dir: Directorio de salida para reportes
                - regenerate_report: Boolean para generar reportes
                - api_key: API key de OpenAI (opcional)
                - model: Modelo de AI a usar
                - temperature: Temperatura del modelo
                - per_doc: Boolean para análisis por documento
        
        Returns:
            Dict con los resultados del procesamiento
        """
        case_id = config["case_id"]
        logger.info(f"Iniciando replay para el caso: {case_id} con config: {config}")

        # Obtener información del caso
        case_index = self.cache_manager.get_case_index(case_id)
        if not case_index:
            raise ValueError(f"No se encontró información del caso {case_id}")

        # Cargar resultados OCR del cache
        ocr_results = []
        for doc_path in case_index.get('documents', []):
            doc_path = Path(doc_path)
            if self.cache_manager.has_cache(doc_path, case_id=None):
                ocr_result = self.cache_manager.get_cache(doc_path)
                ocr_results.append({
                    'filename': doc_path.name,
                    'ocr_result': ocr_result,
                    'document_type': None
                })
            else:
                logger.warning(f"No hay cache para {doc_path.name}")

        if not ocr_results:
            raise RuntimeError("No se encontraron resultados OCR en cache")

        # Procesar según las opciones
        if config.get('use_ai'):
            return await self._process_with_ai(
                ocr_results=ocr_results,
                case_id=case_id,
                options=config
            )
        else:
            return await self._process_legacy(
                ocr_results=ocr_results,
                case_id=case_id,
                options=config
            )

    def _clean_existing_files(self, case_id: str, output_path: Path) -> None:
        """
        Elimina archivos existentes para un case_id antes de generar nuevos.
        
        Args:
            case_id: ID del caso
            output_path: Directorio donde buscar archivos existentes
        """
        patterns_to_clean = [
            f"INF-{case_id}.html",  # Formato antiguo
            f"INF-{case_id}.pdf",   # Formato antiguo
            f"INF-*-*.html",        # Formato nuevo
            f"INF-*-*.pdf",         # Formato nuevo
            f"replay_{case_id}_*.json",
            f"{case_id}_*.html",
            f"{case_id}_*.pdf"
        ]
        
        for pattern in patterns_to_clean:
            for file in output_path.glob(pattern):
                try:
                    file.unlink()
                    logger.info(f"Archivo existente eliminado: {file.name}")
                except Exception as e:
                    logger.warning(f"No se pudo eliminar {file.name}: {e}")
    
    async def _process_with_ai(
        self,
        ocr_results: List[Dict],
        case_id: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa con el sistema de IA usando los datos cacheados
        """
        # Asegurar que el directorio de salida existe
        output_dir = options.get('output_dir')
        if output_dir is None:
            output_dir = 'data/reports'
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio de salida: {output_path.absolute()}")
        
        # Obtener información del caso para nomenclatura
        case_index = self.cache_manager.get_case_index(case_id)
        if case_index:
            insured_name = case_index.get('insured_name', 'DESCONOCIDO')
            claim_number = case_index.get('claim_number', case_id)
        else:
            insured_name = 'DESCONOCIDO'
            claim_number = case_id
        
        # Limpiar archivos existentes antes de generar nuevos
        self._clean_existing_files(case_id, output_path)

        # 1) Resolver API key (UI/CLI > entorno)
        api_key = (options.get("api_key") or os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "No se encontró OPENAI_API_KEY. Cárgala desde .env o introdúcela en la UI/CLI."
            )
        # Exportar al entorno por si algún componente la lee de os.getenv
        if os.getenv("OPENAI_API_KEY") != api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # 2) Resolver config de IA (modelo/temperatura)
        model = options.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        try:
            temperature = float(options.get("temperature", 0.1))
        except Exception:
            temperature = 0.1

        # 3) Inicializar componentes de IA
        extractor = AIFieldExtractor(api_key=api_key)
        consolidator = AIConsolidator(api_key=api_key)
        report_generator = AIReportGenerator()

        # Fase 1: Extracción
        logger.info("Extrayendo campos con IA...")
        extractions = await extractor.extract_from_documents_batch(
            documents=ocr_results,
            parallel_limit=3
        )

        # Fase 2: Consolidación
        logger.info("Consolidando datos...")
        consolidated = await consolidator.consolidate_extractions(
            extractions=extractions,
            case_id=case_id,
            use_advanced_reasoning=True
        )

        # Fase de análisis de fraude eliminada

        # Fase 4: Generar reporte si se solicita
        if options.get('regenerate_report', True):
            logger.info("Generando reporte...")
            
            # Sanitizar nombres para uso en archivos
            import re
            def sanitize_filename(name):
                if not name:
                    return "SIN_NOMBRE"
                return re.sub(r'[^a-zA-Z0-9_.-]+', '_', name).strip('_')
            
            s_insured = sanitize_filename(insured_name)
            s_claim = sanitize_filename(claim_number)
            
            html_path = output_path / f"INF-{s_insured}-{s_claim}.html"
            html_content = report_generator.generate_report(
                consolidated_data=consolidated,
                output_path=html_path
            )

            # Intentar generar PDF
            pdf_path = output_path / f"INF-{s_insured}-{s_claim}.pdf"
            report_generator.generate_pdf(html_content, pdf_path)

        # Preparar resultados
        try:
            consolidated_dict = consolidated.model_dump()
        except Exception:
            try:
                consolidated_dict = consolidated.dict()
            except Exception:
                consolidated_dict = consolidated

        try:
            extractions_list = [e.model_dump() for e in extractions]
        except Exception:
            try:
                extractions_list = [e.dict() for e in extractions]
            except Exception:
                extractions_list = extractions

        results = {
            "case_id": case_id,
            "replay_date": datetime.now().isoformat(),
            "options_used": {**options, "api_key": "***redacted***"},
            "extraction_results": extractions_list,
            "consolidated_data": consolidated_dict,
            "output_path": str(output_path)
        }

        # Guardar JSON de resultados
        json_path = output_path / f"replay_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Replay completado. Resultados en: {output_path}")
        return results

    async def _process_legacy(
        self,
        ocr_results: List[Dict],
        case_id: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa con el sistema legacy
        """
        logger.info("Procesamiento legacy no implementado aún")
        return {"status": "legacy_not_implemented"}

    def clear_cache(self, cases_to_delete: List[str]) -> Dict[str, Any]:
        """Limpia el caché para una lista de case_id o para todos si 'all' está en la lista."""
        if "all" in cases_to_delete:
            # Implementar limpieza total
            try:
                base_dir = Path(getattr(self.cache_manager, "cache_dir", "data/ocr_cache"))
                if base_dir.exists():
                    shutil.rmtree(base_dir)
                base_dir.mkdir(parents=True, exist_ok=True)
                return {"status": "success", "message": "Todo el caché ha sido limpiado."}
            except Exception as e:
                return {"status": "error", "message": f"Error limpiando todo el caché: {e}"}
        
        cleared_cases = []
        errors = []
        for case_id in cases_to_delete:
            try:
                # Cargar índice del caso primero (antes de eliminarlo) para conocer artefactos
                case_index = self.cache_manager.get_case_index(case_id)
                
                # Limpiar archivos de cache asociados (shards por hash)
                if case_index and "cache_files" in case_index:
                    for doc_path_str in case_index["cache_files"]:
                        cache_path = self.cache_manager._get_cache_path(Path(doc_path_str))
                        if cache_path.exists():
                            cache_path.unlink()

                # Limpiar carpeta reorganizada si existe (Nombre - Reclamo)
                try:
                    insured = (case_index or {}).get('insured_name') or ""
                    claim = (case_index or {}).get('claim_number') or ""
                    if insured or claim:
                        s_insured = self.cache_manager._sanitize_filename(insured)
                        s_claim = self.cache_manager._sanitize_filename(claim or case_id)
                        case_folder = self.cache_manager.cache_dir / f"{s_insured} - {s_claim}"
                        if case_folder.exists():
                            shutil.rmtree(case_folder)
                except Exception:
                    pass

                # Limpiar índice del caso (después)
                index_path = self.cache_manager.index_dir / f"{case_id}.json"
                if index_path.exists():
                    index_path.unlink()
                
                cleared_cases.append(case_id)
            except Exception as e:
                errors.append({"case_id": case_id, "error": str(e)})

        return {
            "status": "success",
            "message": f"Limpieza completada. Casos limpiados: {len(cleared_cases)}.",
            "cleared": cleared_cases,
            "errors": errors
        }
    
    async def purge_case(self, case_id: str) -> bool:
        """
        Limpia artefactos del caso especificado sin tocar métricas globales.
        """
        try:
            # Limpiar archivos del caso en el caché
            case_index = self.cache_manager.get_case_index(case_id)
            if case_index and "cache_files" in case_index:
                for doc_path_str in case_index["cache_files"]:
                    cache_path = self.cache_manager._get_cache_path(Path(doc_path_str))
                    if cache_path.exists():
                        cache_path.unlink()
            
            # Limpiar índice del caso
            index_path = self.cache_manager.index_dir / f"{case_id}.json"
            if index_path.exists():
                index_path.unlink()
            
            # Limpiar carpeta reorganizada si existe (Nombre - Reclamo)
            try:
                insured = (case_index or {}).get('insured_name') or ""
                claim = (case_index or {}).get('claim_number') or ""
                if insured or claim:
                    s_insured = self.cache_manager._sanitize_filename(insured)
                    s_claim = self.cache_manager._sanitize_filename(claim or case_id)
                    case_folder = self.cache_manager.cache_dir / f"{s_insured} - {s_claim}"
                    if case_folder.exists():
                        shutil.rmtree(case_folder)
            except Exception:
                pass
            
            # Limpiar archivos de status/progress
            base = os.getenv("FS_DATA_DIR", "data")
            status_file = Path(base) / "temp" / "pipeline_cache" / f"{case_id}.status.jsonl"
            if status_file.exists():
                status_file.unlink()
            
            # Resetear métricas del caso (pero no las globales)
            from ..storage.db import reset_cache_stats
            reset_cache_stats(case_id)
            
            logger.info(f"✅ Caso {case_id} purgado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error purgando caso {case_id}: {e}")
            return False
    
    async def purge_orphans(self) -> int:
        """
        Elimina archivos de caché sin entrada correspondiente en la DB.
        """
        orphan_count = 0
        
        try:
            # Obtener todos los casos conocidos de la DB
            with get_conn() as conn:
                rows = conn.execute("SELECT DISTINCT case_id FROM cases").fetchall()
                valid_cases = {row["case_id"] for row in rows}
            
            # Revisar archivos de índice
            for index_file in self.cache_manager.index_dir.glob("*.json"):
                case_id = index_file.stem
                if case_id not in valid_cases:
                    logger.info(f"Eliminando índice huérfano: {case_id}")
                    # Leer primero para conocer archivos asociados, luego eliminar el índice
                    case_data = None
                    try:
                        with open(index_file, 'r', encoding='utf-8') as f:
                            case_data = json.load(f)
                    except Exception:
                        case_data = None
                    
                    # Eliminar archivos de caché asociados (shards)
                    try:
                        for doc_path_str in (case_data or {}).get("cache_files", []):
                            cache_path = self.cache_manager._get_cache_path(Path(doc_path_str))
                            if cache_path.exists():
                                cache_path.unlink()
                                orphan_count += 1
                    except Exception:
                        pass
                    
                    # Eliminar carpeta reorganizada si existiera (Nombre - Reclamo)
                    try:
                        insured = (case_data or {}).get('insured_name') or ""
                        claim = (case_data or {}).get('claim_number') or ""
                        if insured or claim:
                            s_insured = self.cache_manager._sanitize_filename(insured)
                            s_claim = self.cache_manager._sanitize_filename(claim or case_id)
                            case_folder = self.cache_manager.cache_dir / f"{s_insured} - {s_claim}"
                            if case_folder.exists():
                                shutil.rmtree(case_folder)
                                orphan_count += 1
                    except Exception:
                        pass
                    
                    # Eliminar índice al final
                    try:
                        index_file.unlink()
                        orphan_count += 1
                    except Exception:
                        pass
            
            # Revisar carpetas reorganizadas
            for folder in self.cache_manager.cache_dir.iterdir():
                if folder.is_dir() and folder.name != "case_index":
                    # Extraer case_id del nombre de la carpeta si es posible
                    folder_has_valid_case = False
                    for case_id in valid_cases:
                        if case_id in folder.name:
                            folder_has_valid_case = True
                            break
                    
                    if not folder_has_valid_case and "-" in folder.name:
                        logger.info(f"Eliminando carpeta huérfana: {folder.name}")
                        shutil.rmtree(folder)
                        orphan_count += 1
            
            logger.info(f"✅ {orphan_count} archivos huérfanos eliminados")
            return orphan_count
            
        except Exception as e:
            logger.error(f"Error eliminando huérfanos: {e}")
            return orphan_count
