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
from ..processors.ai.document_analyzer import AIDocumentAnalyzer
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
        """Obtiene las estadísticas del caché que se mostrarán en el dashboard."""
        try:
            # Encuentra todos los archivos de índice de casos, que son la fuente de verdad
            case_files = list(self.cache_manager.index_dir.glob("*.json"))
            case_count = len(case_files)

            # Para los archivos y el tamaño, necesitamos leer los índices
            total_files = 0
            total_size_bytes = 0
            for case_file in case_files:
                with open(case_file, 'r') as f:
                    data = json.load(f)
                    # Sumamos los archivos cacheados en este caso
                    num_files = len(data.get("cache_files", []))
                    total_files += num_files
                    # Obtenemos el tamaño de cada archivo cacheado
                    for doc_path_str in data.get("cache_files", []):
                        cache_path = self.cache_manager._get_cache_path(Path(doc_path_str))
                        if cache_path.exists():
                            total_size_bytes += cache_path.stat().st_size
            
            total_size_mb = round(total_size_bytes / (1024 * 1024), 2)

            return {
                "case_count": case_count,
                "file_count": total_files,
                "total_size_mb": total_size_mb,
            }
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
            if self.cache_manager.has_cache(doc_path):
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
            f"INF-{case_id}.html",
            f"INF-{case_id}.pdf",
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
        output_path = Path(options.get('output_dir', 'data/reports'))
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio de salida: {output_path.absolute()}")
        
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
        analyzer = AIDocumentAnalyzer(api_key=api_key, model=model, temperature=temperature)

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

        # Fase 3: Análisis de fraude
        logger.info("Analizando fraude...")
        docs_for_analysis = []
        for extraction in extractions:
            docs_for_analysis.append({
                "document_type": getattr(extraction, "document_type", None),
                "key_value_pairs": getattr(extraction, "extracted_fields", {}) or {},
                "specific_fields": getattr(extraction, "extracted_fields", {}) or {},
                "raw_text": "",
                "entities": []
            })

        ai_analysis = await analyzer.analyze_claim_documents(docs_for_analysis)

        # Fase 4: Generar reporte si se solicita
        if options.get('regenerate_report', True):
            logger.info("Generando reporte...")
            html_path = output_path / f"INF-{case_id}.html"
            html_content = report_generator.generate_report(
                consolidated_data=consolidated,
                ai_analysis=ai_analysis,
                output_path=html_path
            )

            # Intentar generar PDF
            pdf_path = output_path / f"INF-{case_id}.pdf"
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
            "fraud_analysis": ai_analysis,
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
                # Limpiar índice del caso
                index_path = self.cache_manager.index_dir / f"{case_id}.json"
                if index_path.exists():
                    index_path.unlink()
                
                # Limpiar archivos de cache asociados
                case_index = self.cache_manager.get_case_index(case_id)
                if case_index and "cache_files" in case_index:
                    for doc_path_str in case_index["cache_files"]:
                        cache_path = self.cache_manager._get_cache_path(Path(doc_path_str))
                        if cache_path.exists():
                            cache_path.unlink()
                
                cleared_cases.append(case_id)
            except Exception as e:
                errors.append({"case_id": case_id, "error": str(e)})

        return {
            "status": "success",
            "message": f"Limpieza completada. Casos limpiados: {len(cleared_cases)}.",
            "cleared": cleared_cases,
            "errors": errors
        }