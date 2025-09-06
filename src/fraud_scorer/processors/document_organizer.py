"""
Sistema de organización de documentos en 2 fases
Fase A: Clasificación barata y staging
Fase B: Extracción guiada y renombrado final
"""

from __future__ import annotations

import os
import re
import json
import shutil
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime
import time

import PyPDF2
from openai import AsyncOpenAI

from fraud_scorer.settings import ExtractionConfig, DOCUMENT_PRIORITIES
from fraud_scorer.processors.document_classifier import DocumentClassifier, DocumentType
from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
from fraud_scorer.processors.ai.ai_field_extractor import AIFieldExtractor
from fraud_scorer.processors.ai.ai_consolidator import AIConsolidator
from fraud_scorer.storage.ocr_cache import (
    ensure_document_registered,
    try_get_cached_ocr,
    persist_ocr
)

logger = logging.getLogger(__name__)


class DocumentOrganizer:
    """
    Organizador de documentos con clasificación y extracción
    """
    
    def __init__(self):
        self.config = ExtractionConfig()
        self.classifier = DocumentClassifier()
        self.metrics = OrganizationMetrics()
        self._skip_ocr_sample = False  # Flag para tests
    
    async def organize_documents_phase_a(
        self,
        input_folder: Path,
        staging_base: Optional[Path] = None,
        use_llm_fallback: bool = True
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Fase A: Clasificación barata y organización en staging
        
        Args:
            input_folder: Carpeta con documentos a organizar
            staging_base: Carpeta base para staging (default: data/uploads/renombre_de_documentos)
            use_llm_fallback: Si usar LLM cuando la heurística falla
            
        Returns:
            - staging_folder: Ruta de la carpeta staging creada
            - mapping: Diccionario con el mapeo de archivos
        """
        start_time = time.time()
        
        # 1. Crear carpeta staging con timestamp
        if staging_base is None:
            from fraud_scorer.settings import STAGING_DIR
            staging_base = STAGING_DIR
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        staging_folder = staging_base / timestamp
        staging_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Iniciando Fase A - Staging en: {staging_folder}")
        
        # 2. Descubrir archivos soportados
        files = self._discover_supported_files(input_folder)
        logger.info(f"Encontrados {len(files)} archivos soportados")
        
        # 3. Clasificar y renombrar
        mapping = {
            "files": [],
            "metadata": {
                "timestamp": timestamp,
                "source_folder": str(input_folder),
                "total_files": len(files),
                "phase_a_completed": False,
                "phase_b_completed": False
            }
        }
        
        # Agrupar archivos para manejar subnumeración de guias_y_facturas
        guias_counter = 0
        
        for idx, file_path in enumerate(sorted(files), 1):
            try:
                logger.info(f"Procesando {idx}/{len(files)}: {file_path.name}")
                
                # Determinar ruta de procesamiento
                route = self._determine_route_for_file(file_path)
                
                # Obtener texto de muestra para clasificación
                sample_text = await self._get_sample_text(file_path, route)
                
                # Clasificar documento
                classification_start = time.time()
                doc_type, confidence, reasons = await self.classifier.classify(
                    sample_text,
                    file_path.name,
                    use_llm_fallback=use_llm_fallback
                )
                classification_time = int((time.time() - classification_start) * 1000)
                
                # Registrar métricas
                self.metrics.log_classification(
                    doc_type, 
                    confidence < 0.6 and use_llm_fallback,
                    classification_time
                )
                
                # Manejar subnumeración para guias_y_facturas
                if doc_type == DocumentType.GUIAS_Y_FACTURAS.value:
                    guias_counter += 1
                    # Detectar destinatario si es posible
                    destinatario = self._extract_destinatario(sample_text)
                    if destinatario:
                        doc_type_display = f"{doc_type}_{guias_counter}"
                        reasons.append(f"Destinatario: {destinatario}")
                    else:
                        doc_type_display = f"{doc_type}_{guias_counter}"
                else:
                    doc_type_display = doc_type
                
                # Obtener alias corto para el nombre de archivo
                from fraud_scorer.settings import CANONICAL_TO_ALIAS
                alias = CANONICAL_TO_ALIAS.get(doc_type, doc_type[:15])
                
                # Generar nombre nuevo
                from fraud_scorer.settings import FILE_NAMING_CONFIG
                route_label = FILE_NAMING_CONFIG["route_labels"].get(route, "UNK")
                base_name = self._sanitize_filename(file_path.stem[:50])
                new_name = f"{idx:03d}__{alias}__{route_label}__{base_name}{file_path.suffix}"
                new_path = staging_folder / new_name
                
                # Copiar archivo
                shutil.copy2(file_path, new_path)
                logger.debug(f"Copiado: {file_path.name} → {new_name}")
                
                # Registrar en mapping
                file_info = {
                    "index": idx,
                    "original": str(file_path),
                    "staged": str(new_path),
                    "document_type": doc_type,
                    "document_type_display": doc_type_display,
                    "route": route,
                    "confidence": confidence,
                    "reasons": reasons,
                    "classification_time_ms": classification_time,
                    "file_size": file_path.stat().st_size
                }
                
                if doc_type == DocumentType.GUIAS_Y_FACTURAS.value and destinatario:
                    file_info["destinatario"] = destinatario
                
                mapping["files"].append(file_info)
                
            except Exception as e:
                logger.error(f"Error procesando {file_path}: {e}")
                # Copiar a carpeta de errores
                error_folder = staging_folder / "_errores"
                error_folder.mkdir(exist_ok=True)
                shutil.copy2(file_path, error_folder / file_path.name)
                
                mapping["files"].append({
                    "index": idx,
                    "original": str(file_path),
                    "staged": str(error_folder / file_path.name),
                    "document_type": "error",
                    "route": "unknown",
                    "confidence": 0.0,
                    "reasons": [f"Error: {str(e)}"],
                    "error": str(e)
                })
        
        # 4. Manejar archivos no soportados
        unsupported = self._find_unsupported_files(input_folder)
        if unsupported:
            unsupported_folder = staging_folder / "_no_soportados"
            unsupported_folder.mkdir(exist_ok=True)
            
            for file_path in unsupported:
                shutil.copy2(file_path, unsupported_folder / file_path.name)
                logger.debug(f"Archivo no soportado: {file_path.name}")
            
            mapping["metadata"]["unsupported_files"] = len(unsupported)
        
        # 5. Guardar mapping.json
        mapping_file = staging_folder / "mapping.json"
        mapping["metadata"]["phase_a_completed"] = True
        mapping["metadata"]["phase_a_duration_ms"] = int((time.time() - start_time) * 1000)
        
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Fase A completada en {mapping['metadata']['phase_a_duration_ms']}ms")
        
        # 6. Generar resumen
        self._print_phase_a_summary(mapping)
        
        return staging_folder, mapping
    
    async def organize_documents_phase_b(
        self,
        staging_folder: Path,
        extract_all_fields: bool = False
    ) -> Path:
        """
        Fase B: Extracción guiada y renombrado final
        
        Args:
            staging_folder: Carpeta staging de Fase A
            extract_all_fields: Si False, solo extrae nombre_asegurado y numero_siniestro
            
        Returns:
            final_folder: Carpeta final renombrada
        """
        start_time = time.time()
        logger.info(f"Iniciando Fase B - Extracción desde: {staging_folder}")
        
        # 1. Cargar mapping
        mapping_file = staging_folder / "mapping.json"
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        
        if not mapping["metadata"].get("phase_a_completed"):
            raise ValueError("Fase A no completada en esta carpeta")
        
        # 2. Preparar extractores
        ocr_processor = AzureOCRProcessor()
        ai_extractor = AIFieldExtractor()
        consolidator = AIConsolidator()
        
        # Campos objetivo
        if extract_all_fields:
            target_fields = self.config.REQUIRED_FIELDS
        else:
            target_fields = ["nombre_asegurado", "numero_siniestro"]
        
        logger.info(f"Extrayendo campos: {target_fields}")
        
        # 3. Ejecutar extracción guiada
        extractions = []
        successful_extractions = 0
        
        valid_files = [f for f in mapping["files"] if f.get("document_type") != "error"]
        
        for i, file_info in enumerate(valid_files, 1):
            staged_path = Path(file_info["staged"])
            doc_type = file_info["document_type"]
            route = file_info["route"]
            
            logger.info(f"Extrayendo {i}/{len(valid_files)}: {staged_path.name}")
            
            try:
                # Obtener OCR (con cache)
                case_id = f"organize_{mapping['metadata']['timestamp']}"
                doc_id, file_hash = ensure_document_registered(case_id, str(staged_path))
                
                ocr_result = try_get_cached_ocr(doc_id, file_hash, case_id=case_id)
                
                if not ocr_result:
                    logger.debug(f"Procesando OCR para {staged_path.name}")
                    ocr_obj = await ocr_processor.analyze_document_async(str(staged_path))
                    if ocr_obj:
                        ocr_result = ocr_obj.to_dict()
                        persist_ocr(doc_id, ocr_result, "azure", "full")
                
                # Extraer campos usando sistema guiado
                if ocr_result:
                    extraction = await ai_extractor.extract_from_document(
                        ocr_result=ocr_result,
                        document_name=staged_path.name,
                        document_type=doc_type
                    )
                    
                    if extraction and extraction.extracted_fields:
                        extractions.append(extraction)
                        successful_extractions += 1
                        
                        # Actualizar file_info con campos extraídos
                        file_info["extracted_fields"] = extraction.extracted_fields
                        
                        # Registrar métricas
                        for field in target_fields:
                            if extraction.extracted_fields.get(field):
                                self.metrics.phase_b_stats["fields_found"][field] = \
                                    self.metrics.phase_b_stats["fields_found"].get(field, 0) + 1
                    
            except Exception as e:
                logger.error(f"Error extrayendo de {staged_path.name}: {e}")
                file_info["extraction_error"] = str(e)
                self.metrics.phase_b_stats["extraction_failures"] += 1
        
        self.metrics.phase_b_stats["extraction_success"] = successful_extractions
        
        # 4. Consolidar solo campos de cabecera
        consolidated_values = {}
        
        if extractions:
            logger.info("Consolidando campos extraídos...")
            try:
                consolidated = await consolidator.consolidate_extractions(
                    extractions=extractions,
                    guided_mode=True,
                    # No pasamos target_fields para que consolide todo
                )
                
                # Tomar solo los campos que necesitamos
                for field in target_fields:
                    if field in consolidated.final_values:
                        consolidated_values[field] = consolidated.final_values[field]
                        
            except Exception as e:
                logger.error(f"Error en consolidación: {e}")
        
        # 5. Determinar nombre de carpeta final
        nombre_asegurado = consolidated_values.get("nombre_asegurado", "").strip()
        numero_siniestro = consolidated_values.get("numero_siniestro", "").strip()
        
        final_name = self._generate_final_folder_name(nombre_asegurado, numero_siniestro)
        
        # 6. Renombrar carpeta staging a nombre final
        final_folder = staging_folder.parent.parent / final_name
        
        # Si ya existe, añadir sufijo incremental
        final_folder = self._handle_folder_collision(final_folder)
        
        # Mover y renombrar
        shutil.move(str(staging_folder), str(final_folder))
        logger.info(f"Carpeta renombrada: {staging_folder.name} → {final_folder.name}")
        
        # 7. Actualizar mapping con información final
        mapping["metadata"]["phase_b_completed"] = True
        mapping["metadata"]["phase_b_duration_ms"] = int((time.time() - start_time) * 1000)
        mapping["metadata"]["final_folder"] = str(final_folder)
        mapping["metadata"]["nombre_asegurado"] = nombre_asegurado
        mapping["metadata"]["numero_siniestro"] = numero_siniestro
        mapping["consolidated_fields"] = consolidated_values
        mapping["extraction_stats"] = {
            "total_documents": len(valid_files),
            "successful_extractions": successful_extractions,
            "fields_found": self.metrics.phase_b_stats["fields_found"]
        }
        
        # Guardar mapping actualizado
        mapping_file_new = final_folder / "mapping.json"
        with open(mapping_file_new, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Fase B completada en {mapping['metadata']['phase_b_duration_ms']}ms")
        logger.info(f"Documentos organizados en: {final_folder}")
        
        # Imprimir resumen
        self._print_phase_b_summary(mapping)
        
        return final_folder
    
    def _discover_supported_files(self, folder: Path) -> List[Path]:
        """Descubre archivos soportados en la carpeta"""
        files = []
        for file_path in folder.iterdir():
            from fraud_scorer.settings import SUPPORTED_EXTENSIONS
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(file_path)
        return sorted(files)
    
    def _find_unsupported_files(self, folder: Path) -> List[Path]:
        """Encuentra archivos no soportados"""
        unsupported = []
        for file_path in folder.iterdir():
            from fraud_scorer.settings import SUPPORTED_EXTENSIONS
            if file_path.is_file() and file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                unsupported.append(file_path)
        return unsupported
    
    def _determine_route_for_file(self, file_path: Path) -> str:
        """Determina ruta de procesamiento para un archivo"""
        ext = file_path.suffix.lower()
        
        # Mapeo básico por extensión
        route_map = {
            '.pdf': 'ocr_text',
            '.jpg': 'direct_ai',
            '.jpeg': 'direct_ai',
            '.png': 'direct_ai',
            '.docx': 'ocr_text',
            '.xlsx': 'ocr_text',
            '.csv': 'ocr_text'
        }
        
        return route_map.get(ext, 'ocr_text')
    
    async def _get_sample_text(self, file_path: Path, route: str, max_chars: int = 2000) -> str:
        """
        Obtiene texto de muestra para clasificación
        Usa cache si existe, sino extrae rápido
        """
        # Si está activado el flag de skip OCR (para tests), retornar vacío
        if self._skip_ocr_sample:
            return ""
        
        # Intentar cache primero
        try:
            doc_id, file_hash = ensure_document_registered("temp_classify", str(file_path))
            cached = try_get_cached_ocr(doc_id, file_hash)
            if cached:
                return cached.get("text", "")[:max_chars]
        except:
            pass
        
        # Extraer según tipo
        if file_path.suffix.lower() == '.pdf':
            # Para PDFs, intentar extracción de texto nativo primero
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    # Solo primeras 2 páginas para muestra
                    for i, page in enumerate(reader.pages[:2]):
                        text += page.extract_text()
                        if len(text) >= max_chars:
                            break
                    return text[:max_chars]
            except Exception as e:
                logger.debug(f"Error extrayendo texto PDF nativo: {e}")
        
        # Si es imagen o fallo la extracción, usar OCR sample
        if route in ["ocr_text", "direct_ai"]:
            try:
                # OCR rápido de página 1 solamente
                ocr = AzureOCRProcessor()
                result = await ocr.analyze_document_async(str(file_path))
                if result:
                    # Persistir en cache para reutilizar
                    doc_id, _ = ensure_document_registered("temp_classify", str(file_path))
                    persist_ocr(doc_id, result.to_dict(), "azure", "sample")
                    return result.text[:max_chars]
            except Exception as e:
                logger.warning(f"Error en OCR sample: {e}")
        
        return ""
    
    def _extract_destinatario(self, text: str) -> Optional[str]:
        """Extrae el nombre del destinatario de guías y facturas"""
        
        # Patrones comunes para identificar destinatarios
        patterns = [
            r"Destinatario:?\s*([^\n]+)",
            r"Cliente:?\s*([^\n]+)",
            r"Consignado a:?\s*([^\n]+)",
            r"Nombre del Cliente:?\s*([^\n]+)",
            r"Razón Social:?\s*([^\n]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                destinatario = match.group(1).strip()
                # Limpiar y validar
                destinatario = re.sub(r'[^\w\s\.\-]', '', destinatario)
                if 3 < len(destinatario) < 100:
                    return destinatario
        
        return None
    
    def _sanitize_filename(self, filename: str, max_length: int = 50) -> str:
        """Sanitiza nombre de archivo para ser seguro en filesystem"""
        # Remover caracteres no seguros
        safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remover espacios múltiples
        safe = re.sub(r'\s+', '_', safe)
        # Limitar longitud
        return safe[:max_length]
    
    def _generate_final_folder_name(
        self,
        nombre_asegurado: str,
        numero_siniestro: str,
        max_length: int = 100
    ) -> str:
        """
        Genera nombre de carpeta final sanitizado
        Formato: <NOMBRE_ASEGURADO> - <NUMERO_SINIESTRO>
        """
        # Validar y limpiar nombre asegurado
        if nombre_asegurado:
            # Remover caracteres no seguros
            nombre_clean = re.sub(r'[<>:"/\\|?*]', '', nombre_asegurado)
            # Normalizar espacios
            nombre_clean = ' '.join(nombre_clean.split())
            # Convertir a mayúsculas
            nombre_clean = nombre_clean.upper()[:50]  # Max 50 chars
        else:
            nombre_clean = "DESCONOCIDO"
        
        # Validar número de siniestro (14 dígitos esperados)
        if numero_siniestro:
            # Extraer solo dígitos
            siniestro_clean = ''.join(filter(str.isdigit, numero_siniestro))
            
            # Validar longitud
            if len(siniestro_clean) == 14:
                # Mantener los 14 dígitos
                siniestro_clean = siniestro_clean
            else:
                # Si no es válido, usar como está (truncado)
                siniestro_clean = re.sub(r'[<>:"/\\|?*]', '', numero_siniestro)[:20]
        else:
            # Fallback: timestamp
            siniestro_clean = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Combinar
        folder_name = f"{nombre_clean} - {siniestro_clean}"
        
        # Truncar si es muy largo
        if len(folder_name) > max_length:
            # Priorizar el número de siniestro
            available = max_length - len(siniestro_clean) - 3  # " - "
            nombre_truncated = nombre_clean[:available]
            folder_name = f"{nombre_truncated} - {siniestro_clean}"
        
        return folder_name
    
    def _handle_folder_collision(self, base_path: Path) -> Path:
        """Maneja colisiones de carpetas añadiendo sufijo incremental"""
        if not base_path.exists():
            return base_path
        
        counter = 2
        while True:
            new_path = base_path.parent / f"{base_path.name}_{counter}"
            if not new_path.exists():
                return new_path
            counter += 1
            if counter > 100:  # Prevenir loops infinitos
                raise ValueError(f"Demasiadas colisiones para {base_path}")
    
    def _print_phase_a_summary(self, mapping: Dict[str, Any]):
        """Imprime resumen de Fase A"""
        print("\n" + "="*60)
        print("RESUMEN FASE A - CLASIFICACIÓN Y STAGING")
        print("="*60)
        
        # Distribución por tipo
        type_counts = {}
        for f in mapping["files"]:
            dt = f.get("document_type", "error")
            type_counts[dt] = type_counts.get(dt, 0) + 1
        
        print("\nDistribución por tipo de documento:")
        for doc_type, count in sorted(type_counts.items(), 
                                     key=lambda x: DOCUMENT_PRIORITIES.get(x[0], 99)):
            percentage = (count / len(mapping["files"])) * 100
            print(f"  {doc_type:40} : {count:3} ({percentage:5.1f}%)")
        
        # Estadísticas
        print(f"\nEstadísticas:")
        print(f"  Total archivos procesados: {len(mapping['files'])}")
        print(f"  Archivos no soportados: {mapping['metadata'].get('unsupported_files', 0)}")
        print(f"  Tiempo total: {mapping['metadata'].get('phase_a_duration_ms', 0) / 1000:.2f}s")
        
        # Métricas de clasificación
        metrics_report = self.metrics.generate_report()
        if metrics_report["phase_a"]["total_files"] > 0:
            print(f"\nMétodo de clasificación:")
            print(f"  Heurística: {metrics_report['phase_a']['classified_heuristic']}")
            print(f"  LLM (fallback): {metrics_report['phase_a']['classified_llm']}")
            print(f"  Tiempo promedio: {metrics_report['phase_a']['avg_classification_time_ms']:.0f}ms")
    
    def _print_phase_b_summary(self, mapping: Dict[str, Any]):
        """Imprime resumen de Fase B"""
        print("\n" + "="*60)
        print("RESUMEN FASE B - EXTRACCIÓN Y RENOMBRADO")
        print("="*60)
        
        print(f"\nCarpeta final: {mapping['metadata'].get('final_folder', 'N/A')}")
        print(f"\nCampos consolidados:")
        print(f"  Nombre Asegurado: {mapping['metadata'].get('nombre_asegurado', 'NO ENCONTRADO')}")
        print(f"  Número Siniestro: {mapping['metadata'].get('numero_siniestro', 'NO ENCONTRADO')}")
        
        if "extraction_stats" in mapping:
            stats = mapping["extraction_stats"]
            print(f"\nEstadísticas de extracción:")
            print(f"  Documentos procesados: {stats['total_documents']}")
            print(f"  Extracciones exitosas: {stats['successful_extractions']}")
            print(f"  Campos encontrados:")
            for field, count in stats['fields_found'].items():
                print(f"    - {field}: {count} documentos")
        
        print(f"\nTiempo total Fase B: {mapping['metadata'].get('phase_b_duration_ms', 0) / 1000:.2f}s")


class OrganizationMetrics:
    """Métricas del proceso de organización"""
    
    def __init__(self):
        self.phase_a_stats = {
            "total_files": 0,
            "classified_heuristic": 0,
            "classified_llm": 0,
            "classification_time_ms": [],
            "types_distribution": {}
        }
        
        self.phase_b_stats = {
            "extraction_success": 0,
            "extraction_failures": 0,
            "fields_found": {},
            "total_time_ms": 0
        }
    
    def log_classification(self, doc_type: str, used_llm: bool, time_ms: int):
        """Registra una clasificación"""
        self.phase_a_stats["total_files"] += 1
        
        if used_llm:
            self.phase_a_stats["classified_llm"] += 1
        else:
            self.phase_a_stats["classified_heuristic"] += 1
        
        self.phase_a_stats["classification_time_ms"].append(time_ms)
        
        # Distribución por tipo
        types = self.phase_a_stats["types_distribution"]
        types[doc_type] = types.get(doc_type, 0) + 1
    
    def generate_report(self) -> Dict[str, Any]:
        """Genera reporte de métricas"""
        avg_time = 0
        if self.phase_a_stats["classification_time_ms"]:
            avg_time = sum(self.phase_a_stats["classification_time_ms"]) / \
                      len(self.phase_a_stats["classification_time_ms"])
        
        llm_rate = 0
        if self.phase_a_stats["total_files"] > 0:
            llm_rate = self.phase_a_stats["classified_llm"] / \
                      self.phase_a_stats["total_files"]
        
        return {
            "phase_a": {
                **self.phase_a_stats,
                "avg_classification_time_ms": avg_time,
                "llm_usage_rate": llm_rate
            },
            "phase_b": self.phase_b_stats
        }