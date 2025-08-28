from __future__ import annotations
from typing import Optional, Dict, Any
from .db import (
    upsert_document, get_ocr_by_document_id, get_any_ocr_by_hash, copy_ocr_to_document,
    save_ocr_result, mark_ocr_success, sha256_of_file, get_extracted_by_document_id, save_extracted_data
)
from pathlib import Path
import json
import os
import shutil
import re
import logging

logger = logging.getLogger(__name__)

def ensure_document_registered(case_id: str, filepath: str) -> tuple[str, str]:
    """
    Garantiza que el documento esté en la tabla documents y devuelve (document_id, file_hash).
    """
    p = Path(filepath)
    # upsert_document internamente calcula hash, pero queremos el hash explícito para reuso global
    file_hash = sha256_of_file(p)
    doc_id, _is_new = upsert_document(case_id, str(p), mime_type=None, page_count=None, language=None)
    return doc_id, file_hash

def try_get_cached_ocr(document_id: str, file_hash: str, allow_global: bool = True) -> Optional[Dict[str, Any]]:
    """
    Devuelve el OCR (dict) si ya existe para este document_id o por hash global.
    """
    row = get_ocr_by_document_id(document_id)
    if row:
        return _row_to_ocr_dict(row)

    if allow_global:
        any_ocr = get_any_ocr_by_hash(file_hash)
        if any_ocr:
            # copiar y devolver
            copy_ocr_to_document(any_ocr, document_id)
            return _row_to_ocr_dict(any_ocr)

    return None

def persist_ocr(document_id: str, ocr_dict: Dict[str, Any], engine: str, engine_version: Optional[str] = None) -> None:
    save_ocr_result(document_id, ocr_dict, engine, engine_version)
    mark_ocr_success(document_id, True)

def try_get_cached_extraction(document_id: str) -> Optional[Dict[str, Any]]:
    row = get_extracted_by_document_id(document_id)
    if not row:
        return None
    return {
        "document_type": row["document_type"],
        "entities": json.loads(row["entities"] or "{}"),
        "key_value_pairs": json.loads(row["key_value_pairs"] or "{}"),
        "extra": json.loads(row["extra"] or "{}"),
    }

def persist_extraction(document_id: str, extracted: Dict[str, Any], extractor_version: str = "v1") -> None:
    save_extracted_data(document_id, extracted, extractor_version)

def _row_to_ocr_dict(row) -> Dict[str, Any]:
    return {
        "text": row["raw_text"] or "",
        "key_value_pairs": json.loads(row["key_value_pairs"] or "{}"),
        "tables": json.loads(row["tables"] or "[]"),
        "entities": json.loads(row["entities"] or "[]"),
        "confidence_scores": json.loads(row["confidence"] or "{}"),
        "metadata": json.loads(row["metadata"] or "{}"),
        "errors": json.loads(row["errors"] or "[]"),
    }


class OCRCacheManager:
    """
    Gestor de caché para resultados de OCR con soporte para reorganización de archivos por caso.
    """
    
    def __init__(self, cache_base_dir: Optional[Path] = None):
        if cache_base_dir is None:
            # Usar la misma ubicación que se ve en la estructura actual
            cache_base_dir = Path("data/ocr_cache")
        
        self.cache_dir = Path(cache_base_dir)
        self.index_dir = self.cache_dir / "case_index"
        
        # Crear directorios si no existen
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"OCRCacheManager inicializado con directorio: {self.cache_dir}")
    
    def _get_cache_path(self, document_path: Path) -> Path:
        """
        Genera la ruta del archivo de caché basada en el hash del documento.
        Mantiene compatibilidad con la estructura actual (por hash).
        """
        file_hash = sha256_of_file(document_path)
        # Usar los primeros 2 caracteres para crear subdirectorio
        subdir = file_hash[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(parents=True, exist_ok=True)
        
        return cache_subdir / f"{file_hash}.json"
    
    def has_cache(self, document_path: Path) -> bool:
        """
        Verifica si existe caché para el documento.
        """
        cache_path = self._get_cache_path(document_path)
        return cache_path.exists()
    
    def get_cache(self, document_path: Path) -> Optional[Dict[str, Any]]:
        """
        Obtiene el resultado de OCR desde el caché.
        """
        if not self.has_cache(document_path):
            return None
        
        try:
            cache_path = self._get_cache_path(document_path)
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error leyendo caché para {document_path}: {e}")
            return None
    
    def save_cache(self, document_path: Path, ocr_result: Dict[str, Any]) -> None:
        """
        Guarda el resultado de OCR en el caché.
        """
        try:
            cache_path = self._get_cache_path(document_path)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(ocr_result, f, ensure_ascii=False, indent=2, default=str)
            logger.debug(f"Caché guardado para {document_path.name}: {cache_path}")
        except Exception as e:
            logger.error(f"Error guardando caché para {document_path}: {e}")
    
    def save_case_index(self, case_id: str, case_data: Dict[str, Any]) -> None:
        """
        Guarda el índice de archivos de un caso para futura reorganización.
        """
        try:
            index_path = self.index_dir / f"{case_id}.json"
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(case_data, f, ensure_ascii=False, indent=2, default=str)
            logger.debug(f"Índice de caso guardado: {index_path}")
        except Exception as e:
            logger.error(f"Error guardando índice del caso {case_id}: {e}")
    
    def _sanitize_filename(self, name: str) -> str:
        """
        Elimina caracteres no válidos de un string para que sea un nombre de archivo/carpeta seguro.
        """
        if not name:
            return "SIN_NOMBRE"
        # Reemplaza secuencias de caracteres no alfanuméricos por un solo guion bajo
        name = re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)
        # Elimina guiones bajos al principio o al final
        return name.strip('_')
    
    def reorganize_cache_for_case(self, case_id: str, insured_name: str, claim_number: str):
        """
        Reorganiza los archivos de caché de un caso en una nueva estructura de carpetas.
        """
        logger.info(f"Reorganizando caché para el caso {case_id}...")
        case_index_path = self.index_dir / f"{case_id}.json"

        if not case_index_path.exists():
            logger.warning(f"No se encontró el índice del caso {case_id}. No se puede reorganizar el caché.")
            return

        try:
            with open(case_index_path, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
        except Exception as e:
            logger.error(f"Error leyendo el índice del caso {case_id}: {e}")
            return

        # 1. Crear el nombre de la nueva carpeta
        sanitized_insured_name = self._sanitize_filename(insured_name)
        sanitized_claim_number = self._sanitize_filename(claim_number)
        new_case_folder_name = f"{sanitized_insured_name} - {sanitized_claim_number}"
        new_case_path = self.cache_dir / new_case_folder_name

        # Creamos el directorio principal del caso si no existe
        new_case_path.mkdir(parents=True, exist_ok=True)

        # 2. Mover cada archivo de caché a su nueva ubicación
        if "cache_files" in case_data:
            for original_doc_path_str in case_data["cache_files"]:
                original_doc_path = Path(original_doc_path_str)
                cache_path = self._get_cache_path(original_doc_path)

                if cache_path.exists():
                    try:
                        # Crear subcarpeta para el documento específico
                        doc_folder_name = self._sanitize_filename(original_doc_path.stem)
                        doc_specific_path = new_case_path / doc_folder_name
                        doc_specific_path.mkdir(parents=True, exist_ok=True)

                        # El nuevo nombre del archivo JSON será más descriptivo
                        new_cache_filename = f"ocr_results_for_{self._sanitize_filename(original_doc_path.name)}.json"
                        destination_path = doc_specific_path / new_cache_filename
                        
                        logger.info(f"Moviendo {cache_path} -> {destination_path}")
                        shutil.move(str(cache_path), str(destination_path))
                    except Exception as e:
                        logger.error(f"No se pudo mover el archivo de caché {cache_path}: {e}")
        
        logger.info(f"Reorganización del caché completada para el caso {case_id} en: {new_case_path}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del cache para mostrar en la UI.
        """
        stats = {
            'total_cases': 0,
            'total_cached_files': 0,
            'cache_size_mb': 0.0,
            'cache_directory': str(self.cache_dir)
        }
        
        try:
            # Contar archivos de índice de casos
            if self.index_dir.exists():
                index_files = list(self.index_dir.glob("*.json"))
                stats['total_cases'] = len(index_files)
            
            # Contar archivos de cache y calcular tamaño
            total_size = 0
            total_files = 0
            
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.rglob("*.json"):
                    if cache_file.is_file():
                        total_files += 1
                        try:
                            total_size += cache_file.stat().st_size
                        except OSError:
                            pass  # Ignorar archivos que no se pueden leer
            
            stats['total_cached_files'] = total_files
            stats['cache_size_mb'] = round(total_size / (1024 * 1024), 2)
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas del cache: {e}")
        
        return stats

    def list_cached_cases(self) -> list[Dict[str, Any]]:
        """
        Lista todos los casos que tienen índice en el cache.
        Retorna una lista de diccionarios con información de cada caso.
        """
        cases = []
        
        if not self.index_dir.exists():
            return cases
        
        try:
            for index_file in self.index_dir.glob("*.json"):
                case_id = index_file.stem
                
                try:
                    with open(index_file, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)
                    
                    case_info = {
                        'case_id': case_id,
                        'case_title': case_data.get('case_title', case_id),
                        'total_documents': case_data.get('total_documents', 0),
                        'processed_at': case_data.get('processed_at', ''),
                        'folder_path': case_data.get('folder_path', '')
                    }
                    cases.append(case_info)
                    
                except Exception as e:
                    logger.error(f"Error leyendo índice del caso {case_id}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error listando casos del cache: {e}")
        
        # Ordenar por fecha de procesamiento (más reciente primero)
        cases.sort(key=lambda x: x.get('processed_at', ''), reverse=True)
        
        return cases

    def get_case_index(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene la información del índice de un caso específico.
        """
        index_path = self.index_dir / f"{case_id}.json"
        
        if not index_path.exists():
            return None
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error leyendo índice del caso {case_id}: {e}")
            return None
