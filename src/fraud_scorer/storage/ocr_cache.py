from __future__ import annotations
from typing import Optional, Dict, Any
from .db import (
    upsert_document, get_ocr_by_document_id, get_any_ocr_by_hash, copy_ocr_to_document,
    save_ocr_result, mark_ocr_success, sha256_of_file, get_extracted_by_document_id, save_extracted_data,
    increment_cache_stats, update_cache_avg
)
from pathlib import Path
import json
import os
import shutil
import re
import logging
import time

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

def try_get_cached_ocr(document_id: str, file_hash: str, allow_global: bool = True, case_id: str = None) -> Optional[Dict[str, Any]]:
    """
    Devuelve el OCR (dict) si ya existe para este document_id o por hash global.
    """
    start_time = time.time()
    row = get_ocr_by_document_id(document_id)
    if row:
        # Cache hit
        elapsed_ms = int((time.time() - start_time) * 1000)
        increment_cache_stats('global', 'ocr_hits')
        increment_cache_stats('global', 'ms_saved', elapsed_ms)
        if case_id:
            increment_cache_stats(case_id, 'ocr_hits')
            increment_cache_stats(case_id, 'ms_saved', elapsed_ms)
        return _row_to_ocr_dict(row)

    if allow_global:
        any_ocr = get_any_ocr_by_hash(file_hash)
        if any_ocr:
            # copiar y devolver (hit global)
            copy_ocr_to_document(any_ocr, document_id)
            elapsed_ms = int((time.time() - start_time) * 1000)
            increment_cache_stats('global', 'ocr_hits')
            increment_cache_stats('global', 'ms_saved', elapsed_ms)
            if case_id:
                increment_cache_stats(case_id, 'ocr_hits')
                increment_cache_stats(case_id, 'ms_saved', elapsed_ms)
            return _row_to_ocr_dict(any_ocr)

    # Cache miss
    increment_cache_stats('global', 'ocr_misses')
    if case_id:
        increment_cache_stats(case_id, 'ocr_misses')
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
    
    def _find_cache_in_reorganized_structure(self, document_path: Path, case_id: Optional[str] = None) -> Optional[Path]:
        """
        Busca el archivo de cache en la estructura reorganizada [ASEGURADO] - [SINIESTRO]
        Restringe la búsqueda al caso actual si case_id está disponible.
        """
        try:
            import unicodedata
            doc_name = document_path.name
            # Normalizar diacríticos para evitar discrepancias Póliza vs Póliza
            stem = document_path.stem
            stem_norm = ''.join(c for c in unicodedata.normalize('NFKD', stem) if not unicodedata.combining(c))
            sanitized_stem = self._sanitize_filename(stem)
            sanitized_stem_norm = self._sanitize_filename(stem_norm)
            sanitized_name = self._sanitize_filename(doc_name)

            # Si no hay case_id, no buscar globalmente por nombre (evita falsos positivos entre casos)
            if not case_id:
                return None

            # Cargar el índice del caso para obtener carpeta exacta
            index_path = self.index_dir / f"{case_id}.json"
            if not index_path.exists():
                return None
            with open(index_path, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
            insured = case_data.get('insured_name') or "SIN_NOMBRE"
            claim_raw = case_data.get('claim_number') or case_id
            sanitized_claim = self._sanitize_filename(claim_raw)
            case_folder = self.cache_dir / f"{self._sanitize_filename(insured)} - {sanitized_claim}"
            # Construir candidatos de carpeta de caso: actual y otras que compartan el mismo reclamo
            case_candidates = []
            if case_folder.exists():
                case_candidates.append(case_folder)
            try:
                # Buscar otras carpetas con el mismo sufijo de reclamo
                suffix = f" - {sanitized_claim}"
                for sub in self.cache_dir.iterdir():
                    if not sub.is_dir():
                        continue
                    if sub == case_folder:
                        continue
                    if sub.name.endswith(suffix):
                        case_candidates.append(sub)
            except Exception:
                pass

            # Buscar dentro de todas las carpetas candidatas del caso
            candidates = []
            for case_dir in case_candidates:
                for folder_name in {sanitized_stem, sanitized_stem_norm}:
                    if not folder_name:
                        continue
                    df = case_dir / folder_name
                    if df.exists() and df.is_dir():
                        candidates.append(df)

            # Fallback: intentar localizar carpeta por prefijo aproximado
            if not candidates:
                try:
                    for sub in case_folder.iterdir():
                        if not sub.is_dir():
                            continue
                        # Coincidencia laxa: quitar guiones bajos repetidos y comparar inicios
                        def _canon(s: str) -> str:
                            s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
                            s = self._sanitize_filename(s)
                            return s.replace('__', '_')
                        if _canon(sub.name).startswith(_canon(stem)[:8]):  # 8 chars de margen
                            candidates.append(sub)
                except Exception:
                    pass

            # Inspeccionar candidatos
            for doc_folder in candidates:
                pattern1 = doc_folder / f"ocr_results_for_{sanitized_name}.json"
                pattern2 = doc_folder / f"ocr_results_for_{doc_name}.json"
                if pattern1.exists():
                    return pattern1
                if pattern2.exists():
                    return pattern2
                # Último recurso: cualquier json de ocr_results dentro
                for p in doc_folder.glob('ocr_results_for_*.json'):
                    return p
            return None
        except Exception as e:
            logger.error(f"Error buscando caché reorganizado: {e}")
            return None
    
    def has_cache(self, document_path: Path, case_id: Optional[str] = None) -> bool:
        """
        Verifica si existe caché para el documento.
        Si se proporciona case_id, busca primero en la estructura reorganizada de ese caso.
        En cualquier caso, verifica también por hash (shards) para coincidencia exacta de contenido.
        """
        debug = os.getenv("OCR_CACHE_DEBUG", "false").lower() == "true"
        # 1) Vista humana del caso (opcional)
        if case_id:
            reorganized_path = self._find_cache_in_reorganized_structure(document_path, case_id=case_id)
            if reorganized_path is not None:
                if debug:
                    logger.debug(f"OCR_CACHE_DEBUG: hit vista humana: {reorganized_path}")
                return True

        # 2) Shards por hash
        cache_path = self._get_cache_path(document_path)
        if cache_path.exists():
            if debug:
                logger.debug(f"OCR_CACHE_DEBUG: hit shard: {cache_path}")
            return True

        # 3) Fallback robusto: buscar en DB por hash global (si existe OCR previo)
        try:
            file_hash = sha256_of_file(document_path)
            any_ocr = get_any_ocr_by_hash(file_hash)
            if any_ocr:
                if debug:
                    logger.debug(f"OCR_CACHE_DEBUG: hit DB por hash: {file_hash}")
                return True
            elif debug:
                logger.debug(f"OCR_CACHE_DEBUG: miss DB por hash: {file_hash}")
        except Exception:
            pass

        return False
    
    def get_cache(self, document_path: Path, case_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Obtiene el resultado de OCR desde el caché.
        Busca primero en la estructura reorganizada (vista humana), luego en hash (shards).
        """
        try:
            debug = os.getenv("OCR_CACHE_DEBUG", "false").lower() == "true"
            start_time = time.time()
            # 1) Vista humana primero
            reorganized_path = self._find_cache_in_reorganized_structure(document_path, case_id=case_id)
            if reorganized_path:
                with open(reorganized_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if debug:
                    logger.debug(f"OCR_CACHE_DEBUG: lectura desde vista humana: {reorganized_path}")
                # Registrar hit de caché
                elapsed_ms = int((time.time() - start_time) * 1000)
                increment_cache_stats('global', 'ocr_hits')
                increment_cache_stats('global', 'ms_saved', elapsed_ms)
                if case_id:
                    increment_cache_stats(case_id, 'ocr_hits')
                    increment_cache_stats(case_id, 'ms_saved', elapsed_ms)
                # Estimar bytes ahorrados
                bytes_saved = reorganized_path.stat().st_size
                increment_cache_stats('global', 'bytes_saved', bytes_saved)
                if case_id:
                    increment_cache_stats(case_id, 'bytes_saved', bytes_saved)
                # Persistir en DB para habilitar futuras búsquedas por hash
                try:
                    doc_id, _ = ensure_document_registered(case_id or 'unknown', str(document_path))
                    persist_ocr(doc_id, data, "azure", "full")
                except Exception:
                    pass
                return data

            # 2) Shards por hash
            cache_path = self._get_cache_path(document_path)
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if debug:
                    logger.debug(f"OCR_CACHE_DEBUG: lectura desde shard: {cache_path}")
                # Registrar hit de caché
                elapsed_ms = int((time.time() - start_time) * 1000)
                increment_cache_stats('global', 'ocr_hits')
                increment_cache_stats('global', 'ms_saved', elapsed_ms)
                if case_id:
                    increment_cache_stats(case_id, 'ocr_hits')
                    increment_cache_stats(case_id, 'ms_saved', elapsed_ms)
                # Estimar bytes ahorrados (tamaño del archivo)
                bytes_saved = cache_path.stat().st_size
                increment_cache_stats('global', 'bytes_saved', bytes_saved)
                if case_id:
                    increment_cache_stats(case_id, 'bytes_saved', bytes_saved)
                return data
            
            # Cache miss
            increment_cache_stats('global', 'ocr_misses')
            if case_id:
                increment_cache_stats(case_id, 'ocr_misses')
            # 3) Fallback a DB: intentar por hash global y/o por documento registrado
            try:
                # Registrar/asegurar documento y obtener hash
                doc_id, file_hash = ensure_document_registered(case_id or 'unknown', str(document_path))
                cached = try_get_cached_ocr(doc_id, file_hash, allow_global=True, case_id=case_id)
                if cached:
                    if debug:
                        logger.debug(f"OCR_CACHE_DEBUG: lectura desde DB por hash: {file_hash} -> doc_id {doc_id}")
                    # Importante: rehidratar a FS para que exista caché físico
                    # y luego pueda reorganizarse a la estructura humana.
                    try:
                        self.save_cache(document_path, cached, case_id)
                    except Exception as write_err:
                        # No bloquear por fallo de escritura; ya tenemos los datos en memoria/DB
                        logger.debug(f"No se pudo materializar cache en FS desde DB: {write_err}")
                    return cached
            except Exception as e:
                logger.debug(f"Fallback DB cache fallo para {document_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error leyendo caché para {document_path}: {e}")
            return None
    
    def save_cache(self, document_path: Path, ocr_result: Dict[str, Any], case_id: Optional[str] = None) -> None:
        """
        Guarda el resultado de OCR en el caché.
        """
        try:
            # Si conocemos el caso y su carpeta con nombre, guardar directo en vista humana
            if case_id:
                index_path = self.index_dir / f"{case_id}.json"
                if index_path.exists():
                    try:
                        with open(index_path, 'r', encoding='utf-8') as f:
                            case_data = json.load(f)
                        insured = case_data.get('insured_name') or "SIN_NOMBRE"
                        claim = case_data.get('claim_number') or case_id
                        folder_name = f"{self._sanitize_filename(insured)} - {self._sanitize_filename(claim)}"
                        doc_folder = self._sanitize_filename(document_path.stem)
                        dest_dir = self.cache_dir / folder_name / doc_folder
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_dir / f"ocr_results_for_{self._sanitize_filename(document_path.name)}.json"
                        with open(dest_path, 'w', encoding='utf-8') as f:
                            json.dump(ocr_result, f, ensure_ascii=False, indent=2, default=str)
                        logger.debug(f"Caché guardado (vista humana) para {document_path.name}: {dest_path}")
                        # Persistir también en DB (para fallback robusto por hash)
                        try:
                            doc_id, _ = ensure_document_registered(case_id, str(document_path))
                            persist_ocr(doc_id, ocr_result, "azure", "full")
                        except Exception:
                            pass
                        return
                    except Exception as e:
                        logger.warning(f"Fallo guardando en vista humana; guardando en shard. Detalle: {e}")

            # Fallback a shards por hash
            cache_path = self._get_cache_path(document_path)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(ocr_result, f, ensure_ascii=False, indent=2, default=str)
            logger.debug(f"Caché guardado (shard) para {document_path.name}: {cache_path}")
            # Persistir en DB aunque no exista índice aún
            try:
                doc_id, _ = ensure_document_registered(case_id or 'unknown', str(document_path))
                persist_ocr(doc_id, ocr_result, "azure", "full")
            except Exception:
                pass
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
        Limpia las carpetas de hash vacías después de mover los archivos.
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

        # 2. Mantener registro de carpetas de hash para limpiar
        hash_folders_to_clean = set()
        
        # 3. Mover cada archivo de caché a su nueva ubicación
        #    Tomar rutas de 'cache_files' y asegurar con 'documents' (por si falta alguno)
        original_paths: list[str] = []
        try:
            if "cache_files" in case_data and isinstance(case_data["cache_files"], list):
                original_paths.extend([str(p) for p in case_data["cache_files"]])
        except Exception:
            pass
        try:
            if "documents" in case_data and isinstance(case_data["documents"], list):
                for p in case_data["documents"]:
                    sp = str(p)
                    if sp not in original_paths:
                        original_paths.append(sp)
        except Exception:
            pass

        for original_doc_path_str in original_paths:
                original_doc_path = Path(original_doc_path_str)
                cache_path = self._get_cache_path(original_doc_path)

                if cache_path.exists():
                    try:
                        # Guardar la carpeta de hash para limpiar después
                        hash_folder = cache_path.parent
                        hash_folders_to_clean.add(hash_folder)
                        
                        # Crear subcarpeta para el documento específico
                        doc_folder_name = self._sanitize_filename(original_doc_path.stem)
                        doc_specific_path = new_case_path / doc_folder_name
                        doc_specific_path.mkdir(parents=True, exist_ok=True)

                        # El nuevo nombre del archivo JSON será más descriptivo
                        new_cache_filename = f"ocr_results_for_{self._sanitize_filename(original_doc_path.name)}.json"
                        destination_path = doc_specific_path / new_cache_filename
                        
                        # Verificar que no exista el destino antes de mover
                        if destination_path.exists():
                            logger.warning(f"El archivo destino ya existe: {destination_path}. Sobrescribiendo...")
                            destination_path.unlink()
                        
                        logger.info(f"Moviendo {cache_path} -> {destination_path}")
                        shutil.move(str(cache_path), str(destination_path))
                        
                    except Exception as e:
                        logger.error(f"No se pudo mover el archivo de caché {cache_path}: {e}")
                else:
                    # Si no hay shard, verificar si ya existe en vista humana; de estar en ambos, eliminar shard duplicado
                    doc_folder_name = self._sanitize_filename(original_doc_path.stem)
                    candidate = new_case_path / doc_folder_name / f"ocr_results_for_{self._sanitize_filename(original_doc_path.name)}.json"
                    if candidate.exists():
                        # Nada que mover
                        continue
        
        # 4. Fusionar carpeta placeholder "SIN_NOMBRE - SIN_NOMBRE" en la carpeta con nombre real (si existe)
        try:
            placeholder_folder = self.cache_dir / "SIN_NOMBRE - SIN_NOMBRE"
            if placeholder_folder.exists() and placeholder_folder.is_dir() and placeholder_folder != new_case_path:
                for sub in list(placeholder_folder.iterdir()):
                    if not sub.is_dir():
                        continue
                    dest_sub = new_case_path / sub.name
                    dest_sub.mkdir(parents=True, exist_ok=True)
                    for item in list(sub.iterdir()):
                        target = dest_sub / item.name
                        try:
                            if not target.exists():
                                shutil.move(str(item), str(target))
                        except Exception as e:
                            logger.warning(f"No se pudo mover {item} -> {target}: {e}")
                    # Intentar borrar subcarpeta si quedó vacía (limpiar .DS_Store)
                    ds = sub / ".DS_Store"
                    if ds.exists():
                        try:
                            ds.unlink()
                        except Exception:
                            pass
                    try:
                        sub.rmdir()
                    except OSError:
                        pass
                # Intentar borrar carpeta placeholder
                ds = placeholder_folder / ".DS_Store"
                if ds.exists():
                    try:
                        ds.unlink()
                    except Exception:
                        pass
                try:
                    placeholder_folder.rmdir()
                except OSError:
                    pass
        except Exception as e:
            logger.error(f"Error fusionando carpeta placeholder: {e}")

        # 5. Limpiar carpetas de hash vacías (ignorando .DS_Store)
        for hash_folder in hash_folders_to_clean:
            try:
                if hash_folder.exists() and hash_folder.is_dir():
                    # Eliminar .DS_Store si existe
                    ds = hash_folder / ".DS_Store"
                    if ds.exists():
                        try:
                            ds.unlink()
                        except Exception:
                            pass
                    # Verificar si la carpeta quedó vacía
                    if not any(hash_folder.iterdir()):
                        try:
                            hash_folder.rmdir()
                            logger.debug(f"Carpeta de hash vacía eliminada: {hash_folder}")
                        except Exception as ex:
                            logger.warning(f"No se pudo eliminar carpeta vacía {hash_folder}: {ex}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar la carpeta de hash {hash_folder}: {e}")
        
        # 6. Limpieza global de shards vacíos
        try:
            self.cleanup_shards()
        except Exception as e:
            logger.warning(f"cleanup_shards falló: {e}")

        logger.info(f"Reorganización del caché completada para el caso {case_id} en: {new_case_path}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del cache para mostrar en la UI.
        Cuenta correctamente las carpetas de casos con nombre y excluye carpetas de hash.
        """
        stats = {
            'total_cases': 0,
            'total_cached_files': 0,
            'cache_size_mb': 0.0,
            'cache_directory': str(self.cache_dir)
        }
        
        try:
            # Contar casos de múltiples fuentes
            case_count_index = 0
            case_count_folders = 0
            
            # 1. Contar archivos de índice de casos
            if self.index_dir.exists():
                index_files = list(self.index_dir.glob("*.json"))
                case_count_index = len(index_files)
            
            # 2. Contar carpetas de casos con nombre (formato: "Nombre - Número")
            # Excluir carpetas de hash (solo 2 caracteres) y case_index
            if self.cache_dir.exists():
                for folder in self.cache_dir.iterdir():
                    if folder.is_dir():
                        folder_name = folder.name
                        # Excluir carpetas de hash (2 caracteres hex) y case_index
                        if (len(folder_name) > 2 and 
                            folder_name != 'case_index' and 
                            '-' in folder_name):  # Carpetas con formato "Nombre - Número"
                            case_count_folders += 1
            
            # Usar el máximo entre los dos conteos
            stats['total_cases'] = max(case_count_index, case_count_folders)
            
            # 3. Contar archivos de cache y calcular tamaño
            total_size = 0
            total_files = 0
            
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.rglob("*.json"):
                    if cache_file.is_file():
                        # Excluir archivos en case_index
                        if 'case_index' not in str(cache_file.parent):
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
                    
                    # Construir título dinámicamente a partir de insured_name y claim_number
                    insured_name = case_data.get('insured_name', '')
                    claim_number = case_data.get('claim_number', '')
                    
                    if insured_name and claim_number:
                        case_title = f"{insured_name} - {claim_number}"
                    elif insured_name:
                        case_title = insured_name
                    elif claim_number:
                        case_title = f"Reclamo {claim_number}"
                    else:
                        # Fallback al título existente o al case_id
                        case_title = case_data.get('case_title', case_id)
                    
                    # Contar documentos correctamente
                    total_documents = case_data.get('total_documents', 0)
                    if total_documents == 0 and 'documents' in case_data:
                        # Si no hay total_documents pero hay lista de documentos, contar
                        total_documents = len(case_data.get('documents', []))
                    
                    # Obtener fecha de procesamiento o usar fecha de modificación del archivo
                    processed_at = case_data.get('processed_at', '')
                    if not processed_at:
                        try:
                            # Usar fecha de modificación del archivo como fallback
                            file_stat = index_file.stat()
                            from datetime import datetime
                            processed_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                        except:
                            processed_at = ''
                    
                    case_info = {
                        'case_id': case_id,
                        'case_title': case_title,
                        'total_documents': total_documents,
                        'processed_at': processed_at,
                        'folder_path': case_data.get('folder_path', ''),
                        'insured_name': insured_name,
                        'claim_number': claim_number
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

    # Utilidad pública: limpieza manual de shards
    def cleanup_shards(self) -> None:
        """Elimina .DS_Store y borra shards vacíos (subcarpetas de 2 hex)."""
        try:
            for entry in self.cache_dir.iterdir():
                if not entry.is_dir():
                    continue
                name = entry.name
                if len(name) == 2 and all(c in '0123456789abcdef' for c in name.lower()):
                    # Borrar .DS_Store si está
                    ds = entry / '.DS_Store'
                    if ds.exists():
                        try:
                            ds.unlink()
                        except Exception:
                            pass
                    # Borrar shard si quedó vacío
                    try:
                        if not any(entry.iterdir()):
                            entry.rmdir()
                    except OSError:
                        pass
        except Exception as e:
            logger.error(f"Error durante cleanup_shards: {e}")
