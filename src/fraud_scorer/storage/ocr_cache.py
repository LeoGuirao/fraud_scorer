from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
from .db import (
    upsert_document, get_ocr_by_document_id, get_any_ocr_by_hash, copy_ocr_to_document,
    save_ocr_result, mark_ocr_success, sha256_of_file, get_extracted_by_document_id, save_extracted_data,
    increment_cache_stats, update_cache_avg, get_conn, get_document_by_case_and_hash
)
from pathlib import Path
import json
import os
import shutil
import re
import logging
import time
from datetime import datetime, timezone

from fraud_scorer.storage.gps_cache import GPSCacheManager

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

        self.gps_cache = GPSCacheManager()

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

        # Detección directa: si nos pasan el JSON reorganizado 'ocr_results_for_*.json'
        try:
            if document_path.suffix.lower() == ".json" and document_path.name.startswith("ocr_results_for_"):
                return document_path.exists()
        except Exception:
            pass

        case_hashes: Dict[str, str] = {}
        if case_id:
            try:
                case_index = self.get_case_index(case_id) or {}
                raw_hashes = case_index.get('document_hashes') or {}
                if isinstance(raw_hashes, dict):
                    case_hashes = dict(raw_hashes)
            except Exception:
                case_hashes = {}

        # 1) Vista humana del caso (opcional)
        if case_id:
            reorganized_path = self._find_cache_in_reorganized_structure(document_path, case_id=case_id)
            if reorganized_path is not None:
                if debug:
                    logger.debug(f"OCR_CACHE_DEBUG: hit vista humana: {reorganized_path}")
                return True

        # 2) Shards por hash (sin requerir archivo físico)
        doc_hash: Optional[str] = None
        if case_hashes:
            key = str(document_path)
            doc_hash = case_hashes.get(key)
            if not doc_hash:
                try:
                    doc_hash = case_hashes.get(str(Path(key).resolve()))
                except Exception:
                    pass

        cache_path: Optional[Path] = None
        if doc_hash:
            cache_path = self.cache_dir / doc_hash[:2] / f"{doc_hash}.json"
        else:
            try:
                cache_path = self._get_cache_path(document_path)
            except FileNotFoundError:
                cache_path = None
            except Exception:
                cache_path = None

        if cache_path and cache_path.exists():
            if debug:
                logger.debug(f"OCR_CACHE_DEBUG: hit shard: {cache_path}")
            return True

        # 3) Fallback robusto: buscar en DB por hash global (si existe OCR previo)
        try:
            file_hash = None
            if document_path.exists():
                file_hash = sha256_of_file(document_path)
            elif doc_hash:
                file_hash = doc_hash
            if not file_hash:
                raise FileNotFoundError(document_path)
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

            # Soporte directo: si nos pasan la ruta del JSON de OCR 'ocr_results_for_*.json', leerlo directo
            try:
                if document_path.suffix.lower() == ".json" and document_path.name.startswith("ocr_results_for_"):
                    if document_path.exists():
                        with open(document_path, 'r', encoding='utf-8') as fh:
                            data = json.load(fh)
                        if debug:
                            logger.debug(f"OCR_CACHE_DEBUG: lectura directa de JSON: {document_path}")
                        return data
            except Exception:
                pass

            case_hashes: Dict[str, str] = {}
            doc_ids_by_name: Dict[str, str] = {}
            doc_ids_by_hash: Dict[str, str] = {}
            if case_id:
                try:
                    case_index = self.get_case_index(case_id) or {}
                    raw_hashes = case_index.get('document_hashes') or {}
                    if isinstance(raw_hashes, dict):
                        case_hashes = dict(raw_hashes)
                except Exception:
                    case_hashes = {}

                try:
                    with get_conn() as conn:
                        rows = conn.execute(
                            "SELECT id, filename, file_hash FROM documents WHERE case_id=?",
                            (case_id,)
                        ).fetchall()
                        for row in rows:
                            fname = row["filename"]
                            fhash = row["file_hash"]
                            if fname and fname not in doc_ids_by_name:
                                doc_ids_by_name[fname] = row["id"]
                            if fhash and fhash not in doc_ids_by_hash:
                                doc_ids_by_hash[fhash] = row["id"]
                except Exception:
                    pass

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
                    target_id = None
                    if case_id:
                        target_id = doc_ids_by_name.get(document_path.name)
                        if not target_id:
                            doc_hash = case_hashes.get(str(document_path))
                            if not doc_hash:
                                try:
                                    doc_hash = case_hashes.get(str(Path(str(document_path)).resolve()))
                                except Exception:
                                    doc_hash = None
                            if doc_hash:
                                target_id = doc_ids_by_hash.get(doc_hash)
                            if not target_id and doc_hash:
                                row = get_document_by_case_and_hash(case_id, doc_hash)
                                if row:
                                    target_id = row["id"]
                    if not target_id:
                        target_id, _ = ensure_document_registered(case_id or 'unknown', str(document_path))
                    persist_ocr(target_id, data, "azure", "full")
                except Exception:
                    pass
                return data

            # 2) Shards por hash
            doc_hash = case_hashes.get(str(document_path)) if case_id else None
            if not doc_hash and case_id:
                try:
                    doc_hash = case_hashes.get(str(Path(str(document_path)).resolve()))
                except Exception:
                    doc_hash = None
            cache_path: Optional[Path]
            if doc_hash:
                cache_path = self.cache_dir / doc_hash[:2] / f"{doc_hash}.json"
            else:
                try:
                    cache_path = self._get_cache_path(document_path)
                except FileNotFoundError:
                    cache_path = None
                except Exception:
                    cache_path = None
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
                try:
                    target_id = None
                    if case_id:
                        if doc_hash:
                            target_id = doc_ids_by_hash.get(doc_hash)
                            if not target_id:
                                row = get_document_by_case_and_hash(case_id, doc_hash)
                                if row:
                                    target_id = row["id"]
                        if not target_id:
                            target_id = doc_ids_by_name.get(document_path.name)
                    if not target_id:
                        target_id, doc_hash = ensure_document_registered(case_id or 'unknown', str(document_path))
                    persist_ocr(target_id, data, "azure", "full")
                except Exception:
                    pass
                return data
            
            # Cache miss
            increment_cache_stats('global', 'ocr_misses')
            if case_id:
                increment_cache_stats(case_id, 'ocr_misses')
            # 3) Fallback a DB: intentar por hash global y/o por documento registrado
            try:
                doc_id = None
                file_hash = None
                doc_hash = case_hashes.get(str(document_path)) if case_id else None
                if not doc_hash and case_id:
                    try:
                        doc_hash = case_hashes.get(str(Path(str(document_path)).resolve()))
                    except Exception:
                        doc_hash = None

                if case_id and doc_hash:
                    file_hash = doc_hash
                    doc_id = doc_ids_by_hash.get(doc_hash)
                    if not doc_id:
                        row = get_document_by_case_and_hash(case_id, doc_hash)
                        if row:
                            doc_id = row["id"]

                if not doc_id:
                    try:
                        doc_id, file_hash = ensure_document_registered(case_id or 'unknown', str(document_path))
                    except Exception as register_exc:
                        logger.debug(f"No se pudo asegurar documento {document_path}: {register_exc}")
                        doc_id = None
                        file_hash = None

                if doc_id and file_hash:
                    cached = try_get_cached_ocr(doc_id, file_hash, allow_global=True, case_id=case_id)
                    if cached:
                        if debug:
                            logger.debug(f"OCR_CACHE_DEBUG: lectura desde DB por hash: {file_hash} -> doc_id {doc_id}")
                        try:
                            self.save_cache(document_path, cached, case_id)
                        except Exception as write_err:
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
            case_data: Dict[str, Any] = {}
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
                            doc_id = None
                            try:
                                with get_conn() as conn:
                                    row = conn.execute(
                                        "SELECT id FROM documents WHERE case_id=? AND filename=?",
                                        (case_id, document_path.name)
                                    ).fetchone()
                                    if row:
                                        doc_id = row["id"]
                            except Exception:
                                doc_id = None

                            if not doc_id:
                                try:
                                    doc_id, _ = ensure_document_registered(case_id, str(document_path))
                                except Exception:
                                    doc_id = None

                            if doc_id:
                                engine, version = self._resolve_engine_hint(ocr_result)
                                persist_ocr(doc_id, ocr_result, engine, version)
                        except Exception as persist_exc:
                            logger.debug(f"No se pudo persistir OCR en DB para {document_path}: {persist_exc}")
                        if not COPY_DOCUMENTS_IN_REORG:
                            pdf_destination = dest_dir / document_path.name
                            if pdf_destination.exists():
                                try:
                                    pdf_destination.unlink()
                                except Exception:
                                    pass
                        try:
                            self._persist_gps_artifacts(case_id, document_path, ocr_result)
                        except Exception:
                            logger.debug("No se pudieron persistir artefactos GPS para %s", document_path.name)
                        try:
                            self._update_case_index_metadata(case_id, document_path, ocr_result)
                        except Exception as exc:
                            logger.debug("No se pudo actualizar case_index con metadata GPS %s: %s", document_path.name, exc)
                        return
                    except Exception as e:
                        logger.warning(f"Fallo guardando en vista humana; guardando en shard. Detalle: {e}")

            # Fallback a shards por hash
            cache_path = None
            try:
                cache_path = self._get_cache_path(document_path)
            except FileNotFoundError:
                cache_path = None
            except Exception:
                cache_path = None
            if cache_path:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(ocr_result, f, ensure_ascii=False, indent=2, default=str)
                logger.debug(f"Caché guardado (shard) para {document_path.name}: {cache_path}")
            # Persistir en DB aunque no exista índice aún
            try:
                doc_id = None
                if case_id:
                    try:
                        with get_conn() as conn:
                            row = conn.execute(
                                "SELECT id FROM documents WHERE case_id=? AND filename=?",
                                (case_id, document_path.name)
                            ).fetchone()
                            if row:
                                doc_id = row["id"]
                    except Exception:
                        doc_id = None
                if not doc_id:
                    try:
                        doc_id, _ = ensure_document_registered(case_id or 'unknown', str(document_path))
                    except Exception:
                        doc_id = None
                if doc_id:
                    engine, version = self._resolve_engine_hint(ocr_result)
                    persist_ocr(doc_id, ocr_result, engine, version)
            except Exception as persist_exc:
                logger.debug(f"No se pudo persistir OCR en DB para {document_path}: {persist_exc}")
            if case_id:
                try:
                    self._persist_gps_artifacts(case_id, document_path, ocr_result)
                except Exception:
                    logger.debug("No se pudieron persistir artefactos GPS para %s", document_path.name)
                try:
                    self._update_case_index_metadata(case_id, document_path, ocr_result)
                except Exception as exc:
                    logger.debug("No se pudo actualizar case_index con metadata GPS %s: %s", document_path.name, exc)
        except Exception as e:
            logger.error(f"Error guardando caché para {document_path}: {e}")

    def save_case_index(self, case_id: str, case_data: Dict[str, Any]) -> None:
        """
        Guarda el índice de archivos de un caso para futura reorganización.
        """
        from fraud_scorer.storage.cases import get_conn

        try:
            if case_data is None:
                case_data = {}

            # Mantener nombre de carpeta del caso alineado con los datos actuales
            insured = case_data.get('insured_name') or "SIN_NOMBRE"
            claim = case_data.get('claim_number') or case_id
            case_data['case_folder'] = f"{self._sanitize_filename(insured)} - {self._sanitize_filename(claim)}"

            try:
                self.gps_cache.attach_to_case_index(case_id, case_data)
            except Exception as exc:
                logger.debug("No se pudo adjuntar metadata GPS a case_index %s: %s", case_id, exc)

            # Sincronizar con BD: actualizar base_path si hay carpeta reorganizada
            try:
                if "case_folder" in case_data:
                    reorganized_path = self.cache_dir / case_data["case_folder"]
                    if reorganized_path.exists():
                        with get_conn() as conn:
                            conn.execute(
                                "UPDATE cases SET base_path = ? WHERE case_id = ?",
                                (str(reorganized_path), case_id)
                            )
                            conn.commit()
                            logger.debug(f"Base_path sincronizado en BD para {case_id}: {reorganized_path}")
            except Exception as e:
                logger.debug(f"No se pudo sincronizar base_path: {e}")

            # Guardar índice
            index_path = self.index_dir / f"{case_id}.json"
            case_data["processed_at"] = datetime.now().isoformat()
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(case_data, f, ensure_ascii=False, indent=2, default=str)
            logger.debug(f"Índice de caso guardado: {index_path}")
        except Exception as e:
            logger.error(f"Error guardando índice del caso {case_id}: {e}")

    def _persist_gps_artifacts(self, case_id: str, document_path: Path, ocr_result: Dict[str, Any]) -> None:
        self.gps_cache.persist_direct_output(case_id, document_path, ocr_result)

    def _resolve_engine_hint(self, ocr_result: Dict[str, Any]) -> Tuple[str, str]:
        metadata = ocr_result.get("metadata") or {}
        gps_meta = metadata.get("gps_direct") or {}
        if gps_meta.get("enabled"):
            hint = gps_meta.get("hint") or {}
            version = hint.get("detector_version") or "direct"
            return "gps_dal", str(version)
        return "azure", "full"

    def _update_case_index_metadata(
        self,
        case_id: str,
        document_path: Path,
        ocr_result: Dict[str, Any],
    ) -> None:
        if not case_id:
            return

        case_data = self.get_case_index(case_id) or {"case_id": case_id}
        case_data.setdefault("case_id", case_id)

        metadata = ocr_result.get("metadata") or {}
        hint_payload = metadata.get("processing_hint")
        if hint_payload:
            hints = case_data.setdefault("document_hints", {})
            hints[document_path.name] = hint_payload

        gps_meta = metadata.get("gps_direct") or {}
        if gps_meta:
            hint_dict = hint_payload if isinstance(hint_payload, dict) else {}
            source = "manual" if hint_dict.get("manual_override") else (hint_dict.get("source") or "auto")
            detector_version = hint_dict.get("detector_version") or gps_meta.get("schema_version")
            audit_entry = {
                "document_name": document_path.name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "enabled": bool(gps_meta.get("enabled")),
                "detector_version": detector_version,
                "confidence": hint_dict.get("confidence"),
                "source": source,
                "plugins_used": gps_meta.get("plugins_used"),
                "normalized_row_count": gps_meta.get("normalized_row_count"),
                "warnings": gps_meta.get("normalization_warnings"),
                "ingestion_stats": gps_meta.get("ingestion_stats"),
                "dataset_checksum": (gps_meta.get("dataset") or {}).get("checksum"),
                "dataset_rows": (gps_meta.get("dataset") or {}).get("row_count"),
            }
            audit_list = [
                entry
                for entry in case_data.get("gps_ingestion_audit", [])
                if entry.get("document_name") != document_path.name
            ]
            audit_list.append(audit_entry)
            case_data["gps_ingestion_audit"] = sorted(
                audit_list,
                key=lambda entry: entry.get("document_name") or "",
            )

        self.save_case_index(case_id, case_data)

    def get_case_folder_path(self, case_id: str, case_data: Optional[Dict[str, Any]] = None) -> Path:
        """Devuelve la ruta de carpeta reorganizada para un caso."""
        if case_data is None:
            case_data = self.get_case_index(case_id) or {}

        folder_name = case_data.get('case_folder')
        if not folder_name:
            insured = case_data.get('insured_name') or "SIN_NOMBRE"
            claim = case_data.get('claim_number') or case_id
            folder_name = f"{self._sanitize_filename(insured)} - {self._sanitize_filename(claim)}"

        return self.cache_dir / folder_name
    
    def _reconstruct_index_from_db(self, case_id: str, insured_name: str = None, claim_number: str = None) -> Optional[Dict[str, Any]]:
        """
        Reconstruye el índice de un caso desde la base de datos.
        """
        try:
            with get_conn() as conn:
                # Obtener información del caso
                case_row = conn.execute(
                    "SELECT * FROM cases WHERE case_id = ?",
                    (case_id,)
                ).fetchone()

                if not case_row:
                    return None

                # Obtener documentos del caso
                doc_rows = conn.execute(
                    "SELECT * FROM documents WHERE case_id = ? ORDER BY filename",
                    (case_id,)
                ).fetchall()

                # Reconstruir el índice
                case_index = {
                    "case_id": case_id,
                    "case_title": case_row["name"],
                    "insured_name": insured_name or "UNKNOWN",
                    "claim_number": claim_number or case_id,
                    "documents": [],
                    "cache_files": [],
                    "document_hashes": {},
                    "total_documents": len(doc_rows),
                    "folder_path": case_row["base_path"] or "",
                    "status": "reconstructed_from_db",
                    "processed_at": datetime.now().isoformat()
                }

                # Agregar información de cada documento
                for doc in doc_rows:
                    filepath = doc["filepath"] or doc["filename"]
                    if filepath:
                        case_index["documents"].append(filepath)
                        case_index["cache_files"].append(filepath)
                        if doc["file_hash"]:
                            case_index["document_hashes"][filepath] = doc["file_hash"]

                # Intentar obtener datos de extracción para mejorar el índice
                try:
                    extract_rows = conn.execute(
                        """
                        SELECT e.entities
                        FROM extracted_data e
                        JOIN documents d ON e.document_id = d.id
                        WHERE d.case_id = ?
                        LIMIT 1
                        """,
                        (case_id,)
                    ).fetchone()

                    if extract_rows and extract_rows["entities"]:
                        entities = json.loads(extract_rows["entities"])
                        # Buscar insured_name y claim_number en las entidades
                        for entity in entities:
                            if "insured" in entity.lower() and not insured_name:
                                case_index["insured_name"] = entities[entity]
                            if "claim" in entity.lower() and not claim_number:
                                case_index["claim_number"] = entities[entity]
                except Exception:
                    pass

                return case_index

        except Exception as e:
            logger.error(f"Error reconstruyendo índice desde BD: {e}")
            return None

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
            # Intentar reconstruir el índice desde la BD
            logger.warning(f"No se encontró el índice del caso {case_id}. Intentando reconstruir desde BD...")
            try:
                case_data = self._reconstruct_index_from_db(case_id, insured_name, claim_number)
                if case_data:
                    self.save_case_index(case_id, case_data)
                    logger.info(f"  ✓ Índice reconstruido exitosamente")
                else:
                    logger.error(f"No se pudo reconstruir el índice para {case_id}")
                    return
            except Exception as e:
                logger.error(f"Error reconstruyendo índice: {e}")
                return

        try:
            with open(case_index_path, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
        except Exception as e:
            logger.error(f"Error leyendo el índice del caso {case_id}: {e}")
            return

        document_hashes = case_data.get('document_hashes') or {}
        if not isinstance(document_hashes, dict):
            document_hashes = {}
        else:
            document_hashes = dict(document_hashes)

        hash_by_filename: Dict[str, str] = {}
        for stored_path, file_hash in document_hashes.items():
            try:
                filename = Path(stored_path).name
            except Exception:
                continue
            if filename and file_hash and filename not in hash_by_filename:
                hash_by_filename[filename] = file_hash

        db_hash_by_path: Dict[str, str] = {}
        db_hash_by_filename: Dict[str, str] = {}
        db_rows_by_path: Dict[str, Any] = {}
        db_rows_by_filename: Dict[str, Any] = {}
        try:
            with get_conn() as conn:
                rows = conn.execute(
                    "SELECT filename, filepath, file_hash FROM documents WHERE case_id = ?",
                    (case_id,)
                ).fetchall()
            for row in rows:
                filepath = row.get('filepath') if isinstance(row, dict) else row['filepath']
                file_hash = row.get('file_hash') if isinstance(row, dict) else row['file_hash']
                filename = row.get('filename') if isinstance(row, dict) else row['filename']

                if filepath and file_hash:
                    db_hash_by_path[filepath] = file_hash
                    db_rows_by_path[filepath] = row
                name_candidate = filename or (Path(filepath).name if filepath else '')
                if name_candidate and file_hash:
                    if name_candidate not in db_hash_by_filename:
                        db_hash_by_filename[name_candidate] = file_hash
                    if name_candidate not in db_rows_by_filename:
                        db_rows_by_filename[name_candidate] = row
        except Exception as db_exc:
            logger.warning(f"No se pudieron cargar hashes desde DB para {case_id}: {db_exc}")

        updated_hashes = dict(document_hashes)

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

        new_document_paths: list[str] = []
        for original_doc_path_str in original_paths:
            old_path_str = original_doc_path_str
            try:
                original_doc_path = Path(original_doc_path_str)
            except Exception as path_exc:
                logger.warning(f"Ruta inválida en índice para {case_id}: {original_doc_path_str} ({path_exc})")
                continue

            doc_filename = original_doc_path.name
            doc_hash = document_hashes.get(original_doc_path_str) or hash_by_filename.get(doc_filename)

            db_row = db_rows_by_path.get(original_doc_path_str) or db_rows_by_filename.get(doc_filename)
            if not doc_hash:
                if db_row:
                    try:
                        doc_hash = db_row['file_hash']
                    except Exception:
                        doc_hash = None
            if not doc_hash and original_doc_path.exists():
                try:
                    doc_hash = sha256_of_file(original_doc_path)
                except Exception as hash_exc:
                    logger.warning(f"No se pudo calcular hash para {original_doc_path}: {hash_exc}")
                    doc_hash = None
            if not doc_hash and db_row:
                db_path = db_row['filepath'] if isinstance(db_row, dict) else db_row['filepath']
                if db_path:
                    candidate_path = Path(db_path)
                    if candidate_path.exists():
                        try:
                            doc_hash = sha256_of_file(candidate_path)
                            original_doc_path = candidate_path
                            original_doc_path_str = str(candidate_path)
                        except Exception as hash_exc:
                            logger.warning(f"No se pudo calcular hash usando fallback {candidate_path}: {hash_exc}")
            doc_filename = original_doc_path.name
            if not doc_hash:
                logger.warning(f"No se encontró hash ni archivo para {original_doc_path_str}. Se omite de la reorganización.")
                continue

            hash_by_filename.setdefault(doc_filename, doc_hash)
            document_hashes.setdefault(original_doc_path_str, doc_hash)

            # Manejar directamente archivos JSON reorganizados (ocr_results_for_*.json)
            try:
                if original_doc_path.suffix.lower() == ".json" and original_doc_path.name.startswith("ocr_results_for_"):
                    # Derivar carpeta de documento a partir del nombre original incluido en el JSON
                    try:
                        base_part = original_doc_path.name[len("ocr_results_for_"):-len(".json")]
                        doc_folder_name = self._sanitize_filename(Path(base_part).stem)
                    except Exception:
                        doc_folder_name = self._sanitize_filename(original_doc_path.stem)

                    doc_specific_path = new_case_path / doc_folder_name
                    doc_specific_path.mkdir(parents=True, exist_ok=True)
                    destination_path = doc_specific_path / original_doc_path.name

                    if original_doc_path != destination_path:
                        try:
                            if destination_path.exists():
                                destination_path.unlink()
                            logger.info(f"Moviendo JSON {original_doc_path} -> {destination_path}")
                            shutil.move(str(original_doc_path), str(destination_path))
                        except Exception as move_exc:
                            logger.warning(f"No se pudo mover JSON de caché {original_doc_path}: {move_exc}")

                    new_document_paths.append(str(destination_path))
                    updated_hashes.pop(old_path_str, None)
                    updated_hashes[str(destination_path)] = doc_hash
                    # No procesar este archivo como si fuera un original
                    continue
            except Exception:
                pass

            shard_candidate = self.cache_dir / doc_hash[:2] / f"{doc_hash}.json"
            cache_path: Optional[Path] = shard_candidate if shard_candidate.exists() else None
            cache_from_shard = cache_path is not None

            doc_folder_name = self._sanitize_filename(original_doc_path.stem)
            doc_specific_path = new_case_path / doc_folder_name
            doc_specific_path.mkdir(parents=True, exist_ok=True)

            sanitized_name = self._sanitize_filename(original_doc_path.name)
            destination_path = doc_specific_path / f"ocr_results_for_{sanitized_name}.json"

            if (not cache_path or not cache_path.exists()) and destination_path.exists():
                cache_path = destination_path
                cache_from_shard = False

            if cache_path and cache_path.exists() and cache_path != destination_path:
                try:
                    if cache_from_shard:
                        hash_folders_to_clean.add(cache_path.parent)
                    if destination_path.exists():
                        destination_path.unlink()
                    logger.info(f"Moviendo {cache_path} -> {destination_path}")
                    shutil.move(str(cache_path), str(destination_path))
                except Exception as move_exc:
                    logger.error(f"No se pudo mover el archivo de caché {cache_path}: {move_exc}")

            source_for_copy = original_doc_path
            if not source_for_copy.exists() and db_row:
                db_path = db_row['filepath'] if isinstance(db_row, dict) else db_row['filepath']
                if db_path and Path(db_path).exists():
                    source_for_copy = Path(db_path)

            pdf_destination = doc_specific_path / original_doc_path.name
            if COPY_DOCUMENTS_IN_REORG and source_for_copy.exists():
                try:
                    if not pdf_destination.exists():
                        shutil.copy2(str(source_for_copy), str(pdf_destination))
                except Exception as copy_exc:
                    logger.warning(f"No se pudo copiar original {source_for_copy} -> {pdf_destination}: {copy_exc}")
            elif not COPY_DOCUMENTS_IN_REORG and pdf_destination.exists():
                try:
                    pdf_destination.unlink()
                except Exception:
                    pass

            if not pdf_destination.exists():
                logger.debug(f"No se materializó copia para {pdf_destination}; se usará solo el JSON cache")

            # Registrar el JSON como documento de referencia
            new_document_paths.append(str(destination_path))
            updated_hashes.pop(old_path_str, None)
            updated_hashes[str(destination_path)] = doc_hash

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

        try:
            if new_document_paths:
                unique_docs = []
                seen = set()
                for path_str in new_document_paths:
                    if path_str not in seen:
                        unique_docs.append(path_str)
                        seen.add(path_str)
                case_data['documents'] = unique_docs
                # Mantener 'cache_files' alineado con documentos (JSON) y sin duplicados
                case_data['cache_files'] = list(unique_docs)
                # Filtrar 'document_hashes' para que sólo incluya claves presentes en 'documents'
                try:
                    doc_hashes = case_data.get('document_hashes') or {}
                    if isinstance(doc_hashes, dict):
                        filtered = {k: v for k, v in doc_hashes.items() if k in set(unique_docs)}
                        case_data['document_hashes'] = filtered
                except Exception:
                    pass
                # Ajustar total de documentos
                case_data['total_documents'] = len(unique_docs)
                # Ya no actualizamos 'filepath' en DB con rutas a JSON; mantener DB con metadatos del original

            case_data['case_folder'] = new_case_folder_name
            case_data['folder_path'] = str(new_case_path)
            if updated_hashes:
                case_data['document_hashes'] = updated_hashes
            self.save_case_index(case_id, case_data)

            try:
                with get_conn() as conn:
                    conn.execute(
                        "UPDATE cases SET base_path=? WHERE case_id=?",
                        (str(new_case_path), case_id)
                    )
            except Exception as db_exc:
                logger.warning(f"No se pudo actualizar base_path en DB para {case_id}: {db_exc}")
        except Exception as exc:
            logger.warning(f"No se pudo actualizar case_folder en índice {case_id}: {exc}")
    
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

    def get_case_index(self, case_id: str, auto_reconstruct: bool = False) -> Optional[Dict[str, Any]]:
        """
        Obtiene la información del índice de un caso específico.
        Si auto_reconstruct=True y el índice no existe, intenta reconstruirlo desde la BD.
        """
        index_path = self.index_dir / f"{case_id}.json"

        if not index_path.exists():
            if auto_reconstruct:
                try:
                    case_data = self._reconstruct_index_from_db(case_id)
                    if case_data:
                        # Persistir para uso futuro y mantener nomenclatura consistente
                        self.save_case_index(case_id, case_data)
                        return case_data
                except Exception as e:
                    logger.debug(f"Auto-reconstrucción falló para {case_id}: {e}")
            return None

        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error leyendo índice del caso {case_id}: {e}")
            return None

    def list_cases_from_cache(self) -> List[Dict[str, Any]]:
        """Devuelve un resumen de los casos disponibles en el índice."""
        cases: List[Dict[str, Any]] = []

        if not self.index_dir.exists():
            return cases

        try:
            for index_file in sorted(self.index_dir.glob("*.json")):
                case_id = index_file.stem
                try:
                    with open(index_file, "r", encoding="utf-8") as fh:
                        case_data = json.load(fh)
                except Exception as exc:
                    logger.error(f"No se pudo leer índice {index_file.name}: {exc}")
                    continue

                case_title = case_data.get("case_title") or case_id
                insured = case_data.get("insured_name") or ""
                claim = case_data.get("claim_number") or ""
                processed_at = case_data.get("processed_at") or ""
                total_documents = case_data.get("total_documents")
                if total_documents is None:
                    documents = case_data.get("documents") or []
                    total_documents = len(documents)

                cases.append(
                    {
                        "case_id": case_id,
                        "case_title": case_title,
                        "insured_name": insured,
                        "claim_number": claim,
                        "processed_at": processed_at,
                        "folder_path": case_data.get("folder_path", ""),
                        "total_documents": total_documents,
                    }
                )

        except Exception as exc:
            logger.error(f"Error listando casos del cache: {exc}")
            return cases

        cases.sort(key=lambda item: item.get("processed_at", ""), reverse=True)
        return cases

    def save_manual_classifications(
        self,
        case_id: str,
        classifications: Dict[str, str],
        ai_classifications: Optional[Dict[str, str]] = None,
        *,
        replace: bool = False,
    ) -> None:
        """
        Guarda las clasificaciones manuales del usuario para un caso, junto con las originales de AI.

        Args:
            case_id: ID del caso
            classifications: Dict con {filename: document_type} - clasificaciones finales
            ai_classifications: Dict con {filename: document_type} - clasificaciones originales de AI
        """
        try:
            # Cargar índice existente
            case_data = self.get_case_index(case_id) or {}

            # Añadir o actualizar clasificaciones manuales
            if replace:
                case_data['manual_classifications'] = dict(classifications)
            else:
                if 'manual_classifications' not in case_data:
                    case_data['manual_classifications'] = {}
                case_data['manual_classifications'].update(classifications)

            if not case_data.get('manual_classifications'):
                case_data.pop('manual_classifications', None)
                case_data.pop('last_manual_update', None)
            else:
                case_data['last_manual_update'] = time.strftime('%Y-%m-%d %H:%M:%S')

            # Guardar también las clasificaciones originales de AI si se proporcionan
            if ai_classifications is not None:
                if replace:
                    case_data['ai_classifications'] = dict(ai_classifications)
                else:
                    if 'ai_classifications' not in case_data:
                        case_data['ai_classifications'] = {}
                    case_data['ai_classifications'].update(ai_classifications)

            # Guardar índice actualizado
            self.save_case_index(case_id, case_data)
            logger.info(f"Clasificaciones guardadas para caso {case_id}")

        except Exception as e:
            logger.error(f"Error guardando clasificaciones: {e}")

    def get_manual_classifications(self, case_id: str) -> Optional[Dict[str, str]]:
        """
        Obtiene las clasificaciones manuales previas de un caso.

        Returns:
            Dict con {filename: document_type} o None si no hay clasificaciones previas
        """
        case_data = self.get_case_index(case_id)
        if case_data and 'manual_classifications' in case_data:
            return case_data['manual_classifications']
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
COPY_DOCUMENTS_IN_REORG = os.getenv("FS_COPY_DOCS_IN_CACHE", "false").lower() == "true"
