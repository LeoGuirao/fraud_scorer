#!/usr/bin/env python3
"""
Script para reconstruir el sistema de índices de cache desde la base de datos.
Crea los archivos de índice y reconstruye los archivos de cache OCR.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import logging
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def rebuild_cache_from_db():
    """
    Reconstruye los índices de cache y los archivos OCR desde la base de datos.
    """
    
    db_path = Path("data/cases.db")
    cache_dir = Path("data/ocr_cache")
    index_dir = cache_dir / "case_index"
    
    if not db_path.exists():
        logger.error(f"Base de datos no encontrada: {db_path}")
        return
    
    # Crear directorios necesarios
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorios creados/verificados")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Obtener el caso único
        cursor.execute("SELECT * FROM cases")
        case = cursor.fetchone()
        
        if not case:
            logger.error("No se encontró ningún caso en la base de datos")
            return
            
        case_id = case['case_id']
        case_name = case['name']
        
        # Extraer información del nombre
        parts = case_name.split(' - ', 1)
        if len(parts) == 2:
            claim_number = parts[0].strip()
            insured_name = parts[1].strip()
        else:
            claim_number = ""
            insured_name = case_name.strip()
        
        logger.info(f"Procesando caso {case_id}: {insured_name} - {claim_number}")
        
        # Obtener documentos con sus resultados OCR
        cursor.execute("""
            SELECT 
                d.id as doc_id,
                d.filename,
                d.filepath,
                d.file_hash,
                d.mime_type,
                d.size_bytes,
                d.page_count,
                d.language,
                d.created_at as doc_created,
                o.raw_text,
                o.key_value_pairs,
                o.tables,
                o.entities,
                o.confidence,
                o.metadata,
                o.errors,
                o.engine,
                o.processed_at as ocr_date
            FROM documents d
            LEFT JOIN ocr_results o ON d.id = o.document_id
            WHERE d.case_id = ?
            ORDER BY d.created_at
        """, (case_id,))
        
        documents = cursor.fetchall()
        logger.info(f"Encontrados {len(documents)} documentos")
        
        doc_paths = []
        cache_files = []
        ocr_count = 0
        
        for doc in documents:
            # Agregar a la lista de documentos
            if doc['filepath']:
                doc_paths.append(doc['filepath'])
            
            # Si tiene resultado OCR, reconstruir el archivo de cache
            if doc['raw_text'] is not None:
                # Reconstruir el resultado OCR
                ocr_result = {
                    "text": doc['raw_text'] or "",
                    "key_value_pairs": json.loads(doc['key_value_pairs'] or "{}"),
                    "tables": json.loads(doc['tables'] or "[]"),
                    "entities": json.loads(doc['entities'] or "[]"),
                    "confidence_scores": json.loads(doc['confidence'] or "{}"),
                    "metadata": json.loads(doc['metadata'] or "{}"),
                    "errors": json.loads(doc['errors'] or "[]"),
                    "engine": doc['engine'],
                    "processed_at": doc['ocr_date']
                }
                
                # Usar el hash del archivo
                file_hash = doc['file_hash']
                if file_hash:
                    # Crear la estructura de directorios para el cache (por hash)
                    subdir = file_hash[:2]
                    cache_subdir = cache_dir / subdir
                    cache_subdir.mkdir(parents=True, exist_ok=True)
                    
                    # Guardar el archivo de cache
                    cache_file_path = cache_subdir / f"{file_hash}.json"
                    with open(cache_file_path, 'w', encoding='utf-8') as f:
                        json.dump(ocr_result, f, ensure_ascii=False, indent=2, default=str)
                    
                    if doc['filepath']:
                        cache_files.append(doc['filepath'])
                    ocr_count += 1
                    logger.debug(f"  ✓ Cache OCR creado: {cache_file_path}")
        
        # Crear índice del caso
        index_file = index_dir / f"{case_id}.json"
        
        case_index = {
            "case_id": case_id,
            "case_title": case_name,
            "insured_name": insured_name,
            "claim_number": claim_number,
            "total_documents": len(documents),
            "documents": doc_paths,
            "cache_files": cache_files,
            "folder_path": case['base_path'] or "",
            "processed_at": case['created_at'],
            "status": case['status'],
            "rebuilt_at": datetime.now().isoformat(),
            "rebuilt_from": "database"
        }
        
        # Guardar índice
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(case_index, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"✓ Índice creado para caso {case_id}")
        logger.info(f"  - Documentos: {len(documents)}")
        logger.info(f"  - Archivos OCR reconstruidos: {ocr_count}")
        
        # Verificación final
        index_files = list(index_dir.glob("*.json"))
        cache_files_created = list(cache_dir.rglob("*.json"))
        cache_files_created = [f for f in cache_files_created if 'case_index' not in str(f)]
        
        logger.info("\n=== RECONSTRUCCIÓN COMPLETADA ===")
        logger.info(f"✓ Archivos de índice: {len(index_files)}")
        logger.info(f"✓ Archivos de cache OCR: {len(cache_files_created)}")
        logger.info(f"✓ Directorio de índices: {index_dir}")
        logger.info(f"✓ Directorio de cache: {cache_dir}")
        
    except Exception as e:
        logger.error(f"Error durante la reconstrucción: {e}", exc_info=True)
    finally:
        conn.close()

if __name__ == "__main__":
    logger.info("Iniciando reconstrucción del sistema de cache...")
    rebuild_cache_from_db()