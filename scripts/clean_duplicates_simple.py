#!/usr/bin/env python3
"""
Script simplificado para limpiar casos duplicados.
Estrategia: mantener solo documentos únicos por hash y un solo caso.
"""

import sqlite3
from pathlib import Path
import logging
from datetime import datetime
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_duplicates_simple():
    db_path = Path("data/cases.db")
    backup_path = Path(f"data/cases_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
    
    if not db_path.exists():
        logger.error(f"Base de datos no encontrada: {db_path}")
        return
    
    # Crear backup
    logger.info(f"Creando backup en: {backup_path}")
    shutil.copy2(db_path, backup_path)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        conn.execute("BEGIN TRANSACTION")
        
        # Análisis inicial
        logger.info("=== ANÁLISIS INICIAL ===")
        total_cases = cursor.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        total_docs = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        unique_hashes = cursor.execute("SELECT COUNT(DISTINCT file_hash) FROM documents WHERE file_hash IS NOT NULL").fetchone()[0]
        total_ocr = cursor.execute("SELECT COUNT(*) FROM ocr_results").fetchone()[0]
        
        logger.info(f"Casos: {total_cases}, Documentos: {total_docs}, Únicos: {unique_hashes}, OCR: {total_ocr}")
        
        # Paso 1: Eliminar documentos duplicados (mantener el más antiguo por hash)
        logger.info("\n=== PASO 1: Eliminar documentos duplicados ===")
        cursor.execute("""
            DELETE FROM documents 
            WHERE id NOT IN (
                SELECT MIN(id) 
                FROM documents 
                WHERE file_hash IS NOT NULL 
                GROUP BY file_hash
            )
        """)
        deleted_docs = cursor.rowcount
        logger.info(f"✓ Eliminados {deleted_docs} documentos duplicados")
        
        # Paso 2: Mantener solo un caso (el más reciente con documentos)
        logger.info("\n=== PASO 2: Consolidar en un solo caso ===")
        cursor.execute("""
            SELECT c.case_id, c.name, c.created_at, COUNT(d.id) as doc_count
            FROM cases c
            LEFT JOIN documents d ON c.case_id = d.case_id
            GROUP BY c.case_id
            HAVING doc_count > 0
            ORDER BY c.created_at DESC
            LIMIT 1
        """)
        keep_case = cursor.fetchone()
        
        if not keep_case:
            logger.error("No se encontró caso con documentos")
            conn.rollback()
            return
            
        case_to_keep = keep_case['case_id']
        logger.info(f"✓ Manteniendo caso: {case_to_keep}")
        
        # Paso 3: Mover todos los documentos únicos al caso elegido
        cursor.execute("""
            UPDATE documents 
            SET case_id = ? 
            WHERE case_id != ?
        """, (case_to_keep, case_to_keep))
        updated_docs = cursor.rowcount
        logger.info(f"✓ Movidos {updated_docs} documentos al caso {case_to_keep}")
        
        # Paso 4: Eliminar casos vacíos
        cursor.execute("DELETE FROM cases WHERE case_id != ?", (case_to_keep,))
        deleted_cases = cursor.rowcount
        logger.info(f"✓ Eliminados {deleted_cases} casos vacíos")
        
        # Paso 5: Limpiar OCR huérfanos
        cursor.execute("""
            DELETE FROM ocr_results 
            WHERE document_id NOT IN (SELECT id FROM documents)
        """)
        deleted_ocr = cursor.rowcount
        logger.info(f"✓ Eliminados {deleted_ocr} resultados OCR huérfanos")
        
        # Paso 6: Actualizar el caso mantenido
        cursor.execute("""
            UPDATE cases 
            SET name = '20250000002494 - MODA YKT, S.A. DE C.V',
                status = 'processed',
                updated_at = ?
            WHERE case_id = ?
        """, (datetime.now().isoformat(), case_to_keep))
        
        # Verificación final
        logger.info("\n=== RESULTADO FINAL ===")
        final_cases = cursor.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        final_docs = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        final_ocr = cursor.execute("SELECT COUNT(*) FROM ocr_results").fetchone()[0]
        
        logger.info(f"Casos: {final_cases} (antes: {total_cases})")
        logger.info(f"Documentos: {final_docs} (antes: {total_docs})")
        logger.info(f"OCR: {final_ocr} (antes: {total_ocr})")
        
        # Confirmar
        conn.commit()
        logger.info(f"\n✅ LIMPIEZA EXITOSA - Caso único: {case_to_keep}")
        
        # Optimizar
        conn.execute("VACUUM")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        conn.rollback()
        logger.info("❌ Operación revertida")
    finally:
        conn.close()

if __name__ == "__main__":
    clean_duplicates_simple()