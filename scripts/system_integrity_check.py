#!/usr/bin/env python3
"""
Script para verificar la integridad del sistema y detectar inconsistencias.
"""

import sqlite3
from pathlib import Path
import json
import sys
from collections import defaultdict

def check_system_integrity():
    """Verifica la integridad del sistema completo"""

    print("🔍 VERIFICACIÓN DE INTEGRIDAD DEL SISTEMA")
    print("=" * 60)

    issues = []
    warnings = []

    # 1. Verificar base de datos
    try:
        conn = sqlite3.connect("data/cases.db")

        # Contar registros
        cases_count = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        docs_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        ocr_count = conn.execute("SELECT COUNT(*) FROM ocr_results").fetchone()[0]

        print(f"\n📊 BASE DE DATOS:")
        print(f"  - Casos: {cases_count}")
        print(f"  - Documentos: {docs_count}")
        print(f"  - Resultados OCR: {ocr_count}")

        # Verificar casos con rutas inválidas
        cursor = conn.execute("SELECT case_id, base_path FROM cases")
        for row in cursor:
            case_id, base_path = row
            if base_path and not Path(base_path).exists():
                issues.append(f"Caso {case_id}: ruta base no existe ({base_path})")

        # Verificar documentos huérfanos
        orphan_docs = conn.execute("""
            SELECT COUNT(*) FROM documents d
            WHERE NOT EXISTS (SELECT 1 FROM cases c WHERE c.case_id = d.case_id)
        """).fetchone()[0]

        if orphan_docs > 0:
            issues.append(f"Hay {orphan_docs} documentos huérfanos en la BD")

        conn.close()
    except Exception as e:
        issues.append(f"Error accediendo a la BD: {e}")

    # 2. Verificar archivos de índice
    print(f"\n📁 ÍNDICES DE CACHE:")
    index_dir = Path("data/ocr_cache/case_index")
    if index_dir.exists():
        index_files = list(index_dir.glob("*.json"))
        print(f"  - Archivos de índice: {len(index_files)}")

        # Verificar que cada índice corresponda a un caso válido
        conn = sqlite3.connect("data/cases.db")
        valid_cases = {row[0] for row in conn.execute("SELECT case_id FROM cases")}
        conn.close()

        for index_file in index_files:
            case_id = index_file.stem
            if case_id not in valid_cases:
                issues.append(f"Índice huérfano: {case_id} (no existe en BD)")

            # Verificar contenido del índice
            try:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)

                # Verificar que los documentos referenciados existan
                docs = index_data.get('documents', [])
                for doc_path in docs:
                    if not Path(doc_path).exists():
                        warnings.append(f"Caso {case_id}: documento no existe ({Path(doc_path).name})")
            except Exception as e:
                issues.append(f"Error leyendo índice {case_id}: {e}")
    else:
        warnings.append("No existe directorio de índices")

    # 3. Verificar carpetas reorganizadas
    print(f"\n📂 CARPETAS REORGANIZADAS:")
    ocr_cache_dir = Path("data/ocr_cache")
    reorganized_folders = []

    if ocr_cache_dir.exists():
        for folder in ocr_cache_dir.iterdir():
            if folder.is_dir() and folder.name != "case_index":
                reorganized_folders.append(folder.name)

        print(f"  - Carpetas encontradas: {len(reorganized_folders)}")

        # Verificar que cada carpeta corresponda a un caso
        conn = sqlite3.connect("data/cases.db")
        for folder_name in reorganized_folders:
            cursor = conn.execute(
                "SELECT case_id FROM cases WHERE base_path LIKE ?",
                (f"%{folder_name}%",)
            )
            if not cursor.fetchone():
                # Buscar en índices
                found_in_index = False
                if index_dir.exists():
                    for index_file in index_dir.glob("*.json"):
                        try:
                            with open(index_file, 'r') as f:
                                idx = json.load(f)
                                if folder_name == idx.get('case_folder'):
                                    found_in_index = True
                                    break
                        except:
                            pass

                if not found_in_index:
                    issues.append(f"Carpeta huérfana: {folder_name}")
        conn.close()

    # 4. Verificar reportes
    print(f"\n📄 REPORTES:")
    reports_dir = Path("data/reports")
    report_count = 0

    if reports_dir.exists():
        report_files = [f for f in reports_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
        report_count = len(report_files)
        print(f"  - Reportes encontrados: {report_count}")

        # Verificar reportes huérfanos
        conn = sqlite3.connect("data/cases.db")
        valid_cases = {row[0] for row in conn.execute("SELECT case_id FROM cases")}
        conn.close()

        for report in report_files:
            belongs_to_case = False
            for case_id in valid_cases:
                if case_id in report.name:
                    belongs_to_case = True
                    break

            if not belongs_to_case:
                issues.append(f"Reporte huérfano: {report.name}")

    # 5. Verificar integridad de cache
    print(f"\n🔗 INTEGRIDAD DE CACHE:")
    cache_stats = defaultdict(int)

    # Contar archivos de cache por tipo
    if ocr_cache_dir.exists():
        # Shards (archivos hash)
        for shard_dir in ocr_cache_dir.iterdir():
            if shard_dir.is_dir() and len(shard_dir.name) == 2:  # Directorios de 2 caracteres
                for json_file in shard_dir.glob("*.json"):
                    cache_stats['shards'] += 1

    print(f"  - Shards de cache: {cache_stats['shards']}")

    # 6. Verificar archivos temporales
    print(f"\n🗑️ ARCHIVOS TEMPORALES:")
    temp_dir = Path("data/temp")
    if temp_dir.exists():
        temp_folders = [f for f in temp_dir.iterdir() if f.is_dir() and f.name != "pipeline_cache"]
        print(f"  - Carpetas temporales: {len(temp_folders)}")

        pipeline_cache = Path("data/temp/pipeline_cache")
        if pipeline_cache.exists():
            pipeline_files = list(pipeline_cache.glob("*"))
            print(f"  - Archivos en pipeline_cache: {len(pipeline_files)}")

    # 7. Verificar duplicación de bases de datos
    print(f"\n💾 ARCHIVOS DE BASE DE DATOS:")
    db_files = list(Path("data").glob("*.db"))
    for db_file in db_files:
        size = db_file.stat().st_size
        print(f"  - {db_file.name}: {size:,} bytes")

        if db_file.name == "fraud_scorer.db" and size == 0:
            issues.append("Archivo fraud_scorer.db vacío (debe ser eliminado)")

    # Resumen
    print("\n" + "=" * 60)
    print("📋 RESUMEN:")

    if not issues and not warnings:
        print("✅ Sistema en perfecto estado - No se detectaron problemas")
    else:
        if warnings:
            print(f"\n⚠️ ADVERTENCIAS ({len(warnings)}):")
            for warning in warnings[:5]:
                print(f"  - {warning}")
            if len(warnings) > 5:
                print(f"  ... y {len(warnings) - 5} más")

        if issues:
            print(f"\n❌ PROBLEMAS DETECTADOS ({len(issues)}):")
            for issue in issues[:10]:
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... y {len(issues) - 10} más")

            print("\n💡 RECOMENDACIÓN:")
            print("  Ejecuta: python scripts/clean_orphaned_files.py")
            print("  para limpiar archivos huérfanos")

    return len(issues) == 0

if __name__ == "__main__":
    success = check_system_integrity()
    sys.exit(0 if success else 1)