#!/usr/bin/env python3
"""
Script para limpiar archivos huérfanos y casos antiguos del sistema.
"""

import sqlite3
from pathlib import Path
import shutil
import json
import sys

def get_valid_cases():
    """Obtiene los case_ids válidos de la BD"""
    try:
        conn = sqlite3.connect("data/cases.db")
        cursor = conn.execute("SELECT case_id FROM cases")
        valid_cases = {row[0] for row in cursor.fetchall()}
        conn.close()
        return valid_cases
    except Exception as e:
        print(f"Error leyendo BD: {e}")
        return set()

def clean_orphaned_files(dry_run=False):
    """Limpia archivos que no corresponden a ningún caso válido"""

    print("🔍 Buscando archivos huérfanos...")
    valid_cases = get_valid_cases()

    if not valid_cases:
        print("⚠️ No se encontraron casos válidos en la BD")
        response = input("¿Deseas limpiar TODOS los archivos? (s/n): ")
        if response.lower() != 's':
            print("Cancelado")
            return
    else:
        print(f"✓ Casos válidos encontrados: {valid_cases}")

    files_to_delete = []
    dirs_to_delete = []

    # 1. Limpiar reportes
    reports_dir = Path("data/reports")
    if reports_dir.exists():
        for file in reports_dir.iterdir():
            if file.is_file():
                # Verificar si el archivo corresponde a algún caso válido
                file_belongs_to_valid_case = False
                for case_id in valid_cases:
                    if case_id in file.name:
                        file_belongs_to_valid_case = True
                        break

                if not file_belongs_to_valid_case:
                    files_to_delete.append(file)

    # 2. Limpiar pipeline_cache
    pipeline_dir = Path("data/temp/pipeline_cache")
    if pipeline_dir.exists():
        for file in pipeline_dir.iterdir():
            if file.is_file():
                file_belongs_to_valid_case = False
                for case_id in valid_cases:
                    if case_id in file.name:
                        file_belongs_to_valid_case = True
                        break

                if not file_belongs_to_valid_case:
                    files_to_delete.append(file)

    # 3. Limpiar carpetas temporales
    temp_dir = Path("data/temp")
    if temp_dir.exists():
        for folder in temp_dir.iterdir():
            if folder.is_dir() and folder.name != "pipeline_cache":
                # Verificar si esta carpeta corresponde a algún caso válido
                folder_is_valid = False

                # Buscar en la BD si algún caso apunta a esta carpeta
                try:
                    conn = sqlite3.connect("data/cases.db")
                    cursor = conn.execute(
                        "SELECT case_id FROM cases WHERE base_path LIKE ?",
                        (f"%{folder.name}%",)
                    )
                    if cursor.fetchone():
                        folder_is_valid = True
                    conn.close()
                except:
                    pass

                if not folder_is_valid:
                    dirs_to_delete.append(folder)

    # 4. Limpiar índices huérfanos
    index_dir = Path("data/ocr_cache/case_index")
    if index_dir.exists():
        for index_file in index_dir.glob("*.json"):
            case_id = index_file.stem
            if case_id not in valid_cases:
                files_to_delete.append(index_file)

    # 5. Limpiar carpetas reorganizadas huérfanas
    ocr_cache_dir = Path("data/ocr_cache")
    if ocr_cache_dir.exists():
        for folder in ocr_cache_dir.iterdir():
            if folder.is_dir() and folder.name != "case_index":
                # Verificar si corresponde a algún caso válido
                folder_is_valid = False

                # Buscar en la BD
                try:
                    conn = sqlite3.connect("data/cases.db")
                    cursor = conn.execute(
                        "SELECT case_id FROM cases WHERE base_path LIKE ?",
                        (f"%{folder.name}%",)
                    )
                    if cursor.fetchone():
                        folder_is_valid = True
                    conn.close()
                except:
                    pass

                # También verificar si existe un índice que apunte a esta carpeta
                if not folder_is_valid and index_dir.exists():
                    for index_file in index_dir.glob("*.json"):
                        try:
                            with open(index_file, 'r') as f:
                                index_data = json.load(f)
                                if folder.name == index_data.get('case_folder'):
                                    folder_is_valid = True
                                    break
                        except:
                            pass

                if not folder_is_valid:
                    dirs_to_delete.append(folder)

    # Mostrar resumen
    print("\n📊 RESUMEN DE LIMPIEZA:")
    print(f"  - Archivos a eliminar: {len(files_to_delete)}")
    print(f"  - Carpetas a eliminar: {len(dirs_to_delete)}")

    if files_to_delete or dirs_to_delete:
        print("\n📁 ARCHIVOS A ELIMINAR:")
        for file in files_to_delete[:10]:  # Mostrar máximo 10
            print(f"  - {file}")
        if len(files_to_delete) > 10:
            print(f"  ... y {len(files_to_delete) - 10} más")

        print("\n📂 CARPETAS A ELIMINAR:")
        for folder in dirs_to_delete[:10]:  # Mostrar máximo 10
            print(f"  - {folder}")
        if len(dirs_to_delete) > 10:
            print(f"  ... y {len(dirs_to_delete) - 10} más")

        if not dry_run:
            response = input("\n¿Proceder con la eliminación? (s/n): ")
            if response.lower() == 's':
                # Eliminar archivos
                for file in files_to_delete:
                    try:
                        file.unlink()
                        print(f"  ✓ Eliminado: {file.name}")
                    except Exception as e:
                        print(f"  ✗ Error eliminando {file.name}: {e}")

                # Eliminar carpetas
                for folder in dirs_to_delete:
                    try:
                        shutil.rmtree(folder)
                        print(f"  ✓ Eliminada carpeta: {folder.name}")
                    except Exception as e:
                        print(f"  ✗ Error eliminando carpeta {folder.name}: {e}")

                print("\n✅ LIMPIEZA COMPLETADA")
            else:
                print("❌ Limpieza cancelada")
        else:
            print("\n⚠️ MODO DRY-RUN: No se eliminó nada")
    else:
        print("\n✅ No hay archivos huérfanos para limpiar")

def clean_all():
    """Limpia TODO el sistema (reseteo completo)"""
    print("\n⚠️ ⚠️ ⚠️  ADVERTENCIA: LIMPIEZA TOTAL ⚠️ ⚠️ ⚠️")
    print("Esto eliminará:")
    print("  - TODOS los casos de la base de datos")
    print("  - TODOS los archivos de cache")
    print("  - TODOS los reportes")
    print("  - TODAS las carpetas temporales")

    response = input("\n¿Estás SEGURO? Escribe 'ELIMINAR TODO' para confirmar: ")

    if response == "ELIMINAR TODO":
        print("\n🧹 Iniciando limpieza total...")

        # 1. Limpiar BD
        try:
            conn = sqlite3.connect("data/cases.db")
            conn.execute("DELETE FROM fraud_analyses")
            conn.execute("DELETE FROM ai_analyses")
            conn.execute("DELETE FROM extracted_data")
            conn.execute("DELETE FROM ocr_results")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM cases")
            conn.execute("DELETE FROM cache_stats")
            conn.execute("DELETE FROM runs")
            conn.commit()
            conn.close()
            print("  ✓ Base de datos limpiada")
        except Exception as e:
            print(f"  ✗ Error limpiando BD: {e}")

        # 2. Limpiar carpetas
        dirs_to_clean = [
            ("data/ocr_cache", True),   # Recrear
            ("data/temp", False),       # No eliminar la carpeta en sí
            ("data/reports", False),
            ("data/temp_reports", False),
            ("data/uploads", False),
            ("data/feedback_archive", False),
            ("data/raw", False),
        ]

        for dir_path, recreate in dirs_to_clean:
            dir_obj = Path(dir_path)
            if dir_obj.exists():
                if recreate:
                    # Eliminar y recrear
                    shutil.rmtree(dir_obj)
                    dir_obj.mkdir(parents=True, exist_ok=True)
                    print(f"  ✓ {dir_path} limpiado y recreado")
                else:
                    # Solo limpiar contenido
                    for item in dir_obj.iterdir():
                        if item.name.startswith('.'):  # Mantener .gitkeep
                            continue
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    print(f"  ✓ {dir_path} limpiado")

        print("\n🎆 LIMPIEZA TOTAL COMPLETADA - Sistema reiniciado")
    else:
        print("❌ Limpieza cancelada")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Limpiador de archivos huérfanos")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar qué se eliminaría sin hacerlo")
    parser.add_argument("--all", action="store_true", help="Limpiar TODO el sistema")

    args = parser.parse_args()

    if args.all:
        clean_all()
    else:
        clean_orphaned_files(dry_run=args.dry_run)
