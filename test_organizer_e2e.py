#!/usr/bin/env python3
"""
Test end-to-end del sistema de organización de documentos
"""

import asyncio
from pathlib import Path
import sys
import logging
import json
import tempfile
import shutil
from datetime import datetime

# Añadir el src al path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from fraud_scorer.processors.document_organizer import DocumentOrganizer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def create_test_documents(test_dir: Path):
    """Crear documentos de prueba en un directorio temporal"""
    logger.info(f"📁 Creando documentos de prueba en: {test_dir}")
    
    # Crear estructura de carpetas
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Documentos de prueba con contenido simulado
    test_files = {
        "DENUNCIA_INICIAL.pdf": "Documento de denuncia penal",
        "factura_12345.pdf": "Factura de mercancía",
        "guia_embarque_001.pdf": "Guía de embarque",
        "poliza_seguro.pdf": "Póliza de seguro de transporte",
        "bitacora_viaje.xlsx": "Bitácora del viaje",
        "GPS_tracking.csv": "Datos de rastreo GPS",
        "carta_reclamacion.pdf": "Carta de reclamación formal",
        "candados_seguridad.jpg": "Imagen de candados",
        "documento_random.txt": "Archivo no soportado",
        "._hidden_file.pdf": "Archivo oculto de macOS",
    }
    
    created_files = []
    for filename, content in test_files.items():
        file_path = test_dir / filename
        
        # Crear archivo según extensión
        if filename.endswith(('.pdf', '.xlsx', '.csv', '.jpg')):
            # Para archivos binarios, crear un archivo vacío
            file_path.touch()
        else:
            # Para archivos de texto
            file_path.write_text(content)
        
        created_files.append(filename)
        logger.info(f"  ✓ Creado: {filename}")
    
    return created_files


async def test_phase_a_classification():
    """Test de Fase A: Clasificación y Staging"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST FASE A: Clasificación y Staging")
    logger.info("=" * 60)
    
    # Crear directorio temporal
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_docs"
        await create_test_documents(test_dir)
        
        # Inicializar organizador
        organizer = DocumentOrganizer()
        
        # Ejecutar Fase A
        logger.info("\n🔍 Ejecutando Fase A...")
        staging_folder, mapping = await organizer.organize_documents_phase_a(
            input_folder=test_dir,
            use_llm_fallback=False  # Solo usar heurísticas para el test
        )
        
        # Verificar resultados
        assert staging_folder.exists(), "La carpeta de staging no fue creada"
        assert mapping, "El mapping está vacío"
        
        logger.info(f"\n📊 Métricas de Fase A:")
        total_files = len(mapping.get('files', []))
        classified = sum(1 for f in mapping.get('files', []) if f.get('document_type') != 'otros')
        unsupported = len(mapping.get('unsupported_files', []))
        logger.info(f"  - Total archivos: {total_files}")
        logger.info(f"  - Clasificados: {classified}")
        logger.info(f"  - No soportados: {unsupported}")
        logger.info(f"  - Staging path: {staging_folder}")
        
        # Verificar que se clasificaron correctamente
        assert total_files >= 8, "Deberían procesarse al menos 8 archivos"
        assert classified >= 5, "Deberían clasificarse al menos 5 documentos"
        assert unsupported >= 1, "Debería haber al menos 1 archivo no soportado"
        
        # Verificar que hay archivos clasificados
        doc_types = set(f.get('document_type') for f in mapping.get('files', []))
        logger.info(f"  - Tipos detectados: {doc_types}")
        assert len(doc_types) > 0, "No se detectaron tipos de documentos"
        
        # Verificar mapping.json
        mapping_file = staging_folder / "mapping.json"
        assert mapping_file.exists(), "mapping.json no fue creado"
        
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        
        logger.info(f"\n📋 Archivos en staging: {len(mapping.get('files', []))}")
        for file_info in mapping.get("files", [])[:3]:  # Mostrar primeros 3
            logger.info(f"  - {file_info.get('staged_name', 'N/A')} ({file_info.get('document_type', 'N/A')})")
        
        logger.info("\n✅ Fase A completada exitosamente")
        return staging_folder, mapping


async def test_phase_b_extraction():
    """Test de Fase B: Extracción y Renombrado (simulado)"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST FASE B: Extracción y Renombrado")
    logger.info("=" * 60)
    
    # Crear directorio temporal
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_docs"
        await create_test_documents(test_dir)
        
        # Inicializar organizador
        organizer = DocumentOrganizer()
        
        # Ejecutar Fase A primero
        logger.info("🔍 Ejecutando Fase A como preparación...")
        staging_path, mapping = await organizer.organize_documents_phase_a(
            input_folder=test_dir,
            use_llm_fallback=False
        )
        assert staging_path.exists(), "Fase A falló - staging path no existe"
        
        # Ejecutar Fase B (con extracción simulada)
        logger.info("\n🧠 Ejecutando Fase B...")
        
        # Para el test, simularemos la extracción modificando el mapping
        # En producción, esto llamaría al AIFieldExtractor real
        mapping_file = staging_path / "mapping.json"
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        
        # Simular extracción de campos clave
        mapping["extracted_fields"] = {
            "nombre_asegurado": "TEST_COMPANY_SA",
            "numero_siniestro": "2025-TEST-001"
        }
        
        with open(mapping_file, "w") as f:
            json.dump(mapping, f, indent=2)
        
        # Crear carpeta final simulada
        final_name = f"{mapping['extracted_fields']['nombre_asegurado']} - {mapping['extracted_fields']['numero_siniestro']}"
        final_path = staging_path.parent / final_name
        
        # Simular renombrado
        if staging_path.exists():
            shutil.copytree(staging_path, final_path, dirs_exist_ok=True)
        
        logger.info(f"\n📁 Carpeta final creada: {final_path.name}")
        assert final_path.exists(), "Carpeta final no fue creada"
        
        # Verificar estructura final
        files_in_final = list(final_path.glob("*"))
        logger.info(f"📄 Archivos en carpeta final: {len(files_in_final)}")
        
        logger.info("\n✅ Fase B completada exitosamente (simulada)")
        return final_path


async def test_full_pipeline():
    """Test del pipeline completo de organización"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST PIPELINE COMPLETO")
    logger.info("=" * 60)
    
    # Crear directorio temporal
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_docs"
        await create_test_documents(test_dir)
        
        # Inicializar organizador
        organizer = DocumentOrganizer()
        
        # Ejecutar pipeline completo
        logger.info("🚀 Ejecutando pipeline completo...")
        
        # Fase A
        staging_path, mapping = await organizer.organize_documents_phase_a(
            input_folder=test_dir,
            use_llm_fallback=False
        )
        assert staging_path.exists(), "Fase A falló - staging path no existe"
        
        # Verificar métricas
        total_files = len(mapping.get('files', []))
        classified = sum(1 for f in mapping.get('files', []) if f.get('document_type') != 'otros')
        doc_types = set(f.get('document_type') for f in mapping.get('files', []))
        
        logger.info(f"\n📊 Resumen del Pipeline:")
        logger.info(f"  - Archivos procesados: {total_files}")
        logger.info(f"  - Documentos clasificados: {classified}")
        logger.info(f"  - Tipos únicos: {len(doc_types)}")
        logger.info(f"  - Staging path: {staging_path.name}")
        
        # Validaciones finales
        assert classified > 0, "No se clasificó ningún documento"
        assert staging_path.exists(), "Staging path no existe"
        
        logger.info("\n✅ Pipeline completo ejecutado exitosamente")
        return True


async def main():
    """Ejecutar todos los tests"""
    logger.info("\n🧪 INICIANDO TESTS END-TO-END DEL ORGANIZADOR")
    logger.info("=" * 80)
    
    all_passed = True
    
    # Test 1: Fase A - Clasificación y Staging
    try:
        staging_path, result = await test_phase_a_classification()
        logger.info("✅ Test Fase A pasó")
    except Exception as e:
        logger.error(f"❌ Test Fase A falló: {e}")
        all_passed = False
    
    # Test 2: Fase B - Extracción y Renombrado (simulado)
    try:
        final_path = await test_phase_b_extraction()
        logger.info("✅ Test Fase B pasó")
    except Exception as e:
        logger.error(f"❌ Test Fase B falló: {e}")
        all_passed = False
    
    # Test 3: Pipeline completo
    try:
        result = await test_full_pipeline()
        logger.info("✅ Test Pipeline Completo pasó")
    except Exception as e:
        logger.error(f"❌ Test Pipeline Completo falló: {e}")
        all_passed = False
    
    # Resultado final
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ TODOS LOS TESTS E2E PASARON EXITOSAMENTE")
        logger.info("\n💡 Próximos pasos:")
        logger.info("  1. Ejecutar: python scripts/run_report.py --folder ./docs --organize-only")
        logger.info("  2. Verificar la carpeta de staging creada")
        logger.info("  3. Ejecutar: python scripts/run_report.py --folder ./docs --organize-first")
        logger.info("  4. Verificar el pipeline completo con organización")
    else:
        logger.error("❌ ALGUNOS TESTS E2E FALLARON")
    logger.info("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)