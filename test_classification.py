#!/usr/bin/env python3
"""
Test básico del sistema de clasificación de documentos
"""

import asyncio
from pathlib import Path
import sys
import logging

# Añadir el src al path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from fraud_scorer.processors.document_classifier import DocumentClassifier

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_heuristic_classification():
    """Test de clasificación heurística basada en nombres de archivo"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Clasificación Heurística")
    logger.info("=" * 60)
    
    classifier = DocumentClassifier()
    
    # Casos de prueba usando los tipos reales del sistema
    test_cases = [
        # (filename, expected_type basado en keywords)
        ("carta_reclamacion_aseguradora.pdf", "carta_de_reclamacion_formal_a_la_aseguradora"),
        ("reclamacion_transportista.pdf", "carta_de_reclamacion_formal_al_transportista"),
        ("factura_1234.pdf", "guias_y_facturas"),
        ("guia_embarque.pdf", "guias_y_facturas"),
        ("salida_almacen.xlsx", "salida_de_almacen"),
        ("tarjeta_circulacion.pdf", "tarjeta_de_circulacion_vehiculo"),
        ("licencia_operador.pdf", "licencia_del_operador"),
        ("aviso_siniestro.pdf", "aviso_de_siniestro_transportista"),
        ("GPS_tracking.csv", "reporte_gps"),
        ("carpeta_investigacion.pdf", "carpeta_de_investigacion"),
        ("poliza_seguro.pdf", "poliza_de_la_aseguradora"),
        ("informe_ajustador.pdf", "informe_preliminar_del_ajustador"),
        ("checklist_antifraude.xlsx", "checklist_antifraude"),
        ("unknown_document.pdf", "otro"),  # Debería clasificar como "otro"
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for filename, expected_type in test_cases:
        # Usar el método classify con sample_text vacío y filename
        doc_type, confidence, reasons = await classifier.classify(
            sample_text="",
            filename=filename,
            use_llm_fallback=False  # Solo usar heurística
        )
        
        is_correct = doc_type == expected_type
        
        if is_correct:
            success_count += 1
            logger.info(f"✅ {filename:40} → {doc_type:40} (confianza: {confidence:.2f})")
        else:
            logger.error(f"❌ {filename:40} → {doc_type:40} (esperado: {expected_type})")
    
    logger.info(f"\n📊 Resultado: {success_count}/{total_count} clasificaciones correctas")
    return success_count == total_count


async def test_document_priority():
    """Test del sistema de prioridades de documentos"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Prioridades de Documentos")
    logger.info("=" * 60)
    
    import fraud_scorer.settings as settings
    
    # Verificar que los documentos tienen prioridades asignadas
    high_priority = [
        "carpeta_de_investigacion",
        "poliza_de_la_aseguradora",
        "guias_y_facturas"
    ]
    medium_priority = [
        "carta_de_reclamacion_formal_a_la_aseguradora",
        "carta_de_reclamacion_formal_al_transportista",
        "reporte_gps"
    ]
    low_priority = [
        "checklist_antifraude",
        "otro"
    ]
    
    # Usar la configuración directamente del módulo
    priorities = getattr(settings, 'DOCUMENT_PRIORITIES', {})
    
    for doc_type in high_priority:
        priority = priorities.get(doc_type, 999)
        logger.info(f"Prioridad de '{doc_type}': {priority}")
        assert priority <= 5, f"'{doc_type}' debería tener alta prioridad"
    
    for doc_type in medium_priority:
        priority = priorities.get(doc_type, 999)
        logger.info(f"Prioridad de '{doc_type}': {priority}")
        assert 6 <= priority <= 10, f"'{doc_type}' debería tener prioridad media"
    
    for doc_type in low_priority:
        priority = priorities.get(doc_type, 999)
        logger.info(f"Prioridad de '{doc_type}': {priority}")
        assert priority >= 11, f"'{doc_type}' debería tener baja prioridad"
    
    logger.info("✅ Todas las prioridades están correctamente asignadas")
    return True


async def test_sample_text_extraction():
    """Test de extracción de texto de muestra (simulado)"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Clasificación con Texto de Muestra")
    logger.info("=" * 60)
    
    classifier = DocumentClassifier()
    
    # Simular clasificación con texto de muestra
    sample_texts = [
        ("CARPETA DE INVESTIGACIÓN FGR/2025/001", "carpeta_de_investigacion"),
        ("FACTURA No. 12345 Total: $50,000 Guía de embarque", "guias_y_facturas"),
        ("PÓLIZA DE SEGURO Cobertura de transporte marítimo", "poliza_de_la_aseguradora"),
        ("REPORTE GPS Coordenadas: 19.4326, -99.1332", "reporte_gps"),
        ("CHECKLIST ANTIFRAUDE - Evaluación de riesgo", "checklist_antifraude"),
    ]
    
    success_count = 0
    for text, expected_type in sample_texts:
        # Clasificar usando el texto de muestra
        logger.info(f"Analizando texto: '{text[:50]}...'")
        
        # El clasificador debería poder usar el texto para mejorar la clasificación
        doc_type, confidence, reasons = await classifier.classify(
            sample_text=text,
            filename="documento.pdf",
            use_llm_fallback=False
        )
        
        if doc_type == expected_type:
            success_count += 1
            logger.info(f"✅ Clasificado correctamente como: {doc_type}")
        else:
            logger.error(f"❌ Clasificado como {doc_type}, esperado: {expected_type}")
    
    logger.info(f"\n📊 Resultado: {success_count}/{len(sample_texts)} clasificaciones correctas")
    return success_count == len(sample_texts)


async def main():
    """Ejecutar todos los tests"""
    logger.info("\n🧪 INICIANDO TESTS DE CLASIFICACIÓN")
    logger.info("=" * 80)
    
    all_passed = True
    
    # Test 1: Clasificación heurística
    try:
        result = await test_heuristic_classification()
        if not result:
            all_passed = False
    except Exception as e:
        logger.error(f"❌ Error en test heurístico: {e}")
        all_passed = False
    
    # Test 2: Prioridades de documentos
    try:
        result = await test_document_priority()
        if not result:
            all_passed = False
    except Exception as e:
        logger.error(f"❌ Error en test de prioridades: {e}")
        all_passed = False
    
    # Test 3: Extracción de texto
    try:
        result = await test_sample_text_extraction()
        if not result:
            all_passed = False
    except Exception as e:
        logger.error(f"❌ Error en test de extracción: {e}")
        all_passed = False
    
    # Resultado final
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
    else:
        logger.error("❌ ALGUNOS TESTS FALLARON")
    logger.info("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)