#!/usr/bin/env python3
"""
Test para verificar que las rutas de extracción se asignan correctamente por tipo de documento
"""

import sys
from pathlib import Path

# Añadir el src al path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from fraud_scorer.settings import ExtractionConfig, ExtractionRoute


def test_document_extraction_routes():
    """Test que verifica la configuración de rutas por tipo de documento"""
    print("🧪 TESTEO DE RUTAS DE EXTRACCIÓN POR TIPO DE DOCUMENTO")
    print("=" * 60)
    
    # Inicializar solo configuración (sin extractor para evitar OpenAI)
    config = ExtractionConfig()
    
    # Test cases basados en tu especificación
    test_cases = [
        # OCR + AI (ExtractionRoute.OCR_TEXT)
        ("carta_de_reclamacion_formal_a_la_aseguradora", "ocr_text"),
        ("carta_de_reclamacion_formal_al_transportista", "ocr_text"),
        ("guias_y_facturas", "ocr_text"),
        ("tarjeta_de_circulacion_vehiculo", "ocr_text"),
        ("licencia_del_operador", "ocr_text"),
        ("aviso_de_siniestro_transportista", "ocr_text"),
        ("carpeta_de_investigacion", "ocr_text"),
        ("acreditacion_de_propiedad_y_representacion", "ocr_text"),
        ("salida_de_almacen", "ocr_text"),
        ("reporte_gps", "ocr_text"),
        ("guias_y_facturas_consolidadas", "ocr_text"),
        ("expediente_de_cobranza", "ocr_text"),
        ("checklist_antifraude", "ocr_text"),
        
        # AI Directo (ExtractionRoute.DIRECT_AI)
        ("poliza_de_la_aseguradora", "direct_ai"),
        ("informe_preliminar_del_ajustador", "direct_ai"),
        ("informe_final_del_ajustador", "direct_ai"),
    ]
    
    print("Verificando configuración en DOCUMENT_EXTRACTION_ROUTES:")
    success_count = 0
    
    for document_type, expected_route in test_cases:
        # Test 1: Verificar configuración en settings
        if hasattr(config, 'DOCUMENT_EXTRACTION_ROUTES'):
            actual_route_config = config.DOCUMENT_EXTRACTION_ROUTES.get(document_type)
            if actual_route_config:
                actual_route = actual_route_config.value if hasattr(actual_route_config, 'value') else actual_route_config
                if actual_route == expected_route:
                    success_count += 1
                    print(f"✅ {document_type:45} → {actual_route}")
                else:
                    print(f"❌ {document_type:45} → {actual_route} (esperaba {expected_route})")
            else:
                print(f"❌ {document_type:45} → NO CONFIGURADO (esperaba {expected_route})")
        else:
            print("❌ DOCUMENT_EXTRACTION_ROUTES no existe en configuración")
            break
    
    print(f"\n📊 Resultado: {success_count}/{len(test_cases)} rutas configuradas correctamente")
    
    # Resultado final
    print("\n" + "=" * 60)
    
    if success_count == len(test_cases):
        print("✅ TODOS LOS TESTS PASARON - Configuración de rutas correcta")
        print("\n💡 Próximos pasos:")
        print("  1. Las rutas se aplicarán automáticamente durante la extracción de campos")
        print("  2. Documentos OCR + AI usarán análisis de texto detallado")
        print("  3. Documentos AI Directo usarán visión GPT para análisis de imagen")
        return True
    else:
        print(f"❌ {len(test_cases) - success_count} tests fallaron de {len(test_cases)} totales")
        return False


if __name__ == "__main__":
    success = test_document_extraction_routes()
    sys.exit(0 if success else 1)