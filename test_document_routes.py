#!/usr/bin/env python3
"""
Test dinámico para verificar que las rutas de extracción se asignan
correctamente para todos los tipos de documento definidos.

Detecta automáticamente categorías nuevas que no tengan ruta configurada
o que usen una ruta inválida.
"""

import sys
from pathlib import Path

# Añadir el src al path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from fraud_scorer.settings import ExtractionConfig, ExtractionRoute
from fraud_scorer.processors.document_classifier import DocumentType


def test_document_extraction_routes():
    """Verifica que cada categoría tenga una ruta válida (dinámico)."""
    print("🧪 TEST DINÁMICO: Rutas de extracción por tipo de documento")
    print("=" * 70)

    config = ExtractionConfig()

    # Conjunto de rutas válidas (acepta enum o string)
    valid_routes = {ExtractionRoute.OCR_TEXT.value, ExtractionRoute.DIRECT_AI.value}

    missing = []
    invalid = []
    ok_count = 0
    total = 0

    for dt in DocumentType:
        if dt == DocumentType.OTRO:
            continue  # Excluir comodín
        total += 1
        name = dt.value

        route = None
        if hasattr(config, 'DOCUMENT_EXTRACTION_ROUTES'):
            route = config.DOCUMENT_EXTRACTION_ROUTES.get(name)

        if route is None:
            missing.append(name)
            print(f"❌ {name:45} → NO CONFIGURADO")
            continue

        route_val = route.value if hasattr(route, 'value') else route
        if route_val in valid_routes:
            ok_count += 1
            print(f"✅ {name:45} → {route_val}")
        else:
            invalid.append((name, route_val))
            print(f"❌ {name:45} → {route_val} (ruta inválida)")

    print("\n" + "-" * 70)
    print(f"Resumen: {ok_count}/{total} rutas válidas")
    if missing:
        print(f"Faltantes ({len(missing)}): {missing}")
    if invalid:
        print(f"Inválidas ({len(invalid)}): {invalid}")

    all_ok = (ok_count == total)
    if all_ok:
        print("\n✅ TODAS LAS RUTAS ESTÁN CONFIGURADAS CORRECTAMENTE")
    else:
        print("\n❌ HAY RUTAS FALTANTES O INVÁLIDAS")

    return all_ok


if __name__ == "__main__":
    success = test_document_extraction_routes()
    sys.exit(0 if success else 1)
