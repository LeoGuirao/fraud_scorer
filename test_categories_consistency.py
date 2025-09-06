#!/usr/bin/env python3
"""
Test dinámico de consistencia para nuevas categorías.

Verifica que cada tipo de DocumentType (excepto 'otro') esté correctamente
integrado en las configuraciones clave del sistema:
- Definición en el clasificador (type_definitions)
- Mapeo tipo→campos en ExtractionConfig.DOCUMENT_FIELD_MAPPING (aunque sea [])
- Ruta de extracción en ExtractionConfig.DOCUMENT_EXTRACTION_ROUTES (ocr_text/direct_ai)
- Alias canónico en settings.CANONICAL_TO_ALIAS (para renombrado)
- Prioridad accesible vía DocumentClassifier.get_document_priority()
"""

import sys
from pathlib import Path

# Añadir el src al path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from fraud_scorer.processors.document_classifier import DocumentClassifier, DocumentType
from fraud_scorer.settings import ExtractionConfig, CANONICAL_TO_ALIAS, ExtractionRoute


def test_categories_consistency() -> bool:
    print("🧪 TEST DINÁMICO: Consistencia de categorías")
    print("=" * 70)

    classifier = DocumentClassifier()
    config = ExtractionConfig()

    problems = {
        "missing_definition": [],
        "missing_field_mapping": [],
        "missing_route": [],
        "invalid_route": [],
        "missing_alias": [],
        "invalid_priority": [],
    }

    allowed_routes = {ExtractionRoute.OCR_TEXT, ExtractionRoute.DIRECT_AI}

    for dt in DocumentType:
        if dt == DocumentType.OTRO:
            continue  # 'otro' es la clase comodín

        name = dt.value

        # Definición en clasificador
        if name not in classifier.type_definitions:
            problems["missing_definition"].append(name)

        # Mapeo de campos (puede ser lista vacía, pero debe existir la clave)
        if not hasattr(config, 'DOCUMENT_FIELD_MAPPING') or name not in config.DOCUMENT_FIELD_MAPPING:
            problems["missing_field_mapping"].append(name)

        # Ruta de extracción definida y válida
        route = None
        if hasattr(config, 'DOCUMENT_EXTRACTION_ROUTES'):
            route = config.DOCUMENT_EXTRACTION_ROUTES.get(name)
        if route is None:
            problems["missing_route"].append(name)
        else:
            # Aceptar tanto enums como strings (por robustez)
            route_val = route.value if hasattr(route, 'value') else route
            if route_val not in {r.value for r in allowed_routes}:
                problems["invalid_route"].append((name, route_val))

        # Alias para renombrado (usado por el organizador)
        if name not in CANONICAL_TO_ALIAS:
            problems["missing_alias"].append(name)

        # Prioridad disponible vía clasificador (entero válido)
        try:
            prio = classifier.get_document_priority(name)
            if not isinstance(prio, int) or prio < 1 or prio > 99:
                problems["invalid_priority"].append((name, prio))
        except Exception as e:
            problems["invalid_priority"].append((name, f"ERROR: {e}"))

    # Reporte
    total_types = len([d for d in DocumentType if d != DocumentType.OTRO])
    print(f"Tipos evaluados: {total_types}")

    ok = True
    for key, items in problems.items():
        if items:
            ok = False
            print(f"❌ {key}: {items}")

    if ok:
        print("✅ Consistencia verificada para todas las categorías")
    else:
        print("❌ Inconsistencias detectadas. Revisa los listados anteriores.")

    return ok


if __name__ == "__main__":
    success = test_categories_consistency()
    sys.exit(0 if success else 1)

