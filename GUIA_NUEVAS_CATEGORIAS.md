# Guía Rápida: Añadir Nuevas Categorías de Documento

Esta guía explica cómo integrar correctamente una nueva categoría (tipo de documento) al sistema para que clasifique, rote, renombre y pase los tests.

## Pasos Obligatorios

1) Declarar la categoría
- Archivo: `src/fraud_scorer/processors/document_classifier.py`
- Acción: Añadir el nuevo valor al enum `DocumentType` (usar snake_case exacto y coherente).

2) Definir heurísticas de clasificación
- Archivo: `src/fraud_scorer/processors/document_classifier.py`
- Acción: En `_initialize_type_definitions()`, añadir un `DocumentTypeDefinition` con:
  - `keywords`: palabras clave relevantes (contenido y/o nombre)
  - `must_have`: indicadores obligatorios (si aplica)
  - `may_have`: indicadores opcionales
  - `exclude`: exclusiones para evitar colisiones
  - `description`: descripción breve del documento
- Opcional: Si quieres reglas rápidas por nombre, añade condiciones en `_heuristic_classify(...)` (útil para identificar por filename sin OCR).

3) Configuración de extracción y rutas
- Archivo: `src/fraud_scorer/settings.py`
- Acción:
  - En `ExtractionConfig.DOCUMENT_FIELD_MAPPING`, añade la clave del nuevo tipo con la lista de campos permitidos (o `[]` si no extrae cabeceras).
  - En `ExtractionConfig.DOCUMENT_EXTRACTION_ROUTES`, asigna la ruta: `ExtractionRoute.OCR_TEXT` o `ExtractionRoute.DIRECT_AI`.
  - Si el documento debe participar en extracción de cabecera, evalúa incluirlo en `ExtractionConfig.EXTRACTION_TARGET_TYPES`.

4) Alias de renombrado (nombres de archivo cortos)
- Archivo: `src/fraud_scorer/settings.py`
- Acción: En el diccionario superior `DOCUMENT_TYPE_ALIASES` (claves cortas en MAYÚSCULAS → nombre canónico), añade un alias corto. `CANONICAL_TO_ALIAS` se actualiza automáticamente.

5) Prioridad de ordenamiento
- Archivos:
  - `src/fraud_scorer/processors/document_classifier.py` → método `get_document_priority(...)`
  - (Opcional) `src/fraud_scorer/settings.py` → `DOCUMENT_PRIORITIES` usado en resúmenes/ordenamientos
- Acción: Asigna un entero 1–99 (1 = mayor prioridad). Mantén coherencia con documentos similares.

6) Subnumeración (opcional)
- Archivo: `src/fraud_scorer/processors/document_organizer.py`
- Acción: Si el nuevo tipo requiere subnumeración (ej. series como guías o notas), replica la lógica usada para `guias_y_facturas` o `notas_de_reparacion` y ajusta el contador.

## Tests a Ejecutar (mínimos)

- `python test_categories_consistency.py`
  - Verifica integración global: definición, mapeo de campos, ruta válida, alias y prioridad.
- `python test_document_routes.py`
  - Revisa dinámicamente que todas las categorías tengan ruta (`ocr_text`/`direct_ai`).

## Tests de Regresión (recomendados)

- `python test_classification.py`
  - Añade casos específicos de filename y/o texto para la nueva categoría.
- `python test_organizer_e2e.py`
  - Confirma que Fase A (clasificación y staging) sigue estable.
- `python test_optimal_models.py`
  - Asegura que la selección de modelos no sufrió regresiones.

## Checklist de Integración

- [ ] Enum `DocumentType` actualizado con el nombre canónico correcto
- [ ] `DocumentTypeDefinition` añadido con keywords/exclusiones adecuados
- [ ] `DOCUMENT_FIELD_MAPPING` contiene la clave (aunque sea `[]`)
- [ ] `DOCUMENT_EXTRACTION_ROUTES` define `ocr_text` o `direct_ai`
- [ ] `DOCUMENT_TYPE_ALIASES` tiene alias corto (y por ende `CANONICAL_TO_ALIAS`)
- [ ] `get_document_priority(...)` retorna prioridad válida (1–99)
- [ ] (Opcional) `DOCUMENT_PRIORITIES` en `settings.py` actualizado para ordenamientos
- [ ] (Si aplica) Subnumeración en `document_organizer.py`
- [ ] Tests mínimos pasan (consistency y routes)

## Consejos de Diseño

- Usa `exclude` para evitar colisiones con categorías similares (p. ej., diferenciar “licencia del operador” vs “identificación oficial”).
- Prefiere `keywords` de contenido cuando sea posible; los atajos por nombre son útiles pero más frágiles.
- Si el documento no aporta campos de cabecera, mantenlo con `[]` en `DOCUMENT_FIELD_MAPPING` para forzar extracción nula.

## Ejemplo (esquemático)

1) Enum y definición:
- `DocumentType.NUEVO_TIPO = "nuevo_tipo"`
- En `_initialize_type_definitions()`: añadir un `DocumentTypeDefinition` con keywords, exclude, etc.

2) Settings:
- `ExtractionConfig.DOCUMENT_FIELD_MAPPING["nuevo_tipo"] = []`
- `ExtractionConfig.DOCUMENT_EXTRACTION_ROUTES["nuevo_tipo"] = ExtractionRoute.OCR_TEXT`
- `DOCUMENT_TYPE_ALIASES["NUEVO"] = "nuevo_tipo"`
- `get_document_priority("nuevo_tipo") → 21` (por ejemplo)

3) Tests:
- Ejecutar: `python test_categories_consistency.py` y `python test_document_routes.py`
- Añadir casos en `test_classification.py` si quieres validar heurísticas nuevas.

