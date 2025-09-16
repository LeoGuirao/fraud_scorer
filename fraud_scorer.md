# Fraud Scorer v2 — Guía Técnica y Arquitectura

Este documento resume con precisión cómo funciona el sistema, su arquitectura, los procesos clave y los puntos críticos para operación y evolución.

## Resumen
- Objetivo: analizar expedientes de siniestros a partir de documentos heterogéneos (PDF/imagen/Office), extraer datos clave con IA, consolidarlos y generar un informe HTML/PDF.
- Capas principales: Parsing + OCR (Azure), Clasificación, Extracción Guiada (IA), Consolidación (IA + reglas), Reportería, API y Sistema de Replay.
- Almacenamiento: SQLite para metadatos/resultados, caché de OCR en filesystem, staging organizado por lotes.

## Componentes Clave (ubicación)
- API FastAPI: `src/fraud_scorer/api` (routers en `endpoints/`, plantillas en `api/templates/`). Entrada: `api/main.py`.
- Procesadores: `src/fraud_scorer/processors/`
  - Clasificación: `document_classifier.py`, `classification/engine.py` (LLM-first con heurísticas de respaldo).
  - Extracción: `ai/ai_field_extractor.py` (Sistema de Extracción Guiada + validaciones).
  - Consolidación: `ai/ai_consolidator.py` (resolución de conflictos + validación conservadora).
  - OCR: `ocr/azure_ocr.py` (Azure Document Intelligence, salida normalizada `OCRResult`).
- Parsers: `src/fraud_scorer/parsers/` (unificación de formatos; `document_parser.py`).
- Modelos: `src/fraud_scorer/models/extraction.py` (Pydantic: `DocumentExtraction`, `ConsolidatedFields`, `ConsolidatedExtraction`).
- Plantillas de reporte: `src/fraud_scorer/templates/` (`ai_report_generator.py`, `report_template.html`).
- Almacenamiento y caché: `src/fraud_scorer/storage/` (`db.py`, `ocr_cache.py`, `cases.py`, `temporal_cache.py`).
- Replay y UI: `src/fraud_scorer/services/replay_service.py`, `src/fraud_scorer/ui/` y vistas en `api/templates/`.
- Configuración: `src/fraud_scorer/settings.py` (rutas, mapeos, modelos óptimos, alias, prioridades).

## Flujo de Proceso
1) Ingesta y Parsing
- Descubrimiento y parsing por extensión (`parsers/types.py`).
- Imágenes/PDF → Azure OCR vía `AzureOCRProcessor` → salida unificada (texto, tablas, KV, metadatos).
- DOCX/XLSX/CSV → parsers nativos a la misma estructura.

2) Clasificación (fase de organización A)
- `DocumentClassifier` + `classification/engine.py`.
- Estrategia: LLM-first (modelo `gpt-4o-mini`) con fallback heurístico; “contenido primero” (no sesgo por nombre de archivo).
- Resultado: tipo canónico (p. ej., `poliza_de_la_aseguradora`, `carpeta_de_investigacion`, `guias_y_facturas`, etc.).

3) Extracción guiada por documento
- `AIFieldExtractor` utiliza el Sistema de Extracción Guiada:
  - Mapeo tipo→campos permitidos: `ExtractionConfig.DOCUMENT_FIELD_MAPPING`.
  - Rutas: `direct_ai` (visión) u `ocr_text` (texto), elegidas por `get_model_for_task` y `ROUTE_CONFIG`.
  - Prompts construidos por `prompts/extraction_prompts.py` con guías estrictas y truncamiento de contexto.
  - Validador `FieldValidator` para formatos (fechas, montos, pólizas) y reglas específicas (p. ej. HDI EN MI CASA).
- Salida por documento: `DocumentExtraction` con `extracted_fields` normalizados y metadatos de extracción.

4) Consolidación multi-documento
- `AIConsolidator` agrupa opciones por campo y decide:
  - Reglas determinísticas (prioridad por tipo de documento) + razonamiento IA cuando hay conflicto.
  - Modo guiado: aplica máscara estricta de campos válidos y evita “inventar” valores.
  - Validación final conservadora; si no hay datos, no llama IA para evitar alucinaciones.
- Salida final: `ConsolidatedExtraction` con `ConsolidatedFields`, fuentes y puntuaciones de confianza.

5) Reporte
- `AIReportGenerator` renderiza `report_template.html` con `ConsolidatedExtraction` y genera HTML (y PDF con WeasyPrint si está instalado).
- Nomenclatura: `INF-<ASEGURADO>-<SINIESTRO>.{html,pdf}` con sanitización segura.

6) Replay de casos (desde caché)
- `ReplayService` consume el índice del caché (`OCRCacheManager`) y la base SQLite.
- Permite reprocesar un `case_id` con IA, regenerar reportes y limpiar artefactos (purge/deep-purge, manejo de huérfanos).
- Endpoints HTML/JSON en `api/endpoints/replay.py` y plantillas asociadas.

## Datos y Configuración
- Campos objetivo (header): `numero_siniestro`, `nombre_asegurado`, `numero_poliza`, `vigencia_inicio`, `vigencia_fin`, `domicilio_poliza`, `bien_reclamado`, `monto_reclamacion`, `tipo_siniestro`, `fecha_ocurrencia`, `fecha_reclamacion`, `lugar_hechos`, `ajuste`, `conclusiones`.
- Mapeos críticos en `settings.py`:
  - `DOCUMENT_FIELD_MAPPING`: qué campos extrae cada tipo de documento.
  - `EXTRACTION_TARGET_TYPES`: tipos de documento que sí aportan cabecera.
  - `DOCUMENT_PRIORITIES`: prioridad por tipo para consolidación y orden.
  - `FIELD_VALIDATION_RULES` y `FIELD_SYNONYMS`: formatos y sinónimos por campo.
  - `CLASSIFICATION_CONFIG` y `CLASSIFICATION_ENGINE`: estrategia y modelo de clasificación.
  - `ROUTE_CONFIG` y `DOCUMENT_EXTRACTION_ROUTES`: rutas por extensión/tipo.
  - `get_model_for_task(task, route)`: selección de modelos GPT-5/GPT-4o optimizados por costo/latencia.

## Almacenamiento y Caché
- Base de datos SQLite (`storage/db.py`, ruta `data/cases.db` por defecto):
  - `cases`, `documents`, `ocr_results`, `extracted_data`, `runs`, `ai_analyses`, `cache_stats`.
  - Utilidades: registro de documentos (`upsert_document`), guardado OCR (`save_ocr_result`), extracciones (`save_extracted_data`), métricas de caché.
- Caché de OCR en filesystem (`storage/ocr_cache.py`):
  - Shards por hash en `data/ocr_cache/` y vista “humana” por caso (Nombre - Reclamo).
  - Índice por caso: `data/ocr_cache/case_index/<case_id>.json`.
  - Reuso por hash entre casos y métricas de hits/misses.
- Staging de organización: `data/uploads/renombre_de_documentos/<timestamp>/mapping.json` (Fase A).

## API y Vistas
- App FastAPI (`api/main.py`) con routers:
  - General: salud y raíz (`endpoints/general.py`).
  - Documentos: upload/OCR básico (`endpoints/documents.py`).
  - Reportes: generación asincrónica y previsualización (`endpoints/reports.py`).
  - Replay: dashboard, listado, procesamiento y limpieza (`endpoints/replay.py`, vistas en `api/templates/`).

## Operación y Entorno
- Variables de entorno (en `.env`):
  - `OPENAI_API_KEY`: requerido para IA.
  - `AZURE_ENDPOINT`, `AZURE_OCR_KEY` (o `AZURE_DOCUMENT_INTELLIGENCE_*`): requerido para OCR.
  - `FRAUD_DB_PATH` (opcional): ruta a la DB SQLite.
- Ejecución local API: `python -m uvicorn fraud_scorer.api.main:app --reload` (o `start.sh`).
- Docker: `docker-compose.yml` define servicios para despliegue local.

## Decisiones y Buenas Prácticas
- Extracción guiada: reduce alucinaciones limitando campos por tipo de documento y validando formato/contexto.
- Clasificación “contenido primero”: prompts sin sesgo por nombre de archivo; heurística de respaldo robusta.
- Vision multimodal: reservada para preview/escenarios puntuales; pipeline principal prioriza OCR+texto por costo/latencia.
- Trazabilidad: cada campo consolidado incluye fuente y confianza; se guardan snapshots intermedios (DB y caché).
- Seguridad: nunca persistir credenciales en código; usar `.env`.

## Extensión del Sistema (cómo añadir un tipo nuevo)
1) Agregar tipo y campos en `settings.DOCUMENT_FIELD_MAPPING` y prioridades en `DOCUMENT_PRIORITIES`.
2) Ampliar definiciones del clasificador (`processors/document_classifier.py`).
3) Si aplica, alias en `DOCUMENT_TYPE_ALIASES` y rutas en `DOCUMENT_EXTRACTION_ROUTES`.
4) (Opcional) Ajustar prompts/validadores si hay reglas específicas.
5) Probar con `test_*` existentes (hay tests para clasificación, organizer y e2e en raíz).

## Limitaciones y pendientes
- `pipelines/data_flow.render_report` mantiene un stub de compatibilidad; la generación real de reportes se realiza con `AIReportGenerator`.
- WeasyPrint para PDF es opcional; si no está instalado, se omite la exportación a PDF.

---
Esta guía cubre lo esencial para operar, depurar y extender Fraud Scorer v2 con confianza.

