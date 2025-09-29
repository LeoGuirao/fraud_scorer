# 🎯 VISIÓN DEL SISTEMA DE ANÁLISIS INTELIGENTE DE SINIESTROS
## Documento de Arquitectura y Diseño Estratégico v2.2

---

## 📋 RESUMEN EJECUTIVO

El sistema de **Fraud Scorer v2** opera end-to-end sobre la ingesta de expedientes de siniestros, normaliza todos los documentos disponibles y entrega reportes profesionales con indicadores de fraude por documento. Actualmente:

- `scripts/run_report.py` orquesta la cadena completa **DocumentOrganizer → OCR → Clasificación → Extracción Guiada → Consolidación → Análisis de Fraude → Reportes HTML/PDF**, con seguimiento en tiempo real mediante eventos JSONL.
- La capa de IA combina prompts especializados y modelos OpenAI (`gpt-4o`/`gpt-4o-mini`) con validadores Pydantic para garantizar coherencia y trazabilidad.
- `fraud_analyses` en `data/cases.db` mantiene histórico de resultados, enlazado con el caché de OCR (`OCRCacheManager`) y los artefactos generados para cada caso.
- Se cuenta con herramientas de operación (`scripts/system_integrity_check.py`, `scripts/clean_orphaned_files.py`, `src/fraud_scorer/storage/post_process_verifier.py`) que aseguran la integridad del ecosistema.

La visión estratégica sigue apuntando a tres frentes: automatización inteligente por documento, correlación inter-documental y decisiones globales asistidas. El presente documento resume el estado actual, la arquitectura viva y los siguientes pasos priorizados.

---

## 🛰️ ESTADO DEL SISTEMA (Q1 2026)

| Componente | Ubicación clave | Estado | Notas relevantes |
|------------|-----------------|--------|------------------|
| **Organización & parsing** | `src/fraud_scorer/processors/document_organizer.py`, `src/fraud_scorer/parsers/document_parser.py` | ✅ Productivo | Fase A/B para staging, renombrado controlado, heurísticas de rutas, soporta PDF/TXT/DOCX, integra con caché. |
| **OCR** | `src/fraud_scorer/processors/ocr/azure_ocr.py` | ✅ Estable | Azure Document Intelligence; fallback para textos planos; resultados normalizados vía `pipelines/data_flow.ocr_result_to_dict`. |
| **Clasificación** | `src/fraud_scorer/processors/document_classifier.py` | ✅ Estable | 40+ tipos canónicos, heurística + LLM fallback (`ClassifierEngine`), alias centralizados en `settings.CANONICAL_TO_ALIAS`. |
| **Extracción guiada** | `src/fraud_scorer/processors/ai/ai_field_extractor.py` | ✅ Estable | Prompts dinámicos (`ExtractionPromptBuilder`), rutas configurables (`ExtractionConfig`), validación automática (`FieldValidator`). |
| **Consolidación** | `src/fraud_scorer/processors/ai/ai_consolidator.py` | ✅ Estable | Resuelve conflictos, entrega `ConsolidatedExtraction` con `confidence_scores` y `consolidation_sources`. |
| **Motor fraude por documento** | `src/fraud_scorer/analyzers/fraud_analyzer.py`, `src/fraud_scorer/prompts/fraud_prompts.py`, `src/fraud_scorer/analyzers/fraud_guide_manager.py`, `src/fraud_scorer/guides/` | ✅ Productivo | Guías YAML versionadas, prompts defensivos, persistencia en DB, análisis en paralelo controlado, hashing de prompts y reuse cache. |
| **Reportes** | `src/fraud_scorer/templates/fraud_report_generator.py`, `src/fraud_scorer/templates/report_template.html` | ✅ Productivo | Render HTML/PDF con WeasyPrint, integra métricas `FraudMetrics`, secciones dinámicas y métricas agregadas. |
| **Orquestación v2** | `scripts/run_report.py` | ✅ Productivo | Clase `FraudAnalysisSystemV2`, control de cancelación, seguimiento de progreso (`_ProgressEmitter`), limpieza automatizada de artefactos previo a generar reportes. |
| **Persistencia** | `src/fraud_scorer/storage/db.py`, `src/fraud_scorer/storage/ocr_cache.py`, `src/fraud_scorer/storage/post_process_verifier.py` | ✅ Productivo | Esquema único (cases, documents, ocr_results, extracted_data, fraud_analyses, runs), índices por riesgo y hash. Verificador post-proceso asegura consistencia DB/FS. |
| **Replay & UI** | `src/fraud_scorer/services/replay_service.py`, `src/fraud_scorer/api/web_interface.py` | ✅ Operativo | Replay para casos históricos, cálculo de savings, UI consume la misma pipeline `FraudAnalysisSystemV2`. |
| **Monitoreo interno** | `scripts/system_integrity_check.py`, JSONL status (`data/temp/pipeline_cache/`) | ✅ En uso | Health-check integral, métricas por etapa vía EWMA, limpieza de duplicados en DB y caché. |
| **Machine Learning adaptativo** | `ML/` | 🟡 Exploratorio | Estructura inicial (datasets, notebooks) pendiente de entrenamiento con nuevos labels de fraude. |
| **Correlación inter-documentos & scoring global** | `src/fraud_scorer/analyzers/correlation/` | ✅ Productivo | Reglas determinísticas + correlador estadístico + RAG integrado, hallazgos persistidos en `fraud_correlations`. |

### Stack de modelos y configuración

- `settings.get_model_for_task(task)` centraliza selección (`extraction`, `consolidation`, `generation`). En producción sigue apuntando a `gpt-4o`/`gpt-4o-mini`; `gpt-5`/`gpt-5-mini` están preparados como override vía env.
- `FraudAnalyzer` permite override con `FRAUD_ANALYSIS_MODEL`/`FRAUD_ANALYSIS_MODEL_FALLBACK`. Default: `gpt-4o-mini` (por compatibilidad JSON).
- Todas las llamadas pasan por `instructor.patch` para respuesta estructurada y validada por Pydantic.
- Backoff y tolerancia a rate-limit implementados en `fraud_analyzer._call_ai_with_retry` y `ai_field_extractor._call_ai_with_retry`.

---

## 🔄 FLUJO OPERATIVO ACTUAL (`FraudAnalysisSystemV2`)

```
Carga/Staging (DocumentOrganizer)
        │
        ▼
OCR + Cache (AzureOCRProcessor + OCRCacheManager)
        │
        ▼
Clasificación (DocumentClassifier + aliases)
        │
        ▼
Extracción guiada (AIFieldExtractor + ExtractionConfig)
        │
        ▼
Consolidación (AIConsolidator → ConsolidatedExtraction)
        │
        ▼
Análisis fraude (FraudAnalyzer + FraudGuideManager + YAML)
        │
        ▼
Reportes HTML/PDF (FraudReportGenerator) + persistencia DB
```

### Destacados por fase

- **Fase A/B DocumentOrganizer** (`organize_documents_phase_a/b`): staging con renombrado automático, agrupación de guías/facturas, detección de destinatarios, copiado a rutas `data/uploads/renombre_de_documentos/YYYYMMDD-HHMMSS`.
- **OCR + Cache**: `OCRCacheManager` mantiene `case_index.json` y shards `cache_dir/<hash>.json`. `try_get_cached_ocr` evita re-procesos, `persist_ocr` sincroniza FS y DB.
- **Clasificación**: heurística base y fallback LLM (`ClassifierEngine`); usa alias para mapear a tipos canónicos antes de prompts.
- **Extracción**: prompts guiados incluyen contexto de póliza (`set_policy_context`). Validaciones y normalizaciones (`FieldValidator`, conversiones de montos).
- **Consolidación**: `AIConsolidator` emite conflictos resueltos y fuentes. Output alimenta generador de reportes y fraud analyzer.
- **Fraude**: `FraudAnalyzer.analyze_batch` aplica semáforos de paralelo y guarda `FraudAnalysisResult`. Resiliencia ante guías faltantes (análisis genérico) y almacenamiento en `fraud_analyses`.
- **Reportes**: `FraudReportGenerator.prepare_fraud_report_data` enriquece la estructura base y agrega métricas agregadas (documentos críticos, confianza promedio, etc.). WeasyPrint genera PDF final.
- **Progreso & cancelación**: `_ProgressEmitter` genera eventos JSONL (`data/temp/pipeline_cache/{case_id}.status.jsonl`) con EWMA y ETA. Métodos `cancel/is_cancelled` permiten abortar pipeline segura.

---

## 🗄️ PERSISTENCIA Y GESTIÓN DE DATOS

- **Base SQLite (`data/cases.db`)**  
  Tablas: `cases`, `documents`, `ocr_results`, `extracted_data`, `ai_analyses`, `fraud_analyses`, `runs`.  
  - `fraud_analyses`: guarda `risk_level`, `fraud_score`, `analysis_model`, `guide_version`, `analysis_uuid`, `prompt_hash`, timestamps.  
  - `storage/db.ensure_editor_columns` asegura compatibilidad con el editor de analistas (columnas `tentative_*`, `savings_amount`).
  - `post_process_verifier.verify_case_artifacts` cruza DB, caché y reorganizados; elimina duplicados por `file_hash` y alerta sobre shards faltantes.

- **Caché OCR** (`src/fraud_scorer/storage/ocr_cache.py`)  
  - Estructura `cache_dir/<hash_prefix>/<hash>.json` + `case_index/`.  
  - Mantiene index `document_hashes`, `documents`, `original_names`.  
  - Métricas disponibles via `OCRCacheManager.get_cache_stats()` y CLI `scripts/clean_orphaned_files.py`.

- **Repositorio de guías** (`src/fraud_scorer/guides/`)  
  - Guías YAML versionadas por tipo documental (19 guías activas).  
  - `FraudGuideManager` soporta aliases (`poliza`, `factura`, `tarjeta_circulacion`, etc.) y carga diferida.  
  - Validación vía `scripts/validate_guides.py`.

- **Plantillas & Templates**  
  - HTML base `templates/report_template.html` y respaldo `report_template_backup.html`.  
  - Reporte de fraude hereda de generador general para mantener consistencia visual.

---

## 📈 OBSERVABILIDAD Y OPERACIÓN

- **Monitoreo local**:  
  - `scripts/system_integrity_check.py` ejecuta health check (dependencias, credenciales, directorios, pipeline smoke test).  
  - `scripts/test_system.py` realiza pipeline de prueba con documento sintético.  
  - `scripts/db_maintenance.py` ofrece tareas de compactación y backup de `cases.db`.

- **Logs y métricas**:  
  - Logging central en cada módulo (`logger = logging.getLogger(__name__)`), niveles INFO/DEBUG.  
  - `ProgressEvent` (pydantic) define payload para seguimiento; UI consume JSONL para barra de progreso.  
  - `FraudMetrics` resume porcentaje de riesgo, indicadores encontrados, confianza promedio por reporte.

- **Reprocesos / Replay**:  
  - `services/replay_service.ReplayService` permite regenerar reportes, recalcular ahorros (`get_total_savings`) y limpiar artefactos huérfanos.  
  - Integración con UI (`api/web_interface.py`) reutiliza exactamente `FraudAnalysisSystemV2`.

- **Scripts auxiliares**:  
  - `scripts/replay_case.py`, `scripts/preview_classification.py`, `scripts/system_integrity_check.py` facilitan diagnósticos.  
  - `start_web_server.py` levanta UI FastAPI para uso interno.

---

## ⚠️ RIESGOS Y SIGUIENTES PASOS PRIORITARIOS

1. **Correlación inter-documental (Prioridad Alta)**  
   - Motor completo en producción: reglas determinísticas + correlador estadístico + RAG + persistencia en `fraud_correlations`.  
   - Tareas actuales: ampliar catálogo de reglas (VIN/GPS), métricas en Prometheus y surfacing completo en UI/dashboard.

2. **Sistema de verificaciones manuales inteligente (Prioridad Alta)**  
   - Gestor de checklists automáticos por tipo de documento.  
  - Integración planificada con APIs SAT/REPUVE/SCT (mock de datos listo, pendiente conectores productivos).

3. **Narrativa global y decisión de siniestro (Prioridad Media)**  
   - Síntesis en Observaciones/Conclusión utilizando indicadores individuales.  
   - Motor de scoring global con explicación (depende de correlación).

4. **Optimización de performance y costos (Prioridad Media)**  
   - Migración gradual a modelos `gpt-5` cuando disponibilidad/costos lo permitan.  
   - Paralelización OCR (batch) y fallback Tesseract para documentos fuera de Azure.

5. **Monitoreo productivo (Prioridad Media)**  
   - Dashboards Prometheus/Grafana para métricas de procesamiento, latencia y costos.  
   - Alertas automáticas ante acumulación de caché huérfano.

6. **Aprendizaje activo y dataset etiquetado (Prioridad Baja)**  
   - Consolidar dataset de intentos de fraude, alimentar modelos tradicionales (LightGBM) y calibrar recomendaciones.

---

## 📊 KPIs Y MÉTRICAS

- **Precisión en detección de tentativa** (objetivo operacional): 60-70% estimado con guías actuales → meta 90% con correlación y verificaciones manuales asistidas.
- **Tiempo de procesamiento**: 35-45 min/caso (pipeline actual) vs meta 30 min con paralelización OCR y caching agresivo.
- **Cobertura de guías**: 19 tipos activos; objetivo 30 para cubrir totalidad de expediente logístico.
- **Uso de caché**: reutilización >65% en re-procesos; meta 80% con sincronización DB/FS automática.
- **Calidad del reporte**: 100% secciones documentales completas cuando hay extracción; Observaciones/Conclusión pendientes de automatizar.

---

## 📚 ANEXOS

### A. Glosario Técnico
- **OCR**: Optical Character Recognition (Azure Document Intelligence).
- **LLM**: Large Language Model (OpenAI GPT-4o/gpt-5 family).
- **EWMA**: Exponential Weighted Moving Average para ETA.
- **REPUVE / SAT / SCT**: Fuentes regulatorias mexicanas para validaciones.
- **FraudGuide**: Definición YAML que describe metodología, indicadores y template de salida por documento.

### B. Stack Tecnológico Actual
- **Backend / API**: FastAPI + Uvicorn (`src/fraud_scorer/api`).
- **Procesamiento**: Python 3.10+, AsyncIO, OpenAI SDK, Instructor, Tenacity.
- **Persistencia**: SQLite (`data/cases.db`), caché en filesystem (`data/cache/`).
- **Infra**: Docker compose base, scripts CLI (`scripts/*.py`).
- **Generación de Reportes**: Jinja2 + WeasyPrint.
- **Observabilidad**: Logging estándar + JSONL, plan Prometheus/Grafana.

### C. Estimación de Recursos
- **Equipo**: 4 devs (backend/AI), 1 ML engineer, 1 analista de fraude/documental.
- **Timeline**: +3 meses para correlación + verificaciones manuales; +6 meses para decisión global asistida.
- **Costo incremental**: 25-30% adicional en créditos OpenAI durante fase de tuning (`gpt-5`).
- **ROI esperado**: 300% primer año con reducción de TAT y aumento de detección (mantiene estimación previa).

---

*Documento actualizado con estado real del sistema, arquitectura operativa y roadmap priorizado para continuar la evolución hacia decisiones globales automatizadas.*
