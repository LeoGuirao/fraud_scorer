# 📚 Guía Completa de Implementación - Sistema de Análisis de Fraudes por Documento
## Versión 1.2 (Actualización Q1 2026)

## 📋 Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Preparación del Entorno](#preparación-del-entorno)
4. [Fase 1: Infraestructura Base](#fase-1-infraestructura-base)
5. [Fase 2: Guías Especializadas](#fase-2-guías-especializadas)
6. [Fase 3: Motor de Análisis de Fraude](#fase-3-motor-de-análisis-de-fraude)
7. [Fase 4: Integración con el Pipeline Principal](#fase-4-integración-con-el-pipeline-principal)
8. [Fase 5: Generación de Reportes](#fase-5-generación-de-reportes)
9. [Testing y Validación Continua](#testing-y-validación-continua)
10. [Deployment y Operación](#deployment-y-operación)
11. [Mejores Prácticas](#mejores-prácticas)
12. [Troubleshooting](#troubleshooting)
13. [Referencias y Recursos](#referencias-y-recursos)

---

## 🎯 Resumen Ejecutivo

- El motor de fraude por documento está **operativo** dentro de `FraudAnalysisSystemV2` y reutiliza el 85% del pipeline existente.
- La información clave queda persistida en `data/cases.db` (`fraud_analyses`) y sincronizada con el caché OCR en filesystem.
- El sistema genera reportes enriquecidos (HTML/PDF) que incluyen métricas agregadas, indicadores por documento y secciones condicionales.
- Pendientes principales: correlación inter-documental, generación asistida de conclusiones globales y gestor de verificaciones manuales.

---

## 🏗️ Arquitectura del Sistema

### Visión general

```
DocumentOrganizer (Fase A/B)
      ↓
DocumentParser + AzureOCRProcessor
      ↓
DocumentClassifier (heurístico + LLM)
      ↓
AIFieldExtractor (prompts guiados)
      ↓
AIConsolidator (resuelve conflictos)
      ↓
FraudAnalyzer (guías YAML + prompts)
      ↓
FraudReportGenerator (HTML/PDF)
      ↓
Persistencia (SQLite + caché OCR + artefactos)
```

### Repositorio y módulos clave

| Dominio | Ubicación | Descripción |
|---------|-----------|-------------|
| Orquestación | `scripts/run_report.py` | Define `FraudAnalysisSystemV2`, manejo de progreso, cancelación y limpieza de artefactos. |
| Parsing & staging | `src/fraud_scorer/parsers/document_parser.py`, `src/fraud_scorer/processors/document_organizer.py` | Ingresa archivos, normaliza texto, aplica renombrado estándar y rutas. |
| OCR & caché | `src/fraud_scorer/processors/ocr/azure_ocr.py`, `src/fraud_scorer/storage/ocr_cache.py` | Procesa documentos con Azure; caché con índices por caso y hash. |
| Clasificación | `src/fraud_scorer/processors/document_classifier.py`, `src/fraud_scorer/classification/` | Determina tipo documental y alias canónicos. |
| Extracción | `src/fraud_scorer/processors/ai/ai_field_extractor.py`, `src/fraud_scorer/prompts/extraction_prompts.py` | Guías de extracción, validaciones y rutas `ExtractionRoute`. |
| Consolidación | `src/fraud_scorer/processors/ai/ai_consolidator.py` | Combina resultados y produce `ConsolidatedExtraction`. |
| Motor de fraude | `src/fraud_scorer/analyzers/fraud_analyzer.py`, `src/fraud_scorer/analyzers/fraud_guide_manager.py`, `src/fraud_scorer/prompts/fraud_prompts.py`, `src/fraud_scorer/guides/` | Analiza por documento usando guías YAML. |
| Plantillas | `src/fraud_scorer/templates/` | `AIReportGenerator` base y `FraudReportGenerator` extendido para fraude. |
| Persistencia | `src/fraud_scorer/storage/` | DB SQLite (`db.py`), caché OCR, verificador post-proceso (`post_process_verifier.py`), casos (`cases.py`). |
| Servicios & UI | `src/fraud_scorer/services/replay_service.py`, `src/fraud_scorer/api/web_interface.py` | Replay, estadísticas y endpoints FastAPI. |
| Pipelines auxiliares | `src/fraud_scorer/pipelines/data_flow.py` | Normalizadores de OCR, build de docs para templates, mapping canónico. |

---

## 🔧 Preparación del Entorno

1. **Dependencias**
   - Instalar con `pip install -r requirements.txt`.
   - Herramientas clave: `openai`, `azure-ai-formrecognizer`, `instructor`, `tenacity`, `jinja2`, `weasyprint`, `pyyaml`.

2. **Variables de entorno (`.env`)**
   - `AZURE_ENDPOINT`, `AZURE_OCR_KEY`, `OPENAI_API_KEY`.
   - Configurables opcionales: `FRAUD_ANALYSIS_MODEL`, `FRAUD_ANALYSIS_MODEL_FALLBACK`, `FS_DATA_DIR`, `FRAUD_DB_PATH`, `FRAUD_CONFIDENCE_THRESHOLD`.

3. **Directorios esperados**
   - `data/uploads`, `data/reports`, `data/cache`, `data/temp`.
   - `scripts/system_integrity_check.py` valida la estructura y credenciales.

4. **Inicialización de base de datos**
   - `python scripts/db_maintenance.py --init` (o `python -m fraud_scorer.storage.db` si se expone).
   - Verificar tablas via `sqlite3 data/cases.db ".tables"`.

5. **Smoke tests**
   - `python scripts/test_system.py` (verifica dependencias, conexiones y pipeline básico).
   - `python scripts/system_integrity_check.py` para diagnóstico completo.

---

## 🧱 Fase 1: Infraestructura Base

### 1.1 DocumentOrganizer

- Fase A (`organize_documents_phase_a`): staging y clasificación barata.
- Fase B (`organize_documents_phase_b`): extracción guiada, renombrado definitivo, sincronización con `OCRCacheManager`.
- Métricas internas (`OrganizationMetrics`) registran tiempos y uso de fallback LLM.

### 1.2 DocumentParser

- Normaliza archivos PDF/TXT/DOCX → `{"text", "tables", "metadata"}`.
- Interactúa con `AzureOCRProcessor` cuando requiere OCR completo.

### 1.3 OCR y caché

- `AzureOCRProcessor.process_document` produce `OCRResult`.
- `OCRCacheManager.try_get_cached_ocr` reutiliza resultados (key: hash de contenido).
- `case_index` guarda correspondencia `document_id` ↔ hash ↔ nombre original.

### 1.4 Almacenamiento

- `storage/db.py` controla conexión (`get_conn`), inicialización (`init_db`), columnas adicionales para editor analista (`ensure_editor_columns`).
- `storage/post_process_verifier.verify_case_artifacts` se ejecuta al finalizar pipeline para asegurar consistencia DB/FS.

---

## 🧭 Fase 2: Guías Especializadas

- Repositorio: `src/fraud_scorer/guides/` (YAML).
- Estructura base:
  ```yaml
  metadata:
    type: carta_de_reclamacion_formal_a_la_aseguradora
    version: "1.2"
  methodology:
    fraud_indicators:
      high_risk:
        - pattern: ...
          detection: ...
          severity: alto
    validation_rules:
      required_fields: [...]
    cross_reference_documents: [...]
  response_template:
    output_format:
      risk_level: ""
      fraud_score: 0.0
      confidence: 0.0
      analisis_completo: ""
      indicators: []
      evidence: []
      recommendations: []
      validation_tasks: []
  ```
- `FraudGuideManager`:
  - Carga en inicialización, soporta YAML/JSON.
  - Aliases integrados (poliza → poliza_de_la_aseguradora, etc.).
  - Loggea guías disponibles.
- Validación: `python scripts/validate_guides.py --list` / `--check cfd_carta_porte`.

---

## 🧠 Fase 3: Motor de Análisis de Fraude

### 3.1 FraudAnalyzer (`src/fraud_scorer/analyzers/fraud_analyzer.py`)

- Cliente OpenAI via `AsyncOpenAI` + `instructor.patch`.
- Modelo por defecto: `gpt-4o-mini`; fallback configurable.
- Pipeline:
  1. Obtener guía (`FraudGuideManager.get_guide`).
  2. Construir prompt (`FraudPromptBuilder.build_fraud_analysis_prompt`).
  3. Llamar AI con retry (`_call_ai_with_retry`, fallback sin `response_format` si falla).
  4. Parsear a `FraudAnalysisResult` (Pydantic).
  5. Enriquecer (`_enrich_analysis`): normaliza indicadores/evidencias, aplica reglas guía.
  6. Persistir (`_save_analysis_to_db`).

- Concurrencia controlada (`asyncio.Semaphore` en `analyze_batch`).
- Manejo de errores: genera análisis de error (`_create_error_analysis`) para diagnóstico.

### 3.2 FraudPromptBuilder

- Prompt defensivo (anti prompt-injection, foco en evidencia, JSON estricto).
- Incluye indicadores high/medium risk y contexto opcional (ej. datos consolidados).
- Limita OCR text a 8k chars, incluye key-values y tablas recortadas.

### 3.3 Modelos Pydantic (`models/fraud_analysis.py`)

- `FraudAnalysisResult`, `FraudIndicator`, `RiskLevel`, `FraudMetrics`.
- Validación cruzada score ↔ riesgo.
- Campos `analysis_id`, `prompt_hash`, `processing_time_ms`, `timestamp`.

---

## 🔌 Fase 4: Integración con el Pipeline Principal

- `scripts/run_report.py` → `FraudAnalysisSystemV2`.
- Fases:

  1. **Ingesta**: staging, registro en DB (`create_case`), deduplicación por hash (`sha256_of_file`).
  2. **OCR / Extracción / Consolidación**: reuso de componentes v2.
  3. **Análisis de fraude**:
     ```python
     fraud_analyses = await analyzer.analyze_batch(
         documents=docs_for_fraud,
         case_id=case_id,
         parallel_limit=3,
         context={"consolidated_fields": consolidated.consolidated_fields.model_dump()}
     )
     ```
  4. **Persistencia**: guardado en `fraud_analyses`, actualización `case_index`.
  5. **Reportes**: generación HTML/PDF + guardado JSON (payload completo).
  6. **Verificación**: `verify_case_artifacts(case_id)` se ejecuta antes de finalizar.

- **Tracking**:
  - `_ProgressEmitter`: emite eventos `ProgressEvent` (`stage`, `status`, `eta_ms`) → UI.
  - Callbacks de cancelación (`cancel`, `is_cancelled`) disponibles para UI/CLI.
  - Limpieza automática de artefactos previos (`_clean_previous_case_files`).

---

## 🪄 Fase 5: Generación de Reportes

- **HTML/PDF**: 
  - `FraudReportGenerator` hereda de `AIReportGenerator`.
  - `prepare_fraud_report_data` combina `ConsolidatedExtraction` + `FraudAnalysisResult`.
  - Plantilla `report_template.html` muestra secciones condicionales (`mostrar_seccion_fraude`).

- **Artifacts**:
  - Nomenclatura: `{asegurado}_{numero_siniestro}_INFORME.html/pdf`.
  - JSON del caso se guarda en `data/reports/{case_id}.json` (ver run_report).
  - Limpieza de versiones previas antes de generar nuevos.

- **Métricas**:
  - `FraudMetrics` (documentos totales, críticos, alto riesgo, confianza promedio, indicadores totales).
  - Visualización de riesgo por documento con códigos de color (`_get_risk_color`).

---

## ✅ Testing y Validación Continua

| Tipo | Herramienta | Descripción |
|------|-------------|-------------|
| Smoke test pipeline | `python scripts/test_system.py` | Crea documento sintético y ejecuta pipeline v2. |
| Health check integral | `python scripts/system_integrity_check.py` | Ambiente, dependencias, credenciales, pipeline, bases. |
| Validación guías | `python scripts/validate_guides.py --list` | Revisa estructura YAML y campos obligatorios. |
| Pruebas unitarias | `pytest` (ver `tests/`) | Cobertura para normalización OCR, validadores y parsers. |
| Replay | `python scripts/replay_case.py --case CASE_ID` | Regenera reportes, verifica consistencia DB/caché. |

Recomendaciones:
- Ejecutar health check antes de deploy.
- Añadir fixtures de `fraud_analyses` en tests para validar nuevos indicadores/guías.
- Configurar pipeline CI (GitHub Actions) para `pytest` + lint y validación de guías.

---

## 🚀 Deployment y Operación

1. **Modo CLI**:  
   - `python scripts/run_report.py --input <folder> --case CASE123 --fraud`  
   - Flags: `--fraud` (habilita análisis por documento), `--no-clean` (opcional), `--reuse-ocr`.

2. **Modo API/UI**:  
   - `python scripts/start_web_server.py` (FastAPI).  
   - Endpoint: `/api/v1/reports/generate` admite payload con `enable_fraud_analysis`.

3. **Base de datos y backups**:  
   - `scripts/db_maintenance.py --vacuum` y `--backup`.  
   - Automatizar backup diario de `data/cases.db` + `data/cache`.

4. **Limpieza periódica**:  
   - `scripts/clean_orphaned_files.py` elimina shards huérfanos.  
   - Revisar métricas `OCRCacheManager.get_cache_stats()`.

5. **Monitoreo**:  
   - Consumir JSONL de progreso para dashboards internos.  
   - Propuesta: exponer métricas Prometheus (`fraud_analyses_total`, `high_risk_cases_current`, etc.) — ver sección TODO en doc.

6. **Correlación inter-documental**:  
   - Validar reglas previa liberación con `python scripts/lint_correlation_rules.py`.  
   - Reindexar el Agente Rick para cada caso impactado (botón *Reindexar* del editor o `POST /api/rick/reindex`).  
   - Ejecutar `python scripts/system_integrity_check.py` y revisar la sección *Motor de correlación* (hallazgos y auditoría RAG).

---

## 🛡️ Mejores Prácticas

- **Gestión de prompts**: versionar en Git, documentar cambios en commits, realizar A/B testing controlado.
- **Control de costos**: usar `gpt-4o-mini` para extracción/fraude y reservar `gpt-4o` para consolidación; medir tokens por etapa.
- **Seguridad**: sanitizar PII en logs (`logger.info` evita dumping de OCR completo), cifrar `data/reports` si se montan en entornos compartidos.
- **Caché**: mantener `case_index` sincronizado (no modificar manualmente), usar `replay_service` para limpieza.
- **Extensibilidad**: añadir nuevos tipos documentales actualizando `DocumentType`, `ExtractionConfig.DOCUMENT_FIELD_MAPPING` y creando guía YAML.

---

## 🧯 Troubleshooting

| Problema | Síntoma | Diagnóstico | Solución |
|----------|---------|-------------|----------|
| **Rate limiting OpenAI** | Retries agotados, logs `RateLimitError` | Revisar `_call_ai_with_retry` | Ajustar `parallel_limit`, añadir tiempos de espera, considerar claves dedicadas. |
| **Guía faltante** | `No se encontró guía` en logs | `FraudGuideManager.get_guide` devolvió `None` | Crear guía YAML o añadir alias en manager. |
| **Schema mismatch DB** | Error al guardar en `fraud_analyses` | Tabla sin columnas nuevas | Ejecutar `ensure_editor_columns()` o `init_db()` tras despliegue. |
| **OCR inconsistente** | Texto vacío | Azure falló o documento no soportado | Revisar `OCRCacheManager.try_get_cached_ocr`, usar fallback parse (TXT/CSV), validar formato. |
| **Reporte sin sección fraude** | `mostrar_seccion_fraude = False` | No hubo análisis válido | Confirmar `--fraud`/`enable_fraud_analysis`, revisar logs de `FraudAnalyzer` para excepciones. |
| **Caché desincronizado** | `hashes_missing_in_index` en verificador | FS o DB manualmente modificados | Ejecutar `clean_orphaned_files.py`, re-generar case index. |

---

## 📚 Referencias y Recursos

- Documentación interna adicional:  
  - `VISION_SISTEMA_ANALISIS_SINIESTROS.md` (visión estratégica).  
  - `GUIA_IMPLEMENTACION_ANALISIS_INDIVIDUAL_DOCUMENTOS.md` (detalle por documento).

- Recursos externos recomendados:  
  - [OpenAI API](https://platform.openai.com/docs)  
  - [Azure Document Intelligence](https://learn.microsoft.com/azure/ai-services/document-intelligence/)  
  - [WeasyPrint](https://weasyprint.org/)  
  - [Pydantic](https://docs.pydantic.dev/)  

---

**Última actualización:** Febrero 2026  
**Autoría:** Equipo de Plataforma de Fraude Documental (CTO + Ingeniería IA)
