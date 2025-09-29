# Guía de Implementación — Visibilidad y Personalización del Reporte en el Editor del Analista

## Contexto
- Objetivo: permitir que el analista marque cada documento para mostrarse u ocultarse en el reporte final, habilitar el reproceso individual del análisis de fraude (fase 3.5) sin impactar el resto del caso y ofrecer edición manual, por secciones, del informe HTML/PDF.
- La guía asume el estado actual descrito en `BETTER_PRACTICES.md` y en `scripts/run_report.py` (`FraudAnalysisSystemV2`). Cualquier desviación debe detectarse antes de ejecutar los cambios.
- Este plan privilegia la reutilización del pipeline vigente, respeta el flujo *docless* documentado en Better Practices §1 y mantiene intacta la plantilla HTML/PDF de producción (`src/fraud_scorer/templates/report_template.html`).

## Fase 0 — Preparación y Auditoría
- Leer `BETTER_PRACTICES.md` (en especial §§1, 2 y 6) para refrescar las protecciones sobre reprocesos selectivos, consistencia del `case_index` y limpieza posterior.
- Respaldar `data/cases.db` y el índice del caso (`data/ocr_cache/case_index/CASE-*.json`) antes de cualquier migración.
- Ejecutar `python scripts/post_process_verifier.py CASE-XXXX` sobre un caso reciente para tener una foto base de conteos (ver Better Practices §6.2).
- Confirmar variables sensibles en `.env` (`OPENAI_API_KEY`, `AZURE_OCR_KEY`) antes de disparar reprocesos, tal como se documenta en Better Practices.

- [x] **Fase 1 — Modelo de Datos y Persistencia**
  - Modelos extendidos con `include_in_report` en `FraudAnalysisResult` y `DocumentAnalysisSection`.
  - Esquema `fraud_analyses` migrado con columna `include_in_report` y helper idempotente `ensure_fraud_visibility_column()` integrado en `get_conn()`/`init_db()`.
  - `_save_analysis_to_db` persiste el flag y `_hydrate_fraud_results` asegura un valor por defecto para resultados antiguos. El índice del caso conserva la bandera.
  - Backfill ejecutado (`UPDATE fraud_analyses SET include_in_report = 1 WHERE include_in_report IS NULL`) para sanear datos históricos y alinear índices existentes.

- [x] **Fase 2 — Servicios Backend y Reproceso Selectivo**
  - `FraudDocumentCatalog` y `FraudDocumentReprocessService` centralizan catálogo, toggles y reprocesos individuales (`src/fraud_scorer/services/fraud_document_service.py`). Se reaprovecha `_prepare_docless_ocr`, se normaliza el `document_type` con `FraudGuideManager.normalize_type()` y se rehidratan correlaciones/reportes al cierre.
  - `/api/editor/{case_id}/bootstrap` incluye `fraud_documents_preview` con id, nombre, tipo, riesgo, score y `include_in_report` para alimentar la UI (`src/fraud_scorer/api/web_interface.py`).
  - Catálogo completo expuesto vía `GET /api/editor/{case_id}/fraud-documents`; cada entrada trae `FraudAnalysisResult.model_dump()` más metadatos (`last_run`, `analysis_uuid`, `include_in_report`).
  - Toggle defensivo `PATCH /api/editor/{case_id}/fraud-documents/{document_id}` persiste el flag en DB/case_index, devuelve preview actualizado y registra un estatus instantáneo en `processing_status` (`fraud-toggle-*`).
  - Reproceso individual `POST /api/editor/{case_id}/fraud-documents/{document_id}/reprocess` ejecuta en background: reconstruye OCR/extractions, llama a `FraudAnalyzer.analyze_document`, sincroniza `fraud_correlations`, regenera HTML/PDF y publica progreso con estados `queued → re-analyzing → refreshing-report → completed`. El resultado se consulta en `/status/{process_id}`.
  - Validaciones adicionales bloquean reprocesos concurrentes por `case_id` + `document_id` tanto en `active_jobs` como en `processing_status`. Las excepciones producen `HTTP 409` para que la UI maneje el estado.
  - Endpoints expuestos y consumibles por la UI del editor; devuelven catálogo completo, vista previa y progreso en tiempo real para que la capa visual (Fase 4) se conecte sin lógica adicional (`src/fraud_scorer/api/web_interface.py`).

- [x] **Fase 3 — Generación de Reportes**
  - `FraudReportGenerator.prepare_fraud_report_data()` filtra los análisis con `include_in_report=False`, mantiene métricas globales y añade indicadores `documentos_publicables`, confianza e indicadores de la vista filtrada (`src/fraud_scorer/templates/fraud_report_generator.py`).
  - La plantilla `report_template.html` permanece intacta; sólo la data se filtra, por lo que los documentos ocultos desaparecen automáticamente del HTML/PDF.
  - Los toggles del editor disparan una regeneración ligera del informe (HTML inmediato, PDF diferido para no bloquear) con `FraudDocumentReprocessService.refresh_report`, garantizando que la vista final responda al instante a los cambios de visibilidad sin saturar WeasyPrint (`src/fraud_scorer/api/web_interface.py`, `src/fraud_scorer/services/fraud_document_service.py`).
  - Se introdujo `report_override_html` en el `case_index` para almacenar, de forma sanitizada, la versión manual completa del HTML. Cuando existe ese override se escribe sobre el archivo final y se utiliza también para generar el PDF.
  - Se preserva cache busting al regenerar el reporte mediante el refresco del iframe con query param temporal desde la UI (`static/js/editor_analista.js`).

- [x] **Fase 4 — UI del Editor del Analista**
  - Rediseño del `editor_analista.html`: tabs "Panel de análisis" / "Vista final", listado dinámico con tarjetas, estados colapsados para documentos ocultos y botones `Reprocesar` (`src/fraud_scorer/api/templates/editor_analista.html`).
  - Hoja de estilos extendida para toggles accesibles (`aria-pressed`), badges de riesgo, bloqueo durante reproceso y estados vacíos; la pestaña "Vista final" ahora resalta el iframe cuando se activa el modo edición (`static/css/editor_analista.css`).
  - Frontend con `refreshFraudDocuments`, toggles `PATCH`, reprocesos individuales monitorizados y cache busting del iframe. Además incorpora edición inline del reporte completo: `enterReportEditMode` habilita `contentEditable` dentro del iframe, `PUT /api/editor/{case_id}/report-html` guarda el HTML personalizado y `DELETE /api/editor/{case_id}/report-html` restaura el generado automáticamente (`static/js/editor_analista.js`, `src/fraud_scorer/api/web_interface.py`).
  - El HTML manual se sanitiza (sin `<script>` ni manejadores `on*`) antes de persistirse; los reprocesos siguen respetando el override y pueden revertirse con un click desde la UI (`src/fraud_scorer/services/fraud_document_service.py`).

## Fase 5 — QA y Verificación Operativa
- **Pruebas unitarias**:
  - `tests/test_fraud_analysis_model.py`: validar que `include_in_report` se conserva al serializar/rehidratar.
  - `tests/api/test_editor_toggle.py`: cubrir toggle y respuesta esperada en DB/índice.
  - `tests/api/test_editor_reprocess_document.py`: mockear `FraudAnalyzer` para confirmar rehidratación docless y actualización de correlaciones.
- **Pruebas de integración**:
  - Correr `pytest tests/editor` (crear carpeta si no existe) y asegurar compatibilidad con la red restringida (mock de OpenAI según patrón existente).
  - Ejecutar `python scripts/post_process_verifier.py CASE-XXXX` antes y después para validar que `fraud_analyses` y `fraud_correlations` permanecen alineados (Better Practices §6).
  - Validar que `GET /report/{case_id}` no muestre documentos ocultos y que las métricas del dashboard (conteos de documentos) sigan intactas.
  - Resultado actual: `pytest` completo (23 tests OK, 4 skipped por fixtures async) tras ajustar fixtures de correlación para los nuevos requisitos de `DocumentExtraction` y `FraudAnalysisResult`.
- **Pruebas manuales**:
  - Flujos con caso sin archivos originales (docless) para confirmar que el reproceso individual reutiliza el caché.
  - Reprocesar múltiples documentos en paralelo: confirmar que la UI maneja colas y que `processing_status` no se corrompe.
  - Alternar visibilidad varias veces y generar PDF para verificar que la versión descargada respeta los toggles.
- **Monitoreo**: agregar logs temporales en el backend que indiquen quién ejecutó el toggle/reproceso. Revisar Splunk/ELK si aplica.

## Fase 6 — Despliegue y Rollback
- Desplegar primero en ambiente de staging ejecutando el script de backfill y validando un caso real.
- Plan de rollback: revertir commits + volver a cargar respaldo de `cases.db` y restaurar los `case_index` respaldados.
- Mantener activo el flag `ENABLE_CLASSIFICATION_REVIEW` durante las primeras horas post-despliegue para capturar reprocesos inesperados como recomienda Better Practices.
- Documentar el cambio en `EDITOR_ANALISTA_IMPLEMENTATION_GUIDE.md` y en el changelog interno.

## Notas y Preguntas Abiertas
- Definir junto al área de negocio si `include_in_report=False` debe excluir al documento de los cálculos de riesgo tentativo (`ReplayService._detect_tentative_fraud`). Si la respuesta es sí, actualizar esa función para filtrar.
- Revisar si `AgenteRick` o la ingesta vectorial (`ai/ingestion/document_loader.py`) necesita ignorar documentos ocultos o seguir ingestándolos para búsquedas internas.
- Alinear el mensaje al analista cuando ocultar todos los documentos deje la sección vacía: podría mostrarse un aviso "No hay documentos publicados en el reporte final".
- Considerar auditoría más granular (quién ocultó/mostró cada documento y cuándo) para futuras versiones.
- Evaluar versionado y diff del HTML manual (p.ej. histórico por edición) cuando la personalización sea adoptada de forma masiva.
