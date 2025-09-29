# Guía de Implementación — Motor de Correlación de Fraude

## 📌 Propósito
Diseñar e implementar un motor de correlación inter-documental que complemente el análisis de fraude por documento (Fase 3.5) sin romper los flujos existentes de `FraudAnalysisSystemV2`. El motor debe detectar contradicciones, anomalías temporales y discrepancias de entidades entre documentos, apoyándose en reglas determinísticas, correlación estadística y evidencia contextual (RAG) mientras mantiene la trazabilidad y principios descritos en `BETTER_PRACTICES.md`.

## 🧭 Principios Rectores (en línea con `BETTER_PRACTICES.md`)
- **Cero impacto en fases previas**: Reutilizar `case_index`, caché OCR y estructuras existentes. No reintroducir dependencias con archivos originales; operar con JSON reorganizados si el caso está en modo docless.
- **Persistencia única**: Continuar usando `data/cases.db` y `OCRCacheManager` como fuentes de verdad. Cualquier nueva tabla debe seguir las reglas de índices, vaciado y verificación (`post_process_verifier`).
- **Reproceso seguro**: Permitir invocar el motor en reprocesos 3.5 sin requerir OCR/extracción adicionales. Respetar flags `reprocess_fraud`, `reprocess_ocr`, `reprocess_extraction`.
- **Auditabilidad**: Guardar `rule_id`, `rule_version`, `documents_involved`, `evidence`, `prompt_hash` (cuando intervenga LLM). Integrarse con auditorías existentes.
- **Extensibilidad controlada**: Reglas como YAML versionadas; código desacoplado en módulos claros. Cambios en reglas no deben requerir despliegues de código.
- **Fallback defensivo**: Cualquier fallo (falta de datos, evidencias insuficientes) debe degradarse a estado `needs_manual_review` sin romper pipeline.

## 📊 Estado actual (marzo 2026)
- Cobertura de reglas extendida: montos (reclamación vs suma asegurada), cronología de denuncia/carpeta, identificación vehicular (VIN/placas) y rutas GPS vs Carta Porte (`rules/correlation_rules.yaml`).
- Nuevo comparador numérico (`numeric_order`) en el RuleEngine para validar umbrales con tolerancias relativas/absolutas.
- Métricas operativas expuestas (`fraud_correlation_*`) y resumen RAG en `CorrelationReport.metadata`, con auditoría JSONL etiquetada por módulo.
- El editor analista y el dashboard exponen hallazgos (`/api/case/{id}/correlations`) con filtros de estado/severidad y tarjetas de resumen.
- Script de lint unificado (`scripts/lint_correlation_rules.py`) que valida reglas determinísticas, mapeos y configuración estadística antes de desplegar.

## 🏗️ Arquitectura Objetivo
```
src/
└── fraud_scorer/
    └── analyzers/
        └── correlation/
            ├── orchestrator.py
            ├── engines/
            │   ├── rule_engine.py
            │   ├── statistical_correlator.py
            │   └── rag_evidence_builder.py
            ├── models/
            │   ├── case_context.py
            │   ├── correlation_result.py
            │   └── finding.py
            └── rules/
                ├── correlation_rules.yaml
                └── entity_mappings.yaml
```

- **Orchestrator** (`CorrelationEngine`): coordina la carga de contexto, ejecución de reglas, análisis estadístico y evidencia RAG.
- **Rule Engine**: evalúa condiciones determinísticas usando datos estructurados (`DocumentExtraction`, `ConsolidatedExtraction`, `fraud_analyses`).
- **Statistical Correlator**: calcula correlaciones Kendall/Spearman/Pearson para datos cuantitativos/temporales y genera hallazgos estructurados.
- **RAG Evidence Builder**: reutiliza `AgenteRickService` con consultas adaptativas para obtener fragmentos textuales que respalden hallazgos `needs_context`.
- **Modelos**: definen contratos (`CaseContext`, `CorrelationFinding`, `CorrelationReport`).
- **Reglas**: catálogo YAML versionado + mapeos de entidades/campos y documentos canónicos (alineados con `settings.CANONICAL_TO_ALIAS`).

## 🔌 Entradas de Datos
- `fraud_analyses` (tabla SQLite) → indicadores, risk_level, prompt_hash.
- `extracted_data` y `ConsolidatedExtraction` → campos normalizados.
- `OCRCacheManager` → metadatos (páginas, lenguaje, nombres originales) cuando se requiera cita textual.
- `AgenteRickService` → vector store por `case_id` (ya existente en `data/chroma/`).

## 🚦 Flujo de Ejecución
1. **Crear `CaseContext`**: Combinar datos de DB y caché.
   ```python
   context = CaseContext.from_case(
       case_id=case_id,
       consolidated=consolidated,
       extractions=extractions,
       fraud_results=fraud_analyses,
       cache_manager=OCRCacheManager(),
   )
   ```
2. **Evaluar Reglas Determinísticas**: `rule_engine.evaluate(context)` devuelve lista de `CorrelationFinding` con estados `pass`, `fail`, `needs_context`.
3. **Análisis Estadístico**: `statistical_correlator.analyze(context)` produce hallazgos adicionales (anomalías de montos/fechas).
4. **Evidencia RAG**: Para hallazgos `needs_context`, `rag_evidence_builder.build(findings, case_id)` consulta Agente Rick, adjunta evidencias y actualiza el estado del hallazgo.
5. **Síntesis**: `CorrelationEngine` consolida y retorna `CorrelationReport` con métricas agregadas.
6. **Persistencia**: Guardar cada hallazgo en la tabla `fraud_correlations`, anexar el reporte al JSON final y exponerlo en el dashboard/editor.

## 🧱 Implementación Paso a Paso

### Fase 1 — Fundación (2 sprints)
1. **Modelado** (`src/fraud_scorer/analyzers/correlation/models/`)
   - `CaseContext`: loader desde DB (`get_conn`), agrupa `documents`, `entities`, `timeline`.
   - `CorrelationFinding`: campos `id`, `severity`, `status`, `documents`, `evidence`, `recommendation`.
   - `CorrelationReport`: resumen global + matriz de hallazgos.
2. **Reglas YAML (`rules/correlation_rules.yaml`)**
   ```yaml
   - id: policy_vs_invoice_amount
     version: "1.0"
     severity: high
     description: Montos reclamados deben coincidir entre póliza y facturas.
     documents:
       - poliza_de_la_aseguradora
       - facturas_comerciales_internacionales
     entities:
       - monto_reclamacion
     condition:
       type: equality
       source: consolidated.consolidated_fields.monto_reclamacion
       target: aggregates.facturas.monto_total
       tolerance: 0.05
   ```
   - Reutilizar mapeos existentes (`settings.DOCUMENT_FIELD_MAPPING`, `FraudGuideManager` aliases).
   - Versionar en Git y documentar cambios en encabezado del archivo.
3. **Rule Engine (`engines/rule_engine.py`)**
   - Parser de reglas YAML → objetos en memoria.
   - Evaluador de condiciones (`equality`, `temporal_order`, `set_overlap`, `exists`).
   - Genera `CorrelationFinding` con campos poblados y evidencia estructurada (sin LLM).
4. **Integración con Pipeline** (`scripts/run_report.py`)
   ```python
   from fraud_scorer.analyzers.correlation.orchestrator import CorrelationEngine

   correlation_engine = CorrelationEngine()
   correlation_report = correlation_engine.run(
       case_id=case_id,
       consolidated=consolidated,
       fraud_results=fraud_analyses,
       extractions=extractions,
   )
   ```
   - Solo se ejecuta si `self.enable_fraud`.
   - Respetar `reprocess_mode`: si es reproceso 3.5, cargar datos desde cache JSON.
5. **Testing**
   - Unit tests (`tests/correlation/test_rule_engine.py`) con fixtures de casos reales.
   - Aprovechar `real_cases/` y generar datos sintéticos para cada regla P0.

### Fase 2 — Estadística & RAG Adaptativo (completa)
1. **Statistical Correlator (`engines/statistical_correlator.py`)**
   - Configurado vía `rules/statistical_config.yaml` (outliers, ratios y brechas temporales).
   - Emite hallazgos `finding_type="statistical"`, suma metadatos como media, desviación y z-score.
   - Degrada a `needs_context` cuando faltan datos suficientes (ej. ratio sin referencia).
2. **RAG Evidence Builder (`engines/rag_evidence_builder.py`)**
   - Integra `AgenteRickService`, construye prompts con contexto y adjunta evidencia cuando hay respuesta.
   - Añade `prompt_hash`, latencia y token usage al hallazgo; conserva el estado original si Rick responde `NO_CONTEXT`.
3. **Instrumentación**
   - `CorrelationReport.metadata` incluye conteos por origen (reglas/estadística) y bandera `rag_enabled`.
   - Hallazgos persistidos en `fraud_correlations` para auditoría y consultas posteriores.
4. **Reportes & UI**
   - `FraudReportGenerator` incorpora tabla con estado, severidad, evidencia y etiquetas.
   - El dashboard/Editor consumen `case_index['fraud_correlations']` o, en su defecto, leen desde DB.
5. **QA & Validación**
   - Suite de tests en `tests/correlation/test_correlation_engine.py` cubre reglas, estadísticas y RAG.
   - Reproceso 3.5 docless verificado: motor reutiliza extracciones y fraude previos.

### Fase 3 — Backlog Investigativo (condicionado)
- **Entity Graph + GNN (experimento)**: solo si existe dataset etiquetado y bandwidth ML. Mantener en rama aparte hasta demostrar valor (precision/recall > reglas).
- **Policy-aware rules**: evaluar cuando exista necesidad de actualizar reglas en caliente; precaución con drift.
- Cualquier feature experimental debe respetar principios de auditabilidad y fallback.

## 🧰 Buenas Prácticas Operativas
- Versionar reglas y mapeos junto con plantillas; usar convención `vMAJOR.MINOR`.
- Actualizar `scripts/validate_guides.py` o crear `scripts/validate_correlation_rules.py` para lint de YAML (duplicados, campos requeridos).
- Registrar hallazgos manuales/overrides en DB para retroalimentar futuro ML (hook en UI).
- Documentar nuevos comandos en `README.md` y agregar entradas en `BETTER_PRACTICES.md` cuando afecten flujos existentes.

## ✅ Checklist de Calidad
- [x] `CorrelationEngine` se integra en `run_report.py` sin romper reprocesos.
- [x] Reglas base cubren montos, fechas clave, VIN/placas, rutas vs GPS.
- [x] Tests unitarios/integración ejecutados (`pytest tests/correlation`).
- [x] Logs y auditoría actualizados (`agent_rick_audit.jsonl`).
- [x] Reporte HTML y UI (dashboard/editor) muestran correlaciones con filtros y citas.
- [x] Documentación actualizada (`README`, esta guía, roadmap en `VISION_SISTEMA_ANALISIS_SINIESTROS.md`).


## 📈 Métricas Iniciales a Monitorear
- % de reglas evaluadas (`pass/fail/needs_context`).
- Tasa de hallazgos confirmados por RAG vs. rechazados.
- Tiempo de ejecución promedio del módulo (< 2s completo, < 100ms reglas).
- Incidencias manuales registradas (para retroalimentar reglas).

## 📚 Referencias
- `BETTER_PRACTICES.md`
- `VISION_SISTEMA_ANALISIS_SINIESTROS.md`
- `GUIA_IMPLEMENTACION_SISTEMA_FRAUDES copy.md`
- `AGENTE_RICK_IMPLEMENTATION_GUIDE.md`

---

*Versión inicial 1.0 — Febrero 2026. Esta guía debe evolucionar con cada release mayor del motor de correlación.*
