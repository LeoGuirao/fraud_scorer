# Better Practices Fraud Scorer: Reproceso 3.5, Detección de Duplicados y Conteos

Este documento (Better Practices) describe en detalle los problemas detectados y las soluciones implementadas en el sistema Fraud Scorer en seis áreas:

- Reprocesamiento exclusivo de la Fase 3.5 (análisis de fraude) sin archivos originales.
- Activación confiable del menú de reprocesamiento al subir archivos ya procesados.
- Conteos erróneos de documentos (índices inflados) en la UI y normalización del índice del caso.
- Prevención de la creación de la base de datos innecesaria `data/fraud_scorer.db`.
- Reutilización de análisis de fraude en reprocesos parciales sin romper plantillas.
- Limpieza completa de casos (deep purge y purgas parciales) sin residuos en el filesystem.
- Variables sensibles (.env): las claves de servicios externos (por ejemplo `OPENAI_API_KEY`, `AZURE_OCR_KEY`) se almacenan en la raíz del proyecto en `.env`. Cualquier instalación o reparación debe verificar este archivo antes de ejecutar scripts que dependan de esos proveedores.
- Gobernanza LLM GPS: antes de habilitar `GPS_LLM_ENABLED`, revisa las salvaguardas descritas en `guides_md/GPS_LLM_GOVERNANCE.md` (umbrales de costo, registro de prompts y auditoría por caso).
- Motor de correlación inter-documental (CaseContext + RuleEngine) y generación de reportes consolidados.
- Actualización del pipeline de extracción/consolidación IA (septiembre 2025).

Se incluyen causas raíz, decisiones de diseño y fragmentos de código exactos con rutas de archivo reales, alineados con la estructura actual del proyecto.

---

## 3) Motor de extracción/consolidación para carátula

Esta sección documenta el pipeline de IA que alimenta la carátula. Cualquier mejora futura (Rick, correlación, fraude) debe respetar estas reglas para no romper los campos consolidados.

### 3.1 Modelos y sampling

- Extracción OCR (`ocr_text`): `gpt-4o-mini`
- Extracción multimodal (`direct_ai`): `gpt-4o`
- Consolidación estándar y generación: `gpt-4o-mini`
- Consolidación escalada (`complexity=high`): `gpt-4o`

Los overrides se definen en `MODEL_SAMPLING_CONFIG`; `_chat_with_retry` sólo envía `temperature`/`top_p` cuando el modelo lo permite.

### 3.2 Guías de extracción

- **Informe final del ajustador**: recuperar folio de 14 dígitos, monto total (“RECLAMACIÓN Y AJUSTE”), fecha y lugar específicos (carretera/km/municipio), tipo de siniestro mapeado al catálogo y bien sin cantidades.
- **Denuncia**: leer la narrativa para fechas/lugares (“pasamos por el kilómetro… entronque…”), ignorando el encabezado.
- **Campo `lugar_hechos`**: guía específica exige ubicaciones detalladas; `bien_reclamado` evita números; `tipo_siniestro` enlaza a la taxonomía oficial.
- El preprocesamiento de OCR conserva hasta 6 000 caracteres y guarda 8 000 caracteres en metadata (`raw_text_snippet`) para auditoría o fallbacks.

### 3.3 Priorización y conflictos

- `FIELD_SOURCE_RULES` y `DOCUMENT_PRIORITIES` dan prioridad máxima a los informes del ajustador.
- `_group_by_field` prefiltra opciones cuando existe `informe_final_del_ajustador` para `monto_reclamacion`, `fecha_ocurrencia` y `lugar_hechos`; el LLM sólo ve la entrada prioritaria.
- El prompt de consolidación recuerda explícitamente: “si hay informe final, debes usarlo”.
- `_enforce_field_consistency` valida folios, normaliza `tipo_siniestro` y elimina fechas inconsistentes.
- Fallbacks determinísticos (regex) siguen siendo la red de seguridad.

### 3.4 Principios a mantener

1. **No romper la carátula**: después de cualquier cambio, reproduce `CASE-2025-0001` y verifica los campos críticos.
2. **Actualizar mapeos y guías** si se agregan nuevos documentos o campos.
3. **Prompts antes que heurísticas**: refuerza la extracción antes de aumentar fallbacks. Las guías deben priorizar modelos GPT para cada dato; expresiones regulares se reservan como red de seguridad cuando la IA no logra poblar un campo aun después de reforzar el prompt.
4. **No eliminar `raw_text_snippet`**: se usa para debugging y para motores secundarios.
5. **Instrumenta métricas**: vigila cuántos campos provienen de consolidación vs. fallback para detectar regresiones.

---

## ⬆️ Nueva actualización 2025: Extracción y Consolidación IA

El cambio de la familia GPT‑5 a **GPT‑4o-mini** resolvió los problemas de carátula y consolidado inconsistentes. Los puntos clave que debes recordar (ver detalles en [guides_md/AI_FIELD_EXTRACTION.md](guides_md/AI_FIELD_EXTRACTION.md)) son:

- **Modelos y sampling**: `get_model_for_task` ahora usa `gpt-4o-mini` para extracción, consolidación estándar y generación; se reserva `gpt-4o` para visión y consolidaciones escaladas (`src/fraud_scorer/settings.py:885-918`). `MODEL_SAMPLING_CONFIG` deja `temperature/top_p=None` para estos modelos, evitando errores 400 (`src/fraud_scorer/processors/ai/ai_field_extractor.py:563`).
- **Prompts reforzados**: el bloque de `informe_final_del_ajustador` instruye al LLM a recuperar folio de 14 dígitos, monto total, fecha y ubicación con carretera/km, tipo y bien sin cantidades (`src/fraud_scorer/prompts/extraction_prompts.py:420`). Las denuncias ahora recalcan que hay que leer la narrativa (“pasamos por… kilómetro… entronque…”) en lugar del encabezado (`src/fraud_scorer/prompts/extraction_prompts.py:432`).
- **Priorización por campo**: `_group_by_field` prefiltra opciones cuando existe un valor del informe final para `monto_reclamacion`, `fecha_ocurrencia` o `lugar_hechos`, de modo que el LLM ya no puede preferir denuncias o cartas por “contexto” (`src/fraud_scorer/processors/ai/ai_consolidator.py:524`).
- **Normalizaciones**: `_enforce_field_consistency` expulsa folios inválidos y mapea variantes de `tipo_siniestro` a la categoría oficial (`src/fraud_scorer/processors/ai/ai_consolidator.py:734`).
- **Reglas de fuente**: `FIELD_SOURCE_RULES` y `DOCUMENT_PRIORITIES` ubican al informe final y preliminar del ajustador como máxima prioridad para los campos críticos (`src/fraud_scorer/settings.py:548`, `src/fraud_scorer/settings.py:732`).

> ✅ Resultado: en el caso de QA `CASE-2025-0001`, `numero_siniestro`, `monto_reclamacion`, `fecha_ocurrencia` y `lugar_hechos` ahora se poblan con los valores esperados; el consolidador marca prioridad correcta y la carátula reporta “Carretera Matehuala, San Luis Potosí, kilómetro 57”.

---

## 1) Reproceso exclusivo de la Fase 3.5

### Síntoma

Al intentar reprocesar solo la Fase 3.5 (sin re‑OCR ni re‑extracción), el proceso fallaba con:

```
RuntimeError: No se encontraron documentos para procesar
```

Esto sucedía porque `process_case` exigía construir `documents` desde archivos originales (PDF/DOCX/XLSX/…), y en el nueva arquitectura los originales no deben existir en `data/ocr_cache` (solo los JSON `ocr_results_for_*.json`).

### Causa raíz

- `scripts/run_report.py::process_case` abortaba si `documents` quedaba vacío, incluso en reproceso “solo 3.5”.
- La Fase 1 dependía de `documents` para cargar `ocr_results`, a pesar de que el reproceso 3.5 debe operar únicamente con los JSON reorganizados.

### Solución

- Permitir un **modo “docsless_fraud_only”** cuando `reprocess_mode=True`, `reprocess_fraud=True`, y NO se pide `reprocess_ocr` ni `reprocess_extraction`, incluso si `documents` está vacío.
- Construir `ocr_results` **directamente desde los JSON reorganizados** usando como lista maestra las **extracciones previas** (`extraction_results`) y el **case folder**. Si no existen extracciones previas, abortar con error claro (Opción A).

Archivo clave: `scripts/run_report.py`

Fragmentos relevantes:

```python
# (1) Permitir continuar sin documentos si es SOLO 3.5
if not documents:
    allow_docsless_fraud = False
    try:
        opts = dict(reprocess_options or {})
        allow_docsless_fraud = bool(
            reprocess_mode
            and opts.get("reprocess_fraud")
            and not opts.get("reprocess_ocr")
            and not opts.get("reprocess_extraction")
        )
    except Exception:
        allow_docsless_fraud = False

    if not allow_docsless_fraud:
        raise RuntimeError("No se encontraron documentos para procesar")
    else:
        logger.info("ℹ️ Reproceso 3.5: sin archivos originales; se usará cache JSON reorganizado")
```

```python
# (2) Fase 1: construir ocr_results desde JSON reorganizado
docsless_fraud_only = (
    self.reprocess_mode
    and wants("reprocess_fraud")
    and not wants("reprocess_ocr")
    and not wants("reprocess_extraction")
    and not documents
)

if docsless_fraud_only:
    prev_extractions = case_data.get("extraction_results") or []
    if not prev_extractions:
        raise RuntimeError("No existen extracciones previas; seleccione re-extracción o ejecute Fase 2 antes")

    case_folder = base_folder  # carpeta "ASEGURADO - RECLAMO"
    built = 0
    for item in prev_extractions:
        name = item.get("source_document") if isinstance(item, dict) else getattr(item, "source_document", None)
        doc_type = item.get("document_type") if isinstance(item, dict) else getattr(item, "document_type", None)
        if not name:
            continue
        doc_folder = self.cache_manager._sanitize_filename(Path(name).stem)
        doc_path = case_folder / doc_folder / name
        ocr_result = self.cache_manager.get_cache(doc_path, case_id)
        if not ocr_result:
            logger.warning(f"No se encontró OCR JSON para {name}; se omite en 3.5")
            continue
        ocr_results.append({
            "filename": name,
            "ocr_result": ocr_result,
            "document_type": doc_type or None,
        })
        built += 1
    if built == 0:
        raise RuntimeError("No se pudo construir OCR desde cache JSON; verifique ocr_results_for_*.json en la carpeta de caso")
```

Decisión explícita: si no hay `extraction_results`, se aborta con mensaje claro (no se intenta una re‑extracción implícita).

### Extensión (septiembre 2025): Reprocesos selectivos 1.4–3 sin originales

Después de estabilizar el flujo “solo 3.5”, observamos el mismo bloqueo al reprocesar clasificación, detección de póliza, extracción o consolidación cuando los PDFs ya no estaban en `data/ocr_cache`. Para resolverlo se generalizó el modo *docless*:

- **Cargador universal de OCR**: `FraudAnalysisSystemV2._prepare_docless_ocr` reconstruye `ocr_results`, `cache_files` y nombres originales usando los JSON reorganizados. Utiliza clasificaciones previas, overrides manuales y extracciones para preservar el `document_type` (scripts/run_report.py:289-437).
- **Derivación del nombre humano**: `_extract_original_filename` lee `metadata.file_name` o interpreta `ocr_results_for_<ORIGINAL>.json` para conservar el nombre mostrado en UI y reportes (scripts/run_report.py:268-287).
- **Guardias flexibles**: si se recibe un `existing_case_id`, no se solicita re-OCR y se selecciona cualquier fase >=1.4, el pipeline permite continuar sin PDFs (scripts/run_report.py:563-588). Se aprovechan los paths del `case_index` como manifiesto principal.
- **Conteos coherentes**: `documents_processed`, `total_documents` y `case_data['documents']` se basan ahora en los JSON cuando los originales no existen (scripts/run_report.py:1011-1024, 1708-1777).

Con esto, cualquier combinación de fases 1.4–3.5 puede ejecutarse reutilizando únicamente el cache reorganizado.

### Sub-bullet (noviembre 2025): Extracción y análisis de carpetas con GPT-5

- **Motivación**: Las narrativas de carpeta de investigación requerían inferir rutas completas, horas del evento/liberación y destino final (Aceros Ocotlán) que GPT‑4o‑mini omitía.
- **Acción**: Forzamos `AIFieldExtractor.extract_from_document` a usar `gpt-5` cuando `document_type == "carpeta_de_investigacion"` y deshabilitamos temperatura/top_p para esa familia (ver `src/fraud_scorer/processors/ai/ai_field_extractor.py:209-223`, `1080-1105`).
- **Consistencia**: `FraudAnalyzer` ya respetaba esa guía, por lo que las narrativas consolidan horarios, lugar de abandono y pesos reportados. Mantén esta configuración para cualquier refactor futuro; si cambias de modelo, actualiza la sección anterior.

### Actualización (2025-09-21): Reproceso sin originales para Fases 1.4–3

**Problema detectado**: al extender los reprocesos selectivos a clasificación (1.4), detección de póliza (1.5), extracción (2) o consolidación (3), el pipeline seguía exigiendo PDFs originales. Tras moverlos fuera de `data/ocr_cache`, cualquier intento de reprocesar estas fases terminaba con `RuntimeError: No se encontraron documentos para procesar`.

**Solución**:

- Generalizar el modo “docless” para cualquier combinación que no requiera re‑OCR y parta de un `case_id` existente. Se reusa el índice de caso y los JSON reorganizados como fuente primaria.
- Incorporar en `FraudAnalysisSystemV2` dos auxiliares:
  - `_prepare_docless_ocr`: reconstruye `ocr_results`, rutas y metadatos a partir de los JSON (`scripts/run_report.py`). Usa clasificaciones, extracciones y overrides previos para mantener tipos de documento coherentes.
  - `_extract_original_filename`: deriva el nombre humano desde `metadata.file_name` del JSON o, en su defecto, desde el patrón `ocr_results_for_<ORIGINAL>.json`.
- Ajustar el guard de entrada (`process_case`) para permitir reprocesos sin originales cuando se recibe un `existing_case_id` y se solicite al menos una fase posterior a OCR (`reprocess_classification` / `reprocess_policy_detection` / `reprocess_extraction` / `reprocess_consolidation` / `reprocess_fraud`).
- Normalizar métricas y conteos (`documents_processed`, `total_documents`) usando la lista proveniente de los JSON, de modo que los dashboards sigan mostrando el total real de documentos.

Resultado: cualquier reproceso de fases 1.4–3.5 funciona aun cuando los PDFs ya no estén en disco; la app reutiliza el cache reorganizado y mantiene la nomenclatura original de los documentos.

---

## 2) Activación confiable del menú de reprocesamiento (detección por hash)

### Síntoma

Al volver a subir los mismos documentos, el sistema no activaba el menú de replay y pasaba a flujo “normal”, aún cuando internamente detectaba duplicados por hash.

### Causa raíz

- La UI decide si mostrar el menú de reprocesado consultando `POST /api/case/null/check-existing`.
- El endpoint hacía fallback por nombre (`case_index['documents']`), pero desde que `documents` apunta a JSONs “vista humana”, los nombres eran **sanitizados** (`_` en vez de espacios, etc.) y dejaban de intersectar con los nombres de archivo “humanos” subidos.
- Aun cuando el pipeline detectaba duplicados por hash más tarde, la decisión del menú ya había pasado.

### Solución

- Implementar **detección por hash SHA‑256** en el endpoint de `check-existing`, con soporte en el frontend para calcular y enviar los hashes de hasta 10 archivos (o todos si son menos de 10).
- Mantener un fallback por nombre (clasificaciones previas o deducción desde JSON) para compatibilidad.

Archivos:

- Backend: `src/fraud_scorer/api/web_interface.py`

```python
@app.post("/api/case/{case_id}/check-existing")
async def check_existing_case(case_id: str = None, payload: dict = Body(...)):
    # ...
    hashes: list[str] = []
    # extrae payload['hashes'] si existe
    if hashes:
        with get_conn() as conn:
            qmarks = ",".join(["?"] * len(hashes))
            sql = f"""
                SELECT case_id, COUNT(*) as cnt
                FROM documents
                WHERE file_hash IN ({qmarks})
                GROUP BY case_id
                ORDER BY cnt DESC
                LIMIT 1
            """
            row = conn.execute(sql, tuple(hashes)).fetchone()
            if row and row["case_id"]:
                candidate_id = row["case_id"]
                case_index = OCRCacheManager().get_case_index(candidate_id, auto_reconstruct=True)
                return {"existing_case": True, "case_id": candidate_id, ...}
    # fallback por nombre: clasificaciones → documentos(JSON)
```

- Frontend: `src/fraud_scorer/api/templates/upload.html`

```javascript
async function computeSHA256(file) {
  const buf = await file.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', buf);
  const view = new DataView(digest);
  let hex = '';
  for (let i = 0; i < view.byteLength; i++) {
    hex += view.getUint8(i).toString(16).padStart(2, '0');
  }
  return hex;
}

async function checkExistingCase() {
  const fileNames = selectedFiles.map(f => f.name);
  const limit = Math.min(10, selectedFiles.length || 0);
  const filesToHash = selectedFiles.slice(0, limit);
  let hashes = await Promise.all(filesToHash.map(computeSHA256));
  const response = await fetch('/api/case/null/check-existing', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ files: fileNames, hashes })
  });
  const result = await response.json();
  if (result.existing_case) { showReprocessOptions(result); return true; }
  return false;
}
```

Resultado: la detección de siniestros existentes es **determinista por hash**, y el menú de reprocesado se muestra siempre que los documentos pertenezcan a un caso ya procesado.

---

## 3) Conteos erróneos en la UI (índices inflados → 21 vs 7)

### Síntoma

La UI mostraba 21 documentos cuando en realidad el siniestro tiene 7.

### Causa raíz

- El índice del caso (`data/ocr_cache/case_index/CASE-2025-0001.json`) fue acumulando entradas en `documents` y `cache_files` a través de sesiones previas:
  - Mezcla de rutas temporales (`data/temp/<uuid>/...`) y rutas reorganizadas.
  - A veces PDFs “virtuales” (nunca copiados) y posteriormente JSONs “vista humana”.
  - Estrategia de “unión” durante guardados incrementales (`existing ∪ new`).
- La función `reorganize_cache_for_case` solo normalizaba cuando corría Fase 3 (no en “solo 3.5”).

### Solución

- Normalizar **siempre** el índice hacia JSONs únicos en la reorganización, con manejo explícito de JSONs ya reorganizados.
- Alinear `cache_files` con `documents` y filtrar `document_hashes` a claves presentes.
- Ajustar `total_documents` al recuento real.

Archivo: `src/fraud_scorer/storage/ocr_cache.py`

Soporte JSON directo en cache:

```python
def get_cache(self, document_path: Path, case_id: str = None):
    # JSON reorganizado directo
    if document_path.suffix.lower() == ".json" and document_path.name.startswith("ocr_results_for_"):
        with open(document_path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    # ... resto de la búsqueda

def has_cache(self, document_path: Path, case_id: Optional[str] = None) -> bool:
    if document_path.suffix.lower() == ".json" and document_path.name.startswith("ocr_results_for_"):
        return document_path.exists()
    # ... resto de la verificación
```

Manejo correcto de JSONs reorganizados en la **reorganización**:

```python
# Si el índice trae ya JSONs reorganizados (ocr_results_for_*.json), tratarlos como finales
if original_doc_path.suffix.lower() == ".json" and original_doc_path.name.startswith("ocr_results_for_"):
    base_part = original_doc_path.name[len("ocr_results_for_"):-len(".json")]
    doc_folder_name = self._sanitize_filename(Path(base_part).stem)
    doc_specific_path = new_case_path / doc_folder_name
    destination_path = doc_specific_path / original_doc_path.name
    if original_doc_path != destination_path:
        shutil.move(str(original_doc_path), str(destination_path))
    new_document_paths.append(str(destination_path))
    updated_hashes.pop(old_path_str, None)
    updated_hashes[str(destination_path)] = doc_hash
    continue  # no procesar como si fuera original
```

De‑duplicado y alineación final del índice:

```python
if new_document_paths:
    unique_docs = []
    seen = set()
    for path_str in new_document_paths:
        if path_str not in seen:
            unique_docs.append(path_str); seen.add(path_str)
    case_data['documents'] = unique_docs
    case_data['cache_files'] = list(unique_docs)
    # Filtrar document_hashes
    doc_hashes = case_data.get('document_hashes') or {}
    if isinstance(doc_hashes, dict):
        case_data['document_hashes'] = {k: v for k, v in doc_hashes.items() if k in set(unique_docs)}
    case_data['total_documents'] = len(unique_docs)
```

Resultado: el índice del caso refleja exactamente los **7 JSONs únicos** del siniestro; la UI deja de mostrar 21.

> Nota: En pantallas que muestren `db_documents` (conteo global en la BD), ese número puede incluir **historial** de corridas previas; es independiente del índice “vista humana”.

---

## 4) Prevención de la base de datos `data/fraud_scorer.db`

### Síntoma

Se creó un archivo `data/fraud_scorer.db` que no debe existir.

### Causa raíz

- Alguna configuración/entorno estableció `FRAUD_DB_PATH` hacia `data/fraud_scorer.db` (legacy o residual).

### Solución

- **Guard** en el módulo de almacenamiento para forzar la ruta canónica `data/cases.db` si se intenta usar `fraud_scorer.db`.

Archivo: `src/fraud_scorer/storage/db.py`

```python
DB_PATH = Path(os.getenv("FRAUD_DB_PATH", "data/cases.db"))
if DB_PATH.name.lower() == "fraud_scorer.db":
    DB_PATH = Path("data/cases.db")
```

Resultado: no volverá a crearse `data/fraud_scorer.db` desde este sistema.

---

## 5) Toggles del editor bloquean la UI (WeasyPrint en cada PATCH)

### Síntoma

En el editor del analista, alternar la visibilidad de un documento congelaba la interfaz durante varios segundos e incluso abortaba el proceso (`pointer being freed was not allocated`). Los logs mostraban múltiples inicializaciones de `AIReportGenerator` y advertencias repetidas de WeasyPrint; cada toggle disparaba la generación completa del PDF.

### Causa raíz

- El endpoint `PATCH /api/editor/{case_id}/fraud-documents/{document_id}` reutilizaba `FraudDocumentReprocessService._regenerate_report` con `generate_pdf=True`.
- WeasyPrint abre fuentes WOFF2 y recalcula el layout completo. Esta operación es costosa y no está pensada para ejecutarse decenas de veces por sesión interactiva.

### Solución

- Exponer `FraudDocumentReprocessService.refresh_report()` para regenerar **solo el HTML** del informe (misma plantilla de producción) tras cada toggle y conservar el PDF anterior.
- Ajustar `_regenerate_report(..., generate_pdf: bool = True)` para que únicamente los reprocesos “grandes” (Fase 3.5 o full replay) generen PDF nuevo.
- El frontend mantiene el cache busting del iframe (`reportFrame.src = ...?t=<timestamp>`), por lo que la “Vista final” se actualiza al instante sin bloquear al usuario.
- Las ediciones manuales completas del informe se almacenan como `report_override_html` y se aplican al regenerar el archivo; los toggles respetan esa versión sin invocar WeasyPrint, y el PDF sólo se actualiza al ejecutar un reproceso explícito.

Archivos clave: `src/fraud_scorer/services/fraud_document_service.py`, `src/fraud_scorer/api/web_interface.py`, `static/js/editor_analista.js`.

Resultado: los toggles son inmediatos, el iframe refleja la visibilidad actual y el servidor deja de saturarse con conversiones PDF innecesarias. Si se requiere un PDF actualizado, se obtiene automáticamente al ejecutar un reproceso individual o global que sí pasa `generate_pdf=True`.

---

## 6) Reutilización de análisis de fraude en reprocesos parciales

### Síntoma

Al reprocesar fases 1.4–3 sin marcar `reprocess_fraud`, el reporte final dejaba de mostrar la sección de fraude pese a existir resultados previos en el índice.

### Causa raíz

- `case_index['fraud_analyses']` guarda los resultados de 3.5 como diccionarios (`model_dump`).
- Durante el reproceso, el pipeline asignaba esa lista directamente a `fraud_analyses`, pero el generador de reportes espera instancias `FraudAnalysisResult` (accede a propiedades como `.document_type`, `.risk_level.value`).
- Resultado: la plantilla lanzaba `AttributeError` y se caía al reporte estándar sin fraude.

### Solución

- Se agregó `_hydrate_fraud_results` para validar cada entrada contra `FraudAnalysisResult` antes de reutilizarla (scripts/run_report.py:439-462).
- Fase 3.5 ahora invoca el helper al cargar `previous_fraud` y únicamente ejecuta el análisis si se solicita `reprocess_fraud` o no existen resultados previos (scripts/run_report.py:1494-1514).
- Cuando se reutiliza, la sección de fraude vuelve a renderizarse correctamente y los modelos pueden convivir con dicts legacy sin romper el pipeline.

Resultado: cualquier reproceso parcial conserva la sección de fraude previa, eliminando la necesidad de re-analizar documentos si no es requerido.

---

## 6) Limpieza completa de casos (residuos en FS después del purge)

### Síntoma

Tras ejecutar `DELETE /replay/api/deep-purge/<case_id>` la base de datos quedaba limpia, pero persistían archivos relacionados al caso:

- Respaldos de índice `case_index/CASE-*.json.backup_*`.
- Reportes HTML/PDF/JSON cuyo nombre no coincidía exactamente con el `insured_name` guardado (variaciones con acentos o reemplazos de caracteres).

Esto provocaba que un “deep purge” pareciera exitoso, pero al inspeccionar `data/ocr_cache/case_index/` o `data/reports/` seguían presentes artefactos con `case_id` y `claim_number` del siniestro.

### Causa raíz

- `ReplayService.clear_cache()` únicamente eliminaba el archivo índice principal y reportes que coincidieran con patrones simples (`case_id`, `INF-{case_id}` o el nombre sanitizado literal). No consideraba respaldos automáticos (`*.backup_*`) ni variantes de nombres con tildes/guiones.
- No existía una búsqueda basada en contenido del reporte (JSON) para detectar el `case_id` cuando el nombre del archivo era diferente al asegurado registrado en el índice.

### Solución

- Incorporar auxiliares dedicados en `src/fraud_scorer/services/replay_service.py`:
  - `_collect_name_variations()` genera combinaciones (original, sanitizada, sin acentos, sin espacios, minúsculas) para cubrir discrepancias entre nomenclaturas.
  - `_remove_report_family()` elimina el trío `json/html/pdf` sin importar el sufijo (`_RESULTADOS`, `_INFORME`, `_REPORTE`, `_FINAL`).
  - `_remove_reports_by_case_id()` abre cada JSON en `data/reports`, busca el `case_id` y elimina toda la familia aunque el nombre no coincida con el asegurado.
- Durante el purge:
  - Conservar `insured_name`/`claim_number`, inferirlos desde el nombre de la carpeta (`<ASEGURADO> - <SINIESTRO>`) si el índice ya no está disponible.
  - Eliminar respaldos `case_index/{case_id}.json.backup_*` además del índice principal.
 - Para el pipeline cache (`data/temp/pipeline_cache`) usar patrones basados en `case_id` y `claim_number`.

### Verificación posterior al procesamiento

- Después de consolidar un caso se ejecuta `verify_case_artifacts(case_id)`:
  - Confirma que `cases`, `documents`, `ocr_results`, `extracted_data`, `fraud_analyses`, `runs` y `ai_analyses` tengan registros para el caso.
  - Valida que exista `case_index/<case_id>.json` y que las rutas reorganizadas sigan disponibles.
  - Localiza duplicados por `file_hash` y elimina automáticamente los registros sobrantes (con cascada en OCR/extracción).
  - Reporta advertencias si faltan shards hash o si el índice y la BD se desalinean.
- Si detecta un problema crítico (caso inexistente, sin documentos o sin índice), el pipeline levanta una excepción para revisar el proceso antes de continuar.

Archivo clave: `src/fraud_scorer/services/replay_service.py`

### Sistema de borrado completo (checklist)

Al ejecutar `DELETE /replay/api/deep-purge/<case_id>` o la acción equivalente desde la UI, se debe verificar que:

1. **BD**: se elimina la fila en `cases` y, por cascada, `documents`, `ocr_results`, `extracted_data`, `fraud_analyses`, `runs` y métricas (`reset_cache_stats(case_id)`).
2. **Índices**: se eliminan `case_index/{case_id}.json` y cualquier `*.backup_*` asociado.
3. **Carpetas reorganizadas**: `data/ocr_cache/<ASEGURADO - SINIESTRO>/` desaparece (incluye subcarpetas y shards).
4. **Artefactos GPS**: `data/gps/<case_id>/` se elimina con todos los Parquet, manifest y `raw_text.txt`. Si existen variaciones sanitizadas del `case_id`, también deben desaparecer.
5. **Archivos temporales**: se borran carpetas bajo `data/temp/...` que contengan el `case_id` o el siniestro (incluye `pipeline_cache` y staging temporales).
6. **Índices vectoriales**: `data/chroma/<case_id>*` y snapshots asociados desaparecen; en `clear_cache(['all'])` se recrea la estructura vacía.
7. **Auditoría Rick**: las entradas del caso se purgan de `data/logs/agent_rick_audit.jsonl` y el archivo se recrea vacío (se toca el fichero tras el purge para que futuros eventos no fallen).
8. **Reportes**: para cada coincidencia de `case_id`, `claim_number` o variantes del asegurado, se eliminan los archivos `HTML/PDF/JSON` de `data/reports/`.
9. **Residuos adicionales**: si la limpieza se ejecuta mediante `clear_cache(['all'])`, también se vacía `case_index/` por completo antes de reconstruir la estructura.

Con estas garantías, volver a subir los mismos documentos hará que `/api/case/null/check-existing` responda `existing_case: false` y el pipeline procese el caso desde cero sin depender de artefactos previos.

#### Actualización 2026: tokens defensivos y verificación cruzada

Para que la eliminación resista nuevas integraciones (Agente Rick, correlaciones, consolidadores personalizados), todas las rutas de borrado (`clear_cache`, `purge_case`, `deep_purge_case`) generan un conjunto de **tokens sanitizados** a partir de `case_id`, número de siniestro y nombre del asegurado. Estos tokens se usan para:

- Escanear `data/reports`, `data/temp/pipeline_cache`, `data/gps/`, `data/chroma/` y `data/logs/agent_rick_audit.jsonl` buscando coincidencias parciales en nombres de archivo o contenido JSON; cualquier match deriva en borrado del artefacto completo (familia HTML/PDF/JSON, carpetas de índice, datasets GPS, líneas de auditoría).
- Eliminar shards y carpetas reorganizadas incluso si el índice del caso ya no existe (p. ej. se detecta `GRUPO_ACEROS_OCOTLAN_SA_DE_CV - 20240000001361` solo a partir de los tokens).
- Reconstruir la tabla `fraud_correlations` incorporando `ON DELETE CASCADE` de forma idempotente para evitar correlaciones huérfanas cuando se eliminan casos desde UI.

Adicionalmente, tras cada purge se ejecuta una **verificación cruzada**:

- Se consulta la base de datos para confirmar que `cases`, `documents`, `ocr_results`, `extracted_data`, `fraud_analyses`, `fraud_correlations`, `runs` y `cache_stats` no conserven filas del caso.
- Se recorren nuevamente los directorios críticos; si aún quedan coincidencias por token (por permisos u open handles), se registran en log y se intenta una segunda eliminación. El proceso no concluye en “éxito” sin un FS limpio.
- Se limpian shards hash vacíos bajo `data/ocr_cache/<shard>` para evitar que queden carpetas huérfanas después del purge.

**Recomendaciones operativas**

- Los tests o scripts manuales que simulan un purge deben validar tanto la BD como el FS; si se encuentra cualquier residuo, actualizar los tokens de limpieza o añadir el directorio al barrido defensivo.
- Si se añade un nuevo subsistema que persista artefactos por caso (logs, índices, caches), debe exponer un helper tipo `_remove_<feature>_artifacts(tokens)` y registrarse en el flujo de purge con verificación posterior.
- Mantener `PYTHONPYCACHEPREFIX` o limpiar `__pycache__` tras compilar/validar para que los purges no dejen residuos inesperados en la carpeta de trabajo.

---

## 7) Estado actual del almacenamiento y caché (2025)

### Directorios canónicos

- `data/ocr_cache` y `data/ocr_cache/case_index`: única fuente para resultados OCR y metadatos por hash/caso. Todo nuevo desarrollo debe consumirlos a través de `OCRCacheManager` (`src/fraud_scorer/storage/ocr_cache.py`).
- `data/temp/pipeline_cache`: staging oficial para consolidado y marcadores de pipeline (`*_CONSOLIDADO.json`, `*.status.jsonl`).
- `data/uploads` (`/renombre_de_documentos`): staging de organización/clasificación en dos fases (`document_organizer.py`).
- `data/reports`: carpeta de salida estable tanto para CLI como API (`scripts/run_report.py`, endpoints `reports`).

### Directorios retirados o solo temporales

- `data/raw`, `data/temp_reports`, `data/feedback_archive`: se eliminaron junto con el endpoint legacy `/documents/upload` y scripts asociados. No volver a crearlos salvo que exista un productor claro; ajustamos limpieza/test para ignorarlos.
- `data/temp`: sigue siendo staging general para uploads/API; cualquier limpieza debe conservar la carpeta base pero purgar contenido dinámico.
- `data/training_examples`: contiene ejemplos utilizados por prompts; mantener únicamente archivos necesarios para los modelos.

### Base de datos y métricas

- `data/cases.db` permanece como origen de verdad: `cases`, `documents`, `ocr_results`, `extracted_data`, `fraud_analyses`, `ai_analyses`, `runs`, `cache_stats` (`src/fraud_scorer/storage/db.py`).
- Persistimos hashing por archivo y reuso global; los índices únicos (`idx_docs_case_hash`) previenen duplicados cuando el pipeline sube correctamente los hashes.
- La UI/replay consulta `cache_stats` combinando métricas de BD y conteos en FS; cualquier nuevo contador debe agregarse aquí para mantener coherencia.

### Limpieza y scripts auxiliares

- `scripts/clean_orphaned_files.py` y `ReplayService.clear_cache` solo tocan directorios vigentes (`reports`, `temp`, `uploads`, `ocr_cache`). Si se añade una ruta nueva, debe registrarse explícitamente.
- `scripts/test_system.py` verifica que `data/uploads`, `data/reports`, `data/temp` existan. Mantener esta lista en sincronía con la arquitectura.
- Los scripts de purge no deben reintroducir rutas retiradas; de hacerlo, actualizar primero la documentación y el verificador post-proceso.

### Buenas prácticas de ampliación

- Centralizar rutas nuevas en `src/fraud_scorer/settings.py` para evitar hardcodes dispersos.
- Antes de añadir cualquier carpeta, definir quién la pobla y qué script la limpia. Si no hay productor + limpiador, no crearla.
- Preferir persistencia y deduplicación vía BD. Si se necesita caché adicional, extender `OCRCacheManager` o `post_process_verifier` en lugar de crear carpetas huérfanas.
- Documentar cualquier cambio en la jerarquía en esta sección y añadir el nodo a los chequeos automáticos (tests, clean scripts, verificador).

## Consideraciones finales

- Reproceso 3.5: ahora funciona sin archivos originales, consumiendo únicamente JSONs reorganizados, siempre que existan extracciones previas (carátula previa incluida).
- Replay: la detección por hash en `check-existing` garantiza la activación del menú de reprocesado con los mismos archivos, evitando dependencias frágiles por nombre/sanitización.
- Índice de caso: queda normalizado a JSONs únicos, con conteos consistentes (`total_documents`) y sin duplicados entre sesiones.
- BD canónica: se fuerza `data/cases.db` para evitar la creación de bases no deseadas.

Si deseas, podemos reforzar `get_cache_stats()` para contar únicamente `ocr_results_for_*.json` (excluyendo shards) y mostrar explícitamente en la UI “documentos (FS)” vs “documentos (BD)” para mayor claridad operacional.

---

## Buenas prácticas operativas

- Cache OCR “vista humana” (FS):
  - Mantener `FS_COPY_DOCS_IN_CACHE=false` (por defecto). Nunca copiar originales (PDF/DOCX/XLSX/…) a `data/ocr_cache`.
  - Conservar únicamente JSONs `ocr_results_for_*.json` bajo `data/ocr_cache/<ASEGURADO - RECLAMO>/<doc_folder>/`.
  - Si se detectan originales en esa estructura, es seguro eliminarlos (el sistema ya los limpia si la variable está en `false`).

- Reprocesar solo 3.5 (fraude):
  - Requiere que el índice del caso tenga `extraction_results` previos (se genera en la Fase 2). Si no existe, lanzar reproceso con `reprocess_extraction=true` primero o ejecutar Fase 2 una vez.
  - En reproceso 3.5, no se re‑OCR ni se re‑extrae; la carátula proviene de `consolidated_data` y los OCR del **cache JSON**.

- Detección de siniestros existentes:
  - En el upload, calcular y enviar hashes SHA‑256 de hasta 10 archivos para detección determinista por DB; conservar nombres de archivos sin alterar.
  - Si no es posible calcular hashes en el cliente, la API caerá al fallback por nombre (menos robusto).

- Índices del caso (`case_index/*.json`):
  - No editar manualmente salvo para limpieza controlada. Tras consolidación o limpieza, ejecutar la reorganización (implícita desde el pipeline) para normalizar.
  - Si se realizaron ediciones manuales, verificar que:
    - `documents` apunte solo a JSONs únicos `ocr_results_for_*.json`.
    - `cache_files` esté alineado con `documents`.
    - `document_hashes` contenga claves que existan en `documents`.

- Base de datos:
  - Asegurar que `FRAUD_DB_PATH` apunte a `data/cases.db`. Evitar `data/fraud_scorer.db`; el sistema ya redirige, pero conviene no configurarlo así.

- Estadísticas y conteos:
  - Para conteos por caso, preferir `total_documents` del índice (ya normalizado) sobre conteos globales en DB.
  - Para “FS file count”, contar únicamente `ocr_results_for_*.json` en `data/ocr_cache` (excluyendo `case_index` y shards).

- Limpieza y mantenimiento:
  - `scripts/clean_orphaned_files.py --dry-run` para vista previa y `--all` para limpieza total.
  - `scripts/system_integrity_check.py` para diagnóstico cruzado BD ↔ FS (índices y reorganizados).

---

## Diagramas de flujo (alto nivel)

### A) Nuevo caso (por upload)

```
Usuario → UI Upload
  ├─ Calcula SHA‑256 (hasta 10) y envía /check-existing
  ├─ API check-existing: NO encuentra coincidencias por hash ni por nombre
  ├─ UI inicia procesamiento normal
  │    └─ run_report.process_case
  │        ├─ Descubre documentos (originales)
  │        ├─ Fase 1: OCR/Parser → guarda JSONs con save_cache
  │        ├─ Fase 1.4: Clasificación (LLM/heurística)
  │        ├─ Fase 2: Extracción (genera extraction_results)
  │        ├─ Fase 3: Consolidación (genera consolidated_data)
  │        ├─ Reorganiza cache a "vista humana" (JSONs únicos)
  │        └─ Guarda índice de caso (case_index/CASE-*.json)
  └─ UI muestra carátula, checkpoint, reporte
```

### B) Reprocesado por hash (upload de archivos de un caso existente)

```
Usuario → UI Upload
  ├─ Calcula SHA‑256 (hasta 10) y envía /check-existing
  ├─ API check-existing: ENCUENTRA case_id por hash(s)
  ├─ UI muestra menú "Reprocesar" (opciones: OCR, Clasificación, Extracción, Consolidación, Fraude)
  │    └─ Usuario selecciona SOLO Fase 3.5 (fraude)
  │         └─ run_report.process_case (reprocess_mode=true)
  │             ├─ NO requiere documentos originales
  │             ├─ Construye ocr_results desde JSONs reorganizados
  │             ├─ Reusa extraction_results (previos)
  │             ├─ Reusa consolidated_data (carátula previa)
  │             ├─ Fase 3.5: Análisis de fraude por documento
  │             └─ Genera reporte con carátula + sección de fraude actualizada
  └─ UI entrega reporte completo
```

Notas:
- Si el usuario elige otras fases (OCR, Extracción, Consolidación), el pipeline respetará las opciones y reaprovechará artefactos previos cuando corresponda.

---

## Ejemplos de logs y respuestas del endpoint

### A) Detección por hash exitosa (al subir archivos ya procesados)

Solicitud (UI → API):

```http
POST /api/case/null/check-existing
Content-Type: application/json

{
  "files": [
    "1 1 Carta reclamacion gastos HDI.pdf",
    "13 Carta Porte 16BC2T.pdf",
    "4 RATIFICACION IAD-comprimido.pdf"
  ],
  "hashes": [
    "c07fdc3f1ebfb69f099e1ae322a2cb9af2a5097e0b44aa0c839ec3b5680be44e",
    "43a393d08d86e7ef09c0d5fa61e86dc6c52253a5167ae525d21755fd85fa2967",
    "795c2cf7d2ee99d1f59bd2e4c6bf1bc5b6adb1d8fddfe3a459c1425d62fac1bf"
  ]
}
```

Respuesta (API → UI):

```json
{
  "existing_case": true,
  "case_id": "CASE-2025-0001",
  "case_info": {
    "case_id": "CASE-2025-0001",
    "case_title": "GRUPO ACEROS OCOTLAN SA DE CV - 20240000001361",
    "insured_name": "GRUPO ACEROS OCOTLAN SA DE CV",
    "claim_number": "20240000001361",
    "processed_at": "2025-09-16T20:25:16.162158",
    "folder_path": "data/ocr_cache/GRUPO_ACEROS_OCOTLAN_SA_DE_CV - 20240000001361",
    "total_documents": 7
  },
  "has_ocr": true,
  "has_classifications": true,
  "has_extraction": true,
  "has_consolidation": true,
  "has_fraud_analysis": true
}
```

UI: con `existing_case: true` se muestra el menú de **Reproceso**.

### B) Sin coincidencias (nuevo caso)

Solicitud igual que arriba, pero con hashes no presentes en DB.

Respuesta:

```json
{ "existing_case": false }
```

UI: inicia flujo de **nuevo caso**.

### C) Logs típicos durante reproceso 3.5 sin originales

```
INFO  fraud_scorer.run_report  ============================================================
INFO  fraud_scorer.run_report  📁 Procesando caso: GRUPO_ACEROS_OCOTLAN_SA_DE_CV - 20240000001361
INFO  fraud_scorer.run_report  🤖 Modo: IA Avanzada v2.0
INFO  fraud_scorer.run_report  ============================================================
INFO  fraud_scorer.run_report  🔍 DEPURACIÓN: Procesando con case_id=CASE-2025-0001
INFO  fraud_scorer.run_report    Carpeta: data/ocr_cache/GRUPO_ACEROS_OCOTLAN_SA_DE_CV - 20240000001361
INFO  fraud_scorer.run_report    Título: Reproceso_CASE-2025-0001
INFO  fraud_scorer.run_report    Modo reproceso: True
INFO  fraud_scorer.run_report    Opciones reproceso: {"reprocess_ocr": false, "reprocess_extraction": false, "reprocess_consolidation": false, "reprocess_fraud": true}
INFO  fraud_scorer.run_report  ℹ️ Reproceso 3.5: sin archivos originales; se usará cache JSON reorganizado
INFO  fraud_scorer.run_report  ✓ Documentos listos desde OCR reorganizado (JSON): 7
INFO  fraud_scorer.run_report  🔎 FASE 3.5: Análisis de fraude por documento
INFO  fraud_scorer.run_report  ✓ Análisis de fraude completado: 7 documentos analizados (elegibles: 7)
INFO  fraud_scorer.run_report  ✓ HTML generado: data/reports/INF-GRUPO_ACEROS_OCOTLAN_SA_DE_CV-20240000001361.html
```

### D) Logs durante reorganización con JSONs existentes

```
INFO  fraud_scorer.storage.ocr_cache  Reorganizando caché para el caso CASE-2025-0001...
INFO  fraud_scorer.storage.ocr_cache  Moviendo JSON data/ocr_cache/.../ocr_results_for_1_1_Carta_reclamacion_gastos_HDI.pdf.json -> data/ocr_cache/.../1_1_Carta_reclamacion_gastos_HDI/ocr_results_for_1_1_Carta_reclamacion_gastos_HDI.pdf.json
INFO  fraud_scorer.storage.ocr_cache  Moviendo JSON data/ocr_cache/.../ocr_results_for_13_Carta_Porte_16BC2T.pdf.json -> data/ocr_cache/.../13_Carta_Porte_16BC2T/ocr_results_for_13_Carta_Porte_16BC2T.pdf.json
...
INFO  fraud_scorer.storage.ocr_cache  Reorganización del caché completada para el caso CASE-2025-0001 en: data/ocr_cache/GRUPO_ACEROS_OCOTLAN_SA_DE_CV - 20240000001361
```

### E) Verificación DB rápida (opcional)

Consulta los hashes y case_id en SQLite:

```sql
SELECT case_id, filename, file_hash
FROM documents
WHERE case_id = 'CASE-2025-0001'
ORDER BY filename;
```

Comprobar un hash específico:

```sql
SELECT case_id, COUNT(*) cnt
FROM documents
WHERE file_hash IN (
  'c07fdc3f1ebfb69f099e1ae322a2cb9af2a5097e0b44aa0c839ec3b5680be44e',
  '43a393d08d86e7ef09c0d5fa61e86dc6c52253a5167ae525d21755fd85fa2967'
)
GROUP BY case_id
ORDER BY cnt DESC;
```

---

## 7) Checkpoint de Clasificación y Catálogo de Documentos

### Reglas clave al modificar clasificación manual o tipos de documentos

- **Sincronizar enum y catálogo UI**
  - Cada vez que se agregue un valor a `DocumentType` (`src/fraud_scorer/processors/document_classifier.py`), actualizar `_get_grouped_document_types()` en `src/fraud_scorer/api/web_interface.py`.
  - Tras el cambio, revisar los logs de arranque: `_audit_document_type_groups()` debe imprimir listas vacías para `missing_document_types` y `duplicate_assignments`.
  - Validar también `GET /api/system/document-type-audit`; usarlo como chequeo de smoke test en QA/CI.

- **Revisión manual con baseline consistente**
  - `update_classifications` debe reemplazar `case["ai_classifications"]` con los datos más recientes y registrar `logger.info("Actualizando baseline de IA...")` para trazabilidad.
  - Persistir overrides mediante `OCRCacheManager.save_manual_classifications(..., replace=True)` para evitar mezclar sesiones.
  - Mantener `ai_prediction_details` y `ai_predictions_history` con un máximo razonable (actualmente 10) para auditar cambios de modelo sin inflar el índice.

- **Pipeline post-checkpoint**
  - Antes de continuar a extracción/consolidación, recargar clasificaciones desde cache (`process_case` aplica `_ai_document_type` y respeta overrides manuales).
  - En reprocesos (`reprocess_case_background`), reactivar `ENABLE_CLASSIFICATION_REVIEW`, monitorear marcadores `.awaiting_review` y limpiar solo los del `case_id` actual.

- **UI del checkpoint**
  - Cargar `ai_classifications`, `ai_confidence` y `ai_reasons` para que “Reclasificar con AI” reproduzca el último baseline.
  - Calcular badges de confianza con el valor disponible (`doc.confidence` o, si no hay, `doc.ai_confidence`).

- **Pruebas recomendadas**
  - Crear caso → revisar checkpoint → modificar manualmente → continuar → reprocesar (con y sin reclasificar) y confirmar persistencia de overrides.
- Invocar `GET /api/system/document-type-audit` y verificar `{ "missing_document_types": [] }` antes de liberar.

> Estas prácticas mantienen alineados el motor, la UI y los reprocesos, evitando sorpresas cuando se añaden documentos o se ajusta el checkpoint manual.

### Reutilizar tipado en reprocesos (Replay)

- **Síntoma**: Los reprocesos CLI tomaban los OCR `ocr_results_for_*.json` sin el tipo original y la heurística de `AIFieldExtractor` devolvía `expediente_de_cobranza` ante cualquier referencia a “vigencia”, bloqueando campos como `fecha_ocurrencia` o `numero_siniestro`.
- **Corrección**: `ReplayService._build_case_document_types` (`src/fraud_scorer/services/replay_service.py`) lee `classified_types`, `manual_classifications` y `ai_classifications` del índice (`data/ocr_cache/case_index/*`). Normaliza los nombres (`ocr_results_for_*`) y asigna el tipo canónico (usando `ExtractionConfig.DOCUMENT_TYPE_ALIASES`). El replay injerta ese tipo en cada documento antes de invocar la extracción guiada.
- **Buenas prácticas**:
  - Siempre que se añadan nuevos tipos en el clasificador, actualizar alias/canónicos en `ExtractionConfig` para que `_map_case_document_type` los reconozca.
  - Si se cambian las reglas de renombrado en el reorganizador (`8_3_Tarja_*.pdf` → `8 3 Tarja *.pdf`), mantener la misma normalización en `ReplayService._normalize_case_filename` para evitar perder el enlace.
  - Tras ajustes, ejecutar `python scripts/replay_case.py --case-id CASE-2025-0001 --list` y revisar el JSON `replay_CASE-XXXX.json` para validar que `extraction_results[].document_type` ya no colapsa a un único valor.

---

## 8) Guías de Fraude: Escapes en Regex y Validador Preventivo

### Síntoma

- La Fase 3.5 comenzó a registrar múltiples errores `Error cargando guía ... while scanning a double-quoted scalar`.
- Los patrones en `validation_rules` usaban comillas dobles con secuencias tipo `\d`, `\s`, `\.`; PyYAML intentó interpretar los escapes y falló con `unknown escape character`.
- Resultado: 19 guías no se cargaron y el análisis de fraude solo procesó 3 documentos elegibles.

### Causa raíz

- Se escribieron regex en YAML con comillas dobles; `PyYAML` interpreta `"..."` al estilo JSON y requiere duplicar las barras (`\\d`).
- Al copiar patrones desde Python sin duplicar escapes, la carga del YAML falla antes de que `FraudGuideManager` pueda registrar la guía.

### Solución

1. **Normalizar las guías**
   - Cambiar los `pattern` a comillas simples (`'...'`) para evitar el post-procesado de escapes por YAML.
   - Revisar y corregir todas las guías impactadas (`reporte_gps.yaml`, `cfdi_carta_porte.yaml`, `carta_porte_simple.yaml`, etc.).
2. **Validador dedicado**
   - Se añadió `scripts/validate_guides.py`, que carga cada guía (YAML/JSON), exige `metadata.type`/`metadata.version` y compila cada regex.
   - El script devuelve `exit code 1` cuando una guía no es válida, listo para CI o hooks locales.
3. **Pipeline endurecido**
   - `FraudGuideManager` vuelve a cargar todas las guías sin errores; el pipeline reporta el número real de documentos elegibles.

### Procedimiento recomendado

```bash
# Validar todas las guías antes de desplegar
./venv/bin/python scripts/validate_guides.py

# Opcional: validar un directorio alterno de guías
./venv/bin/python scripts/validate_guides.py --guides-dir path/a/otro_set
```

- Agregar el script a CI (p. ej. job “Guide Validation”) o a un pre-commit hook para bloquear pushes con YAML mal formado.
- Si una guía falla, el log mostrará `Regex inválido:` o `metadata.type ausente`, señalando el archivo y la sección exacta.

> Con esta práctica se evita repetir los cortes en Fase 3.5 por escapes inválidos y se obtiene una verificación rápida antes de cada despliegue.

## 9) Doble diálogo y FileList vacío en el dashboard

### Síntoma

- Al seleccionar archivos mediante el cuadro de diálogo en macOS/Chromium el selector se abría dos veces y la UI solo mostraba archivos hasta el segundo intento.
- `handleFiles` recibía una lista vacía en Safari/WebKit, por lo que `dashboardState.selectedFiles` seguía sin cambios.

### Causa raíz

- `src/fraud_scorer/api/templates/dashboard.html:227` posiciona el `<input type="file">` sobre toda el área de carga con opacidad 0; el primer clic ya ocurre en ese control.
- El handler del contenedor (`uploadArea.addEventListener('click', () => fileInput.click());`) relanzaba `click()` sobre el input después del evento real. Chrome/Chromium reabrían el diálogo inmediatamente.
- Además, el reset inmediato (`event.target.value = ''`) sobre el mismo `FileList` provocaba que `handleFiles` leyera `length = 0` (listas "vivas" en WebKit).

### Solución

- Clonar la selección antes de limpiar el control: `const files = Array.from(event.target.files || []); handleFiles(files);` en `dashboard.html:737`.
- Evitar el doble `click()` verificando el target del evento: si el clic ya viene del `<input>`, no repetir `fileInput.click()` (`dashboard.html:720`).

### Pruebas recomendadas

1. Refrescar el dashboard y seleccionar múltiples archivos una sola vez; verificar que la lista se llena sin reabrir el diálogo.
2. Repetir en Safari/Chrome para confirmar que `handleFiles` recibe la selección (agregar `console.log(files.length)` si se requiere diagnosticar).
3. Validar arrastre y suelta; debe seguir funcionando porque usa `event.dataTransfer.files`.

---

## 7) Agente Rick RAG — Embeddings y Recuperación

### 7.1 Manejo de residencia (HTTP 451)

**Síntoma**

Al reconstruir índices con `tasks/index_builder.py`, OpenAI devolvía `HTTP 451 missing_compute_residency_info` tras varios lotes exitosos, dejando el índice a medio escribir.

**Causa raíz**

La cuenta de OpenAI requería que cada llamada a embeddings declarara explícitamente la región de residencia. El cliente default de LangChain no envía ese header.

**Solución**

- Se añadió `AGENTE_RICK_OPENAI_RESIDENCY` en `.env` y se propagó a `RickAgentConfig` (`src/fraud_scorer/ai/config.py`).
- `RickVectorStoreManager` genera el header `OpenAI-Compute-Residency` cuando la variable está presente (`src/fraud_scorer/ai/vector_store/manager.py`).
- `index_builder` captura la excepción, limpia el directorio parcial y re-lanza con mensaje explícito (`tasks/index_builder.py`).

Resultado: la indexación queda protegida ante residencias obligatorias y no deja residuos en disco si el proveedor rechaza la petición.

### 7.2 Duplicados al reindexar (DuplicateIDError)

**Síntoma**

Reejecutar `index_builder` sobre un caso ya indexado provocaba `chromadb.errors.DuplicateIDError`, deteniendo la operación.

**Causa raíz**

El manager volvía a insertar chunks con el mismo ID sin inspeccionar si ya existían en Chroma; Chroma 1.x exige que los IDs sean únicos por colección.

**Solución**

- `_filter_new_records` deduplica en memoria y consulta al store los IDs ya persistidos antes de llamar a `add_documents` (`src/fraud_scorer/ai/vector_store/manager.py`).
- Se añadió prueba idempotente (`tests/ai/test_vector_store_manager.py::test_upsert_documents_is_idempotent`).

Resultado: `index_builder` puede ejecutarse sin `--rebuild` y solo agrega chunks realmente nuevos.

### 7.3 Umbral de similitud y carga de variables

**Síntoma**

Las consultas devolvían `status: low_similarity` aun con documentos relevantes recuperados; el API no respetaba el umbral ajustado en `.env`.

**Causa raíz**

1. `AGENTE_RICK_SIMILARITY_THRESHOLD` estaba definido a 0.35, por encima de las similitudes reales (~0.23).
2. FastAPI no cargaba `.env`, por lo que usaba el valor por defecto.

**Solución**

- Se invocó `load_dotenv()` al cargar la configuración (`src/fraud_scorer/ai/config.py`) y al iniciar la aplicación (`src/fraud_scorer/api/web_interface.py`).
- Se calibró el umbral vía `.env` (`AGENTE_RICK_SIMILARITY_THRESHOLD=0.22`).

Resultado: las consultas guardan `status: answered`, incluyen fuentes y registran similitudes reales en la auditoría.

> Nota: cualquier ajuste futuro de umbral debe documentarse en este archivo y validar que la tasa de respuestas sin contexto se mantiene dentro del rango esperado.

Actualización 2025-02 — Se habilitó el histograma de similitudes en la auditoría de Rick para medir la distribución real por pregunta y se fijó el umbral por defecto en `0.35` con búsqueda híbrida (dense + BM25). Validar semanalmente que la cola `0.30-0.40` sostenga el 85 % de respuestas útiles antes de mover el umbral.

### 7.4 Compatibilidad con ChromaDB 1.1.0

- No incluir `"ids"` en `collection.get()`. Chroma 1.1.0 devuelve los IDs por defecto; pedirlos explícitamente provoca `ValueError` y corta la carga del caché híbrido. Mantén `include=["documents", "metadatas", "embeddings"]` y recrea `vector_id` desde metadata cuando falte.
- Si encuentras chunks sin `vector_id`, descártalos y sincroniza la matriz de embeddings con los documentos filtrados. Evita filas desalineadas que distorsionan el score denso o rompen BM25.
- No evalúes arreglos de NumPy con operadores booleanos (`or`, `and`). Accede directamente a `payload["embeddings"]` y valida su longitud explícitamente para evitar `ValueError: truth value of an array is ambiguous`.
- `_filter_new_records` debe usar `store.get(ids=..., include=["metadatas"])` para detectar duplicados sin violar la API. Así preservas la idempotencia en reindexados y obtienes `ids` válidos.
- Sanitiza los metadatos antes de `add_documents`: conserva `str/int/float/bool/None`, y transforma listas/tuplas en cadenas (`", ".join(...)`) o serializa dicts con `json.dumps`. Chroma 1.1.0 rechaza valores complejos y aborta el lote si quedan estructuras anidadas.
- Registra las excepciones inesperadas con el tipo (`exc.__class__.__name__`) y el mensaje original antes de elevar el `RuntimeError`. Mejora el diagnóstico de fallos reales (duplicados, conectividad, límites del proveedor) en lugar de culpar falsamente a los tokens.


## 10) Editor del Analista — Bootstrap, decisiones y ahorro

### Contexto
La vista `editor_analista.html` consolida reporte, reprocesos selectivos y Agente Rick. Mantenerla alineada con los pipelines previos evita duplicar lógica y garantiza que los guardas descritos en las secciones 1 y 2 (docless y detección de duplicados) sigan vigentes.

### Bootstrap y servicios compartidos
- Reutilizar `ReplayService` en `web_interface.py` para obtener resúmenes y montos. Su método `compute_claim_amount` aplica las mismas heurísticas que el dashboard/replay.
- El payload de `GET /api/editor/{case}/bootstrap` debe basarse en `_build_case_summary`, que ya calcula `has_ocr`, `has_extraction`, etc. Nunca reconstruir esos campos en frontend.
- Incluir en el resumen los campos `tentative_decision`, `tentative_by`, `tentative_at` y `savings_amount` para que editor y dashboard tengan una fuente de verdad compartida.

### Migración idempotente
- Añadir columnas (`tentative_decision`, `tentative_by`, `tentative_at`, `savings_amount`) usando `ensure_editor_columns()` dentro de `storage/db.py`. Este helper ejecuta `PRAGMA table_info` y solo aplica `ALTER TABLE` cuando falta una columna, permitiendo despliegues sin scripts manuales.

### Guardas para reprocesos
- Las tarjetas de reproceso en el editor deben respetar las banderas de `_build_case_summary`:
  - Fase 1.4 requiere `has_ocr`.
  - Fase 2 depende de `has_classifications`.
  - Fase 3 y 3.5 exigen `has_extraction` (modo docless de la sección 1).
- Mostrar mensajes claros cuando una opción se bloquee para evitar reprocesos fallidos.

### Flujo post-proceso
- `process_monitor.html` ya no presenta el panel legacy de reprocesos. Si `checkExistingCase()` detecta duplicado, redirige directo a `/analyst/{case_id}`.
- Tras un procesamiento nuevo, mostrar botón “Ir al Editor” y programar un redirect automático (4 s) hacia `/analyst/{case_id}`; así todo caso pasa por la revisión central.

### Registro de decisiones y ahorro
- `POST /api/case/{case}/decision` debe:
  1. Validar `{'with','without'}`.
  2. Calcular ahorro con `replay_service.compute_claim_amount`.
  3. Persistir con `set_case_decision` (actualiza columnas y `updated_at`).
  4. Guardar índice de caso (`OCRCacheManager.save_case_index`).
- El dashboard debe consumir `savings_amount` al mostrar métricas (`SUM(savings_amount)`) y la columna de decisión.

### Pruebas rápidas
1. `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile …` tras modificar endpoints del editor (detecta variables como `replay_service`).
2. Subir archivos que ya existen → confirmar redirección al editor sin residuos en `temp/`.
3. Marcar “Con tentativa” → verificar columnas en `cases`, índice JSON y actualización en dashboard.
4. Intentar reproceso 3.5 sin extracciones → la UI debe bloquearlo y detallar el motivo.

## 11) Editor del Analista — Integridad de assets y controles interactivos

### Síntoma
- Tras un refactor de layout el editor cargaba sin tarjetas de reproceso, sin panel de Rick y con todos los botones (zoom, PDF, decisiones) inoperantes.
- El inspector mostraba errores de JavaScript porque el módulo `editor_analista.js` contenía literales `\n` y fragmentos incompletos; el navegador abortó la ejecución antes de inicializar cualquier handler.

### Causa raíz
- El script se reescribió usando sustituciones rápidas que introdujeron secuencias escapadas (`\n`) en lugar de saltos de línea reales.
- Se añadieron listeners y helpers en puntos intermedios sin recompilar/validar el bundle, de modo que el archivo quedó incoherente y sin pruebas posteriores.

### Solución aplicada
1. **Reescritura completa del módulo:** se regeneró `editor_analista.js` con el estado del zoom, listeners y bootstrap en un solo bloque coherente.
2. **Validación posterior:** se ejecutó `PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=./.pycache_tmp python3 -m py_compile …` para comprobar que no quedaran errores de sintaxis o importaciones.
3. **Controles de zoom seguros:** se inicializa el zoom en `applyZoom` después de `bootstrap`, se persiste por `case_id` y se vuelve a aplicar cuando carga el iframe.
4. **Scroll controlado:** los paneles ahora usan `height:100vh`, `min-height:0` y `overflow-y:auto` para evitar que el contenido desaparezca.

### Prevención recomendada
- Al editar JS almacenado, evitar scripts de reemplazo masivo con secuencias escapadas; preferir utilidades tipo `apply_patch` sin literales `\n` o reescribir el bloque completo.
- Tras cambios en `static/js/` o `static/css/`, ejecutar `python3 -m py_compile …` sobre los módulos Python afectados y abrir el editor en local para revisar que los elementos interactivos aparezcan.
- Cada vez que se toque el layout del editor:
  - Verificar que `REPROCESS_TASKS` se renderice (guard docless activo).
  - Confirmar respuesta de botones clave (Descargar PDF, Abrir reporte, Con/Sin tentativa, Reindexar Rick).
  - Probar los niveles de zoom extremos (50 % / 150 %) y asegurar que la preferencia persiste al recargar.
- Documentar en la PR cuáles pruebas UI manuales se ejecutaron (captura o lista) antes de fusionar.

## 12) Motor de correlación inter-documental (Fase 3.5+)

### Síntoma
- Los reportes solo mostraban hallazgos por documento; contradicciones entre póliza, facturas, carta porte o carpetas de investigación se detectaban manualmente.
- Las reglas de fraude individuales no podían cruzar consolidado + extracciones + resultados previos, lo que elevaba el riesgo de falsos negativos.

### Solución
- Se creó el motor en `src/fraud_scorer/analyzers/correlation/` con:
  - `CaseContext`: hidrata consolidado, extracciones y resultados de fraude, construye agregados (`aggregates`) y timeline ordenado.
  - `RuleEngine`: evalúa reglas YAML (`equality`, `temporal_order`, `set_overlap`, `exists`) y devuelve `CorrelationFinding`.
  - `CorrelationEngine`: orquesta reglas, correlador estadístico, RAG y devuelve `CorrelationReport` con métricas.
- Integración en `scripts/run_report.py` tras la Fase 3.5; el reporte HTML incorpora la sección *Correlaciones inter-documentos* con métricas de estado/severidad.
- Los hallazgos se guardan en `case_index['fraud_correlations']` y en el JSON de resultados del caso.

### Buenas prácticas para no romper el motor
- **Contexto**: Siempre pasar objetos Pydantic (`DocumentExtraction`, `FraudAnalysisResult`). Si el input viene como dict, validar con `model_validate` antes de invocar `CaseContext.from_case`.
- **Rutas en reglas**: Verificar con `CaseContext.resolve()` en shell/REPL antes de publicar una regla nueva. Rutas inexistentes degradan automáticamente a `needs_context`.
- **Tolerancias**: Definir `tolerance` (porcentaje) o `absolute_tolerance` para montos. Sin tolerancia, se exige igualdad exacta.
- **Fechas**: Normalizar extracciones a `YYYY-MM-DD`; las cadenas inválidas provocan fallbacks y el hallazgo pasa a `needs_context`.
- **Evidencia**: Conservar la política de degradación defensiva. Nunca lanzar excepciones desde evaluadores; usar `missing_summary` en YAML para hallazgos con datos insuficientes.
- **Reportes**: Mantener `FraudReportGenerator.prepare_fraud_report_data` con `correlation_report` opcional y preservar la clave `mostrar_seccion_correlacion` para evitar romper plantillas existentes.
- **Configuración estadística**: Versionar `rules/statistical_config.yaml` igual que las reglas determinísticas. Documentar en la PR los cambios de umbrales, tolerancias y nuevos checkers.
- **RAG**: Habilitar `CORRELATION_ENABLE_RAG=true` solo cuando el Agente Rick esté indexado para el caso. El builder debe degradar a `FAIL` únicamente si recibe respuesta distinta a `NO_CONTEXT_MESSAGE`; en otro caso conservar el `status` original.
- **Persistencia**: Grabar hallazgos con `save_correlation_findings()` inmediatamente después de construir el reporte. Si se agregan campos nuevos, actualizar el esquema de `fraud_correlations` y los helpers de lectura/escritura.

### Consejos de implementación
- Reutilizar agregados de `CaseContext.aggregates` para reglas numéricas: evita recontar montos en código.
- Versionar cada cambio de reglas en `rules/correlation_rules.yaml` (`meta.version`) y documentarlo en la PR.
- Añadir aliases en `rules/entity_mappings.yaml` cuando un nuevo documento reutiliza campos existentes (ej. variantes de factura o VIN).
- Migrar cualquier validador Pydantic que toque extracciones a la sintaxis v2 (`@field_validator`). Así evitamos warnings y mantenemos coherencia con el resto del proyecto.
- Generar timestamps con `datetime.now(timezone.utc)` cuando se creen reportes o entradas de auditoría; facilita comparar registros entre servicios.

### Testing recomendado
- Ejecutar `python3 -m pytest tests/correlation -v` al tocar reglas, agregados o evaluadores para garantizar compatibilidad.
- Usar fixtures con `tmp_path` y YAML sintético para ensayar nuevas reglas antes de incorporarlas al catálogo oficial.
- Correr `python3 -m compileall src/fraud_scorer/analyzers/correlation` dentro del `venv` para detectar errores de sintaxis o imports.
- Definir `asyncio_default_fixture_loop_scope = "function"` en `pyproject.toml` para que los tests async dejen de depender de heurísticas implícitas de `pytest-asyncio`.
- Validar las transformaciones críticas (montos, fechas, listas) tras cada migración de Pydantic; las suites de correlación (`tests/correlation/test_correlation_engine.py`) cubren los escenarios mínimos.

### Observabilidad y fallback
- El log `"✓ Motor de correlación ejecutado: %s hallazgos"` confirma la ejecución. Si falta, revisar `self.enable_fraud` y la configuración de reproceso.
- Si una regla falla o faltan datos, el motor degrada a `needs_context`; monitorizar los conteos en el JSON y en la tabla `fraud_correlations` para detectar reglas demasiado restrictivas.
- Registrar periódicamente las métricas (`statistical_count`, `rag_enabled`, latencias Rick) para descubrir degradaciones. Un pico de `needs_context` suele indicar datos faltantes o tolerancias mal calibradas.
- Conservar índices (`idx_fcorr_case`, `idx_fcorr_rule`) y limpiar registros antiguos cuando se repite un caso en múltiples iteraciones; evita crecimiento descontrolado de la tabla.

## 13) Análisis de fraude individual — Capa unificada y brechas de evidencia

### Síntoma
- El análisis dependía solo de los campos extraídos por documento; si un dato existía en el consolidado pero no en el documento actual, el LLM lo trataba como ausente, generando falsos positivos de "datos faltantes".
- La respuesta mezclaba indicadores de fraude y brechas de evidencia en un mismo JSON, impidiendo distinguir entre señales de riesgo y tareas pendientes.
- El reporte HTML no mostraba la trazabilidad de brechas ni existía un modelo estructurado para persistirlas en `fraud_analyses`.

### Solución aplicada
- Se construyó `UnifiedDataLayer` (`src/fraud_scorer/analyzers/unified_data_layer.py`), que prioriza consolidado y extracciones confiables, y solo recurre a OCR/regex cuando falta información. Expone `build_case_context` y `build_document_context` que reutilizamos en todos los pipelines.
- `FraudPromptBuilder` diferenciá prompts de indicadores (`build_indicator_prompt`) y brechas (`build_evidence_gap_prompt`), ambos alimentados con el contexto unificado.
- `FraudAnalyzer.analyze_document` ejecuta ambos prompts, fusiona los resultados, calcula un `prompt_hash` combinado y normaliza la salida en `FraudAnalysisResult`, que ahora incorpora `EvidenceGap` (`src/fraud_scorer/models/fraud_analysis.py`). La persistencia añade la columna `evidence_gaps` de forma automática (`src/fraud_scorer/analyzers/fraud_analyzer.py`).
- El generador de reportes incluye la sección *Brechas de evidencia* con totales y fuentes (`src/fraud_scorer/templates/fraud_report_generator.py`).
- Las rutas de reproceso y generación inicial invocan la capa unificada antes de llamar al analizador (`src/fraud_scorer/services/fraud_document_service.py`, `src/fraud_scorer/api/endpoints/reports.py`, `scripts/run_report.py`).

### Consideraciones de implementación
- **Presencia por documento**: `build_document_context` marca un campo como cubierto solo si proviene del mismo documento (extracción u OCR regex). Los valores heredados de consolidado se mantienen en `resolved_fields`, pero el campo aparece en `missing_fields` para solicitar evidencia adicional.
- **Migración de base**: al primer guardado se ejecuta `ALTER TABLE` para añadir `evidence_gaps` si falta. En entornos con migraciones estrictas conviene forzar un reproceso controlado después del despliegue.
- **Sincronización del esquema**: `src/fraud_scorer/storage/db.get_conn` invoca `ensure_evidence_gaps_column`, que vuelve idempotente la migración (`ALTER TABLE … evidence_gaps TEXT NOT NULL DEFAULT '[]'`) en cada conexión. Antes de tocar `fraud_analyses`, mantiene este helper y actualiza también el DDL base (`init_db`) para instalaciones nuevas.
- **Lecturas defensivas**: al consumir resultados desde la DB, usa `COALESCE(fa.evidence_gaps, '[]')` (ver `fraud_document_service._hydrate_from_db`) para blindar queries antiguas. Evita asumir que la columna existe o que siempre trae JSON válido.
- **Hash combinado**: cambios en cualquiera de los prompts invalidan el `prompt_hash`; documentar en PRs cuándo se modifiquen para evitar interpretaciones erróneas en auditoría.
- **Integraciones externas**: cuando se invoque `FraudAnalyzer.analyze_batch` fuera del pipeline estándar, construir el `UnifiedDataLayer` con consolidado, extracciones y clasificaciones manuales como referencia.
- **Fallbacks**: si la IA falla (timeout, JSON inválido) se registrará en `analysis.evidence` y la brecha permanece vacía para no bloquear la ejecución.

### Observabilidad
- Revisar `total_brechas` y `brechas_evidencia` en el reporte HTML como indicador temprano de plantillas incompletas o extracción deficiente.
- Consultar la columna `evidence_gaps` en `fraud_analyses` para identificar patrones recurrentes (por tipo de documento o aseguradora).
- Confirmar que los reportes usen `consolidated.consolidated_fields` y accedan con `getattr` para tolerar cambios en `ConsolidatedExtraction`. Cualquier refactor debe evitar referencias directas a atributos inexistentes (`fields`).
- Incorporar conteos de brechas vs. indicadores en los dashboards existentes para distinguir ausencia de datos de hallazgos de fraude.

### Testing recomendado
- `python3 -m pytest tests/analyzers/test_unified_data_layer.py` — valida prioridades (estructurado vs OCR), detección de faltantes y regex.
- `python3 -m pytest tests/correlation -q` — asegura que la nueva estructura de `FraudAnalysisResult` no rompe la fase de correlación.
- Ejecutar los tests de RAG/fraude que ejercitan el reproceso (`tests/api/test_editor_reprocess_document.py` cuando esté disponible) para confirmar que las brechas se reflejan en la respuesta JSON del editor.
- Tras despliegue, reprocesar un caso de referencia (p. ej. `CASE-2025-0001`) y verificar manualmente la sección de brechas en el reporte generado.

## 14) Consolidación guiada — Presupuesto de tokens y escalaciones

- **Síntoma**: `python scripts/replay_case.py --case-id …` seguía cayendo al respaldo determinístico porque `gpt-5` agotaba tokens o entregaba JSON incompleto para campos críticos.
- **Ajustes aplicados**:
  - `ExtractionConfig.get_openai_params("extraction")` fija `temperature=0.1`, `top_p=0.2` y `max_completion_tokens=1600`; la variante `extraction_escalated` amplía presupuesto/timeout para `gpt-5-thinking` (`src/fraud_scorer/settings.py`).
  - `AIFieldExtractor.extract_from_document_guided` detecta huecos en `suma_asegurada`, `monto_reclamacion` y `tipo_siniestro` y relanza automáticamente con `gpt-5-thinking`, registrando `models_attempted` en la metadata (`src/fraud_scorer/processors/ai/ai_field_extractor.py`).
  - `AIConsolidator._resolve_conflict_with_ai` utiliza `gpt-5-mini` y escala a `gpt-5` cuando la confianza en campos de alto riesgo cae por debajo de 0.65 o la fuente proviene de `fallback:`; los modelos usados quedan en `_models_attempted` (`src/fraud_scorer/processors/ai/ai_consolidator.py`).
  - `_completion_budget` se alimenta ahora de `OPENAI_CONFIG['consolidation']` (1500/2000 tokens) y mantiene el techo en 3500 para evitar nuevos truncamientos.
- `DOCUMENT_EXTRACTION_ROUTES` mantiene pólizas e informes en OCR, pero la ruta `direct_ai` pasa a usar `gpt-5-vision` para escaneos con bajo texto (`src/fraud_scorer/settings.py`).
- `_apply_post_consolidation_fallbacks` continúa como red de seguridad, marcando la fuente con `fallback:` y confianza ≈0.55.
- **Recomendaciones**:
  - Antes de tocar temperaturas o top_p verificar soporte del proveedor; valores fuera del rango recomendado reintroducen `400 invalid_parameter`.
  - Monitorizar `consolidation_sources._models_attempted`: si escala a `gpt-5` con frecuencia, revisa prompts y prioridades antes de fijar `complexity="high"` por defecto.
  - Permitir override de `OPENAI_MAX_COMPLETION_TOKENS` sólo después de auditar `replay_CASE-*.json` para confirmar que el problema es longitud y no formato.
- **Guía rápida de selección de modelos**:
  - Extracción (OCR texto): `gpt-5` es la base; `gpt-5-thinking` se reserva para huecos >5 % en campos críticos.
  - Extracción directa/visión: `gpt-5-vision` es el estándar para escaneos; no requiere escalación adicional.
  - Consolidación: `gpt-5-mini` + IA guiada cubre la mayoría de conflictos; `gpt-5` sólo cuando la confianza cae o persisten discrepancias entre póliza e informe.

## 15) GPS directo — Fallback de texto plano y PDFs verticales

### Síntoma
- Los reportes GPS devolvían `row_count = 0` aunque el PDF contenía coordenadas y eventos; los registros venían en formato vertical (cada campo en una línea distinta), por lo que el extractor buscaba la columna `GPS` y la latitud en líneas separadas sin encontrarlas.

### Solución aplicada
- `extract_tables_from_text_segments` (`src/fraud_scorer/parsers/gps_direct_extractor.py`) agrupa líneas consecutivas con `_iter_candidate_records` y arma un registro sintético antes de aplicar regex.
- `_parse_candidate_line` reutiliza los patrones de timestamp, latitud/longitud y velocidad para construir filas canónicas listas para `normalize_gps_tables`, manteniendo el texto original en `raw_line`.
- Pruebas dedicadas (`tests/gps/test_gps_direct_extractor.py`) cubren registros en una sola línea, formato vertical y entradas sin coordenadas.

### Consideraciones de implementación
- Mantener `_is_candidate` ligero: valida longitud mínima, presencia de “gps”, timestamp y coordenadas; evita evaluaciones costosas antes de tiempo.
- Usar `join` con espacios al fusionar líneas para no perder separadores; si el PDF trae ruido (paginas con encabezados), filtrar buffers que no cumplan con timestamp/coord.
- Al extender soporte para nuevos proveedores, añadir tokens al patrón `_EVENT_PATTERN` y al diccionario `_EVENT_CANONICAL` para preservar nomenclatura normalizada.
- No elimines `raw_line`: sirve para auditoría y debugging cuando las regex necesitan ajustes.

### Testing recomendado
- `python3 -m pytest tests/gps/test_gps_direct_extractor.py -q` tras tocar regex o heurísticas de `_iter_candidate_records`.
- Reprocesar un caso real (ej. `CASE-2025-0001`) y comprobar que `gps_summary.row_count` > 0 y que el cache Parquet contiene las filas reconstruidas.
- Revisar los warnings en `metadata['gps_direct']['normalization_warnings']`; si aparece `missing_coordinates`, ajustar regex o delimitadores antes de cerrar la PR.

## 16) Clasificación documental — Modelos LLM soportados

- **Síntoma**: el clasificador textual devolvía `temperature not supported` y pasaba al heurístico.
- **Causa**: `CLASSIFICATION_CONFIG['llm_model'] = "gpt-5-mini"` (alias inexistente).
- **Acción**: usar `gpt-4o-mini` (mismo stack que OCR/consolidación) sin alterar temperatura ni tokens.
- **Recomendaciones**: validar aliases con `OPENAI_MODEL_CONFIG`, probar en lote pequeño antes de generalizar y, si se experimenta con otros modelos, envolverlo detrás de `CLASSIFICATION_ENGINE.strategy`.
- **Checks**: `python3 -m pytest tests/analyzers/test_unified_data_layer.py`, reprocesar un caso y confirmar en `/api/editor/{case}/bootstrap` que la clasificación reporta `gpt-4o-mini`.

## 17) Carta aclaratoria de peaje — buenas prácticas de análisis (marzo 2026)

- **Preparación de datos**: antes de tocar la guía `carta_aclatoria_comprobantes_peaje.yaml` o su post-proceso, verifica que el `case_index` incluya los datasets GPS reconstruidos (`gps_direct_documents`). Sin ellos, reporta explícitamente la ausencia en `ruta_vs_gps` y evita generar indicadores no sustentados.
- **Tolerancias y evidencia**: conserva los umbrales actuales (±600 m, ±20 min) en `fraud_analyzer._postprocess_carta_aclaratoria_peaje`. Si necesitas ajustarlos, reejecuta `preview_fraud_analysis.py --no-save --refresh-extraction` sobre CASE-2025-0001 y documenta en la PR el impacto en coincidencias vs. desviaciones.
- **Narrativa consistente**: el resumen debe omitir el conteo de páginas, usar solo el nombre real del destinatario (sin “Estimado”) y redactar el propósito como una frase corrida. Si cambias la plantilla YAML, comprueba que el post-proceso siga produciendo esa estructura.
- **Trazabilidad por ECO**: revisa que `verificaciones.ruta_vs_gps.detalle` y `validacion_cruzada.monitoreo_gps.detalle_por_unidad` enumeren coincidencias y discrepancias por unidad (ECO 006 / ECO 010) con timestamp y coordenadas. Es la referencia principal para auditores; no borres esos bloques aunque modifiques recomendaciones.
