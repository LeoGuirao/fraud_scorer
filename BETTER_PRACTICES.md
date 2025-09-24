# Postmortem y Guía Técnica: Reproceso 3.5, Detección de Duplicados y Conteos

Este documento describe en detalle los problemas detectados y las soluciones implementadas en el sistema Fraud Scorer en cuatro áreas:

- Reprocesamiento exclusivo de la Fase 3.5 (análisis de fraude) sin archivos originales.
- Activación confiable del menú de reprocesamiento al subir archivos ya procesados.
- Conteos erróneos de documentos (índices inflados) en la UI y normalización del índice del caso.
- Prevención de la creación de la base de datos innecesaria `data/fraud_scorer.db`.

Se incluyen causas raíz, decisiones de diseño y fragmentos de código exactos con rutas de archivo reales, alineados con la estructura actual del proyecto.

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

## 5) Reutilización de análisis de fraude en reprocesos parciales

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
  - `scripts/clean_orphaned_files.py --dry-run` para vista previa y `--all` para limpieza total (incluye `feedback_archive` y `raw`).
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
