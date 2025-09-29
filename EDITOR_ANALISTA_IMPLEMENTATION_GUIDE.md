# Guía de Implementación: Editor del Analista (v2 – septiembre 2024)

## 1. Objetivo
Crear la vista central del analista que combine reporte, reprocesos selectivos y Agente Rick, respetando la estética de `login.html`, reutilizando la infraestructura existente y las recomendaciones de `BETTER_PRACTICES.md`.

## 2. Resumen funcional
- Vista de tres columnas con sombra negra y tipografía Montserrat.
- Panel izquierdo: reprocesos parciales (migrados desde `upload.html`) ejecutados sin salir de la página.
- Panel central: visor del reporte HTML + acciones (descarga, ventana nueva).
- Panel derecho: Agente Rick + botones de decisión (con/sin tentativa, eliminar siniestro, abrir reporte completo).
- Botón superior “Volver al dashboard”.

## 3. Cambios en el flujo
1. Renombrar `src/fraud_scorer/api/templates/upload.html` a `process_monitor.html` (mantiene barra de progreso, revisión 1.4.1 y SSE). Ajustar `get_upload_page` para renderizar la nueva plantilla.
2. Crear `src/fraud_scorer/api/templates/editor_analista.html` y servirla desde `GET /analyst/{case_id}` (nuevo handler en `web_interface.py`).
3. En `dashboard.html`, actualizar el botón "Editar" para redirigir a `/analyst/{{ case_id }}` y, al finalizar un procesamiento nuevo, redirigir automáticamente al editor.
4. En `process_monitor.html`, eliminar el panel de reprocesos y mostrar únicamente estado/progreso; cuando `monitorProgress` reciba `status.status === 'completed'`, mostrar botón “Ir al Editor” que abra `/analyst/${status.case_id}`.

## 4. Arquitectura UI
```html
<!-- src/fraud_scorer/api/templates/editor_analista.html -->
<body class="editor-body" data-case-id="{{ case_id }}">
  <div class="editor-shell">
    <aside class="reprocess-panel" id="reprocessPanel">
      <button class="ghost-btn" onclick="window.location.href='/'">⟵ Volver al dashboard</button>
      <h2>Opciones de reprocesamiento</h2>
      <div id="reprocessCards"></div>
      <div class="progress" id="reprocessProgress" hidden>
        <div class="bar"><span id="reprocessFill"></span></div>
        <span id="reprocessLabel">Preparando reproceso…</span>
        <button class="ghost-btn" id="reprocessAbort" hidden>Cancelar tarea</button>
      </div>
      <section class="case-meta" id="caseMeta"></section>
    </aside>

    <main class="report-viewer">
      <header class="report-header">
        <h1 id="caseTitle">Editor del analista</h1>
        <div class="report-actions">
          <button class="primary" id="btnDownloadPdf">Descargar PDF</button>
          <button class="secondary" id="btnOpenFull">Abrir reporte completo</button>
        </div>
      </header>
      <iframe id="reportFrame" src="/report/{{ case_id }}" title="Reporte" loading="lazy"></iframe>
    </main>

    <aside class="right-rail">
      <section id="rickPanel"></section>
      <section class="case-actions">
        <button class="danger" id="btnDeleteCase">Eliminar siniestro</button>
        <div class="decision-group">
          <button class="accent" data-decision="with">Con tentativa</button>
          <button class="neutral" data-decision="without">Sin tentativa</button>
        </div>
        <div class="status-chip" id="tentativeStatus"></div>
        <div class="status-chip" id="savingsStatus"></div>
      </section>
    </aside>
  </div>
  <script src="/static/js/editor_analista.js" type="module"></script>
  <script src="/static/js/agente_rick.js" type="module"></script>
</body>
```

## 5. Estilos
Crear `static/css/editor_analista.css` inspirado en `login.html`:
```css
:root {
  --bg: #f9f9f9;
  --card: #ffffff;
  --border: 2.75px solid #000;
  --shadow: 14px 17px 0 0 #000;
  --accent: #111111;
  --accent-light: #f3f3f3;
  --success: #0f9d58;
  --danger: #d32f2f;
  --font: 'Montserrat', sans-serif;
}
body.editor-body { background:#ffffff; font-family:var(--font); }
.editor-shell { display:grid; grid-template-columns: 320px 1fr 340px; min-height:100vh; border:var(--border); box-shadow:var(--shadow); background:var(--card); }
.reprocess-panel, .right-rail { border-right: var(--border); padding:32px 24px; background:var(--card); }
.report-viewer { padding:32px 24px; display:flex; flex-direction:column; }
.report-viewer iframe { flex:1; border:var(--border); box-shadow:6px 8px 0 0 #000; }
@media(max-width:1280px){ .editor-shell{grid-template-columns:280px 1fr 320px;} }
@media(max-width:1024px){ .editor-shell{grid-template-columns:1fr; grid-template-rows:auto 1fr auto;} .right-rail{border-left:none;border-top:var(--border);} }
```

## 6. Lógica frontend (`static/js/editor_analista.js`)
Responsabilidades clave:
1. Leer `caseId` desde `document.body.dataset.caseId`.
2. `bootstrap()` → `GET /api/editor/${caseId}/bootstrap` para obtener:
   - Resumen (`/api/case/{case_id}/summary` reutilizado internamente).
   - Último estado de reprocesos en `processing_status`.
   - Enlaces al reporte (iframe y PDF).
   - Estado manual `tentative_decision` y ahorro (nuevo backend, ver §7.3).
3. Renderizar tarjetas de reproceso usando la misma taxonomía que existía en `upload.html`. Cada tarjeta hace `POST /api/case/{caseId}/reprocess` con `options`.
4. Al recibir `process_id`, reutilizar `monitorProgress` (adaptado) para actualizar la barra y, cuando finalice, recargar el iframe y actualizar el resumen. Consumir `GET /status/${process_id}`.
5. Botones de decisión llaman a `POST /api/case/{caseId}/decision` (payload `{ "decision": "with" | "without" }`).
6. “Eliminar siniestro” reutiliza `/replay/api/deep-purge/{case_id}` y redirige al dashboard si tiene éxito.
7. Inicializar Agente Rick llamando a `window.rickChat.init(caseId)` (ver guía de Rick).

## 7. Cambios de backend
### 7.1 Rutas
- `GET /analyst/{case_id}`: retorna la plantilla con `case_id`, `user`, `report_url = f"/report/{case_id}"`.
- `GET /api/editor/{case_id}/bootstrap`: compone `{ summary, report_url, pdf_url, tentative_decision, savings_amount, active_reprocess }` reutilizando `OCRCacheManager`, `processing_status` y utilidades del dashboard.
- `GET /api/editor/{case_id}/report/pdf`: genera o sirve el PDF si existe; fallback a mensaje “no disponible”.

### 7.2 Panel de reprocesos
- El endpoint `POST /api/case/{case_id}/reprocess` ya existe; la UI debe enviar las mismas banderas (`reprocess_fraud`, `reprocess_extraction`, etc.).
- Mantener las recomendaciones de `BETTER_PRACTICES` §1 y §2: si no hay `extraction_results` bloquear el modo docless, limpiar sólo marcadores del caso (`case_id.awaiting_review`).

### 7.3 Decisiones y tablero
1. Migración SQL (idempotente) para añadir columnas:
   ```sql
   ALTER TABLE cases ADD COLUMN tentative_decision TEXT CHECK(tentative_decision IN ('with','without')) DEFAULT NULL;
   ALTER TABLE cases ADD COLUMN tentative_by TEXT;
   ALTER TABLE cases ADD COLUMN tentative_at TEXT;
   ALTER TABLE cases ADD COLUMN savings_amount REAL DEFAULT 0;
   ```
2. Nuevos helpers en `storage/cases.py`:
   - `set_case_decision(case_id, decision, user, savings)`.
   - `get_total_savings()`.
3. Nuevo endpoint `POST /api/case/{case_id}/decision`:
   - Valida payload `{"decision": "with"|"without"}`.
   - Calcula ahorro usando `consolidated_fields.monto_reclamacion` cuando exista.
   - Actualiza DB, índice (`OCRCacheManager.save_case_index`) y retorna `{ tentative_decision, savings_amount }`.
   - Registra auditoría.
4. Actualizar `/replay/api/stats` para usar `SUM(savings_amount)` en vez de `N/A` y mostrar la decisión real en la tabla.

### 7.4 Reporte y PDF
- Mantener `iframe` apuntando a `/report/{case_id}`.
- Botón “Descargar PDF” genera/servir `data/reports/{case_id}.pdf` mediante `weasyprint`.
- Botón “Abrir reporte completo” abre `/report/{case_id}` en pestaña nueva.

## 8. Integración con Agente Rick
- Incluir `static/js/agente_rick.js` en la plantilla solo cuando `caseId` exista.
- Proveer contenedor `#rickPanel` (ver guía del Agente Rick para montar la UI).

## 9. Consideraciones de `BETTER_PRACTICES.md`
- Reutilizar el modo docless descrito en la sección 1 para reprocesos 3.5 y fases superiores.
- Mantener `ENABLE_CLASSIFICATION_REVIEW` activo mientras se esté en reproceso desde el editor.
- Antes de exponer reanudación, revisar `_audit_document_type_groups()` y mostrar advertencia si faltan tipos.

## 10. Pruebas recomendadas
1. Crear caso nuevo → tras completarse, abrir editor y validar resumen, reporte y botones.
2. Ejecutar cada reproceso individual (1, 1.4, 2, 3, 3.5) y verificar bloqueos “awaiting_review” cuando aplique.
3. Marcar "Con tentativa" → refrescar dashboard y comprobar actualización de estado y ahorro real.
4. Eliminar caso desde el editor y comprobar que desaparece del dashboard y que `/report/{case_id}` devuelve 404.
5. Validar responsive ≥1024px (layout en columnas) y <1024px (paneles apilados).
6. Ejecutar `scripts/validate_guides.py` y smoke tests existentes.

## 11. Checklist de entrega
- [ ] `process_monitor.html` sin panel de reprocesos.
- [ ] Ruta `/analyst/{case_id}` operativa y autenticada.
- [ ] CSS/JS del editor versionados en `static/`.
- [ ] Endpoint de bootstrap y de decisión funcionando.
- [ ] Métrica de ahorro visible en dashboard.
- [ ] Documentación actualizada en guías y README.
