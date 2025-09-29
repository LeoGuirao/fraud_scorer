const caseId = document.body.dataset.caseId;

if (!caseId) {
  console.error('Editor del analista requiere un caseId válido');
}

const ZOOM_LEVELS = [0.5, 0.75, 1, 1.25, 1.5];
const DEFAULT_ZOOM_INDEX = Math.max(ZOOM_LEVELS.indexOf(1), 0);
const ZOOM_STORAGE_KEY = caseId ? `editor.zoom.${caseId}` : 'editor.zoom';

function getStoredZoomIndex() {
  try {
    const raw = localStorage.getItem(ZOOM_STORAGE_KEY);
    if (raw === null) {
      return DEFAULT_ZOOM_INDEX;
    }
    const parsed = Number.parseInt(raw, 10);
    if (Number.isInteger(parsed) && parsed >= 0 && parsed < ZOOM_LEVELS.length) {
      return parsed;
    }
  } catch (error) {
    console.warn('No se pudo leer zoom almacenado', error);
  }
  return DEFAULT_ZOOM_INDEX;
}

function storeZoomIndex(index) {
  try {
    localStorage.setItem(ZOOM_STORAGE_KEY, String(index));
  } catch (error) {
    console.warn('No se pudo persistir zoom del editor', error);
  }
}

const state = {
  caseId,
  bootstrap: null,
  polling: null,
  activeProcessId: null,
  zoomIndex: getStoredZoomIndex(),
  zoomInitialized: false,
  fraudCatalog: null,
  fraudDocumentLocks: new Set(),
  processHooks: {},
  activeTab: 'analysis',
  reportEdit: {
    isEditing: false,
    originalHTML: null,
    hasManualOverride: false,
  },
  correlations: {
    filters: { status: 'all', severity: 'all' },
    findings: [],
    counts: {},
    summary: {},
  },
};

const elements = {
  editorMain: document.getElementById('editorMain'),
  editorTabs: document.getElementById('editorTabs'),
  analysisPanel: document.getElementById('analysisPanel'),
  reportPanel: document.getElementById('reportPanel'),
  fraudDocumentsList: document.getElementById('fraudDocumentsList'),
  fraudDocumentsEmpty: document.getElementById('fraudDocumentsEmpty'),
  fraudDocumentsCount: document.getElementById('fraudDocumentsCount'),
  reprocessCards: document.getElementById('reprocessCards'),
  reprocessProgress: document.getElementById('reprocessProgress'),
  reprocessFill: document.getElementById('reprocessFill'),
  reprocessLabel: document.getElementById('reprocessLabel'),
  reprocessAbort: document.getElementById('reprocessAbort'),
  caseMeta: document.getElementById('caseMeta'),
  reportFrame: document.getElementById('reportFrame'),
  reportFrameInner: document.getElementById('reportFrameInner'),
  reportFrameViewport: document.getElementById('reportFrameViewport'),
  zoomOut: document.querySelector('[data-zoom="out"]'),
  zoomIn: document.querySelector('[data-zoom="in"]'),
  zoomLabel: document.getElementById('zoomLabel'),
  caseTitle: document.getElementById('caseTitle'),
  caseSubtitle: document.getElementById('caseSubtitle'),
  btnDownloadPdf: document.getElementById('btnDownloadPdf'),
  btnOpenFull: document.getElementById('btnOpenFull'),
  btnEditReport: document.getElementById('btnEditReport'),
  btnSaveReport: document.getElementById('btnSaveReport'),
  btnCancelEdit: document.getElementById('btnCancelEdit'),
  btnResetReport: document.getElementById('btnResetReport'),
  btnDeleteCase: document.getElementById('btnDeleteCase'),
  decisionButtons: Array.from(document.querySelectorAll('[data-decision]')),
  tentativeStatus: document.getElementById('tentativeStatus'),
  savingsStatus: document.getElementById('savingsStatus'),
  rickPanel: document.getElementById('rickPanel'),
  correlationPanel: document.getElementById('correlationPanel'),
  correlationSummary: document.getElementById('correlationSummary'),
  correlationBadge: document.getElementById('correlationSummaryBadge'),
  correlationTable: document.getElementById('correlationTable'),
  correlationEmpty: document.getElementById('correlationEmpty'),
  correlationStatusFilter: document.getElementById('correlationStatusFilter'),
  correlationSeverityFilter: document.getElementById('correlationSeverityFilter'),
};

const REPROCESS_TASKS = [
  {
    id: 'phase1',
    title: 'Reprocesar OCR',
    description: 'Reconstruye la fase 1 completa con nuevos OCR shards.',
    badge: 'Fase 1',
    options: {
      reprocess_ocr: true,
      reprocess_classification: true,
      reprocess_policy_detection: true,
      reprocess_extraction: true,
      reprocess_consolidation: true,
      reprocess_fraud: true,
    },
  },
  {
    id: 'phase14',
    title: 'Clasificación 1.4',
    description: 'Reclasifica documentos sin repetir OCR.',
    badge: 'Fase 1.4',
    options: {
      reprocess_classification: true,
      reprocess_policy_detection: true,
      reprocess_extraction: true,
      reprocess_consolidation: true,
      reprocess_fraud: true,
    },
    guard(summary) {
      if (!summary?.has_ocr) {
        return {
          allowed: false,
          reason: 'Requiere resultados de OCR previos.',
        };
      }
      return { allowed: true };
    },
  },
  {
    id: 'phase2',
    title: 'Extracción',
    description: 'Repite extracción de campos y consolida nuevamente.',
    badge: 'Fase 2',
    options: {
      reprocess_extraction: true,
      reprocess_consolidation: true,
      reprocess_fraud: true,
    },
    guard(summary) {
      if (!summary?.has_classifications) {
        return {
          allowed: false,
          reason: 'Ejecuta primero la fase 1.4 para reclasificar documentos.',
        };
      }
      return { allowed: true };
    },
  },
  {
    id: 'phase3',
    title: 'Consolidación',
    description: 'Regenera el consolidado principal sin recalcular fraude.',
    badge: 'Fase 3',
    options: {
      reprocess_consolidation: true,
      reprocess_fraud: true,
    },
    guard(summary) {
      if (!summary?.has_extraction) {
        return {
          allowed: false,
          reason: 'Es necesario contar con extracciones vigentes (fase 2).',
        };
      }
      return { allowed: true };
    },
  },
  {
    id: 'phase35',
    title: 'Análisis de fraude',
    description: 'Ejecuta únicamente la fase 3.5 utilizando cache docless.',
    badge: 'Fase 3.5',
    options: {
      reprocess_fraud: true,
    },
    guard(summary) {
      if (!summary?.has_extraction) {
        return {
          allowed: false,
          reason: 'Requiere extracciones previas para modo docless (Better Practices §1).',
        };
      }
      return { allowed: true };
    },
  },

];

const STATUS_LABELS = {
  pass: 'OK',
  fail: 'Falla',
  needs_context: 'Contexto',
  error: 'Error',
  unsupported: 'No soportado',
};

const SEVERITY_LABELS = {
  low: 'Baja',
  medium: 'Media',
  high: 'Alta',
  critical: 'Crítica',
};

function clampZoomIndex(index) {

  return Math.max(0, Math.min(index, ZOOM_LEVELS.length - 1));
}

function applyZoom(index, { persist = true } = {}) {
  const clamped = clampZoomIndex(index);
  const level = ZOOM_LEVELS[clamped];
  state.zoomIndex = clamped;
  if (persist) {
    storeZoomIndex(clamped);
  }
  if (elements.zoomLabel) {
    elements.zoomLabel.textContent = `${Math.round(level * 100)}%`;
  }
  if (elements.reportFrameInner) {
    elements.reportFrameInner.style.transform = `scale(${level})`;
    const dimension = (1 / level) * 100;
    elements.reportFrameInner.style.width = `${dimension}%`;
    elements.reportFrameInner.style.height = `${dimension}%`;
  }
  if (elements.reportFrameViewport) {
    elements.reportFrameViewport.scrollTo({ top: 0, left: 0 });
  }
}

function changeZoom(delta) {
  const target = clampZoomIndex(state.zoomIndex + delta);
  applyZoom(target);
}

function initializeZoomControls() {
  if (state.zoomInitialized) {
    applyZoom(state.zoomIndex, { persist: false });
    return;
  }
  if (elements.zoomOut) {
    elements.zoomOut.addEventListener('click', () => changeZoom(-1));
  }
  if (elements.zoomIn) {
    elements.zoomIn.addEventListener('click', () => changeZoom(1));
  }
  applyZoom(state.zoomIndex, { persist: false });
  state.zoomInitialized = true;
}

async function bootstrap(options = {}) {
  const { loadDocuments = false } = options;
  if (!caseId) {
    return;
  }

  try {
    const response = await fetch(`/api/editor/${encodeURIComponent(caseId)}/bootstrap`);
    if (!response.ok) {
      throw new Error('No se pudo cargar la información del caso.');
    }

    state.bootstrap = await response.json();
    renderCaseSummary(state.bootstrap);
    renderReprocessCards(state.bootstrap);
    bindActions(state.bootstrap);
    state.reportEdit.hasManualOverride = Boolean(state.bootstrap.report_manual_override?.has_manual_html);
    updateReportEditButtons();
    hydrateDecisionChips(state.bootstrap);
    initRick(state.bootstrap);
    await loadCorrelations();
    applyZoom(state.zoomIndex, { persist: false });

    if (loadDocuments) {
      await refreshFraudDocuments({ initial: true, silent: true });
    }

    if (state.bootstrap.active_reprocess?.process_id) {
      monitorProgress(state.bootstrap.active_reprocess.process_id, {
        resume: true,
      });
    }
  } catch (error) {
    console.error('Error durante bootstrap del editor:', error);
    showToast(error.message || 'No fue posible inicializar el editor.');
  }
}

function renderCaseSummary(payload) {
  const summary = payload?.summary || {};
  const title = summary.claim_number || summary.case_id || caseId;
  if (elements.caseTitle) {
    elements.caseTitle.textContent = title;
  }
  if (elements.caseSubtitle) {
    const insured = summary.insured_name ? `Asegurado: ${summary.insured_name}` : '';
    const docs = Number.isFinite(summary.total_documents) ? `${summary.total_documents} documentos` : '';
    elements.caseSubtitle.textContent = [insured, docs].filter(Boolean).join(' · ');
  }

  if (!elements.caseMeta) {
    return;
  }

  const meta = [
    ['ID del caso', summary.case_id || caseId],
    ['Número de siniestro', summary.claim_number || '—'],
    ['Asegurado', summary.insured_name || '—'],
    ['Procesado', formatDate(summary.processed_at) || '—'],
  ];

const STATUS_LABELS = {
  pass: 'OK',
  fail: 'Falla',
  needs_context: 'Contexto',
  error: 'Error',
  unsupported: 'No soportado',
};

const SEVERITY_LABELS = {
  low: 'Baja',
  medium: 'Media',
  high: 'Alta',
  critical: 'Crítica',
};

  const fragment = document.createElement('div');
  meta.forEach(([label, value]) => {
    const dl = document.createElement('dl');
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = label;
    dd.textContent = value;
    dl.append(dt, dd);
    fragment.append(dl);
  });

  elements.caseMeta.replaceChildren(fragment);
}
function bindCorrelationFilters() {
  if (elements.correlationStatusFilter && !elements.correlationStatusFilter.dataset.bound) {
    elements.correlationStatusFilter.addEventListener('change', (event) => {
      state.correlations.filters.status = event.target.value;
      renderCorrelationPanel();
    });
    elements.correlationStatusFilter.dataset.bound = 'true';
  }
  if (elements.correlationSeverityFilter && !elements.correlationSeverityFilter.dataset.bound) {
    elements.correlationSeverityFilter.addEventListener('change', (event) => {
      state.correlations.filters.severity = event.target.value;
      renderCorrelationPanel();
    });
    elements.correlationSeverityFilter.dataset.bound = 'true';
  }
}

function hydrateCorrelationFilters() {
  if (!elements.correlationStatusFilter || !elements.correlationSeverityFilter) {
    return;
  }
  const findings = state.correlations.findings || [];
  const statusCounts = new Map();
  const severityCounts = new Map();
  findings.forEach((item) => {
    const statusKey = String(item.status || '').toLowerCase();
    const severityKey = String(item.severity || '').toLowerCase();
    if (statusKey) {
      statusCounts.set(statusKey, (statusCounts.get(statusKey) || 0) + 1);
    }
    if (severityKey) {
      severityCounts.set(severityKey, (severityCounts.get(severityKey) || 0) + 1);
    }
  });

  const currentStatus = state.correlations.filters.status || 'all';
  const statusOptions = ['all', ...statusCounts.keys()];
  elements.correlationStatusFilter.innerHTML = statusOptions
    .map((value) => {
      if (value === 'all') {
        return '<option value="all">Todos</option>';
      }
      const label = STATUS_LABELS[value] || value;
      const count = statusCounts.get(value) || 0;
      return `<option value="${value}">${label} (${count})</option>`;
    })
    .join('');
  elements.correlationStatusFilter.value = currentStatus;

  const currentSeverity = state.correlations.filters.severity || 'all';
  const severityOptions = ['all', ...severityCounts.keys()];
  elements.correlationSeverityFilter.innerHTML = severityOptions
    .map((value) => {
      if (value === 'all') {
        return '<option value="all">Todas</option>';
      }
      const label = SEVERITY_LABELS[value] || value;
      const count = severityCounts.get(value) || 0;
      return `<option value="${value}">${label} (${count})</option>`;
    })
    .join('');
  elements.correlationSeverityFilter.value = currentSeverity;
}

function applyCorrelationFilters(findings) {
  const filters = state.correlations.filters || { status: 'all', severity: 'all' };
  const statusFilter = (filters.status || 'all').toLowerCase();
  const severityFilter = (filters.severity || 'all').toLowerCase();

  return (findings || []).filter((item) => {
    const statusValue = String(item.status || '').toLowerCase();
    if (statusFilter !== 'all' && statusValue !== statusFilter) {
      return false;
    }
    const severityValue = String(item.severity || '').toLowerCase();
    if (severityFilter !== 'all' && severityValue !== severityFilter) {
      return false;
    }
    return true;
  });
}

function formatStatusLabel(value) {
  const key = String(value || '').toLowerCase();
  return STATUS_LABELS[key] || key || '—';
}

function formatSeverityLabel(value) {
  const key = String(value || '').toLowerCase();
  return SEVERITY_LABELS[key] || key || '—';
}

async function loadCorrelations() {
  if (!caseId || !elements.correlationPanel) {
    return;
  }
  try {
    const resp = await fetch(`/api/case/${encodeURIComponent(caseId)}/correlations`);
    if (!resp.ok) {
      throw new Error('No se pudo obtener el detalle de correlaciones.');
    }
    const payload = await resp.json();
    const filters = state.correlations?.filters || { status: 'all', severity: 'all' };
    state.correlations = {
      filters,
      findings: Array.isArray(payload.findings) ? payload.findings : [],
      counts: payload.counts || {},
      summary: payload.summary || {},
    };
    bindCorrelationFilters();
    renderCorrelationPanel();
  } catch (error) {
    console.warn('No se pudieron cargar correlaciones', error);
    state.correlations.findings = [];
    state.correlations.counts = {};
    state.correlations.summary = {};
    state.correlations.error = error?.message || 'No fue posible cargar correlaciones.';
    renderCorrelationPanel();
  }
}

function renderCorrelationPanel() {
  if (!elements.correlationPanel) {
    return;
  }
  const { findings, counts, filters, error } = state.correlations;

  hydrateCorrelationFilters();

  if (elements.correlationBadge) {
    const criticalCount = counts?.fail ?? 0;
    elements.correlationBadge.textContent = `${criticalCount} fallas`;
  }

  if (elements.correlationSummary) {
    const total = counts?.total ?? (findings ? findings.length : 0);
    const failCount = counts?.fail ?? 0;
    const needsContext = counts?.needs_context ?? 0;
    const metrics = [
      { label: 'Total', value: total },
      { label: 'Fallas', value: failCount },
      { label: 'Necesita contexto', value: needsContext },
    ];
    elements.correlationSummary.innerHTML = metrics
      .map((metric) => `
        <div class="correlation-metric">
          <span>${metric.label}</span>
          <strong>${metric.value}</strong>
        </div>
      `)
      .join('');
  }

  const table = elements.correlationTable;
  const empty = elements.correlationEmpty;

  if (error) {
    if (table) {
      table.innerHTML = '';
    }
    if (empty) {
      empty.textContent = error;
      empty.hidden = false;
    }
    return;
  }

  const filtered = applyCorrelationFilters(findings || []);

  if (!filtered.length) {
    if (table) {
      table.innerHTML = '';
    }
    if (empty) {
      empty.textContent = 'Sin hallazgos para mostrar.';
      empty.hidden = false;
    }
    return;
  }

  if (table) {
    const rows = filtered.map((item) => {
      const ruleId = escapeHtml(item.rule_id || 'Regla');
      const summary = escapeHtml(item.summary || '');
      const documents = Array.isArray(item.documents_involved) && item.documents_involved.length
        ? escapeHtml(item.documents_involved.join(', '))
        : '—';
      const statusLabel = escapeHtml(formatStatusLabel(item.status));
      const severityLabel = escapeHtml(formatSeverityLabel(item.severity));
      return `
        <div class="correlation-row">
          <div>
            <h3>${ruleId}</h3>
            <p>${summary}</p>
            <p>Docs: ${documents}</p>
          </div>
          <div><span class="correlation-chip">${statusLabel}</span></div>
          <div><span class="correlation-chip">${severityLabel}</span></div>
        </div>
      `;
    });
    table.innerHTML = rows.join('');
  }

  if (empty) {
    empty.hidden = true;
  }
}

function renderReprocessCards(payload) {
  if (!elements.reprocessCards) {
    return;
  }
  const summary = payload?.summary || {};
  const cards = REPROCESS_TASKS.map(task => buildReprocessCard(task, summary));
  elements.reprocessCards.replaceChildren(...cards);
}

function buildReprocessCard(task, summary) {
  const card = document.createElement('article');
  card.className = 'reprocess-card';
  const header = document.createElement('div');
  header.className = 'reprocess-card-header';
  const title = document.createElement('h3');
  title.textContent = task.title;
  header.append(title);

  const desc = document.createElement('p');
  desc.textContent = task.description;

  const badge = document.createElement('span');
  badge.className = 'reprocess-card-badge';
  badge.textContent = task.badge;

  const action = document.createElement('button');
  action.type = 'button';
  action.textContent = 'Ejecutar';
  action.addEventListener('click', () => triggerReprocess(task.options, task));

  let allowed = { allowed: true };
  if (typeof task.guard === 'function') {
    allowed = task.guard(summary) || { allowed: true };
  }

  if (!allowed.allowed) {
    card.classList.add('disabled');
    action.disabled = true;
    const warning = document.createElement('p');
    warning.className = 'reprocess-warning';
    warning.textContent = allowed.reason || 'Reprocesamiento no disponible.';
    card.append(header, desc, badge, warning, action);
    return card;
  }

  card.append(header, desc, badge, action);
  return card;
}

function initializeTabs() {
  if (!elements.editorTabs || elements.editorTabs.dataset.bound) {
    return;
  }
  elements.editorTabs.addEventListener('click', (event) => {
    const button = event.target.closest('[data-tab]');
    if (!button) {
      return;
    }
    const targetTab = button.dataset.tab;
    if (targetTab) {
      setActiveTab(targetTab);
    }
  });
  elements.editorTabs.dataset.bound = 'true';
  setActiveTab(state.activeTab, { force: true });
}

function setActiveTab(tab, { force = false } = {}) {
  if (!tab) {
    return;
  }
  if (!force && state.activeTab === tab) {
    return;
  }
  state.activeTab = tab;
  if (tab !== 'report' && state.reportEdit.isEditing) {
    exitReportEditMode({ restoreOriginal: true });
  }
  if (elements.editorMain) {
    elements.editorMain.dataset.activeTab = tab;
    elements.editorMain.classList.toggle('is-report-active', tab === 'report');
  }

  if (elements.editorTabs) {
    const buttons = elements.editorTabs.querySelectorAll('[data-tab]');
    buttons.forEach((btn) => {
      btn.classList.toggle('is-active', btn.dataset.tab === tab);
    });
  }

  const container = elements.editorMain || document;
  container.querySelectorAll('[data-tab-panel]').forEach((panel) => {
    const isTarget = panel.dataset.tabPanel === tab;
    panel.hidden = !isTarget;
    panel.classList.toggle('is-active', isTarget);
  });

  if (tab === 'report') {
    applyZoom(state.zoomIndex, { persist: false });
  }
}

function bindFraudDocumentEvents() {
  if (elements.fraudDocumentsList && !elements.fraudDocumentsList.dataset.bound) {
    elements.fraudDocumentsList.addEventListener('click', onFraudDocumentsClick);
    elements.fraudDocumentsList.dataset.bound = 'true';
  }
}

function bindReportEditControls() {
  if (elements.btnEditReport && !elements.btnEditReport.dataset.bound) {
    elements.btnEditReport.addEventListener('click', enterReportEditMode);
    elements.btnEditReport.dataset.bound = 'true';
  }

  if (elements.btnSaveReport && !elements.btnSaveReport.dataset.bound) {
    elements.btnSaveReport.addEventListener('click', handleSaveReportEdit);
    elements.btnSaveReport.dataset.bound = 'true';
  }

  if (elements.btnCancelEdit && !elements.btnCancelEdit.dataset.bound) {
    elements.btnCancelEdit.addEventListener('click', handleCancelReportEdit);
    elements.btnCancelEdit.dataset.bound = 'true';
  }

  if (elements.btnResetReport && !elements.btnResetReport.dataset.bound) {
    elements.btnResetReport.addEventListener('click', handleResetReportEdit);
    elements.btnResetReport.dataset.bound = 'true';
  }

  updateReportEditButtons();
}

function updateReportEditButtons() {
  const editing = state.reportEdit.isEditing;
  const hasManual = state.reportEdit.hasManualOverride;

  if (elements.btnEditReport) {
    elements.btnEditReport.hidden = editing;
  }
  if (elements.btnSaveReport) {
    elements.btnSaveReport.hidden = !editing;
    elements.btnSaveReport.disabled = false;
  }
  if (elements.btnCancelEdit) {
    elements.btnCancelEdit.hidden = !editing;
    elements.btnCancelEdit.disabled = false;
  }
  if (elements.btnResetReport) {
    elements.btnResetReport.hidden = editing || !hasManual;
    elements.btnResetReport.disabled = editing;
  }

  if (elements.editorMain) {
    elements.editorMain.classList.toggle('is-report-editing', editing);
  }
}

function enterReportEditMode() {
  const doc = getReportDocument();
  if (!doc) {
    showToast('Carga primero la vista final antes de editar.');
    return;
  }
  if (state.reportEdit.isEditing) {
    return;
  }
  state.reportEdit.isEditing = true;
  const serializer = new XMLSerializer();
  state.reportEdit.originalHTML = `<!DOCTYPE html>${serializer.serializeToString(doc.documentElement)}`;
  try {
    doc.designMode = 'on';
  } catch (error) {
    console.warn('No se pudo activar designMode:', error);
  }
  if (doc.body) {
    doc.body.contentEditable = 'true';
    doc.body.classList.add('editor-report-editing');
    doc.body.focus();
  }
  updateReportEditButtons();
  showToast('Modo edición activo. Realiza tus cambios y guarda.');
}

function exitReportEditMode({ restoreOriginal = false } = {}) {
  if (!state.reportEdit.isEditing) {
    state.reportEdit.originalHTML = null;
    updateReportEditButtons();
    return;
  }
  const doc = getReportDocument();
  if (doc) {
    try {
      doc.designMode = 'off';
    } catch (error) {
      console.warn('No se pudo desactivar designMode:', error);
    }
    if (doc.body) {
      doc.body.contentEditable = 'false';
      doc.body.classList.remove('editor-report-editing');
    }
    if (restoreOriginal && state.reportEdit.originalHTML) {
      doc.open();
      doc.write(state.reportEdit.originalHTML);
      doc.close();
      applyZoom(state.zoomIndex, { persist: false });
    }
  }
  state.reportEdit.isEditing = false;
  state.reportEdit.originalHTML = null;
  updateReportEditButtons();
}

async function handleSaveReportEdit() {
  const doc = getReportDocument();
  if (!doc) {
    showToast('No se pudo acceder al reporte para guardar.');
    return;
  }
  const serializer = new XMLSerializer();
  const html = `<!DOCTYPE html>${serializer.serializeToString(doc.documentElement)}`;
  if (elements.btnSaveReport) {
    elements.btnSaveReport.disabled = true;
  }
  try {
    await saveManualReportHtml(html);
    showToast('Reporte personalizado guardado.');
    exitReportEditMode();
    refreshReport();
  } catch (error) {
    console.error('Error guardando el reporte editado', error);
    showToast(error.message || 'No se pudo guardar el reporte editado.');
  } finally {
    if (elements.btnSaveReport) {
      elements.btnSaveReport.disabled = false;
    }
  }
}


async function handleResetReportEdit() {
  if (!state.reportEdit.hasManualOverride) {
    return;
  }
  if (!confirm('Se restaurará el reporte automático y se perderán los cambios manuales. ¿Continuar?')) {
    return;
  }
  if (elements.btnResetReport) {
    elements.btnResetReport.disabled = true;
  }
  try {
    await deleteManualReportHtml();
    showToast('Reporte restaurado a su versión automática.');
    if (state.reportEdit.isEditing) {
      exitReportEditMode({ restoreOriginal: false });
    }
    refreshReport();
    updateReportEditButtons();
  } catch (error) {
    console.error('Error restaurando el reporte automático', error);
    showToast(error.message || 'No se pudo restaurar el reporte automático.');
  } finally {
    if (elements.btnResetReport) {
      elements.btnResetReport.disabled = false;
    }
  }
}

function handleCancelReportEdit() {
  exitReportEditMode({ restoreOriginal: true });
  showToast('Edición cancelada.');
}

function onFraudDocumentsClick(event) {
  const action = event.target.closest('[data-action]');
  const card = event.target.closest('[data-document-id]');
  if (!card) {
    return;
  }
  const documentId = card.dataset.documentId;
  if (!documentId) {
    return;
  }

  if (action) {
    const actionType = action.dataset.action;
    if (actionType === 'toggle') {
      const nextInclude = action.getAttribute('aria-pressed') !== 'true';
      handleDocumentToggle(documentId, nextInclude, { control: action });
      return;
    }
    if (actionType === 'reprocess') {
      handleDocumentReprocess(documentId, { button: action });
      return;
    }
  }

  const header = event.target.closest('.fraud-doc-header');
  if (header && card.classList.contains('is-hidden')) {
    card.classList.toggle('is-collapsed');
  }
}

async function handleDocumentToggle(documentId, include, { control } = {}) {
  if (!caseId || !documentId) {
    return;
  }

  if (control) {
    control.disabled = true;
  }

  try {
    const resp = await fetch(`/api/editor/${encodeURIComponent(caseId)}/fraud-documents/${encodeURIComponent(documentId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ include_in_report: include }),
    });
    if (!resp.ok) {
      const payload = await safeJson(resp);
      throw new Error(payload?.detail || 'No se pudo actualizar la visibilidad del documento.');
    }
    const payload = await resp.json();
    if (state.bootstrap) {
      state.bootstrap.fraud_documents_preview = payload?.fraud_documents_preview || state.bootstrap.fraud_documents_preview;
    }
    showToast(payload?.include_in_report ? 'Documento marcado para reporte.' : 'Documento oculto del reporte.');
    await refreshFraudDocuments({ silent: true });
    refreshReport();
  } catch (error) {
    console.error('Error actualizando visibilidad de documento', error);
    showToast(error.message || 'No fue posible actualizar la visibilidad.');
  } finally {
    if (control) {
      control.disabled = false;
    }
  }
}

async function handleDocumentReprocess(documentId, { button } = {}) {
  if (!caseId || !documentId) {
    return;
  }

  if (button) {
    button.disabled = true;
  }

  try {
    const resp = await fetch(`/api/editor/${encodeURIComponent(caseId)}/fraud-documents/${encodeURIComponent(documentId)}/reprocess`, {
      method: 'POST',
    });
    if (!resp.ok) {
      const payload = await safeJson(resp);
      throw new Error(payload?.detail || 'No se pudo iniciar el reproceso individual.');
    }
    const data = await resp.json();
    const processId = data?.process_id;
    if (!processId) {
      throw new Error('Respuesta inválida: no se recibió process_id.');
    }
    showToast('Reproceso individual iniciado.');
    setDocumentPending(documentId, true);
    monitorProgress(processId, {
      label: 'Reprocesando documento',
      context: { type: 'fraud-document', documentId },
    });
  } catch (error) {
    console.error('Error reprocesando documento', error);
    showToast(error.message || 'No fue posible iniciar el reproceso individual.');
    setDocumentPending(documentId, false);
  } finally {
    if (button) {
      button.disabled = false;
    }
  }
}

function setDocumentPending(documentId, pending) {
  if (!documentId) {
    return;
  }
  if (pending) {
    state.fraudDocumentLocks.add(documentId);
  } else {
    state.fraudDocumentLocks.delete(documentId);
  }

  const card = findDocumentCard(documentId);
  if (!card) {
    return;
  }
  card.classList.toggle('is-disabled', pending);
  const toggle = card.querySelector('[data-action="toggle"]');
  if (toggle) {
    toggle.disabled = pending;
  }
  const button = card.querySelector('[data-action="reprocess"]');
  if (button) {
    const defaultLabel = button.dataset.defaultLabel || 'Reprocesar';
    button.disabled = pending;
    if (pending) {
      button.innerHTML = '<i class="ri-time-line"></i><span>En reproceso…</span>';
    } else {
      button.innerHTML = `<i class="ri-refresh-line"></i><span>${escapeHtml(defaultLabel)}</span>`;
    }
  }
}

function findDocumentCard(documentId) {
  if (!elements.fraudDocumentsList || !documentId) {
    return null;
  }
  const selector = `[data-document-id="${escapeSelector(documentId)}"]`;
  return elements.fraudDocumentsList.querySelector(selector);
}

function renderFraudDocuments(catalog) {
  const container = elements.fraudDocumentsList;
  if (!container) {
    return;
  }
  const docs = Array.isArray(catalog?.documents) ? catalog.documents : [];
  if (elements.fraudDocumentsCount) {
    const visibleCount = docs.filter((item) => item?.analysis?.include_in_report !== false).length;
    elements.fraudDocumentsCount.textContent = docs.length
      ? `${visibleCount}/${docs.length} visibles`
      : '0 documentos';
  }

  if (!docs.length) {
    container.innerHTML = '';
    if (elements.fraudDocumentsEmpty) {
      elements.fraudDocumentsEmpty.hidden = false;
    }
    return;
  }

  const cards = docs
    .map((entry, index) => composeDocumentCard(entry, index))
    .filter(Boolean)
    .join('');

  container.innerHTML = cards;
  if (elements.fraudDocumentsEmpty) {
    elements.fraudDocumentsEmpty.hidden = true;
  }
}

function composeDocumentCard(entry, index = 0) {
  const analysis = entry?.analysis || {};
  const metadata = entry?.metadata || {};
  const documentId = analysis.document_id || metadata.document_id || `doc-${index}`;
  const documentName = analysis.document_name || documentId;
  const include = analysis.include_in_report !== false;
  const riskKey = normalizeRiskKey(analysis.risk_level);
  const riskLabel = formatRiskLabel(riskKey);
  const scoreLabel = formatScore(analysis.fraud_score);
  const docType = (analysis.document_type || 'otro').toLowerCase();
  const lastRun = formatDate(metadata.last_run || metadata.updated_at || metadata.created_at || analysis.timestamp) || 'Sin registros recientes';
  const model = analysis.analysis_model || '—';
  const guide = analysis.guide_version || '—';
  const pending = state.fraudDocumentLocks.has(documentId);
  const classes = ['fraud-doc-card'];
  if (!include) {
    classes.push('is-hidden', 'is-collapsed');
  }
  if (pending) {
    classes.push('is-disabled');
  }

  const toggleIcon = include ? 'ri-checkbox-circle-line' : 'ri-indeterminate-circle-line';
  const toggleLabel = include ? 'Visible en reporte' : 'Oculto del reporte';
  const buttonIcon = pending ? 'ri-time-line' : 'ri-refresh-line';
  const buttonLabel = pending ? 'En reproceso…' : 'Reprocesar';
  const toggleDisabled = pending ? ' disabled' : '';
  const buttonDisabled = pending ? ' disabled' : '';
  const safeToggleIcon = escapeHtml(toggleIcon);
  const safeButtonIcon = escapeHtml(buttonIcon);

  return `
    <article class="${classes.join(' ')}" data-document-id="${escapeHtml(documentId)}" data-include="${include}">
      <div class="fraud-doc-header">
        <button type="button" class="fraud-toggle" data-action="toggle" aria-pressed="${include}"${toggleDisabled}>
          <i class="${safeToggleIcon}"></i>
          <span>${escapeHtml(toggleLabel)}</span>
        </button>
        <div class="fraud-doc-info">
          <h3>${escapeHtml(documentName)}</h3>
          <p>Tipo: ${escapeHtml(docType)}</p>
        </div>
        <div class="fraud-doc-risk">
          <span class="risk-badge risk-${riskKey}">${escapeHtml(riskLabel)}</span>
          <span class="fraud-doc-score">Score: ${escapeHtml(scoreLabel)}</span>
        </div>
      </div>
      <div class="fraud-doc-body">
        <div class="fraud-doc-meta">
          <span><i class="ri-time-line"></i>${escapeHtml(lastRun)}</span>
          <span><i class="ri-cpu-line"></i>Modelo: ${escapeHtml(model)}</span>
          <span><i class="ri-compass-3-line"></i>Guía: ${escapeHtml(guide)}</span>
        </div>
        <div class="fraud-doc-actions">
          <button type="button" class="fraud-doc-action secondary" data-action="reprocess" data-default-label="Reprocesar"${buttonDisabled}>
            <i class="${safeButtonIcon}"></i>
            <span>${escapeHtml(buttonLabel)}</span>
          </button>
        </div>
      </div>
    </article>
  `;
}

function normalizeRiskKey(value) {
  const key = String(value || '').toLowerCase();
  if (['bajo', 'medio', 'alto', 'critico'].includes(key)) {
    return key;
  }
  return 'medio';
}

function formatRiskLabel(value) {
  const labels = {
    bajo: 'Bajo',
    medio: 'Medio',
    alto: 'Alto',
    critico: 'Crítico',
  };
  return labels[value] || 'Medio';
}

function formatScore(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return `${Math.round(Math.max(0, Math.min(value, 1)) * 100)}%`;
}

async function refreshFraudDocuments({ initial = false, silent = false } = {}) {
  if (!caseId) {
    return;
  }
  try {
    const resp = await fetch(`/api/editor/${encodeURIComponent(caseId)}/fraud-documents`);
    if (!resp.ok) {
      throw new Error('No se pudo cargar el catálogo de fraude.');
    }
    const catalog = await resp.json();
    state.fraudCatalog = catalog;
    if (state.bootstrap) {
      state.bootstrap.fraud_documents_preview = catalog?.preview || state.bootstrap.fraud_documents_preview;
    }
    renderFraudDocuments(catalog);
  } catch (error) {
    console.error('Error cargando documentos de fraude', error);
    if (!silent) {
      showToast(error.message || 'No fue posible cargar los documentos de fraude.');
    }
  }
}

async function saveManualReportHtml(html) {
  const resp = await fetch(`/api/editor/${encodeURIComponent(caseId)}/report-html`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ html }),
  });
  if (!resp.ok) {
    const payload = await safeJson(resp);
    throw new Error(payload?.detail || 'No se pudo guardar el reporte editado.');
  }
  state.reportEdit.hasManualOverride = true;
}

async function deleteManualReportHtml() {
  const resp = await fetch(`/api/editor/${encodeURIComponent(caseId)}/report-html`, {
    method: 'DELETE',
  });
  if (!resp.ok) {
    const payload = await safeJson(resp);
    throw new Error(payload?.detail || 'No se pudo restaurar el reporte automático.');
  }
  state.reportEdit.hasManualOverride = false;
}

function escapeHtml(value) {
  if (value === null || value === undefined) {
    return '';
  }
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escapeSelector(value) {
  if (window.CSS && typeof window.CSS.escape === 'function') {
    return window.CSS.escape(value);
  }
  return String(value).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
}

function getReportDocument() {
  if (!elements.reportFrame) {
    return null;
  }
  try {
    return elements.reportFrame.contentDocument || null;
  } catch (error) {
    console.error('No se pudo acceder al documento del reporte:', error);
    return null;
  }
}

function bindActions(payload) {
  if (elements.btnOpenFull) {
    const reportUrl = payload?.report_url || `/report/${encodeURIComponent(caseId)}`;
    elements.btnOpenFull.dataset.url = reportUrl;
    if (!elements.btnOpenFull.dataset.bound) {
      elements.btnOpenFull.addEventListener('click', () => {
        const target = elements.btnOpenFull.dataset.url || reportUrl;
        window.open(target, '_blank');
      });
      elements.btnOpenFull.dataset.bound = 'true';
    }
  }

  if (elements.btnDownloadPdf) {
    const pdfUrl = payload?.pdf_url || `/api/editor/${encodeURIComponent(caseId)}/report/pdf`;
    elements.btnDownloadPdf.dataset.url = pdfUrl;
    if (!elements.btnDownloadPdf.dataset.bound) {
      elements.btnDownloadPdf.addEventListener('click', () => {
        const target = elements.btnDownloadPdf.dataset.url || pdfUrl;
        window.open(target, '_blank');
      });
      elements.btnDownloadPdf.dataset.bound = 'true';
    }
  }

  if (elements.btnDeleteCase && !elements.btnDeleteCase.dataset.bound) {
    elements.btnDeleteCase.addEventListener('click', async () => {
      if (!confirm('Esta acción eliminará el siniestro y todos sus artefactos. ¿Continuar?')) {
        return;
      }
      try {
        const resp = await fetch(`/replay/api/deep-purge/${encodeURIComponent(caseId)}`, {
          method: 'DELETE',
        });
        if (!resp.ok) {
          throw new Error('No se pudo eliminar el caso.');
        }
        showToast('Caso eliminado correctamente.');
        window.location.href = '/';
      } catch (error) {
        console.error('Error eliminando caso', error);
        showToast(error.message || 'Error eliminando el caso.');
      }
    });
    elements.btnDeleteCase.dataset.bound = 'true';
  }

  if (elements.reprocessAbort && !elements.reprocessAbort.dataset.bound) {
    elements.reprocessAbort.addEventListener('click', async () => {
      if (!state.activeProcessId) {
        return;
      }
      try {
        const resp = await fetch(`/cancel/${encodeURIComponent(state.activeProcessId)}`, {
          method: 'POST',
        });
        if (!resp.ok) {
          throw new Error('No se pudo cancelar el reproceso.');
        }
        showToast('Reproceso cancelado.');
        stopProgress();
      } catch (error) {
        console.error('Error cancelando reproceso', error);
        showToast(error.message || 'No fue posible cancelar el reproceso.');
      }
    });
    elements.reprocessAbort.dataset.bound = 'true';
  }

  elements.decisionButtons.forEach(btn => {
    if (!btn.dataset.bound) {
      btn.addEventListener('click', () => setDecision(btn.dataset.decision));
      btn.dataset.bound = 'true';
    }
  });

  if (elements.reportFrame && !elements.reportFrame.dataset.zoomBound) {
    elements.reportFrame.addEventListener('load', () => applyZoom(state.zoomIndex, { persist: false }));
    elements.reportFrame.dataset.zoomBound = 'true';
  }

  initializeTabs();
  bindFraudDocumentEvents();
  bindReportEditControls();
  setActiveTab(state.activeTab, { force: true });
}


async function triggerReprocess(options, task) {
  if (!caseId) {
    return;
  }
  if (state.polling) {
    showToast('Ya existe un reproceso en ejecución.');
    return;
  }

  try {
    const resp = await fetch(`/api/case/${encodeURIComponent(caseId)}/reprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ options }),
    });
    if (!resp.ok) {
      const payload = await safeJson(resp);
      throw new Error(payload?.detail || 'Fallo al iniciar el reproceso.');
    }

    const data = await resp.json();
    const processId = data?.process_id;
    if (!processId) {
      throw new Error('Respuesta inválida: no se recibió process_id.');
    }

    showToast(`Reproceso iniciado (${task.badge}).`);
    monitorProgress(processId, { label: task.title, context: { type: 'case-reprocess', taskId: task.id } });
  } catch (error) {
    console.error('Error iniciando reproceso', error);
    showToast(error.message || 'No fue posible iniciar el reproceso.');
  }
}

function monitorProgress(processId, { label = 'Reproceso', resume = false, context = null } = {}) {
  state.activeProcessId = processId;
  updateProgress({ visible: true, progress: resume ? null : 5, message: `${label} en ejecución…` });

  if (context) {
    state.processHooks[processId] = context;
  } else if (!state.processHooks[processId]) {
    state.processHooks[processId] = { type: resume ? 'resumed' : 'generic' };
  }

  if (state.polling) {
    clearInterval(state.polling);
  }

  state.polling = setInterval(async () => {
    try {
      const resp = await fetch(`/status/${encodeURIComponent(processId)}`);
      if (!resp.ok) {
        throw new Error('No se pudo obtener el estado del reproceso.');
      }
      const status = await resp.json();
      applyProgressStatus(status);
      if (['completed', 'error', 'cancelled'].includes(status.status)) {
        clearInterval(state.polling);
        state.polling = null;
        handleProcessFinished(status, processId);
      }
    } catch (error) {
      console.error('Error monitoreando reproceso', error);
      clearInterval(state.polling);
      state.polling = null;
      showToast(error.message || 'Error monitoreando reproceso.');
    }
  }, 1200);
}

function applyProgressStatus(status) {
  updateProgress({
    visible: true,
    progress: typeof status.progress === 'number' ? status.progress : null,
    message: status.message || 'Procesando…',
  });
}

async function handleProcessFinished(status, processId) {
  const finalStatus = status?.status || 'completed';
  const message = status?.message || (finalStatus === 'completed' ? 'Procesamiento completado.' : 'Proceso finalizado.');
  const progressValue = typeof status?.progress === 'number' ? status.progress : finalStatus === 'completed' ? 100 : null;

  updateProgress({
    visible: true,
    progress: progressValue,
    message,
  });

  const context = state.processHooks?.[processId];
  const isFraudDocument = context?.type === 'fraud-document' && context.documentId;

  if (isFraudDocument) {
    setDocumentPending(context.documentId, false);
  }

  try {
    if (finalStatus === 'completed') {
      showToast(message);
      refreshReport();
      await bootstrap();
      await refreshFraudDocuments({ silent: true });
      setTimeout(() => {
        stopProgress();
      }, 1600);
    } else {
      showToast(message);
      stopProgress();
    }
  } catch (error) {
    console.error('Error al refrescar vista tras reproceso', error);
    stopProgress();
  } finally {
    if (isFraudDocument && context?.documentId) {
      state.fraudDocumentLocks.delete(context.documentId);
    }
    delete state.processHooks[processId];
  }
}

function updateProgress({ visible, progress, message }) {
  if (!elements.reprocessProgress) {
    return;
  }
  elements.reprocessProgress.hidden = !visible;
  if (typeof progress === 'number' && elements.reprocessFill) {
    elements.reprocessFill.style.width = `${Math.max(0, Math.min(progress, 100))}%`;
  }
  if (elements.reprocessLabel && message) {
    elements.reprocessLabel.textContent = message;
  }
  if (elements.reprocessAbort) {
    elements.reprocessAbort.hidden = !state.activeProcessId;
  }
}

function stopProgress() {
  state.activeProcessId = null;
  updateProgress({ visible: false });
}

function refreshReport() {
  if (!elements.reportFrame) {
    return;
  }
  if (state.reportEdit.isEditing) {
    exitReportEditMode({ restoreOriginal: false });
  }
  const url = new URL(elements.reportFrame.src, window.location.origin);
  url.searchParams.set('t', String(Date.now()));
  elements.reportFrame.src = url.toString();
  applyZoom(state.zoomIndex, { persist: false });
}

async function setDecision(decision) {
  if (!['with', 'without'].includes(decision)) {
    return;
  }
  try {
    const resp = await fetch(`/api/case/${encodeURIComponent(caseId)}/decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision }),
    });
    if (!resp.ok) {
      const payload = await safeJson(resp);
      throw new Error(payload?.detail || 'No se pudo registrar la decisión.');
    }
    const payload = await resp.json();
    hydrateDecisionChips(payload);
    showToast('Decisión guardada.');
  } catch (error) {
    console.error('Error guardando decisión', error);
    showToast(error.message || 'No fue posible guardar la decisión.');
  }
}

function hydrateDecisionChips(payload) {
  const decision = payload?.tentative_decision || payload?.summary?.tentative_decision;
  const savings = payload?.savings_amount ?? payload?.summary?.savings_amount;

  if (elements.decisionButtons.length) {
    elements.decisionButtons.forEach(btn => {
      btn.classList.toggle('is-active', btn.dataset.decision === decision);
    });
  }

  if (elements.tentativeStatus) {
    const label = decision === 'with'
      ? 'Marcado con tentativa'
      : decision === 'without'
        ? 'Marcado sin tentativa'
        : 'Sin decisión manual';
    elements.tentativeStatus.textContent = label;
    elements.tentativeStatus.classList.toggle('success', Boolean(decision));
  }

  if (elements.savingsStatus) {
    if (typeof savings === 'number' && Number.isFinite(savings) && savings !== 0) {
      elements.savingsStatus.textContent = `Ahorro estimado: ${formatCurrency(savings)}`;
      elements.savingsStatus.classList.add('success');
    } else {
      elements.savingsStatus.textContent = 'Ahorro pendiente de cálculo';
      elements.savingsStatus.classList.remove('success');
    }
  }
}

function initRick(payload) {
  if (!window.rickChat || typeof window.rickChat.init !== 'function') {
    console.warn('RickChat no disponible');
    return;
  }
  try {
    window.rickChat.init({
      caseId,
      container: elements.rickPanel,
      summary: payload?.summary,
    });
  } catch (error) {
    console.error('No se pudo inicializar el Agente Rick', error);
  }
}

function showToast(message) {
  if (!message) {
    return;
  }
  console.info(message);
}

function formatDate(value) {
  if (!value) {
    return null;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString('es-MX', {
    dateStyle: 'short',
    timeStyle: 'short',
  });
}

function formatCurrency(value) {
  try {
    return new Intl.NumberFormat('es-MX', { style: 'currency', currency: 'MXN' }).format(value);
  } catch (error) {
    return String(value);
  }
}

async function safeJson(resp) {
  try {
    return await resp.json();
  } catch (error) {
    return null;
  }
}

initializeZoomControls();
bootstrap({ loadDocuments: true });
