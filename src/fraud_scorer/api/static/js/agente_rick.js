const rickChat = {
  init(config) {
    const { caseId, container } = config || {};
    if (!caseId) {
      console.warn('Agente Rick requiere caseId');
      return;
    }

    const host = container || document.getElementById('rickPanel');
    if (!host) {
      console.warn('No se encontró contenedor para Agente Rick');
      return;
    }

    host.innerHTML = '';
    host.classList.add('rick-chat');

    const header = document.createElement('header');
    header.className = 'rick-chat__header';
    header.innerHTML = `<h2>Agente Rick</h2><span class="rick-chat__subtitle">Caso ${escapeHtml(caseId)}</span>`;

    const history = document.createElement('div');
    history.className = 'rick-chat__history';

    const form = document.createElement('form');
    form.className = 'rick-chat__form';

    const textarea = document.createElement('textarea');
    textarea.placeholder = 'Pregunta lo que necesites del caso…';
    textarea.rows = 2;

    const actions = document.createElement('div');
    actions.className = 'rick-chat__actions';

    const askButton = document.createElement('button');
    askButton.type = 'submit';
    askButton.textContent = 'Consultar';

    const reindexButton = document.createElement('button');
    reindexButton.type = 'button';
    reindexButton.textContent = 'Reindexar';
    reindexButton.className = 'secondary';

    actions.append(askButton, reindexButton);
    form.append(textarea, actions);

    host.append(header, history, form);

    const state = {
      busy: false,
    };

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const question = textarea.value.trim();
      if (!question || state.busy) {
        return;
      }
      appendMessage(history, { role: 'user', content: question });
      textarea.value = '';
      await ask(caseId, question, history, state);
    });

    textarea.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        form.dispatchEvent(new Event('submit', { cancelable: true }));
      }
    });

    reindexButton.addEventListener('click', async () => {
      if (state.busy) {
        return;
      }
      if (!confirm('Esto reconstruirá el índice del caso. ¿Deseas continuar?')) {
        return;
      }
      try {
        state.busy = true;
        reindexButton.disabled = true;
        reindexButton.textContent = 'Reindexando…';
        const resp = await fetch('/api/rick/reindex', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ case_id: caseId, rebuild: true }),
        });
        if (!resp.ok) {
          throw new Error('No se pudo reindexar el caso.');
        }
        appendMessage(history, {
          role: 'system',
          content: 'Índice reconstruido exitosamente. Puedes consultar de nuevo.',
        });
      } catch (error) {
        console.error('Error reindexando caso', error);
        appendMessage(history, {
          role: 'system',
          content: error.message || 'No se pudo reindexar el caso. Intenta más tarde.',
          tone: 'error',
        });
      } finally {
        reindexButton.disabled = false;
        reindexButton.textContent = 'Reindexar';
        state.busy = false;
      }
    });
  },
};

async function ask(caseId, question, historyNode, state) {
  try {
    state.busy = true;
    const placeholderId = appendMessage(historyNode, {
      role: 'assistant',
      content: 'Pensando…',
      pending: true,
    });

    const resp = await fetch('/api/rick/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ case_id: caseId, question }),
    });

    if (!resp.ok) {
      throw new Error('No se pudo obtener respuesta.');
    }

    const payload = await resp.json();
    replaceMessage(historyNode, placeholderId, {
      role: 'assistant',
      content: payload.answer || 'No encontré información relevante en el caso.',
      sources: payload.sources,
    });
  } catch (error) {
    console.error('Error consultando a Rick', error);
    appendMessage(historyNode, {
      role: 'system',
      content: error.message || 'No fue posible consultar al agente.',
      tone: 'error',
    });
  } finally {
    state.busy = false;
  }
}

function appendMessage(history, message) {
  const id = crypto.randomUUID();
  const item = document.createElement('article');
  item.className = `rick-chat__message rick-chat__message--${message.role}`;
  item.dataset.messageId = id;

  const content = document.createElement('div');
  content.className = 'rick-chat__bubble';
  content.innerHTML = renderMarkdown(message.content);

  if (message.tone === 'error') {
    content.classList.add('rick-chat__bubble--error');
  }

  item.append(content);

  if (Array.isArray(message.sources) && message.sources.length) {
    const list = document.createElement('ul');
    list.className = 'rick-chat__sources';
    message.sources.forEach((source) => {
      const li = document.createElement('li');
      li.textContent = `${source.metadata?.document || 'Fragmento'} (${Number.parseFloat(source.score || source.similarity || 0).toFixed(2)})`;
      list.append(li);
    });
    item.append(list);
  }

  history.append(item);
  scrollHistory(history);
  return id;
}

function replaceMessage(history, id, message) {
  const node = history.querySelector(`[data-message-id="${id}"]`);
  if (!node) {
    appendMessage(history, message);
    return;
  }
  node.className = `rick-chat__message rick-chat__message--${message.role}`;
  node.innerHTML = '';

  const content = document.createElement('div');
  content.className = 'rick-chat__bubble';
  content.innerHTML = renderMarkdown(message.content);
  node.append(content);

  if (Array.isArray(message.sources) && message.sources.length) {
    const list = document.createElement('ul');
    list.className = 'rick-chat__sources';
    message.sources.forEach((source) => {
      const li = document.createElement('li');
      li.textContent = `${source.metadata?.document || 'Fragmento'} (${Number.parseFloat(source.score || source.similarity || 0).toFixed(2)})`;
      list.append(li);
    });
    node.append(list);
  }

  scrollHistory(history);
}

function scrollHistory(history) {
  if (!history) {
    return;
  }
  history.scrollTop = history.scrollHeight;
}

function renderMarkdown(raw) {
  try {
    return escapeHtml(String(raw || ''))
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  } catch (error) {
    return escapeHtml(String(raw || ''));
  }
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

window.rickChat = rickChat;

export {};
