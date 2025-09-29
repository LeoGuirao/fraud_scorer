# Guía de Implementación: Agente Rick

## 1. Objetivo
Entregar al analista un asistente que responda preguntas puntuales sobre el caso activo reutilizando los artefactos existentes del pipeline (`OCRCacheManager`, extracciones, consolidaciones y análisis de fraude) sin introducir dependencias externas innecesarias ni romper los flujos descritos en `BETTER_PRACTICES.md`.

## 2. Principios clave
- **Sin LangChain ni stacks futuristas**: usar `sentence-transformers`, `numpy`, `scikit-learn`, `openai` (ya presentes en `pyproject.toml`).
- **Indexación docless**: construir el conocimiento desde la cache reorganizada y JSONs consolidados (ver `BETTER_PRACTICES` §1 y §6). No depender de PDFs originales.
- **Persistencia local**: guardar embeddings y chunks en `data/agent_rick/{case_id}/` para evitar recalcular.
- **Auditoría obligatoria**: registrar cada pregunta/respuesta con timestamp, usuario y fuentes.
- **Respuestas factuales**: si la información no está en el contexto, el agente debe indicarlo explícitamente.

## 3. Fuentes de datos
1. **Índice del caso** (`OCRCacheManager.get_case_index(case_id, auto_reconstruct=True)`):
   - `extraction_results` (campos por documento con `source_document` y `extracted_fields`).
   - `consolidated_data.consolidated_fields` (carátula consolidada para el reporte).
   - `fraud_analyses` (indicadores y resúmenes por documento).
2. **OCR reorganizado**: archivos `ocr_results_for_*.json` en la carpeta del caso (modo docless descrito en `BETTER_PRACTICES`).
3. **Reporte HTML** (fallback): parsear con BeautifulSoup cuando falten datos estructurados.

Antes de embebedar textos, normaliza (trim, colapsa espacios, limita longitud) para evitar ruido.

## 4. Estructura del módulo
```
src/fraud_scorer/ai/
├── __init__.py
├── agente_rick.py          # Servicio principal (embeddings + Q&A)
├── agente_rick_store.py    # Persistencia npz/jsonl
└── prompts/
    └── agente_rick_system.txt
```

### 4.1 `AgentRickStore`
Encargado de:
- `data/agent_rick/{case_id}/chunks.jsonl` (texto + metadata para inspección).
- `embeddings.npz` (matriz float32 y vector con ids de chunk).
- `index_meta.json` (modelo, versión, timestamp, hash del case index) para invalidar cuando cambie el caso.
- Métodos clave: `has_index`, `is_stale(case_id, *, processed_at)`, `save_index`, `load_index`, `prune(expire_days=45)`.

### 4.2 `AgentRickService`
```python
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from fraud_scorer.api.web_interface import OCRCacheManager

class AgentRickService:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", llm_model="gpt-4o-mini", max_chunks=6):
        self.embedder = SentenceTransformer(embedding_model)
        self.llm_model = llm_model
        self.max_chunks = max_chunks
        self.store = AgentRickStore(Path("data/agent_rick"))
        self.client = AsyncOpenAI()

    async def ensure_index(self, case_id: str, processed_at: str | None = None) -> None:
        if self.store.has_index(case_id) and not self.store.is_stale(case_id, processed_at=processed_at):
            return
        payload = self._build_dataset(case_id)
        texts = [item["text"] for item in payload]
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        self.store.save_index(case_id, vectors, payload, processed_at=processed_at)

    def _build_dataset(self, case_id: str) -> List[Dict]:
        cm = OCRCacheManager()
        case = cm.get_case_index(case_id, auto_reconstruct=True) or {}
        chunks: List[Dict] = []

        # Consolidated fields
        fields = ((case.get("consolidated_data") or {}).get("consolidated_fields") or {})
        for key, value in fields.items():
            if value:
                text = f"{key.replace('_', ' ')}: {value}"
                chunks.append({"text": text, "metadata": {"source": "consolidated_fields", "field": key}})

        # Extraction results por documento
        for item in case.get("extraction_results") or []:
            source = item.get("source_document") or "documento_sin_nombre"
            extracted = item.get("extracted_fields") or {}
            for field, value in extracted.items():
                if value:
                    chunks.append({
                        "text": f"{field.replace('_', ' ')}: {value}",
                        "metadata": {"source": source, "type": "extraction_field", "field": field}
                    })

        # Fraud analyses
        for analysis in case.get("fraud_analyses") or []:
            summary = analysis.get("analisis_completo") or analysis.get("summary")
            if summary:
                chunks.append({
                    "text": summary,
                    "metadata": {
                        "source": analysis.get("document_id") or "fraud_analysis",
                        "type": "fraud_analysis"
                    }
                })

        # OCR docless (BETTER PRACTICES §1)
        case_folder = cm.get_case_folder_path(case_id, case)
        for json_file in case_folder.glob("ocr_results_for_*.json"):
            data = cm.get_cache(json_file, case_id) or {}
            text = (data.get("text") or "").strip()
            if text:
                snippet = text[:2000]
                chunks.append({
                    "text": snippet,
                    "metadata": {
                        "source": json_file.stem.replace("ocr_results_for_", ""),
                        "type": "ocr_chunk"
                    }
                })

        if not chunks:
            raise RuntimeError(
                "No hay información indexable para el caso; asegúrate de tener extracciones o consolidación previa."
            )
        return chunks

    async def ask(self, case_id: str, question: str) -> Dict:
        cm = OCRCacheManager()
        case = cm.get_case_index(case_id, auto_reconstruct=True) or {}
        await self.ensure_index(case_id, processed_at=case.get("processed_at"))
        vectors, payload = self.store.load_index(case_id)

        q_embed = self.embedder.encode([question], normalize_embeddings=True)
        scores = cosine_similarity(q_embed, vectors)[0]
        order = scores.argsort()[::-1]
        selected = [idx for idx in order if scores[idx] > 0.35][: self.max_chunks]
        context = [payload[i] for i in selected]
        prompt = self._build_prompt(question, context)

        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self._load_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_completion_tokens=220
        )

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            answer = "No encuentro esa información en el expediente." 

        confidence = 0.0
        if selected:
            confidence = min(1.0, float(sum(scores[idx] for idx in selected) / len(selected)))

        return {
            "answer": answer,
            "sources": [
                {
                    "source": ctx["metadata"].get("source"),
                    "type": ctx["metadata"].get("type"),
                    "score": float(scores[idx])
                }
                for idx, ctx in zip(selected, context)
            ],
            "confidence": confidence
        }

    def _build_prompt(self, question: str, context: List[Dict]) -> str:
        if context:
            context_block = "\n\n".join(
                f"[Fuente: {item['metadata'].get('source','desconocido')}]\n{item['text']}"
                for item in context
            )
        else:
            context_block = "(no hay fragmentos relevantes en el índice)"
        return (
            f"Contexto disponible:\n{context_block}\n\n"
            f"Pregunta: {question}\n"
            "Responde en máximo tres oraciones, citando explícitamente cuando la información no está disponible."
        )

    def _load_system_prompt(self) -> str:
        path = Path("src/fraud_scorer/ai/prompts/agente_rick_system.txt")
        return path.read_text(encoding="utf-8") if path.exists() else (
            "Eres Agente Rick, asistente para analistas de fraude."
            " Responde en español, de forma concisa y factual usando solo el contexto proporcionado."
            " Si faltan datos, dilo claramente."
        )
```

## 5. API
Agregar rutas autenticadas en `web_interface.py`:
```python
@app.post("/api/editor/{case_id}/agent/index")
async def agent_index(case_id: str):
    case = OCRCacheManager().get_case_index(case_id, auto_reconstruct=True) or {}
    await agente_rick_service.ensure_index(case_id, processed_at=case.get("processed_at"))
    return {"status": "ok"}

@app.post("/api/editor/{case_id}/agent/query")
async def agent_query(case_id: str, payload: dict = Body(...)):
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "Pregunta vacía")
    result = await agente_rick_service.ask(case_id, question)
    audit_logger.record(case_id, current_user(), question, result)
    return result

@app.get("/api/editor/{case_id}/agent/suggestions")
async def agent_suggestions(case_id: str):
    return {"suggestions": DEFAULT_QUESTIONS}
```
`DEFAULT_QUESTIONS` debe incluir las preguntas proporcionadas por el usuario (ministerio público, lugar del siniestro, testigos, antigüedad de la póliza, etc.).

## 6. Frontend (`static/js/agente_rick.js`)
- Montar UI siguiendo el wireframe (encabezado “AGENTE RICK”, chat bubbles, chips de sugerencias, barra de confianza).
- Flujo:
  1. `await fetch('/api/editor/${caseId}/agent/index', { method: 'POST' })` al cargar.
  2. `GET /agent/suggestions` para renderizar chips (cada chip dispara `askQuestion(text)`).
  3. `askQuestion()` añade mensaje del analista, muestra indicador “Rick está escribiendo…”, llama a `/agent/query` y pinta respuesta + fuentes + confianza.
- Sanitizar contenido (`textContent`) y limitar historial a ~100 mensajes (remover antiguos si es necesario).
- Guardar historial en `sessionStorage` (clave `${caseId}-agent-rick`) para restaurar tras refresh.

## 7. Seguridad y auditoría
- Endpoints protegidos por el middleware de autenticación.
- Validar longitud de la pregunta (`1 <= len <= 500`), rechazar inputs vacíos.
- Limitar tráfico: contador en memoria (por usuario) con ventana móvil de 1 hora ≤ 30 preguntas; usar `asyncio.Lock` o Redis si está disponible.
- Registrar cada interacción en `data/logs/agent_rick.jsonl`:
  ```json
  {"ts": "2024-09-15T10:32:11Z", "case_id": "CASE-2024-0010", "user": "analista", "question": "...", "answer": "...", "sources": [...]} 
  ```
  Redactar información sensible (tokens, IDs internos) antes de loggear.
- Sanitizar texto antes de enviarlo a la UI y al LLM para evitar inyección.

## 8. Mantenimiento
- `AgentRickStore.prune(expire_days=45)` para limpiar índices viejos.
- Invalidar índice cuando cambie `case_index["processed_at"]` o tras un reproceso desde el editor (llamar `ensure_index(..., processed_at=...)`).
- Versionar embeddings: si cambia `embedding_model`, borrar índices incompatibles.
- Monitorear consumo de OpenAI y añadir métricas básicas (`rick_questions_total`, `rick_avg_latency_ms`).

## 9. Pruebas recomendadas
1. Caso con información completa → lanzar todas las preguntas sugeridas y validar que las respuestas concuerden con el reporte.
2. Caso sin `extraction_results` → confirmar que se alimenta de OCR/reporte y que responde “no disponible” cuando aplica.
3. Después de un reproceso 3.5 → verificar reindex (tocar `processed_at` y confirmar que se reconstruye).
4. Simular error de OpenAI (clave inválida) → comprobar mensaje de fallback y que la UI siga operativa.
5. Test de carga ligera: 20 preguntas concurrentes para validar caching y rate limiting.

## 10. Checklist de entrega
- [ ] Directorio `data/agent_rick/{case_id}/` creado tras la primera consulta.
- [ ] Endpoints `/agent/*` protegidos y registrando auditoría.
- [ ] UI del chat alineada con el wireframe, mostrando fuentes y confianza.
- [ ] Reindex automático tras reprocesos o cambios en `processed_at`.
- [ ] Scripts de limpieza documentados y accesibles.

