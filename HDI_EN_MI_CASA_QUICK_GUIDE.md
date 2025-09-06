## Guía Rápida — Proceso HDI EN MI CASA

Esta guía resume, de forma precisa, cómo funciona el pipeline cuando la póliza es “HDI EN MI CASA”. Para detalles ampliados, consulta `HDI_EN_MI_CASA_IMPLEMENTATION_GUIDE.md`.

**Cuándo aplica**
- Se activa si, tras el OCR, se detecta “HDI EN MI CASA” en la póliza del caso.
- Detection window: inmediatamente después del OCR y antes de la extracción (FASE 1.5).

**Prerequisitos**
- Env vars: `ENABLE_HDI_SPECIAL_RULES=true`, `OPENAI_API_KEY`, credenciales Azure OCR.
- Rutas clave:
  - Detección: `scripts/run_report.py` (FASE 1.5 dentro de `_process_with_ai`).
  - Extractor/Prompts: `src/fraud_scorer/processors/ai/ai_field_extractor.py`, `src/fraud_scorer/prompts/extraction_prompts.py`.
  - Consolidación: `src/fraud_scorer/processors/ai/ai_consolidator.py`.
  - Config: `src/fraud_scorer/settings.py`.

**1) Detección Temprana (FASE 1.5)**
- Escaneo de candidatos de póliza:
  - Normaliza nombre de archivo (NFKD, sin diacríticos) y usa heurística `_detect_document_type(...)`.
  - Si es “poliza_de_la_aseguradora”, busca “HDI EN MI CASA” en KV o texto OCR.
- Si detecta HDI:
  - Setea contexto global: `extractor.set_policy_context("HDI_EN_MI_CASA")` y `consolidator.set_policy_context("HDI_EN_MI_CASA")`.
  - Log: `HDI_DETECTION: Policy type detected: HDI_EN_MI_CASA`.

**2) Extracción Guiada con Reglas HDI**
- Documentos objetivo incluyen: `informe_preliminar_del_ajustador`, `informe_final_del_ajustador`, `poliza_de_la_aseguradora`, `carta_de_reclamacion_formal_a_la_aseguradora`.
- Prompt guiado recibe `policy_type` y añade instrucciones HDI:
  - Número de siniestro NO es 14 dígitos (ej. `3925/25 R - 4735611`).
  - Lugar de hechos en póliza: priorizar campo “UBICACIÓN DEL RIESGO”.
  - Informe final: extraer “tipo_siniestro” desde narrativa.
- Campos permitidos (mask) con ajustes HDI:
  - En póliza, se permite dinámicamente `lugar_hechos`.
- Control de tokens (para evitar respuestas incompletas):
  - Texto OCR truncado (~4000 chars), KV acotado (hasta 40), Tablas incluidas (hasta 8 tablas, 3 filas c/u, headers truncados razonablemente).
- Log: `HDI_EXTRACTION: contexto activo para extracción guiada`.

**3) Validaciones Específicas**
- `numero_siniestro` (HDI): validación flexible (barras/letras/guiones). No se fuerza a 14 dígitos.
- `numero_poliza`: formato alfanumérico con guiones/espacios (se recomienda normalizar valores con “Inciso” adjunto).

**4) Consolidación con Prioridades HDI**
- Prioridades personalizadas cuando `policy_context == HDI_EN_MI_CASA`:
  - `tipo_siniestro` → `informe_final_del_ajustador` (narrativo).
  - `lugar_hechos` → `poliza_de_la_aseguradora` (UBICACIÓN DEL RIESGO), luego `informe_preliminar_del_ajustador`.
- Fallback determinístico si el LLM recorta salida por tokens.

**5) Cache OCR — Seguridad por Caso**
- Búsqueda por nombre en cache reorganizado se restringe al `case_id` actual.
- Fuera del caso, el cache solo se considera por hash de contenido (evita “hits” falsos por nombres iguales).

**6) Logging/Monitoreo**
- Señales útiles: `HDI_DETECTION`, `HDI_EXTRACTION`, `HDI_CONTEXT`.
- `results.json` incluye `policy_type` para monitoreo.

**Criterios de aceptación**
- Detecta HDI antes de extracción (FASE 1.5) y propaga contexto.
- `numero_siniestro` preserva formato HDI (no 14 dígitos).
- `lugar_hechos` se obtiene de “UBICACIÓN DEL RIESGO” en póliza.
- `tipo_siniestro` proviene del informe final (narrativo).
- Pólizas estándar no se ven afectadas (flag desactivable).

**Comandos útiles**
- Ejecutar caso: `python scripts/run_report.py "<carpeta_del_caso>" --out data/reports --debug`
- Flag HDI: en `.env` → `ENABLE_HDI_SPECIAL_RULES=true`.

