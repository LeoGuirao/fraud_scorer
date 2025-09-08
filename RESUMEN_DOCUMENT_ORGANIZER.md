# 📘 Resumen del Sistema de Organización de Documentos

Este documento explica cómo funciona el pipeline de organización y reporte, sus componentes principales y la integración con el motor de clasificación compartido. El enfoque es operativo (cómo trabaja), no una guía paso a paso de implementación.

## 🎯 Objetivo
- Preparar, clasificar y organizar documentos de un caso.
- Extraer y consolidar información clave con IA para generar reportes HTML/PDF.

## 🧭 Flujo del Pipeline
1. Ingesta: se recibe la carpeta del caso con los documentos.
2. OCR + Parser: procesamiento con Azure/Cache OCR para obtener texto estructurado.
3. Clasificación: usa el motor unificado (LLM-first) y, si falla, heurística.
4. Extracción IA: prompts guiados para obtener campos relevantes por tipo.
5. Consolidación: resolución de conflictos entre fuentes e inferencias.
6. Organización/Nombrado: staging temporal y/o carpeta final organizada.
7. Reportes: HTML/PDF + JSON de resultados.

## 🧩 Componentes
- OCR/Cache: `AzureOCRProcessor`, `OCRCacheManager`, `DocumentParser`.
- Clasificación: `DocumentClassifier` (wrapper) → `ClassifierEngine` (núcleo en `src/fraud_scorer/classification/engine.py`).
- Extracción/Consolidación: `AIFieldExtractor` + `AIConsolidator`.
- Generación de reportes: `AIReportGenerator`.

## 🤝 Unificación con Preview
- Tanto el preview como el pipeline comparten el mismo motor de clasificación.
- Diferencias operativas:
  - Preview: puede usar visión (PDF/imagen) para evaluar escaneos.
  - Pipeline: por defecto usa texto (visión opcional por configuración global).

## 🔍 Clasificación (Engine)
- Texto (pipeline): un llamado LLM textual por documento; valida tipo y maneja errores; heurística como respaldo.
- Visión (preview): convierte PDF/imagen a contenido visual y clasifica con el mismo contrato.
- Prompt: la guía de tipos se genera desde las definiciones canónicas del `DocumentClassifier` (keywords, must/may/exclude, descripción).

## ⚙️ Configuración
- `settings.CLASSIFICATION_CONFIG`: `llm_model`, `llm_temperature`, `llm_max_completion_tokens`, `sample_text_length`.
- `settings.CLASSIFICATION_ENGINE` (opcional):
  - `strategy`: normalmente `"llm_first"`.
  - `use_vision`: `false/true` para pipeline (visión aumenta costo/latencia y requiere PyMuPDF).

## 📂 Organización de Archivos
- Staging temporal: `data/uploads/renombre_de_documentos/<timestamp>/` con `mapping.json`.
- Carpeta final del caso: `<ASEGURADO + SINIESTRO>` tras consolidación y renombrado final.

## 🧱 Definiciones Canónicas (Categorías)
- Archivo: `src/fraud_scorer/processors/document_classifier.py`.
- Cada tipo especifica: `keywords` (coincidencias), `must_have` (obligatorios), `may_have` (opcionales), `exclude` (exclusiones) y una `description`.
- Cualquier cambio en estas definiciones impacta tanto en preview como en pipeline.

## ⏱️ Rendimiento y Costos
- Pipeline: LLM textual → latencia/costo controlados (1 llamada/doc).
- Pipeline con visión (opcional): requiere `PyMuPDF`, mayor latencia/costo por envío de imágenes.
- Recomendación: usar visión selectivamente (p. ej., para lotes escaneados).

## 🧪 Confiabilidad y Fallbacks
- Fallback: si el LLM falla o no hay credenciales, se usa la heurística del clasificador base.
- Logs de clasificación: incluyen tipo y confianza para trazabilidad.

## 🧯 Troubleshooting
- OPENAI_API_KEY ausente → se usará heurística (menor precisión).
- Azure/OCR: revisar credenciales y cache si hay fallas o latencias altas.
- Visión activada sin PyMuPDF: instalar `pymupdf` (`pip install pymupdf`).

## 🔚 Notas
- Este resumen describe la operación del sistema ya integrado al engine. Para ajustes finos, modificar definiciones de `DocumentClassifier` o parámetros en `settings`.

