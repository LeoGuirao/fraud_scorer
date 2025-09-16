# Guía de Implementación: Checkpoint de Clasificación Manual (Inyección 1.4.1)

## 📋 Resumen Ejecutivo

Se implementará un checkpoint obligatorio de revisión manual inmediatamente después de la clasificación automática (post‑OCR) y antes de continuar con el flujo actual de extracción/consolidación/reporte. Este checkpoint se inyecta como una micro‑fase dentro del pipeline existente, sin separar en nuevas “Fases 1 y 2”, sin refactorizar firmas públicas ni mover lógica entre módulos. Todo el sistema permanece igual, salvo por la inserción de una pausa controlada llamada “Fase 1.4.1: Revisión manual de clasificación de documentos”.

### Objetivos
- Pausar tras clasificación para revisión/corrección manual en la UI de `upload.html` (ya incluida).
- Reanudar desde clasificaciones corregidas sin repetir OCR.
- Mantener el comportamiento y contratos actuales del sistema.

### Principios de diseño
- No separar fases ni exponer nuevas funciones públicas del pipeline.
- Endpoints mínimos solo para lectura/actualización de clasificaciones (sin tocar otras rutas ni contratos).
- No modificar nomenclaturas de archivos ni rutas de reportes.
- La inyección 1.4.1 es una pausa operativa, no un refactor técnico.

---

## 🔄 Flujo del Sistema

### Estados del pipeline (ejemplo)
```
processing → awaiting_review (1.4.1) → processing → completed
```

### Flujo de datos
Conceptualmente:
- 1.4 (clasificación automática) produce una propuesta de tipo por documento.
- 1.4.1 (revisión manual) permite confirmar o ajustar esas propuestas en la UI.
- Luego el pipeline continúa con extracción, consolidación y reporte como hoy.

---
## 🧠 Fase 1.4.1: Revisión Manual (definición mínima)

Propósito: dar al analista un punto único para validar/ajustar la clasificación automática antes de continuar, sin alterar contratos existentes.

Características clave (no prescriptivas):
- La revisión ocurre inmediatamente después de la clasificación automática (1.4) y antes de la extracción.
- La UI de `upload.html` ya contiene una sección de revisión; se utiliza tal cual, sin añadir páginas.
- La continuidad del flujo tras la revisión se mantiene igual a la actual.

Implementación mínima requerida (sin refactor de fases):
- Persistir al finalizar 1.4 un arreglo `classified_types` en `case_index` con al menos `{ filename, document_type }`. Si se dispone, incluir `confidence` (0–1) y `reasons` (lista corta). No es obligatorio cambiar el esquema existente; los campos extra son opcionales.
- Exponer 3 endpoints mínimos para la UI (ver sección siguiente), que solo leen/escriben `classified_types` en `case_index` y señalan la reanudación. No se modifica el resto del sistema.
- Insertar una pausa breve tras 1.4 que establezca `status="awaiting_review"` en `processing_status` y quede a la espera de la señal de continuación.

---

## 🌐 Endpoints mínimos para la UI

- `GET /api/case/{case_id}/classifications`
  - Retorna `{ case_id, classifications, document_types }`.
  - `classifications`: toma `classified_types` de `case_index` y puede incluir `{ filename, document_type, confidence?, reasons? }`.
  - `document_types`: lista derivada de `DocumentType` (`value`, `label`).

- `POST /api/case/{case_id}/update-classifications`
  - Payload: `{ classifications: { "fileA.pdf": "tipo", ... } }`.
  - Valida `tipo` ∈ `DocumentType`. Actualiza `case_index.classified_types` reemplazando `document_type` por archivo. Puede añadir `manually_reviewed: true` si se desea; no es obligatorio.

- `POST /api/case/{case_id}/continue-processing`
  - Marca la señal de reanudación (en memoria o con un archivo marcador por `case_id`). No lanza otro proceso ni separa fases.

Notas:
- Estos endpoints son locales al checkpoint; no cambian contratos existentes.
- La UI actual (`upload.html`) ya consume estas rutas; este mínimo habilita la edición manual.

## ⚙️ Operación (nivel técnico acotado)

- Al terminar 1.4 (clasificación), el backend:
  - Persiste `classified_types` y establece `processing_status[process_id].status = "awaiting_review"` con el `case_id`.
  - Entra en un bucle asíncrono ligero esperando la señal de `continue` (p. ej., `await asyncio.sleep(0.5)` mientras no exista `.resume` para ese `case_id`).
- En la UI, el usuario revisa/ajusta tipos y envía `update-classifications`; luego llama `continue-processing`.
- El backend sale de la pausa, vuelve a cargar `classified_types` desde `case_index` y sobre‑escribe en memoria el `document_type` de cada documento antes de iniciar la extracción. Continúa con consolidación y reporte como hoy.

Nota: La sincronización entre UI y backend queda a criterio del equipo, reusando mecanismos existentes. Esta guía no dicta cambios técnicos.

## 🚪 Alcance y No‑Objetivos

- Solo se añaden tres endpoints mínimos para este checkpoint.
- No se cambian contratos del resto del sistema ni nomenclaturas.
- El esquema de `case_index` no es obligatorio ampliarlo; `confidence`/`reasons` son opcionales.
- No se separa el pipeline en fases públicas ni se crean nuevas firmas.

---

## 🧱 Persistencia (referencial)

- Se mantiene la persistencia actual del sistema. Si existe un registro de clasificación, puede utilizarse como apoyo visual en la UI. Esta guía no añade ni modifica campos.

---

## ✅ Ajustes respecto a versiones anteriores

- Se elimina la propuesta de separar el pipeline y cualquier mención a nuevos endpoints o cambios de esquema.
- Se mantiene la nomenclatura, contratos y rutas actuales.

---

## 🧪 Pruebas mínimas

- `GET classifications`: devuelve lista de documentos con su tipo actual (y confianza si existe).
- `POST update-classifications`: persiste cambios; un segundo `GET` refleja la actualización.
- Pausa 1.4.1: `status` pasa a `awaiting_review` y la UI entra a la sección de revisión.
- `POST continue-processing`: reanuda y el reporte se genera; los tipos usados en extracción coinciden con los corregidos.

---

## 🧭 Checklist mínimo (sin cambios de contrato)

- Documentar internamente el punto del flujo donde sucede 1.4.1.
- Asegurar que el equipo conozca cómo realizar la revisión manual en la UI actual.
- Añadir logs de “inicio/fin de revisión” si es útil para auditoría (opcional, sin alterar contratos).

---

## 🚨 Troubleshooting

- La UI no entra a revisión: confirmar que `status="awaiting_review"` se establece tras la clasificación.
- Cambios no persisten: validar que los `filename` del payload existan en `classified_types` y el `tipo` sea válido.
- No reanuda: verificar que la señal `.resume` o flag en memoria se estableció al llamar `continue-processing` y que la tarea background sigue activa.

---

## ✅ Conclusión

Esta guía introduce un checkpoint mínimo “Fase 1.4.1: Revisión manual de clasificación” sin separar el pipeline ni cambiar su estructura, contratos o esquemas. Permite validar la clasificación antes de continuar, reutilizando la UI existente y manteniendo el comportamiento actual del sistema.
