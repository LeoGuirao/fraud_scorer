# Implementacion de Sistema GPS Monitor

## Contexto

Los reportes de monitoreo GPS son documentos extensos y pesados que hoy atraviesan el mismo pipeline que el resto de los archivos. Esto genera cuellos de botella en OCR, vectorizacion y correlacion, ademas de elevar costos operativos. Siguiendo los principios de `BETTER_PRACTICES.md` (cero impacto en fases previas, reaprovechamiento de cache, trazabilidad y fallback defensivo), documentamos los problemas y las opciones de solucion.

## Problemas Identificados

1. **OCR costoso y lento**
   - 200+ paginas por reporte implican tiempos de 10 minutos y costos acumulados.
   - La respuesta JSON de Azure resulta masiva y presiona memoria/disco.

2. **Vectorizacion en Agente Rick**
   - Excede limites de tokens y tasa al fragmentar cientos de paginas.
   - Los costos de embedding escalan de forma no controlada.
   - El vector store crece de manera desproporcionada.

3. **Uso ineficiente en el motor de correlacion**
   - Buscar horarios y rutas dentro de texto plano es impreciso y lento.
   - El tabular original se pierde en el proceso de OCR.

4. **Flujo de carga unificado**
   - Todos los documentos ingresan al OCR sin discriminar.
   - No existe bypass ni tratamiento especializado para GPS.

5. **Trazabilidad en reprocesos docless**
   - Los JSON de OCR no preservan tablas completas.
   - Reprocesos 3.5 carecen de metadatos suficientes para RAG o correlacion.

## Lineamientos de diseno

- Mantener compatibilidad con docless y case_index.
- Registrar cualquier cambio en el indice del caso y en `fraud_correlations`.
- Agregar flags de fallback para volver al flujo actual si ocurre un error.
- Versionar nuevos artefactos (p.ej. `gps_data.parquet`, `gps_summary.json`).
- Agregar pruebas unitarias e integrales para cada capa introducida.

## Propuestas de Solucion

### Opcion A: Pipeline Hibrido con Bypass OCR

1. **Clasificacion temprana y bypass**
   - Extender `DocumentParser` para detectar tipos `reporte_gps` y omitir OCR cuando el archivo sea digital.
   - Generar un `ParsedDocument` especializado con metadatos equivalentes (document_id, hash, rutas) para no romper reprocesos.
   - Registrar en `case_index` un flag `gps_direct=true` y la ubicacion del dataset tabular.

2. **Extractor directo (PDF/Excel/CSV)**
   - Implementar `GPSDirectExtractor` basado en PyMuPDF, pandas o librerias existentes.
   - Normalizar el schema resultante (`timestamp`, `lat`, `lon`, `speed`, `eventos`).
   - Guardar la salida en formato Parquet y un resumen JSON para UI/LLM.
   - Incluir fallback a OCR cuando la lectura directa falle (por ejemplo, PDF escaneado).

3. **Servicio de consultas puntuales**
   - Crear `GPSQueryService` que cargue el Parquet en memoria bajo demanda.
   - Exponer metodos como `find_location_at_time` y `validate_route` empleando indices y filtros.
   - Retornar trazabilidad (indices de fila, coordenadas, confianza) para auditar la respuesta.

4. **Integracion con Agente Rick**
   - Evitar vectorizar datos brutos; incluir solo resuenes (fechas, kilometraje, ruta declarada).
   - Detectar preguntas GPS y delegar en `GPSQueryService`, devolviendo evidencia estructurada.
   - Registrar en auditoria la fuente (archivo, fila) usada en la respuesta.

5. **Motor de correlacion**
   - Ajustar `CaseContext` para cargar resuenes GPS desde `gps_summary.json`.
   - Agregar reglas especificas (p.ej. comparacion de checkpoints) usando el DataFrame y no texto plano.
   - Asegurar que el fallback degrade a `needs_context` sin romper el pipeline.

6. **Persistencia y limpieza**
   - Ampliar `OCRCacheManager` para gestionar `gps_data/` y purgar Parquet/indices en deep purge.
   - Documentar nuevas rutas en `BETTER_PRACTICES.md` y scripts de verificacion (`post_process_verifier`).

### Opcion B: OCR selectivo con Layout API

1. Continuar usando Azure pero invocar `Document Intelligence Layout` para extraer tablas con menor costo.
2. Configurar un reintento manual solo en paginas relevantes (ej. conteniendo etiquetas de ruta/tiempo).
3. Requiere ajustes minimos al pipeline actual, pero mantiene dependencia en OCR y no elimina costos.

### Opcion C: Preprocesamiento en origen

1. Solicitar al proveedor GPS archivos CSV/JSON ya normalizados.
2. Implementar un conector que consuma APIs propias del proveedor (si existen).
3. Riesgos: depende de terceros y de acuerdos contractuales; escalamiento mas lento.

### Medidas complementarias

- **Limite de tamaño**: rechazar o partir archivos >250 MB antes de subirlos.
- **Compresion y particionado**: almacenar DataFrames en Parquet particionados por fecha para lecturas rapidas.
- **Metadatos enriquecidos**: incluir distancia total, origen/destino geocodificado, eventos críticos (paradas, desvíos).
- **Alertas operativas**: monitorear errores de lectura directa y activar fallback automatico.

## Plan de implementacion sugerido

1. **Sprint 1**
   - Prototipo de `GPSDirectExtractor` para PDF y CSV.
   - Dataset de prueba con varias variantes de formato.

2. **Sprint 2**
   - Bypass en `DocumentParser` con bandera de feature toggle.
   - Actualizacion de `case_index` y compatibilidad docless.

3. **Sprint 3**
   - `GPSQueryService` con endpoints internos y pruebas unitarias.
   - Integracion con motor de correlacion (nuevas reglas y tests).

4. **Sprint 4**
   - Ajustes en Agente Rick (handler especifico para preguntas GPS).
   - Auditoria y logs siguiendo guias RAG de Better Practices.

5. **Sprint 5**
   - Actualizacion de UI / upload (opcional) y guias operativas.
   - Documentacion en `BETTER_PRACTICES.md` y manuales.

6. **Sprint 6**
   - Tests de carga y costos.
   - Feature toggle a produccion, monitoreo cercano y retroalimentacion.

## Riesgos y mitigaciones

- **Dependencias nativas**: PyMuPDF y pandas requieren paquetes del sistema. Mitigacion: construir imagen Docker con base reproducible y CI.
- **Variabilidad de formatos**: algunos PDF pueden ser escaneados. Mitigacion: fallback a OCR + alerta para ajustar proveedores.
- **Consistencia de datos**: asegurar que los campos extraidos coincidan con los usados por reglas. Mitigacion: validadores schema-first y pruebas de regresion.
- **Impacto en docless**: nuevos archivos deben registrarse en `case_index` para reprocesos. Mitigacion: actualizar `OCRCacheManager`, guardar resuenes y rutas absolutas.

## Proximos pasos

- Definir responsables y esfuerzo estimado para cada sprint.
- Preparar datasets reales anonimizados para pruebas de rendimiento.
- Elaborar estrategia de monitoreo (costos, tiempos, tasa de fallos).
- Revisar implicaciones legales/seguridad de almacenar datos GPS en crudo.

Con este plan se alinean los objetivos de rendimiento, costo y auditabilidad sin romper la arquitectura actual, manteniendo un camino de fallback y pruebas que respete Better Practices.
