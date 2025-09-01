# Reporte de Avance: Optimización del Sistema Fraud Scorer

## Introducción al Proyecto de Optimización del Sistema Fraud Scorer

### Contexto General
Estás a punto de trabajar en la optimización de un sistema de detección de fraude de seguros que procesa documentos mediante OCR y análisis con IA. El sistema actual funciona pero tiene varios problemas críticos de estructura, cache, nomenclatura y experiencia de usuario que necesitan ser corregidos sistemáticamente.

### Arquitectura del Sistema
El proyecto está estructurado como una aplicación web Python con las siguientes características:

- **Frontend**: Interfaz web con templates HTML para upload y replay de casos
- **Backend**: API endpoints para procesamiento y consulta de casos
- **Procesamiento**: Scripts que manejan OCR, análisis IA y generación de reportes
- **Cache**: Sistema de almacenamiento de resultados OCR en fraud_scorer/data/ocr_cache/
- **Reportes**: Generación de informes HTML/PDF en fraud_scorer/data/reports/

### Problemas Críticos Identificados

#### 1. Sistema de Cache Desconectado
- Las estadísticas muestran 0 casos cuando existen casos reales procesados
- Desconexión entre el directorio de cache y el sistema de índice de casos
- El archivo case_index no refleja el estado real del sistema

#### 2. Nomenclatura Inconsistente
- Casos se nombran con patrón genérico "INF-CASE-2025-XXXX"
- Carpetas de cache usan nomenclatura alfanumérica confusa (1a, 2b, 5f)
- Falta conexión con datos reales del asegurado y número de siniestro

#### 3. Duplicación de Estructura
- Carpetas duplicadas en cache con/sin punto final
- Sistema temporal con archivos redundantes
- Falta de detección y reemplazo de casos idénticos

#### 4. Inconsistencia Terminal vs Web
- El sistema replay funciona en web pero falla en terminal
- Diferencias en manejo de rutas y parámetros
- Lista de casos muestra datos incorrectos (0 documentos, N/A en procesado)

### Objetivos de la Implementación

#### Fase Crítica (Prioridad 1)
Restaurar la conexión entre casos procesados y sistema de estadísticas para que el dashboard refleje el estado real del sistema.

#### Optimización de Estructura (Prioridad 2)
Implementar nomenclatura consistente basada en datos reales: [ASEGURADO]-[SINIESTRO] en lugar de códigos genéricos.

#### Sincronización (Prioridad 3)
Unificar el comportamiento entre interfaz web y terminal, asegurando consistencia en ambos canales.

#### Mejoras de Experiencia (Prioridad 4)
Añadir feedback en tiempo real durante el procesamiento y sistema de cancelación segura.

### Estrategia de Implementación
El plan sigue un enfoque de "arreglar la base primero": comenzamos con los sistemas críticos de cache e índice, luego construimos las mejoras de nomenclatura y estructura sobre esa base sólida, y finalmente añadimos las optimizaciones de experiencia de usuario.

### Archivos Clave Examinados
- endpoints/replay.py o web_interface.py - Sistema de estadísticas
- scripts/run_report.py - Procesamiento principal
- fraud_scorer/scripts/sreplay_case.py - Sistema replay terminal
- Estructura de fraud_scorer/data/ocr_cache/ - Sistema de cache
- Templates HTML - Interfaz de usuario

---

## IMPLEMENTACIONES REALIZADAS

## FASE 1: DIAGNÓSTICO Y CORRECCIÓN DE CACHE (CRÍTICO) ✅

### Paso 1.1: Investigar Sistema de Estadísticas de Cache ✅

#### **Diagnóstico Realizado:**
1. **Identificación del Problema Principal:**
   - El directorio `data/ocr_cache/case_index/` no existía
   - Encontrados 22 casos duplicados del mismo siniestro (20250000002494 - MODA YKT, S.A. DE C.V)
   - 286 documentos totales, pero solo 26 únicos (repetidos ~11 veces cada uno)

2. **Análisis de Duplicación:**
   - Cada ejecución de `run_report.py` creaba un nuevo caso (CASE-2025-XXXX)
   - No detectaba casos existentes, reprocesaba los mismos documentos
   - Base de datos inflada con registros duplicados

#### **Soluciones Implementadas:**

1. **Limpieza de Base de Datos:**
   - **Script creado:** `scripts/clean_duplicates_simple.py`
   - **Resultado:** Consolidado de 22 casos a 1 único (CASE-2025-0011)
   - **Documentos:** Reducidos de 286 a 26 únicos
   - **Base de datos:** Reducida de 4.4MB a 2.2MB (50% menos)
   - **Backups:** Creado backup automático antes de limpieza

2. **Prevención de Duplicados Futuros:**
   - **Modificado:** `scripts/run_report.py`
     - Agregada detección de casos existentes por ruta y título
     - Implementado reuso de casos en lugar de crear nuevos
   - **Agregado:** Función `get_case_by_title()` en `src/fraud_scorer/storage/cases.py`

3. **Reconstrucción del Sistema de Cache:**
   - **Script creado:** `scripts/rebuild_cache_index.py`
   - **Directorios creados:** `data/ocr_cache/case_index/`
   - **Archivos generados:** 
     - 1 archivo de índice JSON (CASE-2025-0011.json)
     - 26 archivos OCR reconstruidos en estructura hash
   - **Tamaño:** 0.24 MB total de cache

#### **Validación Exitosa:**
- **Script de prueba:** `test_stats.py` (eliminado tras verificación)
- **Resultados:**
  - Sistema reporta 1 caso en cache ✓
  - 26 archivos procesados ✓
  - 0.24 MB de tamaño total ✓
  - Estadísticas funcionando correctamente ✓

### Paso 1.2: Localizar Sistema de Índice de Casos ✅

#### **Estructura del Índice Identificada:**
- **Ubicación:** `data/ocr_cache/case_index/CASE-2025-0011.json`
- **Contenido identificado:**
  ```json
  {
    "case_id": "CASE-2025-0011",
    "case_title": "20250000002494 - MODA YKT, S.A. DE C.V",
    "insured_name": "MODA YKT, S.A. DE C.V",
    "claim_number": "20250000002494",
    "total_documents": 26,
    "documents": [...],
    "cache_files": [...],
    "processed_at": "...",
    "status": "processed"
  }
  ```

#### **Actualización de Sistema de Índices:**
- **Mejorado:** `scripts/run_report.py` (líneas 216-240)
  - Ahora guarda índices completos con toda la información
  - Incluye fecha de procesamiento, estado, y metadatos
  - Extrae automáticamente nombre de asegurado y número de siniestro

#### **Validación del Sistema:**
- Conexión entre casos procesados y sistema de índice restaurada
- Sistema de estadísticas refleja correctamente el estado real
- Índice valida caso único con todos los metadatos

### Limpieza de Archivos Temporales ✅
- **Eliminados:**
  - `scripts/clean_duplicate_cases.py` (versión con errores)
  - 2 backups antiguos de BD (liberados 9MB)
- **Conservados:**
  - `scripts/clean_duplicates_simple.py` (para uso futuro)
  - `scripts/rebuild_cache_index.py` (mantenimiento)
  - 1 backup de seguridad

---

## FASE 2: INVESTIGAR NOMENCLATURA Y EXTRACCIÓN DE DATOS

### Paso 2.1: Localizar Extracción de Datos del Asegurado ✅

#### **Flujo de Extracción Identificado:**

1. **OCR → Extracción (AI):**
   - **Ubicación:** `src/fraud_scorer/processors/ai/ai_field_extractor.py`
   - **Función:** Procesa cada documento OCR individualmente
   - **Salida:** Objetos `DocumentExtraction` con campos extraídos

2. **Consolidación de Datos:**
   - **Ubicación:** `src/fraud_scorer/processors/ai/ai_consolidator.py`
   - **Modelo:** `ConsolidatedFields` en `src/fraud_scorer/models/extraction.py` (líneas 110-111)
   - **Campos clave identificados:**
     - `nombre_asegurado` (línea 111)
     - `numero_siniestro` (línea 110)
     - Otros campos relevantes

3. **Uso en Procesamiento Principal:**
   - **Ubicación:** `scripts/run_report.py` (líneas 294-296)
   - **Código identificado:**
     ```python
     insured_name_from_data = fields_dict.get("nombre_asegurado", "Desconocido")
     claim_number_from_data = fields_dict.get("numero_siniestro", f"SINIESTRO_{case_id}")
     ```

4. **Aplicación en Nomenclatura:**
   - Los datos se usan para nombrar archivos de salida
   - Formato actual en `run_report.py`: `{asegurado}_{siniestro}_INFORME.{ext}`

#### **Ubicaciones Clave Documentadas:**
- **Modelos:** `/src/fraud_scorer/models/extraction.py` (ConsolidatedFields)
- **Procesamiento:** `/scripts/run_report.py` (extracción y nomenclatura)
- **Consolidación:** `/src/fraud_scorer/processors/ai/ai_consolidator.py`
- **Extracción:** `/src/fraud_scorer/processors/ai/ai_field_extractor.py`

### Paso 2.2: Encontrar Generación de Nombres de Reportes 🔄 (EN PROGRESO)

#### **Análisis de Patrones de Nomenclatura:**

1. **Patrón "INF-CASE" Localizado en:**
   - `src/fraud_scorer/services/replay_service.py` (líneas 157, 158, 244, 252)
   - `src/fraud_scorer/ui/replay_ui.py` (líneas 293, 294)
   - `src/fraud_scorer/api/web_interface.py` (líneas 170, 619, 675)
   - `src/fraud_scorer/api/endpoints/reports.py` (líneas 179, 394)
   - `src/fraud_scorer/pipelines/data_flow.py` (línea 323)

2. **Estado Actual de Nomenclatura:**
   - **run_report.py**: ✅ Ya usa formato correcto `{asegurado}_{siniestro}_INFORME`
   - **replay_service.py**: ❌ Usa formato antiguo `INF-{case_id}`
   - **web_interface.py**: ❌ Usa formato antiguo `INF-{case_id}`
   - **Otros archivos**: ❌ Formatos mixtos

#### **Implementaciones Realizadas (Parcial):**

1. **Actualizado replay_service.py:**
   - **Agregada obtención de datos del caso** (líneas 186-193):
     ```python
     case_index = self.cache_manager.get_case_index(case_id)
     if case_index:
         insured_name = case_index.get('insured_name', 'DESCONOCIDO')
         claim_number = case_index.get('claim_number', case_id)
     ```
   
   - **Modificada generación de nombres** (líneas 254-273):
     ```python
     s_insured = sanitize_filename(insured_name)
     s_claim = sanitize_filename(claim_number)
     html_path = output_path / f"INF-{s_insured}-{s_claim}.html"
     pdf_path = output_path / f"INF-{s_insured}-{s_claim}.pdf"
     ```

   - **Actualizada función de limpieza** (líneas 157-163):
     - Agregados patrones para formato nuevo: `f"INF-*-*.html"`, `f"INF-*-*.pdf"`
     - Mantenidos patrones antiguos para compatibilidad

#### **Pendiente de Implementación:**
- **web_interface.py**: Actualizar generación de nombres (líneas 170, 619, 675)
- **Otros archivos**: Revisar y actualizar patrones restantes
- **Pruebas**: Validar funcionamiento del nuevo sistema de nomenclatura

---

## ESTADO ACTUAL DEL PROYECTO

### ✅ **Completado:**
- **Fase 1 completa**: Sistema de cache restaurado y funcional
- **Paso 2.1 completo**: Flujo de extracción de datos localizado y documentado
- **Prevención de duplicados**: Sistema implementado y funcional
- **Base de datos limpia**: Un caso único con 26 documentos
- **Sistema de índices**: Funcional con metadatos completos

### 🔄 **En Progreso:**
- **Paso 2.2**: Actualización de nomenclatura de reportes
  - replay_service.py ✅ completado
  - web_interface.py ❌ pendiente
  - Otros archivos ❌ pendientes

### 📋 **Por Implementar:**
- Sincronización terminal vs web (Fase 3)
- Mejoras de experiencia de usuario (Fase 4)
- Pruebas completas del nuevo sistema
- Validación de nomenclatura en todos los componentes

### 📊 **Métricas de Mejora:**
- **Base de datos**: Reducida 50% (4.4MB → 2.2MB)
- **Casos**: Consolidados de 22 → 1
- **Documentos**: Únicos 26 (eliminados 260 duplicados)
- **Sistema de estadísticas**: 100% funcional
- **Archivos temporales**: Limpieza completa

### 🎯 **Próximo Paso:**
**Completar Paso 2.2** - Finalizar actualización de nomenclatura en web_interface.py y validar funcionamiento completo del nuevo sistema de nombres de reportes basado en datos reales del asegurado y número de siniestro.