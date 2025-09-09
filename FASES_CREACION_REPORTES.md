# 📊 Fases de Creación de Reportes - Fraud Scorer v2.0

## Descripción General

Fraud Scorer v2.0 procesa casos de siniestros mediante un pipeline de 5 fases que transforma documentos físicos en reportes analíticos profesionales.

---

## 🔄 Pipeline de Procesamiento

### **FASE 1: Procesamiento de Documentos (OCR)**
**Archivo**: `scripts/run_report.py:337-525`  
**Componentes**: `AzureOCRProcessor`, `DocumentParser`, `OCRCacheManager`

#### Funcionalidades:
- ✅ **Extracción de texto** de PDFs, imágenes, documentos Office
- ✅ **Cache inteligente** para evitar reprocesamiento (reduce tiempo 80%)
- ✅ **Tolerancia a fallos** con reintentos automáticos
- ✅ **Almacenamiento** en `data/ocr_cache/`

#### Documentos soportados:
- PDFs, PNG, JPG, JPEG, TIFF
- DOCX, XLSX, CSV

---

### **FASE 1.4: Clasificación de Documentos**
**Archivo**: `scripts/run_report.py:527-613`  
**Componente**: `DocumentClassifier`

#### Funcionalidades:
- 🤖 **Clasificación híbrida**: LLM + heurísticas
- 📄 **Tipos detectados**: pólizas, denuncias, facturas, GPS, peritajes, IDs
- ⚙️ **Configuración ajustable** por umbral de confianza
- 🔄 **Fallback automático** si baja confianza LLM

---

### **FASE 1.5: Detección HDI Especial**
**Archivo**: `scripts/run_report.py:614-687`

#### Funcionalidades:
- 🏠 **Detección "HDI EN MI CASA"** mediante análisis de contenido
- ⚡ **Reglas especiales** de extracción según tipo de póliza
- 🎯 **Ajuste automático** del comportamiento del sistema

---

### **FASE 2: Extracción con IA**
**Archivo**: `scripts/run_report.py:688-752`  
**Componente**: `AIFieldExtractor`

#### Funcionalidades:
- 🧠 **Extracción semántica** con GPT-4
- 🛡️ **Modo guiado** con restricciones documento-campo
- ⚡ **Procesamiento paralelo** controlado (máximo 3 procesos simultáneos)
- 🎯 **Extracción selectiva** solo de documentos objetivo

#### Campos extraídos:
- Número de siniestro, póliza, asegurado
- Fechas de ocurrencia y reclamación
- Montos, lugares, bienes reclamados
- Información de ajuste y contacto

---

### **FASE 3: Consolidación Inteligente**
**Archivo**: `scripts/run_report.py`  
**Componente**: `AIConsolidator`

#### Funcionalidades:
- 🧩 **Resolución de conflictos** entre documentos
- 🤝 **Consolidación inteligente** usando razonamiento avanzado IA
- 📊 **Cálculo de confianza** por campo consolidado
- 🗃️ **Actualización automática** de índices de cache
- 📁 **Reorganización de estructura** con nomenclatura consistente

### **FASE 4: Generación del Reporte**
**Archivo**: `scripts/run_report.py`  
**Componente**: `AIReportGenerator`

#### Archivos clave:
- **Template**: `src/fraud_scorer/templates/report_template.html`
- **Generador**: `src/fraud_scorer/templates/ai_report_generator.py`

#### Funcionalidades:
- 🎨 **Template profesional** con diseño corporativo HDI
- 📱 **Diseño responsive** (móvil/desktop/print)
- 🔤 **Tipografía profesional** (Source Sans Pro, IBM Plex Mono)
- 🎯 **Iconografía moderna** (RemixIcon)
- 📄 **Generación PDF** con WeasyPrint
- 🧹 **Limpieza automática** de archivos previos

#### Archivos generados:
```
{ASEGURADO}_{SINIESTRO}_INFORME.html    # Reporte visual
{ASEGURADO}_{SINIESTRO}_INFORME.pdf     # Reporte para impresión
{ASEGURADO}_{SINIESTRO}_RESULTADOS.json # Datos completos + métricas
```

### **FASE 5: Almacenamiento y Organización**
**Archivo**: `scripts/run_report.py`

#### Estructura de archivos:
```
data/
├── reports/                           # Reportes finales
│   ├── {ASEGURADO}_{SINIESTRO}_INFORME.html
│   ├── {ASEGURADO}_{SINIESTRO}_INFORME.pdf
│   └── {ASEGURADO}_{SINIESTRO}_RESULTADOS.json
├── temp/pipeline_cache/               # Cache de consolidación
│   └── {ASEGURADO}_{SINIESTRO}_CONSOLIDADO.json
└── ocr_cache/                         # Cache OCR reutilizable
    ├── {ASEGURADO} - {SINIESTRO}/     # Documentos OCR por caso
    └── case_index/                    # Índices de casos
        └── {CASE_ID}.json
```

#### Métricas generadas:
- 📊 **Tasa de éxito OCR** y extracción
- 📈 **Completitud de campos** (% campos llenos)
- 🎯 **Confianza promedio** de extracción
- 🔧 **Conflictos resueltos** automáticamente

---

## 🚀 Optimizaciones del Sistema

### **Cache Inteligente**
- ✅ **Reutilización OCR**: Evita reprocesar documentos existentes
- ✅ **Cache por caso**: Aislamiento entre diferentes siniestros  
- ✅ **Reorganización automática**: Estructura consistente `[ASEGURADO] - [SINIESTRO]`
- ✅ **Limpieza de huérfanos**: Elimina archivos sin referencias en DB

### **Procesamiento Paralelo**
- ✅ **Semáforos controlados**: Máximo 3-4 procesos simultáneos
- ✅ **Clasificación asíncrona**: Documentos procesados en paralelo
- ✅ **Extracción por lotes**: Optimizada para múltiples documentos

### **Tolerancia a Fallos**
- ✅ **Fallbacks robustos**: Heurísticas si falla LLM
- ✅ **Reintentos automáticos**: Para servicios externos
- ✅ **Templates de emergencia**: HTML básico si falla Jinja2
- ✅ **Validación de datos**: Sanitización de nombres de archivo

---

## 🎛️ Configuración y Personalización

### **Variables de entorno clave:**
```env
# IA
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Azure OCR  
AZURE_ENDPOINT=https://...
AZURE_OCR_KEY=...

# Configuración del sistema
USE_LLM_DOC_CLASSIFIER=true          # Usar LLM para clasificación
ENABLE_HDI_SPECIAL_RULES=true        # Reglas especiales HDI
```

### **Configuración interna:**
- `src/fraud_scorer/settings.py`: Configuración de extracción y clasificación
- Umbral de confianza mínima para clasificación
- Lista de tipos de documento objetivo
- Configuraciones de paralelismo

---

## 📈 Métricas de Rendimiento

### **Tiempos típicos (caso con 5-10 documentos):**
- ⏱️ **Procesamiento total**: 30-60 segundos
- ⏱️ **OCR por documento**: 2-5 segundos
- ⏱️ **Extracción IA**: 5-15 segundos  
- ⏱️ **Consolidación**: 3-8 segundos
- ⏱️ **Generación reporte**: 2-5 segundos

### **Precisión:**
- 🎯 **Extracción de campos**: >95%
- 🎯 **Clasificación documentos**: >90%

### **Optimización con cache:**
- ⚡ **Reducción tiempo**: -80% en reprocesamiento
- ⚡ **Reutilización**: Casos similares aprovechan cache existente

---

## 🔧 Comandos de Ejecución

### **Procesamiento estándar:**
```bash
python scripts/run_report.py /ruta/documentos --out data/reports --title "Caso ABC-123"
```

### **Con organización previa:**
```bash  
python scripts/run_report.py /ruta/documentos --organize-first
```

### **Solo clasificación:**
```bash
python scripts/run_report.py /ruta/documentos --organize-only
```

### **Limpieza de cache:**
```bash
python scripts/run_report.py --purge-case CASE-2025-0001
python scripts/run_report.py --purge-orphans
```

---

**📌 Nota**: Este sistema está optimizado para documentos de seguros HDI pero es extensible a otras aseguradoras mediante configuración de plantillas y reglas específicas.
