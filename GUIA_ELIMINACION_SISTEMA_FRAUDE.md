# 🗑️ Guía de Eliminación Total del Sistema de Análisis de Fraude

## ⚠️ IMPORTANTE: Realizar backup completo antes de proceder
```bash
cp -r fraud_scorer fraud_scorer_backup_$(date +%Y%m%d_%H%M%S)
```

---

## 📋 Resumen de Cambios

Esta guía elimina completamente la **FASE 4: Análisis de Fraude** del sistema, manteniendo todas las demás funcionalidades intactas.

### Lo que se eliminará:
- ❌ Clase `AIDocumentAnalyzer` completa
- ❌ FASE 4 del pipeline de procesamiento
- ❌ Todos los campos relacionados con fraude en reportes
- ❌ Referencias a `fraud_score`, `fraud_indicators`, `inconsistencies`, `risk_level`
- ❌ Imports y dependencias del analizador de fraude

### Lo que se mantiene:
- ✅ Fases 1, 1.4, 1.5, 2, 3, 5 y 6 del pipeline
- ✅ Sistema de OCR y extracción
- ✅ Consolidación de datos
- ✅ Generación de reportes HTML/PDF (sin datos de fraude)
- ✅ Sistema de cache y replay

---

## 🔧 PASO 1: Eliminar el archivo principal del analizador

```bash
# Eliminar el archivo del analizador de fraude
rm src/fraud_scorer/processors/ai/document_analyzer.py
```

---

## 📝 PASO 2: Modificar `scripts/run_report.py`

### 2.1 Eliminar el import (línea ~1046)
```python
# ELIMINAR esta línea:
from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
```

### 2.2 Eliminar FASE 4 completa (líneas 836-864)
```python
# ELIMINAR todo este bloque:
# ============================================
# FASE 4: Análisis de fraude (IA)
# ============================================
logger.info("\n🔎 FASE 4: Análisis de fraude")
logger.info("-" * 40)

# Verificar cancelación antes de fase 4
if self.cancellation_check and await self.cancellation_check():
    await self.cleanup_on_cancel()
    raise asyncio.CancelledError("Proceso cancelado durante fase 4")

# Notificar análisis de fraude
if self.progress_callback:
    self.progress_callback("Analizando indicadores de fraude...", 70)

# Emitir evento de inicio de análisis
if self.progress_emitter:
    self.progress_emitter.emit("analyze", "started", message="Analizando indicadores de fraude")

ai_analysis = await self._analyze_fraud(consolidated, extractions)
fraud_score = ai_analysis.get("fraud_score", 0)
risk_level = "BAJO" if fraud_score < 0.3 else ("MEDIO" if fraud_score < 0.6 else "ALTO")
logger.info(f"✓ Fraud Score: {fraud_score:.2%}")
logger.info(f"✓ Nivel de Riesgo: {risk_level}")

# Emitir evento de finalización de análisis
if self.progress_emitter:
    self.progress_emitter.emit("analyze", "done", message=f"Análisis completado - Riesgo: {risk_level}")
```

### 2.3 Eliminar método `_analyze_fraud` (líneas 1038-1061)
```python
# ELIMINAR todo este método:
async def _analyze_fraud(
    self,
    consolidated: ConsolidatedExtraction,
    extractions: List[DocumentExtraction],
) -> Dict[str, Any]:
    """
    Análisis de fraude usando IA.
    """
    from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer

    analyzer = AIDocumentAnalyzer()

    # Payload ligero para IA
    docs_for_analysis = [
        {
            "document_type": extr.document_type,
            "key_value_pairs": extr.extracted_fields,
        }
        for extr in extractions
    ]

    analysis = await analyzer.analyze_claim_documents(docs_for_analysis)
    return analysis
```

### 2.4 Modificar la llamada a `generate_report` (línea ~912)
```python
# CAMBIAR de:
html_content = self.report_generator.generate_report(
    consolidated_data=consolidated,
    ai_analysis=ai_analysis,  # <-- ELIMINAR ESTE PARÁMETRO
    output_path=html_path,
    insured_name=insured_name_from_data,
    claim_number=claim_number_from_data
)

# A:
html_content = self.report_generator.generate_report(
    consolidated_data=consolidated,
    output_path=html_path,
    insured_name=insured_name_from_data,
    claim_number=claim_number_from_data
)
```

### 2.5 Ajustar porcentajes de progreso
Después de eliminar FASE 4, ajustar los porcentajes en las notificaciones:
```python
# Línea ~847 (antes era 70, ahora será parte de FASE 5):
# ELIMINAR la notificación de fraude

# Línea ~878 (ajustar de 85 a 75):
if self.progress_callback:
    self.progress_callback("Generando reporte HTML y PDF...", 75)

# Línea ~936 (ajustar de 95 a 90):
if self.progress_callback:
    self.progress_callback("Finalizando procesamiento...", 90)
```

### 2.6 Eliminar referencias en results (líneas ~1001, 1011-1013)
```python
# ELIMINAR estas líneas del diccionario results:
"fraud_analysis": ai_analysis,  # línea ~1001
"fraud_score": ai_analysis.get("fraud_score", 0) if ai_analysis else 0,  # línea ~1011
"risk_level": risk_level,  # línea ~1012
```

---

## 📝 PASO 3: Modificar `src/fraud_scorer/templates/ai_report_generator.py`

### 3.1 Modificar método `generate_report` (líneas ~88-95)
```python
def generate_report(
    self,
    consolidated_data: ConsolidatedExtraction,
    # ai_analysis: Optional[Dict[str, Any]] = None,  # ELIMINAR ESTE PARÁMETRO
    output_path: Optional[Path] = None,
    insured_name: Optional[str] = None,
    claim_number: Optional[str] = None
) -> str:
```

### 3.2 Modificar `_prepare_template_data` (líneas ~192-196)
```python
def _prepare_template_data(
    self,
    consolidated_data: ConsolidatedExtraction,
    # ai_analysis: Optional[Dict[str, Any]]  # ELIMINAR ESTE PARÁMETRO
) -> Dict[str, Any]:
```

### 3.3 Eliminar la llamada con ai_analysis (línea ~119)
```python
# CAMBIAR de:
template_data = self._prepare_template_data(consolidated_data, ai_analysis)

# A:
template_data = self._prepare_template_data(consolidated_data)
```

### 3.4 Eliminar sección de análisis de fraude (líneas ~269-279)
```python
# ELIMINAR este bloque completo:
# Agregar análisis de IA si existe (normalizado por si acaso)
ai_analysis_dict = _to_dict(ai_analysis)
if ai_analysis_dict:
    template_data.update({
        "fraud_score": ai_analysis_dict.get("fraud_score", 0),
        "risk_level": self._calculate_risk_level(ai_analysis_dict.get("fraud_score", 0) or 0),
        "inconsistencias": ai_analysis_dict.get("inconsistencies", []),
        "fraud_indicators": ai_analysis_dict.get("fraud_indicators", []),
        "validaciones_externas": ai_analysis_dict.get("external_validations", [])
    })
```

### 3.5 Eliminar método `_calculate_risk_level` (líneas ~315-326)
```python
# ELIMINAR todo este método:
def _calculate_risk_level(self, fraud_score: float) -> str:
    """Calcula el nivel de riesgo basado en el score"""
    try:
        score = float(fraud_score)
    except Exception:
        score = 0.0
    if score < 0.3:
        return "BAJO"
    elif score < 0.6:
        return "MEDIO"
    else:
        return "ALTO"
```

---

## 📝 PASO 4: Modificar `src/fraud_scorer/api/web_interface.py`

### 4.1 Eliminar import (línea ~178)
```python
# ELIMINAR estas líneas:
from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
```

### 4.2 Eliminar sección de análisis (líneas ~176-211)
```python
# ELIMINAR todo este bloque:
# Análisis de fraude
logger.info("Analizando fraude...")
from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
analyzer = AIDocumentAnalyzer()

docs_for_analysis = []
for doc in ocr_results:
    docs_for_analysis.append({
        "document_type": doc.get("document_type", "otro"),
        "key_value_pairs": doc.get("ocr_result", {}).get("key_value_pairs", {}),
        "specific_fields": {}
    })

ai_analysis = await analyzer.analyze_claim_documents(docs_for_analysis)

# ... y las líneas relacionadas con fraud_score y risk_level
```

### 4.3 Eliminar método `_get_risk_level` (líneas ~236-242)
```python
# ELIMINAR todo este método:
def _get_risk_level(self, fraud_score: float) -> str:
    """Calcula nivel de riesgo"""
    if fraud_score < 0.3:
        return "BAJO"
    elif fraud_score < 0.6:
        return "MEDIO"
    else:
        return "ALTO"
```

### 4.4 Eliminar referencias en respuestas (líneas ~507-508, 516)
```python
# ELIMINAR estas líneas:
fraud_score = fraud_analysis.get("fraud_score", 0)
risk_level = "BAJO" if fraud_score < 0.3 else ("MEDIO" if fraud_score < 0.6 else "ALTO")
# ...
"fraud_score": fraud_score,
```

### 4.5 Eliminar de test_template (línea ~640)
```python
# ELIMINAR:
"fraud_score": 0.3,  # Valor por defecto
```

---

## 📝 PASO 5: Modificar `src/fraud_scorer/api/endpoints/reports.py`

### 5.1 Eliminar import (línea ~15)
```python
# ELIMINAR:
from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
```

### 5.2 Eliminar análisis en endpoint (líneas ~381-382)
```python
# ELIMINAR estas líneas:
ai_analyzer = AIDocumentAnalyzer()
ai_analysis = await ai_analyzer.analyze_claim_documents(ocr_results)
```

### 5.3 Eliminar de response (línea ~280)
```python
# ELIMINAR:
response["fraud_indicators"] = fraud_analysis.get("alerts", [])
```

---

## 📝 PASO 6: Modificar `src/fraud_scorer/services/replay_service.py`

### 6.1 Eliminar import (línea ~14)
```python
# ELIMINAR:
from ..processors.ai.document_analyzer import AIDocumentAnalyzer
```

### 6.2 Eliminar análisis (líneas ~252-253)
```python
# ELIMINAR estas líneas:
analyzer = AIDocumentAnalyzer()
ai_analysis = await analyzer.analyze_claim_documents(docs_for_analysis)
```

---

## 📝 PASO 7: Modificar template HTML (si tiene secciones de fraude)

### 7.1 Revisar `src/fraud_scorer/templates/report_template.html`
Si el template tiene secciones que muestren fraud_score, risk_level, o indicadores de fraude, eliminarlas.

```html
<!-- ELIMINAR cualquier sección como esta: -->
<div class="fraud-section">
    <h3>Análisis de Fraude</h3>
    <p>Score: {{ fraud_score }}</p>
    <p>Nivel: {{ risk_level }}</p>
</div>
```

---

## 📝 PASO 8: Actualizar documentación

### 8.1 Modificar `FASES_CREACION_REPORTES.md`
Eliminar toda la sección de FASE 4 y ajustar la numeración:
- FASE 5 → FASE 4 (Generación del Reporte)
- FASE 6 → FASE 5 (Almacenamiento y Organización)

### 8.2 Actualizar `README.md`
Eliminar referencias al análisis de fraude si las hay.

---

## 🧪 PASO 9: Validación

### 9.1 Verificar que no queden imports huérfanos
```bash
# Buscar cualquier referencia restante
grep -r "AIDocumentAnalyzer" src/ scripts/
grep -r "fraud_score" src/ scripts/
grep -r "fraud_indicators" src/ scripts/
grep -r "_analyze_fraud" src/ scripts/
```

### 9.2 Verificar que no queden archivos relacionados
```bash
# Verificar que el archivo fue eliminado
ls -la src/fraud_scorer/processors/ai/document_analyzer.py
# Debe dar error: "No such file or directory"
```

### 9.3 Test de funcionamiento
```bash
# Ejecutar un caso de prueba sin análisis de fraude
python scripts/run_report.py /path/to/test/documents --out data/test_output

# Verificar que el JSON de salida NO contenga campos de fraude
cat data/test_output/*_RESULTADOS.json | grep -i fraud
# No debe encontrar nada
```

### 9.4 Verificar la API (si aplica)
```bash
# Iniciar servidor web
python scripts/start_web_server.py

# Probar que los endpoints funcionen sin errores
curl http://localhost:8000/health
```

---

## ✅ Checklist de Verificación Final

- [ ] Backup completo realizado
- [ ] `document_analyzer.py` eliminado
- [ ] FASE 4 eliminada de `run_report.py`
- [ ] Método `_analyze_fraud` eliminado
- [ ] Referencias a `ai_analysis` eliminadas de `generate_report`
- [ ] Método `_calculate_risk_level` eliminado
- [ ] Imports de `AIDocumentAnalyzer` eliminados en todos los archivos
- [ ] Referencias a `fraud_score`, `risk_level`, `inconsistencies` eliminadas
- [ ] Template HTML actualizado (sin secciones de fraude)
- [ ] Documentación actualizada
- [ ] Tests ejecutados exitosamente
- [ ] No hay errores de import al ejecutar el sistema
- [ ] Reportes generados correctamente sin datos de fraude

---

## 🔄 Rollback (si es necesario)

Si algo sale mal, puedes restaurar desde el backup:
```bash
# Restaurar desde backup
rm -rf fraud_scorer
mv fraud_scorer_backup_[timestamp] fraud_scorer
```

---

## 📌 Notas Importantes

1. **El sistema funcionará sin FASE 4**: Las fases 1-3 procesan y consolidan datos, la fase 5 (ahora 4) genera el reporte, y la fase 6 (ahora 5) guarda los resultados.

2. **Los reportes NO contendrán análisis de fraude**: Los HTML/PDF se generarán solo con los datos consolidados.

3. **El pipeline será más rápido**: Sin el análisis de fraude, el procesamiento será 5-15 segundos más rápido.

4. **Puedes reconstruir después**: Al eliminar todo limpiamente, tendrás una base sólida para implementar un nuevo sistema de análisis de fraude en el futuro.

---

## 🚀 Comando de Ejecución Post-Eliminación

Después de completar todos los pasos:
```bash
# Test completo del sistema sin análisis de fraude
python scripts/test_system.py

# Procesar un caso real
python scripts/run_report.py /ruta/a/documentos --out data/reports --title "Test sin fraude"
```

---

**⚠️ IMPORTANTE**: Esta guía elimina permanentemente el sistema de análisis de fraude. Asegúrate de tener un backup completo antes de proceder.