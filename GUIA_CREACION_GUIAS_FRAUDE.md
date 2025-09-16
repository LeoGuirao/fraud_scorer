# 🎯 Guía para Crear Guías de Análisis de Fraude

## 📊 Resumen del Sistema

### ¿Cómo funciona la aplicación?

El **Fraud Scorer** es un sistema de análisis documental que detecta posibles fraudes en reclamaciones de seguros mediante:

1. **OCR** → Extrae texto de documentos PDF/imágenes
2. **Clasificación** → Identifica el tipo de documento
3. **Extracción** → Obtiene campos clave según el tipo
4. **Análisis de Fraude** → Evalúa riesgos usando guías YAML especializadas
5. **Reporte** → Genera informe HTML con análisis detallado

### Flujo de Análisis con Guías

```mermaid
Documento → Clasificación → ¿Existe guía YAML?
                                    ↓ Sí
                            Carga guía específica
                                    ↓
                            Genera prompt con guía
                                    ↓
                            LLM analiza documento
                                    ↓
                            Genera FraudAnalysisResult
                                    ↓
                            Guarda en DB + Reporte HTML
```

## 🔧 Estructura de una Guía YAML

### Anatomía Completa

```yaml
# METADATA - Información de la guía
metadata:
  type: "tipo_documento_canonico"        # CRÍTICO: debe coincidir con clasificación
  version: "1.0"                          # Versionado para trazabilidad
  author: "Sistema Experto HDI"
  last_updated: "2025-01-15"
  description: "Descripción de qué analiza esta guía"

# DEFINITION - Qué buscar en el documento
definition:
  nature: "Descripción legal/técnica del documento"
  legal_importance: "Alta|Media|Baja - Impacto legal"
  critical_elements:                      # Campos que DEBEN estar presentes
    - campo_critico_1
    - campo_critico_2
    - campo_critico_3

# METHODOLOGY - Cómo analizar
methodology:
  analysis_approach: |                   # Estrategia de análisis paso a paso
    1. Verificar integridad documental
    2. Validar coherencia temporal
    3. Contrastar información
    4. Identificar anomalías

  fraud_indicators:                       # Patrones de fraude categorizados
    high_risk:                           # Riesgo alto/crítico
      - pattern: "nombre_patron"
        detection: "Cómo detectarlo"
        severity: "critico|alto"
    medium_risk:                         # Riesgo medio
      - pattern: "patron_medio"
        detection: "Descripción"
        severity: "medio"
    low_risk:                            # Riesgo bajo
      - pattern: "patron_bajo"
        detection: "Descripción"
        severity: "bajo"

  validation_rules:                      # Reglas de validación específicas
    campo_x:
      type: "pattern|list|range"
      pattern: "^regex$"                 # Si es pattern
      values: ["valor1", "valor2"]       # Si es list
      min: 0                             # Si es range
      max: 100

  cross_reference_documents:            # Documentos relacionados a validar
    - documento_relacionado_1
    - documento_relacionado_2

# RESPONSE TEMPLATE - Estructura de salida esperada
response_template:
  output_format:
    documento_analizado: "tipo_documento"

    # CAMPO CRÍTICO - Análisis narrativo completo
    analisis_completo: |
      Plantilla de texto para el análisis narrativo.
      Debe contener 3-5 párrafos describiendo:
      - Hallazgos principales
      - Validaciones realizadas
      - Inconsistencias detectadas
      - Conclusiones específicas

    # Métricas principales
    risk_level: "bajo|medio|alto|critico"
    fraud_score: 0.0                    # 0.0 a 1.0
    confidence: 0.0                     # 0.0 a 1.0

    # Detalles estructurados
    indicators: []                       # Lista de indicadores encontrados
    evidence: []                         # Evidencia específica
    recommendations: []                  # Recomendaciones de acción

    # Campos específicos del documento (opcionales)
    campo_especifico_1:
      subcampo: valor
    campo_especifico_2: true

    # Tareas de validación pendientes
    validation_tasks:
      - tarea: "Descripción de validación externa"
        fuente: "Dónde validar"
        prioridad: "alta|media|baja"
        estado: "pendiente"
```

## 🔗 Cómo se Conecta con el Sistema

### 1. **Registro de la Guía**
```python
# src/fraud_scorer/analyzers/fraud_guide_manager.py
class FraudGuideManager:
    def __init__(self):
        self.guides_dir = Path("src/fraud_scorer/guides")
        self.guides = self._load_all_guides()

    def get_guide(self, document_type: str) -> Optional[FraudGuide]:
        # Busca la guía por tipo de documento
        # Usa aliases si no encuentra coincidencia exacta
```

### 2. **Uso en el Analizador**
```python
# src/fraud_scorer/analyzers/fraud_analyzer.py
async def analyze_document(...):
    # 1. Obtiene la guía
    guide = self.guides.get_guide(document_type)

    # 2. Construye el prompt con la guía
    prompt = self.prompts.build_fraud_analysis_prompt(
        document_type=document_type,
        document_name=document_name,
        ocr_content=ocr_result,
        extracted_fields=extraction.extracted_fields,
        guide=guide._data,
        context=context
    )

    # 3. Llama al LLM con el prompt
    response = await self.client.chat.completions.create(
        model=self.model,
        response_model=dict,
        messages=[{"role": "user", "content": prompt}]
    )

    # 4. Parsea y valida la respuesta
    return self._parse_llm_response(response, guide, ...)
```

### 3. **Persistencia en Base de Datos**
```sql
-- La guía influye en estos campos de fraud_analyses:
risk_level      -- Determinado por la guía
fraud_score     -- Calculado según indicadores
analisis_completo -- Texto narrativo generado
indicators      -- JSON de indicadores detectados
guide_version   -- Versión de la guía usada
```

### 4. **Renderizado en Reporte**
```html
<!-- report_template.html -->
{% if mostrar_seccion_fraude %}
  {% for documento in documentos_analizados %}
    <!-- Muestra analisis_completo (evitar bloques vacíos con |trim) -->
    {% if documento.analisis_completo|trim %}
      {{ documento.analisis_completo }}
    {% endif %}

    <!-- Muestra indicadores de la guía -->
    {% for ind in documento.analisis.indicadores %}
      {{ ind.tipo }} - {{ ind.descripcion }}
    {% endfor %}
  {% endfor %}
{% endif %}
```

### 5. **Alineación de Tipos y Alias (canónicos)**
- El `metadata.type` de la guía debe corresponder a un tipo canónico reconocido por el sistema o por sus alias.
- Mapeos vigentes en el gestor de guías:
  - `carta_reclamacion` | `carta_respuesta` → `carta_de_reclamacion_formal_a_la_aseguradora`
  - `denuncia` | `denuncia_hechos` → `denuncia_de_los_hechos`
  - `carta_porte` → `cfdi_carta_porte`
  - `factura` | `factura_compra` → `facturas_comerciales_internacionales` (guía especializada)
  - `poliza` → `poliza_de_la_aseguradora`
  - `tarjeta_circulacion` → `tarjeta_de_circulacion_vehiculo`

Recomendación: para nuevos tipos canónicos, añadir metadatos visuales en `fraud_report_generator.py` (título, icono, orden) para una mejor presentación en el reporte.

## 📝 Ejemplo Práctico: Crear Guía para Carta de Reclamación

```yaml
metadata:
  type: "carta_de_reclamacion_formal_a_la_aseguradora"
  version: "1.0"
  author: "Sistema Experto HDI"
  last_updated: "2025-01-15"
  description: "Análisis de cartas de reclamación para detectar inconsistencias"

definition:
  nature: "Documento formal de solicitud de indemnización"
  legal_importance: "Alta - Establece pretensión económica"
  critical_elements:
    - fecha_carta
    - monto_reclamado
    - descripcion_hechos
    - documentos_adjuntos
    - firma_asegurado

methodology:
  analysis_approach: |
    1. Verificar coherencia entre monto reclamado y daños descritos
    2. Validar fecha de carta vs fecha de siniestro
    3. Contrastar con documentación soporte
    4. Identificar lenguaje atípico o exagerado

  fraud_indicators:
    high_risk:
      - pattern: "monto_excesivo"
        detection: "Monto supera valor asegurado o es desproporcionado"
        severity: "critico"
      - pattern: "documentos_faltantes"
        detection: "No menciona o adjunta documentos críticos"
        severity: "alto"
    medium_risk:
      - pattern: "demora_injustificada"
        detection: "Carta enviada meses después sin explicación"
        severity: "medio"

  cross_reference_documents:
    - poliza_de_la_aseguradora
    - denuncia_de_los_hechos
    - guias_y_facturas

response_template:
  output_format:
    documento_analizado: "carta_de_reclamacion_formal_a_la_aseguradora"
    analisis_completo: |
      Análisis exhaustivo de la carta de reclamación evaluando:
      - Coherencia entre el monto reclamado y los daños descritos
      - Validación temporal entre fechas mencionadas
      - Completitud de la documentación referenciada
      - Identificación de exageraciones o inconsistencias
      [Generar 3-5 párrafos con hallazgos específicos]
    risk_level: "bajo|medio|alto|critico"
    fraud_score: 0.0
    confidence: 0.0
    indicators: []
    evidence: []
    recommendations: []
```

## 🚀 Pasos para Crear una Nueva Guía

### 1. **Identificar el Tipo de Documento**
```bash
# Ver tipos de documentos en tu sistema
grep -r "document_type" src/fraud_scorer/classifiers/
```

### 2. **Crear el Archivo YAML**
```bash
# Crear nueva guía
touch src/fraud_scorer/guides/nuevo_documento.yaml
```

### 3. **Definir la Estructura**
- Copiar plantilla base
- Ajustar `metadata.type` al tipo exacto del clasificador
- Definir `critical_elements` específicos
- Crear `fraud_indicators` relevantes
- Personalizar `analisis_completo` en template

### 4. **Probar la Guía**
```bash
# Ejecutar análisis con la nueva guía
python scripts/run_report.py /path/to/case --fraud

# Verificar en logs
tail -f logs/fraud_analysis.log
```

### 5. **Validar Resultados**
- Revisar que `analisis_completo` se genera
- Verificar `risk_level` y `fraud_score` coherentes
- Confirmar que se guarda en DB
- Revisar reporte HTML generado

## 🎯 Mejores Prácticas

### ✅ DO's
1. **Nombre del tipo EXACTO**: El `metadata.type` debe coincidir exactamente con el clasificador
2. **Indicadores específicos**: Crear patrones de fraude específicos del documento
3. **Análisis completo descriptivo**: Dar contexto claro en la plantilla de `analisis_completo` (3–5 párrafos)
4. **Versionado**: Incrementar versión al modificar la guía
5. **Documentación clara**: Explicar cada patrón de fraude
6. **Evitar métricas fijas**: No fijar `risk_level`, `fraud_score`, `confidence`; usar enum/rangos para guiar al modelo
7. **Indicador de ejemplo**: Incluir un ejemplo de estructura de indicador para guiar formato del LLM

### ❌ DON'Ts
1. **No omitir `analisis_completo`**: Es crítico para el reporte
2. **No usar tipos genéricos**: Cada documento necesita su guía específica
3. **No mezclar severidades**: Mantener coherencia entre score y risk_level
4. **No olvidar cross_reference**: Documentar relaciones entre documentos

### 📌 Snippets útiles

Indicador con todos los campos (recomendado como ejemplo de formato):
```yaml
indicators:
  - pattern: "monto_excesivo"
    description: "Monto supera cobertura de póliza"
    severity: "alto"
    confidence: 0.8
    evidence: "p.2, cuadro de montos"
    location: { page: 2 }
```

Regex robusta para montos con miles/decimales:
```yaml
validation_rules:
  monto:
    type: pattern
    pattern: "^[0-9]{1,3}(?:[.,\\s][0-9]{3})*(?:[.,][0-9]{2})?$"
```

Regex de fecha aceptando ISO y DD/MM/YYYY:
```yaml
validation_rules:
  fecha:
    type: pattern
    pattern: "^(?:\\d{4}-\\d{2}-\\d{2}|\\d{2}/\\d{2}/\\d{4})$"
```

## 📊 Mapeo Score ↔ Risk Level

| Risk Level | Fraud Score Range | Color en Reporte |
|------------|------------------|------------------|
| `bajo`     | 0.00 - 0.30      | Verde (#16a34a)  |
| `medio`    | 0.31 - 0.59      | Amarillo (#f59e0b)|
| `alto`     | 0.60 - 0.84      | Rojo (#ef4444)   |
| `critico`  | 0.85 - 1.00      | Rojo Oscuro (#991b1b)|

## 🔍 Debugging

### Ver qué guía se está usando:
```python
# En fraud_analyzer.py, línea ~60
logger.info(f"Usando guía: {guide.type} v{guide.version}")
```

### Verificar salida del LLM:
```python
# En fraud_analyzer.py, línea ~175
logger.debug(f"Respuesta LLM: {response}")
```

### Revisar en base de datos:
```sql
SELECT
    document_type,
    guide_version,
    risk_level,
    fraud_score,
    substr(analisis_completo, 1, 100) as preview
FROM fraud_analyses
WHERE case_id = 'CASE-2025-0001';
```

## 📚 Recursos Adicionales

- **Guías existentes**: `/src/fraud_scorer/guides/`
- **Manager de guías**: `/src/fraud_scorer/analyzers/fraud_guide_manager.py`
- **Analizador principal**: `/src/fraud_scorer/analyzers/fraud_analyzer.py`
- **Modelos de datos**: `/src/fraud_scorer/models/fraud_analysis.py`
- **Template de reporte**: `/src/fraud_scorer/templates/report_template.html`

---

**Última actualización**: 15/01/2025 • **Versión**: 1.0.0
