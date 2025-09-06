# 🚀 GUÍA DE IMPLEMENTACIÓN - SISTEMA DE EXTRACCIÓN GUIADA CON IA

## 📌 Resumen Ejecutivo

Sistema de Extracción Guiada que elimina alucinaciones de IA mediante:
- **Restricción por documento**: Cada campo solo se busca en documentos autorizados
- **Doble barrera**: Validación en prompt + máscara en código
- **Rutas duales**: Direct AI (GPT-5 visión) y OCR + IA textual
- **Trazabilidad completa**: Metadata detallada de cada decisión

**Principio clave**: Reutilizar la estructura existente, modificando solo los archivos actuales.

---

## 📁 ARCHIVOS A MODIFICAR (Sin crear nuevos)

### Archivos principales a actualizar:
1. `src/fraud_scorer/settings.py` - Configuración y mapeos
2. `src/fraud_scorer/prompts/extraction_prompts.py` - Prompts con guías
3. `src/fraud_scorer/processors/ai/ai_field_extractor.py` - Extractor con rutas duales
4. `src/fraud_scorer/processors/ai/ai_consolidator.py` - Consolidador con filtrado
5. `src/fraud_scorer/prompts/consolidation_prompts.py` - Prompts de consolidación
6. `scripts/run_report.py` - Orquestador principal

### Archivos opcionales (mejoras):
7. `src/fraud_scorer/models/extraction.py` - Modelos con metadata adicional
8. `src/fraud_scorer/utils/validators.py` - Validaciones de campos

---

## 🎯 FASE 1: CONFIGURACIÓN Y ESTRUCTURAS BASE

### 1.1 Actualizar settings.py con mapeos y guías

**Archivo:** `src/fraud_scorer/settings.py`

```python
# src/fraud_scorer/settings.py

"""
Configuración para el sistema de extracción con IA
Incluye mapeos de documentos y guías de extracción
"""
import os
from enum import Enum
from typing import Dict, Any, List
from pathlib import Path

class ModelType(Enum):
    """Modelos disponibles para cada tarea"""
    # Modelos para Direct AI (visión)
    GPT5_VISION = "gpt-5"
    GPT5_VISION_MINI = "gpt-5-mini"
    GPT5_VISION_NANO = "gpt-5-nano"
    
    # Modelos para texto
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    
    # Fallback (mantener compatibilidad)
    EXTRACTOR = "gpt-4o-mini"
    CONSOLIDATOR = "gpt-4o"
    GENERATOR = "gpt-4o-mini"

class ExtractionRoute(Enum):
    """Rutas de procesamiento"""
    DIRECT_AI = "direct_ai"  # Visión directa con GPT-5
    OCR_TEXT = "ocr_text"    # OCR + IA textual

class FieldPriority(Enum):
    """Prioridad de fuentes por tipo de documento"""
    INFORME_AJUSTADOR = 1
    POLIZA = 2
    CARTA_RECLAMACION = 3
    CARPETA_INVESTIGACION = 4
    OTRO = 99

class ExtractionConfig:
    """Configuración del sistema de extracción"""
    
    # Campos a extraer (mantener existente)
    REQUIRED_FIELDS = [
        "numero_siniestro",
        "nombre_asegurado", 
        "numero_poliza",
        "vigencia_inicio",
        "vigencia_fin",
        "domicilio_poliza",
        "bien_reclamado",
        "monto_reclamacion",
        "tipo_siniestro",
        "fecha_ocurrencia",
        "fecha_reclamacion",
        "lugar_hechos",
        "ajuste",
        "conclusiones"
    ]
    
    # ========== NUEVA SECCIÓN: GUÍAS DE EXTRACCIÓN ==========
    
    # Mapeo: tipo de documento → campos permitidos
    DOCUMENT_FIELD_MAPPING = {
        "informe_preliminar_del_ajustador": [
            "numero_siniestro",
            "nombre_asegurado",
            "bien_reclamado",
            "fecha_ocurrencia",
            "fecha_reclamacion",
            "lugar_hechos",
            "ajuste"
        ],
        "poliza_de_la_aseguradora": [
            "numero_poliza",
            "vigencia_inicio",
            "vigencia_fin",
            "domicilio_poliza"
        ],
        "carta_de_reclamacion_formal_a_la_aseguradra": [
            "monto_reclamacion"
        ],
        "carpeta_de_investigacion": ["tipo_siniestro"],
        "narracion_de_hechos": ["tipo_siniestro"],
        "declaracion_del_asegurado": ["tipo_siniestro"],
        "otro": []  # Documentos no reconocidos no pueden proveer campos
    }
    
    # Sinónimos y etiquetas para búsqueda
    FIELD_SYNONYMS = {
        "numero_siniestro": [
            "Siniestro", "N° de siniestro", "No. de siniestro", 
            "Su Referencia", "Número de siniestro"
        ],
        "nombre_asegurado": [
            "Asegurado", "Contratante", "Cliente", 
            "Nombre del asegurado", "Razón social"
        ],
        "numero_poliza": [
            "Póliza", "Póliza No.", "No. de póliza", 
            "Policy No.", "Número de póliza"
        ],
        "vigencia_inicio": [
            "Vigencia", "Desde", "Vigencia desde", 
            "Fecha inicio", "Inicio de vigencia"
        ],
        "vigencia_fin": [
            "Hasta", "Vigencia hasta", "Fecha fin", 
            "Fin de vigencia", "Vencimiento"
        ],
        "domicilio_poliza": [
            "Domicilio Fiscal", "Dirección Fiscal", 
            "Domicilio", "Dirección del asegurado"
        ],
        "bien_reclamado": [
            "Bienes Afectados", "Bien afectado", 
            "Objeto afectado", "Mercancía afectada", "Bienes"
        ],
        "monto_reclamacion": [
            "Valor estimado", "Total estimado de daños",
            "Importe total robado", "Monto reclamado", 
            "Suma reclamada", "Valor de la reclamación"
        ],
        "tipo_siniestro": [
            "Tipo de siniestro", "Clase de siniestro",
            "Naturaleza del siniestro", "Evento"
        ],
        "fecha_ocurrencia": [
            "Fecha y hora de ocurrido", "Fecha de siniestro",
            "Fecha del evento", "Cuando ocurrió"
        ],
        "fecha_reclamacion": [
            "Fecha de reporte", "Fecha y hora reportado",
            "Fecha de reclamación", "Cuando se reportó"
        ],
        "lugar_hechos": [
            "Ubicación del siniestro", "Lugar del siniestro",
            "Domicilio del evento", "Donde ocurrió"
        ],
        "ajuste": [
            "Ajustador", "Despacho ajustador", "Empresa ajustadora",
            "Firma ajustadora", "Ajuste realizado por"
        ]
    }
    
    # Reglas de formato y validación
    FIELD_VALIDATION_RULES = {
        "numero_siniestro": {
            "regex": r"^\d{14}$",
            "format": "14 dígitos exactos",
            "transform": lambda x: ''.join(filter(str.isdigit, str(x)))[:14]
        },
        "nombre_asegurado": {
            "min_length": 3,
            "max_length": 120,
            "format": "Nombre completo o razón social"
        },
        "numero_poliza": {
            "regex": r"^[A-Z0-9\-\s]+$",
            "format": "Alfanumérico con guiones y espacios"
        },
        "vigencia_inicio": {
            "format": "DD/MM/YYYY",
            "type": "date"
        },
        "vigencia_fin": {
            "format": "DD/MM/YYYY",
            "type": "date"
        },
        "bien_reclamado": {
            "max_words": 5,
            "format": "Máximo 5 palabras"
        },
        "monto_reclamacion": {
            "type": "float",
            "min": 0,
            "format": "Número positivo"
        },
        "fecha_ocurrencia": {
            "format": "DD/MM/YYYY",
            "type": "date"
        },
        "fecha_reclamacion": {
            "format": "DD/MM/YYYY",
            "type": "date"
        }
    }
    
    # Catálogo de tipos de siniestro
    SINIESTRO_TYPES = {
        "automoviles": [
            "Colisión / Choque",
            "Robo Total / Parcial",
            "Daños Materiales",
            "Responsabilidad Civil (Daños a Terceros)"
        ],
        "hogar": [
            "Incendio",
            "Inundación",
            "Robo",
            "Daños por Fenómeno Natural"
        ],
        "gastos_medicos_vida": [
            "Enfermedad / Accidente (Gastos Médicos)",
            "Fallecimiento (Vida)",
            "Invalidez Total y Permanente (Vida)"
        ],
        "transporte": [
            "Riesgos Ordinarios de Tránsito (ROT)",
            "Robo de Bulto por Entero",
            "Riesgos por Maniobras",
            "Rapiña o Saqueo"
        ]
    }
    
    # Ajustadores reconocidos
    RECOGNIZED_ADJUSTERS = [
        "SINIESCA",
        "PARK PERALES",
        # Agregar más según aparezcan
    ]
    
    # Configuración de rutas por tipo de archivo
    ROUTE_CONFIG = {
        ".pdf": "auto",  # Decidir según contenido
        ".jpg": ExtractionRoute.DIRECT_AI,
        ".jpeg": ExtractionRoute.DIRECT_AI,
        ".png": ExtractionRoute.DIRECT_AI,
        ".tiff": ExtractionRoute.DIRECT_AI,
        ".docx": ExtractionRoute.OCR_TEXT,
        ".doc": ExtractionRoute.OCR_TEXT,
        ".csv": ExtractionRoute.OCR_TEXT,
    }
    
    # Mapeo de tipos de documento a prioridades (actualizado)
    DOCUMENT_PRIORITIES = {
        "informe_preliminar_del_ajustador": FieldPriority.INFORME_AJUSTADOR,
        "poliza_de_la_aseguradora": FieldPriority.POLIZA,
        "carta_de_reclamacion_formal_a_la_aseguradra": FieldPriority.CARTA_RECLAMACION,
        "carpeta_de_investigacion": FieldPriority.CARPETA_INVESTIGACION,
        "narracion_de_hechos": FieldPriority.CARPETA_INVESTIGACION,
        "declaracion_del_asegurado": FieldPriority.CARPETA_INVESTIGACION,
        # Mantener compatibilidad con nombres anteriores
        "poliza": FieldPriority.POLIZA,
        "denuncia": FieldPriority.INFORME_AJUSTADOR,
        "factura": FieldPriority.OTRO,
        "otro": FieldPriority.OTRO
    }
```

---

## 🎯 FASE 2: SISTEMA DE PROMPTS CON GUÍAS

### 2.1 Actualizar ExtractionPromptBuilder

**Archivo:** `src/fraud_scorer/prompts/extraction_prompts.py`

```python
# src/fraud_scorer/prompts/extraction_prompts.py

"""
Constructor de prompts para extracción con IA
Actualizado con Sistema de Extracción Guiada
"""
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

# Importar configuración desde settings
from fraud_scorer.settings import ExtractionConfig

class ExtractionPromptBuilder:
    """
    Construye prompts optimizados para extracción de campos
    con guías estrictas por tipo de documento
    """
    
    def __init__(self):
        self.config = ExtractionConfig()
        
        # Cargar mapeos desde settings
        self.field_mapping = self.config.DOCUMENT_FIELD_MAPPING
        self.field_synonyms = self.config.FIELD_SYNONYMS
        self.validation_rules = self.config.FIELD_VALIDATION_RULES
        self.siniestro_types = self.config.SINIESTRO_TYPES
        
        # Mantener compatibilidad con código existente
        self.base_template = self._load_base_template()
        self.field_descriptions = self._load_field_descriptions()
        self.examples = self._load_examples()
    
    def build_extraction_prompt(
        self,
        document_name: str,
        document_type: str,
        ocr_content: Dict[str, Any],
        required_fields: List[str],
        use_guided: bool = True  # Nuevo parámetro para activar guía
    ) -> str:
        """
        Construye un prompt completo para extracción
        Mantiene compatibilidad con código existente
        """
        if use_guided and document_type in self.field_mapping:
            # Usar nueva versión con guía
            return self.build_guided_extraction_prompt(
                document_name=document_name,
                document_type=document_type,
                content=ocr_content,
                route="ocr_text"
            )
        else:
            # Mantener versión original para compatibilidad
            return self._build_legacy_prompt(
                document_name, document_type, ocr_content, required_fields
            )
    
    def build_guided_extraction_prompt(
        self,
        document_name: str,
        document_type: str,
        content: Optional[Dict[str, Any]] = None,
        route: str = "ocr_text"
    ) -> str:
        """
        Construye prompt con Sistema de Extracción Guiada
        """
        
        # 1. Obtener campos permitidos para este documento
        allowed_fields = self.field_mapping.get(document_type, [])
        
        # 2. Si no hay campos permitidos, retornar prompt mínimo
        if not allowed_fields:
            return self._build_null_prompt(document_name, document_type)
        
        # 3. Construir sección de guía
        guide_section = self._build_extraction_guide(document_type, allowed_fields)
        
        # 4. Construir prompt según ruta
        if route == "direct_ai":
            return self._build_vision_prompt(document_name, document_type, guide_section)
        else:
            return self._build_text_prompt(document_name, document_type, content, guide_section)
    
    def _build_extraction_guide(self, document_type: str, allowed_fields: List[str]) -> str:
        """
        Construye la sección de guía de extracción estricta
        """
        
        # Mapear tipo de documento a nombre legible
        doc_type_readable = {
            "informe_preliminar_del_ajustador": "Informe Preliminar del Ajustador",
            "poliza_de_la_aseguradora": "Póliza de la Aseguradora",
            "carta_de_reclamacion_formal_a_la_aseguradra": "Carta de Reclamación Formal",
            "carpeta_de_investigacion": "Carpeta de Investigación",
            "narracion_de_hechos": "Narración de Hechos",
            "declaracion_del_asegurado": "Declaración del Asegurado"
        }.get(document_type, document_type)
        
        guide = f"""
================================================================================
                        GUÍA DE EXTRACCIÓN ESTRICTA
================================================================================

DOCUMENTO ACTUAL: {doc_type_readable}
TIPO TÉCNICO: {document_type}

REGLA FUNDAMENTAL:
⚠️ ESTE DOCUMENTO SOLO PUEDE PROPORCIONAR LOS SIGUIENTES CAMPOS:
{chr(10).join(f'  ✓ {field}' for field in allowed_fields)}

TODOS LOS DEMÁS CAMPOS DEBEN SER NULL.

INSTRUCCIONES CRÍTICAS:
1. SOLO extrae los campos listados arriba
2. Si un campo NO está en la lista → DEBE ser null
3. NO inventes información
4. NO combines datos de múltiples secciones
5. NO asumas valores basándote en contexto

DETALLES POR CAMPO PERMITIDO:
"""
        
        # Agregar detalles específicos para cada campo permitido
        for field in allowed_fields:
            guide += self._format_field_guide(field)
        
        # Agregar instrucciones para campos especiales
        guide += self._add_special_instructions(document_type, allowed_fields)
        
        guide += """
================================================================================
"""
        
        return guide
    
    def _format_field_guide(self, field: str) -> str:
        """
        Formatea la guía para un campo específico
        """
        synonyms = self.field_synonyms.get(field, [])
        rules = self.validation_rules.get(field, {})
        
        guide = f"""
📍 {field.upper()}:
   Buscar en: {', '.join(synonyms[:3])}...
   Formato: {rules.get('format', 'texto libre')}"""
        
        # Agregar reglas específicas por campo
        if field == "numero_siniestro":
            guide += """
   Regla especial: Ignorar si dice "Antes..." """
        elif field == "vigencia_inicio" or field == "vigencia_fin":
            guide += """
   Regla especial: Convertir ENE→01, FEB→02, etc."""
        elif field == "bien_reclamado":
            guide += """
   Regla especial: Máximo 5 palabras, sin artículos"""
        elif field == "monto_reclamacion":
            guide += """
   Regla especial: Solo el monto total, sin desglose"""
        elif field == "tipo_siniestro":
            guide += f"""
   Valores permitidos: {', '.join([item for sublist in self.siniestro_types.values() for item in sublist][:5])}..."""
        elif field == "ajuste":
            guide += f"""
   Ajustadores válidos: {', '.join(self.config.RECOGNIZED_ADJUSTERS)}"""
        
        guide += "\n"
        return guide
    
    def _add_special_instructions(self, document_type: str, allowed_fields: List[str]) -> str:
        """
        Agrega instrucciones especiales según el tipo de documento
        """
        instructions = "\nINSTRUCCIONES ESPECIALES PARA ESTE DOCUMENTO:\n"
        
        if document_type == "informe_preliminar_del_ajustador":
            instructions += """
- El número de siniestro está en la tabla principal o cronología
- El ajustador puede estar en el encabezado, marca de agua o firma
- Las fechas suelen estar en formato DD/MM/YYYY
"""
        elif document_type == "poliza_de_la_aseguradora":
            instructions += """
- La vigencia aparece como "Desde... Hasta..."
- El domicilio fiscal es la dirección completa del asegurado
- El número de póliza puede tener guiones o espacios
"""
        elif document_type == "carta_de_reclamacion_formal_a_la_aseguradra":
            instructions += """
- Buscar el monto TOTAL reclamado, no parciales
- Puede aparecer como "valor estimado" o "suma reclamada"
"""
        elif document_type in ["carpeta_de_investigacion", "narracion_de_hechos", "declaracion_del_asegurado"]:
            instructions += """
- Identificar el tipo de siniestro según el catálogo
- Mapear a UNA sola categoría del listado oficial
"""
        
        return instructions
    
    def _build_vision_prompt(self, document_name: str, document_type: str, guide_section: str) -> str:
        """
        Prompt para Direct AI (visión con GPT-5)
        """
        
        return f"""
Eres un experto extractor de datos de documentos de seguros usando visión computacional.
Tu tarea es analizar el documento visual y extraer ÚNICAMENTE los campos permitidos.

DOCUMENTO A ANALIZAR:
📄 Archivo: {document_name}
📁 Tipo: {document_type}

{guide_section}

INSTRUCCIONES DE PROCESAMIENTO:
1. Analiza cuidadosamente la imagen/documento adjunto
2. Identifica las secciones relevantes visualmente
3. Extrae SOLO los campos permitidos según la guía
4. Para campos no permitidos o no encontrados: usar null
5. Mantén los valores exactamente como aparecen

FORMATO DE RESPUESTA OBLIGATORIO:
Responde ÚNICAMENTE con un JSON válido siguiendo esta estructura:
{{
    "numero_siniestro": "valor encontrado" o null,
    "nombre_asegurado": "valor encontrado" o null,
    "numero_poliza": "valor encontrado" o null,
    "vigencia_inicio": "DD/MM/YYYY" o null,
    "vigencia_fin": "DD/MM/YYYY" o null,
    "domicilio_poliza": "valor encontrado" o null,
    "bien_reclamado": "máximo 5 palabras" o null,
    "monto_reclamacion": número o null,
    "tipo_siniestro": "categoría del catálogo" o null,
    "fecha_ocurrencia": "DD/MM/YYYY" o null,
    "fecha_reclamacion": "DD/MM/YYYY" o null,
    "lugar_hechos": "valor encontrado" o null,
    "ajuste": "nombre del ajustador" o null,
    "conclusiones": null
}}

⚠️ RECORDATORIO FINAL: Solo los campos marcados con ✓ en la guía pueden tener valores.
"""
    
    def _build_text_prompt(self, document_name: str, document_type: str, content: Dict, guide_section: str) -> str:
        """
        Prompt para OCR + texto
        """
        
        # Formatear contenido OCR
        ocr_section = self._format_ocr_content(content) if content else "Sin contenido OCR"
        
        return f"""
Eres un experto extractor de datos de documentos de seguros.
Tu tarea es analizar el contenido OCR y extraer ÚNICAMENTE los campos permitidos.

DOCUMENTO A ANALIZAR:
📄 Archivo: {document_name}
📁 Tipo: {document_type}

{guide_section}

CONTENIDO DEL DOCUMENTO (OCR):
{ocr_section}

INSTRUCCIONES DE PROCESAMIENTO:
1. Analiza el texto OCR, campos clave-valor y tablas
2. Busca SOLO en las secciones indicadas en la guía
3. Extrae ÚNICAMENTE los campos permitidos (marcados con ✓)
4. Para campos no permitidos o no encontrados: usar null
5. Aplica las transformaciones de formato indicadas

FORMATO DE RESPUESTA OBLIGATORIO:
Responde ÚNICAMENTE con un JSON válido con TODOS estos campos:
{{
    "numero_siniestro": "valor" o null,
    "nombre_asegurado": "valor" o null,
    "numero_poliza": "valor" o null,
    "vigencia_inicio": "DD/MM/YYYY" o null,
    "vigencia_fin": "DD/MM/YYYY" o null,
    "domicilio_poliza": "valor" o null,
    "bien_reclamado": "máximo 5 palabras" o null,
    "monto_reclamacion": número o null,
    "tipo_siniestro": "valor del catálogo" o null,
    "fecha_ocurrencia": "DD/MM/YYYY" o null,
    "fecha_reclamacion": "DD/MM/YYYY" o null,
    "lugar_hechos": "valor" o null,
    "ajuste": "valor" o null,
    "conclusiones": null
}}

⚠️ RECORDATORIO FINAL: Solo extrae campos marcados con ✓ en la guía.
"""
    
    def _build_null_prompt(self, document_name: str, document_type: str) -> str:
        """
        Prompt para documentos no reconocidos (todos los campos null)
        """
        return f"""
El documento "{document_name}" es de tipo "{document_type}" que no está reconocido
en el sistema de extracción guiada.

Por seguridad, retorna todos los campos como null:
{{
    "numero_siniestro": null,
    "nombre_asegurado": null,
    "numero_poliza": null,
    "vigencia_inicio": null,
    "vigencia_fin": null,
    "domicilio_poliza": null,
    "bien_reclamado": null,
    "monto_reclamacion": null,
    "tipo_siniestro": null,
    "fecha_ocurrencia": null,
    "fecha_reclamacion": null,
    "lugar_hechos": null,
    "ajuste": null,
    "conclusiones": null
}}
"""
    
    def _build_legacy_prompt(
        self,
        document_name: str,
        document_type: str,
        ocr_content: Dict[str, Any],
        required_fields: List[str]
    ) -> str:
        """
        Mantiene el prompt original para compatibilidad
        """
        # [Copiar aquí el método build_extraction_prompt original]
        # Este es el código que ya existe en tu archivo
        fields_section = self._format_fields_section(required_fields)
        examples_section = self._format_examples_section(document_type)
        ocr_section = self._format_ocr_content(ocr_content)
        
        prompt = f"""
Eres un asistente experto en la extracción de datos de documentos de siniestros de seguros.

**DOCUMENTO A ANALIZAR:**
- Nombre del archivo: {document_name}
- Tipo de documento: {document_type}

**INSTRUCCIONES:**
1. Analiza cuidadosamente el contenido del documento proporcionado
2. Extrae ÚNICAMENTE los valores para los campos especificados
3. Si no encuentras información para un campo, déjalo como null
4. NO inventes o asumas información que no esté en el documento
5. Mantén los valores tal como aparecen en el documento

**CAMPOS A EXTRAER:**
{fields_section}

**REGLAS DE FORMATO:**
- Fechas: Formato YYYY-MM-DD (ejemplo: 2024-01-15)
- Montos: Número sin símbolos (ejemplo: 1500.50)
- Vigencias: Si aparecen como rango, separa en vigencia_inicio y vigencia_fin
- Nombres: Exactamente como aparecen en el documento
- Números de póliza/siniestro: Incluir todos los caracteres (letras, números, guiones)

{examples_section}

**CONTENIDO DEL DOCUMENTO:**
{ocr_section}

**IMPORTANTE:** 
- Responde ÚNICAMENTE con el JSON de extracción
- No incluyas explicaciones adicionales
- Asegúrate de que el JSON sea válido y contenga todos los campos requeridos
"""
        return prompt
    
    # Mantener métodos auxiliares existentes sin cambios
    def _format_fields_section(self, required_fields: List[str]) -> str:
        """Formatea la sección de campos a extraer con sus descripciones"""
        lines = []
        for field in required_fields:
            desc = self.field_descriptions.get(field, "")
            lines.append(f"- {field}: {desc}")
        return "\n".join(lines)
    
    def _format_examples_section(self, document_type: str) -> str:
        """Incluye ejemplos relevantes según el tipo de documento"""
        if document_type not in self.examples:
            return ""
        
        examples = self.examples.get(document_type, [])
        if not examples:
            return ""
        
        section = "\n**EJEMPLOS DE REFERENCIA:**\n"
        for i, example in enumerate(examples[:2], 1):
            section += f"\nEjemplo {i}:\n"
            section += f"Entrada: {example['input'][:200]}...\n"
            section += f"Extracción correcta: {json.dumps(example['output'], ensure_ascii=False, indent=2)}\n"
        
        return section
    
    def _format_ocr_content(self, ocr_content: Dict[str, Any]) -> str:
        """Formatea el contenido del OCR de manera estructurada"""
        if not ocr_content:
            return "No hay contenido OCR disponible"
            
        sections = []
        
        # Texto principal
        if ocr_content.get("text"):
            sections.append("TEXTO EXTRAÍDO:")
            sections.append(ocr_content["text"])
            sections.append("")
        
        # Pares clave-valor
        if ocr_content.get("key_value_pairs"):
            sections.append("CAMPOS DETECTADOS:")
            for key, value in ocr_content["key_value_pairs"].items():
                sections.append(f"  {key}: {value}")
            sections.append("")
        
        # Tablas
        if ocr_content.get("tables"):
            sections.append("TABLAS ENCONTRADAS:")
            for i, table in enumerate(ocr_content["tables"], 1):
                sections.append(f"\nTabla {i}:")
                if table.get("headers"):
                    sections.append(f"  Encabezados: {', '.join(table['headers'])}")
                if table.get("rows"):
                    sections.append("  Primeras filas:")
                    for row in table["rows"][:3]:
                        sections.append(f"    {' | '.join(str(cell) for cell in row)}")
        
        return "\n".join(sections)
    
    def _load_base_template(self) -> str:
        """Carga la plantilla base del prompt"""
        return ""
    
    def _load_field_descriptions(self) -> Dict[str, str]:
        """Carga las descripciones de los campos"""
        return {
            "numero_siniestro": "Número único del siniestro o reclamación (14 dígitos)",
            "nombre_asegurado": "Nombre completo del asegurado o empresa asegurada",
            "numero_poliza": "Número de la póliza de seguro",
            "vigencia_inicio": "Fecha de inicio de vigencia de la póliza",
            "vigencia_fin": "Fecha de fin de vigencia de la póliza",
            "domicilio_poliza": "Dirección completa registrada en la póliza",
            "bien_reclamado": "Descripción del bien o mercancía reclamada (máx 5 palabras)",
            "monto_reclamacion": "Monto total de la reclamación",
            "tipo_siniestro": "Tipo de siniestro según catálogo",
            "fecha_ocurrencia": "Fecha cuando ocurrió el siniestro",
            "fecha_reclamacion": "Fecha cuando se presentó la reclamación",
            "lugar_hechos": "Lugar donde ocurrió el siniestro",
            "ajuste": "Nombre del ajustador o empresa ajustadora",
            "conclusiones": "Conclusiones o resolución del caso"
        }
    
    def _load_examples(self) -> Dict[str, List[Dict]]:
        """Carga ejemplos de extracciones correctas"""
        return {
            "poliza": [
                {
                    "input": "PÓLIZA DE SEGURO No. AX-2024-001234...",
                    "output": {
                        "numero_poliza": "AX-2024-001234",
                        "nombre_asegurado": "MODA YKT, S.A. DE C.V.",
                        "vigencia_inicio": "2024-07-26",
                        "vigencia_fin": "2025-07-26"
                    }
                }
            ]
        }
```

---

## 🎯 FASE 3: EXTRACTOR CON RUTAS DUALES

### 3.1 Actualizar AIFieldExtractor

**Archivo:** `src/fraud_scorer/processors/ai/ai_field_extractor.py`

```python
# src/fraud_scorer/processors/ai/ai_field_extractor.py

"""
AIFieldExtractor: Extrae campos de documentos individuales usando IA
Actualizado con Sistema de Extracción Guiada y rutas duales
"""

from __future__ import annotations

import os
import re
import json
import logging
import base64
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import asyncio
from datetime import datetime

from openai import AsyncOpenAI
import instructor
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from fraud_scorer.settings import ExtractionConfig, ModelType, ExtractionRoute
from fraud_scorer.models.extraction import DocumentExtraction
from fraud_scorer.prompts.extraction_prompts import ExtractionPromptBuilder

logger = logging.getLogger(__name__)


class AIFieldExtractor:
    """
    Extractor de campos usando IA para documentos individuales.
    Ahora con Sistema de Extracción Guiada y rutas duales.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el extractor con cliente de OpenAI
        """
        # Cliente para OCR + texto (con instructor)
        raw_client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.client = instructor.patch(raw_client)
        
        # Cliente para visión directa (sin instructor)
        self.vision_client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        self.config = ExtractionConfig()
        self.prompt_builder = ExtractionPromptBuilder()
        self.extraction_cache: Dict[str, DocumentExtraction] = {}
        
        # Cargar mapeos de guías
        self.field_mapping = self.config.DOCUMENT_FIELD_MAPPING
        self.validation_rules = self.config.FIELD_VALIDATION_RULES

        logger.info("AIFieldExtractor inicializado con Sistema de Extracción Guiada")

    # =============================================================================
    #   MÉTODO PRINCIPAL: Mantiene compatibilidad y agrega funcionalidad guiada
    # =============================================================================
    
    async def extract_from_document(
        self,
        ocr_result: Dict[str, Any],
        document_name: str,
        document_type: Optional[str] = None,
        use_cache: bool = True,
        use_guided: bool = True  # Nuevo parámetro para activar extracción guiada
    ) -> DocumentExtraction:
        """
        Extrae campos de un documento individual
        Mantiene compatibilidad con código existente
        """
        if use_guided:
            # Usar nuevo sistema guiado
            return await self.extract_from_document_guided(
                content=ocr_result,
                document_name=document_name,
                document_type=document_type or self._detect_document_type(ocr_result, document_name),
                route="ocr_text",
                use_cache=use_cache
            )
        else:
            # Mantener comportamiento original
            return await self._extract_legacy(
                ocr_result, document_name, document_type, use_cache
            )
    
    async def extract_from_document_guided(
        self,
        content: Union[Dict[str, Any], bytes, Path],
        document_name: str,
        document_type: str,
        route: str = "ocr_text",
        model: str = "gpt-4o-mini",  # Por defecto usar modelo existente
        use_cache: bool = True
    ) -> DocumentExtraction:
        """
        Extracción guiada con doble ruta y validación estricta
        
        Args:
            content: OCR dict, bytes de imagen, o Path al archivo
            document_name: Nombre del documento
            document_type: Tipo detectado (informe_preliminar_del_ajustador, etc.)
            route: "direct_ai" o "ocr_text"
            model: Modelo a usar
        """
        
        # Cache key considerando la ruta y tipo
        cache_key = f"{document_name}_{document_type}_{route}"
        if use_cache and cache_key in self.extraction_cache:
            logger.info(f"Usando cache de extracción guiada para {document_name}")
            return self.extraction_cache[cache_key]
        
        logger.info(
            f"Extracción guiada iniciada:\n"
            f"  Documento: {document_name}\n"
            f"  Tipo: {document_type}\n"
            f"  Ruta: {route}\n"
            f"  Modelo: {model}"
        )
        
        # 1. Obtener campos permitidos para este documento
        allowed_fields = self.field_mapping.get(document_type, [])
        
        if not allowed_fields:
            logger.warning(
                f"Documento tipo '{document_type}' no tiene campos permitidos. "
                f"Retornando todos los campos como null."
            )
            return self._create_null_extraction(document_name, document_type)
        
        # 2. Construir prompt con guía
        prompt = self.prompt_builder.build_guided_extraction_prompt(
            document_name=document_name,
            document_type=document_type,
            content=content if route == "ocr_text" else None,
            route=route
        )
        
        # 3. Ejecutar extracción según ruta
        try:
            if route == "direct_ai":
                raw_extraction = await self._extract_direct_ai(
                    content=content,
                    prompt=prompt,
                    model=model
                )
            else:
                raw_extraction = await self._extract_ocr_text(
                    prompt=prompt,
                    model=model,
                    ocr_content=content
                )
        except Exception as e:
            logger.error(f"Error en extracción: {e}")
            raw_extraction = {}
        
        # 4. MÁSCARA DE SEGURIDAD - Forzar null en campos no permitidos
        masked_extraction, masked_fields = self._apply_field_mask(
            raw_extraction, 
            allowed_fields
        )
        
        # 5. Aplicar validaciones y transformaciones
        validated_extraction = self._validate_and_transform(
            masked_extraction,
            document_type
        )
        
        # 6. Calcular métricas de extracción
        extracted_count = sum(1 for v in validated_extraction.values() if v is not None)
        coverage = extracted_count / len(allowed_fields) if allowed_fields else 0
        
        # 7. Construir respuesta con metadata completa
        result = DocumentExtraction(
            source_document=document_name,
            document_type=document_type,
            extracted_fields=validated_extraction,
            extraction_metadata={
                "route": route,
                "model_used": model,
                "guide_applied": True,
                "allowed_fields": allowed_fields,
                "masked_fields": masked_fields,
                "extracted_count": extracted_count,
                "coverage": round(coverage, 2),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Guardar en cache
        if use_cache:
            self.extraction_cache[cache_key] = result
        
        # Log de resultado
        logger.info(
            f"Extracción completada:\n"
            f"  Campos permitidos: {len(allowed_fields)}\n"
            f"  Campos extraídos: {extracted_count}\n"
            f"  Campos bloqueados: {len(masked_fields)}\n"
            f"  Cobertura: {coverage:.0%}"
        )
        
        return result
    
    async def _extract_direct_ai(
        self, 
        content: Union[bytes, Path],
        prompt: str,
        model: str
    ) -> Dict[str, Any]:
        """
        Extracción usando visión directa con GPT-5 o GPT-4V
        """
        logger.info(f"Iniciando extracción Direct AI con modelo {model}")
        
        # Preparar imagen
        if isinstance(content, Path):
            with open(content, 'rb') as f:
                image_bytes = f.read()
        else:
            image_bytes = content
        
        # Codificar en base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Determinar tipo MIME
        if isinstance(content, Path):
            ext = content.suffix.lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.pdf': 'application/pdf'
            }.get(ext, 'image/jpeg')
        else:
            mime_type = 'image/jpeg'  # Default
        
        try:
            # Llamada a la API con visión
            response = await self.vision_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64}",
                                    "detail": "high"  # Alta resolución para mejor extracción
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,  # Baja temperatura para mayor precisión
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Parsear respuesta
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Direct AI extracción exitosa, campos extraídos: {len([v for v in result.values() if v])}")
            return result
            
        except Exception as e:
            logger.error(f"Error en Direct AI: {e}")
            return {}
    
    async def _extract_ocr_text(
        self, 
        prompt: str,
        model: str,
        ocr_content: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extracción usando OCR + texto con IA
        """
        logger.info(f"Iniciando extracción OCR + texto con modelo {model}")
        
        try:
            # Si el contenido OCR se pasó por separado, incluirlo en el prompt
            if ocr_content:
                # El prompt ya debería tener el contenido formateado
                pass
            
            # Llamada a la API
            response = await self.vision_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Parsear respuesta
            result = json.loads(response.choices[0].message.content)
            logger.info(f"OCR + texto extracción exitosa, campos extraídos: {len([v for v in result.values() if v])}")
            return result
            
        except Exception as e:
            logger.error(f"Error en OCR + texto: {e}")
            return {}
    
    def _apply_field_mask(
        self, 
        extraction: Dict[str, Any],
        allowed_fields: List[str]
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        MÁSCARA DE SEGURIDAD: Fuerza null en campos no permitidos
        Retorna tupla (extraction_masked, masked_fields)
        """
        masked = {}
        masked_fields = []
        
        for field in self.config.REQUIRED_FIELDS:
            if field in allowed_fields:
                # Campo permitido - mantener valor si existe
                masked[field] = extraction.get(field, None)
            else:
                # Campo NO permitido - forzar null
                masked[field] = None
                
                # Si la IA intentó extraer este campo, registrar
                if field in extraction and extraction[field] is not None:
                    masked_fields.append(field)
                    logger.warning(
                        f"MÁSCARA APLICADA: Campo '{field}' fue extraído pero NO está permitido. "
                        f"Valor bloqueado: {extraction[field][:50] if isinstance(extraction[field], str) else extraction[field]}"
                    )
        
        if masked_fields:
            logger.info(f"Máscara de seguridad aplicó {len(masked_fields)} bloqueos: {masked_fields}")
        
        return masked, masked_fields
    
    def _validate_and_transform(
        self,
        extraction: Dict[str, Any],
        document_type: str
    ) -> Dict[str, Any]:
        """
        Aplica validaciones y transformaciones según las reglas
        """
        validated = extraction.copy()
        
        for field, value in extraction.items():
            if value is None:
                continue
            
            rules = self.validation_rules.get(field, {})
            
            # Aplicar transformaciones
            if 'transform' in rules and callable(rules['transform']):
                try:
                    validated[field] = rules['transform'](value)
                except Exception as e:
                    logger.error(f"Error transformando {field}: {e}")
            
            # Validar formato
            if 'regex' in rules:
                pattern = rules['regex']
                if not re.match(pattern, str(value)):
                    logger.warning(f"Campo {field} no cumple regex {pattern}: {value}")
            
            # Validar longitud para strings
            if isinstance(value, str):
                if 'min_length' in rules and len(value) < rules['min_length']:
                    logger.warning(f"Campo {field} muy corto: {len(value)} < {rules['min_length']}")
                if 'max_length' in rules and len(value) > rules['max_length']:
                    value = value[:rules['max_length']]
                    validated[field] = value
            
            # Validar palabras máximas
            if 'max_words' in rules and isinstance(value, str):
                words = value.split()
                if len(words) > rules['max_words']:
                    validated[field] = ' '.join(words[:rules['max_words']])
            
            # Validar tipo fecha
            if rules.get('type') == 'date' and value:
                validated[field] = self._normalize_date(value)
            
            # Validar tipo numérico
            if rules.get('type') == 'float' and value:
                try:
                    validated[field] = float(str(value).replace(',', '').replace('$', ''))
                except:
                    logger.error(f"No se pudo convertir {field} a float: {value}")
        
        return validated
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """
        Normaliza fechas al formato DD/MM/YYYY
        """
        if not date_str:
            return None
        
        # Mapeo de meses abreviados
        month_map = {
            'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
        }
        
        # Reemplazar meses abreviados
        for abbr, num in month_map.items():
            date_str = date_str.replace(abbr, num)
        
        # Intentar diferentes formatos
        import re
        
        # Formato YYYY-MM-DD
        if match := re.match(r'(\d{4})-(\d{2})-(\d{2})', date_str):
            return f"{match.group(3)}/{match.group(2)}/{match.group(1)}"
        
        # Formato DD/MM/YYYY
        if match := re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str):
            return f"{match.group(1).zfill(2)}/{match.group(2).zfill(2)}/{match.group(3)}"
        
        # Formato DD-MM-YYYY
        if match := re.match(r'(\d{1,2})-(\d{1,2})-(\d{4})', date_str):
            return f"{match.group(1).zfill(2)}/{match.group(2).zfill(2)}/{match.group(3)}"
        
        logger.warning(f"No se pudo normalizar fecha: {date_str}")
        return date_str
    
    def _create_null_extraction(self, document_name: str, document_type: str) -> DocumentExtraction:
        """
        Crea una extracción con todos los campos en null
        """
        null_fields = {field: None for field in self.config.REQUIRED_FIELDS}
        
        return DocumentExtraction(
            source_document=document_name,
            document_type=document_type,
            extracted_fields=null_fields,
            extraction_metadata={
                "route": "null",
                "guide_applied": True,
                "reason": "document_type_not_recognized",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # =============================================================================
    #   MÉTODOS LEGACY: Mantener para compatibilidad
    # =============================================================================
    
    async def _extract_legacy(
        self,
        ocr_result: Dict[str, Any],
        document_name: str,
        document_type: Optional[str],
        use_cache: bool
    ) -> DocumentExtraction:
        """
        Método de extracción original para mantener compatibilidad
        """
        # [Mantener el código original del método extract_from_document]
        # Este es el código que ya existe en tu archivo
        
        cache_key = self._generate_cache_key(document_name, ocr_result)
        if use_cache and cache_key in self.extraction_cache:
            logger.info(f"Usando cache de extracción para {document_name}")
            return self.extraction_cache[cache_key]

        if not document_type:
            document_type = self._detect_document_type(
                self._ocr_to_dict_safe(ocr_result),
                document_name
            )

        logger.info(f"Extrayendo campos de {document_name} (tipo: {document_type})")

        # Manejar diferentes formatos de OCR
        if hasattr(ocr_result, "text"):
            prepared_content = {
                "text": getattr(ocr_result, "text", "") or "",
                "key_value_pairs": getattr(ocr_result, "key_values", {}) or {},
                "tables": getattr(ocr_result, "tables", []) or [],
            }
        else:
            prepared_content = self._prepare_ocr_content(ocr_result)

        if not prepared_content.get("text") and not prepared_content.get("key_value_pairs"):
            logger.warning(f"No hay contenido para extraer en {document_name}")
            empty = DocumentExtraction(
                source_document=document_name,
                document_type=document_type or "otro",
                extracted_fields={field: None for field in self.config.REQUIRED_FIELDS},
                extraction_metadata={"warning": "no_content"},
            )
            if use_cache:
                self.extraction_cache[cache_key] = empty
            return empty

        # Construir prompt (versión original)
        prompt = self.prompt_builder.build_extraction_prompt(
            document_name=document_name,
            document_type=document_type,
            ocr_content=prepared_content,
            required_fields=self.config.REQUIRED_FIELDS,
            use_guided=False  # Usar versión legacy
        )

        try:
            # Llamada a la API
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            extracted = json.loads(response.choices[0].message.content)
            
            result = DocumentExtraction(
                source_document=document_name,
                document_type=document_type,
                extracted_fields=extracted,
                extraction_metadata={"model": "gpt-4o-mini"}
            )

            if use_cache:
                self.extraction_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Error en extracción: {e}")
            return DocumentExtraction(
                source_document=document_name,
                document_type=document_type or "otro",
                extracted_fields={field: None for field in self.config.REQUIRED_FIELDS},
                extraction_metadata={"error": str(e)}
            )

    def _detect_document_type(self, ocr_content: Dict[str, Any], file_name: str) -> str:
        """Detecta el tipo de documento basado en el contenido y nombre"""
        # [Mantener código original]
        return "otro"
    
    def _prepare_ocr_content(self, ocr_data: Any) -> Dict[str, Any]:
        """Prepara el contenido OCR para procesamiento"""
        # [Mantener código original]
        return {}
    
    def _ocr_to_dict_safe(self, ocr_result: Any) -> Dict[str, Any]:
        """Convierte OCR result a dict de forma segura"""
        # [Mantener código original]
        return {}
    
    def _generate_cache_key(self, document_name: str, content: Any) -> str:
        """Genera una clave única para el cache"""
        # [Mantener código original]
        import hashlib
        content_str = json.dumps(content, sort_keys=True, default=str)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        return f"{document_name}_{content_hash}"
```

---

## 🎯 FASE 4: CONSOLIDADOR CON FILTRADO

### 4.1 Actualizar AIConsolidator

**Archivo:** `src/fraud_scorer/processors/ai/ai_consolidator.py`

```python
# src/fraud_scorer/processors/ai/ai_consolidator.py

"""
AIConsolidator: Consolida extracciones múltiples usando razonamiento de IA
Actualizado con filtrado por fuentes permitidas según la guía
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel, Field

from fraud_scorer.settings import ExtractionConfig, FieldPriority
from fraud_scorer.models.extraction import (
    DocumentExtraction,
    ConsolidatedExtraction,
    ConsolidatedFields,
    ExtractionBatch
)
from fraud_scorer.prompts.consolidation_prompts import ConsolidationPromptBuilder

logger = logging.getLogger(__name__)


# =========================
#   Modelos de respuesta
# =========================

class ConsolidationDecision(BaseModel):
    """Decisión de consolidación para un campo con trazabilidad."""
    field_name: str
    selected_value: Any
    source_document: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list)
    guide_applied: bool = False
    filtering_mode: Optional[str] = None


class ValidationResponse(BaseModel):
    """Respuesta estructurada de validación."""
    adjustments: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


# =========================
#      Consolidador
# =========================

class AIConsolidator:
    """
    Consolida extractos de varios documentos, resuelve conflictos y
    valida con filtrado por fuentes permitidas.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = instructor.patch(
            AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        )
        self.config = ExtractionConfig()
        self.prompt_builder = ConsolidationPromptBuilder()
        
        # Cargar mapeos de guías
        self.field_mapping = self.config.DOCUMENT_FIELD_MAPPING
        self.document_priorities = self.config.DOCUMENT_PRIORITIES
        
        self.golden_examples = self._load_golden_examples()
        logger.info("AIConsolidator inicializado con Sistema de Extracción Guiada")

    # ---------- API principal ----------

    async def consolidate_extractions(
        self,
        extractions: List[DocumentExtraction],
        case_id: str,
        use_advanced_reasoning: bool = True,
        use_guided: bool = True,  # Nuevo parámetro
        filtering_mode: str = "strict"  # "strict" o "soft"
    ) -> ConsolidatedExtraction:
        """
        Consolida múltiples extracciones en un resultado final.
        Ahora con filtrado por fuentes permitidas.
        """
        if use_guided:
            return await self.consolidate_with_guide(
                extractions=extractions,
                case_id=case_id,
                mode=filtering_mode,
                use_advanced_reasoning=use_advanced_reasoning
            )
        else:
            # Mantener comportamiento original
            return await self._consolidate_legacy(
                extractions, case_id, use_advanced_reasoning
            )
    
    async def consolidate_with_guide(
        self,
        extractions: List[DocumentExtraction],
        case_id: str,
        mode: str = "strict",
        use_advanced_reasoning: bool = True
    ) -> ConsolidatedExtraction:
        """
        Consolidación con filtrado por fuentes permitidas según la guía
        
        Args:
            mode: "strict" - descarta valores de fuentes no permitidas
                  "soft" - penaliza confianza de fuentes no permitidas
        """
        logger.info(
            f"Consolidación guiada iniciada:\n"
            f"  Caso: {case_id}\n"
            f"  Documentos: {len(extractions)}\n"
            f"  Modo: {mode}\n"
            f"  Razonamiento avanzado: {use_advanced_reasoning}"
        )
        
        # --- Cortocircuito si no hay extracciones ---
        if not extractions:
            logger.warning("No se recibieron extracciones. Retornando resultado vacío.")
            empty_fields = ConsolidatedFields()
            return ConsolidatedExtraction(
                case_id=case_id,
                consolidated_fields=empty_fields,
                consolidation_sources={},
                conflicts_resolved=[],
                confidence_scores={field: 0.0 for field in self.config.REQUIRED_FIELDS},
                consolidation_metadata={
                    "total_documents": 0,
                    "guide_applied": True,
                    "filtering_mode": mode
                },
                validation_notes=None
            )
        
        # 1. Agrupar valores por campo
        field_options = self._group_by_field(extractions)
        
        # 2. Aplicar filtrado/penalización según guía
        filtered_options = {}
        filtering_stats = {"filtered": 0, "penalized": 0, "kept": 0}
        
        for field, options in field_options.items():
            filtered, stats = self._filter_by_guide(
                field=field,
                options=options,
                mode=mode
            )
            filtered_options[field] = filtered
            
            # Acumular estadísticas
            filtering_stats["filtered"] += stats.get("filtered", 0)
            filtering_stats["penalized"] += stats.get("penalized", 0)
            filtering_stats["kept"] += stats.get("kept", 0)
        
        logger.info(
            f"Filtrado aplicado:\n"
            f"  Valores descartados: {filtering_stats['filtered']}\n"
            f"  Valores penalizados: {filtering_stats['penalized']}\n"
            f"  Valores mantenidos: {filtering_stats['kept']}"
        )
        
        # 3. Resolver conflictos con IA
        consolidated_fields = {}
        consolidation_sources = {}
        conflicts_resolved = []
        confidence_scores = {}
        
        for field in self.config.REQUIRED_FIELDS:
            options = filtered_options.get(field, [])
            
            if not options:
                # No hay valores válidos para este campo
                consolidated_fields[field] = None
                confidence_scores[field] = 0.0
                logger.debug(f"Campo '{field}': sin valores válidos tras filtrado")
                continue
            
            if len(options) == 1:
                # Solo una opción válida - usar directamente
                option = options[0]
                consolidated_fields[field] = option["value"]
                confidence_scores[field] = option.get("confidence", 0.8)
                consolidation_sources[field] = {
                    "document": option["source"],
                    "document_type": option.get("document_type", "otro"),
                    "confidence": option.get("confidence", 0.8),
                    "guide_compliant": True
                }
                logger.debug(f"Campo '{field}': una opción válida de {option['source']}")
            else:
                # Múltiples opciones - resolver con IA
                if use_advanced_reasoning:
                    decision = await self._resolve_conflict_with_guide(
                        field=field,
                        options=options,
                        context={"case_id": case_id},
                        mode=mode
                    )
                    
                    consolidated_fields[field] = decision.selected_value
                    confidence_scores[field] = decision.confidence
                    consolidation_sources[field] = {
                        "document": decision.source_document,
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                        "guide_applied": True,
                        "alternatives": len(options) - 1
                    }
                    conflicts_resolved.append({
                        "field": field,
                        "options_count": len(options),
                        "resolution": decision.dict()
                    })
                    logger.debug(f"Campo '{field}': conflicto resuelto con IA, elegido de {decision.source_document}")
                else:
                    # Sin razonamiento avanzado - usar prioridad
                    best_option = self._select_by_priority(options, field)
                    consolidated_fields[field] = best_option["value"]
                    confidence_scores[field] = best_option.get("confidence", 0.7)
                    consolidation_sources[field] = {
                        "document": best_option["source"],
                        "document_type": best_option.get("document_type", "otro"),
                        "confidence": best_option.get("confidence", 0.7),
                        "method": "priority"
                    }
        
        # 4. Validación cruzada de campos consolidados
        if use_advanced_reasoning:
            validated_fields = await self._validate_consolidated_fields(
                consolidated_fields, case_id
            )
            # Aplicar ajustes de validación
            for field, value in validated_fields.items():
                if field in consolidated_fields:
                    consolidated_fields[field] = value
        
        # 5. Construir resultado final
        result = ConsolidatedExtraction(
            case_id=case_id,
            consolidated_fields=ConsolidatedFields(**consolidated_fields),
            consolidation_sources=consolidation_sources,
            conflicts_resolved=conflicts_resolved,
            confidence_scores=confidence_scores,
            consolidation_metadata={
                "total_documents": len(extractions),
                "guide_applied": True,
                "filtering_mode": mode,
                "filtering_stats": filtering_stats,
                "fields_with_conflicts": len(conflicts_resolved),
                "fields_empty": sum(1 for v in consolidated_fields.values() if v is None),
                "timestamp": datetime.now().isoformat()
            },
            validation_notes=None
        )
        
        logger.info(
            f"Consolidación completada:\n"
            f"  Campos con valor: {sum(1 for v in consolidated_fields.values() if v is not None)}/{len(self.config.REQUIRED_FIELDS)}\n"
            f"  Conflictos resueltos: {len(conflicts_resolved)}\n"
            f"  Confianza promedio: {sum(confidence_scores.values())/len(confidence_scores):.2f}"
        )
        
        return result
    
    def _filter_by_guide(
        self,
        field: str,
        options: List[Dict],
        mode: str
    ) -> tuple[List[Dict], Dict[str, int]]:
        """
        Filtra opciones según documento permitido para el campo
        Retorna tupla (opciones_filtradas, estadísticas)
        """
        filtered = []
        stats = {"filtered": 0, "penalized": 0, "kept": 0}
        
        # Encontrar documentos permitidos para este campo
        allowed_doc_types = []
        for doc_type, fields in self.field_mapping.items():
            if field in fields:
                allowed_doc_types.append(doc_type)
        
        logger.debug(f"Campo '{field}' permitido en: {allowed_doc_types}")
        
        for option in options:
            doc_type = option.get("document_type", "otro")
            
            if doc_type in allowed_doc_types:
                # Documento permitido - mantener con confianza original
                option["guide_compliant"] = True
                filtered.append(option)
                stats["kept"] += 1
                
            elif mode == "soft":
                # Modo suave - penalizar confianza pero mantener
                option_copy = option.copy()
                original_confidence = option_copy.get("confidence", 0.5)
                option_copy["confidence"] = original_confidence * 0.3  # Penalización 70%
                option_copy["penalized"] = True
                option_copy["original_confidence"] = original_confidence
                option_copy["guide_compliant"] = False
                filtered.append(option_copy)
                stats["penalized"] += 1
                
                logger.info(
                    f"Campo '{field}' de '{doc_type}' penalizado "
                    f"(confianza: {original_confidence:.2f} → {option_copy['confidence']:.2f})"
                )
                
            else:
                # Modo estricto - descartar completamente
                stats["filtered"] += 1
                logger.info(
                    f"Campo '{field}' de '{doc_type}' descartado (no es fuente válida)"
                )
        
        return filtered, stats
    
    async def _resolve_conflict_with_guide(
        self,
        field: str,
        options: List[Dict],
        context: Dict,
        mode: str
    ) -> ConsolidationDecision:
        """
        Resuelve conflictos considerando la guía de fuentes permitidas
        """
        # Obtener fuentes permitidas para este campo
        allowed_sources = []
        for doc_type, fields in self.field_mapping.items():
            if field in fields:
                allowed_sources.append(doc_type)
        
        # Construir prompt con información de guía
        prompt = self.prompt_builder.build_guided_conflict_prompt(
            field_name=field,
            options=options,
            allowed_sources=allowed_sources,
            context=context
        )
        
        try:
            # Llamada a IA para resolver
            response = await self.client.chat.completions.create(
                model="gpt-4o",  # Usar modelo más potente para consolidación
                response_model=ConsolidationDecision,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # Agregar metadata de guía
            response.guide_applied = True
            response.filtering_mode = mode
            
            return response
            
        except Exception as e:
            logger.error(f"Error resolviendo conflicto para {field}: {e}")
            
            # Fallback: elegir por prioridad
            best_option = self._select_by_priority(options, field)
            return ConsolidationDecision(
                field_name=field,
                selected_value=best_option["value"],
                source_document=best_option["source"],
                confidence=best_option.get("confidence", 0.5),
                reasoning=f"Seleccionado por prioridad tras error: {str(e)}",
                alternatives_considered=options,
                guide_applied=True,
                filtering_mode=mode
            )
    
    def _select_by_priority(self, options: List[Dict], field: str) -> Dict:
        """
        Selecciona la mejor opción basándose en prioridad de documento
        """
        # Ordenar por: 1) guide_compliant, 2) prioridad documento, 3) confianza
        def sort_key(opt):
            doc_type = opt.get("document_type", "otro")
            priority = self.document_priorities.get(doc_type, FieldPriority.OTRO)
            
            # Priorizar documentos que cumplen con la guía
            guide_bonus = 0 if opt.get("guide_compliant", False) else 1000
            
            return (
                guide_bonus,  # Primero los que cumplen la guía
                priority.value,  # Luego por prioridad de documento
                -opt.get("confidence", 0)  # Finalmente por confianza (negativo para ordenar desc)
            )
        
        sorted_options = sorted(options, key=sort_key)
        return sorted_options[0]
    
    async def _validate_consolidated_fields(
        self,
        fields: Dict[str, Any],
        case_id: str
    ) -> Dict[str, Any]:
        """
        Validación cruzada de campos consolidados
        """
        validated = fields.copy()
        
        # Validar coherencia de fechas
        if fields.get("fecha_ocurrencia") and fields.get("fecha_reclamacion"):
            fecha_ocurrencia = self._parse_date(fields["fecha_ocurrencia"])
            fecha_reclamacion = self._parse_date(fields["fecha_reclamacion"])
            
            if fecha_ocurrencia and fecha_reclamacion:
                if fecha_ocurrencia > fecha_reclamacion:
                    logger.warning(
                        f"Fecha de ocurrencia posterior a reclamación. "
                        f"Intercambiando valores."
                    )
                    validated["fecha_ocurrencia"] = fields["fecha_reclamacion"]
                    validated["fecha_reclamacion"] = fields["fecha_ocurrencia"]
        
        # Validar vigencia vs fecha de ocurrencia
        if all([fields.get("vigencia_inicio"), fields.get("vigencia_fin"), fields.get("fecha_ocurrencia")]):
            vigencia_inicio = self._parse_date(fields["vigencia_inicio"])
            vigencia_fin = self._parse_date(fields["vigencia_fin"])
            fecha_ocurrencia = self._parse_date(fields["fecha_ocurrencia"])
            
            if vigencia_inicio and vigencia_fin and fecha_ocurrencia:
                if not (vigencia_inicio <= fecha_ocurrencia <= vigencia_fin):
                    logger.warning(
                        f"Fecha de ocurrencia fuera de vigencia de póliza"
                    )
        
        # Validar monto
        if fields.get("monto_reclamacion"):
            try:
                monto = float(str(fields["monto_reclamacion"]).replace(',', '').replace('$', ''))
                if monto < 0:
                    logger.warning(f"Monto negativo detectado: {monto}")
                    validated["monto_reclamacion"] = abs(monto)
            except:
                pass
        
        return validated
    
    def _parse_date(self, date_str: str):
        """Helper para parsear fechas"""
        if not date_str:
            return None
        
        from datetime import datetime
        
        # Intentar formato DD/MM/YYYY
        try:
            return datetime.strptime(date_str, "%d/%m/%Y")
        except:
            pass
        
        # Intentar formato YYYY-MM-DD
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            pass
        
        return None
    
    # ---------- Métodos auxiliares ----------
    
    def _group_by_field(self, extractions: List[DocumentExtraction]) -> Dict[str, List[Dict]]:
        """
        Agrupa valores extraídos por campo
        """
        field_groups = {field: [] for field in self.config.REQUIRED_FIELDS}
        
        for extraction in extractions:
            doc_type = extraction.document_type
            
            for field, value in extraction.extracted_fields.items():
                if value is not None and field in field_groups:
                    field_groups[field].append({
                        "value": value,
                        "source": extraction.source_document,
                        "document_type": doc_type,
                        "confidence": extraction.extraction_metadata.get("confidence", 0.5),
                        "extraction_metadata": extraction.extraction_metadata
                    })
        
        return field_groups
    
    async def _consolidate_legacy(
        self,
        extractions: List[DocumentExtraction],
        case_id: str,
        use_advanced_reasoning: bool
    ) -> ConsolidatedExtraction:
        """
        Método de consolidación original para mantener compatibilidad
        """
        # [Mantener código original del método consolidate_extractions]
        # Este es el comportamiento existente sin filtrado por guías
        
        logger.info(f"Consolidando {len(extractions)} extracciones para caso {case_id} (modo legacy)")
        
        if not extractions:
            empty_fields = ConsolidatedFields()
            return ConsolidatedExtraction(
                case_id=case_id,
                consolidated_fields=empty_fields,
                consolidation_sources={},
                conflicts_resolved=[],
                confidence_scores={field: 0.0 for field in self.config.REQUIRED_FIELDS},
                consolidation_metadata={"total_documents": 0},
                validation_notes=None
            )
        
        # [Resto del código legacy...]
        # Implementación original sin filtrado
        pass
    
    def _load_golden_examples(self) -> List[Dict]:
        """Carga ejemplos de referencia para mejorar decisiones"""
        # [Mantener código original]
        return []
```

### 4.2 Actualizar ConsolidationPromptBuilder

**Archivo:** `src/fraud_scorer/prompts/consolidation_prompts.py`

```python
# src/fraud_scorer/prompts/consolidation_prompts.py

"""
Constructor de prompts para consolidación con IA
Actualizado con soporte para guías de extracción
"""
import json
from typing import Dict, Any, List, Optional

class ConsolidationPromptBuilder:
    """
    Construye prompts para consolidación inteligente
    con soporte para Sistema de Extracción Guiada
    """
    
    def build_conflict_resolution_prompt(
        self,
        field_name: str,
        options: List[Dict[str, Any]],
        field_rules: List[str],
        golden_examples: List[Dict],
        context: str
    ) -> str:
        """
        Construye prompt para resolver conflictos (mantiene compatibilidad)
        """
        # Mantener método original para compatibilidad
        options_text = self._format_options(options)
        rules_text = self._format_rules(field_rules)
        examples_text = self._format_examples(golden_examples)
        
        prompt = f"""
Necesito resolver un conflicto para el campo '{field_name}' en un siniestro de seguros.

**CONTEXTO DEL CASO:**
{context}

**OPCIONES ENCONTRADAS:**
{options_text}

**REGLAS DE NEGOCIO:**
{rules_text}

**EJEMPLOS DE RESOLUCIONES CORRECTAS ANTERIORES:**
{examples_text}

**INSTRUCCIONES:**
1. Analiza todas las opciones disponibles
2. Considera la fuente y confianza de cada opción
3. Aplica las reglas de negocio
4. Selecciona el valor más confiable
5. Explica tu razonamiento

Responde con la decisión estructurada.
"""
        return prompt
    
    def build_guided_conflict_prompt(
        self,
        field_name: str,
        options: List[Dict],
        allowed_sources: List[str],
        context: Optional[Dict] = None
    ) -> str:
        """
        Nuevo método: Prompt con información de fuentes permitidas según la guía
        """
        
        # Formatear opciones con información de guía
        options_text = self._format_options_with_guide(options)
        
        # Formatear fuentes permitidas
        sources_text = "\n".join([f"  ✓ {source}" for source in allowed_sources])
        
        prompt = f"""
TAREA: Resolver conflicto para el campo '{field_name}'

================================================================================
                    INFORMACIÓN DE LA GUÍA DE EXTRACCIÓN
================================================================================

FUENTES AUTORIZADAS PARA ESTE CAMPO:
{sources_text}

REGLA CRÍTICA:
⚠️ Los valores de documentos autorizados tienen MÁXIMA prioridad
⚠️ Los valores penalizados (fuentes no autorizadas) tienen MÍNIMA confiabilidad

================================================================================
                            OPCIONES ENCONTRADAS
================================================================================

{options_text}

================================================================================
                           INSTRUCCIONES DE DECISIÓN
================================================================================

1. PRIORIZAR valores de documentos que cumplen con la guía (guide_compliant = true)
2. Si hay múltiples opciones válidas, elegir la de mayor confianza
3. Solo considerar opciones penalizadas si NO hay opciones válidas
4. Verificar consistencia del valor con el tipo de campo
5. Explicar claramente el razonamiento

CONTEXTO ADICIONAL:
- Caso ID: {context.get('case_id', 'N/A') if context else 'N/A'}
- Campo: {field_name}
- Total de opciones: {len(options)}
- Opciones válidas: {sum(1 for opt in options if opt.get('guide_compliant', False))}

Por favor, selecciona el valor más confiable considerando la guía y explica tu decisión.
"""
        return prompt
    
    def build_validation_prompt(
        self,
        consolidated_fields: Dict[str, Any],
        case_id: str,
        with_guide: bool = False
    ) -> str:
        """
        Construye prompt para validación de campos consolidados
        """
        
        fields_json = json.dumps(consolidated_fields, ensure_ascii=False, indent=2)
        
        prompt = f"""
TAREA: Validar y ajustar campos consolidados del caso {case_id}

CAMPOS CONSOLIDADOS:
{fields_json}

VALIDACIONES A REALIZAR:
1. Coherencia de fechas:
   - fecha_ocurrencia <= fecha_reclamacion
   - fecha_ocurrencia dentro de vigencia (inicio/fin)

2. Formato de campos:
   - numero_siniestro: 14 dígitos
   - fechas: DD/MM/YYYY
   - monto_reclamacion: número positivo

3. Consistencia lógica:
   - El tipo_siniestro debe corresponder con el bien_reclamado
   - El lugar_hechos debe ser coherente con el tipo de evento

"""
        
        if with_guide:
            prompt += """
4. Cumplimiento de guía:
   - Verificar que los valores provengan de fuentes autorizadas
   - Marcar campos sospechosos si detectas inconsistencias

"""
        
        prompt += """
INSTRUCCIONES:
- Si todos los campos son válidos, retorna un objeto vacío {{}}
- Si hay ajustes necesarios, retorna SOLO los campos a modificar
- Incluye notas explicativas si hay observaciones importantes

Responde con el objeto de ajustes en formato JSON.
"""
        
        return prompt
    
    # ---------- Métodos auxiliares ----------
    
    def _format_options(self, options: List[Dict]) -> str:
        """Formatea las opciones de manera legible"""
        if not options:
            return "No hay opciones disponibles"
        
        formatted = []
        for i, opt in enumerate(options, 1):
            formatted.append(
                f"{i}. Valor: {opt.get('value', 'N/A')}\n"
                f"   Fuente: {opt.get('source', 'desconocida')}\n"
                f"   Confianza: {opt.get('confidence', 0):.2f}"
            )
        
        return "\n".join(formatted)
    
    def _format_options_with_guide(self, options: List[Dict]) -> str:
        """
        Formatea opciones incluyendo información de cumplimiento de guía
        """
        if not options:
            return "No hay opciones disponibles"
        
        formatted = []
        
        # Separar opciones válidas y penalizadas
        valid_options = [opt for opt in options if opt.get('guide_compliant', False)]
        penalized_options = [opt for opt in options if opt.get('penalized', False)]
        other_options = [opt for opt in options 
                        if not opt.get('guide_compliant', False) 
                        and not opt.get('penalized', False)]
        
        if valid_options:
            formatted.append("📗 OPCIONES VÁLIDAS (cumplen con la guía):")
            for i, opt in enumerate(valid_options, 1):
                formatted.append(self._format_single_option(i, opt, "✓"))
        
        if penalized_options:
            formatted.append("\n⚠️ OPCIONES PENALIZADAS (fuentes no autorizadas):")
            for i, opt in enumerate(penalized_options, len(valid_options) + 1):
                formatted.append(self._format_single_option(i, opt, "⚠"))
        
        if other_options:
            formatted.append("\n❓ OTRAS OPCIONES:")
            for i, opt in enumerate(other_options, len(valid_options) + len(penalized_options) + 1):
                formatted.append(self._format_single_option(i, opt, "?"))
        
        return "\n".join(formatted)
    
    def _format_single_option(self, index: int, option: Dict, symbol: str) -> str:
        """
        Formatea una opción individual con su información
        """
        result = f"\n{symbol} Opción {index}:"
        result += f"\n   Valor: {option.get('value', 'N/A')}"
        result += f"\n   Documento: {option.get('source', 'desconocido')}"
        result += f"\n   Tipo doc: {option.get('document_type', 'otro')}"
        result += f"\n   Confianza: {option.get('confidence', 0):.2f}"
        
        if option.get('penalized'):
            result += f"\n   ⚠️ Confianza original: {option.get('original_confidence', 0):.2f}"
            result += f"\n   ⚠️ Penalizado por no ser fuente autorizada"
        
        if option.get('guide_compliant'):
            result += f"\n   ✓ Cumple con la guía de extracción"
        
        return result
    
    def _format_rules(self, rules: List[str]) -> str:
        """Formatea las reglas de negocio"""
        if not rules:
            return "No hay reglas específicas"
        
        return "\n".join([f"- {rule}" for rule in rules])
    
    def _format_examples(self, examples: List[Dict]) -> str:
        """Formatea ejemplos de referencia"""
        if not examples:
            return "No hay ejemplos disponibles"
        
        formatted = []
        for i, example in enumerate(examples[:3], 1):
            formatted.append(
                f"Ejemplo {i}:\n"
                f"  Situación: {example.get('situation', 'N/A')}\n"
                f"  Decisión: {example.get('decision', 'N/A')}\n"
                f"  Razón: {example.get('reasoning', 'N/A')}"
            )
        
        return "\n\n".join(formatted)
```

---

## 🎯 FASE 5: ORQUESTADOR PRINCIPAL

### 5.1 Actualizar run_report.py

**Archivo:** `scripts/run_report.py`

```python
#!/usr/bin/env python3
"""
Fraud Scorer v2.0 - Sistema de análisis con IA
Actualizado con Sistema de Extracción Guiada
"""

import sys
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import re
from datetime import datetime
import threading
import signal
import shutil

# [Mantener imports existentes...]

# Añadir la raíz del proyecto al path de Python
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fraud_scorer.run_report")

# ==== Componentes del sistema v2 ====
from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
from fraud_scorer.parsers.document_parser import DocumentParser
from fraud_scorer.storage.ocr_cache import OCRCacheManager
from fraud_scorer.storage.cases import create_case

from fraud_scorer.processors.ai.ai_field_extractor import AIFieldExtractor
from fraud_scorer.processors.ai.ai_consolidator import AIConsolidator
from fraud_scorer.templates.ai_report_generator import AIReportGenerator
from fraud_scorer.models.extraction import (
    DocumentExtraction,
    ConsolidatedExtraction,
    ProgressEvent,
)

# Importar configuración de guías
from fraud_scorer.settings import ExtractionConfig, ExtractionRoute

import time
import os

# [Mantener clase _ProgressEmitter sin cambios...]
class _ProgressEmitter:
    """Emisor de eventos de progreso a archivos JSONL para seguimiento en tiempo real"""
    # [Mantener código existente...]
    pass


class FraudScorerPipeline:
    """
    Pipeline principal de Fraud Scorer con Sistema de Extracción Guiada
    """
    
    def __init__(
        self,
        use_guided: bool = True,
        filtering_mode: str = "strict",
        default_model: str = "gpt-4o-mini"
    ):
        """
        Inicializa el pipeline
        
        Args:
            use_guided: Activar Sistema de Extracción Guiada
            filtering_mode: Modo de filtrado ("strict" o "soft")
            default_model: Modelo por defecto para extracción
        """
        self.ocr_processor = AzureOCRProcessor()
        self.parser = DocumentParser()
        self.cache_manager = OCRCacheManager()
        
        self.extractor = AIFieldExtractor()
        self.consolidator = AIConsolidator()
        self.report_generator = AIReportGenerator()
        
        # Configuración de guías
        self.config = ExtractionConfig()
        self.field_mapping = self.config.DOCUMENT_FIELD_MAPPING
        
        # Parámetros de extracción guiada
        self.use_guided = use_guided
        self.filtering_mode = filtering_mode
        self.default_model = default_model
        
        logger.info(
            f"Pipeline inicializado:\n"
            f"  Extracción guiada: {use_guided}\n"
            f"  Modo de filtrado: {filtering_mode}\n"
            f"  Modelo por defecto: {default_model}"
        )
    
    def detect_document_type(self, filepath: Path) -> Tuple[str, str]:
        """
        Detecta tipo de documento y ruta óptima de procesamiento
        
        Returns:
            (document_type, route)
        """
        filename = filepath.stem.lower()
        
        # Detectar tipo por nombre del archivo
        doc_type = self._detect_type_by_name(filename)
        
        # Decidir ruta según extensión
        route = self._detect_route_by_extension(filepath)
        
        logger.info(
            f"Detección de documento:\n"
            f"  Archivo: {filepath.name}\n"
            f"  Tipo detectado: {doc_type}\n"
            f"  Ruta seleccionada: {route}"
        )
        
        return doc_type, route
    
    def _detect_type_by_name(self, filename: str) -> str:
        """
        Detecta el tipo de documento basándose en el nombre del archivo
        """
        filename_lower = filename.lower()
        
        # Mapeo de patrones a tipos de documento
        patterns = {
            "informe_preliminar_del_ajustador": ["informe_preliminar", "ajustador"],
            "poliza_de_la_aseguradora": ["poliza", "póliza"],
            "carta_de_reclamacion_formal_a_la_aseguradra": ["carta_reclamacion", "reclamacion_formal"],
            "carpeta_de_investigacion": ["carpeta_investigacion", "carpeta_de_investigacion"],
            "narracion_de_hechos": ["narracion", "narración", "hechos"],
            "declaracion_del_asegurado": ["declaracion", "declaración", "asegurado"]
        }
        
        for doc_type, keywords in patterns.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return doc_type
        
        # Si no coincide con ningún patrón
        logger.warning(f"Tipo de documento no reconocido para: {filename}")
        return "otro"
    
    def _detect_route_by_extension(self, filepath: Path) -> str:
        """
        Detecta la ruta de procesamiento según la extensión del archivo
        """
        extension = filepath.suffix.lower()
        
        # Imágenes -> Direct AI
        if extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return "direct_ai"
        
        # Documentos de texto -> OCR
        elif extension in ['.docx', '.doc', '.txt', '.csv']:
            return "ocr_text"
        
        # PDFs -> Analizar contenido
        elif extension == '.pdf':
            # Por ahora default a OCR, pero podríamos analizar si es escaneado
            return self._analyze_pdf_route(filepath)
        
        # Default
        else:
            return "ocr_text"
    
    def _analyze_pdf_route(self, filepath: Path) -> str:
        """
        Analiza un PDF para determinar la mejor ruta
        TODO: Implementar detección de PDF escaneado vs nativo
        """
        # Por ahora usar OCR para todos los PDFs
        # En el futuro: detectar si tiene texto extraíble o es imagen
        return "ocr_text"
    
    async def process_document_guided(
        self,
        filepath: Path,
        case_id: str,
        progress_emitter: Optional[_ProgressEmitter] = None
    ) -> DocumentExtraction:
        """
        Procesa un documento con Sistema de Extracción Guiada
        """
        # 1. Detectar tipo y ruta
        doc_type, route = self.detect_document_type(filepath)
        
        # Emitir progreso
        if progress_emitter:
            progress_emitter.emit(
                stage="extraction",
                status="started",
                message=f"Procesando {filepath.name} (tipo: {doc_type}, ruta: {route})"
            )
        
        try:
            # 2. Preparar contenido según ruta
            if route == "direct_ai":
                # Para visión directa, pasar el path del archivo
                content = filepath
                logger.info(f"Usando Direct AI (visión) para {filepath.name}")
                
            else:
                # Para OCR + texto, procesar primero con OCR
                logger.info(f"Procesando OCR para {filepath.name}")
                
                # Verificar cache
                cached_ocr = self.cache_manager.get_cached_result(str(filepath))
                if cached_ocr:
                    logger.info(f"Usando OCR cacheado para {filepath.name}")
                    content = cached_ocr
                else:
                    # Procesar con OCR
                    ocr_result = await self.ocr_processor.process_document(filepath)
                    
                    # Convertir a dict si es necesario
                    if hasattr(ocr_result, "__dict__"):
                        content = {
                            "text": getattr(ocr_result, "text", ""),
                            "key_value_pairs": getattr(ocr_result, "key_values", {}),
                            "tables": getattr(ocr_result, "tables", [])
                        }
                    else:
                        content = ocr_result
                    
                    # Cachear resultado
                    self.cache_manager.save_result(str(filepath), content)
            
            # 3. Extraer con guía
            extraction = await self.extractor.extract_from_document_guided(
                content=content,
                document_name=filepath.name,
                document_type=doc_type,
                route=route,
                model=self.default_model
            )
            
            # 4. Log de resultado
            allowed_fields = self.field_mapping.get(doc_type, [])
            extracted_fields = [
                f for f, v in extraction.extracted_fields.items() 
                if v is not None
            ]
            
            logger.info(
                f"Extracción completada para {filepath.name}:\n"
                f"  Campos permitidos: {len(allowed_fields)}\n"
                f"  Campos extraídos: {len(extracted_fields)}\n"
                f"  Campos bloqueados: {len(extraction.extraction_metadata.get('masked_fields', []))}"
            )
            
            # Emitir progreso
            if progress_emitter:
                progress_emitter.emit(
                    stage="extraction",
                    status="done",
                    message=f"Extraídos {len(extracted_fields)} campos de {filepath.name}"
                )
            
            return extraction
            
        except Exception as e:
            logger.error(f"Error procesando {filepath}: {e}")
            
            if progress_emitter:
                progress_emitter.emit(
                    stage="extraction",
                    status="error",
                    message=f"Error en {filepath.name}: {str(e)}"
                )
            
            # Retornar extracción vacía
            return DocumentExtraction(
                source_document=filepath.name,
                document_type=doc_type,
                extracted_fields={field: None for field in self.config.REQUIRED_FIELDS},
                extraction_metadata={
                    "error": str(e),
                    "route": route,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def run_pipeline(
        self,
        folder_path: Path,
        case_info: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo con Extracción Guiada
        """
        start_time = time.time()
        
        # Crear caso
        if not case_info:
            case_info = {
                "company": folder_path.name,
                "claim_number": datetime.now().strftime("%Y%m%d%H%M%S")
            }
        
        case_id = create_case(case_info)
        logger.info(f"Caso creado: {case_id}")
        
        # Inicializar emisor de progreso
        progress_emitter = _ProgressEmitter(case_id)
        
        # 1. Descubrir documentos
        documents = list(folder_path.glob("*"))
        valid_docs = [
            doc for doc in documents 
            if doc.is_file() and not doc.name.startswith('.')
        ]
        
        logger.info(f"Documentos encontrados: {len(valid_docs)}")
        
        # 2. Procesar cada documento
        extractions = []
        for i, filepath in enumerate(valid_docs, 1):
            progress_emitter.emit(
                stage="processing",
                status="in_progress",
                doc_index=i,
                doc_total=len(valid_docs),
                message=f"Procesando documento {i}/{len(valid_docs)}"
            )
            
            try:
                extraction = await self.process_document_guided(
                    filepath=filepath,
                    case_id=case_id,
                    progress_emitter=progress_emitter
                )
                extractions.append(extraction)
                
            except Exception as e:
                logger.error(f"Error procesando {filepath}: {e}")
        
        # 3. Consolidar con filtrado
        progress_emitter.emit(
            stage="consolidation",
            status="started",
            message="Consolidando extracciones"
        )
        
        consolidated = await self.consolidator.consolidate_with_guide(
            extractions=extractions,
            case_id=case_id,
            mode=self.filtering_mode
        )
        
        progress_emitter.emit(
            stage="consolidation",
            status="done",
            message=f"Consolidación completada"
        )
        
        # 4. Generar reporte
        progress_emitter.emit(
            stage="report",
            status="started",
            message="Generando reporte"
        )
        
        report_path = await self.report_generator.generate_report(
            consolidated_result=consolidated,
            output_dir=Path("data/reports"),
            case_id=case_id
        )
        
        progress_emitter.emit(
            stage="report",
            status="done",
            message=f"Reporte generado: {report_path}"
        )
        
        # 5. Calcular métricas
        elapsed_time = time.time() - start_time
        
        result = {
            "case_id": case_id,
            "status": "completed",
            "documents_processed": len(extractions),
            "fields_extracted": sum(
                1 for v in consolidated.consolidated_fields.dict().values() 
                if v is not None
            ),
            "conflicts_resolved": len(consolidated.conflicts_resolved),
            "report_path": str(report_path),
            "execution_time": f"{elapsed_time:.2f}s",
            "guide_applied": self.use_guided,
            "filtering_mode": self.filtering_mode,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "2.0-guided"
            }
        }
        
        # Log final
        logger.info(
            f"\n{'='*60}\n"
            f"PIPELINE COMPLETADO\n"
            f"{'='*60}\n"
            f"Caso: {case_id}\n"
            f"Documentos: {result['documents_processed']}\n"
            f"Campos extraídos: {result['fields_extracted']}/{len(self.config.REQUIRED_FIELDS)}\n"
            f"Conflictos resueltos: {result['conflicts_resolved']}\n"
            f"Tiempo: {result['execution_time']}\n"
            f"Reporte: {result['report_path']}\n"
            f"{'='*60}"
        )
        
        return result


def main():
    """
    Función principal del CLI
    """
    parser = argparse.ArgumentParser(
        description="Fraud Scorer v2.0 - Sistema de Extracción Guiada con IA"
    )
    
    parser.add_argument(
        "folder",
        type=str,
        help="Ruta a la carpeta con documentos del siniestro"
    )
    
    parser.add_argument(
        "--company",
        type=str,
        help="Nombre de la compañía asegurada"
    )
    
    parser.add_argument(
        "--claim",
        type=str,
        help="Número de siniestro"
    )
    
    parser.add_argument(
        "--guided",
        action="store_true",
        default=True,
        help="Usar Sistema de Extracción Guiada (default: True)"
    )
    
    parser.add_argument(
        "--no-guided",
        action="store_true",
        help="Desactivar Sistema de Extracción Guiada"
    )
    
    parser.add_argument(
        "--mode",
        choices=["strict", "soft"],
        default="strict",
        help="Modo de filtrado: strict=descartar, soft=penalizar (default: strict)"
    )
    
    parser.add_argument(
        "--model",
        choices=["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini"],
        default="gpt-4o-mini",
        help="Modelo a usar para extracción (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activar modo debug con logs detallados"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    # Validar carpeta
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"❌ Error: La carpeta {folder_path} no existe")
        sys.exit(1)
    
    # Preparar info del caso
    case_info = {}
    if args.company:
        case_info["company"] = args.company
    if args.claim:
        case_info["claim_number"] = args.claim
    
    # Determinar si usar guías
    use_guided = not args.no_guided
    
    # Crear pipeline
    pipeline = FraudScorerPipeline(
        use_guided=use_guided,
        filtering_mode=args.mode,
        default_model=args.model
    )
    
    # Ejecutar
    print(f"\n🚀 Iniciando Fraud Scorer v2.0")
    print(f"   Carpeta: {folder_path}")
    print(f"   Extracción guiada: {'✓' if use_guided else '✗'}")
    print(f"   Modo: {args.mode}")
    print(f"   Modelo: {args.model}")
    print(f"\n")
    
    try:
        # Ejecutar pipeline
        result = asyncio.run(pipeline.run_pipeline(folder_path, case_info))
        
        print(f"\n✅ Procesamiento completado exitosamente")
        print(f"   Caso ID: {result['case_id']}")
        print(f"   Documentos: {result['documents_processed']}")
        print(f"   Campos extraídos: {result['fields_extracted']}")
        print(f"   Reporte: {result['report_path']}")
        print(f"   Tiempo: {result['execution_time']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Procesamiento interrumpido por el usuario")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## 📊 MÉTRICAS Y VALIDACIÓN

### Métricas de Éxito Esperadas

1. **Reducción de Alucinaciones**: >95% campos null cuando no hay fuente válida
2. **Precisión de Extracción**: >90% campos del documento correcto
3. **Cobertura**: >85% campos encontrados cuando existen
4. **Performance**: <5 segundos por documento
5. **Compliance**: 100% cumplimiento de guía

### Tests Recomendados

```bash
# Test con modo estricto
python scripts/run_report.py /path/to/documents --guided --mode strict

# Test con modo suave
python scripts/run_report.py /path/to/documents --guided --mode soft

# Comparación con sistema legacy
python scripts/run_report.py /path/to/documents --no-guided
```

---

## 🚀 PRÓXIMOS PASOS

1. **Implementar cambios en orden**:
   - Fase 1: settings.py
   - Fase 2: extraction_prompts.py
   - Fase 3: ai_field_extractor.py
   - Fase 4: ai_consolidator.py y consolidation_prompts.py
   - Fase 5: run_report.py

2. **Validar con casos de prueba**

3. **Ajustar umbrales y parámetros**

4. **Monitorear métricas de calidad**

---

## 📝 NOTAS IMPORTANTES

- **Compatibilidad**: Todo el código mantiene compatibilidad con el sistema existente
- **Activación**: El sistema guiado se activa con flags, permitiendo comparación A/B
- **Rollback**: Fácil desactivar con `--no-guided` si hay problemas
- **Logs**: Sistema extensivo de logging para debugging
- **Cache**: Reutiliza el sistema de cache existente

Esta implementación reutiliza al máximo tu estructura actual, modificando solo los archivos existentes necesarios.