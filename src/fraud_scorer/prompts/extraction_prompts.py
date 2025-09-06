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
    
    def _build_legacy_prompt(
        self,
        document_name: str,
        document_type: str,
        ocr_content: Dict[str, Any],
        required_fields: List[str]
    ) -> str:
        """
        Construye un prompt legacy (versión original)
        """
        # Formatear la lista de campos con descripciones
        fields_section = self._format_fields_section(required_fields)
        
        # Obtener ejemplos relevantes
        examples_section = self._format_examples_section(document_type)
        
        # Formatear el contenido del OCR
        ocr_section = self._format_ocr_content(ocr_content)
        
        # Construir el prompt final
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
    
    def _format_fields_section(self, required_fields: List[str]) -> str:
        """
        Formatea la sección de campos a extraer con sus descripciones
        """
        lines = []
        for field in required_fields:
            desc = self.field_descriptions.get(field, "")
            lines.append(f"- {field}: {desc}")
        return "\n".join(lines)
    
    def _format_examples_section(self, document_type: str) -> str:
        """
        Incluye ejemplos relevantes según el tipo de documento
        """
        if document_type not in self.examples:
            return ""
        
        examples = self.examples.get(document_type, [])
        if not examples:
            return ""
        
        section = "\n**EJEMPLOS DE REFERENCIA:**\n"
        for i, example in enumerate(examples[:2], 1):  # Máximo 2 ejemplos
            section += f"\nEjemplo {i}:\n"
            section += f"Entrada: {example['input'][:200]}...\n"
            section += f"Extracción correcta: {json.dumps(example['output'], ensure_ascii=False, indent=2)}\n"
        
        return section
    
    def _format_ocr_content(self, ocr_content: Dict[str, Any]) -> str:
        """
        Formatea el contenido del OCR de manera estructurada
        """
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
        # Aquí podrías cargar desde archivo
        return ""
    
    def _load_field_descriptions(self) -> Dict[str, str]:
        """Carga las descripciones de los campos"""
        return {
            "numero_siniestro": "Número único del siniestro o reclamación",
            "nombre_asegurado": "Nombre completo del asegurado o empresa asegurada",
            "numero_poliza": "Número de la póliza de seguro",
            "vigencia_inicio": "Fecha de inicio de vigencia de la póliza",
            "vigencia_fin": "Fecha de fin de vigencia de la póliza",
            "domicilio_poliza": "Dirección completa registrada en la póliza",
            "bien_reclamado": "Descripción del bien o mercancía reclamada",
            "monto_reclamacion": "Monto total de la reclamación",
            "tipo_siniestro": "Tipo de siniestro (robo, colisión, incendio, etc.)",
            "fecha_ocurrencia": "Fecha cuando ocurrió el siniestro",
            "fecha_reclamacion": "Fecha cuando se presentó la reclamación",
            "lugar_hechos": "Lugar donde ocurrió el siniestro",
            "ajuste": "Nombre del ajustador asignado",
            "conclusiones": "Conclusiones o resolución del caso"
        }
    
    def _load_examples(self) -> Dict[str, List[Dict]]:
        """Carga ejemplos de extracciones correctas"""
        # Aquí cargarías desde archivos JSON en data/training_examples
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
    
    def _build_null_prompt(self, document_name: str, document_type: str) -> str:
        """
        Construye un prompt para documentos sin campos permitidos
        """
        return f"""
DOCUMENTO: {document_name}
TIPO: {document_type}

Este tipo de documento NO está autorizado para proporcionar campos de extracción.

Retorna TODOS los campos como null:
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
    
    def _build_vision_prompt(self, document_name: str, document_type: str, guide_section: str) -> str:
        """
        Construye prompt para ruta Direct AI (visión)
        """
        return f"""
Eres un experto en análisis visual de documentos de seguros.

DOCUMENTO: {document_name}
TIPO: {document_type}

{guide_section}

INSTRUCCIONES PARA ANÁLISIS VISUAL:
1. Examina visualmente el documento completo
2. Identifica elementos estructurales: encabezados, tablas, sellos, firmas
3. Lee cuidadosamente el texto visible
4. Extrae SOLO los campos permitidos según la guía
5. Mantén el formato exacto como aparece en el documento

RESPONDE ÚNICAMENTE con el JSON de extracción.
"""
    
    def _build_text_prompt(self, document_name: str, document_type: str, content: Dict[str, Any], guide_section: str) -> str:
        """
        Construye prompt para ruta OCR + IA textual
        """
        ocr_section = self._format_ocr_content(content) if content else "No hay contenido OCR disponible"
        
        return f"""
Eres un experto en extracción de datos de documentos de seguros.

DOCUMENTO: {document_name}
TIPO: {document_type}

{guide_section}

CONTENIDO DEL DOCUMENTO (OCR):
{ocr_section}

INSTRUCCIONES FINALES:
1. Analiza el contenido OCR proporcionado
2. Extrae SOLO los campos permitidos según la guía
3. Si un campo no está permitido → DEBE ser null
4. NO inventes información
5. Responde ÚNICAMENTE con el JSON de extracción

FORMATO DE RESPUESTA:
{{
    "numero_siniestro": "valor o null",
    "nombre_asegurado": "valor o null",
    "numero_poliza": "valor o null",
    "vigencia_inicio": "valor o null",
    "vigencia_fin": "valor o null",
    "domicilio_poliza": "valor o null",
    "bien_reclamado": "valor o null",
    "monto_reclamacion": valor_numerico_o_null,
    "tipo_siniestro": "valor o null",
    "fecha_ocurrencia": "valor o null",
    "fecha_reclamacion": "valor o null",
    "lugar_hechos": "valor o null",
    "ajuste": "valor o null",
    "conclusiones": "valor o null"
}}
"""