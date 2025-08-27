# src/fraud_scorer/ai_extractors/prompts/extraction_prompts.py

"""
Constructor de prompts para extracción con IA
"""
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

class ExtractionPromptBuilder:
    """
    Construye prompts optimizados para extracción de campos
    """
    
    def __init__(self):
        self.base_template = self._load_base_template()
        self.field_descriptions = self._load_field_descriptions()
        self.examples = self._load_examples()
    
    def build_extraction_prompt(
        self,
        document_name: str,
        document_type: str,
        ocr_content: Dict[str, Any],
        required_fields: List[str]
    ) -> str:
        """
        Construye un prompt completo para extracción
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