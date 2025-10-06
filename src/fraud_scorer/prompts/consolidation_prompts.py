# src/fraud_scorer/ai_extractors/prompts/consolidation_prompts.py

"""
Constructor de prompts para consolidación con IA
Sistema v2.0 con awareness del modo guiado y restricciones documento-campo
"""
import json
from typing import Dict, Any, List, Optional

class ConsolidationPromptBuilder:
    """
    Construye prompts para consolidación inteligente con awareness del sistema guiado
    """
    
    def build_conflict_resolution_prompt(
        self,
        field_name: str,
        options: List[Dict[str, Any]],
        field_rules: List[str],
        golden_examples: List[Dict],
        context: str,
        guided_mode: bool = True  # Nueva bandera para modo guiado
    ) -> str:
        """
        Construye prompt para resolver conflictos con awareness del modo guiado
        """
        # Formatear opciones
        options_text = self._format_options(options)
        
        # Formatear reglas
        rules_text = self._format_rules(field_rules)
        
        # Formatear ejemplos
        examples_text = self._format_examples(golden_examples)
        
        # Añadir restricciones si está en modo guiado
        guided_notice = ""
        if guided_mode:
            guided_notice = """
**⚠️ MODO GUIADO ACTIVADO:**
- Solo se consolidan valores extraídos de documentos autorizados para cada campo
- Los valores han sido pre-validados según reglas de negocio
- La prioridad de documentos ya ha sido aplicada en la extracción
"""
        
        extra_rules = ""
        if field_name == "fecha_ocurrencia":
            extra_rules = """
REGLA ABSOLUTA PARA FECHA_OCURRENCIA:
1. Si existe un valor proveniente de 'informe_final_del_ajustador', DEBES seleccionarlo.
2. Solo cuando falte esa fuente puedes considerar informes preliminares o denuncias.
3. No utilices razonamientos contextuales que contradigan este orden.
"""
        elif field_name == "lugar_hechos":
            extra_rules = """
REGLA PARA LUGAR_HECHOS:
1. Prefiere ubicaciones detalladas (carreteras, kilómetros, entronques) provenientes de informes o peritajes.
2. Las denuncias solo deben usarse si no existen esas descripciones específicas.
3. Evita respuestas genéricas (solo ciudad/estado) cuando haya detalles más precisos disponibles.
"""

        prompt = f"""
Necesito resolver un conflicto para el campo '{field_name}' en un siniestro de seguros.
{guided_notice}

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
2. Aplica las reglas de negocio si son relevantes
3. Considera los ejemplos previos como guía
4. Selecciona el valor más confiable y correcto
5. Explica tu razonamiento de forma clara
{extra_rules}

**CRITERIOS DE DECISIÓN:**
- Prioridad del tipo de documento según las reglas
- Consistencia con otros campos del caso
- Formato estándar del campo
- Frecuencia del valor (si aparece múltiples veces)
- Coherencia lógica con el contexto del siniestro

Proporciona tu decisión con el valor seleccionado y una explicación clara.
"""
        return prompt
    
    def build_validation_prompt(
        self,
        consolidated_fields: Dict[str, Any],
        original_extractions: List[Dict],
        guided_mode: bool = True  # Nueva bandera para modo guiado
    ) -> str:
        """
        Construye prompt para validación final con awareness del modo guiado
        """
        guided_info = ""
        if guided_mode:
            guided_info = """
**🛡️ VALIDACIÓN EN MODO GUIADO:**
Este modo garantiza que:
- Todos los valores provienen de documentos autorizados
- Se aplicaron máscaras de seguridad para prevenir alucinaciones
- Los campos siguen las reglas de negocio y validaciones específicas
"""
        
        prompt = f"""
Valida los siguientes datos consolidados de un siniestro de seguros:
{guided_info}

**DATOS CONSOLIDADOS:**
{json.dumps(consolidated_fields, ensure_ascii=False, indent=2)}

**DOCUMENTOS ORIGINALES PROCESADOS:**
Total de documentos: {len(original_extractions)}
Tipos: {', '.join(set(e.get('document_type', 'desconocido') for e in original_extractions))}

**VALIDACIONES A REALIZAR:**
1. Coherencia temporal (fechas en orden lógico)
2. Consistencia de montos
3. Formato correcto de campos
4. Relaciones lógicas entre campos
5. Completitud de información crítica

**REGLAS DE VALIDACIÓN:**
- La fecha de ocurrencia debe ser anterior a la fecha de reclamación
- La fecha de ocurrencia debe estar dentro del período de vigencia
- El monto de reclamación debe ser un número positivo
- El número de póliza debe tener formato válido
- Todos los campos críticos deben estar presentes

Si todos los datos son válidos y coherentes, responde: "VALIDADO_OK"

Si hay ajustes necesarios, responde:
"AJUSTES:
campo1: nuevo_valor1
campo2: nuevo_valor2"

**ANÁLISIS:**
"""
        return prompt
    
    def _format_options(self, options: List[Dict]) -> str:
        """Formatea las opciones de forma clara"""
        lines = []
        for i, opt in enumerate(options, 1):
            lines.append(f"{i}. Valor: '{opt['value']}'")
            lines.append(f"   Fuente: {opt['source']}")
            lines.append(f"   Tipo de documento: {opt.get('document_type', 'desconocido')}")
            lines.append("")
        return "\n".join(lines)
    
    def _format_rules(self, rules: List[str]) -> str:
        """Formatea las reglas de prioridad"""
        if not rules:
            return "No hay reglas específicas para este campo."
        
        lines = ["Orden de prioridad de fuentes:"]
        for i, rule in enumerate(rules, 1):
            lines.append(f"{i}. {rule}")
        return "\n".join(lines)
    
    def _format_examples(self, examples: List[Dict]) -> str:
        """Formatea ejemplos de resoluciones anteriores"""
        if not examples:
            return "No hay ejemplos previos disponibles."
        
        lines = []
        for i, example in enumerate(examples, 1):
            lines.append(f"Ejemplo {i}:")
            lines.append(f"  Opciones: {example.get('options', [])}")
            lines.append(f"  Decisión correcta: {example.get('correct_choice')}")
            lines.append(f"  Razonamiento: {example.get('reasoning')}")
            lines.append("")
        return "\n".join(lines)
