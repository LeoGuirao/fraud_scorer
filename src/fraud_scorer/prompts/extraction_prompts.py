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
        Formatea el contenido del OCR de manera estructurada, controlando tamaño.
        - Texto: truncado a ~4000 chars
        - KV pairs: hasta 40 entradas
        - Tablas: hasta 8 tablas, 3 filas por tabla (muestra representativa)
        """
        sections: list[str] = []

        # Texto principal (truncado)
        text = (ocr_content or {}).get("text") or ""
        if text:
            max_chars = 4000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...[texto truncado]"
            sections.append("TEXTO EXTRAÍDO:")
            sections.append(text)
            sections.append("")

        # Pares clave-valor (limitado)
        kv = (ocr_content or {}).get("key_value_pairs") or {}
        if kv:
            sections.append("CAMPOS DETECTADOS:")
            for i, (key, value) in enumerate(kv.items()):
                if i >= 40:
                    sections.append("  ...[kv truncados]")
                    break
                sections.append(f"  {key}: {value}")
            sections.append("")

        # Tablas (limitadas)
        tables = (ocr_content or {}).get("tables") or []
        if tables:
            sections.append("TABLAS ENCONTRADAS:")
            table_limit = 8
            row_limit = 3
            for i, table in enumerate(tables, 1):
                if i > table_limit:
                    sections.append("...[tablas truncadas]")
                    break
                sections.append(f"\nTabla {i}:")
                headers = table.get("headers") or []
                if headers:
                    # Limitar headers si son demasiados
                    hdrs = headers[:15]
                    if len(headers) > 15:
                        hdrs.append("...[cols truncadas]")
                    sections.append(f"  Encabezados: {', '.join(str(h) for h in hdrs)}")
                # Compatibilidad: algunas fuentes usan 'rows' y otras 'data_rows'
                rows = table.get("rows") or table.get("data_rows") or []
                if rows:
                    sections.append("  Primeras filas:")
                    for row in rows[:row_limit]:
                        try:
                            sections.append(f"    {' | '.join(str(cell) for cell in row)}")
                        except Exception:
                            sections.append(f"    {row}")

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
            "numero_carpeta": "Clave oficial asignada a la carpeta de investigación",
            "fiscalia": "Nombre completo de la fiscalía o autoridad ministerial que recibe la denuncia",
            "agente_ministerio_publico": "Nombre del agente del Ministerio Público que tomó la declaración",
            "denuncias": "Listado de denuncias incluidas en la carpeta con su narrativa y datos críticos",
            "acreditaciones": "Listado de acreditaciones de propiedad anexadas a la carpeta",
            "resumen_conjunto": "Resumen general que cruza la información de todas las denuncias",
            "monto_reclamacion": "Monto total de la reclamación",
            "tipo_siniestro": "Tipo de siniestro (robo, colisión, incendio, etc.)",
            "fecha_ocurrencia": "Fecha cuando ocurrió el siniestro",
            "fecha_reclamacion": "Fecha cuando se presentó la reclamación",
            "lugar_hechos": "Lugar donde ocurrió el siniestro",
            "ajuste": "Nombre del ajustador asignado",
            "conclusiones": "Conclusiones o resolución del caso",
            "fecha_carta": "Fecha en que se emitió la carta",
            "emisor_carta": "Entidad o persona que emite la carta",
            "firmante_nombre": "Nombre de quien firma la carta",
            "firmante_cargo": "Cargo o puesto del firmante",
            "destinatario_nombre": "Nombre de la persona a quien va dirigida",
            "destinatario_cargo": "Cargo o puesto del destinatario",
            "asunto_principal": "Asunto principal descrito en la carta",
            "descripcion_evento": "Descripción breve del evento informado",
            "consecuencia_evento": "Consecuencias descritas a raíz del evento",
            "detalle_carta": "Detalle adicional incluido en la carta",
            "proposito_notificacion": "Objetivo explícito de la notificación",
            "casetas_involucradas": "Lista de casetas de peaje citadas",
            "horarios_reportados": "Fechas y horas asociadas a las casetas",
            "evidencia_respaldo": "Documentos o soportes mencionados",
            "numero_interno_documento": "Folio interno o número de control de la carta porte",
            "serie_cfdi": "Serie del CFDI tal como aparece en el encabezado (ej. \"A\")",
            "folio_cfdi": "Número de folio consecutivo del CFDI (ej. \"499\")",
            "empresa_transportista": "Nombre de la empresa transportista que emite el documento",
            "nombre_transportista": "Nombre comercial o marca del transportista destacado en el membrete",
            "representante_emisor": "Nombre completo de la persona física que firma o representa al emisor del CFDI",
            "emisor_nombre": "Razón social o nombre legal del emisor del CFDI",
            "destinatario": "Nombre del cliente o consignatario al que se entrega la mercancía",
            "issuer_rfc": "RFC del emisor exactamente como aparece en el CFDI",
            "recipient_rfc": "RFC del receptor exactamente como aparece en el CFDI",
            "receptor_nombre": "Nombre o razón social del receptor indicado en el CFDI",
            "fecha_emision": "Fecha de emisión de la carta porte en formato YYYY-MM-DD",
            "uuid_fiscal": "UUID o folio fiscal cuando exista",
            "fecha_timbrado": "Fecha de timbrado fiscal del CFDI en formato YYYY-MM-DD (sin hora)",
            "fecha_certificacion_sat": "Fecha y hora de certificación por el SAT si se muestra",
            "pac_certificador": "Nombre del PAC que certificó el CFDI",
            "sello_digital_cfdi": "Sello digital del CFDI (emisor) completo, sin saltos de línea",
            "sello_digital_sat": "Sello digital completo del CFDI en Base64 (sin recortar)",
            "cadena_original_complemento": "Cadena original del complemento con separadores '|' si aparece en el documento",
            "operador_nombre": "Nombre del operador o chofer responsable del traslado",
            "licencia_operador": "Número de licencia o permiso SCT del operador",
            "placas": "Placas de la unidad (tractor y/o remolques) en formato alfanumérico limpio",
            "cantidad": "Cantidad principal de mercancía declarada como número",
            "unidad_medida": "Unidad de medida asociada a la cantidad de mercancía",
            "descripcion_mercancia": "Descripción principal de la mercancía declarada (ej. \"placas de acero\")",
            "mercancias": "Lista de mercancías con claves: descripcion, cantidad, unidad, peso y valor declarado",
            "origen": "Lugar exacto de origen del traslado",
            "destino": "Lugar exacto de destino del traslado",
            "ruta_planeada": "Ruta o trayecto planeado para la unidad"
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
        route: str = "ocr_text",
        policy_type: Optional[str] = None
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

        # 3.1 Instrucciones especiales por tipo de póliza (HDI EN MI CASA)
        if (policy_type or "") == "HDI_EN_MI_CASA":
            guide_section += """
INSTRUCCIONES ESPECIALES (HDI EN MI CASA):
- NÚMERO DE SINIESTRO no es de 14 dígitos; puede ser como "3925/25 R - 4735611". Mantén el formato completo.
- LUGAR DE LOS HECHOS: en póliza, prioriza el campo "UBICACIÓN DEL RIESGO".
- INFORME FINAL DEL AJUSTADOR: el tipo de siniestro puede venir en texto narrativo; extráelo si aparece.
"""
            # En informe final, reforzar búsqueda exhaustiva y patrones del dominio hogar
            if document_type == "informe_final_del_ajustador":
                guide_section += """
- Buscar en todo el documento (encabezados, narrativa y secciones como: NATURALEZA DEL SINIESTRO, HECHOS, DESCRIPCIÓN DE LOS HECHOS, CRITERIO SOBRE PROCEDENCIA, RESUMEN).
- Patrones útiles (ejemplos reales):
  • "Siniestro de [TIPO]"
  • "Naturaleza del siniestro: [TIPO]"
  • "Daños a [OBJETO] por [CAUSA]"
  • "Pérdidas por [CAUSA]"
  • "Reclamación por [CAUSA]"
- Palabras clave frecuentes en hogar (no exhaustivo):
  • Incendio, Explosión, Humo
  • Robo (con/sin violencia) en Casa Habitación, Allanamiento
  • Fenómenos Hidrometeorológicos (Viento, Granizo, Lluvia, Inundación)
  • Variación/Sobre-voltaje, Daños Eléctricos, Corto circuito
  • Daños por Agua, Fuga de Agua, Rotura de Tubería
  • Rotura de Cristales/Vidrios
  • Vandalismo/Actos malintencionados
  • Impacto de Vehículo, Caída de Árbol, Colapso estructural
- Regla de precisión:
  1) Prefiere la frase exacta que describe el evento (p. ej., "Daños a Equipo Electrónico por Variación de Voltaje").
  2) Si hay varias, elige la más específica y directamente atribuida a la causa del daño.
  3) No confundas listados de coberturas o condiciones generales con el evento ocurrido.
  4) Si no hay mención clara, deja tipo_siniestro como null (NO inventes).
"""
        
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
            "carta_de_reclamacion_formal_a_la_aseguradora": "Carta de Reclamación Formal",
            "carpeta_de_investigacion": "Carpeta de Investigación",
            "narracion_de_hechos": "Narración de Hechos",
            "declaracion_del_asegurado": "Declaración del Asegurado",
            "identificacion_oficial": "Identificación Oficial",
            "notas_de_reparacion": "Notas de Reparación",
            "dictamen_tecnico": "Dictamen Técnico",
            "comprobante_de_domicilio": "Comprobante de Domicilio"
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
   Regla especial: Describe el bien de forma concisa (hasta ~15 palabras) y sin cantidades ni unidades.
   Ejemplo correcto: "placas de acero". Ejemplos incorrectos: "187.104 toneladas de placas de acero"."""
        elif field == "monto_reclamacion":
            guide += """
   Regla especial: Solo el monto total, sin desglose"""
        elif field == "tipo_siniestro":
            guide += f"""
   Valores permitidos: {', '.join([item for sublist in self.siniestro_types.values() for item in sublist][:5])}.
   Si encuentras una variante (ej. "robo total"), mapea al valor del catálogo (ej. "Robo de Bulto por Entero")."""
        elif field == "lugar_hechos":
            guide += """
   Regla especial: Extrae la ubicación más específica del siniestro.
   Prioriza descripciones con carretera, kilómetro, entronques y municipio.
   Ejemplo correcto: "Carretera Matehuala, San Luis Potosí, kilómetro 57".
   Ejemplo incorrecto: "San Luis Potosí" (demasiado genérico)."""
        elif field == "ajuste":
            guide += f"""
   Ajustadores válidos: {', '.join(self.config.RECOGNIZED_ADJUSTERS)}
   Regla especial: Extrae el NOMBRE de la persona o empresa ajustadora, nunca montos ni porcentajes."""
        elif field == "numero_interno_documento":
            guide += """
   Regla especial: Identifica folios como "Folio", "No. Carta Porte" o "Folio interno". Conserva el texto exacto."""
        elif field == "numero_conocimiento":
            guide += """
   Regla especial: Usa el folio literal del conocimiento/tarja (ej. "BOL-001" o "Folio 02"). Conserva letras, guiones y ceros iniciales."""
        elif field == "empresa_transportista":
            guide += """
   Regla especial: Usa la razón social/literal del transportista que emite el documento (encabezado o sello)."""
        elif field == "destinatario":
            guide += """
   Regla especial: Extrae el nombre del cliente/consignatario al que se entrega la mercancía."""
        elif field == "operador_nombre":
            guide += """
   Regla especial: Selecciona el nombre del operador/chofer principal (campo OPERADOR, CONDUCTOR o similar)."""
        elif field == "placas":
            guide += """
   Regla especial: Devuelve todas las placas asociadas a la unidad. Formato alfanumérico sin espacios adicionales."""
        elif field in {"placas_unidad", "semirremolques"}:
            guide += """
   Regla especial: Devuelve cada placa como elemento independiente en una lista JSON (ej. ["16BC2T","34UL2C"]). Limpia espacios, guiones y repeticiones."""
        elif field == "licencia_operador":
            guide += """
   Regla especial: Captura el número de licencia o permiso SCT del operador, si aparece."""
        elif field == "origen":
            guide += """
   Regla especial: Describe el punto de partida (ciudad/estado y, si se menciona, planta o parque industrial)."""
        elif field == "destino":
            guide += """
   Regla especial: Describe el punto de entrega final (ciudad/estado y ubicación específica)."""
        elif field == "fecha_emision":
            guide += """
   Regla especial: Convierte cualquier formato textual a YYYY-MM-DD (ej. "12 de febrero de 2024" → 2024-02-12)."""
        elif field == "fecha_salida":
            guide += """
   Regla especial: Convierte la fecha del encabezado ("12-02-24") a formato YYYY-MM-DD (ej. 2024-02-12)."""
        elif field in {"hora_salida", "hora_inicio", "hora_termino"}:
            guide += """
   Regla especial: Normaliza a formato 24h HH:MM (ej. "17:38"). Usa campos "Hora inicio/termino" si aparecen."""
        elif field == "emisor_documento":
            guide += """
   Regla especial: Toma la razón social del encabezado (terminal o empresa emisora) tal como aparece impresa."""
        elif field == "agente_aduanal":
            guide += """
   Regla especial: Extrae el nombre del agente aduanal reportado (línea 'Agente Aduanal')."""
        elif field == "nombre_transportista":
            guide += """
   Regla especial: Prefiere la razón social completa del transportista; si solo hay alias, devuelve el texto literal indicado."""
        elif field == "numero_pedimento":
            guide += """
   Regla especial: Devuelve únicamente los dígitos consecutivos del pedimento (sin espacios ni guiones)."""
        elif field == "importador":
            guide += """
   Regla especial: Usa la razón social completa del importador tal como aparece en el recuadro 'Importador' del pedimento (sin abreviarla ni traducirla)."""
        elif field in {"aduana_numero", "aduana_codigo"}:
            guide += """
   Regla especial: Captura el código numérico de la aduana (normalmente 2 dígitos) exactamente como aparece en el encabezado."""
        elif field == "aduana_nombre":
            guide += """
   Regla especial: Devuelve el nombre de la aduana o sección aduanera literal, incluyendo ciudad y entidad si se mencionan."""
        elif field == "fecha_entrada":
            guide += """
   Regla especial: Convierte la 'Fecha de entrada/pago' del pedimento a formato YYYY-MM-DD."""
        elif field == "fecha_pago":
            guide += """
   Regla especial: Normaliza la fecha de pago a formato YYYY-MM-DD."""
        elif field == "cantidad_mercancia":
            guide += """
   Regla especial: Resume la cantidad total (ej. "28 bultos" o "90.36 toneladas"). Incluye unidad si se especifica."""
        elif field in {"cantidad", "cantidad_total"}:
            guide += """
   Regla especial: Toma el valor numérico total registrado en el pedimento; si la tabla incluye varias fracciones, suma solo cuando el documento lo indica de forma explícita."""
        elif field == "unidad_medida":
            guide += """
   Regla especial: Devuelve la unidad de medida asociada a la cantidad (ej. KG, TON, PZA)."""
        elif field in {"descripcion_mercancias", "mercancias"}:
            guide += """
   Regla especial: Resume la descripción comercial de las mercancías; si existen varios renglones, sintetiza en una frase clara (ej. "Placas de acero laminado en caliente")."""
        elif field == "peso":
            guide += """
   Regla especial: Prioriza el peso neto principal (ej. "47068 kg"). Si hay neto/bruto, usa el neto."""
        elif field in {"peso_neto", "peso_bruto"}:
            guide += """
   Regla especial: Extrae el valor numérico con su unidad (kg, t, etc.) tal como aparece en el pedimento."""
        elif field in {"valor_mercancia", "valor_aduana"}:
            guide += """
   Regla especial: Devuelve el monto numérico con decimales (sin signo $); respeta los separadores decimales del documento."""

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
- La suma asegurada suele mencionarse como "Límite máximo por embarque" o "Límite de responsabilidad"
"""
        elif document_type == "conocimiento_de_embarque":
            instructions += """
- Usa el encabezado (TARJA/terminal) para identificar `emisor_documento` y `nombre_transportista`.
- Convierte fechas como "12-02-24" a formato YYYY-MM-DD y horas a HH:MM.
- El pedimento debe entregarse solo con dígitos consecutivos (sin espacios).
- Devuelve cada placa (tractor y remolques) como elementos independientes en una lista.
- Extrae la mercancía principal más su cantidad/peso declarados; si hay varios renglones, consólidalos en una frase breve.
- Si existen varios conocimientos de embarque en el expediente, verifica que la fecha de salida coincida con los demás."""
        elif document_type == "pedimento_importacion":
            instructions += """
- Asegúrate de capturar el número de pedimento completo (15 dígitos: aduana + ejercicio + patente + progresivo) sin guiones ni espacios.
- Identifica la razón social del importador en el recuadro correspondiente y devuélvela literal, respetando mayúsculas/abreviaturas.
- Normaliza `fecha_pago` y `fecha_entrada` a formato YYYY-MM-DD (ej. 24/01/2024 → 2024-01-24).
- Resume la mercancía declarada en una frase clara; si hay varias fracciones arancelarias, sintetiza los conceptos relevantes.
- Extrae cantidades, unidad de medida, pesos neto/bruto y valores (valor aduana, valor mercancía) como números con decimales cuando existan.
- Si el pedimento lista varias partidas, conserva los valores tal como vienen; no inventes sumas que el documento no muestre expresamente."""
        elif document_type == "informe_final_del_ajustador":
            instructions += """
INSTRUCCIONES ESPECÍFICAS PARA INFORME FINAL DEL AJUSTADOR:

1. NUMERO_SINIESTRO:
   - Busca en la portada/encabezado expresiones como "Siniestro:", "No. de siniestro:", "Su referencia:".
   - DEBE contener exactamente 14 dígitos (formato 20XXXXXXXXXXXX). Si ves números más cortos (ej. FED/SLP/SLP/0000231/2024), no son el folio.

2. MONTO_RECLAMACION:
   - Revisa la sección "RECLAMACIÓN Y AJUSTE" o párrafos donde menciona "Recibimos del Asegurado... por la cantidad de MXN$...".
   - Esta información suele estar en las últimas páginas del informe.
   - Extrae el monto TOTAL reclamado por la mercancía (no gastos de transporte, subrogaciones ni deducibles).
   - Puede estar escrito en número y en letra; conserva el valor numérico completo (ej. MXN$3,145,997.60).
   - Si aparecen varios montos ("Total pérdida", deducible, ajuste), PRIORIZA el monto reclamado por el asegurado. Ignora "Total pérdida" u otros importes contables.

3. FECHA_OCURRENCIA:
   - Busca "Fecha de siniestro", "Fecha del evento" o similar.
   - Normaliza a formato DD/MM/AAAA.
   - No confundas la fecha de emisión del informe ni la fecha de la denuncia.

4. LUGAR_HECHOS:
   - Debe incluir carretera, kilómetro y municipio/estado cuando estén disponibles.
   - Frases clave: "al circular sobre...", "en el km...". Evita respuestas generales como "San Luis Potosí".
   - Si el texto menciona varios tramos, captura el más específico.

5. AJUSTE:
   - Debe ser el nombre de la empresa/persona ajustadora (ej. "PARK PERALES", "SINIESCA").
   - Ignora montos y porcentajes asociados al ajuste.

6. TIPO_SINIESTRO:
   - Prioriza la cobertura listada en "Riesgos cubiertos" o "Riesgos amparados" (ej. "Robo de Bulto por Entero").
   - Si no está explícita, usa la narrativa de "Causa y circunstancia" y mapea la frase resultante a la categoría oficial.

7. BIEN_RECLAMADO:
   - Extrae únicamente el nombre o tipo del bien (ej. "placas de acero", "varilla de acero").
   - Elimina cantidades, pesos y unidades si aparecen en el texto.
"""
        elif document_type == "denuncia_de_los_hechos":
            instructions += """
- El encabezado contiene datos de la comparecencia (lugar/fecha de denuncia) y la carpeta de investigación. NO los confundas con el siniestro.
- Para NUMERO_SINIESTRO busca etiquetas como "SINIESTRO", "No. de siniestro", "SINIESTRO:". Ignora números de carpeta (p.ej. FED/XXX/0001234/2024).
- La FECHA_OCURRENCIA debe salir de la narrativa donde se describe cuándo sucedió el evento ("el día...", "aproximadamente a las...").
- El LUGAR_HECHOS también se encuentra en la narrativa (carreteras, kilometrajes, ubicaciones). Evita usar "En la ciudad de..." del encabezado.
- Incluye detalles específicos como carretera, kilómetro y municipio cuando estén disponibles.
- Si el encabezado no tiene la información correcta, recorre toda la narrativa.
- Para lugar_hechos busca frases como "pasamos por", "en el kilómetro", "entronque", "carretera".
- Ejemplo correcto: "Carretera Matehuala, San Luis Potosí, kilómetro 57"; ejemplo incorrecto: "San Luis Potosí".
"""
        elif document_type == "carta_de_reclamacion_formal_a_la_aseguradora":
            instructions += """
- Identifica al emisor (persona moral o física) y regístralo en `nombre_asegurado`.
- Extrae el número de póliza y el número de siniestro tal como aparecen en el encabezado.
- Captura la fecha de la carta en `fecha_reclamacion` (formato YYYY-MM-DD) y, si se menciona, la fecha del siniestro en `fecha_ocurrencia`.
- Verifica la ubicación del siniestro (carretera, kilómetro, municipio) y colócala en `lugar_hechos`.
- Normaliza el monto TOTAL reclamado (sin IVA) en `monto_reclamacion` usando formato numérico o con prefijo "$".
- Si describe la mercancía o bienes reclamados, regístralos en `bien_reclamado` como texto descriptivo.
- Evita capturar gastos adicionales cuando el documento distinga entre mercancía robada y otros cargos; prioriza el valor de la mercancía reclamada.
"""
        elif document_type == "carta_porte_simple":
            instructions += """
- `numero_interno_documento`: toma el folio literal (ej. "500", "Folio 500") indicado en encabezado o tabla principal.
- `empresa_transportista`: copia la razón social que emite la carta porte (membrete o sello del transportista).
- Si existen varios nombres de empresa, prioriza la razón social destacada en el encabezado (ej. "Logística, Carga y Más Transportaciones") y no alias abreviados.
- `destinatario`: captura el nombre del cliente/consignatario indicado en campos "Destinatario", "Cliente" o similares.
- `fecha_emision`: convierte cualquier formato textual al estándar YYYY-MM-DD (ej. "12 de febrero de 2024" → 2024-02-12).
- `operador_nombre`: utiliza el nombre del operador/chofer principal. Si se listan varios operadores, prioriza el que conduce la unidad.
- `placas`: extrae todas las placas indicadas (tractor y/o remolques) en formato alfanumérico limpio, separadas por comas si hay varias.
- `origen` y `destino`: transcribe la ubicación completa (ciudad, estado y referencias) tal como aparece en el documento.
- Si algún campo no aparece, devuélvelo como null (no lo infieras)."""
        elif document_type == "cfdi_carta_porte":
            instructions += """
- `numero_interno_documento`: captura el folio interno o número de control impreso; si solo aparece el UUID, deja este campo en null.
- `serie_cfdi` y `folio_cfdi`: conserva exactamente la serie y el folio que aparecen en el encabezado. Si solo hay folio, deja `serie_cfdi` en null.
- `representante_emisor`: identifica a la persona física que firma o aparece como responsable de la emisión (normalmente debajo del domicilio del emisor).
- `nombre_transportista`: prioriza el nombre comercial visible en el membrete (ej. "LC&+ Transportaciones"); si no existe, reutiliza la razón social destacada.
- `emisor_nombre`: captura la razón social legal indicada en el campo "Razón Social" del CFDI.
- `receptor_nombre`: registra el nombre o razón social del receptor textual, sin RFC.
- `descripcion_mercancia`: resume la mercancía principal en minúsculas (ej. "placas de acero"), sin repetir cantidades.
- `mercancias`: si existe tabla, devuelve lista de objetos con `descripcion`, `cantidad`, `unidad`, `peso` (en kilogramos) y `valor`.
- `operador_nombre`: captura el nombre completo con apellidos tal como se muestra en el CFDI.
- `placas`: incluye tanto el tractor como los remolques, separados por comas y sin etiquetas adicionales.
- `sello_digital_sat`: copia el sello completo sin recortarlo; elimina saltos de línea.
- `sello_digital_cfdi`: captura íntegro el sello digital del emisor (sello del CFDI), sin espacios ni saltos de línea.
- `sello_digital_sat`: copia el sello completo del SAT sin recortarlo; elimina saltos de línea.
- `origen` y `destino`: transcribe la dirección completa de las ubicaciones del complemento (calle, número, colonia, municipio, estado y país).
- `fecha_certificacion_sat` y `pac_certificador`: captura literalmente lo indicado en la sección del timbre fiscal digital.
- Si un dato no aparece de forma explícita, devuélvelo como null (no lo inventes)."""
        elif document_type == "carta_aclatoria_comprobantes_peaje":
            instructions += """
- Extrae `fecha_carta` en formato YYYY-MM-DD exactamente como aparece en la carta.
- `emisor_carta` debe conservar el nombre comercial o razón social del remitente (respetando mayúsculas y símbolos como "LC&+").
- Captura `firmante_nombre` y `firmante_cargo` a partir de la firma (normalmente debajo de "Atentamente").
- Registra `destinatario_nombre` y `destinatario_cargo` si se mencionan en el encabezado o saludo.
- `asunto_principal`, `descripcion_evento`, `consecuencia_evento` y `proposito_notificacion` deben describirse con texto literal del documento (sin inferencias).
- `casetas_involucradas` debe ser una lista de objetos con las claves `nombre`, `fecha` y/o `hora` cuando estén presentes. Ejemplo:
  [{"nombre": "Plaza de Cobro LIBRAMIENTO PONIENTE TAMPICO", "fecha": "2024-02-12", "hora": "23:01:17"}]
- `horarios_reportados` debe incluir únicamente timestamps (YYYY-MM-DD HH:MM:SS) asociados a las casetas.
- Si se mencionan soportes (tickets, facturas, bitácoras), agrégalos en `evidencia_respaldo` como lista de textos.
- Si algún campo no aparece en el documento, devuélvelo como null (no inventes datos).
"""
        elif document_type == "carpeta_de_investigacion":
            instructions += """
- Prioriza la transcripción textual. NO inventes ni completes datos fuera del expediente.
- Devuelve `numero_carpeta`, `fiscalia` y `agente_ministerio_publico` exactamente como aparecen (ej. "Fiscalía General de la República").
- Usa el siguiente esquema para `denuncias` (lista con cada declaración principal, incluye al menos la del operador asegurado y la del operador escolta):
  [
    {
      "orden": 1,
      "declarante_titulo": "Sr.",
      "declarante_nombre": "Enrique Hernández García",
      "declarante_rol": "operador del tractocamión asegurado",
      "autoridad": "Lic. Jonathan Josué Zuviri Alonso",
      "fiscalia": "Fiscalía General de la República",
      "numero_carpeta": "FED/SLP/SLP/0000231/2024",
      "fecha_inicio": "2024-02-14",
      "fecha_siniestro": "2024-02-13",
      "hora_evento": "19:30",
      "hora_liberacion": "10:00",
      "origen": "Puerto Altamira, Tampico, Tamaulipas",
      "destino": "Aceros Ocotlán, Guadalajara, Jalisco",
      "lugar": "Kilómetro 57 del entronque de la carretera Matehuala, San Luis Potosí",
      "stop_reason": "consumir alimentos",
      "vehiculos": [
        {"placa": "16BC2T", "tipo": "tractocamión"},
        {"placa": "97UL4C", "tipo": "semirremolque"},
        {"placa": "15TZ2Y", "tipo": "semirremolque"}
      ],
      "mercancias": [
        "Placas de acero (90.36 toneladas)"
      ],
      "descripcion_evento": "Despojo del tractocamión y la mercancía al reanudar el trayecto.",
      "narrativa_detallada": "Incluye el detalle completo de la denuncia en prosa.",
      "assailant_detail": "dos individuos armados",
      "detention_detail": "los mantuvieron en una construcción en obra negra con la vista cubierta",
      "post_event_detail": "Posteriormente los abandonaron cerca de San Lorenzo, Municipio de Villa Hidalgo, San Luis Potosí.",
      "abandon_location": "San Lorenzo, Municipio de Villa Hidalgo, San Luis Potosí",
      "companion_reference": "Irwin Rueda Rubio"
    }
  ]
- Duplica la estructura anterior para cada declarante adicional (cambia `orden` y los datos correspondientes).
- `destino` debe reflejar el destino FINAL declarado (ej. Aceros Ocotlán, Guadalajara, Jalisco), no el punto de descanso.
- `hora_evento` debe corresponder a la hora del robo descrita en la narrativa (ej. 19:30); evita registrar horarios de ratificación o de traslado posterior.
- Si ambos operadores describen el mismo abandono, utiliza el MISMO texto en `abandon_location` y en `post_event_detail`.
- Incluye pesos o toneladas relevantes dentro de `mercancias` conservando la cifra exacta mencionada.
- `acreditaciones` debe ser una lista con objetos que indiquen tipo de bien, presentante, rol y documentos soporte (ej. pedimentos, cartas porte con folio fiscal).
- `resumen_conjunto` debe resumir coincidencias críticas entre denuncias (origen, destino final, hora del evento, agresores, lugar de abandono y hora de liberación).
- Si no se localiza algún dato, devuelve null sin generar conjeturas.
"""
        elif document_type in ["narracion_de_hechos", "declaracion_del_asegurado"]:
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
