"""
Clasificador de documentos para el sistema de organización
Implementa clasificación híbrida: heurística primero, LLM si necesario
"""

from __future__ import annotations

import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Tipos de documento reconocidos por el sistema"""
    
    # Cartas de reclamación
    CARTA_RECLAMACION_ASEGURADORA = "carta_de_reclamacion_formal_a_la_aseguradora"
    CARTA_RECLAMACION_TRANSPORTISTA = "carta_de_reclamacion_formal_al_transportista"
    
    # Documentos de transporte y facturación
    GUIAS_Y_FACTURAS = "guias_y_facturas"
    GUIAS_Y_FACTURAS_CONSOLIDADAS = "guias_y_facturas_consolidadas"
    SALIDA_DE_ALMACEN = "salida_de_almacen"
    
    # Documentos vehiculares
    TARJETA_CIRCULACION = "tarjeta_de_circulacion_vehiculo"
    LICENCIA_OPERADOR = "licencia_del_operador"
    
    # Documentos de siniestro
    AVISO_SINIESTRO_TRANSPORTISTA = "aviso_de_siniestro_transportista"
    REPORTE_GPS = "reporte_gps"
    
    # Documentos legales
    CARPETA_INVESTIGACION = "carpeta_de_investigacion"
    ACREDITACION_PROPIEDAD = "acreditacion_de_propiedad_y_representacion"
    
    # Documentos de seguro
    POLIZA_ASEGURADORA = "poliza_de_la_aseguradora"
    INFORME_PRELIMINAR_AJUSTADOR = "informe_preliminar_del_ajustador"
    INFORME_FINAL_AJUSTADOR = "informe_final_del_ajustador"
    EXPEDIENTE_COBRANZA = "expediente_de_cobranza"
    CHECKLIST_ANTIFRAUDE = "checklist_antifraude"
    
    # Nuevos documentos generales
    IDENTIFICACION_OFICIAL = "identificacion_oficial"
    NOTAS_DE_REPARACION = "notas_de_reparacion"
    DICTAMEN_TECNICO = "dictamen_tecnico"
    COMPROBANTE_DOMICILIO = "comprobante_de_domicilio"
    
    # Otros
    OTRO = "otro"


@dataclass
class DocumentTypeDefinition:
    """Definición de un tipo de documento con sus características"""
    type_name: str
    keywords: List[str]
    must_have: List[str]  # Palabras que DEBEN estar presentes
    may_have: List[str]   # Palabras que PUEDEN estar presentes
    exclude: List[str]     # Palabras que NO deben estar presentes
    description: str
    
    def matches_content(self, text: str, filename: str = "") -> Tuple[bool, float, List[str]]:
        """
        Evalúa si el contenido coincide con este tipo de documento
        Returns: (matches, confidence, reasons)
        """
        text_lower = text.lower()
        filename_lower = filename.lower()
        reasons = []
        
        # Check must-have keywords
        must_have_found = 0
        for keyword in self.must_have:
            if keyword.lower() in text_lower or keyword.lower() in filename_lower:
                must_have_found += 1
                reasons.append(f"Contiene '{keyword}' (requerido)")
        
        # Si no tiene todos los must-have, no coincide
        if self.must_have and must_have_found < len(self.must_have):
            return False, 0.0, []
        
        # Check exclusions
        for exclude_word in self.exclude:
            if exclude_word.lower() in text_lower:
                reasons.append(f"Excluido por contener '{exclude_word}'")
                return False, 0.0, reasons
        
        # Count keyword matches
        keyword_matches = 0
        filename_matches = 0
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                keyword_matches += 2  # Peso mayor para contenido
                reasons.append(f"texto: '{keyword}'")
            elif keyword.lower() in filename_lower:
                keyword_matches += 1  # Peso menor para nombre
                filename_matches += 1
                reasons.append(f"nombre: '{keyword}'")
        
        # Count may-have matches
        may_have_matches = 0
        for keyword in self.may_have:
            if keyword.lower() in text_lower:
                may_have_matches += 1
                reasons.append(f"puede tener: '{keyword}'")
        
        # Calculate confidence
        total_possible = len(self.keywords) * 2 + len(self.may_have)
        if total_possible == 0:
            confidence = 0.5 if must_have_found > 0 else 0.0
        else:
            confidence = min((keyword_matches + may_have_matches * 0.5) / total_possible, 1.0)
        
        # Boost confidence if must-have are present
        if must_have_found > 0:
            confidence = max(confidence, 0.3)
        
        # Si hay coincidencias solo en nombre de archivo y no hay texto, dar más peso
        if filename_matches > 0 and len(text_lower.strip()) == 0:
            confidence = max(confidence, 0.3)
        
        # Si hay coincidencias de nombre de archivo, reducir umbral mínimo
        threshold = 0.15 if filename_matches > 0 else 0.2
        matches = confidence > threshold
        
        return matches, confidence, reasons


class DocumentClassifier:
    """
    Clasificador híbrido de documentos
    Usa heurística primero, LLM como fallback
    """
    
    def __init__(self):
        self.type_definitions = self._initialize_type_definitions()
        self.llm_guide = self._build_llm_guide()
        
    def _initialize_type_definitions(self) -> Dict[str, DocumentTypeDefinition]:
        """Inicializa las definiciones de tipos de documento"""
        
        definitions = {
            DocumentType.CARTA_RECLAMACION_ASEGURADORA.value: DocumentTypeDefinition(
                type_name="carta_de_reclamacion_formal_a_la_aseguradora",
                keywords=["carta reclamación", "aviso de siniestro", "solicitud de indemnización", 
                         "reclamación formal", "reembolso"],
                must_have=["aseguradora"],
                may_have=["siniestro", "póliza", "facturas", "mercancía", "importes", "firma"],
                exclude=["transportista", "guía de embarque exclusivamente"],
                description="Carta formal a la aseguradora solicitando indemnización"
            ),
            
            DocumentType.CARTA_RECLAMACION_TRANSPORTISTA.value: DocumentTypeDefinition(
                type_name="carta_de_reclamacion_formal_al_transportista",
                keywords=["carta reclamación", "reclamacion_transportista", "reclamación transportista", "mercancía no entregada", "incumplimiento"],
                must_have=[],
                may_have=["guías", "facturas", "importes", "explicación", "causas"],
                exclude=["aseguradora", "indemnización del seguro"],
                description="Carta al transportista reclamando por mercancía no entregada"
            ),
            
            DocumentType.GUIAS_Y_FACTURAS.value: DocumentTypeDefinition(
                type_name="guias_y_facturas",
                keywords=["número de guía", "factura", "guia_embarque", "guía embarque", "guía de embarque", "cfdi", "refacciones", "código sat", "total"],
                must_have=[],
                may_have=["peso", "remitente", "destinatario", "subtotal", "iva", "total", "pagare"],
                exclude=["consolidado", "múltiples clientes en un archivo"],
                description="Guías de transporte con facturas comerciales"
            ),
            
            DocumentType.GUIAS_Y_FACTURAS_CONSOLIDADAS.value: DocumentTypeDefinition(
                type_name="guias_y_facturas_consolidadas",
                keywords=["consolidado", "múltiples guías", "varios clientes"],
                must_have=["consolidado"],
                may_have=["etiquetas", "paquetería", "cfdi", "pagares"],
                exclude=[],
                description="Múltiples guías y facturas en un solo archivo consolidado"
            ),
            
            DocumentType.TARJETA_CIRCULACION.value: DocumentTypeDefinition(
                type_name="tarjeta_de_circulacion_vehiculo",
                keywords=["tarjeta de circulación", "tarjeta_circulacion", "tarjeta circulación", "tipo de transporte", "niv", "placas"],
                must_have=[],
                may_have=["folio", "marca", "modelo", "año", "propietario", "estado emisor"],
                exclude=["licencia", "operador", "fotografía"],
                description="Documento oficial de circulación vehicular"
            ),
            
            DocumentType.LICENCIA_OPERADOR.value: DocumentTypeDefinition(
                type_name="licencia_del_operador",
                keywords=["licencia", "operador", "chofer", "conductor"],
                must_have=["licencia"],
                may_have=["fotografía", "curp", "tipo de licencia", "vigencia", "firma", "huella"],
                exclude=["tarjeta de circulación", "vehículo", "placas"],
                description="Licencia oficial del conductor/operador"
            ),
            
            DocumentType.AVISO_SINIESTRO_TRANSPORTISTA.value: DocumentTypeDefinition(
                type_name="aviso_de_siniestro_transportista",
                keywords=["ficha", "aviso de siniestro", "aviso_siniestro", "aviso siniestro", "reporte siniestro", "transportista"],
                must_have=[],
                may_have=["fecha", "hora", "lugar", "operador", "unidad", "accidente", "asalto", "robo"],
                exclude=["fiscalía", "ministerio público", "ajustador"],
                description="Reporte de siniestro emitido por la transportista"
            ),
            
            DocumentType.CARPETA_INVESTIGACION.value: DocumentTypeDefinition(
                type_name="carpeta_de_investigacion",
                keywords=["carpeta de investigación", "carpeta_investigacion", "carpeta investigación", "fgr", "fiscalía", "ministerio público", "querella"],
                must_have=[],
                may_have=["investigación", "folio", "acuerdos", "denunciante", "ofendidos", "hechos", "acta"],
                exclude=[],
                description="Documento oficial de investigación judicial"
            ),
            
            DocumentType.ACREDITACION_PROPIEDAD.value: DocumentTypeDefinition(
                type_name="acreditacion_de_propiedad_y_representacion",
                keywords=["acreditación", "propiedad", "representación legal", "poder notarial"],
                must_have=["acreditación"],
                may_have=["poderes", "identificaciones", "constancias", "facturas", "guías"],
                exclude=[],
                description="Documentos que acreditan propiedad y representación legal"
            ),
            
            DocumentType.SALIDA_DE_ALMACEN.value: DocumentTypeDefinition(
                type_name="salida_de_almacen",
                keywords=["salida de almacén", "salida_almacen", "salida almacén", "embarque", "control interno"],
                must_have=[],
                may_have=["códigos", "piezas", "cantidades", "firmas", "responsable", "transportista"],
                exclude=["cfdi", "factura", "guía de paquetería"],
                description="Control interno de salida de mercancías del almacén"
            ),
            
            DocumentType.REPORTE_GPS.value: DocumentTypeDefinition(
                type_name="reporte_gps",
                keywords=["rastreo satelital", "gps", "reporte gps", "coordenadas", "recorrido", "telemetría"],
                must_have=["gps"],
                may_have=["periodo", "distancia", "velocidad", "combustible", "horarios", "direcciones"],
                exclude=[],
                description="Informe de rastreo satelital del vehículo"
            ),
            
            DocumentType.POLIZA_ASEGURADORA.value: DocumentTypeDefinition(
                type_name="poliza_de_la_aseguradora",
                keywords=["póliza", "poliza_seguro", "póliza seguro", "póliza de seguro", "cobertura", "vigencia", "asegurado", "prima", "transporte marítimo"],
                must_have=[],
                may_have=["carátula", "límites", "deducibles", "cláusulas", "endosos", "exclusiones"],
                exclude=["reclamación", "siniestro ocurrido"],
                description="Contrato formal de seguro"
            ),
            
            DocumentType.INFORME_PRELIMINAR_AJUSTADOR.value: DocumentTypeDefinition(
                type_name="informe_preliminar_del_ajustador",
                keywords=["informe preliminar", "informe_ajustador", "informe ajustador", "ajustador", "estimación inicial"],
                must_have=[],
                may_have=["asegurado", "póliza", "evento", "pérdida", "deducible", "observaciones"],
                exclude=["final", "definitivo", "conclusión"],
                description="Valoración inicial del ajustador"
            ),
            
            DocumentType.INFORME_FINAL_AJUSTADOR.value: DocumentTypeDefinition(
                type_name="informe_final_del_ajustador",
                keywords=["informe final", "ajustador", "recomendación", "indemnización"],
                must_have=["final", "ajustador"],
                may_have=["narrativa", "análisis", "cotejo", "salvamento", "subrogación"],
                exclude=["preliminar", "inicial"],
                description="Conclusión definitiva del ajustador"
            ),
            
            DocumentType.EXPEDIENTE_COBRANZA.value: DocumentTypeDefinition(
                type_name="expediente_de_cobranza",
                keywords=["cobranza", "recibos pagados", "validez cobertura"],
                must_have=["cobranza"],
                may_have=["comunicaciones", "pantallas", "comprobantes", "pago", "vigente"],
                exclude=[],
                description="Validación del estado de cobranza de la póliza"
            ),
            
            DocumentType.CHECKLIST_ANTIFRAUDE.value: DocumentTypeDefinition(
                type_name="checklist_antifraude",
                keywords=["checklist", "antifraude", "checklist antifraude", "evaluación", "evaluación de riesgo", "parámetros alerta", "validación"],
                must_have=["antifraude"],
                may_have=["crítico", "no crítico", "analista", "firmas", "riesgos"],
                exclude=[],
                description="Formato de evaluación de riesgo de fraude"
            ),
            
            # ---- Nuevos documentos generales ----
            DocumentType.IDENTIFICACION_OFICIAL.value: DocumentTypeDefinition(
                type_name="identificacion_oficial",
                keywords=["ine", "ife", "credencial", "pasaporte", "cédula", "cedula", "licencia", "curp", "fotografía", "foto"],
                must_have=[],
                may_have=["curp", "clave de elector", "vigencia", "sexo", "nacionalidad"],
                exclude=["póliza", "poliza", "siniestro", "vehículo", "tarjeta de circulación"],
                description="Identificación oficial de persona (INE/IFE, pasaporte, cédula, licencia)"
            ),
            DocumentType.NOTAS_DE_REPARACION.value: DocumentTypeDefinition(
                type_name="notas_de_reparacion",
                keywords=["nota de reparación", "nota de reparacion", "nota de servicio", "servicio técnico", "refacciones", "mano de obra", "importe", "total"],
                must_have=["nota"],
                may_have=["servicio", "reparación", "taller", "cliente", "firma", "responsable"],
                exclude=["cfdi", "factura electrónica", "xml"],
                description="Comprobante interno de servicio/reparación (no CFDI)"
            ),
            DocumentType.DICTAMEN_TECNICO.value: DocumentTypeDefinition(
                type_name="dictamen_tecnico",
                keywords=["dictamen técnico", "dictamen tecnico", "diagnóstico", "diagnostico", "causa raíz", "falla", "peritaje", "evaluación técnica"],
                must_have=["dictamen"],
                may_have=["diagnóstico", "causas", "pruebas", "procedimiento", "resultado"],
                exclude=["nota de reparación", "cfdi"],
                description="Informe técnico con diagnóstico y justificación de intervención"
            ),
            DocumentType.COMPROBANTE_DOMICILIO.value: DocumentTypeDefinition(
                type_name="comprobante_de_domicilio",
                keywords=["comprobante de domicilio", "cfe", "agua", "luz", "teléfono", "telefono", "número de servicio", "periodo facturado"],
                must_have=["domicilio"],
                may_have=["colonia", "cp", "código postal", "titular", "periodo", "importe"],
                exclude=["póliza", "poliza", "siniestro"],
                description="Recibo oficial para acreditar domicilio (CFE/agua/teléfono)"
            ),
        }
        
        return definitions
    
    async def classify(
        self,
        sample_text: str,
        filename: str,
        use_llm_fallback: bool = True
    ) -> Tuple[str, float, List[str]]:
        """
        Clasifica un documento
        
        Returns:
            - document_type: Tipo de documento identificado
            - confidence: Nivel de confianza (0.0 - 1.0)
            - reasons: Lista de razones para la clasificación
        """
        # 1. Intentar clasificación heurística
        doc_type, confidence, reasons = self._heuristic_classify(sample_text, filename)
        
        # 2. Si necesita subnumeración (guias_y_facturas), detectar destinatario
        if doc_type == DocumentType.GUIAS_Y_FACTURAS.value:
            destinatario = self._extract_destinatario(sample_text)
            if destinatario:
                reasons.append(f"Destinatario detectado: {destinatario}")
        
        # 3. Si confianza baja y LLM habilitado, usar fallback
        if confidence < 0.6 and use_llm_fallback and doc_type == DocumentType.OTRO.value:
            try:
                doc_type, confidence, reasons = await self._llm_classify(
                    sample_text[:1500],
                    filename
                )
            except Exception as e:
                logger.warning(f"Error en clasificación LLM: {e}")
                # Mantener clasificación heurística
        
        return doc_type, confidence, reasons
    
    def _heuristic_classify(
        self, text: str, filename: str
    ) -> Tuple[str, float, List[str]]:
        """Clasificación basada en heurísticas y keywords"""
        import unicodedata
        
        # Primero, verificar reglas estrictas por nombre de archivo (normalizando acentos)
        filename_lower = (filename or "").lower()
        # Normalizar para remover diacríticos (p.ej., "Póliza" => "poliza")
        filename_norm = unicodedata.normalize('NFKD', filename_lower)
        filename_norm = ''.join(c for c in filename_norm if not unicodedata.combining(c))
        
        # Reglas prioritarias por nombre (retornan tipos canónicos existentes)
        if "informe preliminar" in filename_norm or "informe_preliminar" in filename_norm:
            return DocumentType.INFORME_PRELIMINAR_AJUSTADOR.value, 0.95, [
                "Nombre del archivo contiene 'Informe Preliminar'"
            ]

        if "informe final" in filename_norm or "informe_final" in filename_norm:
            return DocumentType.INFORME_FINAL_AJUSTADOR.value, 0.95, [
                "Nombre del archivo contiene 'Informe Final'"
            ]

        if "poliza" in filename_norm or "póliza" in filename_norm:
            return DocumentType.POLIZA_ASEGURADORA.value, 0.90, [
                "Nombre del archivo contiene 'Póliza'"
            ]

        if "carta_reclamacion" in filename_norm or "carta reclamacion" in filename_norm:
            # Distinguir entre aseguradora y transportista
            if "transportista" in filename_norm:
                return DocumentType.CARTA_RECLAMACION_TRANSPORTISTA.value, 0.90, [
                    "Nombre del archivo indica carta a transportista"
                ]
            else:
                return DocumentType.CARTA_RECLAMACION_ASEGURADORA.value, 0.90, [
                    "Nombre del archivo indica carta de reclamación"
                ]

        if "denuncia" in filename_norm:
            return DocumentType.CARPETA_INVESTIGACION.value, 0.85, [
                "Nombre del archivo contiene 'Denuncia'"
            ]

        if "checklist" in filename_lower and "antifraude" in filename_lower:
            return DocumentType.CHECKLIST_ANTIFRAUDE.value, 0.95, [
                "Nombre indica Checklist Antifraude"
            ]

        if "reporte gps" in filename_lower or "gps" in filename_lower:
            return DocumentType.REPORTE_GPS.value, 0.90, ["Nombre indica Reporte GPS"]

        if "salida de almacen" in filename_lower or "salida_de_almacen" in filename_lower:
            return DocumentType.SALIDA_DE_ALMACEN.value, 0.90, ["Nombre indica Salida de Almacén"]

        if "licencia" in filename_lower and ("chofer" in filename_lower or "operador" in filename_lower or True):
            return DocumentType.LICENCIA_OPERADOR.value, 0.85, ["Nombre indica Licencia de Operador"]

        if "tarjeta" in filename_lower and "circulacion" in filename_lower:
            return DocumentType.TARJETA_CIRCULACION.value, 0.85, ["Nombre indica Tarjeta de Circulación"]

        if "guia" in filename_lower or "guias" in filename_lower or "factura" in filename_lower:
            return DocumentType.GUIAS_Y_FACTURAS.value, 0.90, ["Nombre indica Guías y/o Facturas"]

        # Nuevos tipos por nombre de archivo
        if any(x in filename_lower for x in ["identificacion", "identificación", "ine", "ife", "pasaporte", "cedula", "cédula"]) or (
            "licencia" in filename_lower and "operador" not in filename_lower and "chofer" not in filename_lower
        ):
            return DocumentType.IDENTIFICACION_OFICIAL.value, 0.9, ["Nombre indica Identificación Oficial"]

        if "nota" in filename_lower and ("reparacion" in filename_lower or "reparación" in filename_lower or "servicio" in filename_lower):
            return DocumentType.NOTAS_DE_REPARACION.value, 0.9, ["Nombre indica Nota de Reparación/Servicio"]

        if "dictamen" in filename_lower or "diagnostico" in filename_lower or "diagnóstico" in filename_lower:
            return DocumentType.DICTAMEN_TECNICO.value, 0.85, ["Nombre indica Dictamen/Diagnóstico Técnico"]

        if "comprobante de domicilio" in filename_lower or any(x in filename_lower for x in ["cfe", "agua", "luz", "teléfono", "telefono"]):
            return DocumentType.COMPROBANTE_DOMICILIO.value, 0.8, ["Nombre indica Comprobante de Domicilio"]
        
        # Si no hay coincidencia directa en el nombre, usar la lógica existente
        best_match = None
        best_confidence = 0.0
        best_reasons = []
        
        # Evaluar cada tipo de documento
        for type_name, definition in self.type_definitions.items():
            matches, confidence, reasons = definition.matches_content(text, filename)
            
            if matches and confidence > best_confidence:
                best_match = type_name
                best_confidence = confidence
                best_reasons = reasons
        
        if best_match is None:
            return DocumentType.OTRO.value, 0.0, ["No se encontraron indicadores claros"]
        
        return best_match, best_confidence, best_reasons
    
    def _extract_destinatario(self, text: str) -> Optional[str]:
        """Extrae el nombre del destinatario de guías y facturas"""
        
        # Patrones comunes para identificar destinatarios
        patterns = [
            r"Destinatario:?\s*([^\n]+)",
            r"Cliente:?\s*([^\n]+)",
            r"Consignado a:?\s*([^\n]+)",
            r"Nombre:?\s*([^\n]+)",
            r"Razón Social:?\s*([^\n]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                destinatario = match.group(1).strip()
                # Limpiar y validar
                if len(destinatario) > 3 and len(destinatario) < 100:
                    return destinatario
        
        return None
    
    async def _llm_classify(
        self, sample_text: str, filename: str
    ) -> Tuple[str, float, List[str]]:
        """Clasificación usando LLM como fallback"""
        
        prompt = f"""Clasifica este documento de siniestro de seguros.

TIPOS PERMITIDOS (usa el nombre exacto):
{self.llm_guide}

DOCUMENTO:
Archivo: {filename}
Contenido (muestra):
{sample_text}

INSTRUCCIONES:
1. Analiza el contenido y nombre del archivo
2. Identifica el tipo más apropiado de la lista
3. Si no encaja claramente en ninguno, usa "otro"

Responde SOLO con JSON válido:
{{
  "document_type": "tipo_exacto_de_la_lista",
  "confidence": 0.0-1.0,
  "reasons": ["razón 1", "razón 2", "máximo 3 razones"]
}}"""

        try:
            from fraud_scorer.settings import CLASSIFICATION_CONFIG
            model = CLASSIFICATION_CONFIG.get("llm_model", "gpt-4o-mini")
            temperature = float(CLASSIFICATION_CONFIG.get("llm_temperature", 0.1))
            max_tokens = int(CLASSIFICATION_CONFIG.get("llm_max_completion_tokens", 200))

            client = AsyncOpenAI()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Eres un experto en clasificación de documentos de seguros. Responde solo con JSON válido."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens
            )
            
            # Parsear respuesta
            content = response.choices[0].message.content
            # Limpiar posibles marcadores de código
            content = content.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(content)
            
            # Validar que el tipo esté en los permitidos
            valid_types = [dt.value for dt in DocumentType]
            if result["document_type"] not in valid_types:
                result["document_type"] = DocumentType.OTRO.value
                result["confidence"] *= 0.5
                result["reasons"].append("Tipo no reconocido, clasificado como 'otro'")
            
            return (
                result["document_type"],
                float(result["confidence"]),
                result["reasons"][:3]  # Máximo 3 razones
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando respuesta LLM: {e}")
            return DocumentType.OTRO.value, 0.0, [f"Error parsing LLM: {str(e)}"]
        except Exception as e:
            logger.error(f"Error en clasificación LLM: {e}")
            return DocumentType.OTRO.value, 0.0, [f"Error LLM: {str(e)}"]
    
    def _build_llm_guide(self) -> str:
        """Construye guía compacta para el LLM"""
        guide_lines = []
        
        for type_name, definition in self.type_definitions.items():
            # Descripción breve con keywords principales
            keywords = definition.keywords[:3] if definition.keywords else []
            guide_lines.append(
                f"- {type_name}: {definition.description}"
            )
        
        guide_lines.append("- otro: documentos que no encajan en las categorías anteriores")
        
        return "\n".join(guide_lines)
    
    def get_document_priority(self, document_type: str) -> int:
        """Retorna la prioridad de un tipo de documento para ordenamiento"""
        
        priority_map = {
            DocumentType.CARPETA_INVESTIGACION.value: 1,
            DocumentType.INFORME_PRELIMINAR_AJUSTADOR.value: 2,
            DocumentType.INFORME_FINAL_AJUSTADOR.value: 3,
            DocumentType.POLIZA_ASEGURADORA.value: 4,
            DocumentType.CARTA_RECLAMACION_ASEGURADORA.value: 5,
            DocumentType.CARTA_RECLAMACION_TRANSPORTISTA.value: 6,
            DocumentType.ACREDITACION_PROPIEDAD.value: 7,
            DocumentType.AVISO_SINIESTRO_TRANSPORTISTA.value: 8,
            DocumentType.GUIAS_Y_FACTURAS.value: 9,
            DocumentType.GUIAS_Y_FACTURAS_CONSOLIDADAS.value: 10,
            DocumentType.TARJETA_CIRCULACION.value: 11,
            DocumentType.LICENCIA_OPERADOR.value: 12,
            DocumentType.SALIDA_DE_ALMACEN.value: 13,
            DocumentType.REPORTE_GPS.value: 14,
            DocumentType.EXPEDIENTE_COBRANZA.value: 15,
            DocumentType.CHECKLIST_ANTIFRAUDE.value: 16,
            DocumentType.NOTAS_DE_REPARACION.value: 17,
            DocumentType.DICTAMEN_TECNICO.value: 18,
            DocumentType.IDENTIFICACION_OFICIAL.value: 19,
            DocumentType.COMPROBANTE_DOMICILIO.value: 20,
            DocumentType.OTRO.value: 99
        }
        
        return priority_map.get(document_type, 99)
