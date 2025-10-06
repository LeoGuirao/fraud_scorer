"""
Clasificador de documentos para el sistema de organización
Implementa clasificación híbrida: LLM primero, heurística como respaldo
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
    
    # Nuevas categorías (no usadas para extracción de cabecera)
    CFDI_CARTA_PORTE = "cfdi_carta_porte"
    CONSTANCIA_IMSS_OPERADOR = "constancia_imss_del_operador"
    FACTURA_COMERCIAL_CFDI = "factura_comercial_cfdi"
    PEDIDO_POR_CORREO = "pedido_por_correo"
    REPORTE_COSTOS_RENDIMIENTOS = "reporte_de_costos_y_rendimientos"
    FACTURAS_COMERCIALES_INTERNACIONALES = "facturas_comerciales_internacionales"
    PEDIMENTO_IMPORTACION = "pedimento_importacion"
    PROTOCOLO_ACCION_REACCION = "protocolo_de_accion_y_reaccion"
    CONOCIMIENTO_EMBARQUE = "conocimiento_de_embarque"
    CONTRATO_PRESTACION_SERVICIO_TRANSPORTISTA = "contrato_prestacion_servicio_transportista"
    OFICIO_DESADUANADO = "oficio_de_desaduanado"
    OFICIO_DENUNCIA = "oficio_denuncia"
    CARTA_ACLARATORIA_PEAJE = "carta_aclatoria_comprobantes_peaje"
    CARTA_PORTE_SIMPLE = "carta_porte_simple"
    DECLARACION_UNIVERSAL_ACCIDENTE = "declaracion_universal_de_accidente"
    NARRACION_DE_HECHOS = "narracion_de_hechos"
    FICHA_TECNICA_VEHICULO = "ficha_tecnica_de_vehiculo"
    DETERMINACION_DE_PERDIDA = "determinacion_de_perdida"
    
    # Nuevos documentos generales
    IDENTIFICACION_OFICIAL = "identificacion_oficial"
    NOTAS_DE_REPARACION = "notas_de_reparacion"
    DICTAMEN_TECNICO = "dictamen_tecnico"
    COMPROBANTE_DOMICILIO = "comprobante_de_domicilio"
    
    # Documentos legales adicionales
    DENUNCIA_DE_LOS_HECHOS = "denuncia_de_los_hechos"
    
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
        try:
            from fraud_scorer.classification.engine import ClassifierEngine
            from fraud_scorer.settings import CLASSIFICATION_CONFIG
            self._engine = ClassifierEngine(
                model_name=CLASSIFICATION_CONFIG.get("llm_model", "gpt-5-mini"),
                base_classifier=self,
            )
        except Exception as e:
            logger.warning(f"No se pudo inicializar ClassifierEngine: {e}")
            self._engine = None
        
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
                exclude=["salida de almacén", "salida_almacen", "almacén", "firmas", "responsable", "recibe", "entrega"],
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
                keywords=[
                    "carpeta de investigación", "carpeta_investigacion", "carpeta investigación", "n.u.c", "nuc",
                    "fiscalía", "ministerio público", "actas", "diligencias", "acuerdos"
                ],
                must_have=["carpeta", "investigacion"],
                may_have=["averiguación", "nuc", "folio", "oficio", "peritajes", "policía"],
                exclude=["oficio de denuncia", "narración de hechos", "formato de denuncia"],
                description="Expediente formal del Ministerio Público (folios, acuerdos, diligencias, sellos) que integra la investigación penal del siniestro"
            ),
            
            DocumentType.ACREDITACION_PROPIEDAD.value: DocumentTypeDefinition(
                type_name="acreditacion_de_propiedad_y_representacion",
                keywords=[
                    "acreditación", "propiedad", "representación legal", "poder notarial", "acta constitutiva",
                    "cesión", "mandato", "apoderado"
                ],
                must_have=["acreditación"],
                may_have=["poder", "notario", "factura", "constancia", "cedula", "representante legal"],
                exclude=["denuncia", "carpeta", "narración"],
                description="Paquete documental (poderes notariales, actas, facturas) que acredita propiedad de la mercancía y representación legal ante la aseguradora"
            ),

            # Denuncia de los hechos (noticia criminal)
            DocumentType.DENUNCIA_DE_LOS_HECHOS.value: DocumentTypeDefinition(
                type_name="denuncia_de_los_hechos",
                keywords=[
                    "denuncia", "noticia criminal", "hechos", "ministerio público", "ministerio publico",
                    "fiscalía", "fiscalia", "autoridad", "testigos", "denunciante", "acta",
                    "jurisdicción", "jurisdiccion", "tipificación", "tipificacion"
                ],
                must_have=["denuncia"],
                may_have=["folio", "fecha", "hora", "lugar", "narrativa", "testimonios", "autoridad receptora"],
                exclude=["oficio", "plantilla", "póliza", "factura", "cfdi", "informe preliminar", "informe final"],
                description="Acta ministerial con sello y folio donde la autoridad recibe formalmente la denuncia de los hechos y abre la noticia criminal"
            ),
            
            DocumentType.SALIDA_DE_ALMACEN.value: DocumentTypeDefinition(
                type_name="salida_de_almacen",
                keywords=["salida de almacén", "salida_almacen", "salida almacén", "embarque", "control interno"],
                must_have=[],
                may_have=["códigos", "piezas", "cantidades", "firmas", "responsable", "transportista", "recibe", "entrega"],
                exclude=["cfdi", "factura", "uuid", "sat", "timbrado", "carta porte", "complemento carta porte", "guía de paquetería"],
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
                keywords=[
                    "cobranza", "cumplimiento", "recibo", "recibos", "pagado", "pago", "pagos",
                    "vigencia", "primas", "validación de cobertura", "validez cobertura", "estado de cuenta"
                ],
                # No exigir must-have para no bloquear variantes (algunos usan "cumplimiento" sin decir "cobranza")
                must_have=[],
                may_have=["RFC", "datos fiscales", "aseguradora", "correo", "comprobante", "confirmación"],
                exclude=["credencial para votar", "instituto nacional electoral", "ine", "ife"],
                description="Documento que acredita vigencia en cobranza: comunicaciones, recibos pagados y validación de cobertura"
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

            # ------------------ Nuevos tipos ------------------
            DocumentType.CFDI_CARTA_PORTE.value: DocumentTypeDefinition(
                type_name="cfdi_carta_porte",
                keywords=[
                    "carta porte", "cfdi", "complemento carta porte", "uuid", "sello", "sat",
                    "emisor", "receptor", "traslado", "transporte", "autotransporte",
                    "mercancias", "mercancías", "ubicaciones", "figura transporte", "permsct",
                    "remolques", "transpinternac"
                ],
                must_have=["carta porte", "cfdi"],
                may_have=["origen", "destino", "material", "peso", "folio", "serie"],
                exclude=["informe", "póliza", "carta porte simple"],
                description="Comprobante fiscal digital (CFDI) con complemento Carta Porte timbrado por el SAT para acreditar formalmente el traslado de mercancías ante la autoridad, con UUID, sellos digitales y obligaciones fiscales"
            ),
            DocumentType.CONSTANCIA_IMSS_OPERADOR.value: DocumentTypeDefinition(
                type_name="constancia_imss_del_operador",
                keywords=["imss", "instituto mexicano del seguro social", "nss", "registro patronal", "asegurado", "movimiento afiliatorio", "alta", "baja", "modificación"],
                must_have=[],
                may_have=["sello digital", "patrón", "empresa", "vigencia"],
                exclude=["identificación", "ine", "ife", "póliza"],
                description="Constancia del IMSS con datos del asegurado y movimientos afiliatorios"
            ),
            DocumentType.FACTURA_COMERCIAL_CFDI.value: DocumentTypeDefinition(
                type_name="factura_comercial_cfdi",
                keywords=["cfdi", "factura", "uuid", "timbrado", "sat", "rfc", "subtotal", "iva", "total", "código sat"],
                must_have=["factura"],
                may_have=["emisor", "receptor", "productos", "cantidades", "orden de compra"],
                exclude=["carta porte", "complemento carta porte", "autotransporte", "mercancias", "mercancías", "ubicaciones", "figura transporte"],
                description="CFDI de venta de mercancías con timbrado SAT"
            ),
            DocumentType.PEDIDO_POR_CORREO.value: DocumentTypeDefinition(
                type_name="pedido_por_correo",
                keywords=["correo", "email", "from:", "to:", "subject", "asunto", "pedido", "confirmación"],
                must_have=[],
                may_have=["productos", "cantidades", "adjunto"],
                exclude=["cfdi", "uuid", "timbrado"],
                description="Impresión de correo electrónico de solicitud/confirmación de pedido"
            ),
            DocumentType.REPORTE_COSTOS_RENDIMIENTOS.value: DocumentTypeDefinition(
                type_name="reporte_de_costos_y_rendimientos",
                keywords=["costos", "rendimientos", "márgenes", "lote", "producción", "reporte"],
                must_have=[],
                may_have=["pedimento", "carta porte", "orden de compra"],
                exclude=["cfdi", "factura"],
                description="Reporte interno de costos/rendimientos, no un CFDI de venta"
            ),
            DocumentType.FACTURAS_COMERCIALES_INTERNACIONALES.value: DocumentTypeDefinition(
                type_name="facturas_comerciales_internacionales",
                keywords=["comercial invoice", "factura comercial", "importación", "exportación", "aduana", "pedimento", "moneda", "usd", "eur", "incoterms"],
                must_have=[],
                may_have=["bill of lading", "bl", "puerto"],
                exclude=["cfdi", "sat", "carta porte"],
                description="Facturas de proveedores extranjeros ligadas a import/export"
            ),
            DocumentType.PEDIMENTO_IMPORTACION.value: DocumentTypeDefinition(
                type_name="pedimento_importacion",
                keywords=[
                    "pedimento", "aduana", "clave pedimento", "régimen aduanero", "importador",
                    "agente aduanal", "shcp", "aduanal", "valor aduana", "impuestos"
                ],
                must_have=["pedimento"],
                may_have=[
                    "régimen", "aduana sección", "clave", "contribuciones", "iva", "doda",
                    "fracción arancelaria", "hoja de cálculo", "importe", "descarga"
                ],
                exclude=["cfdi", "factura", "complemento carta porte", "carta porte"],
                description="Documento oficial emitido por la autoridad aduanera mexicana que ampara la entrada de mercancías al país. Contiene número de pedimento, régimen aduanero, datos del importador y agente aduanal, fracciones arancelarias, descripción y cantidad de mercancías, valores aduanales, impuestos y contribuciones pagadas. Se distingue porque es un comprobante aduanal obligatorio en operaciones de comercio exterior, no una factura comercial ni una carta porte."
            ),
            DocumentType.PROTOCOLO_ACCION_REACCION.value: DocumentTypeDefinition(
                type_name="protocolo_de_accion_y_reaccion",
                keywords=[
                    "protocolo de acción", "protocolo de accion", "acción y reacción", "accion y reaccion",
                    "monitoreo satelital", "empresa de seguridad", "bitácora", "seguimiento de unidad",
                    "permiso ssp", "secretaría de seguridad pública", "operador", "evento de robo"
                ],
                must_have=["protocolo"],
                may_have=[
                    "seguimiento", "horarios", "monitoreo", "permiso", "ssp", "operador",
                    "ruta", "unidad", "medidas", "reacción", "acciones ejecutadas"
                ],
                exclude=["cfdi", "factura", "pedimento", "carta porte"],
                description="Documento emitido por la empresa de seguridad o monitoreo satelital donde se establecen las actuaciones y protocolos a seguir ante un evento de robo. Contiene horarios y seguimiento detallado de la unidad, nombre del operador, permisos vigentes ante la Secretaría de Seguridad Pública y todas las medidas ejecutadas por la empresa que brindó el servicio. La información debe ser consistente con lo manifestado por el asegurado en la documentación del siniestro."
            ),
            DocumentType.CONOCIMIENTO_EMBARQUE.value: DocumentTypeDefinition(
                type_name="conocimiento_de_embarque",
                keywords=[
                    "conocimiento de embarque", "bill of lading", "bl", "puerto", "buque",
                    "embarque", "agente aduanal", "transportista", "checador", "bultos",
                    "peso bruto", "peso neto", "dimensiones", "manifiesto", "muelle"
                ],
                must_have=["embarque"],
                may_have=[
                    "buque", "origen", "destino", "peso", "bultos", "firma", "chequeo",
                    "agente", "operador", "vehículo", "vehiculo", "contenedor"
                ],
                exclude=["factura", "cfdi", "carta porte", "pedimento"],
                description="Documento portuario que certifica la recepción y embarque de mercancías en un buque. Incluye datos del agente aduanal, transportista, operador y vehículo, así como la identificación del buque, origen y destino. Detalla la descripción de la carga (dimensiones, piezas, peso neto y bruto) y total de bultos, con firmas de verificación de checador, agente y transportista. Se distingue porque es un comprobante de embarque marítimo/portuario, no una factura, carta porte o guía de transporte terrestre."
            ),
            DocumentType.CONTRATO_PRESTACION_SERVICIO_TRANSPORTISTA.value: DocumentTypeDefinition(
                type_name="contrato_prestacion_servicio_transportista",
                keywords=[
                    "contrato", "prestación de servicios", "prestacion de servicios", "transportista",
                    "traslado de mercancías", "traslado de mercancias", "monitoreo satelital", "gps",
                    "obligaciones", "seguridad", "responsabilidad", "trazabilidad", "cláusulas", "clausulas",
                    "cobertura", "asegurado"
                ],
                must_have=["contrato"],
                may_have=[
                    "transporte", "mercancías", "mercancias", "viaje", "obligaciones", "seguro",
                    "monitoreo", "gps", "responsabilidad", "daños", "perdidas", "robos", "convenio"
                ],
                exclude=["factura", "pedido", "carta porte", "cfdi"],
                description="Documento legal que avala la relación comercial entre el asegurado y el transportista elegido para el traslado de mercancías. Establece condiciones, cláusulas, acuerdos y convenios sobre la trazabilidad del viaje, obligaciones de seguridad y cobertura del seguro, uso de monitoreo satelital GPS y responsabilidad del transportista ante daños, pérdidas o robos."
            ),
            DocumentType.OFICIO_DESADUANADO.value: DocumentTypeDefinition(
                type_name="oficio_de_desaduanado",
                keywords=[
                    "oficio", "desaduanado", "despacho aduanero", "aduana", "autorización",
                    "legal ingreso", "autoridad aduanera", "entrega", "importador", "exportador",
                    "pedimento", "país de origen", "pais de origen"
                ],
                must_have=["oficio"],
                may_have=[
                    "pedimento", "fecha", "agente aduanal", "mercancía", "mercancia", "cantidad",
                    "origen", "destino", "autorización", "despacho"
                ],
                exclude=["factura", "carta porte", "cfdi", "conocimiento de embarque"],
                description="Documento oficial emitido por la autoridad aduanera mediante el cual se acredita el legal ingreso de las mercancías al país y su entrega al importador correspondiente. Incluye número y fecha del pedimento, datos del agente aduanal, descripción y cantidad de la mercancía, país de origen, importador y exportador, así como autorización expresa para el despacho aduanero."
            ),
            DocumentType.OFICIO_DENUNCIA.value: DocumentTypeDefinition(
                type_name="oficio_denuncia",
                keywords=[
                    "oficio", "denuncia", "representante legal", "autoridad", "modo", "tiempo",
                    "lugar", "involucrados", "narrativa jurídica", "presento denuncia", "escrito",
                    "plantilla", "previa", "siniestro", "asegurado", "señor agente", "c.c." 
                ],
                must_have=["oficio", "denuncia"],
                may_have=[
                    "representante", "legal", "asegurado", "narrativa", "hechos", "autoridad competente",
                    "investigación", "investigacion", "ministerio público", "mp", "acuse", "sello"
                ],
                exclude=["carpeta de investigación", "acta circunstanciada"],
                description="Escrito previo preparado por el asegurado o su representante para presentar la denuncia ante la autoridad antes de obtener folio oficial; resume tiempo, modo y lugar para iniciar la carpeta de investigación"
            ),
            DocumentType.CARTA_ACLARATORIA_PEAJE.value: DocumentTypeDefinition(
                type_name="carta_aclatoria_comprobantes_peaje",
                keywords=[
                    "carta", "aclaratoria", "comprobantes de peaje", "tickets de caseta", "peaje",
                    "extravío", "robo", "ausencia", "plazas de cobro", "fechas", "horas",
                    "ruta", "unidad", "traslado", "mercancías", "justificar"
                ],
                must_have=["carta", "peaje"],
                may_have=[
                    "comprobantes", "caseta", "boletos", "plaza", "cobro", "fecha", "hora",
                    "ruta", "unidad", "siniestro", "aseguradora"
                ],
                exclude=["factura", "carta porte", "cfdi", "pedimento"],
                description="Documento emitido por el asegurado o su representante para aclarar una situación relacionada con los comprobantes de peaje utilizados durante el traslado de mercancías. Puede exponer el extravío, robo o ausencia de tickets de caseta y detallar las plazas de cobro, fechas y horas correspondientes. Su objetivo es justificar ante la aseguradora la falta de comprobantes físicos y brindar trazabilidad sobre la ruta seguida por la unidad."
            ),
            DocumentType.CARTA_PORTE_SIMPLE.value: DocumentTypeDefinition(
                type_name="carta_porte_simple",
                keywords=[
                    "carta porte", "simple", "remitente", "destinatario", "mercancía", "peso",
                    "volumen", "placas", "vehículo", "características", "operador", "permiso sct",
                    "ruta", "origen", "destino", "formato administrativo"
                ],
                must_have=["carta", "porte"],
                may_have=[
                    "remitente", "destinatario", "mercancía", "peso", "volumen", "placas",
                    "características", "operador", "permiso", "ruta", "sct", "contrato"
                ],
                exclude=["uuid", "sat", "sello digital", "complemento", "timbrado"],
                description="Documento interno emitido por empresas transportistas para amparar el traslado de mercancías sin timbrado fiscal; funciona como un formato administrativo o contractual y contiene datos del remitente y destinatario, descripción de la mercancía, peso y volumen, identificación del vehículo (placas, características), datos del operador, permisos SCT y ruta de origen y destino; se distingue de cfdi_carta_porte porque la carta porte simple carece de timbrado SAT, UUID y sellos digitales, siendo únicamente evidencia logística y operativa dentro de la empresa, mientras que el cfdi_carta_porte es un comprobante fiscal digital con complemento timbrado por el SAT que acredita formalmente el traslado ante la autoridad con validez fiscal y obligaciones tributarias."
            ),
            DocumentType.DECLARACION_UNIVERSAL_ACCIDENTE.value: DocumentTypeDefinition(
                type_name="declaracion_universal_de_accidente",
                keywords=["declaración universal de accidente", "dua", "aseguradora", "croquis", "firma", "ajustador", "conductor"],
                must_have=[],
                may_have=["narración", "poliza", "siniestro"],
                exclude=["informe final", "informe preliminar"],
                description="Formato estandarizado de aseguradora para documentar accidentes (DUA)"
            ),
            DocumentType.NARRACION_DE_HECHOS.value: DocumentTypeDefinition(
                type_name="narracion_de_hechos",
                keywords=["narración de hechos", "narracion de hechos", "relato", "declarante", "declaración", "de puño y letra", "yo declaro"],
                must_have=["narracion"],
                may_have=["relato", "circunstancias", "fecha", "modo", "lugar"],
                exclude=["carpeta de investigación", "oficio", "denuncia final"],
                description="Relato en primera persona (afectado/testigo) previo a la autoridad, sin sellos ni folios oficiales"
            ),
            DocumentType.FICHA_TECNICA_VEHICULO.value: DocumentTypeDefinition(
                type_name="ficha_tecnica_de_vehiculo",
                keywords=["ficha técnica", "vehículo", "marca", "modelo", "año", "niv", "serie", "motor", "fotografía"],
                must_have=[],
                may_have=["color", "kilometraje", "version"],
                exclude=["tarjeta de circulación", "licencia"],
                description="Ficha comercial/técnica con datos del vehículo (no autorización oficial)"
            ),
            DocumentType.DETERMINACION_DE_PERDIDA.value: DocumentTypeDefinition(
                type_name="determinacion_de_perdida",
                keywords=["determinación de pérdida", "monto a indemnizar", "deducible", "importe reclamado", "monto ajustado", "total a indemnizar", "ajustador"],
                must_have=[],
                may_have=["cuantificación", "desglose", "facturas"],
                exclude=["informe preliminar", "informe final", "póliza"],
                description="Cálculo económico de la pérdida (cuantificación de indemnización)"
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
        Clasifica un documento intentando primero con LLM (si está disponible)
        y utilizando heurísticas como respaldo.

        Returns:
            - document_type: Tipo de documento identificado
            - confidence: Nivel de confianza (0.0 - 1.0)
            - reasons: Lista de razones para la clasificación
        """

        doc_type: str = DocumentType.OTRO.value
        confidence: float = 0.0
        reasons: List[str] = []
        classified_with_llm = False

        # 1) Intento con motor LLM si está habilitado
        if use_llm_fallback:
            if self._engine is not None:
                try:
                    # Visión opcional controlada por settings (off por defecto en pipeline)
                    try:
                        from fraud_scorer.settings import CLASSIFICATION_ENGINE
                        use_vis = bool(CLASSIFICATION_ENGINE.get("use_vision", False))
                    except Exception:
                        use_vis = False

                    doc_type, confidence, reasons = await self._engine.classify(
                        sample_text=sample_text[:1500],
                        filename=filename,
                        document_path=None,
                        use_llm=True,
                        use_vision=use_vis,
                    )
                    classified_with_llm = True
                except Exception as e:
                    logger.warning(f"Engine LLM falló, se intentará heurística: {e}")
            else:
                try:
                    doc_type, confidence, reasons = await self._llm_classify(sample_text[:1500], filename)
                    classified_with_llm = True
                except Exception as e:
                    logger.warning(f"LLM directo falló, se usará heurística: {e}")

        # 2) Fallback heurístico (o camino directo si LLM no está habilitado)
        if not classified_with_llm:
            doc_type, confidence, reasons = self._heuristic_classify(sample_text, filename)

        # Enriquecimiento: destinatario si aplica
        if doc_type == DocumentType.GUIAS_Y_FACTURAS.value:
            destinatario = self._extract_destinatario(sample_text)
            if destinatario:
                reasons.append(f"Destinatario detectado: {destinatario}")

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
            return DocumentType.DENUNCIA_DE_LOS_HECHOS.value, 0.90, [
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
1. Basa tu decisión principalmente en el CONTENIDO; el nombre del archivo es secundario.
2. Identifica el tipo más apropiado de la lista según la semántica del documento.
3. Distingue con cuidado:
   - expediente_de_cobranza: comunicaciones de aseguradora, recibos/estados de pago, datos fiscales, validación de vigencia/cobertura. Propósito: acreditar COBRANZA/VIGENCIA.
   - identificacion_oficial: credenciales oficiales (INE/IFE/pasaporte), con campos como "Instituto Nacional Electoral", "Clave de Elector", fotografía, firma, etc. La palabra "identificación" o "RFC/CURP" por sí sola NO es identificación oficial.
4. Si no encaja claramente en ninguno, usa "otro".

Responde SOLO con JSON válido:
{{
  "document_type": "tipo_exacto_de_la_lista",
  "confidence": 0.0-1.0,
  "reasons": ["razón 1", "razón 2", "máximo 3 razones"]
}}"""

        try:
            from fraud_scorer.settings import CLASSIFICATION_CONFIG
            model = CLASSIFICATION_CONFIG.get("llm_model", "gpt-5-mini")
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
            keywords = ", ".join(definition.keywords[:6]) if definition.keywords else ""
            must = ", ".join(definition.must_have[:4]) if definition.must_have else "(ninguno)"
            may = ", ".join(definition.may_have[:4]) if definition.may_have else "(opcionales)"
            excl = ", ".join(definition.exclude[:4]) if definition.exclude else "(ninguno)"
            guide_lines.append(
                f"- {type_name}: {definition.description}. keywords: [{keywords}] | must_have: [{must}] | may_have: [{may}] | exclude: [{excl}]"
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
            DocumentType.CFDI_CARTA_PORTE.value: 21,
            DocumentType.PEDIMENTO_IMPORTACION.value: 22,
            DocumentType.CONOCIMIENTO_EMBARQUE.value: 23,
            DocumentType.CONTRATO_PRESTACION_SERVICIO_TRANSPORTISTA.value: 24,
            DocumentType.OFICIO_DESADUANADO.value: 25,
            DocumentType.OFICIO_DENUNCIA.value: 26,
            DocumentType.CARTA_ACLARATORIA_PEAJE.value: 27,
            DocumentType.CARTA_PORTE_SIMPLE.value: 28,
            DocumentType.PROTOCOLO_ACCION_REACCION.value: 29,
            DocumentType.FACTURA_COMERCIAL_CFDI.value: 30,
            DocumentType.FACTURAS_COMERCIALES_INTERNACIONALES.value: 31,
            DocumentType.PEDIDO_POR_CORREO.value: 32,
            DocumentType.CONSTANCIA_IMSS_OPERADOR.value: 33,
            DocumentType.DECLARACION_UNIVERSAL_ACCIDENTE.value: 34,
            DocumentType.REPORTE_COSTOS_RENDIMIENTOS.value: 35,
            DocumentType.FICHA_TECNICA_VEHICULO.value: 36,
            DocumentType.DETERMINACION_DE_PERDIDA.value: 37,
            DocumentType.NARRACION_DE_HECHOS.value: 38,
            DocumentType.OTRO.value: 99
        }
        
        return priority_map.get(document_type, 99)
