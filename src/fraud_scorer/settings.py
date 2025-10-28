# src/fraud_scorer/settings.py

"""
Configuración para el sistema de extracción con IA
Incluye mapeos de documentos y guías de extracción
"""
import os
from enum import Enum
from typing import Dict, Any, List, Optional
from pathlib import Path


MODEL_SAMPLING_CONFIG: Dict[str, Dict[str, Optional[float]]] = {
    "gpt-5": {"temperature": None, "top_p": None},
    "gpt-5-nano": {"temperature": None, "top_p": None},
    "gpt-5-mini": {"temperature": None, "top_p": None},
    "gpt-4o": {"temperature": 0.1, "top_p": 0.9},
    "gpt-4o-mini": {"temperature": 0.1, "top_p": 0.9},
}


def normalize_numero_siniestro(value: Any) -> Optional[str]:
    """Normaliza el folio de siniestro asegurando 14 dígitos exactos."""
    digits = ''.join(filter(str.isdigit, str(value or "")))
    if len(digits) == 14:
        return digits
    return None

class ModelType(Enum):
    """Modelos disponibles para cada tarea"""
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"

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
    DENUNCIA = 5  # Mantener compatibilidad
    FACTURA = 6
    PERITAJE = 7
    CARTA_PORTE = 8
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

    # Campos críticos para evaluar escalación en extracción/consolidación
    CRITICAL_EXTRACTION_FIELDS = {
        "suma_asegurada",
        "monto_reclamacion",
        "tipo_siniestro",
    }
    HIGH_RISK_CONSOLIDATION_FIELDS = {
        "suma_asegurada",
        "monto_reclamacion",
        "ajuste",
        "nombre_asegurado",
        "fecha_reclamacion",
        "tipo_siniestro",
    }
    CONSOLIDATION_CONFIDENCE_THRESHOLD = 0.65
    
    # ========== NUEVA SECCIÓN: GUÍAS DE EXTRACCIÓN ==========
    
    # Mapeo: tipo de documento → campos permitidos
    DOCUMENT_FIELD_MAPPING = {
        # Documentos de ajustador
        "informe_preliminar_del_ajustador": [
            "numero_siniestro",
            "nombre_asegurado",
            "bien_reclamado",
            "fecha_ocurrencia",
            "fecha_reclamacion",
            "lugar_hechos",
            "ajuste"
        ],
        "informe_final_del_ajustador": [
            "numero_siniestro",
            "nombre_asegurado",
            "bien_reclamado",
            "monto_reclamacion",
            "fecha_ocurrencia",
            "fecha_reclamacion",
            "lugar_hechos",
            "tipo_siniestro",
            "ajuste",
            "conclusiones"
        ],
        
        # Documentos de seguro
        "poliza_de_la_aseguradora": [
            "numero_poliza",
            "nombre_asegurado",
            "vigencia_inicio",
            "vigencia_fin",
            "domicilio_poliza",
            "suma_asegurada",
            "coberturas",
            "deducible",
        ],
        "expediente_de_cobranza": [
            "numero_poliza",
            "vigencia_inicio",
            "vigencia_fin"
        ],
        "checklist_antifraude": [],
        
        # Cartas de reclamación
        "carta_de_reclamacion_formal_a_la_aseguradora": [
            "numero_siniestro",
            "nombre_asegurado",
            "numero_poliza",
            "fecha_reclamacion",
            "fecha_ocurrencia",
            "lugar_hechos",
            "bien_reclamado",
            "monto_reclamacion"
        ],
        "carta_de_reclamacion_formal_al_transportista": [
            "numero_siniestro",
            "monto_reclamacion"
        ],
        
        # Documentos legales
        "carpeta_de_investigacion": [
            "numero_siniestro",
            "numero_carpeta",
            "fiscalia",
            "agente_ministerio_publico",
            "tipo_siniestro",
            "fecha_ocurrencia",
            "lugar_hechos",
            "fecha_apertura",
            "placas",
            "vin",
            "denuncias",
            "acreditaciones",
            "resumen_conjunto",
        ],
        "denuncia_de_los_hechos": [
            "numero_siniestro",
            "tipo_siniestro",
            "fecha_ocurrencia",
            "lugar_hechos",
            "fecha_denuncia",
            "placas",
            "vin",
        ],
        "acreditacion_de_propiedad_y_representacion": [],
        "narracion_de_hechos": ["tipo_siniestro"],
        "declaracion_del_asegurado": ["tipo_siniestro"],
        
        # Documentos de transporte
        "guias_y_facturas": [
            "bien_reclamado",
            "monto_reclamacion",
            "monto_total",
            "fecha_emision",
            "valor_mercancia",
        ],
        "guias_y_facturas_consolidadas": [
            "monto_total",
            "valor_mercancia",
        ],
        "salida_de_almacen": [
            "bien_reclamado",
            "monto_reclamacion"
        ],
        "carta_porte": [
            "numero_interno_documento",
            "empresa_transportista",
            "destinatario",
            "uuid_fiscal",
            "fecha_emision",
            "fecha_timbrado",
            "placas",
            "vin",
            "operador_nombre",
            "licencia_operador",
            "peso_bruto",
            "valor_mercancia",
            "monto_total",
            "ruta_planeada",
            "origen",
            "destino",
        ],
        
        # Documentos vehiculares
        "tarjeta_de_circulacion_vehiculo": [],
        "licencia_del_operador": [],
        
        # Documentos de siniestro
        "aviso_de_siniestro_transportista": [
            "numero_siniestro",
            "fecha_ocurrencia",
            "lugar_hechos",
            "tipo_siniestro"
        ],
        "reporte_gps": [
            "fecha_inicio",
            "fecha_fin",
            "trayecto",
        ],

        # Otros
        "otro": []  # Documentos no reconocidos no pueden proveer campos
    }

    # Claves alias adicionales para compatibilidad con detectores no canónicos
    # (se reutilizan las mismas listas de campos permitidos)
    DOCUMENT_FIELD_MAPPING.update({
        # Aliases comunes devueltos por detectores heurísticos
        "poliza": DOCUMENT_FIELD_MAPPING["poliza_de_la_aseguradora"],
        "factura": DOCUMENT_FIELD_MAPPING["guias_y_facturas"],
        # Mapear alias 'denuncia' al tipo canónico denuncia_de_los_hechos
        "denuncia": DOCUMENT_FIELD_MAPPING["denuncia_de_los_hechos"],
        
        # Nuevos tipos (sin extracción de cabecera)
        "identificacion_oficial": [],
        "notas_de_reparacion": [],
        "dictamen_tecnico": [],
        "comprobante_de_domicilio": [],
        # Tipos adicionales solicitados (no aportan cabecera)
        "cfdi_carta_porte": DOCUMENT_FIELD_MAPPING["carta_porte"],
        "constancia_imss_del_operador": [],
        "factura_comercial_cfdi": [
            "numero_factura",
            "monto_total",
            "valor_mercancia",
            "fecha_emision",
            "moneda",
        ],
        "pedido_por_correo": [],
        "reporte_de_costos_y_rendimientos": [],
        "facturas_comerciales_internacionales": [
            "numero_factura",
            "monto_total",
            "valor_mercancia",
            "fecha_emision",
            "moneda",
        ],
        "pedimento_importacion": [
            "numero_pedimento",
            "fecha_emision",
            "valor_mercancia",
        ],
        "pedimentos_aduanales": [
            "numero_pedimento",
            "fecha_emision",
            "valor_mercancia",
        ],
        "protocolo_de_accion_y_reaccion": [
            "tipo_siniestro",
            "lugar_hechos"
        ],
        "conocimiento_de_embarque": [
            "bien_reclamado",
            "valor_mercancia",
            "monto_reclamacion"
        ],
        "contrato_prestacion_servicio_transportista": [
            "nombre_asegurado",
            "domicilio_poliza",
            "tipo_siniestro"
        ],
        "oficio_de_desaduanado": [],
        "oficio_denuncia": [
            "numero_siniestro",
            "fecha_ocurrencia",
            "lugar_hechos",
            "tipo_siniestro"
        ],
        "carta_aclatoria_comprobantes_peaje": [
            "fecha_carta",
            "emisor_carta",
            "firmante_nombre",
            "firmante_cargo",
            "destinatario_nombre",
            "destinatario_cargo",
            "asunto_principal",
            "descripcion_evento",
            "consecuencia_evento",
            "detalle_carta",
            "proposito_notificacion",
            "casetas_involucradas",
            "horarios_reportados",
            "evidencia_respaldo",
        ],
        "carta_porte_simple": DOCUMENT_FIELD_MAPPING["carta_porte"],
        "declaracion_universal_de_accidente": [],
        "ficha_tecnica_de_vehiculo": [],
        "determinacion_de_perdida": [],
        "reporte_gps": DOCUMENT_FIELD_MAPPING["reporte_gps"],
    })

    # Tipos de documento en los que SÍ se debe ejecutar extracción de campos
    EXTRACTION_TARGET_TYPES = [
        "informe_preliminar_del_ajustador",
        "informe_final_del_ajustador",
        "poliza_de_la_aseguradora",
        "carta_de_reclamacion_formal_a_la_aseguradora",
        "carta_de_reclamacion_formal_al_transportista",
        "carta_aclatoria_comprobantes_peaje",
        "carpeta_de_investigacion",
        "denuncia_de_los_hechos",
        "narracion_de_hechos",
        "declaracion_del_asegurado",
        "guias_y_facturas",
        "facturas_comerciales_internacionales",
        "factura_comercial_cfdi",
        "cfdi_carta_porte",
        "carta_porte",
        "carta_porte_simple",
        "reporte_gps",
        "pedimento_importacion",
    ]
    
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
            "Firma ajustadora", "Ajuste realizado por",
            "Nombre del ajustador", "Firma de ajustadores",
            "Compañía ajustadora"
        ],
        "suma_asegurada": [
            "Suma asegurada", "Límite máximo por embarque",
            "Límite máximo de responsabilidad", "Límite de responsabilidad"
        ],
        "conclusiones": [
            "Conclusión", "Conclusiones", "Resumen de conclusiones"
        ],
        "fecha_carta": [
            "Fecha", "Fecha de la carta", "Fecha de emisión"
        ],
        "emisor_carta": [
            "Emisor", "Remitente", "Empresa que emite", "Asegurado emisor"
        ],
        "firmante_nombre": [
            "Firmante", "Firma", "Quien firma", "Nombre del firmante"
        ],
        "firmante_cargo": [
            "Cargo", "Puesto", "Título del firmante"
        ],
        "destinatario_nombre": [
            "Destinatario", "A", "Atención", "Señor"
        ],
        "destinatario_cargo": [
            "Cargo destinatario", "Título destinatario", "Puesto destinatario"
        ],
        "asunto_principal": [
            "Asunto", "Motivo", "Notificación"
        ],
        "descripcion_evento": [
            "Descripción", "Narrativa", "Detalle del evento"
        ],
        "consecuencia_evento": [
            "Resultado", "Consecuencia", "Impacto"
        ],
        "detalle_carta": [
            "Detalles", "Se detalla", "Describe"
        ],
        "proposito_notificacion": [
            "Propósito", "Objetivo", "Intención"
        ],
        "casetas_involucradas": [
            "Casetas", "Plazas de cobro", "Peajes", "Pistas"
        ],
        "horarios_reportados": [
            "Horarios", "Fechas y horas", "Tiempos reportados"
        ],
        "evidencia_respaldo": [
            "Evidencia", "Soportes", "Documentos anexos"
        ],
        "numero_interno_documento": [
            "Folio", "Folio No.", "No. de carta porte", "No. documento",
            "Folio interno", "Folio carta porte"
        ],
        "empresa_transportista": [
            "Transportista", "Empresa transportista", "Razón social transportista",
            "Empresa que emite", "Transportista emisor"
        ],
        "destinatario": [
            "Destinatario", "Cliente", "Consignatario", "Cliente destinatario",
            "Nombre del cliente"
        ],
        "fecha_emision": [
            "Fecha de emisión", "Fecha", "Fecha expedición", "Fecha de elaboración"
        ],
        "operador_nombre": [
            "Operador", "Nombre del operador", "Chofer", "Conductor",
            "Operador responsable"
        ],
        "placas": [
            "Placas", "No. de placas", "Placa", "Matrícula", "Número económico"
        ],
        "licencia_operador": [
            "Licencia", "No. de licencia", "Licencia SCT", "Permiso SCT"
        ],
        "origen": [
            "Origen", "Lugar de origen", "Punto de partida", "Ubicación de salida"
        ],
        "destino": [
            "Destino", "Lugar de destino", "Punto de llegada", "Ubicación de entrega"
        ],
        "ruta_planeada": [
            "Ruta", "Trayecto", "Itinerario", "Ruta planeada"
        ]
    }
    
    # Reglas de formato y validación
    FIELD_VALIDATION_RULES = {
        "numero_siniestro": {
            "regex": r"^\d{14}$",
            "format": "14 dígitos exactos",
            "transform": normalize_numero_siniestro
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
            "max_words": 20,
            "format": "Descripción breve del bien sin cantidades"
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
        },
        "lugar_hechos": {
            "max_length": 200,
            "format": "Carretera, kilómetro, municipio y estado"
        },
        "numero_interno_documento": {
            "max_length": 40,
            "format": "Texto alfanumérico corto (folio de carta porte)"
        },
        "fecha_emision": {
            "format": "YYYY-MM-DD",
            "type": "date_iso"
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
            "Rapiña o Saqueo",
            "Robo total",
            "Robo total del embarque"
        ]
    }
    
    # Ajustadores reconocidos
    RECOGNIZED_ADJUSTERS = [
        "SINIESCA",
        "PARK PERALES",
        "NUÑEZ MORA Y ASOCIADOS AJUSTADORES",
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
    
    # Configuración de rutas de extracción por tipo de documento
    DOCUMENT_EXTRACTION_ROUTES = {
        # OCR + AI
        "carta_de_reclamacion_formal_a_la_aseguradora": ExtractionRoute.OCR_TEXT,
        "carta_de_reclamacion_formal_al_transportista": ExtractionRoute.OCR_TEXT,
        "guias_y_facturas": ExtractionRoute.OCR_TEXT,
        "tarjeta_de_circulacion_vehiculo": ExtractionRoute.OCR_TEXT,
        "licencia_del_operador": ExtractionRoute.OCR_TEXT,
        "aviso_de_siniestro_transportista": ExtractionRoute.OCR_TEXT,
        "carpeta_de_investigacion": ExtractionRoute.OCR_TEXT,
        "denuncia_de_los_hechos": ExtractionRoute.OCR_TEXT,
        "acreditacion_de_propiedad_y_representacion": ExtractionRoute.OCR_TEXT,
        "salida_de_almacen": ExtractionRoute.OCR_TEXT,
        "reporte_gps": ExtractionRoute.OCR_TEXT,
        "guias_y_facturas_consolidadas": ExtractionRoute.OCR_TEXT,
        "expediente_de_cobranza": ExtractionRoute.OCR_TEXT,
        "checklist_antifraude": ExtractionRoute.OCR_TEXT,
        "identificacion_oficial": ExtractionRoute.OCR_TEXT,
        "notas_de_reparacion": ExtractionRoute.OCR_TEXT,
        "dictamen_tecnico": ExtractionRoute.OCR_TEXT,
        "comprobante_de_domicilio": ExtractionRoute.OCR_TEXT,
        # Nuevos tipos
        "cfdi_carta_porte": ExtractionRoute.OCR_TEXT,
        "constancia_imss_del_operador": ExtractionRoute.OCR_TEXT,
        "factura_comercial_cfdi": ExtractionRoute.OCR_TEXT,
        "pedido_por_correo": ExtractionRoute.OCR_TEXT,
        "reporte_de_costos_y_rendimientos": ExtractionRoute.OCR_TEXT,
        "facturas_comerciales_internacionales": ExtractionRoute.OCR_TEXT,
        "pedimento_importacion": ExtractionRoute.OCR_TEXT,
        "protocolo_de_accion_y_reaccion": ExtractionRoute.OCR_TEXT,
        "conocimiento_de_embarque": ExtractionRoute.OCR_TEXT,
        "contrato_prestacion_servicio_transportista": ExtractionRoute.OCR_TEXT,
        "oficio_de_desaduanado": ExtractionRoute.OCR_TEXT,
        "oficio_denuncia": ExtractionRoute.OCR_TEXT,
        "carta_aclatoria_comprobantes_peaje": ExtractionRoute.OCR_TEXT,
        "carta_porte_simple": ExtractionRoute.OCR_TEXT,
        "declaracion_universal_de_accidente": ExtractionRoute.OCR_TEXT,
        "ficha_tecnica_de_vehiculo": ExtractionRoute.OCR_TEXT,
        "determinacion_de_perdida": ExtractionRoute.OCR_TEXT,
        
        # AI Directo
        "poliza_de_la_aseguradora": ExtractionRoute.OCR_TEXT,
        "informe_preliminar_del_ajustador": ExtractionRoute.OCR_TEXT,
        "informe_final_del_ajustador": ExtractionRoute.OCR_TEXT,
    }
    
    # Mapeo de tipos de documento a prioridades (actualizado)
    DOCUMENT_PRIORITIES = {
        "informe_preliminar_del_ajustador": FieldPriority.INFORME_AJUSTADOR,
        "poliza_de_la_aseguradora": FieldPriority.POLIZA,
        "carta_de_reclamacion_formal_a_la_aseguradora": FieldPriority.CARTA_RECLAMACION,
        "carpeta_de_investigacion": FieldPriority.CARPETA_INVESTIGACION,
        "narracion_de_hechos": FieldPriority.CARPETA_INVESTIGACION,
        "declaracion_del_asegurado": FieldPriority.CARPETA_INVESTIGACION,
        # Mantener compatibilidad con nombres anteriores
        "poliza": FieldPriority.POLIZA,
        "poliza_seguro": FieldPriority.POLIZA,
        "denuncia": FieldPriority.DENUNCIA,
        "denuncia_de_los_hechos": FieldPriority.DENUNCIA,
        "factura": FieldPriority.FACTURA,
        "factura_compra": FieldPriority.FACTURA,
        "peritaje": FieldPriority.PERITAJE,
        "carta_porte": FieldPriority.CARTA_PORTE,
        "pedimento_importacion": FieldPriority.CARTA_PORTE,
        "protocolo_de_accion_y_reaccion": FieldPriority.CARTA_PORTE,
        "conocimiento_de_embarque": FieldPriority.CARTA_PORTE,
        "contrato_prestacion_servicio_transportista": FieldPriority.CARTA_PORTE,
        "oficio_de_desaduanado": FieldPriority.CARTA_PORTE,
        "oficio_denuncia": FieldPriority.DENUNCIA,
        "carta_aclatoria_comprobantes_peaje": FieldPriority.CARTA_PORTE,
        "carta_porte_simple": FieldPriority.CARTA_PORTE,
        "otro": FieldPriority.OTRO
    }
    
    # Reglas de fuente de datos por campo
    FIELD_SOURCE_RULES = {
        "numero_siniestro": [
            "informe_final_del_ajustador",
            "denuncia",
            "poliza"
        ],
        "nombre_asegurado": [
            "poliza_de_la_aseguradora",
            "poliza",
            "informe_final_del_ajustador",
            "informe_preliminar_del_ajustador",
            "denuncia_de_los_hechos",
            "denuncia",
            "carta_de_reclamacion_formal_a_la_aseguradora"
        ],
        "numero_poliza": ["poliza", "denuncia", "carta_de_reclamacion_formal_a_la_aseguradora"],
        "vigencia_inicio": ["poliza"],
        "vigencia_fin": ["poliza"],
        "domicilio_poliza": ["poliza", "denuncia"],
        "bien_reclamado": [
            "informe_final_del_ajustador",
            "denuncia",
            "factura"
        ],
        "monto_reclamacion": [
            "informe_final_del_ajustador",
            "carta_de_reclamacion_formal_a_la_aseguradora",
            "denuncia",
            "factura",
            "peritaje"
        ],
        "tipo_siniestro": [
            "informe_final_del_ajustador",
            "denuncia",
            "poliza"
        ],
        "fecha_ocurrencia": [
            "informe_final_del_ajustador",
            "denuncia",
            "peritaje"
        ],
        "fecha_reclamacion": ["denuncia", "carta_de_reclamacion_formal_a_la_aseguradora"],
        "lugar_hechos": [
            "informe_final_del_ajustador",
            "peritaje",
            "denuncia_de_los_hechos",
            "denuncia",
            "carta_de_reclamacion_formal_a_la_aseguradora"
        ],
        "ajuste": ["peritaje"],
        "conclusiones": ["peritaje", "denuncia"]
    }
    
    # Configuración de OpenAI
    OPENAI_CONFIG = {
        "extraction": {
            "temperature": 0.1,
            "top_p": 0.2,
            "max_completion_tokens": 2200,
            "max_retries": 3,
            "timeout": 30,
        },
        "extraction_escalated": {
            "temperature": 0.1,
            "top_p": 0.2,
            "max_completion_tokens": 2400,
            "max_retries": 3,
            "timeout": 45,
        },
        "consolidation": {
            "temperature": 0.05,
            "top_p": 0.15,
            "max_completion_tokens": 2000,
            "max_retries": 3,
            "timeout": 45,
        },
        "consolidation_escalated": {
            "temperature": 0.05,
            "top_p": 0.15,
            "max_completion_tokens": 2600,
            "max_retries": 4,
            "timeout": 60,
        },
        "generation": {
            "temperature": 0.2,
            "top_p": 0.3,
            "max_completion_tokens": 1400,
            "max_retries": 2,
            "timeout": 30,
        },
    }

    @classmethod
    def get_openai_params(cls, task: str, escalated: bool = False) -> Dict[str, Any]:
        """Recupera los parámetros de inferencia para una tarea."""
        key = f"{task.strip().lower()}"
        if escalated:
            key = f"{key}_escalated"
        params = cls.OPENAI_CONFIG.get(key, {})
        # Evitar que el sistema comparta referencias mutables
        return dict(params)
    
    # Rutas
    PROMPTS_DIR = Path(__file__).parent / "prompts"
    EXAMPLES_DIR = Path("data/training_examples")
    
    # ========== NUEVA SECCIÓN: CONFIGURACIÓN DE ORGANIZACIÓN ==========
    
    # Directorio base para uploads organizados
    UPLOADS_DIR = Path("data/uploads")
    
    # Directorio temporal para staging
    STAGING_DIR = UPLOADS_DIR / "renombre_de_documentos"
    
    # Alias de tipos de documento (mapeo corto → canónico)
    DOCUMENT_TYPE_ALIASES = {
        # Nombres cortos para archivos (para nombres de archivo más legibles)
        "poliza": "poliza_de_la_aseguradora",
        "informe_preliminar": "informe_preliminar_del_ajustador",
        "informe_final": "informe_final_del_ajustador",
        "reclamacion_aseg": "carta_de_reclamacion_formal_a_la_aseguradora",
        "reclamacion_trans": "carta_de_reclamacion_formal_al_transportista",
        "investigacion": "carpeta_de_investigacion",
        "denuncia_hechos": "denuncia_de_los_hechos",
        "acreditacion": "acreditacion_de_propiedad_y_representacion",
        "guias_facturas": "guias_y_facturas",
        "guias_consolidadas": "guias_y_facturas_consolidadas",
        "salida_almacen": "salida_de_almacen",
        "tarjeta_circ": "tarjeta_de_circulacion_vehiculo",
        "licencia": "licencia_del_operador",
        "aviso_siniestro": "aviso_de_siniestro_transportista",
        "gps": "reporte_gps",
        "cobranza": "expediente_de_cobranza",
        "antifraude": "checklist_antifraude",
        "id_oficial": "identificacion_oficial",
        "nota_reparacion": "notas_de_reparacion",
        "dictamen_tecnico": "dictamen_tecnico",
        "comp_dom": "comprobante_de_domicilio",
        # Nuevos tipos
        "carta_porte": "cfdi_carta_porte",
        "imss_operador": "constancia_imss_del_operador",
        "factura_cfdi": "factura_comercial_cfdi",
        "pedido_correo": "pedido_por_correo",
        "costos_rendimientos": "reporte_de_costos_y_rendimientos",
        "facturas_intl": "facturas_comerciales_internacionales",
        "pedimento": "pedimento_importacion",
        "pedimento_aduanal": "pedimento_importacion",
        "pedimentos_aduanales": "pedimento_importacion",
        "protocolo_accion": "protocolo_de_accion_y_reaccion",
        "protocolo_reaccion": "protocolo_de_accion_y_reaccion",
        "accion_reaccion": "protocolo_de_accion_y_reaccion",
        "conocimiento_embarque": "conocimiento_de_embarque",
        "bill_of_lading": "conocimiento_de_embarque",
        "conocimiento_emb": "conocimiento_de_embarque",
        "contrato_transportista": "contrato_prestacion_servicio_transportista",
        "contrato_prestacion": "contrato_prestacion_servicio_transportista",
        "contrato_servicio_transportista": "contrato_prestacion_servicio_transportista",
        "oficio_desaduanado": "oficio_de_desaduanado",
        "oficio_desaduanamiento": "oficio_de_desaduanado",
        "oficio_denuncia": "oficio_denuncia",
        "oficio_de_denuncia": "oficio_denuncia",
        "escrito_denuncia": "oficio_denuncia",
        "carta_aclaratoria_peaje": "carta_aclatoria_comprobantes_peaje",
        "aclaratoria_peaje": "carta_aclatoria_comprobantes_peaje",
        "carta_peaje": "carta_aclatoria_comprobantes_peaje",
        "carta_porte_simple": "carta_porte_simple",
        "porte_simple": "carta_porte_simple",
        "carta_porte_interna": "carta_porte_simple",
        "dua": "declaracion_universal_de_accidente",
        "narracion_hechos": "narracion_de_hechos",
        "ficha_vehiculo": "ficha_tecnica_de_vehiculo",
        "determinacion_perdida": "determinacion_de_perdida"
    }
    
    # Invertir el mapeo para obtener alias desde canónico
    CANONICAL_TO_ALIAS = {v: k for k, v in DOCUMENT_TYPE_ALIASES.items()}
    
    # Prioridades para ordenamiento de archivos
    DOCUMENT_PRIORITIES = {
        "carpeta_de_investigacion": 1,
        "denuncia_de_los_hechos": 1,
        "oficio_denuncia": 1,
        "informe_preliminar_del_ajustador": 2,
        "informe_final_del_ajustador": 3,
        "poliza_de_la_aseguradora": 4,
        "carta_de_reclamacion_formal_a_la_aseguradora": 5,
        "carta_de_reclamacion_formal_al_transportista": 6,
        "acreditacion_de_propiedad_y_representacion": 7,
        "aviso_de_siniestro_transportista": 8,
        "guias_y_facturas": 9,
        "guias_y_facturas_consolidadas": 10,
        "tarjeta_de_circulacion_vehiculo": 11,
        "licencia_del_operador": 12,
        "salida_de_almacen": 13,
        "reporte_gps": 14,
        "expediente_de_cobranza": 15,
        "checklist_antifraude": 16,
        "notas_de_reparacion": 17,
        "dictamen_tecnico": 18,
        "identificacion_oficial": 19,
        "comprobante_de_domicilio": 20,
        "cfdi_carta_porte": 21,
        "pedimento_importacion": 22,
        "conocimiento_de_embarque": 23,
        "contrato_prestacion_servicio_transportista": 24,
        "oficio_de_desaduanado": 25,
        "protocolo_de_accion_y_reaccion": 26,
        "factura_comercial_cfdi": 27,
        "facturas_comerciales_internacionales": 28,
        "pedido_por_correo": 29,
        "constancia_imss_del_operador": 30,
        "declaracion_universal_de_accidente": 31,
        "reporte_de_costos_y_rendimientos": 32,
        "ficha_tecnica_de_vehiculo": 33,
        "determinacion_de_perdida": 34,
        "narracion_de_hechos": 35,
        "otro": 99
    }


# Acceso directo para compatibilidad con tests  
DOCUMENT_PRIORITIES = {
    "informe_final_del_ajustador": 1,
    "informe_preliminar_del_ajustador": 2,
    "carpeta_de_investigacion": 3,
    "denuncia_de_los_hechos": 3,
    "oficio_denuncia": 3,
    "poliza_de_la_aseguradora": 4,
    "guias_y_facturas": 5,
    "carta_de_reclamacion_formal_a_la_aseguradora": 6,
    "carta_de_reclamacion_formal_al_transportista": 7,
    "acreditacion_de_propiedad_y_representacion": 8,
    "reporte_gps": 9,
    "aviso_de_siniestro_transportista": 10,
    "guias_y_facturas_consolidadas": 11,
    "tarjeta_de_circulacion_vehiculo": 12,
    "licencia_del_operador": 13,
    "salida_de_almacen": 14,
    "expediente_de_cobranza": 15,
    "checklist_antifraude": 16,
    "cfdi_carta_porte": 21,
    "pedimento_importacion": 22,
    "conocimiento_de_embarque": 23,
    "contrato_prestacion_servicio_transportista": 24,
    "oficio_de_desaduanado": 25,
    "protocolo_de_accion_y_reaccion": 26,
    "factura_comercial_cfdi": 27,
    "facturas_comerciales_internacionales": 28,
    "pedido_por_correo": 29,
    "constancia_imss_del_operador": 30,
    "declaracion_universal_de_accidente": 31,
    "reporte_de_costos_y_rendimientos": 32,
    "ficha_tecnica_de_vehiculo": 33,
    "determinacion_de_perdida": 34,
    "narracion_de_hechos": 35,
    "otro": 99
}

# Parámetros de clasificación
CLASSIFICATION_CONFIG = {
    "min_confidence_threshold": 0.6,  # Umbral para usar LLM
    "sample_text_length": 1500,       # Caracteres para clasificación
    "llm_model": "gpt-4o-mini",       # Modelo base para clasificación textual
    "llm_temperature": 0.1,           # Baja temperatura para consistencia
    "llm_max_completion_tokens": 200  # Límite de tokens para respuesta
}

# Configuración del motor de clasificación (engine)
# Estrategia LLM-first y uso de visión (imágenes/PDF) en el pipeline.
# Nota: Activar visión incrementa costo/latencia y requiere PyMuPDF.
CLASSIFICATION_ENGINE = {
    "strategy": "llm_first",   # Alternativas futuras: "heuristic_first"
    "use_vision": True          # True: usar visión (PDF/imagen) en pipeline; False: solo texto
}

# Configuración de nombres de archivo
FILE_NAMING_CONFIG = {
    "max_folder_length": 100,         # Longitud máxima carpeta
    "max_file_length": 150,           # Longitud máxima archivo
    "route_labels": {
        "ocr_text": "OCR",
        "direct_ai": "VIS"
    }
}

# Extensiones de archivo soportadas
SUPPORTED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.docx', '.xlsx', '.csv'}

# Directorio de staging para organización  
from pathlib import Path
STAGING_DIR = Path("data/uploads/renombre_de_documentos")

# Aliases de tipos de documentos (para nombres de archivo cortos)
DOCUMENT_TYPE_ALIASES = {
    "DENUNCIA": "denuncia_de_los_hechos",
    "FACTURA": "guias_y_facturas", 
    "GUIA": "guias_y_facturas",
    "POLIZA": "poliza_de_la_aseguradora",
    "REPORTE": "reporte_gps",
    "LICENCIA": "licencia_del_operador",
    "TARJETA": "tarjeta_de_circulacion_vehiculo",
    "RECLAMACION": "carta_de_reclamacion_formal_a_la_aseguradora",
    "RECLAM_TRANS": "carta_de_reclamacion_formal_al_transportista",
    "GUIAS_CONS": "guias_y_facturas_consolidadas",
    "SALIDA_ALM": "salida_de_almacen",
    "AVISO_SINIESTRO": "aviso_de_siniestro_transportista",
    "CARP_INV": "carpeta_de_investigacion",
    "ACREDITACION": "acreditacion_de_propiedad_y_representacion",
    "INF_PRELIM": "informe_preliminar_del_ajustador",
    "INF_FINAL": "informe_final_del_ajustador",
    "COBRANZA": "expediente_de_cobranza",
    "CHECKLIST": "checklist_antifraude",
    "ID": "identificacion_oficial",
    "NOTA_REP": "notas_de_reparacion",
    "DICTAMEN": "dictamen_tecnico",
    "COMP_DOM": "comprobante_de_domicilio",
    # Nuevos tipos
    "CFDI_CP": "cfdi_carta_porte",
    "IMSS_OPER": "constancia_imss_del_operador",
    "FACTURA_CFDI": "factura_comercial_cfdi",
    "PEDIDO_CORREO": "pedido_por_correo",
    "COSTOS_REND": "reporte_de_costos_y_rendimientos",
    "FACT_INTL": "facturas_comerciales_internacionales",
    "PEDIMENTO": "pedimento_importacion",
    "PEDIMENTO_ADUANAL": "pedimento_importacion",
    "PEDIMENTOS_ADUANALES": "pedimento_importacion",
    "PROTOCOLO_ACCION": "protocolo_de_accion_y_reaccion",
    "PROTOCOLO_REACCION": "protocolo_de_accion_y_reaccion",
    "ACCION_REACCION": "protocolo_de_accion_y_reaccion",
    "CONOCIMIENTO_EMBARQUE": "conocimiento_de_embarque",
    "CONOCIMIENTO_EMB": "conocimiento_de_embarque",
    "BILL_OF_LADING": "conocimiento_de_embarque",
    "CONTRATO_TRANSPORTISTA": "contrato_prestacion_servicio_transportista",
    "CONTRATO_PRESTACION": "contrato_prestacion_servicio_transportista",
    "CONTRATO_SERVICIO_TRANSPORTISTA": "contrato_prestacion_servicio_transportista",
    "OFICIO_DESADUANADO": "oficio_de_desaduanado",
    "OFICIO_DESADUANAMIENTO": "oficio_de_desaduanado",
    "OFICIO_DENUNCIA": "oficio_denuncia",
    "OFICIO_DE_DENUNCIA": "oficio_denuncia",
    "ESCRITO_DENUNCIA": "oficio_denuncia",
    "CARTA_ACLARATORIA_PEAJE": "carta_aclatoria_comprobantes_peaje",
    "ACLARATORIA_PEAJE": "carta_aclatoria_comprobantes_peaje",
    "CARTA_PEAJE": "carta_aclatoria_comprobantes_peaje",
    "CARTA_PORTE_SIMPLE": "carta_porte_simple",
    "PORTE_SIMPLE": "carta_porte_simple",
    "CARTA_PORTE_INTERNA": "carta_porte_simple",
    "DUA": "declaracion_universal_de_accidente",
    "NARR_HECHOS": "narracion_de_hechos",
    "FICHA_VEH": "ficha_tecnica_de_vehiculo",
    "DET_PERDIDA": "determinacion_de_perdida",
    "OTROS": "otro"
}

# Invertir el mapeo para obtener alias desde canónico
CANONICAL_TO_ALIAS = {v: k for k, v in DOCUMENT_TYPE_ALIASES.items()}

# Configuración de nombres de archivo
FILE_NAMING_CONFIG = {
    "max_folder_length": 100,
    "max_file_length": 150, 
    "route_labels": {
        "ocr_text": "OCR",
        "direct_ai": "VIS"
    }
}

# Función para obtener el modelo óptimo
def get_model_for_task(task: str, route: str = "ocr_text", complexity: str = "normal") -> str:
    """Devuelve el modelo recomendado por tarea y complejidad.

    Args:
        task: "extraction", "consolidation" o "generation".
        route: Ruta de procesamiento ("ocr_text" o "direct_ai").
        complexity: "normal" para el flujo estándar o "high" cuando se requiere escalación.
    """
    if hasattr(route, "value"):
        route = route.value  # normalizar Enum ExtractionRoute

    normalized_task = (task or "").lower()
    normalized_route = getattr(route, "value", route) if route else "ocr_text"
    normalized_complexity = (complexity or "normal").lower()

    if normalized_task == "extraction":
        if normalized_route == "direct_ai":
            return ModelType.GPT4O.value
        return ModelType.GPT4O_MINI.value

    if normalized_task == "consolidation":
        if normalized_complexity == "high":
            return ModelType.GPT4O.value
        return ModelType.GPT4O_MINI.value

    if normalized_task == "generation":
        return ModelType.GPT4O_MINI.value

    return ModelType.GPT4O_MINI.value
