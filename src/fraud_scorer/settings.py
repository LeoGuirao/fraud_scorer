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
    EXTRACTOR = "gpt-4o-mini"  # Más barato para extracción simple
    CONSOLIDATOR = "gpt-4o"     # Más potente para razonamiento
    GENERATOR = "gpt-4o-mini"   # Para generación de reportes

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
            "domicilio_poliza"
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
            "monto_reclamacion"
        ],
        "carta_de_reclamacion_formal_al_transportista": [
            "numero_siniestro",
            "monto_reclamacion"
        ],
        
        # Documentos legales
        "carpeta_de_investigacion": [
            "numero_siniestro",
            "tipo_siniestro",
            "fecha_ocurrencia",
            "lugar_hechos"
        ],
        "acreditacion_de_propiedad_y_representacion": [],
        "narracion_de_hechos": ["tipo_siniestro"],
        "declaracion_del_asegurado": ["tipo_siniestro"],
        
        # Documentos de transporte
        "guias_y_facturas": [
            "bien_reclamado",
            "monto_reclamacion"
        ],
        "guias_y_facturas_consolidadas": [],
        "salida_de_almacen": [
            "bien_reclamado",
            "monto_reclamacion"
        ],
        "carta_porte": [],
        
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
        "reporte_gps": [],
        
        # Otros
        "otro": []  # Documentos no reconocidos no pueden proveer campos
    }

    # Claves alias adicionales para compatibilidad con detectores no canónicos
    # (se reutilizan las mismas listas de campos permitidos)
    DOCUMENT_FIELD_MAPPING.update({
        # Aliases comunes devueltos por detectores heurísticos
        "poliza": DOCUMENT_FIELD_MAPPING["poliza_de_la_aseguradora"],
        "factura": DOCUMENT_FIELD_MAPPING["guias_y_facturas"],
        # Mapear 'denuncia' a carpeta de investigación (documento de MP)
        "denuncia": DOCUMENT_FIELD_MAPPING["carpeta_de_investigacion"],
        
        # Nuevos tipos (sin extracción de cabecera)
        "identificacion_oficial": [],
        "notas_de_reparacion": [],
        "dictamen_tecnico": [],
        "comprobante_de_domicilio": [],
        # Tipos adicionales solicitados (no aportan cabecera)
        "cfdi_carta_porte": [],
        "constancia_imss_del_operador": [],
        "factura_comercial_cfdi": [],
        "pedido_por_correo": [],
        "reporte_de_costos_y_rendimientos": [],
        "facturas_comerciales_internacionales": [],
        "declaracion_universal_de_accidente": [],
        "ficha_tecnica_de_vehiculo": [],
        "determinacion_de_perdida": [],
    })

    # Tipos de documento en los que SÍ se debe ejecutar extracción de campos
    EXTRACTION_TARGET_TYPES = [
        "informe_preliminar_del_ajustador",
        "informe_final_del_ajustador",
        "poliza_de_la_aseguradora",
        "carta_de_reclamacion_formal_a_la_aseguradora",
        "carpeta_de_investigacion",
        "narracion_de_hechos",
        "declaracion_del_asegurado",
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
        "declaracion_universal_de_accidente": ExtractionRoute.OCR_TEXT,
        "ficha_tecnica_de_vehiculo": ExtractionRoute.OCR_TEXT,
        "determinacion_de_perdida": ExtractionRoute.OCR_TEXT,
        
        # AI Directo
        "poliza_de_la_aseguradora": ExtractionRoute.DIRECT_AI,
        "informe_preliminar_del_ajustador": ExtractionRoute.DIRECT_AI,
        "informe_final_del_ajustador": ExtractionRoute.DIRECT_AI,
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
        "factura": FieldPriority.FACTURA,
        "factura_compra": FieldPriority.FACTURA,
        "peritaje": FieldPriority.PERITAJE,
        "carta_porte": FieldPriority.CARTA_PORTE,
        "otro": FieldPriority.OTRO
    }
    
    # Reglas de fuente de datos por campo
    FIELD_SOURCE_RULES = {
        "numero_siniestro": ["denuncia", "poliza"],
        "nombre_asegurado": ["poliza", "denuncia"],
        "numero_poliza": ["poliza", "denuncia"],
        "vigencia_inicio": ["poliza"],
        "vigencia_fin": ["poliza"],
        "domicilio_poliza": ["poliza", "denuncia"],
        "bien_reclamado": ["denuncia", "factura"],
        "monto_reclamacion": ["denuncia", "factura", "peritaje"],
        "tipo_siniestro": ["denuncia", "poliza"],
        "fecha_ocurrencia": ["denuncia", "peritaje"],
        "fecha_reclamacion": ["denuncia"],
        "lugar_hechos": ["denuncia", "peritaje"],
        "ajuste": ["peritaje"],
        "conclusiones": ["peritaje", "denuncia"]
    }
    
    # Configuración de OpenAI
    OPENAI_CONFIG = {
        "temperature": 0.1,  # Muy bajo para consistencia
        "max_completion_tokens": 2000,
        "timeout": 30,
        "max_retries": 3
    }
    
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
        "otro": 99
    }


# Acceso directo para compatibilidad con tests  
DOCUMENT_PRIORITIES = {
    "carpeta_de_investigacion": 1,
    "poliza_de_la_aseguradora": 2,
    "guias_y_facturas": 3,
    "informe_preliminar_del_ajustador": 4,
    "informe_final_del_ajustador": 5,
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
    "factura_comercial_cfdi": 22,
    "facturas_comerciales_internacionales": 23,
    "pedido_por_correo": 24,
    "constancia_imss_del_operador": 25,
    "declaracion_universal_de_accidente": 26,
    "reporte_de_costos_y_rendimientos": 27,
    "ficha_tecnica_de_vehiculo": 28,
    "determinacion_de_perdida": 29,
    "otro": 99
}

# Parámetros de clasificación
CLASSIFICATION_CONFIG = {
    "min_confidence_threshold": 0.6,  # Umbral para usar LLM
    "sample_text_length": 1500,       # Caracteres para clasificación
    "llm_model": "gpt-4o-mini",       # Modelo económico para clasificación
    "llm_temperature": 0.1,           # Baja temperatura para consistencia
    "llm_max_completion_tokens": 200             # Límite de tokens para respuesta
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
    "DENUNCIA": "denuncia_inicial",
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

# Función para obtener el modelo óptimo según investigación 2025
def get_model_for_task(task: str, route: str = "ocr_text") -> str:
    """
    Obtiene el modelo óptimo para cada tarea según investigación 2025
    
    Args:
        task: Tipo de tarea ("extraction", "consolidation", "generation")
        route: Ruta de procesamiento ("ocr_text" o "direct_ai")
        
    Returns:
        Nombre del modelo óptimo para la tarea
    """
    if task == "extraction":
        if route == "direct_ai":
            # Para visión: GPT-5 Mini (95% más económico)
            return ModelType.GPT5_VISION_MINI.value
        else:
            # Para OCR + texto: usar GPT-5 Mini para reducir TPM y costos
            return ModelType.GPT5_MINI.value
            
    elif task == "consolidation":
        # Para razonamiento complejo: GPT-5 completo
        return ModelType.GPT5.value
        
    elif task == "generation":
        # Para generación: GPT-5 Mini es eficiente
        return ModelType.GPT5_MINI.value
        
    # Fallback por compatibilidad
    return ModelType.EXTRACTOR.value
