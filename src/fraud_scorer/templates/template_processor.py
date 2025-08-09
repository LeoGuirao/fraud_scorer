"""
Sistema de Llenado Automático de Plantillas para Informes de Siniestros
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json
import re
from jinja2 import Template, Environment, FileSystemLoader
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class TipoDocumento(Enum):
    """Tipos de documentos que el sistema puede procesar"""
    CARTA_RECLAMACION = "carta_reclamacion"
    CARTA_RESPUESTA = "carta_respuesta"
    CARPETA_INVESTIGACION = "carpeta_investigacion"
    TARJETA_CIRCULACION = "tarjeta_circulacion"
    FACTURA_COMPRA = "factura_compra"
    BITACORA_VIAJE = "bitacora_viaje"
    REPORTE_GPS = "reporte_gps"
    DENUNCIA = "denuncia"
    CARTA_PORTE = "carta_porte"
    VALIDACION_MP = "validacion_mp"
    CONSULTA_REPUVE = "consulta_repuve"
    CANDADOS_SEGURIDAD = "candados_seguridad"
    PERITAJE = "peritaje"
    OTRO = "otro"

class NivelAlerta(Enum):
    """Niveles de alerta para hallazgos"""
    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"
    SUCCESS = "success"

class TipoConclusion(Enum):
    """Tipos de conclusión del informe"""
    TENTATIVA = "tentativa"
    PROCEDENTE = "procedente"
    INVESTIGACION = "investigacion"

@dataclass
class DocumentoAnalizado:
    """Representa un documento analizado"""
    tipo_documento: str
    descripcion: str
    hallazgos: List[str] = field(default_factory=list)
    nivel_alerta: str = NivelAlerta.INFO.value
    imagen: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Inconsistencia:
    """Representa una inconsistencia entre documentos"""
    dato: str
    valor_a: str
    valor_b: str
    severidad: str  # critica, alta, media, baja
    documentos_afectados: List[str] = field(default_factory=list)

@dataclass
class ValidacionExterna:
    """Representa una validación externa realizada"""
    tipo: str
    descripcion: str
    resultado: str
    resultado_critico: bool = False
    fecha_validacion: datetime = field(default_factory=datetime.now)

@dataclass
class AnalisisRuta:
    """Análisis técnico de ruta (para siniestros de transporte)"""
    origen_declarado: str
    verificacion_gps_origen: str
    lugar_declarado: str
    verificacion_gps_hechos: str
    analisis_evento: str
    destino_declarado: str
    verificacion_trayectoria: str
    inconsistencias_ruta: List[str] = field(default_factory=list)

@dataclass
class InformeSiniestro:
    """Estructura completa del informe de siniestro"""
    # Información General
    numero_siniestro: str
    nombre_asegurado: str
    numero_poliza: str
    vigencia_desde: str
    vigencia_hasta: str
    domicilio_poliza: str
    bien_reclamado: str
    monto_reclamacion: str
    tipo_siniestro: str
    fecha_ocurrencia: str
    fecha_reclamacion: str
    lugar_hechos: str
    ajustador: str = "PARK PERALES"
    conclusion_preliminar: str = "POR DETERMINAR"
    
    # Contenido del análisis
    analisis_turno: str = ""
    planteamiento_problema: str = ""
    alertas_iniciales: List[str] = field(default_factory=list)
    metodos_investigacion: List[Dict[str, str]] = field(default_factory=list)
    estudio_empresas: str = ""
    empresas_involucradas: List[Dict[str, str]] = field(default_factory=list)
    
    # Documentos analizados
    documentos_analizados: List[DocumentoAnalizado] = field(default_factory=list)
    
    # Análisis técnico
    analisis_ruta: Optional[AnalisisRuta] = None
    
    # Validaciones
    validaciones_externas: List[ValidacionExterna] = field(default_factory=list)
    inconsistencias: List[Inconsistencia] = field(default_factory=list)
    
    # Consideraciones y conclusiones
    consideraciones: List[Dict[str, Any]] = field(default_factory=list)
    conclusion_texto: str = ""
    conclusion_veredicto: str = ""
    conclusion_tipo: str = TipoConclusion.INVESTIGACION.value
    soporte_legal: List[Dict[str, str]] = field(default_factory=list)
    
    # Metadata
    fecha_emision: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    fecha_generacion: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class TemplateProcessor:
    """Procesador principal de plantillas"""
    
    def __init__(self, template_dir: Optional[str] = None):
        project_root = Path(__file__).parents[2]
        self.template_dir = Path(template_dir or project_root / "templates")
        if not self.template_dir.exists():
            raise FileNotFoundError(f"No existe carpeta de plantillas en {self.template_dir}")
        
        # Configurar Jinja2
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )
        
        # Métodos de investigación predefinidos
        self.metodos_base = [
            {
                "nombre": "Cruce de Información",
                "descripcion": "Comparación de datos entre múltiples fuentes para identificar discrepancias y contradicciones."
            },
            {
                "nombre": "Consultas en Fuentes Públicas",
                "descripcion": "Verificación de datos en registros oficiales y bases de datos públicas."
            },
            {
                "nombre": "Verificación Directa con Terceros",
                "descripcion": "Contacto directo con proveedores, transportistas y otras partes involucradas."
            },
            {
                "nombre": "Análisis Documental Forense",
                "descripcion": "Examen técnico de documentos para detectar alteraciones o falsificaciones."
            },
            {
                "nombre": "Análisis de Telemetría",
                "descripcion": "Revisión de datos GPS y sistemas de rastreo vehicular."
            }
        ]
        
        # Artículos legales comunes
        self.articulos_legales = [
            {
                "referencia": "Art. 70 LSCS",
                "texto": "Las obligaciones de la empresa quedarán extinguidas si demuestra que el asegurado, el beneficiario o los representantes de ambos, con el fin de hacerla incurrir en error, disimulan o declaran inexactamente hechos que excluirían o podrían restringir dichas obligaciones."
            },
            {
                "referencia": "Art. 47 LSCS",
                "texto": "Cualquiera omisión o inexacta declaración de los hechos a que se refieren los artículos 8, 9 y 10 de la presente ley, facultará a la empresa aseguradora para considerar rescindido de pleno derecho el contrato."
            }
        ]
    
    def extract_from_documents(self, ocr_results: List[Dict[str, Any]], ai_analysis: Dict[str, Any]) -> InformeSiniestro:
        """
        Extrae información de los resultados de OCR y análisis de AI para crear el informe
        """
        logger.info("Iniciando extracción de información de documentos")
        
        # Crear informe base
        informe = InformeSiniestro(
            numero_siniestro=self._extract_numero_siniestro(ocr_results, ai_analysis),
            nombre_asegurado=self._extract_nombre_asegurado(ocr_results, ai_analysis),
            numero_poliza=self._extract_numero_poliza(ocr_results, ai_analysis),
            vigencia_desde=self._extract_vigencia_desde(ocr_results, ai_analysis),
            vigencia_hasta=self._extract_vigencia_hasta(ocr_results, ai_analysis),
            domicilio_poliza=self._extract_domicilio(ocr_results, ai_analysis),
            bien_reclamado=self._extract_bien_reclamado(ocr_results, ai_analysis),
            monto_reclamacion=self._extract_monto(ocr_results, ai_analysis),
            tipo_siniestro=self._extract_tipo_siniestro(ocr_results, ai_analysis),
            fecha_ocurrencia=self._extract_fecha_ocurrencia(ocr_results, ai_analysis),
            fecha_reclamacion=self._extract_fecha_reclamacion(ocr_results, ai_analysis),
            lugar_hechos=self._extract_lugar_hechos(ocr_results, ai_analysis)
        )
        
        # Agregar análisis y métodos
        informe.analisis_turno = self._generate_analisis_turno(ai_analysis)
        informe.planteamiento_problema = self._generate_planteamiento(ai_analysis)
        informe.metodos_investigacion = self._select_metodos(ai_analysis)
        
        # Procesar documentos individuales
        informe.documentos_analizados = self._process_documents(ocr_results, ai_analysis)
        
        # Detectar inconsistencias
        informe.inconsistencias = self._detect_inconsistencies(ocr_results, ai_analysis)
        
        # Agregar validaciones externas
        informe.validaciones_externas = self._extract_validations(ocr_results, ai_analysis)
        
        # Análisis de ruta si aplica
        if self._is_transport_claim(informe.tipo_siniestro):
            informe.analisis_ruta = self._extract_route_analysis(ocr_results, ai_analysis)
        
        # Generar consideraciones y conclusión
        informe.consideraciones = self._generate_considerations(informe, ai_analysis)
        informe.conclusion_texto, informe.conclusion_veredicto, informe.conclusion_tipo = \
            self._generate_conclusion(informe, ai_analysis)
        
        # Agregar soporte legal si hay tentativa de fraude
        if informe.conclusion_tipo == TipoConclusion.TENTATIVA.value:
            informe.soporte_legal = self.articulos_legales
        
        logger.info(f"Extracción completada para siniestro {informe.numero_siniestro}")
        return informe
    
    def _extract_numero_siniestro(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae el número de siniestro"""
        # Buscar en resultados de OCR
        for doc in ocr_results:
            if 'numero_siniestro' in doc.get('entities', {}):
                return doc['entities']['numero_siniestro']
            
            # Buscar en key-value pairs
            for key, value in doc.get('key_values', {}).items():
                if 'siniestro' in key.lower() and 'numero' in key.lower():
                    return value
        
        # Buscar en análisis AI
        if 'claim_number' in ai_analysis:
            return ai_analysis['claim_number']
        
        return "SIN_NUMERO"
    
    def _extract_nombre_asegurado(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae el nombre del asegurado"""
        for doc in ocr_results:
            # Buscar en entidades
            if 'nombre_asegurado' in doc.get('entities', {}):
                return doc['entities']['nombre_asegurado']
            
            # Buscar en key-value pairs
            for key, value in doc.get('key_values', {}).items():
                if 'asegurado' in key.lower() and 'nombre' in key.lower():
                    return value
        
        return ai_analysis.get('insured_name', 'NO IDENTIFICADO')
    
    def _extract_numero_poliza(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae el número de póliza"""
        for doc in ocr_results:
            for key, value in doc.get('key_values', {}).items():
                if 'poliza' in key.lower() or 'póliza' in key.lower():
                    # Limpiar y formatear número de póliza
                    return re.sub(r'[^\d\-]', '', value)
        
        return ai_analysis.get('policy_number', 'SIN_POLIZA')
    
    def _extract_vigencia_desde(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae fecha de inicio de vigencia"""
        for doc in ocr_results:
            text = doc.get('text', '')
            # Buscar patrón de vigencia
            match = re.search(r'vigencia.*?del?\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
            if match:
                return self._format_date(match.group(1))
        
        return ai_analysis.get('policy_start_date', '')
    
    def _extract_vigencia_hasta(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae fecha de fin de vigencia"""
        for doc in ocr_results:
            text = doc.get('text', '')
            # Buscar patrón de vigencia
            match = re.search(r'vigencia.*?al\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
            if match:
                return self._format_date(match.group(1))
        
        return ai_analysis.get('policy_end_date', '')
    
    def _extract_domicilio(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae el domicilio del asegurado"""
        for doc in ocr_results:
            # Buscar en entidades
            for entity in doc.get('entities', []):
                if entity.get('category') == 'Address':
                    return entity.get('text', '')
        
        return ai_analysis.get('insured_address', 'NO ESPECIFICADO')
    
    def _extract_bien_reclamado(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae descripción del bien reclamado"""
        for doc in ocr_results:
            if doc.get('document_type') == 'factura':
                # Buscar descripción de mercancía
                for key, value in doc.get('key_values', {}).items():
                    if 'descripcion' in key.lower() or 'mercancia' in key.lower():
                        return value
        
        return ai_analysis.get('claimed_goods', 'MERCANCÍA DIVERSA')
    
    def _extract_monto(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae el monto reclamado"""
        montos = []
        for doc in ocr_results:
            # Buscar montos en entidades
            for entity in doc.get('entities', []):
                if entity.get('category') == 'Currency':
                    # Limpiar y convertir a número
                    monto_str = entity.get('normalized', entity.get('text', ''))
                    try:
                        monto = float(re.sub(r'[^\d.]', '', monto_str))
                        montos.append(monto)
                    except:
                        pass
        
        # Retornar el monto más alto encontrado
        if montos:
            return f"{max(montos):,.2f}"
        
        return ai_analysis.get('claim_amount', '0.00')
    
    def _extract_tipo_siniestro(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae el tipo de siniestro"""
        tipos_comunes = {
            'robo': 'ROBO',
            'colision': 'COLISIÓN',
            'incendio': 'INCENDIO',
            'daño': 'DAÑOS',
            'perdida': 'PÉRDIDA TOTAL',
            'bulto': 'ROBO DE BULTO POR ENTERO'
        }
        
        for doc in ocr_results:
            text = doc.get('text', '').lower()
            for keyword, tipo in tipos_comunes.items():
                if keyword in text:
                    return tipo
        
        return ai_analysis.get('claim_type', 'NO ESPECIFICADO')
    
    def _extract_fecha_ocurrencia(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae fecha de ocurrencia del siniestro"""
        for doc in ocr_results:
            if doc.get('document_type') == 'denuncia':
                # Buscar fechas en el documento
                for entity in doc.get('entities', []):
                    if entity.get('category') == 'DateTime':
                        return self._format_date(entity.get('text', ''))
        
        return ai_analysis.get('occurrence_date', '')
    
    def _extract_fecha_reclamacion(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae fecha de reclamación"""
        for doc in ocr_results:
            if doc.get('document_type') == 'carta_reclamacion':
                # Buscar fecha en carta
                for entity in doc.get('entities', []):
                    if entity.get('category') == 'DateTime':
                        return self._format_date(entity.get('text', ''))
        
        return ai_analysis.get('claim_date', '')
    
    def _extract_lugar_hechos(self, ocr_results: List[Dict], ai_analysis: Dict) -> str:
        """Extrae el lugar de los hechos"""
        for doc in ocr_results:
            if doc.get('document_type') in ['denuncia', 'carpeta_investigacion']:
                # Buscar direcciones o lugares
                for entity in doc.get('entities', []):
                    if entity.get('category') in ['Address', 'Location']:
                        return entity.get('text', '')
        
        return ai_analysis.get('incident_location', 'NO ESPECIFICADO')
    
    def _generate_analisis_turno(self, ai_analysis: Dict) -> str:
        """Genera el texto de análisis del turno"""
        return (
            "Como inicio del presente análisis documental y validación de documentación "
            "que se realizó de forma técnica basada en principios de criminalística, "
            "jurídicos y criminológicos para postular una sugerencia con sustentos "
            "indubitables a la compañía."
        )
    
    def _generate_planteamiento(self, ai_analysis: Dict) -> str:
        """Genera el planteamiento del problema basado en el análisis"""
        if ai_analysis.get('fraud_indicators'):
            return (
                "Se presenta una reclamación con documentación que muestra "
                "inconsistencias significativas que requieren investigación detallada. "
                f"Se han identificado {len(ai_analysis['fraud_indicators'])} "
                "indicadores que sugieren posibles irregularidades en la reclamación."
            )
        else:
            return (
                "Se presenta una reclamación por siniestro que requiere validación "
                "exhaustiva de la documentación presentada y verificación de los "
                "hechos declarados para determinar su procedencia."
            )
    
    def _select_metodos(self, ai_analysis: Dict) -> List[Dict[str, str]]:
        """Selecciona los métodos de investigación aplicables"""
        metodos = self.metodos_base.copy()
        
        # Agregar métodos específicos según el tipo de siniestro
        if 'gps' in str(ai_analysis).lower():
            metodos.append({
                "nombre": "Análisis de Telemetría GPS",
                "descripcion": "Revisión detallada de datos de posicionamiento y telemetría vehicular."
            })
        
        if 'repuve' in str(ai_analysis).lower():
            metodos.append({
                "nombre": "Consulta REPUVE",
                "descripcion": "Verificación en el Registro Público Vehicular para validar datos del vehículo."
            })
        
        return metodos
    
    def _process_documents(self, ocr_results: List[Dict], ai_analysis: Dict) -> List[DocumentoAnalizado]:
        """Procesa cada documento y genera su análisis"""
        documentos = []
        
        for doc in ocr_results:
            tipo = doc.get('document_type', 'otro')
            
            # Determinar nivel de alerta basado en hallazgos
            nivel_alerta = NivelAlerta.INFO.value
            hallazgos = []
            
            # Verificar si hay anomalías detectadas
            if 'anomalies' in doc:
                hallazgos.extend(doc['anomalies'])
                nivel_alerta = NivelAlerta.WARNING.value
            
            # Verificar si es documento apócrifo
            if doc.get('is_fake', False):
                hallazgos.append("DOCUMENTO APÓCRIFO CONFIRMADO")
                nivel_alerta = NivelAlerta.DANGER.value
            
            documento = DocumentoAnalizado(
                tipo_documento=self._format_document_type(tipo),
                descripcion=doc.get('description', f'Documento tipo {tipo}'),
                hallazgos=hallazgos,
                nivel_alerta=nivel_alerta,
                imagen=doc.get('image_path'),
                metadata=doc.get('metadata', {})
            )
            
            documentos.append(documento)
        
        return documentos
    
    def _detect_inconsistencies(self, ocr_results: List[Dict], ai_analysis: Dict) -> List[Inconsistencia]:
        """Detecta inconsistencias entre documentos"""
        inconsistencias = []
        
        # Buscar inconsistencias en el análisis de AI
        if 'inconsistencies' in ai_analysis:
            for inc in ai_analysis['inconsistencies']:
                inconsistencia = Inconsistencia(
                    dato=inc.get('field', 'Campo no especificado'),
                    valor_a=inc.get('value_a', ''),
                    valor_b=inc.get('value_b', ''),
                    severidad=inc.get('severity', 'media'),
                    documentos_afectados=inc.get('affected_docs', [])
                )
                inconsistencias.append(inconsistencia)
        
        return inconsistencias
    
    def _extract_validations(self, ocr_results: List[Dict], ai_analysis: Dict) -> List[ValidacionExterna]:
        """Extrae validaciones externas realizadas"""
        validaciones = []
        
        # Buscar validaciones en el análisis
        if 'external_validations' in ai_analysis:
            for val in ai_analysis['external_validations']:
                validacion = ValidacionExterna(
                    tipo=val.get('type', 'Verificación'),
                    descripcion=val.get('description', ''),
                    resultado=val.get('result', ''),
                    resultado_critico=val.get('is_critical', False)
                )
                validaciones.append(validacion)
        
        return validaciones
    
    def _is_transport_claim(self, tipo_siniestro: str) -> bool:
        """Determina si es un siniestro de transporte"""
        keywords = ['robo', 'transporte', 'carga', 'bulto', 'mercancía']
        return any(keyword in tipo_siniestro.lower() for keyword in keywords)
    
    def _extract_route_analysis(self, ocr_results: List[Dict], ai_analysis: Dict) -> AnalisisRuta:
        """Extrae análisis de ruta para siniestros de transporte"""
        route_data = ai_analysis.get('route_analysis', {})
        
        return AnalisisRuta(
            origen_declarado=route_data.get('declared_origin', ''),
            verificacion_gps_origen=route_data.get('gps_origin_verification', ''),
            lugar_declarado=route_data.get('declared_incident_location', ''),
            verificacion_gps_hechos=route_data.get('gps_incident_verification', ''),
            analisis_evento=route_data.get('event_analysis', ''),
            destino_declarado=route_data.get('declared_destination', ''),
            verificacion_trayectoria=route_data.get('trajectory_verification', ''),
            inconsistencias_ruta=route_data.get('route_inconsistencies', [])
        )
    
    def _generate_considerations(self, informe: InformeSiniestro, ai_analysis: Dict) -> List[Dict]:
        """Genera las consideraciones del informe"""
        consideraciones = []
        
        # Si hay documentos apócrifos
        docs_apocrifos = [d for d in informe.documentos_analizados 
                         if 'APÓCRIFO' in ' '.join(d.hallazgos)]
        if docs_apocrifos:
            consideraciones.append({
                "titulo": "Falsificación Comprobada de Documentación Clave",
                "descripcion": (
                    "Se ha confirmado mediante verificación directa que los documentos "
                    "presentados son apócrifos. La presentación de documentos falsos "
                    "es un acto doloso e intencional."
                ),
                "evidencias": [f"Documento {d.tipo_documento} confirmado como falso" 
                              for d in docs_apocrifos]
            })
        
        # Si hay inconsistencias críticas
        inconsistencias_criticas = [i for i in informe.inconsistencias 
                                   if i.severidad == 'critica']
        if inconsistencias_criticas:
            consideraciones.append({
                "titulo": "Inconsistencias Críticas en la Información",
                "descripcion": (
                    "Se han detectado contradicciones fundamentales en la información "
                    "presentada que no pueden explicarse por errores administrativos."
                ),
                "evidencias": [f"{i.dato}: {i.valor_a} vs {i.valor_b}" 
                              for i in inconsistencias_criticas]
            })
        
        return consideraciones
    
    def _generate_conclusion(self, informe: InformeSiniestro, ai_analysis: Dict) -> tuple:
        """Genera la conclusión del informe"""
        # Determinar tipo de conclusión basado en evidencia
        fraud_score = ai_analysis.get('fraud_score', 0)
        
        if fraud_score > 0.7 or any('APÓCRIFO' in str(d.hallazgos) for d in informe.documentos_analizados):
            texto = (
                "La reclamación presentada está viciada de origen por el dolo manifestado "
                "en la presentación de documentación apócrifa, un acto que constituye una "
                "violación fundamental al principio de Máxima Buena Fe que rige el contrato "
                "de seguro."
            )
            veredicto = "CON TENTATIVA DE FRAUDE"
            tipo = TipoConclusion.TENTATIVA.value
        elif fraud_score < 0.3:
            texto = (
                "Tras el análisis exhaustivo de la documentación presentada, no se han "
                "encontrado elementos que sugieran irregularidades en la reclamación."
            )
            veredicto = "PROCEDENTE"
            tipo = TipoConclusion.PROCEDENTE.value
        else:
            texto = (
                "Se han identificado elementos que requieren investigación adicional "
                "antes de poder determinar la procedencia de la reclamación."
            )
            veredicto = "REQUIERE INVESTIGACIÓN ADICIONAL"
            tipo = TipoConclusion.INVESTIGACION.value
        
        return texto, veredicto, tipo
    
    def _format_date(self, date_str: str) -> str:
        """Formatea una fecha a formato estándar"""
        # Implementar lógica de formateo de fecha
        return date_str
    
    def _format_document_type(self, tipo: str) -> str:
        """Formatea el tipo de documento para mostrar"""
        mappings = {
            'carta_reclamacion': 'CARTA DE RECLAMACIÓN',
            'carta_respuesta': 'CARTA RESPUESTA DEL TRANSPORTISTA',
            'carpeta_investigacion': 'CARPETA DE INVESTIGACIÓN',
            'tarjeta_circulacion': 'TARJETA DE CIRCULACIÓN',
            'factura_compra': 'FACTURA DE COMPRA',
            'bitacora_viaje': 'BITÁCORA DE VIAJE',
            'reporte_gps': 'REPORTE GPS',
            'denuncia': 'DENUNCIA',
            'carta_porte': 'CARTA PORTE'
        }
        return mappings.get(tipo, tipo.upper().replace('_', ' '))
    
        # ...existing code...
    def generate_report(self, informe: InformeSiniestro, output_path: str = None) -> str:
        """
        Genera el informe HTML final
        """
        # Convertir dataclass a dict para Jinja2
        data = self._dataclass_to_dict(informe)
        
        # Cargar y renderizar plantilla usando Jinja2 y el nombre correcto
        template = self.env.get_template('report_template.html')
        html_output = template.render(**data)
        
        # Guardar si se especifica path
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_output)
            logger.info(f"Informe generado en: {output_path}")
        
        return html_output
    # ...existing code...
    
    def _dataclass_to_dict(self, obj):
        """Convierte dataclass a diccionario recursivamente"""
        if hasattr(obj, '__dataclass_fields__'):
            return {k: self._dataclass_to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [self._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._dataclass_to_dict(v) for k, v in obj.items()}
        else:
            return obj


# Función principal para procesar un siniestro completo
async def process_claim(documents_folder: str, output_folder: str) -> str:
    """
    Procesa todos los documentos de un siniestro y genera el informe
    
    Args:
        documents_folder: Carpeta con los documentos del siniestro
        output_folder: Carpeta donde guardar el informe generado
    
    Returns:
        Path al informe generado
    """
    from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
    from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
    
    logger.info(f"Procesando siniestro en: {documents_folder}")
    
    # Inicializar procesadores
    ocr_processor = AzureOCRProcessor()
    ai_analyzer = AIDocumentAnalyzer()
    template_processor = TemplateProcessor()
    
    # Procesar todos los documentos con OCR
    ocr_results = []
    documents_path = Path(documents_folder)
    
    for doc_file in documents_path.glob("*"):
        if doc_file.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']:
            logger.info(f"Procesando documento: {doc_file.name}")
            
            # OCR
            ocr_result = ocr_processor.analyze_document(str(doc_file))
            ocr_result['file_name'] = doc_file.name
            ocr_result['document_type'] = _detect_document_type(doc_file.name, ocr_result)
            ocr_results.append(ocr_result)
    
    # Análisis con AI
    ai_analysis = await ai_analyzer.analyze_claim_documents(ocr_results)
    
    # Extraer información y generar informe
    informe = template_processor.extract_from_documents(ocr_results, ai_analysis)
    
    # Generar HTML
    output_path = Path(output_folder) / f"INF-{informe.numero_siniestro}.html"
    template_processor.generate_report(informe, str(output_path))
    
    logger.info(f"Informe generado exitosamente: {output_path}")
    return str(output_path)


def _detect_document_type(filename: str, ocr_result: Dict) -> str:
    """Detecta el tipo de documento basado en el nombre y contenido"""
    filename_lower = filename.lower()
    text_lower = ocr_result.get('text', '').lower()
    
    if 'reclamacion' in filename_lower or 'reclamación' in text_lower:
        return 'carta_reclamacion'
    elif 'respuesta' in filename_lower:
        return 'carta_respuesta'
    elif 'carpeta' in filename_lower or 'investigacion' in filename_lower:
        return 'carpeta_investigacion'
    elif 'circulacion' in filename_lower or 'circulación' in text_lower:
        return 'tarjeta_circulacion'
    elif 'factura' in filename_lower or 'cfdi' in text_lower:
        return 'factura_compra'
    elif 'gps' in filename_lower or 'telemetria' in text_lower:
        return 'reporte_gps'
    elif 'denuncia' in filename_lower:
        return 'denuncia'
    elif 'carta' in filename_lower and 'porte' in filename_lower:
        return 'carta_porte'
    else:
        return 'otro'
    
from weasyprint import HTML

class TemplateProcessor:

    def generate_pdf(self, informe: InformeSiniestro, output_pdf: str) -> None:
        """
        Genera el informe en PDF a partir del HTML.
        """
        html_str = self.generate_report(informe)
        HTML(string=html_str).write_pdf(output_pdf)
        logger.info(f"PDF generado en: {output_pdf}")

# Funciones template_processor.py extras

def generate_pdf_from_html(html_content: str, output_path: str) -> bool:
    """
    Genera un PDF a partir del contenido HTML usando WeasyPrint
    
    Args:
        html_content: String con el HTML del informe
        output_path: Ruta donde guardar el PDF
        
    Returns:
        True si se generó exitosamente, False en caso contrario
    """
    try:
        from weasyprint import HTML, CSS
        import logging
        
        # Silenciar warnings de WeasyPrint
        logging.getLogger('weasyprint').setLevel(logging.ERROR)
        
        # CSS personalizado para mejor renderizado en PDF
        pdf_css = CSS(string='''
            @page {
                size: A4;
                margin: 20mm 15mm;
                @top-center {
                    content: "Sistema de Análisis de Siniestros";
                    font-size: 10pt;
                    color: #666;
                }
                @bottom-right {
                    content: "Página " counter(page) " de " counter(pages);
                    font-size: 10pt;
                    color: #666;
                }
            }
            
            body {
                font-family: 'Arial', 'Helvetica', sans-serif;
                font-size: 10pt;
                line-height: 1.5;
                color: #333;
            }
            
            h1 {
                font-size: 18pt;
                color: #2c3e50;
                margin-bottom: 20px;
            }
            
            h2 {
                font-size: 14pt;
                color: #2c3e50;
                margin-top: 20px;
                margin-bottom: 15px;
                page-break-after: avoid;
            }
            
            h3 {
                font-size: 12pt;
                color: #34495e;
                margin-top: 15px;
                margin-bottom: 10px;
                page-break-after: avoid;
            }
            
            .section {
                page-break-inside: avoid;
                margin-bottom: 20px;
            }
            
            .info-general {
                page-break-inside: avoid;
            }
            
            .info-grid {
                display: grid;
                grid-template-columns: 35% 65%;
                gap: 5px;
                font-size: 9pt;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 9pt;
                page-break-inside: avoid;
            }
            
            th {
                background: #34495e;
                color: white;
                padding: 8px;
                text-align: left;
                font-weight: normal;
            }
            
            td {
                padding: 6px 8px;
                border-bottom: 1px solid #dee2e6;
            }
            
            .alert {
                padding: 10px;
                margin: 15px 0;
                border-radius: 3px;
                border-left: 4px solid;
                page-break-inside: avoid;
            }
            
            .alert-danger {
                background: #f8d7da;
                color: #721c24;
                border-color: #dc3545;
            }
            
            .alert-warning {
                background: #fff3cd;
                color: #856404;
                border-color: #ffc107;
            }
            
            .conclusion {
                page-break-inside: avoid;
                background: #f8f9fa;
                border: 1px solid #2c3e50;
                border-radius: 5px;
                padding: 20px;
                margin-top: 30px;
            }
            
            .conclusion-verdict {
                font-size: 14pt;
                font-weight: bold;
                text-align: center;
                padding: 10px;
                margin: 15px 0;
                border-radius: 3px;
            }
            
            .verdict-tentativa {
                background: #dc3545;
                color: white;
            }
            
            .verdict-procedente {
                background: #28a745;
                color: white;
            }
            
            .verdict-investigacion {
                background: #ffc107;
                color: #212529;
            }
            
            .evidence-box {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                padding: 10px;
                margin: 10px 0;
                page-break-inside: avoid;
                font-size: 9pt;
            }
            
            .legal-article {
                font-family: 'Courier New', monospace;
                background: white;
                padding: 10px;
                border-left: 3px solid #6c757d;
                margin: 10px 0;
                font-size: 9pt;
            }
            
            /* Evitar saltos de página en elementos importantes */
            h1, h2, h3, h4 {
                page-break-after: avoid;
            }
            
            p {
                orphans: 3;
                widows: 3;
            }
            
            /* Numeración de secciones */
            .section {
                counter-increment: section;
            }
            
            .section h2:before {
                content: counter(section) ". ";
            }
            
            /* Mejoras para impresión */
            @media print {
                .no-print {
                    display: none;
                }
                
                a {
                    text-decoration: none;
                    color: #333;
                }
            }
        ''')
        
        # Generar PDF
        HTML(string=html_content).write_pdf(output_path, stylesheets=[pdf_css])
        
        return True
        
    except ImportError:
        print("WeasyPrint no está instalado. Instala con: pip install weasyprint")
        return False
    except Exception as e:
        print(f"Error generando PDF: {e}")
        return False


def extract_from_documents(self, documents: List[Dict[str, Any]], ai_analysis: Dict[str, Any]) -> InformeSiniestro:
    """
    Versión mejorada que maneja mejor los documentos procesados
    """
    logger.info("Iniciando extracción de información de documentos")
    
    # Crear informe base con valores extraídos
    informe = InformeSiniestro(
        numero_siniestro=self._extract_from_all_docs(documents, 'numero_siniestro', ai_analysis),
        nombre_asegurado=self._extract_from_all_docs(documents, 'nombre_asegurado', ai_analysis),
        numero_poliza=self._extract_from_all_docs(documents, 'numero_poliza', ai_analysis),
        vigencia_desde=self._extract_date_field(documents, 'vigencia_inicio', ai_analysis),
        vigencia_hasta=self._extract_date_field(documents, 'vigencia_fin', ai_analysis),
        domicilio_poliza=self._extract_address(documents, ai_analysis),
        bien_reclamado=self._extract_from_all_docs(documents, 'bien_reclamado', ai_analysis),
        monto_reclamacion=self._extract_amount(documents, ai_analysis),
        tipo_siniestro=self._extract_claim_type(documents, ai_analysis),
        fecha_ocurrencia=self._extract_date_field(documents, 'fecha_siniestro', ai_analysis),
        fecha_reclamacion=self._extract_date_field(documents, 'fecha_reclamacion', ai_analysis),
        lugar_hechos=self._extract_location(documents, ai_analysis)
    )
    
    # Agregar análisis
    informe.analisis_turno = self._generate_analisis_turno(ai_analysis)
    informe.planteamiento_problema = self._generate_planteamiento(ai_analysis)
    
    # Alertas iniciales basadas en el análisis
    if ai_analysis.get('fraud_indicators'):
        informe.alertas_iniciales = [
            ind.get('description', 'Indicador de riesgo detectado')
            for ind in ai_analysis['fraud_indicators'][:5]  # Máximo 5 alertas
        ]
    
    # Métodos de investigación
    informe.metodos_investigacion = self._select_metodos(ai_analysis)
    
    # Procesar documentos
    informe.documentos_analizados = self._process_documents_enhanced(documents, ai_analysis)
    
    # Inconsistencias
    if 'inconsistencies' in ai_analysis:
        informe.inconsistencias = [
            Inconsistencia(
                dato=inc.get('field', 'Campo'),
                valor_a=str(inc.get('values', ['', ''])[0] if inc.get('values') else ''),
                valor_b=str(inc.get('values', ['', ''])[1] if len(inc.get('values', [])) > 1 else ''),
                severidad=inc.get('severity', 'media'),
                documentos_afectados=inc.get('affected_documents', [])
            )
            for inc in ai_analysis['inconsistencies']
        ]
    
    # Generar consideraciones y conclusión
    informe.consideraciones = self._generate_considerations_enhanced(informe, ai_analysis)
    
    # Determinar conclusión basada en el score de fraude
    fraud_score = ai_analysis.get('fraud_score', 0)
    informe.conclusion_texto, informe.conclusion_veredicto, informe.conclusion_tipo = \
        self._generate_conclusion_enhanced(fraud_score, informe, ai_analysis)
    
    # Agregar soporte legal si es necesario
    if informe.conclusion_tipo == TipoConclusion.TENTATIVA.value:
        informe.soporte_legal = self.articulos_legales
    
    logger.info(f"Extracción completada para siniestro {informe.numero_siniestro}")
    return informe


def _extract_from_all_docs(self, documents: List[Dict], field: str, ai_analysis: Dict) -> str:
    """Extrae un campo específico de todos los documentos"""
    # Buscar en entidades de cada documento
    for doc in documents:
        if 'entities' in doc and field in doc['entities']:
            values = doc['entities'][field]
            if values and isinstance(values, list):
                return values[0]
            elif values:
                return str(values)
    
    # Buscar en campos específicos
    for doc in documents:
        if 'specific_fields' in doc and field in doc['specific_fields']:
            return str(doc['specific_fields'][field])
    
    # Buscar en key-value pairs
    for doc in documents:
        if 'key_value_pairs' in doc:
            for key, value in doc['key_value_pairs'].items():
                if field.lower() in key.lower():
                    return value
    
    # Fallback al análisis AI
    return ai_analysis.get(field, 'NO ESPECIFICADO')


def _extract_amount(self, documents: List[Dict], ai_analysis: Dict) -> str:
    """Extrae el monto más alto encontrado"""
    amounts = []
    
    for doc in documents:
        if 'entities' in doc and 'moneda' in doc['entities']:
            for amount_str in doc['entities']['moneda']:
                try:
                    # Limpiar y convertir
                    clean_amount = amount_str.replace('$', '').replace(',', '')
                    amount = float(clean_amount)
                    amounts.append(amount)
                except:
                    pass
    
    if amounts:
        max_amount = max(amounts)
        return f"{max_amount:,.2f}"
    
    return ai_analysis.get('claim_amount', '0.00')


def _extract_claim_type(self, documents: List[Dict], ai_analysis: Dict) -> str:
    """Determina el tipo de siniestro"""
    claim_types = {
        'robo': 'ROBO',
        'colision': 'COLISIÓN', 
        'colisión': 'COLISIÓN',
        'incendio': 'INCENDIO',
        'daño': 'DAÑOS',
        'daños': 'DAÑOS',
        'bulto': 'ROBO DE BULTO POR ENTERO',
        'mercancia': 'ROBO DE MERCANCÍA',
        'mercancía': 'ROBO DE MERCANCÍA'
    }
    
    # Buscar en todos los documentos
    for doc in documents:
        text = doc.get('raw_text', '').lower()
        for keyword, claim_type in claim_types.items():
            if keyword in text:
                return claim_type
    
    return ai_analysis.get('claim_type', 'NO ESPECIFICADO')


def _generate_conclusion_enhanced(self, fraud_score: float, informe: InformeSiniestro, ai_analysis: Dict) -> tuple:
    """Genera conclusión mejorada basada en el análisis"""
    
    # Verificar si hay documentos apócrifos
    has_fake_docs = any(
        'APÓCRIFO' in ' '.join(d.hallazgos) 
        for d in informe.documentos_analizados
    )
    
    # Verificar inconsistencias críticas
    critical_inconsistencies = [
        i for i in informe.inconsistencias 
        if i.severidad == 'critica' or i.severidad == 'critical'
    ]
    
    if has_fake_docs or fraud_score > 0.8:
        texto = (
            "La reclamación presentada está viciada de origen por el dolo manifestado "
            "en la presentación de documentación apócrifa, un acto que constituye una "
            "violación fundamental al principio de Máxima Buena Fe que rige el contrato de seguro."
        )
        veredicto = "CON TENTATIVA DE FRAUDE"
        tipo = TipoConclusion.TENTATIVA.value
        
    elif fraud_score > 0.5 or len(critical_inconsistencies) > 0:
        texto = (
            "Se han identificado inconsistencias significativas que requieren "
            "investigación adicional antes de determinar la procedencia de la reclamación."
        )
        veredicto = "REQUIERE INVESTIGACIÓN ADICIONAL"
        tipo = TipoConclusion.INVESTIGACION.value
        
    else:
        texto = (
            "Tras el análisis exhaustivo de la documentación presentada, "
            "no se han encontrado elementos que sugieran irregularidades significativas."
        )
        veredicto = "PROCEDENTE SALVO MEJOR OPINIÓN"
        tipo = TipoConclusion.PROCEDENTE.value
    
    return texto, veredicto, tipo