"""
Sistema de Llenado Automático de Plantillas para Informes de Siniestros
Actualizado para trabajar con IntelligentFieldExtractor
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json
import re
import logging
from enum import Enum
import os
import unicodedata

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

logger = logging.getLogger(__name__)


# =========================
# Enums y Dataclasses
# =========================
class TipoDocumento(Enum):
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
    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"
    SUCCESS = "success"


class TipoConclusion(Enum):
    TENTATIVA = "tentativa"
    PROCEDENTE = "procedente"
    INVESTIGACION = "investigacion"


@dataclass
class DocumentoAnalizado:
    tipo_documento: str
    descripcion: str
    hallazgos: List[str] = field(default_factory=list)
    nivel_alerta: str = NivelAlerta.INFO.value
    imagen: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Inconsistencia:
    dato: str
    valor_a: str
    valor_b: str
    severidad: str  # critica, alta, media, baja
    documentos_afectados: List[str] = field(default_factory=list)


@dataclass
class ValidacionExterna:
    tipo: str
    descripcion: str
    resultado: str
    resultado_critico: bool = False
    fecha_validacion: datetime = field(default_factory=datetime.now)


@dataclass
class AnalisisRuta:
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


# =========================
# Clase principal
# =========================
class TemplateProcessor:
    """
    Procesador de plantillas del informe.
    🔄 ACTUALIZADO para trabajar mejor con IntelligentFieldExtractor
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Prioridad para localizar la carpeta de plantillas:
        1) Param. template_dir (si se pasa)
        2) ENV FRAUD_TEMPLATES_DIR (si existe)
        3) Carpeta del propio archivo (src/fraud_scorer/templates)
        """
        env_dir = os.getenv("FRAUD_TEMPLATES_DIR")
        default_dir = Path(__file__).parent
        self.template_dir = Path(template_dir or env_dir or default_dir)

        if not self.template_dir.exists():
            raise FileNotFoundError(f"No existe carpeta de plantillas en {self.template_dir}")

        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)), autoescape=True)

        # Métodos base
        self.metodos_base: List[Dict[str, str]] = [
            {"nombre": "Cruce de Información", "descripcion": "Comparación de datos entre fuentes para detectar contradicciones."},
            {"nombre": "Consultas en Fuentes Públicas", "descripcion": "Verificación en registros oficiales y bases públicas."},
            {"nombre": "Verificación Directa con Terceros", "descripcion": "Contacto con proveedores/transportistas involucrados."},
            {"nombre": "Análisis Documental Forense", "descripcion": "Examen técnico para detectar alteraciones o falsificaciones."},
            {"nombre": "Análisis de Telemetría", "descripcion": "Revisión de datos GPS y rastreo vehicular."},
        ]

        # Soporte legal base
        self.articulos_legales: List[Dict[str, str]] = [
            {
                "referencia": "Art. 70 LSCS",
                "texto": "Obligaciones extinguidas si se disimulan o declaran inexactamente hechos que excluirían obligaciones.",
            },
            {
                "referencia": "Art. 47 LSCS",
                "texto": "Omisiones o declaraciones inexactas facultan a considerar rescindido el contrato.",
            },
        ]

    # ------------------------
    # API usada por pipelines
    # ------------------------
    def extract_from_documents(self, documents: List[Dict[str, Any]], ai_analysis: Dict[str, Any]) -> InformeSiniestro:
        """
        🔄 ACTUALIZADO: Extrae campos con prioridad de IntelligentFieldExtractor
        Los campos ahora vienen en specific_fields gracias al nuevo extractor
        """
        logger.info("Extrayendo información desde documentos con IntelligentFieldExtractor")

        # Usar el case_id si está disponible, si no usar "1"
        numero_siniestro = ai_analysis.get('case_id', '1')
        if numero_siniestro == '1':
            # Intentar extraer de otro lugar
            for doc in documents:
                sf = doc.get('specific_fields', {})
                if sf.get('numero_siniestro'):
                    numero_siniestro = str(sf['numero_siniestro'])
                    break

        informe = InformeSiniestro(
            numero_siniestro=numero_siniestro,
            nombre_asegurado=self._extract_field_intelligent(documents, 'nombre_asegurado', ai_analysis),
            numero_poliza=self._extract_field_intelligent(documents, 'numero_poliza', ai_analysis),
            vigencia_desde=self._extract_field_intelligent(documents, 'vigencia_inicio', ai_analysis),
            vigencia_hasta=self._extract_field_intelligent(documents, 'vigencia_fin', ai_analysis),
            domicilio_poliza=self._extract_field_intelligent(documents, 'domicilio_poliza', ai_analysis),
            bien_reclamado=self._extract_field_intelligent(documents, 'bien_reclamado', ai_analysis),
            monto_reclamacion=self._format_amount(
                self._extract_field_intelligent(documents, 'monto_reclamacion', ai_analysis) or
                self._extract_field_intelligent(documents, 'total', ai_analysis)
            ),
            tipo_siniestro=self._extract_field_intelligent(documents, 'tipo_siniestro', ai_analysis),
            fecha_ocurrencia=self._format_date(
                self._extract_field_intelligent(documents, 'fecha_siniestro', ai_analysis) or
                self._extract_field_intelligent(documents, 'fecha_ocurrencia', ai_analysis)
            ),
            fecha_reclamacion=self._format_date(
                self._extract_field_intelligent(documents, 'fecha_reclamacion', ai_analysis)
            ),
            lugar_hechos=self._extract_field_intelligent(documents, 'lugar_hechos', ai_analysis),
        )

        # Texto general
        informe.analisis_turno = self._generate_analisis_turno(ai_analysis)
        informe.planteamiento_problema = self._generate_planteamiento(ai_analysis)

        # Alertas iniciales sugeridas por IA
        if ai_analysis.get("fraud_indicators"):
            informe.alertas_iniciales = [
                (ind.get("description") or "Indicador de riesgo detectado")
                for ind in ai_analysis["fraud_indicators"][:5]
            ]

        # Métodos aplicables
        informe.metodos_investigacion = self._select_metodos(ai_analysis)

        # Documentos analizados
        informe.documentos_analizados = self._process_documents_enhanced(documents, ai_analysis)

        # Inconsistencias (de IA)
        informe.inconsistencias = self._build_inconsistencies(ai_analysis)

        # Análisis de ruta (si aplica)
        route = self._extract_route_analysis(ai_analysis)
        if route:
            informe.analisis_ruta = route

        # Consideraciones y conclusión
        informe.consideraciones = self._generate_considerations_enhanced(informe, ai_analysis)
        fraud_score = float(ai_analysis.get("fraud_score", 0.0))
        informe.conclusion_texto, informe.conclusion_veredicto, informe.conclusion_tipo = \
            self._generate_conclusion_enhanced(fraud_score, informe, ai_analysis)

        # Soporte legal si procede
        if informe.conclusion_tipo == TipoConclusion.TENTATIVA.value:
            informe.soporte_legal = self.articulos_legales

        logger.info(f"Extracción completada para siniestro {informe.numero_siniestro}")
        return informe

    def generate_report(self, informe: InformeSiniestro, output_path: Optional[str] = None) -> str:
        """
        Renderiza el HTML del informe usando Jinja2.
        Si `output_path` se pasa, escribe el archivo HTML en disco.
        """
        data = self._dataclass_to_dict(informe)

        try:
            template = self.env.get_template("report_template.html")
            html_output = template.render(**data)
        except TemplateNotFound:
            logger.warning("Plantilla report_template.html no encontrada. Usando fallback simple.")
            html_output = self._fallback_html(data)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_output)
            logger.info(f"Informe HTML generado en: {output_path}")

        return html_output

    def generate_pdf(self, informe: InformeSiniestro, output_pdf: str) -> None:
        """
        Genera PDF usando WeasyPrint (si está instalado).
        """
        html_str = self.generate_report(informe)
        try:
            from weasyprint import HTML, CSS  # type: ignore
            base_css = CSS(string="""
                @page { size: A4; margin: 18mm 14mm; }
                body { font-family: Arial, Helvetica, sans-serif; font-size: 10pt; color: #333; }
                h1 { color: #2c3e50; font-size: 18pt; margin: 0 0 12px 0; }
                h2 { color: #2c3e50; font-size: 14pt; margin: 16px 0 8px 0; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 9pt; }
                th { background: #34495e; color: #fff; text-align: left; padding: 6px 8px; }
                td { border-bottom: 1px solid #e5e5e5; padding: 5px 8px; }
            """)
            HTML(string=html_str).write_pdf(output_pdf, stylesheets=[base_css])
            logger.info(f"PDF generado en: {output_pdf}")
        except ImportError:
            logger.warning("WeasyPrint no instalado. Saltando PDF.")
        except Exception as e:
            logger.warning(f"Error generando PDF: {e}")

    # =========================
    # 🔄 NUEVOS Helpers para IntelligentFieldExtractor
    # =========================
    
    def _extract_field_intelligent(self, documents: List[Dict[str, Any]], field_name: str, ai_analysis: Dict[str, Any]) -> str:
        """
        🔄 NUEVO: Extrae campo con prioridad de specific_fields (IntelligentFieldExtractor)
        
        Orden de prioridad:
        1. specific_fields (campos extraídos por IntelligentFieldExtractor)
        2. key_value_pairs (si existe)
        3. AI analysis (como fallback)
        4. Valor por defecto
        """
        # 1. Buscar en specific_fields (IntelligentFieldExtractor)
        for doc in documents:
            sf = doc.get('specific_fields', {})
            if field_name in sf and sf[field_name]:
                value = str(sf[field_name])
                if value and value not in ['None', 'null', '']:
                    logger.debug(f"Campo '{field_name}' encontrado en specific_fields: {value}")
                    return value
        
        # 2. Buscar en key_value_pairs
        for doc in documents:
            kv = doc.get('key_value_pairs', {})
            if field_name in kv and kv[field_name]:
                value = str(kv[field_name])
                if value and value not in ['None', 'null', '']:
                    logger.debug(f"Campo '{field_name}' encontrado en key_value_pairs: {value}")
                    return value
        
        # 3. Buscar con sinónimos en specific_fields
        synonyms = {
            'nombre_asegurado': ['asegurado', 'contratante', 'cliente', 'titular'],
            'numero_poliza': ['poliza', 'no_poliza', 'policy'],
            'vigencia_inicio': ['vigencia_desde', 'desde', 'inicio_vigencia'],
            'vigencia_fin': ['vigencia_hasta', 'hasta', 'fin_vigencia'],
            'monto_reclamacion': ['monto', 'total', 'importe', 'cantidad'],
            'fecha_siniestro': ['fecha_ocurrencia', 'fecha_evento'],
            'lugar_hechos': ['ubicacion', 'lugar', 'direccion_siniestro'],
        }
        
        if field_name in synonyms:
            for synonym in synonyms[field_name]:
                for doc in documents:
                    sf = doc.get('specific_fields', {})
                    if synonym in sf and sf[synonym]:
                        value = str(sf[synonym])
                        if value and value not in ['None', 'null', '']:
                            logger.debug(f"Campo '{field_name}' encontrado como '{synonym}': {value}")
                            return value
        
        # 4. Fallback a AI analysis
        if field_name in ai_analysis:
            value = str(ai_analysis[field_name])
            if value and value not in ['None', 'null', '']:
                logger.debug(f"Campo '{field_name}' tomado de AI analysis: {value}")
                return value
        
        # 5. Valores por defecto según el campo
        defaults = {
            'nombre_asegurado': 'NO IDENTIFICADO',
            'numero_poliza': 'SIN_POLIZA',
            'vigencia_inicio': '',
            'vigencia_fin': '',
            'domicilio_poliza': 'NO ESPECIFICADO',
            'bien_reclamado': 'MERCANCÍA DIVERSA',
            'monto_reclamacion': '0.00',
            'tipo_siniestro': 'NO ESPECIFICADO',
            'fecha_siniestro': '',
            'fecha_reclamacion': '',
            'lugar_hechos': 'NO ESPECIFICADO',
        }
        
        return defaults.get(field_name, 'NO ESPECIFICADO')
    
    def _format_amount(self, amount_str: str) -> str:
        """
        🔄 MEJORADO: Formatea montos con mejor manejo
        """
        if not amount_str or amount_str in ['NO ESPECIFICADO', '0.00', '']:
            return '0.00'
        
        # Si ya está formateado correctamente
        if re.match(r'^\d{1,3}(,\d{3})*(\.\d{2})?$', amount_str):
            return amount_str
        
        # Limpiar y convertir
        try:
            # Quitar símbolos de moneda y espacios
            clean = amount_str.replace('$', '').replace('MXN', '').replace('MN', '').strip()
            # Si tiene comas como separador de miles, quitarlas
            if ',' in clean and '.' in clean:
                clean = clean.replace(',', '')
            elif ',' in clean and clean.count(',') == 1:
                # Podría ser decimal con coma
                clean = clean.replace(',', '.')
            else:
                clean = clean.replace(',', '')
            
            # Convertir a float
            amount = float(clean)
            # Formatear con comas de miles y 2 decimales
            return f"{amount:,.2f}"
        except:
            return '0.00'
    
    def _format_date(self, date_str: str) -> str:
        """
        🔄 MEJORADO: Formatea fechas con mejor manejo
        """
        if not date_str or date_str in ['NO ESPECIFICADO', '']:
            return ''
        
        # Si ya está en formato ISO
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str
        
        # Intentar parsear diferentes formatos
        s = date_str.strip()
        
        # Mapeo de meses en español
        month_map = {
            'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
            'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
            'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12',
            'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'ago': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12',
        }
        
        # Reemplazar nombres de meses
        for month_name, month_num in month_map.items():
            s = s.lower().replace(month_name, month_num)
        
        # Formatos comunes
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            except:
                pass
        
        # Si no se pudo parsear, buscar cualquier fecha
        m = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', s)
        if m:
            return self._format_date(m[0])  # Recursivo con la primera fecha encontrada
        
        return s  # Retornar como está si no se puede formatear

    # =========================
    # Helpers privados existentes (sin cambios)
    # =========================
    def _dataclass_to_dict(self, obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: self._dataclass_to_dict(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [self._dataclass_to_dict(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._dataclass_to_dict(v) for k, v in obj.items()}
        return obj

    def _fallback_html(self, ctx: Dict[str, Any]) -> str:
        def esc(s: Any) -> str:
            try:
                return str(s)
            except Exception:
                return ""
        header = f"<h1>Informe Siniestro {esc(ctx.get('numero_siniestro'))}</h1>"
        meta = f"""
        <p><strong>Asegurado:</strong> {esc(ctx.get('nombre_asegurado'))} &nbsp;|&nbsp;
           <strong>Póliza:</strong> {esc(ctx.get('numero_poliza'))}</p>
        <p><strong>Tipo:</strong> {esc(ctx.get('tipo_siniestro'))} &nbsp;|&nbsp;
           <strong>Ocurrió:</strong> {esc(ctx.get('fecha_ocurrencia'))}</p>
        """
        return f"<!doctype html><html><head><meta charset='utf-8'><title>Informe</title></head><body>{header}{meta}<pre>{esc(json.dumps(ctx, ensure_ascii=False, indent=2))}</pre></body></html>"

    def _select_metodos(self, ai_analysis: Dict) -> List[Dict[str, str]]:
        methods = list(self.metodos_base)
        s = json.dumps(ai_analysis, ensure_ascii=False).lower()
        if "gps" in s:
            methods.append({"nombre": "Análisis de Telemetría GPS", "descripcion": "Revisión detallada de posicionamiento/telemetría."})
        if "repuve" in s:
            methods.append({"nombre": "Consulta REPUVE", "descripcion": "Verificación en Registro Público Vehicular."})
        return methods

    def _process_documents_enhanced(self, documents: List[Dict[str, Any]], ai_analysis: Dict[str, Any]) -> List[DocumentoAnalizado]:
        """
        🔄 MEJORADO: Procesa documentos mostrando campos extraídos por IntelligentFieldExtractor
        """
        out: List[DocumentoAnalizado] = []
        
        for doc in documents:
            tipo = doc.get("document_type", "otro")
            meta = doc.get("ocr_metadata", {}) or {}
            fname = meta.get("source_name") or meta.get("source_path") or f"{tipo}.pdf"
            
            # Contar campos extraídos
            sf = doc.get("specific_fields", {})
            kv = doc.get("key_value_pairs", {})
            total_fields = len(sf) + len(kv)
            
            desc = f"{fname}"
            if total_fields > 0:
                desc += f" ({total_fields} campos extraídos)"
            
            # Mostrar campos clave extraídos
            highlights: List[str] = []
            
            if tipo in {"factura", "factura_compra"}:
                for field in ["rfc", "total", "fecha", "numero_factura"]:
                    if field in sf and sf[field]:
                        highlights.append(f"{field.upper()}: {sf[field]}")
            
            elif tipo in {"poliza", "poliza_seguro"}:
                for field in ["numero_poliza", "nombre_asegurado", "vigencia_inicio"]:
                    if field in sf and sf[field]:
                        label = field.replace('_', ' ').title()
                        highlights.append(f"{label}: {sf[field]}")
            
            elif tipo == "denuncia":
                for field in ["fecha_siniestro", "lugar_hechos", "numero_denuncia"]:
                    if field in sf and sf[field]:
                        label = field.replace('_', ' ').title()
                        highlights.append(f"{label}: {sf[field]}")
            
            if highlights:
                desc += "\n• " + "\n• ".join(highlights[:3])
            
            doc_analizado = DocumentoAnalizado(
                tipo_documento=self._format_document_type(tipo),
                descripcion=desc,
                hallazgos=[],
                nivel_alerta=NivelAlerta.SUCCESS.value if total_fields > 3 else NivelAlerta.INFO.value,
                imagen=None,
                metadata=meta,
            )
            out.append(doc_analizado)
        
        return out

    def _build_inconsistencies(self, ai_data: Dict[str, Any]) -> List[Inconsistencia]:
        """
        Extrae inconsistencias del análisis de IA
        """
        inconsistencias_list = []
        raw_inconsistencies = ai_data.get("inconsistencies", [])
        
        if not isinstance(raw_inconsistencies, list):
            return []

        for inconsistency in raw_inconsistencies:
            if isinstance(inconsistency, dict):
                inconsistencias_list.append(Inconsistencia(
                    dato=inconsistency.get("field", "N/A"),
                    valor_a=str(inconsistency.get("value_a", "N/A")),
                    valor_b=str(inconsistency.get("value_b", "N/A")),
                    severidad=str(inconsistency.get("severity", "baja")).upper(),
                    documentos_afectados=inconsistency.get("affected_docs", [])
                ))
        
        return inconsistencias_list

    def _generate_analisis_turno(self, ai: Dict[str, Any]) -> str:
        return (
            "Se realizó un análisis documental exhaustivo utilizando extracción inteligente de campos "
            "y validación cruzada de información para determinar la procedencia del siniestro."
        )

    def _generate_planteamiento(self, ai: Dict[str, Any]) -> str:
        if ai.get("fraud_indicators"):
            return (
                f"La reclamación presenta {len(ai['fraud_indicators'])} indicadores de alerta. "
                "Se requiere análisis detallado de la documentación y verificación de hechos."
            )
        return (
            "Se procede con la validación estándar de la documentación presentada "
            "para determinar la procedencia del reclamo según los términos de la póliza."
        )

    def _generate_conclusion_enhanced(self, fraud_score: float, informe: InformeSiniestro, ai: Dict[str, Any]) -> Tuple[str, str, str]:
        has_critical = len([i for i in informe.inconsistencias if i.severidad.lower() in {"critica", "critical"}]) > 0

        if fraud_score > 0.8 or has_critical:
            return (
                "Existen elementos suficientes para considerar que la reclamación presenta irregularidades graves.",
                "RECLAMACIÓN CON INDICIOS DE FRAUDE",
                TipoConclusion.TENTATIVA.value,
            )
        if fraud_score > 0.5:
            return (
                "Se identifican inconsistencias que requieren investigación adicional antes de determinar procedencia.",
                "REQUIERE INVESTIGACIÓN ADICIONAL",
                TipoConclusion.INVESTIGACION.value,
            )
        return (
            "La documentación presentada cumple con los requisitos básicos para proceder con la reclamación.",
            "PROCEDENTE SALVO MEJOR OPINIÓN",
            TipoConclusion.PROCEDENTE.value,
        )

    def _generate_considerations_enhanced(self, informe: InformeSiniestro, ai: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        
        # Si hay muchos campos extraídos exitosamente
        docs_with_fields = [d for d in informe.documentos_analizados if "campos extraídos" in d.descripcion]
        if len(docs_with_fields) > 5:
            out.append({
                "titulo": "Documentación Completa",
                "descripcion": f"Se logró extraer información de {len(docs_with_fields)} documentos exitosamente.",
                "evidencias": [d.tipo_documento for d in docs_with_fields[:5]],
            })
        
        # Si hay inconsistencias críticas
        crit = [i for i in informe.inconsistencias if i.severidad.lower() in {"critica", "critical"}]
        if crit:
            out.append({
                "titulo": "Inconsistencias Detectadas",
                "descripcion": "Se encontraron discrepancias entre los documentos presentados.",
                "evidencias": [f"{i.dato}: {i.valor_a} vs {i.valor_b}" for i in crit[:3]],
            })
        
        return out

    def _extract_route_analysis(self, ai: Dict[str, Any]) -> Optional[AnalisisRuta]:
        r = ai.get("route_analysis") or {}
        if not r:
            return None
        return AnalisisRuta(
            origen_declarado=r.get("declared_origin", ""),
            verificacion_gps_origen=r.get("gps_origin_verification", ""),
            lugar_declarado=r.get("declared_incident_location", ""),
            verificacion_gps_hechos=r.get("gps_incident_verification", ""),
            analisis_evento=r.get("event_analysis", ""),
            destino_declarado=r.get("declared_destination", ""),
            verificacion_trayectoria=r.get("trajectory_verification", ""),
            inconsistencias_ruta=list(r.get("route_inconsistencies", [])),
        )

    def _format_document_type(self, tipo: str) -> str:
        mappings = {
            "carta_reclamacion": "CARTA DE RECLAMACIÓN",
            "carta_respuesta": "CARTA RESPUESTA DEL TRANSPORTISTA",
            "carpeta_investigacion": "CARPETA DE INVESTIGACIÓN",
            "tarjeta_circulacion": "TARJETA DE CIRCULACIÓN",
            "factura_compra": "FACTURA DE COMPRA",
            "factura": "FACTURA",
            "bitacora_viaje": "BITÁCORA DE VIAJE",
            "reporte_gps": "REPORTE GPS",
            "bitacora_gps": "BITÁCORA GPS",
            "denuncia": "DENUNCIA",
            "carta_porte": "CARTA PORTE",
            "poliza": "PÓLIZA DE SEGURO",
            "poliza_seguro": "PÓLIZA DE SEGURO",
        }
        return mappings.get(tipo, tipo.upper().replace("_", " "))