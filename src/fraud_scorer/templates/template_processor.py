"""
Sistema de Llenado Automático de Plantillas para Informes de Siniestros
(versión unificada y compatible con run_report.py / replay_case.py)
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
    Diseñado para trabajar con la estructura que produce `build_docs_for_template_*`:
    - Cada documento es un dict con:
      document_type, raw_text, entities (lista), key_value_pairs, specific_fields, ocr_metadata
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
        Extrae campos del caso a partir de la lista normalizada de documentos + análisis IA.
        `documents`: items con [document_type, raw_text, entities(list), key_value_pairs, specific_fields, ocr_metadata]
        """
        logger.info("Extrayendo información desde documentos normalizados + AI")

        informe = InformeSiniestro(
            numero_siniestro=self._extract_from_all_docs(documents, "numero_siniestro", ai_analysis, fallback="SIN_NUMERO"),
            nombre_asegurado=self._extract_from_all_docs(documents, "nombre_asegurado", ai_analysis, fallback="NO IDENTIFICADO"),
            numero_poliza=self._extract_from_all_docs(documents, "numero_poliza", ai_analysis, fallback="SIN_POLIZA"),
            vigencia_desde=self._extract_date_field(documents, "vigencia_inicio", ai_analysis),
            vigencia_hasta=self._extract_date_field(documents, "vigencia_fin", ai_analysis),
            domicilio_poliza=self._extract_address(documents, ai_analysis),
            bien_reclamado=self._extract_from_all_docs(documents, "bien_reclamado", ai_analysis, fallback="MERCANCÍA DIVERSA"),
            monto_reclamacion=self._extract_amount(documents, ai_analysis),
            tipo_siniestro=self._extract_claim_type(documents, ai_analysis),
            fecha_ocurrencia=self._extract_date_field(documents, "fecha_siniestro", ai_analysis),
            fecha_reclamacion=self._extract_date_field(documents, "fecha_reclamacion", ai_analysis),
            lugar_hechos=self._extract_location(documents, ai_analysis),
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
    # Helpers privados
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
        # Fallback hiper simple para no romper si no hay plantilla
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

    # --------- Extractores de alto nivel (usando documents normalizados) ---------
    def _extract_from_all_docs(self, documents: List[Dict], field: str, ai: Dict, fallback: str = "NO ESPECIFICADO") -> str:
        # 1) specific_fields
        for d in documents:
            sf = d.get("specific_fields") or {}
            if field in sf and sf[field]:
                return str(sf[field])

        # 2) key_value_pairs (por coincidencia parcial en la llave)
        for d in documents:
            kv = d.get("key_value_pairs") or {}
            for k, v in kv.items():
                if field.lower() in k.lower() and v:
                    return str(v)

        # 3) entities (lista de dicts con {type, value} o similar)
        for d in documents:
            ents = d.get("entities") or []
            for e in ents:
                if isinstance(e, dict) and str(e.get("type", "")).lower() == field.lower():
                    val = e.get("value") or e.get("text")
                    if val:
                        return str(val)

        # 4) IA
        return str(ai.get(field, fallback))

    def _extract_amount(self, documents: List[Dict], ai: Dict) -> str:
        amounts: List[float] = []

        # entities con amounts
        for d in documents:
            ents = d.get("entities") or []
            for e in ents:
                t = (e.get("type") or "").lower()
                if t in {"moneda", "currency", "amount", "monto"}:
                    raw = str(e.get("value") or e.get("text") or "").strip()
                    val = self._try_parse_amount(raw)
                    if val is not None:
                        amounts.append(val)

        # kv (por si viene como "monto", "importe", etc.)
        for d in documents:
            kv = d.get("key_value_pairs") or {}
            for k, v in kv.items():
                if any(w in k.lower() for w in ("monto", "importe", "total", "claim_amount")):
                    val = self._try_parse_amount(str(v))
                    if val is not None:
                        amounts.append(val)

        if amounts:
            return f"{max(amounts):,.2f}"

        # IA
        ai_val = ai.get("claim_amount")
        if isinstance(ai_val, (int, float)):
            return f"{float(ai_val):,.2f}"
        if isinstance(ai_val, str):
            val = self._try_parse_amount(ai_val)
            if val is not None:
                return f"{val:,.2f}"
        return "0.00"

    def _try_parse_amount(self, s: str) -> Optional[float]:
        s = s.replace(",", "").replace("$", "").replace("MXN", "").replace("usd", "").strip()
        m = re.findall(r"[-+]?\d*\.?\d+", s)
        if not m:
            return None
        try:
            return float(m[0])
        except Exception:
            return None

    def _extract_claim_type(self, documents: List[Dict], ai: Dict) -> str:
        mapping = {
            "robo": "ROBO",
            "colision": "COLISIÓN",
            "colisión": "COLISIÓN",
            "incendio": "INCENDIO",
            "daño": "DAÑOS",
            "daños": "DAÑOS",
            "bulto": "ROBO DE BULTO POR ENTERO",
            "mercancia": "ROBO DE MERCANCÍA",
            "mercancía": "ROBO DE MERCANCÍA",
        }
        # buscar keywords en raw_text
        for d in documents:
            text = (d.get("raw_text") or "").lower()
            for k, v in mapping.items():
                if k in text:
                    return v
        return ai.get("claim_type", "NO ESPECIFICADO")

    def _extract_date_field(self, documents: List[Dict], field: str, ai: Dict) -> str:
        # specific_fields
        for d in documents:
            sf = d.get("specific_fields") or {}
            if field in sf and sf[field]:
                return self._format_date(str(sf[field]))
        # IA
        v = ai.get(field, "")
        return self._format_date(str(v)) if v else ""

    def _extract_address(self, documents: List[Dict], ai: Dict) -> str:
        for d in documents:
            ents = d.get("entities") or []
            for e in ents:
                if str(e.get("type", "")).lower() in {"address", "domicilio"}:
                    return str(e.get("value") or e.get("text") or "")
        return ai.get("insured_address", "NO ESPECIFICADO")

    def _extract_location(self, documents: List[Dict], ai: Dict) -> str:
        for d in documents:
            ents = d.get("entities") or []
            for e in ents:
                if str(e.get("type", "")).lower() in {"location", "lugar"}:
                    return str(e.get("value") or e.get("text") or "")
        return ai.get("incident_location", "NO ESPECIFICADO")

    def _format_date(self, date_str: str) -> str:
        s = (date_str or "").strip()
        if not s:
            return ""
        # intenta DD/MM/AAAA, DD-MM-AAAA, AAAA-MM-DD, y variantes cortas
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            except Exception:
                pass
        # búsqueda básica
        m = re.findall(r"\d{2,4}[-/]\d{1,2}[-/]\d{2,4}", s)
        return m[0] if m else s

    def _select_metodos(self, ai_analysis: Dict) -> List[Dict[str, str]]:
        methods = list(self.metodos_base)
        s = json.dumps(ai_analysis, ensure_ascii=False).lower()
        if "gps" in s:
            methods.append({"nombre": "Análisis de Telemetría GPS", "descripcion": "Revisión detallada de posicionamiento/telemetría."})
        if "repuve" in s:
            methods.append({"nombre": "Consulta REPUVE", "descripcion": "Verificación en Registro Público Vehicular."})
        return methods

    def _process_documents_enhanced(self, documents: List[Dict[str, Any]], ai_analysis: Dict[str, Any]) -> List[DocumentoAnalizado]:
        out: List[DocumentoAnalizado] = []
        for d in documents:
            tipo = d.get("document_type", "otro")
            desc = d.get("ocr_metadata", {}).get("source_name") or f"Documento tipo {tipo}"
            hallazgos: List[str] = []
            nivel = NivelAlerta.INFO.value

            doc = DocumentoAnalizado(
                tipo_documento=self._format_document_type(tipo),
                descripcion=desc,
                hallazgos=hallazgos,
                nivel_alerta=nivel,
                imagen=None,
                metadata=d.get("ocr_metadata", {}),
            )
            out.append(doc)
        return out

    def _build_inconsistencies(self, ai: Dict[str, Any]) -> List[Inconsistencia]:
        res: List[Inconsistencia] = []
        incs = ai.get("inconsistencies") or []
        for inc in incs:
            res.append(
                Inconsistencia(
                    dato=str(inc.get("field", "Campo")),
                    valor_a=str((inc.get("values") or ["", ""])[0]),
                    valor_b=str((inc.get("values") or ["", ""])[:2][-1]),
                    severidad=str(inc.get("severity", "media")),
                    documentos_afectados=list(inc.get("affected_documents") or []),
                )
            )
        return res

    def _generate_analisis_turno(self, ai: Dict[str, Any]) -> str:
        return (
            "Se realizó un análisis documental y validación técnica basada en criterios "
            "forenses, jurídicos y de telemetría para recomendar una postura sustentada."
        )

    def _generate_planteamiento(self, ai: Dict[str, Any]) -> str:
        if ai.get("fraud_indicators"):
            return (
                "La reclamación presenta documentación con inconsistencias relevantes. "
                f"Se identifican {len(ai['fraud_indicators'])} indicadores potenciales."
            )
        return (
            "Se requiere validación exhaustiva de la documentación presentada y verificación "
            "de los hechos para determinar la procedencia del reclamo."
        )

    def _generate_conclusion_enhanced(self, fraud_score: float, informe: InformeSiniestro, ai: Dict[str, Any]) -> Tuple[str, str, str]:
        has_fake_docs = any("APÓCRIFO" in " ".join(d.hallazgos) for d in informe.documentos_analizados)
        critical = [i for i in informe.inconsistencias if i.severidad.lower() in {"critica", "critical"}]

        if has_fake_docs or fraud_score > 0.8:
            return (
                "La reclamación está viciada por la presentación de documentación apócrifa o indicios graves.",
                "CON TENTATIVA DE FRAUDE",
                TipoConclusion.TENTATIVA.value,
            )
        if fraud_score > 0.5 or critical:
            return (
                "Se identifican inconsistencias significativas que requieren investigación adicional.",
                "REQUIERE INVESTIGACIÓN ADICIONAL",
                TipoConclusion.INVESTIGACION.value,
            )
        return (
            "No se observan irregularidades significativas tras el análisis efectuado.",
            "PROCEDENTE SALVO MEJOR OPINIÓN",
            TipoConclusion.PROCEDENTE.value,
        )

    def _generate_considerations_enhanced(self, informe: InformeSiniestro, ai: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        apocrifos = [d for d in informe.documentos_analizados if "APÓCRIFO" in " ".join(d.hallazgos)]
        if apocrifos:
            out.append({
                "titulo": "Falsificación Comprobada",
                "descripcion": "Se confirmó documentación apócrifa mediante verificación directa.",
                "evidencias": [f"{d.tipo_documento}" for d in apocrifos],
            })
        crit = [i for i in informe.inconsistencias if i.severidad.lower() in {"critica", "critical"}]
        if crit:
            out.append({
                "titulo": "Inconsistencias Críticas",
                "descripcion": "Contradicciones fundamentales que afectan la credibilidad del reclamo.",
                "evidencias": [f"{i.dato}: {i.valor_a} vs {i.valor_b}" for i in crit],
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
            "bitacora_viaje": "BITÁCORA DE VIAJE",
            "reporte_gps": "REPORTE GPS",
            "denuncia": "DENUNCIA",
            "carta_porte": "CARTA PORTE",
        }
        return mappings.get(tipo, tipo.upper().replace("_", " "))
