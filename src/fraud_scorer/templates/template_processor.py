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

        # ⚠️ Pedido explícito: usar un número fijo por ahora
        numero_siniestro_fijo = "1"

        informe = InformeSiniestro(
            numero_siniestro=numero_siniestro_fijo,
            nombre_asegurado=self._extract_nombre_asegurado(documents, ai_analysis),
            numero_poliza=self._extract_numero_poliza(documents, ai_analysis),
            vigencia_desde=self._extract_vigencia(documents, ai_analysis, which="inicio"),
            vigencia_hasta=self._extract_vigencia(documents, ai_analysis, which="fin"),
            domicilio_poliza=self._extract_address(documents, ai_analysis),
            bien_reclamado=self._extract_bien_reclamado(documents, ai_analysis),
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

    # --------- Normalización / utilidades de regex ---------
    def _norm(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return s

    def _all_text(self, documents: List[Dict]) -> str:
        parts: List[str] = []
        for d in documents:
            t = d.get("raw_text") or ""
            if t:
                parts.append(str(t))
        return "\n".join(parts)

    def _search_in_text(self, documents: List[Dict], patterns: List[re.Pattern]) -> Optional[str]:
        txt = self._all_text(documents)
        if not txt:
            return None
        txt_norm = "".join(c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn")
        for pat in patterns:
            m = pat.search(txt)
            if m:
                return (m.group(1) or "").strip()
            m2 = pat.search(txt_norm)
            if m2:
                return (m2.group(1) or "").strip()
        return None

    def _line_search(self, documents: List[Dict], pattern: re.Pattern) -> Optional[str]:
        txt = self._all_text(documents)
        if not txt:
            return None
        for line in txt.splitlines():
            m = pattern.search(line)
            if m:
                return (m.group(1) or "").strip()
        # prueba con versión sin acentos
        txt_norm = "".join(c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn")
        for line in txt_norm.splitlines():
            m = pattern.search(line)
            if m:
                return (m.group(1) or "").strip()
        return None

    def _kv_lookup(self, documents: List[Dict], targets: List[str]) -> Optional[str]:
        norm_targets = [self._norm(t) for t in targets]
        for d in documents:
            kv = d.get("key_value_pairs") or {}
            for k, v in kv.items():
                nk = self._norm(k)
                if any(t in nk for t in norm_targets):
                    if isinstance(v, (list, tuple)) and v:
                        return str(v[0])
                    if isinstance(v, dict) and v:
                        return str(v.get("value") or next(iter(v.values()), ""))
                    if v not in (None, ""):
                        return str(v)
        return None

    # --------- Extractores de alto nivel ---------
    def _extract_nombre_asegurado(self, documents: List[Dict], ai: Dict) -> str:
        # SF
        for d in documents:
            sf = d.get("specific_fields") or {}
            v = sf.get("nombre_asegurado")
            if v:
                return str(v)
        # KV
        hit = self._kv_lookup(documents, ["asegurado", "insured", "insured name", "nombre del asegurado"])
        if hit:
            return hit
        # Texto (regex)
        pats = [
            re.compile(r"(?:asegurado|insured(?:\s+name)?)\s*[:#]?\s*([A-ZÁÉÍÓÚÑ0-9][^\n\r]{3,80})", re.IGNORECASE),
        ]
        hit = self._line_search(documents, pats[0])
        if hit:
            return hit
        # IA
        return str(ai.get("insured_name") or ai.get("nombre_asegurado") or "NO IDENTIFICADO")

    def _extract_numero_poliza(self, documents: List[Dict], ai: Dict) -> str:
        # SF
        for d in documents:
            sf = d.get("specific_fields") or {}
            v = sf.get("numero_poliza")
            if v:
                return str(v)
        # KV
        hit = self._kv_lookup(documents, ["numero de poliza", "no de poliza", "poliza", "policy number", "num poliza", "no. póliza"])
        if hit:
            return hit
        # Texto (regex)
        pats = [
            re.compile(r"(?:n[uú]mero\s*de\s*p[óo]liza|no\.?\s*p[óo]liza|p[óo]liza|policy\s*number)\s*[:#]?\s*([A-Z0-9\-\/]{6,})", re.IGNORECASE),
        ]
        hit = self._search_in_text(documents, pats)
        if hit:
            return hit
        return str(ai.get("numero_poliza") or "SIN_POLIZA")

    def _extract_bien_reclamado(self, documents: List[Dict], ai: Dict) -> str:
        # SF
        for d in documents:
            sf = d.get("specific_fields") or {}
            v = sf.get("bien_reclamado")
            if v:
                return str(v)
        # KV
        hit = self._kv_lookup(documents, ["bien reclamado", "mercancia", "mercancía", "description of goods", "goods", "descripcion de mercancia"])
        if hit:
            return hit
        # Texto (regex)
        pat = re.compile(r"(?:bien\s*reclamado|mercanc[ií]a|description\s*of\s*goods)\s*[:#]?\s*([^\n\r]{3,120})", re.IGNORECASE)
        hit = self._line_search(documents, pat)
        if hit:
            return hit
        return str(ai.get("bien_reclamado") or "MERCANCÍA DIVERSA")

    def _extract_amount(self, documents: List[Dict], ai: Dict) -> str:
        amounts: List[float] = []

        # entities
        for d in documents:
            ents = d.get("entities") or []
            for e in ents:
                t = (e.get("type") or "").lower()
                if t in {"moneda", "currency", "amount", "monto"}:
                    raw = str(e.get("value") or e.get("text") or "").strip()
                    val = self._try_parse_amount(raw)
                    if val is not None:
                        amounts.append(val)

        # kv
        for d in documents:
            kv = d.get("key_value_pairs") or {}
            for k, v in kv.items():
                if any(w in k.lower() for w in ("monto", "importe", "total", "claim_amount")):
                    val = self._try_parse_amount(str(v))
                    if val is not None:
                        amounts.append(val)

        # texto (regex robusta)
        txt = self._all_text(documents)
        if txt:
            for m in re.findall(r"(?:monto|importe|total)\s*(?:reclamad[oa]|reclamaci[óo]n)?\s*[:$]?\s*\$?\s*([\d\.,]{3,})", txt, flags=re.IGNORECASE):
                val = self._try_parse_amount(m)
                if val is not None:
                    amounts.append(val)

        if amounts:
            return f"{max(amounts):,.2f}"

        ai_val = ai.get("claim_amount")
        if isinstance(ai_val, (int, float)):
            return f"{float(ai_val):,.2f}"
        if isinstance(ai_val, str):
            val = self._try_parse_amount(ai_val)
            if val is not None:
                return f"{val:,.2f}"
        return "0.00"

    def _try_parse_amount(self, s: str) -> Optional[float]:
        s = s.replace(",", "").replace("$", "").replace("MXN", "").replace("mxn", "").replace("usd", "").strip()
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
        # keywords en raw_text
        for d in documents:
            text = (d.get("raw_text") or "").lower()
            for k, v in mapping.items():
                if k in text:
                    return v
        return str(ai.get("claim_type") or "NO ESPECIFICADO")

    def _extract_vigencia(self, documents: List[Dict], ai: Dict, which: str) -> str:
        """
        Extrae vigencia inicio/fin buscando:
        - specific_fields (vigencia_inicio/vigencia_fin)
        - KV con sinónimos típicos
        - Texto: "Vigencia ... del DD/MM/AAAA al DD/MM/AAAA"
        """
        field = "vigencia_inicio" if which == "inicio" else "vigencia_fin"

        # 1) specific_fields
        for d in documents:
            sf = d.get("specific_fields") or {}
            if field in sf and sf[field]:
                return self._format_date(str(sf[field]))

        # 2) KV con sinónimos
        synonyms = {
            "vigencia_inicio": ["vigencia desde", "inicio de vigencia", "del", "desde"],
            "vigencia_fin": ["vigencia hasta", "fin de vigencia", "al", "hasta"],
        }
        hit = self._kv_lookup(documents, synonyms[field])
        if hit:
            return self._format_date(hit)

        # 3) Texto: buscar rango del ... al ...
        txt = self._all_text(documents)
        if txt:
            m = re.search(
                r"vigencia[^:\n\r]*?(?:del|desde)\s*([0-3]?\d[\/\-][01]?\d[\/\-]\d{2,4}).{0,40}?(?:al|hasta)\s*([0-3]?\d[\/\-][01]?\d[\/\-]\d{2,4})",
                txt,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m:
                start, end = m.group(1), m.group(2)
                return self._format_date(start if which == "inicio" else end)

        # 4) IA
        v = ai.get(field, "")
        return self._format_date(str(v)) if v else ""

    def _extract_date_field(self, documents, field, ai) -> str:
        date_synonyms = {
            "fecha_siniestro": ["fecha del siniestro", "fecha de ocurrencia", "occurrence date"],
            "fecha_reclamacion": ["fecha de reclamacion", "fecha de reclamo", "claim date"],
        }
        # 1) SF
        for d in documents:
            sf = d.get("specific_fields") or {}
            if field in sf and sf[field]:
                return self._format_date(str(sf[field]))
        # 2) KV
        if field in date_synonyms:
            hit = self._kv_lookup(documents, date_synonyms[field])
            if hit:
                return self._format_date(hit)
        # 3) Texto
        label_map = {
            "fecha_siniestro": r"(?:fecha\s*(?:del\s*)?siniestro|fecha\s*de\s*ocurrencia)\s*[:#]?\s*([0-3]?\d[\/\-][01]?\d[\/\-]\d{2,4})",
            "fecha_reclamacion": r"(?:fecha\s*de\s*reclamaci[óo]n|claim\s*date)\s*[:#]?\s*([0-3]?\d[\/\-][01]?\d[\/\-]\d{2,4})",
        }
        pat = re.compile(label_map.get(field, r""), re.IGNORECASE)
        hit = self._line_search(documents, pat) if label_map.get(field) else None
        if hit:
            return self._format_date(hit)
        # 4) IA
        v = ai.get(field, "")
        return self._format_date(str(v)) if v else ""

    def _extract_address(self, documents: List[Dict], ai: Dict) -> str:
        # entities
        for d in documents:
            ents = d.get("entities") or []
            for e in ents:
                if str(e.get("type", "")).lower() in {"address", "domicilio"}:
                    return str(e.get("value") or e.get("text") or "")
        # KV/texto
        pat = re.compile(r"(?:domicilio(?:\s+de\s+la\s+p[óo]liza)?)\s*[:#]?\s*([^\n\r]{6,140})", re.IGNORECASE)
        hit = self._line_search(documents, pat)
        if hit:
            return hit
        return str(ai.get("insured_address") or "NO ESPECIFICADO")

    def _extract_location(self, documents: List[Dict], ai: Dict) -> str:
        # entities
        for d in documents:
            ents = d.get("entities") or []
            for e in ents:
                if str(e.get("type", "")).lower() in {"location", "lugar"}:
                    return str(e.get("value") or e.get("text") or "")
        # texto
        pat = re.compile(r"(?:lugar\s*(?:de\s*)?los?\s*hechos|ubicaci[óo]n\s*del\s*incidente)\s*[:#]?\s*([^\n\r]{3,120})", re.IGNORECASE)
        hit = self._line_search(documents, pat)
        if hit:
            return hit
        return str(ai.get("incident_location") or "NO ESPECIFICADO")

    def _format_date(self, date_str: str) -> str:
        s = (date_str or "").strip()
        if not s:
            return ""
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            except Exception:
                pass
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
            meta = d.get("ocr_metadata", {}) or {}
            fname = meta.get("source_name") or meta.get("source_path") or f"{tipo}.pdf"
            desc = f"{fname}"

            # Mini-resumen por tipo (si hay datos)
            kv = d.get("key_value_pairs") or {}
            sf = d.get("specific_fields") or {}
            highlights: List[str] = []
            if tipo in {"factura", "factura_compra"}:
                for k in ("rfc", "total", "fecha", "subtotal"):
                    val = sf.get(k) or kv.get(k)
                    if val:
                        highlights.append(f"{k.upper()}: {val}")
            elif tipo in {"poliza", "poliza_seguro"}:
                for k in ("numero_poliza", "vigencia_inicio", "vigencia_fin"):
                    val = sf.get(k) or kv.get(k)
                    if val:
                        highlights.append(f"{k.replace('_',' ').title()}: {val}")

            if highlights:
                desc += " · " + " | ".join(highlights[:3])

            doc = DocumentoAnalizado(
                tipo_documento=self._format_document_type(tipo),
                descripcion=desc,
                hallazgos=[],
                nivel_alerta=NivelAlerta.INFO.value,
                imagen=None,
                metadata=meta,
            )
            out.append(doc)
        return out

    def _build_inconsistencies(self, ai: Dict[str, Any]) -> List[Inconsistencia]:
        res: List[Inconsistencia] = []
        for inc in ai.get("inconsistencies") or []:
            values = inc.get("values")
            a = b = ""
            if isinstance(values, (list, tuple)) and values:
                a = str(values[0] or "")
                b = str(values[1] or "") if len(values) > 1 else ""
            a = a or str(inc.get("value_a", ""))
            b = b or str(inc.get("value_b", ""))

            res.append(Inconsistencia(
                dato=str(inc.get("field", "Campo")),
                valor_a=a,
                valor_b=b,
                severidad=str(inc.get("severity", "media")),
                documentos_afectados=list(inc.get("affected_documents") or inc.get("docs") or []),
            ))
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
