# src/fraud_scorer/ai_extractors/ai_report_generator.py

"""
AIReportGenerator: Genera el reporte final usando los datos consolidados
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel

from ..models.extraction import ConsolidatedExtraction
from fraud_scorer.templates.ai_report_generator import AIReportGenerator

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper de normalización (nuevo)
# -----------------------------------------------------------------------------
def _to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convierte Pydantic v1/v2 o cualquier objeto simple en dict seguro.
    Evita errores tipo "'ConsolidatedFields' object has no attribute 'get'".
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump()  # type: ignore[return-value]
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return obj.dict()  # type: ignore[return-value]
        except Exception:
            pass
    # Fallback genérico
    try:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {}

class AIReportGenerator:
    """
    Generador de reportes que toma los datos consolidados
    y produce el informe final HTML/PDF
    """
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Inicializa el generador con las plantillas
        """
        if template_dir is None:
            template_dir = Path(__file__).parent.parent / "templates"
        
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )
        
        logger.info(f"AIReportGenerator inicializado con plantillas de {template_dir}")
    
    def generate_report(
        self,
        consolidated_data: ConsolidatedExtraction,
        ai_analysis: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Genera el reporte HTML final
        
        Args:
            consolidated_data: Datos consolidados del caso
            ai_analysis: Análisis adicional de IA (fraud score, etc.)
            output_path: Ruta donde guardar el HTML
            
        Returns:
            HTML generado como string
        """
        # case_id robusto incluso si consolidated_data es Pydantic v1/v2
        try:
            logger.info(f"Generando reporte para caso {getattr(consolidated_data, 'case_id', 'SIN_CASE_ID')}")
        except Exception:
            logger.info("Generando reporte para caso (case_id no disponible)")

        # Preparar datos para la plantilla
        template_data = self._prepare_template_data(consolidated_data, ai_analysis)
        
        # Cargar y renderizar plantilla
        try:
            template = self.env.get_template("report_template.html")
            html_content = template.render(**template_data)
        except Exception as e:
            logger.error(f"Error renderizando plantilla: {e}")
            # Usar plantilla de fallback
            html_content = self._generate_fallback_html(template_data)
        
        # Guardar si se especifica path
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Reporte guardado en {output_path}")
        
        return html_content
    
    def generate_pdf(
        self,
        html_content: str,
        output_path: Path
    ) -> bool:
        """
        Genera PDF desde el HTML
        
        Args:
            html_content: Contenido HTML
            output_path: Ruta para el PDF
            
        Returns:
            True si se generó exitosamente
        """
        try:
            from weasyprint import HTML, CSS
            
            # CSS adicional para PDF
            pdf_css = CSS(string="""
                @page { 
                    size: A4; 
                    margin: 18mm 14mm;
                    @bottom-right {
                        content: "Página " counter(page) " de " counter(pages);
                    }
                }
                body { 
                    font-family: Arial, sans-serif;
                    font-size: 10pt;
                }
            """)
            
            HTML(string=html_content).write_pdf(
                output_path,
                stylesheets=[pdf_css]
            )
            
            logger.info(f"PDF generado en {output_path}")
            return True
            
        except ImportError:
            logger.warning("WeasyPrint no instalado, no se puede generar PDF")
            return False
        except Exception as e:
            logger.error(f"Error generando PDF: {e}")
            return False
    
    def _prepare_template_data(
        self,
        consolidated_data: ConsolidatedExtraction,
        ai_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Prepara los datos para la plantilla.
        Acepta tanto modelos Pydantic (ConsolidatedExtraction/ConsolidatedFields)
        como dicts, normalizándolos a dicts seguros.
        """
        # Normalizar el objeto principal (puede ser ConsolidatedExtraction o dict)
        cd_dict = _to_dict(consolidated_data)

        # case_id robusto
        case_id = cd_dict.get("case_id")
        if not case_id and hasattr(consolidated_data, "case_id"):
            case_id = getattr(consolidated_data, "case_id", None)

        # Normalizar consolidated_fields (puede ser ConsolidatedFields o dict)
        fields_obj = cd_dict.get("consolidated_fields")
        if fields_obj is None and hasattr(consolidated_data, "consolidated_fields"):
            fields_obj = getattr(consolidated_data, "consolidated_fields", None)
        fields = _to_dict(fields_obj)

        # Compatibilidad con dos esquemas:
        # - antiguo: vigencia_inicio / vigencia_fin
        # - nuevo:   vigencia (string "YYYY-MM-DD a YYYY-MM-DD" o similar)
        vig_desde = fields.get("vigencia_inicio")
        vig_hasta = fields.get("vigencia_fin")
        if (not vig_desde and not vig_hasta) and "vigencia" in fields and isinstance(fields["vigencia"], str):
            vig_str = fields["vigencia"]
            # split por ' a ' o ' - '
            if " a " in vig_str:
                partes = [p.strip() for p in vig_str.split(" a ", 1)]
            elif " - " in vig_str:
                partes = [p.strip() for p in vig_str.split(" - ", 1)]
            else:
                partes = [vig_str]
            if len(partes) == 2:
                vig_desde, vig_hasta = partes[0], partes[1]
            elif len(partes) == 1:
                vig_desde = partes[0]

        # Mapear campos consolidados a nombres de plantilla (con los mismos keys que ya usabas)
        template_data = {
            "numero_siniestro": fields.get("numero_siniestro", case_id),
            "nombre_asegurado": fields.get("nombre_asegurado", "NO ESPECIFICADO"),
            "numero_poliza": fields.get("numero_poliza", "NO ESPECIFICADO"),
            "vigencia_desde": self._format_date(vig_desde),
            "vigencia_hasta": self._format_date(vig_hasta),
            "domicilio_poliza": fields.get("domicilio_poliza", "NO ESPECIFICADO"),
            "bien_reclamado": fields.get("bien_reclamado", "NO ESPECIFICADO"),
            "monto_reclamacion": self._format_amount(fields.get("monto_reclamacion")),
            "tipo_siniestro": fields.get("tipo_siniestro", "NO ESPECIFICADO"),
            "fecha_ocurrencia": self._format_date(fields.get("fecha_ocurrencia")),
            "fecha_reclamacion": self._format_date(fields.get("fecha_reclamacion")),
            "lugar_hechos": fields.get("lugar_hechos", "NO ESPECIFICADO"),
            "ajuste": fields.get("ajuste", "NO ESPECIFICADO"),
            
            # Metadata
            "fecha_generacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "confidence_scores": cd_dict.get("confidence_scores", getattr(consolidated_data, "confidence_scores", {})),
            "consolidation_sources": cd_dict.get("consolidation_sources", getattr(consolidated_data, "consolidation_sources", {})),
        }
        
        # Agregar análisis de IA si existe (normalizado por si acaso)
        ai_analysis_dict = _to_dict(ai_analysis)
        if ai_analysis_dict:
            template_data.update({
                "fraud_score": ai_analysis_dict.get("fraud_score", 0),
                "risk_level": self._calculate_risk_level(ai_analysis_dict.get("fraud_score", 0) or 0),
                "inconsistencias": ai_analysis_dict.get("inconsistencies", []),
                "fraud_indicators": ai_analysis_dict.get("fraud_indicators", []),
                "validaciones_externas": ai_analysis_dict.get("external_validations", [])
            })
        
        # Agregar información de conflictos resueltos si viene
        conflicts = cd_dict.get("conflicts_resolved", getattr(consolidated_data, "conflicts_resolved", []))
        if conflicts:
            template_data["conflicts_resolved"] = conflicts
        
        return template_data
    
    def _format_date(self, date_value: Any) -> str:
        """Formatea una fecha para display"""
        if not date_value:
            return "NO ESPECIFICADO"
        
        if isinstance(date_value, str):
            # Si ya está en formato YYYY-MM-DD, convertir a DD/MM/YYYY
            if len(date_value) == 10 and date_value[4] == '-':
                parts = date_value.split('-')
                if len(parts) == 3:
                    return f"{parts[2]}/{parts[1]}/{parts[0]}"
        
        return str(date_value)
    
    def _format_amount(self, amount: Any) -> str:
        """Formatea un monto para display"""
        if amount in (None, "", 0, 0.0):
            return "0.00"
        
        try:
            if isinstance(amount, (int, float)):
                return f"${amount:,.2f}"
            # Si viene como string (p. ej., ya formateado), lo devolvemos tal cual
            return str(amount)
        except Exception:
            return "0.00"
    
    def _calculate_risk_level(self, fraud_score: float) -> str:
        """Calcula el nivel de riesgo basado en el score"""
        try:
            score = float(fraud_score)
        except Exception:
            score = 0.0
        if score < 0.3:
            return "BAJO"
        elif score < 0.6:
            return "MEDIO"
        else:
            return "ALTO"
    
    def _generate_fallback_html(self, data: Dict[str, Any]) -> str:
        """Genera HTML de fallback si falla la plantilla principal"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Reporte de Siniestro - {data.get('numero_siniestro', 'Sin número')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        .field {{ margin: 10px 0; }}
        .label {{ font-weight: bold; display: inline-block; width: 200px; }}
        .value {{ display: inline-block; }}
    </style>
</head>
<body>
    <h1>Reporte de Siniestro</h1>
    <div class="field">
        <span class="label">Número de Siniestro:</span>
        <span class="value">{data.get('numero_siniestro', 'NO ESPECIFICADO')}</span>
    </div>
    <div class="field">
        <span class="label">Asegurado:</span>
        <span class="value">{data.get('nombre_asegurado', 'NO ESPECIFICADO')}</span>
    </div>
    <div class="field">
        <span class="label">Póliza:</span>
        <span class="value">{data.get('numero_poliza', 'NO ESPECIFICADO')}</span>
    </div>
    <div class="field">
        <span class="label">Monto Reclamado:</span>
        <span class="value">{data.get('monto_reclamacion', '0.00')}</span>
    </div>
    <hr>
    <p><small>Generado: {data.get('fecha_generacion', '')}</small></p>
</body>
</html>
"""
        return html
