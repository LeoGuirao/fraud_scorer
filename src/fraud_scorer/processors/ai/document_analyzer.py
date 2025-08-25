# src/fraud_scorer/processors/ai/document_analyzer.py

from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import base64
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class AIDocumentAnalyzer:
    """
    Analizador de documentos usando AI que trabaja con:
    1. Datos estructurados del OCR
    2. Imágenes originales para detectar alteraciones
    3. Análisis contextual entre documentos
    4. Análisis global del caso (todos los docs) -> analyze_claim_documents
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        vision_model: Optional[str] = None
    ):
        """
        - api_key: permite inyectar la clave desde la UI o usar OPENAI_API_KEY del entorno.
        - model: modelo base para análisis de texto/JSON (por defecto: env OPENAI_MODEL o 'gpt-4o-mini').
        - temperature: control de creatividad del modelo.
        - vision_model: modelo para análisis de imagen (por defecto: env OPENAI_VISION_MODEL o 'gpt-4o'
                        si el modelo principal no es 4o/4o-mini).
        """
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY no está definido. Pásalo como parámetro (api_key) o configúralo en el entorno/.env."
            )
        self.client = AsyncOpenAI(api_key=key)

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(temperature)

        # Si el modelo principal no es de la familia 4o, usar 4o para visión por defecto.
        default_vision = self.model if "gpt-4o" in self.model else "gpt-4o"
        self.vision_model = vision_model or os.getenv("OPENAI_VISION_MODEL", default_vision)

        logger.info(
            f"AIDocumentAnalyzer inicializado | model={self.model} | "
            f"vision_model={self.vision_model} | temperature={self.temperature}"
        )
        
    async def analyze_document(
        self, 
        structured_data: Dict[str, Any],
        image_path: Optional[str] = None,
        context_documents: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analiza un documento usando los datos estructurados y opcionalmente la imagen
        """
        doc_type = structured_data.get('document_type', 'desconocido')
        
        # Análisis basado en datos estructurados
        content_analysis = await self._analyze_content(structured_data)
        
        # Análisis visual si hay imagen
        visual_analysis = None
        if image_path:
            visual_analysis = await self._analyze_visual_integrity(image_path, doc_type)
        
        # Análisis contextual si hay otros documentos
        contextual_analysis = None
        if context_documents:
            contextual_analysis = await self._analyze_context(
                structured_data, 
                context_documents
            )
        
        # Compilar análisis completo
        return {
            "document_type": doc_type,
            "analysis_timestamp": datetime.now().isoformat(),
            "content_analysis": content_analysis,
            "visual_analysis": visual_analysis,
            "contextual_analysis": contextual_analysis,
            "summary": await self._generate_summary(
                content_analysis, 
                visual_analysis, 
                contextual_analysis
            ),
            "report_points": self._extract_report_points(content_analysis),
            "alerts": self._compile_alerts(
                content_analysis, 
                visual_analysis, 
                contextual_analysis
            )
        }

    async def analyze_claim_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza el caso completo (lista de documentos ya estructurados para informe)
        y devuelve un JSON con:
          - fraud_score (0..1)
          - fraud_indicators: [{description, weight, doc_refs}]
          - inconsistencies: [{field, value_a, value_b, severity, affected_docs}]
          - external_validations: [{type, description, result, is_critical}]
          - route_analysis: {...}  (si aplica a transporte)
        """
        # Resumen compacto de documentos para el LLM
        compact_docs: List[Dict[str, Any]] = []
        for i, d in enumerate(documents):
            compact_docs.append({
                "index": i,
                "document_type": d.get("document_type", "otro"),
                "entities": d.get("entities", {}),
                "key_value_pairs": d.get("key_value_pairs", {}),
                "specific_fields": d.get("specific_fields", {}),
                "text_length": len(d.get("raw_text", "")),
            })

        prompt = (
            "Eres analista senior de siniestros. Con base en los documentos estructurados, "
            "devuelve un JSON con estas llaves:\n"
            "fraud_score (0..1),\n"
            "fraud_indicators: [{description, weight (0..1), doc_refs:[indices]}],\n"
            "inconsistencies: [{field, value_a, value_b, severity in [baja, media, alta, critica], affected_docs:[indices]}],\n"
            "external_validations: [{type, description, result, is_critical}],\n"
            "route_analysis: {declared_origin, gps_origin_verification, declared_incident_location, "
            "gps_incident_verification, event_analysis, declared_destination, trajectory_verification, "
            "route_inconsistencies:[]}\n\n"
            "Responde SOLO con JSON válido."
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Eres un analista experto en fraude de seguros."},
                {"role": "user", "content": prompt},
                {"role": "user", "content": json.dumps({"documents": compact_docs}, ensure_ascii=False)}
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_tokens=1200,
        )

        # Robustez en el parseo
        try:
            content = response.choices[0].message.content
            data = json.loads(content) if content else {}
        except Exception:
            logger.exception("No se pudo parsear la respuesta del análisis de caso. Devolviendo valores por defecto.")
            data = {}

        # Defaults seguros
        return {
            "fraud_score": float(data.get("fraud_score", 0.0)) if isinstance(data.get("fraud_score", 0.0), (int, float)) else 0.0,
            "fraud_indicators": data.get("fraud_indicators", []) or [],
            "inconsistencies": data.get("inconsistencies", []) or [],
            "external_validations": data.get("external_validations", []) or [],
            "route_analysis": data.get("route_analysis", {}) or {},
        }
    
    async def _analyze_content(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el contenido estructurado del documento"""
        # Evitar reventar si faltan keys
        safe_doc_type = structured_data.get('document_type', 'desconocido')
        safe_payload = {
            k: structured_data.get(k) for k in [
                "document_type", "entities", "key_value_pairs", "specific_fields",
                "raw_text", "tables", "ocr_metadata"
            ] if k in structured_data
        }

        prompt = (
            f"Analiza el siguiente documento tipo '{safe_doc_type}' y proporciona un análisis detallado.\n\n"
            f"Datos estructurados:\n{json.dumps(safe_payload, indent=2, ensure_ascii=False)}\n\n"
            "Por favor proporciona en JSON:\n"
            "  resumen, consistencia, faltantes, anomalias (lista), verificacion_externa (lista)\n"
        )
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": "Eres un analista experto en documentos de seguros y detección de fraude."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature
        )

        try:
            return json.loads(response.choices[0].message.content)
        except Exception:
            logger.exception("No se pudo parsear _analyze_content; devolviendo estructura vacía.")
            return {
                "resumen": "",
                "consistencia": "",
                "faltantes": [],
                "anomalias": [],
                "verificacion_externa": []
            }
    
    async def _analyze_visual_integrity(
        self, 
        image_path: str, 
        doc_type: str
    ) -> Dict[str, Any]:
        """Analiza la imagen para detectar alteraciones o anomalías visuales"""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.exception(f"No se pudo leer la imagen para análisis visual: {e}")
            return {
                "visual_inspection": "No se pudo leer la imagen para análisis.",
                "timestamp": datetime.now().isoformat()
            }
        
        prompt = (
            f"Analiza esta imagen de un documento tipo '{doc_type}' y busca:\n"
            "1. Señales de alteración digital (photoshop, edición)\n"
            "2. Inconsistencias en fuentes o alineación\n"
            "3. Borrones, tachaduras o correcciones\n"
            "4. Sellos o firmas sospechosas\n"
            "5. Calidad general del documento\n"
            "6. Códigos QR o elementos de seguridad\n\n"
            "Sé específico sobre cualquier anomalía visual detectada."
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            analysis_text = response.choices[0].message.content if response.choices else ""
        except Exception as e:
            logger.exception(f"Fallo en análisis visual con modelo '{self.vision_model}': {e}")
            analysis_text = (
                "No fue posible realizar análisis visual con el modelo configurado. "
                "Verifique que el modelo soporte entrada de imagen (recomendado: gpt-4o)."
            )
        
        return {
            "visual_inspection": analysis_text,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_context(
        self, 
        current_doc: Dict[str, Any], 
        context_docs: List[Dict]
    ) -> Dict[str, Any]:
        """Analiza el documento en contexto con otros documentos del caso"""
        prompt = f"""
        Analiza la coherencia y consistencia entre estos documentos de un mismo siniestro:
        
        Documento actual:
        Tipo: {current_doc.get('document_type', 'desconocido')}
        Entidades: {json.dumps(current_doc.get('entities', {}), ensure_ascii=False)}
        
        Documentos relacionados:
        {json.dumps([
            {
                'tipo': d.get('document_type', 'otro'),
                'entidades': d.get('entities', {})
            } for d in context_docs
        ], indent=2, ensure_ascii=False)}
        
        Busca:
        1. Inconsistencias en fechas entre documentos
        2. Discrepancias en montos
        3. Diferencias en nombres o identificadores
        4. Cronología ilógica de eventos
        5. Datos contradictorios
        
        Proporciona un análisis detallado.
        """
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un investigador experto en análisis de documentos relacionados."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        
        return {
            "cross_document_analysis": response.choices[0].message.content if response.choices else "",
            "documents_analyzed": len(context_docs) + 1
        }
    
    async def _generate_summary(
        self, 
        content_analysis: Dict,
        visual_analysis: Optional[Dict],
        contextual_analysis: Optional[Dict]
    ) -> str:
        """Genera un resumen ejecutivo del análisis"""
        prompt = (
            "Genera un resumen ejecutivo conciso basado en estos análisis:\n\n"
            f"Análisis de contenido:\n{json.dumps(content_analysis, ensure_ascii=False)}\n\n"
            f"Análisis visual:\n{json.dumps(visual_analysis, ensure_ascii=False) if visual_analysis else 'No realizado'}\n\n"
            f"Análisis contextual:\n{json.dumps(contextual_analysis, ensure_ascii=False) if contextual_analysis else 'No realizado'}\n\n"
            "El resumen debe ser claro, profesional y destacar los puntos más importantes."
        )
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Genera resúmenes ejecutivos claros y concisos."},
                {"role": "user", "content": prompt}
            ],
            temperature=min(max(self.temperature, 0.0), 1.0),
            max_tokens=500
        )
        
        return response.choices[0].message.content if response.choices else ""
    
    def _extract_report_points(self, content_analysis: Dict) -> List[str]:
        """Extrae puntos clave para incluir en el informe"""
        points: List[str] = []
        if isinstance(content_analysis, dict):
            for _, value in content_analysis.items():
                if isinstance(value, str) and len(value) > 20:
                    points.append(value)
                elif isinstance(value, list):
                    points.extend([str(item) for item in value if len(str(item)) > 20])
        return points[:10]  # Limitar a 10 puntos principales
    
    def _compile_alerts(
        self,
        content_analysis: Dict,
        visual_analysis: Optional[Dict],
        contextual_analysis: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Compila todas las alertas y anomalías detectadas"""
        alerts: List[Dict[str, Any]] = []
        
        # Alertas de contenido
        try:
            if isinstance(content_analysis, dict) and content_analysis.get('anomalias'):
                for anomalia in content_analysis['anomalias']:
                    alerts.append({
                        "type": "content",
                        "severity": "medium",
                        "description": anomalia
                    })
        except Exception:
            logger.debug("No se pudieron compilar alertas de contenido.")
        
        # Alertas visuales
        try:
            if visual_analysis and 'alteración' in str(visual_analysis).lower():
                alerts.append({
                    "type": "visual",
                    "severity": "high",
                    "description": "Posibles alteraciones visuales detectadas"
                })
        except Exception:
            logger.debug("No se pudieron compilar alertas visuales.")
        
        # Alertas contextuales
        try:
            if contextual_analysis and 'inconsisten' in str(contextual_analysis).lower():
                alerts.append({
                    "type": "contextual",
                    "severity": "high",
                    "description": "Inconsistencias entre documentos"
                })
        except Exception:
            logger.debug("No se pudieron compilar alertas contextuales.")
        
        return alerts
