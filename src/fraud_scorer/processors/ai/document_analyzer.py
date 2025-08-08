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
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
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
            
            # Análisis de contenido
            "content_analysis": content_analysis,
            
            # Análisis visual
            "visual_analysis": visual_analysis,
            
            # Análisis contextual
            "contextual_analysis": contextual_analysis,
            
            # Resumen ejecutivo
            "summary": await self._generate_summary(
                content_analysis, 
                visual_analysis, 
                contextual_analysis
            ),
            
            # Puntos clave para el informe
            "report_points": self._extract_report_points(content_analysis),
            
            # Alertas y anomalías
            "alerts": self._compile_alerts(
                content_analysis, 
                visual_analysis, 
                contextual_analysis
            )
        }
    
    async def _analyze_content(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el contenido estructurado del documento"""
        
        prompt = f"""
        Analiza el siguiente documento tipo '{structured_data['document_type']}' 
        y proporciona un análisis detallado.
        
        Datos estructurados:
        {json.dumps(structured_data, indent=2, ensure_ascii=False)}
        
        Por favor proporciona:
        1. Resumen de la información clave
        2. Consistencia interna del documento
        3. Datos faltantes o incompletos
        4. Anomalías o inconsistencias detectadas
        5. Elementos que requieren verificación externa
        
        Formato tu respuesta como JSON.
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system", 
                    "content": "Eres un analista experto en documentos de seguros y detección de fraude."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def _analyze_visual_integrity(
        self, 
        image_path: str, 
        doc_type: str
    ) -> Dict[str, Any]:
        """Analiza la imagen para detectar alteraciones o anomalías visuales"""
        
        # Leer y codificar imagen
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = f"""
        Analiza esta imagen de un documento tipo '{doc_type}' y busca:
        
        1. Señales de alteración digital (photoshop, edición)
        2. Inconsistencias en fuentes o alineación
        3. Borrones, tachaduras o correcciones
        4. Sellos o firmas sospechosas
        5. Calidad general del documento
        6. Códigos QR o elementos de seguridad
        
        Sé específico sobre cualquier anomalía visual detectada.
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Parsear respuesta
        analysis_text = response.choices[0].message.content
        
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
        Tipo: {current_doc['document_type']}
        Entidades: {json.dumps(current_doc.get('entities', {}), ensure_ascii=False)}
        
        Documentos relacionados:
        {json.dumps([
            {
                'tipo': doc['document_type'],
                'entidades': doc.get('entities', {})
            } for doc in context_docs
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
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un investigador experto en análisis de documentos relacionados."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return {
            "cross_document_analysis": response.choices[0].message.content,
            "documents_analyzed": len(context_docs) + 1
        }
    
    async def _generate_summary(
        self, 
        content_analysis: Dict,
        visual_analysis: Optional[Dict],
        contextual_analysis: Optional[Dict]
    ) -> str:
        """Genera un resumen ejecutivo del análisis"""
        
        prompt = f"""
        Genera un resumen ejecutivo conciso basado en estos análisis:
        
        Análisis de contenido:
        {json.dumps(content_analysis, ensure_ascii=False)}
        
        Análisis visual:
        {json.dumps(visual_analysis, ensure_ascii=False) if visual_analysis else 'No realizado'}
        
        Análisis contextual:
        {json.dumps(contextual_analysis, ensure_ascii=False) if contextual_analysis else 'No realizado'}
        
        El resumen debe ser claro, profesional y destacar los puntos más importantes.
        """
        
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "Genera resúmenes ejecutivos claros y concisos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _extract_report_points(self, content_analysis: Dict) -> List[str]:
        """Extrae puntos clave para incluir en el informe"""
        points = []
        
        # Extraer puntos del análisis de contenido
        if isinstance(content_analysis, dict):
            for key, value in content_analysis.items():
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
        alerts = []
        
        # Alertas de contenido
        if content_analysis.get('anomalias'):
            for anomalia in content_analysis['anomalias']:
                alerts.append({
                    "type": "content",
                    "severity": "medium",
                    "description": anomalia
                })
        
        # Alertas visuales
        if visual_analysis and 'alteración' in str(visual_analysis).lower():
            alerts.append({
                "type": "visual",
                "severity": "high",
                "description": "Posibles alteraciones visuales detectadas"
            })
        
        # Alertas contextuales
        if contextual_analysis and 'inconsisten' in str(contextual_analysis).lower():
            alerts.append({
                "type": "contextual",
                "severity": "high",
                "description": "Inconsistencias entre documentos"
            })
        
        return alerts
    