"""
Clasificador de documentos para preview (sin persistencia).

Estrategia LLM-first con descripciones dinámicas de categorías derivadas
del clasificador base. Si el LLM falla (error de red/credenciales/timeout),
se utiliza la heurística del clasificador base como fallback.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from openai import AsyncOpenAI
from openai import OpenAIError  # tipo de error del cliente
import os
import re
import unicodedata

# Importar clasificador base y definiciones
from fraud_scorer.processors.document_classifier import DocumentClassifier
from fraud_scorer.classification.engine import ClassifierEngine

logger = logging.getLogger(__name__)


class DocumentPreviewClassifier:
    """
    Clasificador temporal para la vista previa de clasificación.

    - No genera caché ni escribe a disco.
    - Usa LLM como primer intento y heurística como fallback ante errores.
    - Genera una guía de categorías dinámica basada en las definiciones del
      `DocumentClassifier` para evitar desalineaciones.
    """

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        # Cliente OpenAI asíncrono (se intenta con api_key o variable de entorno)
        self.client: Optional[AsyncOpenAI] = None
        try:
            # Permitir inyección explícita; si no, confiar en OPENAI_API_KEY
            key = api_key or os.getenv("OPENAI_API_KEY")
            if key:
                self.client = AsyncOpenAI(api_key=key)
            else:
                # Intento sin pasar api_key (el SDK leerá el entorno si existe)
                self.client = AsyncOpenAI()
        except Exception as e:  # pragma: no cover - robustez
            logger.warning(
                f"No fue posible inicializar cliente OpenAI (se usará heurística si se solicita LLM): {e}"
            )
            self.client = None

        # Modelo textual por defecto (económico)
        self.model_name = model_name or "gpt-4o-mini"
        # Modelo para visión: preferir gpt-4o completo para mejor OCR interno
        self.vision_model = "gpt-4o" if (not self.model_name or "mini" in self.model_name) else self.model_name

        # Historial solo en memoria
        self.classification_history: List[Dict[str, Any]] = []

        # Reutilizar clasificador base para heurísticas y definiciones
        self.base_classifier = DocumentClassifier()
        self.definitions = self.base_classifier.type_definitions  # dict canónico → definición
        # Engine compartido
        self.engine = ClassifierEngine(
            model_name=self.model_name,
            base_classifier=self.base_classifier,
        )

    async def classify(
        self,
        sample_text: str,
        filename: str,
        use_llm: bool = True,
        use_vision: bool = False,
        document_path: Optional[Path] = None,
    ) -> Tuple[str, float, List[str], str]:
        """
        Clasifica un documento (LLM primero; heurística como fallback en error).

        Retorna: (document_type, confidence, reasons, method)
        """
        start_time = datetime.now()

        method = "llm"
        if use_llm and self.engine.client is not None:
            doc_type, confidence, reasons = await self.engine.classify(
                sample_text=sample_text,
                filename=filename,
                document_path=document_path,
                use_llm=True,
                use_vision=use_vision,
            )
            reasons.insert(0, "🤖 Clasificado por LLM con guía de categorías")
            if use_vision and document_path is not None:
                reasons.insert(1, "🖼️ Clasificación por visión (contenido visual)")
        else:
            # Fallback heurístico si LLM no disponible o deshabilitado
            doc_type, confidence, reasons = self.base_classifier._heuristic_classify(sample_text, filename)
            reasons.insert(0, "📏 Clasificado por heurística (LLM no disponible/deshabilitado)")
            method = "heuristic"

        # Registrar en memoria
        elapsed = (datetime.now() - start_time).total_seconds()
        self.classification_history.append(
            {
                "filename": filename,
                "document_type": doc_type,
                "confidence": confidence,
                "method": method,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return doc_type, confidence, reasons, method

    def _heuristic_classify(self, text: str, filename: str) -> Tuple[str, float, List[str]]:
        """Clasificación por reglas heurísticas del clasificador base."""
        return self.base_classifier._heuristic_classify(text, filename)

    async def _llm_classify_with_descriptions(
        self,
        sample_text: str,
        heur_type: str,
        heur_conf: float,
        heur_reasons: List[str],
    ) -> Tuple[str, float, List[str]]:
        """
        Clasificación con LLM usando descripciones detalladas de categorías.
        Genera la guía dinámicamente a partir de `self.definitions` para asegurar
        consistencia con el sistema principal.
        """

        prompt = self._build_enhanced_classification_prompt(
            sample_text, heur_type, heur_conf, heur_reasons
        )

        if self.client is None:
            raise RuntimeError("Cliente LLM no disponible")

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un experto en clasificación de documentos de seguros y siniestros. "
                        "Compara el documento con las descripciones de categorías y responde solo con JSON válido."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )

        # Contenido JSON de la respuesta
        content = response.choices[0].message.content
        # Limpiar posibles marcadores
        content = (content or "").replace("```json", "").replace("```", "").strip()
        result = json.loads(content)

        # Validación de tipo permitido
        if result.get("document_type") not in self._get_valid_types():
            result["document_type"] = "otro"
            # Atenuar confianza si el tipo no es reconocido
            try:
                result["confidence"] = float(result.get("confidence", 0)) * 0.5
            except Exception:
                result["confidence"] = 0.0
            reasons = result.setdefault("reasons", [])
            reasons.append("Tipo no reconocido, asignado a 'otro'")

        return (
            str(result.get("document_type", "otro")),
            float(result.get("confidence", 0.0)),
            list(result.get("reasons", [])),
        )

    async def _vision_classify_image(self, image_path: "Path") -> Tuple[str, float, List[str]]:
        """Clasifica un documento basado en una imagen usando visión del modelo.

        No usa el nombre del archivo; solo el contenido visual.
        """
        import base64
        from pathlib import Path as _P

        if self.client is None:
            raise RuntimeError("Cliente LLM no disponible")

        # Codificar imagen en base64 para pasarla como data URL
        p = _P(image_path)
        img_bytes = p.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("ascii")

        categories_detail = self._build_detailed_categories_guide()
        disambiguation = self._build_disambiguation_rules()
        priorities = self._build_priority_rules()

        system = (
            "Eres un experto en clasificación de documentos de seguros. "
            "Analiza únicamente el contenido visual para decidir la categoría. Responde solo con JSON válido."
        )
        user_parts: List[Dict[str, Any]] = [
            {"type": "text", "text": "Clasifica el documento mostrado en una de las categorías: \n" + categories_detail},
            {"type": "text", "text": "Reglas de desambiguación:\n" + disambiguation},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{p.suffix.lstrip('.').lower()};base64,{b64}"},
            },
            {
                "type": "text",
                "text": (
                    "Responde en JSON con: {\n"
                    "  \"document_type\": \"...\", \n"
                    "  \"confidence\": 0.0, \n"
                    "  \"reasons\": [\"...\"]\n}"
                ),
            },
        ]

        response = await self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_parts},
            ],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)

        if result.get("document_type") not in self._get_valid_types():
            result["document_type"] = "otro"
            try:
                result["confidence"] = float(result.get("confidence", 0)) * 0.5
            except Exception:
                result["confidence"] = 0.0
            reasons = result.setdefault("reasons", [])
            reasons.append("Tipo no reconocido, asignado a 'otro'")

        return (
            str(result.get("document_type", "otro")),
            float(result.get("confidence", 0.0)),
            list(result.get("reasons", [])),
        )

    async def _vision_classify_pdf(self, pdf_path: Path, sample_text: str = "") -> Tuple[str, float, List[str]]:
        """Clasifica un PDF renderizando las primeras páginas a imágenes (en memoria)."""
        if self.client is None:
            raise RuntimeError("Cliente LLM no disponible")

        # Intentar PyMuPDF (pymupdf) para render a imágenes en memoria
        try:
            import fitz  # type: ignore
        except Exception as e:
            # Sin PyMuPDF: no hay visión PDF; degradar a resultado neutro
            return (
                "otro",
                0.0,
                [
                    "Visión PDF no disponible (instala 'pymupdf').",
                    "Se requiere visión para leer contenido visual sin OCR.",
                ],
            )

        import base64
        images_data_urls: List[str] = []
        with fitz.open(str(pdf_path)) as doc:
            page_count = min(len(doc), 2)
            for i in range(page_count):
                page = doc[i]
                pix = page.get_pixmap(dpi=220)
                png_bytes = pix.tobytes("png")
                b64 = base64.b64encode(png_bytes).decode("ascii")
                images_data_urls.append(f"data:image/png;base64,{b64}")

        categories_detail = self._build_detailed_categories_guide()
        disambiguation = self._build_disambiguation_rules()

        system = (
            "Eres un experto en clasificación de documentos de seguros. "
            "Analiza únicamente el contenido visual (imágenes de páginas) para decidir."
        )

        # Armar contenido del usuario: texto + múltiples imágenes
        user_parts: List[Dict[str, Any]] = [
            {"type": "text", "text": "Clasifica el documento en una categoría de la lista:\n" + categories_detail},
            {"type": "text", "text": "Reglas de desambiguación:\n" + disambiguation},
        ]
        for url in images_data_urls:
            user_parts.append({"type": "image_url", "image_url": {"url": url}})
        if (sample_text or "").strip():
            user_parts.append({"type": "text", "text": "Texto extraído (solo apoyo, puede estar incompleto):\n" + sample_text[:1000]})
        user_parts.append(
            {
                "type": "text",
                "text": (
                    "Responde en JSON con: {\n"
                    "  \"document_type\": \"...\", \n"
                    "  \"confidence\": 0.0, \n"
                    "  \"reasons\": [\"...\"]\n}"
                ),
            }
        )

        response = await self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_parts},
            ],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)

        if result.get("document_type") not in self._get_valid_types():
            result["document_type"] = "otro"
            try:
                result["confidence"] = float(result.get("confidence", 0)) * 0.5
            except Exception:
                result["confidence"] = 0.0
            reasons = result.setdefault("reasons", [])
            reasons.append("Tipo no reconocido, asignado a 'otro'")

        return (
            str(result.get("document_type", "otro")),
            float(result.get("confidence", 0.0)),
            list(result.get("reasons", [])),
        )

    def _build_enhanced_classification_prompt(
        self,
        sample_text: str,
        heur_type: str,
        heur_conf: float,
        heur_reasons: List[str],
    ) -> str:
        """Construye un prompt con guía detallada y señales adicionales sin usar el nombre del archivo."""

        categories_detail = self._build_detailed_categories_guide()
        disambiguation = self._build_disambiguation_rules()

        heur_block = "\n".join(
            [
                f"• Tipo sugerido: {heur_type}",
                f"• Confianza heurística: {heur_conf:.2f}",
                "• Razones heurísticas:" if heur_reasons else "• Sin razones heurísticas",
            ]
            + [f"   - {r}" for r in (heur_reasons or [])]
        )

        prompt = f"""
Analiza el siguiente documento y clasifícalo en la categoría más apropiada.

 📋 CATEGORÍAS DISPONIBLES CON DESCRIPCIONES (generadas dinámicamente):

{categories_detail}

🧭 REGLAS DE DESAMBIGUACIÓN IMPORTANTES:

{disambiguation}

⚖️ PRIORIDADES SI HAY CONFLICTO:
{priorities}

📌 SUGERENCIA HEURÍSTICA DEL SISTEMA (puede estar equivocada, úsala como pista):
{heur_block}

📄 DOCUMENTO A ANALIZAR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Contenido del documento (muestra):
{sample_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 INSTRUCCIONES:
1. Compara propósito, estructura, palabras clave y contexto con cada categoría.
2. Selecciona la categoría que mejor corresponda.
3. Si no encaja claramente en ninguna, usa "otro".
 4. Si el texto es muy escaso, reduce la confianza o clasifica como "otro".
 5. Respeta las exclusiones de cada categoría (no las asignes si un término excluyente está presente).

📊 RESPUESTA REQUERIDA (JSON válido):
{{
  "document_type": "nombre_exacto_de_categoria",
  "confidence": 0.0,
  "reasons": [
    "Razón 1 específica",
    "Razón 2 basada en contenido y estructura"
  ]
}}

IMPORTANTE: El "document_type" debe ser EXACTAMENTE uno de los listados arriba.
"""
        return prompt

    def _build_detailed_categories_guide(self) -> str:
        """Genera guía detallada a partir de las definiciones canónicas, con exclusiones."""

        lines: List[str] = []
        for type_name, definition in self.definitions.items():
            # Descripción y listas de palabras clave
            desc = definition.description or "(sin descripción)"
            main_keywords = ", ".join(definition.keywords[:6]) if definition.keywords else ""
            must = ", ".join(definition.must_have[:4]) if definition.must_have else "(ninguno)"
            may = ", ".join(definition.may_have[:4]) if definition.may_have else "(opcionales)"
            excl = ", ".join(definition.exclude[:6]) if definition.exclude else "(ninguno)"

            lines.append(f"📁 {type_name}")
            lines.append(f"   📝 Descripción: {desc}")
            if main_keywords:
                lines.append(f"   🔑 Keywords: {main_keywords}")
            lines.append(f"   ✅ Debe contener: {must}")
            lines.append(f"   ➕ Puede contener: {may}")
            lines.append(f"   ❌ No confundir con (exclusiones): {excl}")
            lines.append("")

        lines.append("📁 otro\n   Documentos que no encajan en las categorías anteriores.")
        return "\n".join(lines)

    def _build_disambiguation_rules(self) -> str:
        """Reglas específicas para pares propensos a confusión."""
        rules = [
            "- Si el contenido menciona INE/IFE, 'Instituto Nacional Electoral' o 'credencial para votar' → identificacion_oficial (NO 'licencia_del_operador').",
            "- 'licencia_del_operador' es licencia de conducir emitida por tránsito; no confundir con INE/pasaporte.",
            "- Si el contenido contiene términos como 'ficha técnica', datos técnicos del vehículo (marca, modelo, año, NIV/serie, motor) y fotos, sugiere 'ficha_tecnica_de_vehiculo'.",
            "- 'reporte_de_costos_y_rendimientos' NO son facturas; si el contenido muestra 'factura(s)', 'commercial invoice', 'BL/bill of lading', 'aduana', 'pedimento', 'incoterms' → 'facturas_comerciales_internacionales'.",
            "- 'tarjeta_de_circulacion_vehiculo' es autorización oficial; no confundir con ficha técnica comercial del vehículo.",
            "- CFDI con 'Carta Porte' o secciones/etiquetas de complemento ('Complemento Carta Porte', 'Mercancías', 'Autotransporte', 'Ubicaciones', 'Figura Transporte', 'PermSCT') → 'cfdi_carta_porte' incluso si aparece 'Factura', 'UUID' y sello SAT.",
            "- Si NO hay campos fiscales (UUID, Sello Digital, SAT, CFDI) y hay tablas con códigos/cantidades y firmas/responsables de almacén/transportista/entrega/recibe → 'salida_de_almacen'.",
            "- Si el documento es un relato en primera persona ('yo', 'me', 'declaro', 'narro') con secciones 'Narración de Hechos'/'Declarante' y párrafos de texto continuo → 'narracion_de_hechos'. 'carpeta_de_investigacion' es el conjunto del expediente (folios, acuerdos, oficios), no el relato.",
        ]
        return "\n".join(rules)

    def _build_priority_rules(self) -> str:
        """Prioridades de clasificación en caso de conflicto entre categorías.

        Ayuda a romper empates cuando aparecen señales cruzadas en documentos reales.
        """
        lines = [
            "- Si existe 'Carta Porte' o secciones del complemento → prioriza 'cfdi_carta_porte' sobre 'factura_comercial_cfdi'.",
            "- Si NO hay CFDI/UUID/SAT y hay firmas/responsables típicas de almacén → prioriza 'salida_de_almacen' sobre 'guias_y_facturas[_consolidadas]'.",
            "- Si el documento es un relato personal en primera persona → prioriza 'narracion_de_hechos' sobre 'carpeta_de_investigacion'.",
            "- 'guias_y_facturas_consolidadas' requiere explícitamente 'consolidado' o referencias claras a múltiples guías/clientes; si no, no asignar.",
        ]
        return "\n".join(lines)

    async def _balanced_classify(self, document_path: Path, sample_text: str, filename: str) -> Tuple[str, float, List[str]]:
        """Implementa la combinación 80/20: contenido (visión+texto) y nombre.
        Incluye reglas de prioridad para pares conflictivos.
        """
        # Ver implementación en la sección auxiliar inferior
        flags: Dict[str, Any] = {
            "has_carta_porte": False,
            "has_ccp_sections": False,
            "has_uuid": False,
            "has_sello_sat": False,
            "has_cfdi": False,
            "has_consolidado": False,
            "has_salida_signoff": False,
            "is_first_person_narrative": False,
            "has_section_narracion": False,
            "has_expediente_markers": False,
        }

        # Señales visuales
        try:
            cp = await self._vision_flags_carta_porte(document_path)
            if isinstance(cp, dict):
                flags["has_carta_porte"] = bool(cp.get("has_carta_porte"))
                flags["has_ccp_sections"] = flags["has_carta_porte"]
        except Exception:
            pass
        try:
            sc = await self._vision_flags_salida_vs_consolidado(document_path)
            if isinstance(sc, dict):
                flags["has_consolidado"] = bool(sc.get("has_consolidado"))
                flags["has_salida_signoff"] = bool(sc.get("has_salida_signoff"))
        except Exception:
            pass
        try:
            nc = await self._vision_flags_narracion_vs_carpeta(document_path)
            if isinstance(nc, dict):
                flags["is_first_person_narrative"] = bool(nc.get("is_first_person_narrative"))
                flags["has_section_narracion"] = bool(nc.get("has_section_narracion"))
                flags["has_expediente_markers"] = bool(nc.get("has_expediente_markers"))
        except Exception:
            pass

        # Señales textuales (si existen)
        t = (sample_text or "").lower()
        if t:
            if re.search(r"\buuid\b|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", t):
                flags["has_uuid"] = True
            if "sello sat" in t or "sello cfdi" in t:
                flags["has_sello_sat"] = True
            if "cfdi" in t:
                flags["has_cfdi"] = True
            if self._text_flags_carta_porte(sample_text):
                flags["has_carta_porte"] = True
                flags["has_ccp_sections"] = True
            if self._text_flags_salida_almacen(sample_text):
                flags["has_salida_signoff"] = True

        # Señales de informes via visión (final/preliminar)
        try:
            inf = await self._vision_flags_informes(document_path)
            flags["is_informe_final"] = bool(inf.get("is_informe_final", False))
            flags["is_informe_preliminar"] = bool(inf.get("is_informe_preliminar", False))
        except Exception:
            flags["is_informe_final"] = False
            flags["is_informe_preliminar"] = False

        # Scoring por nombre (lo movemos antes para poder usar hints en contenido)
        name_scores, name_reasons = self._filename_scores(filename)
        filename_hint_salida = name_scores.get("salida_de_almacen", 0.0) > 0.0

        # Baseline por visión: pedir una clasificación directa para anclar puntaje
        primary_cat: Optional[str] = None
        primary_conf: float = 0.0
        try:
            if document_path.suffix.lower() == ".pdf":
                primary_cat, primary_conf, _ = await self._vision_classify_pdf(document_path, sample_text)
            else:
                primary_cat, primary_conf, _ = await self._vision_classify_image(document_path)
        except Exception:
            primary_cat, primary_conf = None, 0.0

        # Scoring contenido: baseline 0.0, subir con visión directa y señales claras
        content_scores: Dict[str, float] = {k: 0.0 for k in self._get_valid_types()}
        if primary_cat in content_scores:
            content_scores[primary_cat] = max(content_scores[primary_cat], float(primary_conf))
        # Carta Porte vs Factura
        if flags.get("has_carta_porte") or flags.get("has_ccp_sections"):
            content_scores["cfdi_carta_porte"] = 0.95
            content_scores["factura_comercial_cfdi"] = 0.10
        else:
            if flags.get("has_uuid") or flags.get("has_sello_sat") or flags.get("has_cfdi"):
                content_scores["factura_comercial_cfdi"] = 0.85
        # Salida vs Consolidado
        if flags.get("has_consolidado"):
            content_scores["guias_y_facturas_consolidadas"] = 0.90
        if flags.get("has_salida_signoff") and not flags.get("has_cfdi"):
            content_scores["salida_de_almacen"] = 0.90
            if content_scores.get("guias_y_facturas_consolidadas", 0.0) > 0 and not flags.get("has_consolidado"):
                content_scores["guias_y_facturas_consolidadas"] = 0.20
        # Narración vs Carpeta
        if flags.get("is_first_person_narrative") or flags.get("has_section_narracion"):
            content_scores["narracion_de_hechos"] = 0.90
            if not flags.get("has_expediente_markers"):
                content_scores["carpeta_de_investigacion"] = max(content_scores.get("carpeta_de_investigacion", 0.0), 0.20)
        elif flags.get("has_expediente_markers"):
            content_scores["carpeta_de_investigacion"] = 0.90

        # Informes del ajustador
        if flags.get("is_informe_final"):
            content_scores["informe_final_del_ajustador"] = 0.95
            content_scores["salida_de_almacen"] = min(content_scores.get("salida_de_almacen", 0.0), 0.20)
        if flags.get("is_informe_preliminar"):
            content_scores["informe_preliminar_del_ajustador"] = 0.95
            content_scores["salida_de_almacen"] = min(content_scores.get("salida_de_almacen", 0.0), 0.20)

        # Scoring nombre ya calculado arriba

        # Agregación
        W_CONTENT, W_NAME = 0.8, 0.2
        final_scores: Dict[str, float] = {}
        for cat in self._get_valid_types():
            c = content_scores.get(cat, 0.0)
            n = name_scores.get(cat, 0.0)
            final_scores[cat] = W_CONTENT * c + W_NAME * n

        best_cat, best_score = max(final_scores.items(), key=lambda x: x[1])

        # Umbral mínimo: si la evidencia es débil, clasificar como 'otro'
        MIN_SCORE = 0.35
        if best_score < MIN_SCORE:
            return "otro", float(best_score), [
                "Evidencia insuficiente para categorías específicas",
                f"Score={best_score:.2f} (< {MIN_SCORE:.2f})",
            ]

        # Reglas de prioridad
        if flags.get("has_carta_porte") and best_cat == "factura_comercial_cfdi":
            best_cat, best_score = "cfdi_carta_porte", max(best_score, 0.95)
        if best_cat == "guias_y_facturas_consolidadas" and (not flags.get("has_consolidado")) and flags.get("has_salida_signoff"):
            best_cat, best_score = "salida_de_almacen", max(best_score, 0.90)
        if best_cat == "carpeta_de_investigacion" and (flags.get("is_first_person_narrative") or flags.get("has_section_narracion")) and not flags.get("has_expediente_markers"):
            best_cat, best_score = "narracion_de_hechos", max(best_score, 0.90)

        # Razones
        reasons: List[str] = []
        active_flags = [k for k, v in flags.items() if v]
        if active_flags:
            reasons.append("Flags activas (contenido): " + ", ".join(active_flags))
        nr = name_reasons.get(best_cat)
        if nr:
            reasons.append(f"Nombre: {nr}")
        reasons.append(f"Pesos: contenido={W_CONTENT:.0%}, nombre={W_NAME:.0%}")

        return best_cat, float(min(max(best_score, 0.0), 1.0)), reasons

    async def _vision_flags_informes(self, path: Path) -> Dict[str, Any]:
        if self.client is None:
            return {"is_informe_final": False, "is_informe_preliminar": False}
        parts: List[Dict[str, Any]] = []
        import base64
        if path.suffix.lower() == ".pdf":
            try:
                import fitz  # type: ignore
                with fitz.open(str(path)) as doc:
                    for i in range(min(len(doc), 3)):
                        pix = doc[i].get_pixmap(dpi=300)
                        b = base64.b64encode(pix.tobytes("png")).decode("ascii")
                        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
            except Exception:
                parts = []
        else:
            b = base64.b64encode(path.read_bytes()).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/{path.suffix.lstrip('.').lower()};base64,{b}"}})

        prompt = (
            "Devuelve solo JSON con {is_informe_final, is_informe_preliminar}.\n"
            "- is_informe_final = true si aparece título/etiqueta 'INFORME FINAL' claramente.\n"
            "- is_informe_preliminar = true si aparece 'INFORME PRELIMINAR'.\n"
            "No marques ambos a la vez. Ignora palabras sueltas si no son títulos claros."
        )
        user_content = [{"type": "text", "text": prompt}] + parts
        resp = await self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": "Responde solo con JSON válido."},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=40,
            response_format={"type": "json_object"},
        )
        data = resp.choices[0].message.content or "{}"
        import json as _json
        return _json.loads(data)

    def _filename_scores(self, filename: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        scores: Dict[str, float] = {}
        reasons: Dict[str, str] = {}
        name = (filename or "").lower()
        name_norm = unicodedata.normalize("NFKD", name)
        name_norm = "".join(c for c in name_norm if not unicodedata.combining(c))

        tokens_map: Dict[str, Dict[str, List[str]]] = {
            "cfdi_carta_porte": {"exact": ["carta porte", "carta_porte", "cp"]},
            "factura_comercial_cfdi": {"exact": ["factura", "factura comercial", "cfdi"]},
            "salida_de_almacen": {"exact": ["salida de almacen", "salida de almacén", "salida_almacen"]},
            "guias_y_facturas_consolidadas": {"exact": ["consolidado", "consolidada", "consolidadas"]},
            "narracion_de_hechos": {"exact": ["narracion de hechos", "narración de hechos", "narracion_de_hechos"]},
            "carpeta_de_investigacion": {"exact": ["carpeta de investigacion", "carpeta_de_investigacion"]},
        }

        def quality_for(cat: str) -> Tuple[float, str]:
            spec = tokens_map.get(cat, {})
            for tok in spec.get("exact", []):
                # Usar regex para tolerar separadores no alfanuméricos
                pattern = re.escape(tok).replace("\\ ", "\\W*")
                if re.search(pattern, name_norm):
                    return 1.0, f"coincide '{tok}'"
            return 0.0, "(sin señal relevante)"

        for cat in self._get_valid_types():
            q, why = quality_for(cat)
            if q > 0:
                scores[cat] = q
                reasons[cat] = why
        return scores, reasons

    async def _post_disambiguation_refinement(
        self,
        doc_type: str,
        confidence: float,
        reasons: List[str],
        document_path: Optional[Path],
        sample_text: str,
    ) -> Tuple[str, float, List[str]]:
        """Refina la clasificación para pares conflictivos.

        - factura_comercial_cfdi vs cfdi_carta_porte
        - guias_y_facturas_consolidadas vs salida_de_almacen
        """
        # 1) Carta Porte verificación visual/textual
        if doc_type == "factura_comercial_cfdi":
            has_cp = False
            if document_path is not None:
                try:
                    flags = await self._vision_flags_carta_porte(document_path)
                    has_cp = bool(flags.get("has_carta_porte"))
                except Exception:
                    has_cp = False
            if not has_cp and (sample_text or "").strip():
                has_cp = self._text_flags_carta_porte(sample_text)

            if has_cp:
                reasons = [
                    "🔎 Verificación: se detectó complemento Carta Porte (visual/texto)",
                    *reasons,
                ]
                return "cfdi_carta_porte", max(confidence, 0.95), reasons

        # 2) Consolidado vs Salida de almacén
        if doc_type == "guias_y_facturas_consolidadas":
            flags = {"has_consolidado": False, "has_salida_signoff": False}
            if document_path is not None:
                try:
                    resp = await self._vision_flags_salida_vs_consolidado(document_path)
                    if isinstance(resp, dict):
                        flags["has_consolidado"] = bool(resp.get("has_consolidado", False))
                        flags["has_salida_signoff"] = bool(resp.get("has_salida_signoff", False))
                except Exception:
                    pass
            if not flags.get("has_consolidado", False):
                # Si no hay consolidado explícito, pero hay señal de salida
                if flags.get("has_salida_signoff", False) or self._text_flags_salida_almacen(sample_text):
                    reasons = [
                        "🔎 Verificación: sin 'Consolidado' explícito y con firmas/‘recibe/entrega’ típicas de almacén",
                        *reasons,
                    ]
                    return "salida_de_almacen", max(confidence, 0.90), reasons

        return doc_type, confidence, reasons

    def _text_flags_carta_porte(self, text: str) -> bool:
        t = (text or "").lower()
        keys = [
            "carta porte",
            "complemento carta porte",
            "mercancias",
            "mercancías",
            "autotransporte",
            "ubicaciones",
            "figura transporte",
            "permsct",
        ]
        return any(k in t for k in keys)

    def _text_flags_salida_almacen(self, text: str) -> bool:
        t = (text or "").lower()
        # Presencia de términos de salida + firmas responsables
        must_any = ["salida de almacen", "salida de almacén", "embarque", "almacen", "almacén"]
        signoffs = ["firma", "responsable", "recibe", "entrega", "transportista"]
        return (any(k in t for k in must_any) and any(s in t for s in signoffs))

    async def _vision_flags_carta_porte(self, path: Path) -> Dict[str, Any]:
        if self.client is None:
            return {"has_carta_porte": False}

        # Preparar contenido visual (pdf → páginas; imagen → base64)
        parts: List[Dict[str, Any]] = []
        import base64
        if path.suffix.lower() == ".pdf":
            try:
                import fitz  # type: ignore
                with fitz.open(str(path)) as doc:
                    for i in range(min(len(doc), 2)):
                        pix = doc[i].get_pixmap(dpi=220)
                        b = base64.b64encode(pix.tobytes("png")).decode("ascii")
                        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
            except Exception:
                parts = []
        else:
            b = base64.b64encode(path.read_bytes()).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/{path.suffix.lstrip('.').lower()};base64,{b}"}})

        prompt = (
            "Detecta si el documento es un CFDI con COMPLEMENTO CARTA PORTE y responde JSON con {has_carta_porte}.\n"
            "Marca has_carta_porte=true SOLO si aparece explícitamente 'Carta Porte' o 'Complemento Carta Porte', o\n"
            "si ves secciones/etiquetas propias del complemento: 'Mercancías', 'Autotransporte', 'Ubicaciones',\n"
            "'Figura Transporte', 'PermSCT', 'Remolques'. Si solo ves 'CFDI', 'UUID', 'Factura' y sellos SAT,\n"
            "sin las etiquetas del complemento, devuelve false."
        )
        user_content = [{"type": "text", "text": prompt}] + parts
        resp = await self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": "Responde solo con JSON válido."},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=50,
            response_format={"type": "json_object"},
        )
        data = resp.choices[0].message.content or "{}"
        import json as _json
        return _json.loads(data)

    async def _vision_flags_salida_vs_consolidado(self, path: Path) -> Dict[str, Any]:
        if self.client is None:
            return {"has_consolidado": False, "has_salida_signoff": False}

        parts: List[Dict[str, Any]] = []
        import base64
        if path.suffix.lower() == ".pdf":
            try:
                import fitz  # type: ignore
                with fitz.open(str(path)) as doc:
                    for i in range(min(len(doc), 2)):
                        pix = doc[i].get_pixmap(dpi=220)
                        b = base64.b64encode(pix.tobytes("png")).decode("ascii")
                        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
            except Exception:
                parts = []
        else:
            b = base64.b64encode(path.read_bytes()).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/{path.suffix.lstrip('.').lower()};base64,{b}"}})

        prompt = (
            "Evalúa dos señales y responde solo JSON con campos exactos: {has_consolidado, has_salida_signoff}.\n"
            "- has_consolidado = true SOLO si aparece literalmente la palabra 'Consolidado' en título/etiqueta/encabezado.\n"
            "  No infieras 'consolidado' por tener muchas filas o varias páginas.\n"
            "- has_salida_signoff = true si aparecen etiquetas propias de salida de almacén como 'Entrega', 'Recibe',\n"
            "  'Almacén/Almacen', 'Responsable', 'Transportista' (al menos dos de ellas, cercanas a firmas o campos).\n"
            "  No cuentes 'Firma' genérica sin estas etiquetas."
        )
        user_content = [{"type": "text", "text": prompt}] + parts
        resp = await self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": "Responde solo con JSON válido."},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=60,
            response_format={"type": "json_object"},
        )
        data = resp.choices[0].message.content or "{}"
        import json as _json
        return _json.loads(data)

    # Eliminado método de pistas por nombre de archivo para cumplir restricción de no usar filenames

    def _get_valid_types(self) -> List[str]:
        """Retorna lista de tipos canónicos válidos para la sesión."""
        return list(self.definitions.keys())

    def get_classification_stats(self) -> Dict[str, Any]:
        """Estadísticas en memoria de la sesión de clasificación."""
        if not self.classification_history:
            return {
                "total_classified": 0,
                "average_confidence": 0.0,
                "average_time_seconds": 0.0,
                "method_distribution": {},
                "type_distribution": {},
                "low_confidence_count": 0,
            }

        total = len(self.classification_history)
        avg_conf = sum(h["confidence"] for h in self.classification_history) / total

        method_dist: Dict[str, int] = {}
        type_dist: Dict[str, int] = {}
        for h in self.classification_history:
            method_dist[h["method"]] = method_dist.get(h["method"], 0) + 1
            type_dist[h["document_type"]] = type_dist.get(h["document_type"], 0) + 1

        return {
            "total_classified": total,
            "average_confidence": avg_conf,
            "average_time_seconds": sum(h["elapsed_seconds"] for h in self.classification_history) / total,
            "method_distribution": method_dist,
            "type_distribution": type_dist,
            "low_confidence_count": sum(1 for h in self.classification_history if h["confidence"] < 0.6),
        }
