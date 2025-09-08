"""
Motor de clasificación reutilizable (texto y visión opcional).

Provee una interfaz unificada para clasificar documentos usando LLM
(texto) y opcionalmente visión (para PDF/imagen). Se apoya en las
definiciones de DocumentClassifier y mantiene compatibilidad con el
pipeline principal y el preview.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class ClassifierEngine:
    """
    Núcleo compartido para clasificación de documentos.

    - LLM-first para texto; fallback a heurística si hay error.
    - Visión opcional (solo para preview); pipeline la desactiva.
    - No usa el nombre de archivo para influir en la decisión (contenido primero).
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_classifier: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name or "gpt-4o-mini"
        self.base_classifier = base_classifier  # Para fallback heurístico y definiciones

        try:
            self.client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()
        except Exception as e:
            logger.warning(f"No se pudo inicializar cliente OpenAI: {e}")
            self.client = None

        # Construir guía de tipos a partir de definiciones del clasificador base
        self.types_guide = self._build_types_guide()

    def _build_types_guide(self) -> str:
        """Construye guía compacta de categorías desde el clasificador base."""
        if not self.base_classifier or not getattr(self.base_classifier, "type_definitions", None):
            return "- otro: documentos que no encajan en las categorías anteriores"

        lines: List[str] = []
        for type_name, definition in self.base_classifier.type_definitions.items():
            keywords = ", ".join(definition.keywords[:6]) if definition.keywords else ""
            must = ", ".join(definition.must_have[:4]) if definition.must_have else "(ninguno)"
            may = ", ".join(definition.may_have[:4]) if definition.may_have else "(opcionales)"
            excl = ", ".join(definition.exclude[:4]) if definition.exclude else "(ninguno)"
            lines.append(
                f"- {type_name}: {definition.description}. keywords: [{keywords}] | must_have: [{must}] | "
                f"may_have: [{may}] | exclude: [{excl}]"
            )
        lines.append("- otro: documentos que no encajan en las categorías anteriores")
        return "\n".join(lines)

    def _build_prompt_text(self, sample_text: str) -> str:
        """Construye el prompt textual (sin depender del nombre de archivo)."""
        prompt = f"""Clasifica este documento de siniestro de seguros.

TIPOS PERMITIDOS (usa el nombre exacto):
{self.types_guide}

DOCUMENTO (muestra):
{sample_text}

INSTRUCCIONES:
1. Basa tu decisión principalmente en el CONTENIDO.
2. Identifica el tipo más apropiado de la lista según la semántica del documento.
3. Distingue cuidadosamente categorías cercanas; si no encaja, usa "otro".

Responde SOLO con JSON válido:
{{
  "document_type": "tipo_exacto_de_la_lista",
  "confidence": 0.0-1.0,
  "reasons": ["razón 1", "razón 2", "máximo 3 razones"]
}}"""
        return prompt

    async def _classify_text_llm(self, sample_text: str) -> Tuple[str, float, List[str]]:
        if not self.client:
            raise RuntimeError("Cliente OpenAI no disponible")

        from fraud_scorer.settings import CLASSIFICATION_CONFIG
        model = self.model_name or CLASSIFICATION_CONFIG.get("llm_model", "gpt-4o-mini")
        temperature = float(CLASSIFICATION_CONFIG.get("llm_temperature", 0.1))
        max_tokens = int(CLASSIFICATION_CONFIG.get("llm_max_completion_tokens", 200))

        prompt = self._build_prompt_text(sample_text)

        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un experto en clasificación de documentos de seguros. Responde solo con JSON válido."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )

        content = (response.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
        result = json.loads(content)

        # Validar tipo permitido
        valid_types = list(getattr(self.base_classifier, "type_definitions", {}).keys())
        if result.get("document_type") not in valid_types:
            result["document_type"] = "otro"
            try:
                result["confidence"] = float(result.get("confidence", 0)) * 0.5
            except Exception:
                result["confidence"] = 0.0
            reasons = result.setdefault("reasons", [])
            reasons.append("Tipo no reconocido, clasificado como 'otro'")

        return (
            str(result.get("document_type", "otro")),
            float(result.get("confidence", 0.0)),
            list(result.get("reasons", []))[:3],
        )

    async def _classify_vision_llm(self, document_path: Path) -> Tuple[str, float, List[str]]:
        if not self.client:
            raise RuntimeError("Cliente OpenAI no disponible")

        # Intentar enviar imagen base64 (o páginas del PDF)
        parts: List[Dict[str, Any]] = []
        try:
            import base64
            suffix = document_path.suffix.lower()
            if suffix == ".pdf":
                try:
                    import fitz  # type: ignore
                    with fitz.open(str(document_path)) as doc:
                        for i in range(min(len(doc), 2)):
                            pix = doc[i].get_pixmap(dpi=220)
                            b = base64.b64encode(pix.tobytes("png")).decode("ascii")
                            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
                except Exception:
                    # Si no hay PyMuPDF, degradar a texto
                    return await self._classify_text_llm("")
            elif suffix in {".jpg", ".jpeg", ".png"}:
                b = base64.b64encode(document_path.read_bytes()).decode("ascii")
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/{suffix.lstrip('.')};base64,{b}"}})
            else:
                # No compatible para visión directa
                return await self._classify_text_llm("")
        except Exception as e:
            logger.warning(f"Error preparando contenido visual: {e}")
            return await self._classify_text_llm("")

        system = (
            "Eres un experto en clasificación de documentos de seguros. "
            "Analiza el contenido visual y responde solo con JSON válido."
        )
        guide_text = self.types_guide
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Clasifica en uno de los tipos permitidos:\n" + guide_text}
        ] + parts + [
            {
                "type": "text",
                "text": (
                    "Responde en JSON con: {\\n  \\\"document_type\\\": \\\"...\\\", \\n  \\\"confidence\\\": 0.0, \\n"
                    "  \\\"reasons\\\": [\\\"...\\\"]\\n}"
                ),
            }
        ]

        # Usar el mismo modelo textual (4o-mini/4o) para visión multimodal
        from fraud_scorer.settings import CLASSIFICATION_CONFIG
        model = self.model_name or CLASSIFICATION_CONFIG.get("llm_model", "gpt-4o-mini")
        temperature = float(CLASSIFICATION_CONFIG.get("llm_temperature", 0.1))

        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=400,
        )
        content = (response.choices[0].message.content or "").replace("```json", "").replace("```", "").strip()
        result = json.loads(content)

        valid_types = list(getattr(self.base_classifier, "type_definitions", {}).keys())
        if result.get("document_type") not in valid_types:
            result["document_type"] = "otro"
            try:
                result["confidence"] = float(result.get("confidence", 0)) * 0.5
            except Exception:
                result["confidence"] = 0.0
            reasons = result.setdefault("reasons", [])
            reasons.append("Tipo no reconocido, clasificado como 'otro'")

        return (
            str(result.get("document_type", "otro")),
            float(result.get("confidence", 0.0)),
            list(result.get("reasons", []))[:3],
        )

    async def classify(
        self,
        *,
        sample_text: str,
        filename: str = "",
        document_path: Optional[Path] = None,
        use_llm: bool = True,
        use_vision: bool = False,
    ) -> Tuple[str, float, List[str]]:
        """Clasifica un documento con LLM; fallback a heurística en error.

        - use_vision: solo para preview (pipeline debe dejarlo False)
        - filename no se usa en el prompt; se pasa solo por compatibilidad de fallback heurístico
        """
        # Intento LLM
        if use_llm and self.client is not None:
            try:
                if use_vision and document_path is not None:
                    return await self._classify_vision_llm(document_path)
                else:
                    return await self._classify_text_llm(sample_text)
            except Exception as e:
                logger.warning(f"LLM falló, se usará heurística: {e}")
                # cae a heurística

        # Fallback a heurística
        if self.base_classifier is not None:
            try:
                # Evitar sesgo por nombre: pasar filename si así lo deseas;
                # aquí usamos filename para mantener compatibilidad histórica.
                return self.base_classifier._heuristic_classify(sample_text, filename)
            except Exception as e:
                logger.error(f"Heurística también falló: {e}")
        return "otro", 0.0, ["No se pudo clasificar"]

