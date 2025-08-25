# src/fraud_scorer/ai_extractors/ai_consolidator.py

"""
AIConsolidator: Consolida extracciones múltiples usando razonamiento de IA,
evita validación cuando no hay datos (cortocircuito) y nunca inventa valores.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel, Field

from .config import ExtractionConfig, FieldPriority  # FieldPriority puede usarse en reglas
from .models.extraction_models import (
    DocumentExtraction,
    ConsolidatedExtraction,
    ConsolidatedFields,   # << importante: usar el modelo pydantic para los campos
    ExtractionBatch
)
from .prompts.consolidation_prompts import ConsolidationPromptBuilder

logger = logging.getLogger(__name__)


# =========================
#   Modelos de respuesta
# =========================

class ConsolidationDecision(BaseModel):
    """Decisión de consolidación para un campo con trazabilidad."""
    field_name: str
    selected_value: Any
    source_document: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list)


class ValidationResponse(BaseModel):
    """
    Respuesta estructurada de validación.
    - adjustments: dict con únicamente los campos a modificar (si aplica).
    - notes: comentarios opcionales.
    """
    adjustments: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


# =========================
#      Consolidador
# =========================

class AIConsolidator:
    """
    Consolida extractos de varios documentos, resuelve conflictos y
    valida de forma conservadora (sin inventar).
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = instructor.patch(
            AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        )
        self.config = ExtractionConfig()
        self.prompt_builder = ConsolidationPromptBuilder()
        self.golden_examples = self._load_golden_examples()
        logger.info("AIConsolidator inicializado")

    # ---------- API principal ----------

    async def consolidate_extractions(
        self,
        extractions: List[DocumentExtraction],
        case_id: str,
        use_advanced_reasoning: bool = True
    ) -> ConsolidatedExtraction:
        """
        Consolida múltiples extracciones en un resultado final.

        - Cortocircuita si `extractions` está vacío (evita alucinaciones).
        - Solo valida con IA si existe al menos un valor no-nulo.
        """
        logger.info(f"Consolidando {len(extractions)} extracciones para caso {case_id}")

        # --- Cortocircuito si no hay extracciones ---
        if not extractions:
            logger.warning(
                "No se recibieron extracciones para consolidar. Devolviendo resultado vacío sin validar."
            )
            empty_fields = ConsolidatedFields()  # todos los campos None según el modelo
            return ConsolidatedExtraction(
                case_id=case_id,
                consolidated_fields=empty_fields,
                consolidation_sources={},
                conflicts_resolved=[],
                confidence_scores={
                    field: 0.0 for field in self._required_fields()
                }
            )

        # Agrupar por campo (solo valores no nulos)
        field_groups = self._group_by_field(extractions)

        consolidated_fields_dict: Dict[str, Any] = {}
        consolidation_sources: Dict[str, str] = {}
        conflicts_resolved: List[Dict[str, Any]] = []
        confidence_scores: Dict[str, float] = {}

        # Iterar por los campos del modelo (fuente de verdad)
        for field_name in self._required_fields():
            options = field_groups.get(field_name, [])

            if not options:
                # Sin datos para este campo
                consolidated_fields_dict[field_name] = None
                confidence_scores[field_name] = 0.0
                continue

            if len(options) == 1:
                # Única opción
                consolidated_fields_dict[field_name] = options[0]["value"]
                consolidation_sources[field_name] = f"Único valor de {options[0]['source']}"
                confidence_scores[field_name] = 1.0
                continue

            # Conflicto: múltiples opciones
            if use_advanced_reasoning:
                decision = await self._resolve_conflict_with_ai(
                    field_name=field_name,
                    options=options,
                    all_extractions=extractions
                )
            else:
                decision = self._resolve_conflict_with_rules(
                    field_name=field_name,
                    options=options
                )

            consolidated_fields_dict[field_name] = decision.selected_value
            consolidation_sources[field_name] = decision.reasoning
            confidence_scores[field_name] = decision.confidence

            conflicts_resolved.append({
                "field": field_name,
                "options": options,
                "selected": decision.selected_value,
                "reasoning": decision.reasoning
            })

        # ¿Hay al menos un valor real? si no, no tiene sentido validar con IA
        has_any_value = any(v is not None for v in consolidated_fields_dict.values())

        if use_advanced_reasoning and has_any_value:
            consolidated_fields_dict = await self._validate_with_ai_safe(
                consolidated_fields_dict,
                extractions
            )
        elif use_advanced_reasoning and not has_any_value:
            logger.info(
                "Todos los campos están vacíos; se omite validación con IA para evitar invenciones."
            )

        # Instanciar el modelo pydantic final (valida tipos/formato)
        try:
            final_fields = ConsolidatedFields(**consolidated_fields_dict)
        except Exception as e:
            logger.error(f"Error creando ConsolidatedFields: {e}. "
                         f"Se devuelven campos como dict sin validar.")
            # Último recurso: devolver vacío seguro
            final_fields = ConsolidatedFields()

        result = ConsolidatedExtraction(
            case_id=case_id,
            consolidated_fields=final_fields,
            consolidation_sources=consolidation_sources,
            conflicts_resolved=conflicts_resolved,
            confidence_scores=confidence_scores
        )

        self._log_consolidation_metrics(result)
        return result

    # ---------- Resolución de conflictos ----------

    async def _resolve_conflict_with_ai(
        self,
        field_name: str,
        options: List[Dict[str, Any]],
        all_extractions: List[DocumentExtraction]
    ) -> ConsolidationDecision:
        """
        Usa IA (estructura) para resolver conflictos de un campo.
        """
        prompt = self.prompt_builder.build_conflict_resolution_prompt(
            field_name=field_name,
            options=options,
            field_rules=self.config.FIELD_SOURCE_RULES.get(field_name, []),
            golden_examples=self._get_relevant_examples(field_name),
            context=self._build_context(all_extractions)
        )

        try:
            response: ConsolidationDecision = await self.client.chat.completions.create(
                model=self.config.get_model_for_task("consolidation"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un experto ajustador de seguros con 20 años de experiencia. "
                            "Selecciona el mejor valor entre opciones dadas. Explica brevemente."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_model=ConsolidationDecision,
                temperature=0.1,
                max_tokens=1000,
            )
            return response

        except Exception as e:
            logger.error(f"Error en resolución con IA para {field_name}: {e}. "
                         "Aplicando reglas de respaldo.")
            return self._resolve_conflict_with_rules(field_name, options)

    def _resolve_conflict_with_rules(
        self,
        field_name: str,
        options: List[Dict[str, Any]]
    ) -> ConsolidationDecision:
        """
        Reglas determinísticas como respaldo (prioridades por fuente -> valor más común).
        """
        priority_sources = self.config.FIELD_SOURCE_RULES.get(field_name, [])
        if priority_sources:
            for priority_source in priority_sources:
                for option in options:
                    if priority_source in option["source"].lower():
                        return ConsolidationDecision(
                            field_name=field_name,
                            selected_value=option["value"],
                            source_document=option["source"],
                            confidence=0.8,
                            reasoning=f"Seleccionado de {option['source']} por regla de prioridad",
                            alternatives_considered=options,
                        )

        from collections import Counter
        value_counts = Counter(opt["value"] for opt in options)
        most_common_value = value_counts.most_common(1)[0][0]
        source = next(opt["source"] for opt in options if opt["value"] == most_common_value)

        return ConsolidationDecision(
            field_name=field_name,
            selected_value=most_common_value,
            source_document=source,
            confidence=0.6,
            reasoning=f"Valor más común entre {len(options)} documentos",
            alternatives_considered=options,
        )

    # ---------- Validación “segura” ----------

    async def _validate_with_ai_safe(
        self,
        consolidated_fields: Dict[str, Any],
        original_extractions: List[DocumentExtraction]
    ) -> Dict[str, Any]:
        """
        Validación final con IA **sin inventar**:
        - Solo puede normalizar/ajustar campos que YA tienen valor.
        - No debe crear valores nuevos para campos vacíos.
        - Si no hay ajustes, devuelve tal cual.
        """
        # Build prompt con reglas de NO invención
        prompt = self.prompt_builder.build_validation_prompt(
            consolidated_fields=consolidated_fields,
            original_extractions=[self._to_dict(e) for e in original_extractions]
        )
        guardrails = (
            "INSTRUCCIONES ESTRICTAS:\n"
            "- NO inventes valores para campos vacíos (None/null/\"\").\n"
            "- SOLO propone ajustes mínimos (normalización de formato, corrección obvia de OCR) "
            "para campos QUE YA TIENEN VALOR.\n"
            "- Responde en JSON válido con la clave 'adjustments' que contenga únicamente los "
            "campos a modificar. Si no hay cambios, usa 'adjustments': {}.\n"
        )
        full_prompt = f"{guardrails}\n\n{prompt}"

        try:
            response: ValidationResponse = await self.client.chat.completions.create(
                model=self.config.get_model_for_task("consolidation"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Valida coherencia y formato. No inventes datos faltantes. "
                            "Responde SOLO con JSON para ser parseado."
                        ),
                    },
                    {"role": "user", "content": full_prompt},
                ],
                response_model=ValidationResponse,
                temperature=0.0,
                max_tokens=1200,
            )

            adjustments = response.adjustments or {}
            if not adjustments:
                logger.info("Validación IA: sin ajustes.")
                return consolidated_fields

            # Aplicar únicamente a claves existentes con valor (doble protección)
            applied = 0
            for field, new_value in adjustments.items():
                if field in consolidated_fields and consolidated_fields[field] is not None:
                    logger.info(f"Ajuste IA -> {field}: '{consolidated_fields[field]}' → '{new_value}'")
                    consolidated_fields[field] = new_value
                    applied += 1

            if applied == 0:
                logger.info("Validación IA: ajustes ignorados (no aplicables).")
            return consolidated_fields

        except Exception as e:
            logger.error(f"Error en validación con IA: {e}. Se mantiene consolidado original.")
            return consolidated_fields

    # ---------- Utilitarios internos ----------

    def _group_by_field(self, extractions: List[DocumentExtraction]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Agrupa los valores extraídos por campo (solo valores no nulos).
        Devuelve: { field_name: [ {value, source, document_type}, ... ] }
        """
        field_groups: Dict[str, List[Dict[str, Any]]] = {}
        for extraction in extractions:
            # Se asume que extraction.extracted_fields es un dict
            for field_name, value in getattr(extraction, "extracted_fields", {}).items():
                if value is None:
                    continue
                field_groups.setdefault(field_name, []).append({
                    "value": value,
                    "source": getattr(extraction, "source_document", "desconocido"),
                    "document_type": getattr(extraction, "document_type", "desconocido"),
                })
        return field_groups

    def _build_context(self, extractions: List[DocumentExtraction]) -> str:
        """
        Construye un breve contexto para prompts (tipos de doc y conteo de campos con datos).
        """
        doc_types = [getattr(e, "document_type", "desconocido") for e in extractions]
        context_parts = [f"Documentos analizados: {', '.join(sorted(set(doc_types)))}"]

        field_counts: Dict[str, int] = {}
        for e in extractions:
            for field, value in getattr(e, "extracted_fields", {}).items():
                if value is not None:
                    field_counts[field] = field_counts.get(field, 0) + 1

        context_parts.append(f"Campos con datos: {json.dumps(field_counts, ensure_ascii=False)}")
        return "\n".join(context_parts)

    def _get_relevant_examples(self, field_name: str) -> List[Dict[str, Any]]:
        """Devuelve hasta 3 ejemplos dorados relevantes, si existen."""
        examples = self.golden_examples.get(field_name) or []
        return examples[:3]

    def _load_golden_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Carga ejemplos de consolidaciones exitosas anteriores (puedes reemplazar con tu storage).
        """
        return {
            "numero_poliza": [
                {
                    "options": [
                        {"value": "AX-123-B", "source": "poliza.pdf"},
                        {"value": "AX123B", "source": "factura.pdf"},
                    ],
                    "correct_choice": "AX-123-B",
                    "reasoning": "El formato de póliza de 'poliza.pdf' es el estándar.",
                }
            ]
        }

    def _log_consolidation_metrics(self, result: ConsolidatedExtraction) -> None:
        """
        Registra métricas de la consolidación (soporta ConsolidatedFields o dict).
        """
        cf = result.consolidated_fields
        if isinstance(cf, ConsolidatedFields):
            values = cf.model_dump()
        elif isinstance(cf, dict):
            values = cf
        else:
            try:
                values = dict(cf)  # último recurso
            except Exception:
                values = {}

        fields_filled = sum(1 for v in values.values() if v is not None)
        total_fields = len(values)
        conflicts = len(result.conflicts_resolved or [])
        avg_confidence = (
            sum((result.confidence_scores or {}).values()) / max(len(result.confidence_scores or {}), 1)
        )

        logger.info(
            "Consolidación completada | Caso: %s | Campos: %s/%s | Conflictos: %s | Confianza prom.: %.2f%%",
            result.case_id, fields_filled, total_fields, conflicts, avg_confidence * 100.0
        )

    def _required_fields(self) -> List[str]:
        """
        Fuente de verdad para los nombres de campos requeridos.
        Prioriza el modelo Pydantic, con fallback a la config.
        """
        try:
            # Pydantic v2
            return list(ConsolidatedFields.model_fields.keys())
        except Exception:
            # Fallback si fuese Pydantic v1 o estructura legacy
            return getattr(self.config, "REQUIRED_FIELDS", [])

    @staticmethod
    def _to_dict(obj: Any) -> Dict[str, Any]:
        """
        Convierte modelos pydantic u objetos dataclass a dict sin romper si no implementan .model_dump().
        """
        for attr in ("model_dump", "dict"):
            if hasattr(obj, attr):
                try:
                    return getattr(obj, attr)()
                except Exception:
                    pass
        # Fall back naive
        try:
            return obj.__dict__
        except Exception:
            return {}
