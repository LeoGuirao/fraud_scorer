# src/fraud_scorer/ai_extractors/ai_consolidator.py

"""
AIConsolidator: Consolida extracciones múltiples usando razonamiento de IA,
evita validación cuando no hay datos (cortocircuito) y nunca inventa valores.
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel, Field

from fraud_scorer.settings import ExtractionConfig, FieldPriority, get_model_for_task  # FieldPriority puede usarse en reglas
from fraud_scorer.models.extraction import (
    DocumentExtraction,
    ConsolidatedExtraction,
    ConsolidatedFields,   # << importante: usar el modelo pydantic para los campos
    ExtractionBatch
)
from fraud_scorer.prompts.consolidation_prompts import ConsolidationPromptBuilder
from fraud_scorer.utils.validators import FieldValidator  # Nuevo validador
from fraud_scorer.analyzers.correlation.utils.normalization import (
    normalize_decimal_as_str,
    normalize_date,
)

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

    TIPO_SINIESTRO_CANONICAL = {
        "robo total": "Robo de Bulto por Entero",
        "robo total del embarque": "Robo de Bulto por Entero",
        "robo total del embarque por asalto": "Robo de Bulto por Entero",
        "robo de mercancía": "Robo de Bulto por Entero",
        "robo de bulto": "Robo de Bulto por Entero",
        "robo de bulto por entero": "Robo de Bulto por Entero",
        "robo de bulto por entero (robo de mercancía)": "Robo de Bulto por Entero",
        "robo total / parcial": "Robo de Bulto por Entero",
        "robo parcial": "Robo Parcial",
        "robo parcial de mercancía": "Robo Parcial",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.client = instructor.patch(
            AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        )
        self.config = ExtractionConfig()
        self.prompt_builder = ConsolidationPromptBuilder()
        self.validator = FieldValidator()  # Nuevo validador
        self.golden_examples = self._load_golden_examples()
        
        # Configuración del sistema de extracción guiada
        self.field_mapping = self.config.DOCUMENT_FIELD_MAPPING
        self.document_priorities = self.config.DOCUMENT_PRIORITIES
        # Contexto de póliza y prioridades personalizadas
        self.policy_context: Optional[str] = None
        self.custom_priorities: Dict[str, List[str]] = {}
        
        logger.info("AIConsolidator inicializado con Sistema de Extracción Guiada")

    def set_policy_context(self, policy_type: str) -> None:
        """Establece contexto de tipo de póliza y ajusta prioridades si aplica."""
        self.policy_context = policy_type
        if policy_type == "HDI_EN_MI_CASA":
            # Para HDI: tipo_siniestro desde informe final; lugar_hechos puede venir de póliza
            self.custom_priorities = {
                "tipo_siniestro": ["informe_final_del_ajustador"],
                "lugar_hechos": ["poliza_de_la_aseguradora", "informe_preliminar_del_ajustador"],
                "nombre_asegurado": ["poliza_de_la_aseguradora", "informe_preliminar_del_ajustador", "informe_final_del_ajustador"],
            }
            logger.info("Prioridades personalizadas activadas para HDI EN MI CASA")

    # ---------- API principal ----------

    async def consolidate_extractions(
        self,
        extractions: List[DocumentExtraction],
        case_id: str,
        use_advanced_reasoning: bool = True,
        guided_mode: bool = True,  # Añadir parámetro de modo guiado
        *,
        complexity: str = "normal",
        allow_escalation: bool = True,
    ) -> ConsolidatedExtraction:
        """
        Consolida múltiples extracciones en un resultado final.

        - Cortocircuita si `extractions` está vacío (evita alucinaciones).
        - Solo valida con IA si existe al menos un valor no-nulo.
        """
        logger.info(f"Consolidando {len(extractions)} extracciones para caso {case_id}")
        if guided_mode:
            logger.info("🛡️ Modo guiado activado: aplicando restricciones estrictas")

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

            # Aplicar prioridades personalizadas si están configuradas
            if self.custom_priorities and field_name in self.custom_priorities:
                preferred_docs = self.custom_priorities[field_name]
                for doc_type in preferred_docs:
                    for opt in options:
                        if opt.get("document_type") == doc_type and opt.get("value") is not None:
                            consolidated_fields_dict[field_name] = opt["value"]
                            consolidation_sources[field_name] = f"Prioridad por contexto ({doc_type})"
                            confidence_scores[field_name] = 0.95
                            # Pasar a siguiente campo (omitimos resolución de conflicto)
                            break
                    if field_name in consolidated_fields_dict and consolidated_fields_dict[field_name] is not None:
                        break
                if field_name in consolidated_fields_dict and consolidated_fields_dict[field_name] is not None:
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
                    all_extractions=extractions,
                    guided_mode=guided_mode,
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

        # ==========================================================
        #   NUEVO: Extracción de emergencia si todo vino vacío
        # ==========================================================
        if not has_any_value and extractions:
            logger.warning("EMERGENCIA: todos los campos llegaron vacíos; activando extracción de respaldo")
            logger.warning("Documentos procesados en este lote: %s", len(extractions))
            logger.warning(
                "Revisar los logs de AIFieldExtractor para diagnosticar el problema de origen"
            )

            import re

            for extraction in extractions:
                meta = getattr(extraction, "extraction_metadata", None)
                text = ""
                if isinstance(meta, dict):
                    # convenciones más comunes
                    for key in ("raw_text", "text", "raw_text_snippet"):
                        candidate = meta.get(key)
                        if candidate:
                            text = candidate
                            break

                if not text:
                    continue  # no hay material para raspar

                src = getattr(extraction, "source_document", "desconocido")

                # Número de póliza (ej. "Póliza ABC-123-456")
                if not consolidated_fields_dict.get("numero_poliza"):
                    m = re.search(r"[Pp][óo]liza.*?([A-Z0-9\-]{5,})", text)
                    if m:
                        consolidated_fields_dict["numero_poliza"] = m.group(1).strip()
                        consolidation_sources["numero_poliza"] = f"emergencia:{src}"
                        confidence_scores["numero_poliza"] = max(
                            confidence_scores.get("numero_poliza", 0.0), 0.3
                        )

                # Nombre del asegurado (ej. "Nombre: FULANO DE TAL")
                if not consolidated_fields_dict.get("nombre_asegurado"):
                    m = re.search(r"[Nn]ombre:?\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s,\.]+)", text)
                    if m:
                        consolidated_fields_dict["nombre_asegurado"] = m.group(1).strip(" ,.")
                        consolidation_sources["nombre_asegurado"] = f"emergencia:{src}"
                        confidence_scores["nombre_asegurado"] = max(
                            confidence_scores.get("nombre_asegurado", 0.0), 0.2
                        )

                # Monto reclamación (ej. "$ 123,456.78")
                if not consolidated_fields_dict.get("monto_reclamacion"):
                    m = re.search(r"\$\s*([\d\.,]+)", text)
                    if m:
                        consolidated_fields_dict["monto_reclamacion"] = m.group(1).strip()
                        consolidation_sources["monto_reclamacion"] = f"emergencia:{src}"
                        confidence_scores["monto_reclamacion"] = max(
                            confidence_scores.get("monto_reclamacion", 0.0), 0.2
                        )

            # Recalcular si ya tenemos algo tras la emergencia
            has_any_value = any(v is not None for v in consolidated_fields_dict.values())

        # Heurísticas determinísticas adicionales para campos vacíos
        self._apply_post_consolidation_fallbacks(
            consolidated_fields_dict,
            consolidation_sources,
            confidence_scores,
            extractions,
        )

        self._enforce_field_consistency(
            consolidated_fields_dict,
            consolidation_sources,
            confidence_scores,
        )

        has_any_value = any(v is not None for v in consolidated_fields_dict.values())

        # Validación con IA (solo si ya hay algo)
        if use_advanced_reasoning and has_any_value:
            consolidated_fields_dict = await self._validate_with_ai_safe(
                consolidated_fields_dict,
                extractions,
                guided_mode=guided_mode,
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
        all_extractions: List[DocumentExtraction],
        guided_mode: bool = True,
    ) -> ConsolidationDecision:
        """
        Usa IA (estructura) para resolver conflictos de un campo.
        """
        prompt = self.prompt_builder.build_conflict_resolution_prompt(
            field_name=field_name,
            options=options,
            field_rules=self.config.FIELD_SOURCE_RULES.get(field_name, []),
            golden_examples=self._get_relevant_examples(field_name),
            context=self._build_context(all_extractions),
            guided_mode=guided_mode
        )

        try:
            model_name = get_model_for_task("consolidation")
            params = self.config.get_openai_params("consolidation")
            kwargs = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Eres un experto ajustador de seguros con 20 años de experiencia. "
                            "Selecciona el mejor valor entre opciones dadas. Explica brevemente."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_model": ConsolidationDecision,
                "max_completion_tokens": params.get("max_completion_tokens", 1500),
            }
            if "timeout" in params:
                kwargs["timeout"] = params["timeout"]
            response: ConsolidationDecision = await self.client.chat.completions.create(**kwargs)
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
        Reglas determinísticas como respaldo.
        Usa prioridades del documento y luego valor más común.
        """
        # Primero intentar resolver por prioridad del documento
        # (las opciones ya vienen ordenadas por prioridad)
        if options and "priority" in options[0]:
            best_option = options[0]  # Ya está ordenado por prioridad
            return ConsolidationDecision(
                field_name=field_name,
                selected_value=best_option["value"],
                source_document=best_option["source"],
                confidence=0.9,
                reasoning=f"Seleccionado de {best_option['source']} ({best_option['document_type']}) por mayor prioridad",
                alternatives_considered=options,
            )
        
        # Fallback: usar reglas de fuente anteriores
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
                            reasoning=f"Seleccionado de {option['source']} por regla de prioridad legacy",
                            alternatives_considered=options,
                        )

        # Último recurso: valor más común
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
        original_extractions: List[DocumentExtraction],
        guided_mode: bool = True,
        complexity: str = "normal",
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
            original_extractions=[self._to_dict(e) for e in original_extractions],
            guided_mode=guided_mode
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
            model_name = get_model_for_task("consolidation")
            params = self.config.get_openai_params("consolidation")
            kwargs = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Valida coherencia y formato. No inventes datos faltantes. "
                            "Responde SOLO con JSON para ser parseado."
                        ),
                    },
                    {"role": "user", "content": full_prompt},
                ],
                "response_model": ValidationResponse,
                "max_completion_tokens": params.get("max_completion_tokens", 1800),
            }
            if "timeout" in params:
                kwargs["timeout"] = params["timeout"]
            response: ValidationResponse = await self.client.chat.completions.create(**kwargs)

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

    def _group_by_field(self, extractions: List[DocumentExtraction]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Agrupa los valores extraídos por campo (solo valores no nulos).
        Aplica filtrado según documentos autorizados y prioridades.
        Devuelve: { field_name: [ {value, source, document_type, priority}, ... ] }
        """
        field_groups: Dict[str, List[Dict[str, Any]]] = {}
        
        for extraction in extractions:
            doc_type = getattr(extraction, "document_type", "desconocido")
            
            # Obtener campos permitidos para este tipo de documento
            allowed_fields = self.field_mapping.get(doc_type, None)
            
            # Se asume que extraction.extracted_fields es un dict
            for field_name, value in getattr(extraction, "extracted_fields", {}).items():
                if value is None:
                    continue
                
                # Aplicar filtro de campos permitidos
                if allowed_fields is not None and field_name not in allowed_fields:
                    logger.debug(f"Campo '{field_name}' ignorado de documento tipo '{doc_type}' (no autorizado)")
                    continue
                
                # Obtener prioridad del documento
                priority = self._get_document_priority(doc_type)
                
                field_groups.setdefault(field_name, []).append({
                    "value": value,
                    "source": getattr(extraction, "source_document", "desconocido"),
                    "document_type": doc_type,
                    "priority": priority
                })
        
        monto_priorities = {
            "facturas_comerciales_internacionales": 0,
            "factura_comercial_cfdi": 0,
            "conocimiento_de_embarque": 1,
            "guias_y_facturas": 1,
            "informe_final_del_ajustador": 2,
            "informe_preliminar_del_ajustador": 2,
            "poliza_de_la_aseguradora": 3,
            "poliza": 3,
            "carpeta_de_investigacion": 4,
            "denuncia_de_los_hechos": 4,
            "carta_de_reclamacion_formal_a_la_aseguradora": 5,
        }

        # Ordenar opciones por prioridad para cada campo
        for field_name, options in field_groups.items():
            if field_name == "monto_reclamacion":
                sorted_options = sorted(
                    options,
                    key=lambda x: (
                        monto_priorities.get(x["document_type"], 99),
                        x["priority"],
                    ),
                )
                has_informe_final = any(
                    opt["document_type"] == "informe_final_del_ajustador"
                    for opt in sorted_options
                )
                if has_informe_final:
                    field_groups[field_name] = [
                        opt for opt in sorted_options
                        if opt["document_type"] == "informe_final_del_ajustador"
                    ]
                else:
                    field_groups[field_name] = sorted_options
            elif field_name == "fecha_ocurrencia":
                fecha_priorities = {
                    "informe_final_del_ajustador": 0,
                    "informe_preliminar_del_ajustador": 1,
                    "denuncia_de_los_hechos": 2,
                    "denuncia": 2,
                    "carpeta_de_investigacion": 3,
                }
                sorted_options = sorted(
                    options,
                    key=lambda x: (
                        fecha_priorities.get(x["document_type"], 99),
                        x["priority"],
                    ),
                )
                has_informe_final = any(
                    opt["document_type"] == "informe_final_del_ajustador"
                    for opt in sorted_options
                )
                if has_informe_final:
                    field_groups[field_name] = [
                        opt for opt in sorted_options
                        if opt["document_type"] == "informe_final_del_ajustador"
                    ]
                else:
                    field_groups[field_name] = sorted_options
            elif field_name == "lugar_hechos":
                lugar_priorities = {
                    "informe_final_del_ajustador": 0,
                    "peritaje": 1,
                    "denuncia_de_los_hechos": 2,
                    "denuncia": 2,
                }
                sorted_options = sorted(
                    options,
                    key=lambda x: (
                        lugar_priorities.get(x["document_type"], 99),
                        x["priority"],
                    ),
                )
                has_informe_final = any(
                    opt["document_type"] == "informe_final_del_ajustador"
                    for opt in sorted_options
                )
                if has_informe_final:
                    field_groups[field_name] = [
                        opt for opt in sorted_options
                        if opt["document_type"] == "informe_final_del_ajustador"
                    ]
                else:
                    field_groups[field_name] = sorted_options
            else:
                field_groups[field_name] = sorted(
                    options,
                    key=lambda x: x["priority"]
                )

        return field_groups

    def _apply_post_consolidation_fallbacks(
        self,
        fields: Dict[str, Any],
        sources: Dict[str, str],
        confidences: Dict[str, float],
        extractions: List[DocumentExtraction],
    ) -> None:
        """Aplica heurísticas determinísticas cuando la IA deja campos vacíos."""

        if not extractions:
            return

        month_map = {
            "enero": "01",
            "febrero": "02",
            "marzo": "03",
            "abril": "04",
            "mayo": "05",
            "junio": "06",
            "julio": "07",
            "agosto": "08",
            "septiembre": "09",
            "setiembre": "09",
            "octubre": "10",
            "noviembre": "11",
            "diciembre": "12",
        }

        for extraction in extractions:
            meta = getattr(extraction, "extraction_metadata", {}) or {}
            raw_text = (
                meta.get("raw_text")
                or meta.get("text")
                or meta.get("raw_text_snippet")
                or ""
            )
            if not raw_text:
                continue

            doc_type = (getattr(extraction, "document_type", "") or "").lower()
            source_doc = getattr(extraction, "source_document", "desconocido")
            lowered = raw_text.lower()

            # Suma asegurada desde póliza
            if fields.get("suma_asegurada") in (None, "") and doc_type in {"poliza_de_la_aseguradora", "poliza"}:
                match = re.search(
                    r"(?:suma\s+asegurada|l[ií]mite\s+m[áa]ximo(?:\s+por\s+embarque|\s+de\s+responsabilidad)?)[:\s\$]*([\d.,]+)",
                    raw_text,
                    re.IGNORECASE,
                )
                if match:
                    value = normalize_decimal_as_str(match.group(1))
                    if value:
                        fields["suma_asegurada"] = value
                        sources["suma_asegurada"] = f"fallback:{source_doc}"
                        confidences["suma_asegurada"] = max(confidences.get("suma_asegurada", 0.0), 0.6)

            # Monto reclamación desde informes/pólizas
            if fields.get("monto_reclamacion") in (None, "") and doc_type in {
                "informe_final_del_ajustador",
                "informe_preliminar_del_ajustador",
                "poliza_de_la_aseguradora",
                "poliza",
            }:
                match = re.search(
                    r"(?:total\s+perdida|total\s+p[eé]rdida|monto\s+total\s+de\s+la\s+reclamaci[oó]n|reclamaci[oó]n\s+por\s+la\s+cantidad\s+de)[:\s\$]*([\d.,]+)",
                    raw_text,
                    re.IGNORECASE,
                )
                if match:
                    value = normalize_decimal_as_str(match.group(1))
                    if value:
                        fields["monto_reclamacion"] = value
                        sources["monto_reclamacion"] = f"fallback:{source_doc}"
                        confidences["monto_reclamacion"] = max(confidences.get("monto_reclamacion", 0.0), 0.55)

            # Tipo de siniestro desde informes o denuncias
            if fields.get("tipo_siniestro") in (None, "") and doc_type in {
                "informe_final_del_ajustador",
                "informe_preliminar_del_ajustador",
                "denuncia_de_los_hechos",
                "denuncia",
                "carpeta_de_investigacion",
            }:
                candidate = None
                match = re.search(
                    r"tipo\s+de\s+siniestro\s*[:\-]?\s*([A-ZÁÉÍÓÚÑa-z0-9 /\-]+)",
                    raw_text,
                    re.IGNORECASE,
                )
                if match:
                    candidate = match.group(1).split("\n")[0].strip(" .:")
                if not candidate:
                    for options in self.config.SINIESTRO_TYPES.values():
                        for option in options:
                            if option.lower() in lowered:
                                candidate = option
                                break
                        if candidate:
                            break
                if not candidate and "robo" in lowered:
                    candidate = "Robo de bulto por entero (Robo de mercancía)" if "bulto" in lowered else "Robo total"
                if candidate:
                    fields["tipo_siniestro"] = candidate[:120]
                    sources["tipo_siniestro"] = f"fallback:{source_doc}"
                    confidences["tipo_siniestro"] = max(confidences.get("tipo_siniestro", 0.0), 0.55)

            # Fecha de reclamación desde cartas
            if fields.get("fecha_reclamacion") in (None, "") and doc_type in {
                "carta_de_reclamacion_formal_a_la_aseguradora",
                "carta_de_reclamacion_formal_al_transportista",
            }:
                date_match = re.search(
                    r"(?:[A-ZÁÉÍÓÚÑ\s]+,?\s+)?a\s*(\d{1,2})\s+de\s+([A-Za-zÁÉÍÓÚÑ]+)\s+de\s+(\d{4})",
                    raw_text,
                    re.IGNORECASE,
                )
                if date_match:
                    day = date_match.group(1).zfill(2)
                    month_name = date_match.group(2).lower()
                    year = date_match.group(3)
                    month = month_map.get(month_name)
                    if month:
                        normalized = normalize_date(f"{day}/{month}/{year}")
                        if not normalized:
                            normalized = f"{year}-{month}-{day}"
                        fields["fecha_reclamacion"] = normalized
                        sources["fecha_reclamacion"] = f"fallback:{source_doc}"
                        confidences["fecha_reclamacion"] = max(confidences.get("fecha_reclamacion", 0.0), 0.6)

            # Ajustador desde informe final
            if fields.get("ajuste") in (None, "") and doc_type == "informe_final_del_ajustador":
                candidate = None
                for known in getattr(self.config, "RECOGNIZED_ADJUSTERS", []):
                    if known.lower() in lowered:
                        candidate = known
                        break
                if not candidate:
                    name_match = re.search(
                        r"(?:ajustador(?:es)?|firma ajustadora|empresa ajustadora)\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ\s&\.]+)",
                        raw_text,
                        re.IGNORECASE,
                    )
                    if name_match:
                        candidate = " ".join(name_match.group(1).split())[:120]
                if candidate:
                    fields["ajuste"] = candidate
                    sources["ajuste"] = f"fallback:{source_doc}"
                    confidences["ajuste"] = max(confidences.get("ajuste", 0.0), 0.55)

            # Conclusiones resumidas desde informe final
            if fields.get("conclusiones") in (None, "") and doc_type == "informe_final_del_ajustador":
                match = re.search(
                    r"Conclusi[oó]n(?:es)?\s*[:\-]?\s*(.+?)(?:\n\s*\n|$)",
                    raw_text,
                    re.IGNORECASE | re.DOTALL,
                )
                if match:
                    snippet = " ".join(match.group(1).split())[:240]
                    if snippet:
                        fields["conclusiones"] = snippet
                        sources["conclusiones"] = f"fallback:{source_doc}"
                        confidences["conclusiones"] = max(confidences.get("conclusiones", 0.0), 0.5)

            # Bien reclamado desde encabezados
            if fields.get("bien_reclamado") in (None, ""):
                match = re.search(
                    r"bien(?:es)?\s+afectad[oa]s?\s*[:\-]?\s*(.+?)(?:\n|$)",
                    raw_text,
                    re.IGNORECASE,
                )
                if match:
                    snippet = " ".join(match.group(1).split())[:120]
                    if snippet:
                        fields["bien_reclamado"] = snippet
                        sources["bien_reclamado"] = f"fallback:{source_doc}"
                        confidences["bien_reclamado"] = max(confidences.get("bien_reclamado", 0.0), 0.5)

    def _enforce_field_consistency(
        self,
        fields: Dict[str, Any],
        sources: Dict[str, str],
        confidences: Dict[str, float],
    ) -> None:
        """Aplica verificaciones finales de formato y consistencia temporal."""

        numero_siniestro = fields.get("numero_siniestro")
        if numero_siniestro:
            digits = ''.join(filter(str.isdigit, str(numero_siniestro)))
            if len(digits) != 14:
                logger.warning(
                    "Numero de siniestro inválido (%s). Se descarta para evitar confusiones.",
                    numero_siniestro,
                )
                fields["numero_siniestro"] = None
                sources.pop("numero_siniestro", None)
                confidences.pop("numero_siniestro", None)

        fecha_ocurrencia = fields.get("fecha_ocurrencia")
        fecha_reclamacion = fields.get("fecha_reclamacion")
        if fecha_ocurrencia and fecha_reclamacion:
            dt_ocurrencia = self._parse_date_str(fecha_ocurrencia)
            dt_reclamacion = self._parse_date_str(fecha_reclamacion)
            if dt_ocurrencia and dt_reclamacion and dt_reclamacion < dt_ocurrencia:
                logger.warning(
                    "Fecha de reclamación %s anterior a ocurrencia %s. Se elimina para revisión.",
                    fecha_reclamacion,
                    fecha_ocurrencia,
                )
                fields["fecha_reclamacion"] = None
                sources.pop("fecha_reclamacion", None)
                confidences.pop("fecha_reclamacion", None)

        tipo_siniestro = fields.get("tipo_siniestro")
        if isinstance(tipo_siniestro, str) and tipo_siniestro.strip():
            normalized = self.TIPO_SINIESTRO_CANONICAL.get(tipo_siniestro.strip().lower())
            if normalized and normalized != tipo_siniestro:
                fields["tipo_siniestro"] = normalized

    @staticmethod
    def _parse_date_str(value: Any) -> Optional[datetime]:
        if not value:
            return None
        text = str(value).strip()
        candidates = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")
        for fmt in candidates:
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue
        return None

    def _get_document_priority(self, doc_type: str) -> int:
        """
        Obtiene la prioridad de un tipo de documento.
        Menor número = mayor prioridad.
        """
        if doc_type in self.document_priorities:
            val = self.document_priorities[doc_type]
            try:
                # Si es un Enum (FieldPriority), usar .value
                return val.value  # type: ignore[attr-defined]
            except Exception:
                # Si ya es un int, devolver directo
                if isinstance(val, int):
                    return val
                # Cualquier otro caso, prioridad baja
                return 99
        return 99  # Prioridad más baja para documentos desconocidos

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
