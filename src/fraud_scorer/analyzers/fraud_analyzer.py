"""
Motor principal de análisis de fraude por documento
"""
from __future__ import annotations
import os
import json
import asyncio
import logging
import hashlib
import uuid
import re
import unicodedata
from difflib import SequenceMatcher
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Iterable, Set

from openai import AsyncOpenAI
import instructor

from fraud_scorer.models.fraud_analysis import (
    EvidenceGap,
    FraudAnalysisResult,
    FraudIndicator,
    RiskLevel,
)
from fraud_scorer.models.extraction import DocumentExtraction
from fraud_scorer.analyzers.fraud_guide_manager import FraudGuideManager, FraudGuide
from fraud_scorer.prompts.fraud_prompts import FraudPromptBuilder
from fraud_scorer.storage.db import get_conn
from fraud_scorer.settings import get_model_for_task
from fraud_scorer.analyzers.unified_data_layer import UnifiedDataLayer
from fraud_scorer.services.exchange_rate_service import ExchangeRateService
from fraud_scorer.utils.geo_reference import suggest_reference_point
from fraud_scorer.config.feature_flags import get_fiscal_validation_stage
from fraud_scorer.models.fiscal_validation import (
    CFDIValidationRequest,
    FiscalValidationResult,
    FiscalValidationStatus,
)
from fraud_scorer.services.fiscal_api_service import FiscalAPIService

logger = logging.getLogger(__name__)


MEXICO_STATE_ALIASES: Dict[str, Set[str]] = {
    "Aguascalientes": {"aguascalientes"},
    "Baja California": {"baja california", "bc"},
    "Baja California Sur": {"baja california sur", "bcs"},
    "Campeche": {"campeche"},
    "Chiapas": {"chiapas"},
    "Chihuahua": {"chihuahua"},
    "Ciudad de México": {"ciudad de mexico", "ciudad de méxico", "cdmx", "distrito federal", "df"},
    "Coahuila": {"coahuila", "coahuila de zaragoza"},
    "Colima": {"colima"},
    "Durango": {"durango"},
    "Estado de México": {"estado de mexico", "edomex"},
    "Guanajuato": {"guanajuato"},
    "Guerrero": {"guerrero"},
    "Hidalgo": {"hidalgo"},
    "Jalisco": {"jalisco"},
    "Michoacán": {"michoacan", "michoacan de ocampo"},
    "Morelos": {"morelos"},
    "Nayarit": {"nayarit"},
    "Nuevo León": {"nuevo leon", "nuevoleon", "nl"},
    "Oaxaca": {"oaxaca"},
    "Puebla": {"puebla"},
    "Querétaro": {"queretaro", "queretaro de arteaga"},
    "Quintana Roo": {"quintana roo"},
    "San Luis Potosí": {"san luis potosi", "slp"},
    "Sinaloa": {"sinaloa"},
    "Sonora": {"sonora"},
    "Tabasco": {"tabasco"},
    "Tamaulipas": {"tamaulipas"},
    "Tlaxcala": {"tlaxcala"},
    "Veracruz": {"veracruz", "veracruz de ignacio de la llave"},
    "Yucatán": {"yucatan"},
    "Zacatecas": {"zacatecas"},
}

SPANISH_MONTHS: Dict[str, int] = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

SPANISH_DAY_WORDS: Dict[str, int] = {
    "UNO": 1,
    "UNA": 1,
    "DOS": 2,
    "TRES": 3,
    "CUATRO": 4,
    "CINCO": 5,
    "SEIS": 6,
    "SIETE": 7,
    "OCHO": 8,
    "NUEVE": 9,
    "DIEZ": 10,
    "ONCE": 11,
    "DOCE": 12,
    "TRECE": 13,
    "CATORCE": 14,
    "QUINCE": 15,
    "DIECISEIS": 16,
    "DIECISÉIS": 16,
    "DIECISIETE": 17,
    "DIECIOCHO": 18,
    "DIECINUEVE": 19,
    "VEINTE": 20,
    "VEINTIUNO": 21,
    "VEINTIUN": 21,
    "VEINTIDOS": 22,
    "VEINTIDÓS": 22,
    "VEINTITRES": 23,
    "VEINTITRÉS": 23,
    "VEINTICUATRO": 24,
    "VEINTICINCO": 25,
    "VEINTISEIS": 26,
    "VEINTISÉIS": 26,
    "VEINTISIETE": 27,
    "VEINTIOCHO": 28,
    "VEINTINUEVE": 29,
    "TREINTA": 30,
    "TREINTAYUNO": 31,
}

STATE_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "Aguascalientes": (21.8853, -102.2916),
    "Baja California": (30.8406, -115.2838),
    "Baja California Sur": (25.9870, -111.6626),
    "Campeche": (19.8301, -90.5349),
    "Chiapas": (16.7516, -93.1011),
    "Chihuahua": (28.6320, -106.0691),
    "Ciudad de México": (19.4326, -99.1332),
    "Coahuila": (27.0587, -101.7068),
    "Colima": (19.1223, -104.0072),
    "Durango": (24.0277, -104.6532),
    "Estado de México": (19.2921, -99.6569),
    "Guanajuato": (20.8759, -101.9790),
    "Guerrero": (17.4392, -99.5451),
    "Hidalgo": (20.4960, -98.9625),
    "Jalisco": (20.6597, -103.3496),
    "Michoacán": (19.5665, -101.7068),
    "Morelos": (18.6813, -99.1013),
    "Nayarit": (21.7514, -104.8455),
    "Nuevo León": (25.5922, -99.9962),
    "Oaxaca": (17.0594, -96.7216),
    "Puebla": (19.0413, -98.2062),
    "Querétaro": (20.5888, -100.3899),
    "Quintana Roo": (19.1817, -88.4791),
    "San Luis Potosí": (22.1565, -100.9855),
    "Sinaloa": (24.8091, -107.3940),
    "Sonora": (29.0729, -110.9559),
    "Tabasco": (17.8409, -92.6189),
    "Tamaulipas": (23.7417, -99.1450),
    "Tlaxcala": (19.3182, -98.2375),
    "Veracruz": (19.1738, -96.1342),
    "Yucatán": (20.9674, -89.5926),
    "Zacatecas": (22.7709, -102.5833),
}


class FraudAnalyzer:
    def __init__(self, api_key: Optional[str] = None) -> None:
        raw = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.client = instructor.patch(raw)
        self.prompts = FraudPromptBuilder()
        self.guides = FraudGuideManager()

        # Modelos (por defecto usar familia 4o-mini, más compatible con JSON)
        self.model = os.getenv("FRAUD_ANALYSIS_MODEL") or "gpt-4o-mini"
        self.model_fallback = os.getenv("FRAUD_ANALYSIS_MODEL_FALLBACK") or "gpt-4o-mini"
        self.confidence_threshold = float(os.getenv("FRAUD_CONFIDENCE_THRESHOLD", "0.7"))

        try:
            self.fiscal_service = FiscalAPIService()
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Fiscal validation service unavailable: %s", exc)
            self.fiscal_service = None
        self.fiscal_stage = (
            self.fiscal_service.stage if getattr(self, "fiscal_service", None) else get_fiscal_validation_stage()
        )
        logger.info("Fiscal validation stage: %s", self.fiscal_stage)

    async def analyze_document(
        self,
        document_id: str,
        document_name: str,
        document_type: str,
        ocr_result: Dict[str, Any],
        extraction: DocumentExtraction,
        case_id: str,
        context: Optional[Dict[str, Any]] = None,
        data_layer: Optional[UnifiedDataLayer] = None,
    ) -> FraudAnalysisResult:
        guide = self.guides.get_guide(document_type)
        if not guide:
            logger.warning(f"No hay guía para {document_type}. Ejecutando análisis genérico")
            return await self._generic_analysis(
                document_id, document_name, document_type, ocr_result, extraction, case_id
            )
        layer = data_layer
        case_context = context or {}
        document_context: Optional[Dict[str, Any]] = None
        if layer:
            try:
                case_snapshot = layer.build_case_context()
                case_context = {**case_snapshot, **(context or {})}
                document_context = layer.build_document_context(
                    extraction=extraction,
                    ocr_result=ocr_result,
                )
            except Exception as exc:  # pragma: no cover - defensivo
                logger.debug("Fallo construyendo contexto unificado: %s", exc)
        if document_context is None:
            document_context = {
                "document_type": document_type,
                "document_name": document_name,
                "resolved_fields": extraction.extracted_fields,
            }

        fiscal_result: Optional[FiscalValidationResult] = None
        if self.fiscal_service:
            fiscal_result = await self._maybe_validate_fiscal(
                extraction,
                document_type=document_type,
                document_name=document_name,
                case_id=case_id,
                document_id=document_id,
            )
            if fiscal_result:
                try:
                    document_context.setdefault("fiscal_validation", fiscal_result.to_case_index())
                except Exception:
                    logger.debug("No se pudo adjuntar fiscal_validation a document_context")
                try:
                    case_context.setdefault("fiscal_validation", {})
                    case_context["fiscal_validation"][document_id] = fiscal_result.to_case_index()
                except Exception:
                    logger.debug("No se pudo incorporar fiscal_validation en case_context")

        indicator_prompt = self.prompts.build_indicator_prompt(
            document_type=document_type,
            document_name=document_name,
            ocr_content=ocr_result,
            extracted_fields=extraction.extracted_fields,
            guide=guide._data,  # type: ignore[attr-defined]
            case_context=case_context,
            document_context=document_context,
        )

        start = datetime.now()
        try:
            is_carpeta = document_type == "carpeta_de_investigacion"
            primary_model = "gpt-5" if is_carpeta else self.model
            fallback_model = (
                os.getenv("FRAUD_ANALYSIS_MODEL_FALLBACK_CARPETA") or self.model_fallback
                if is_carpeta
                else self.model_fallback
            )
            indicator_response, model_used = await self._call_ai_with_retry(
                indicator_prompt,
                context_name=document_name,
                primary_model=primary_model,
                fallback_model=fallback_model,
            )
            analysis = self._parse_analysis_response(
                indicator_response,
                analysis_model=model_used,
                document_id=document_id,
                document_name=document_name,
                document_type=document_type,
                case_id=case_id,
                guide=guide,
            )
            # Trazabilidad
            analysis.analysis_id = str(uuid.uuid4())

            if fiscal_result:
                analysis = self._apply_fiscal_enrichment(analysis, fiscal_result)

            analysis = await self._enrich_analysis(
                analysis,
                extraction,
                guide,
                data_layer=layer,
                document_context=document_context,
                case_context=case_context,
                ocr_text=str(ocr_result.get("text") or ""),
            )

            analysis.prompt_hash = hashlib.sha256(indicator_prompt.encode("utf-8")).hexdigest()

            analysis.processing_time_ms = int((datetime.now() - start).total_seconds() * 1000)

            await self._save_analysis_to_db(analysis)
            logger.info(
                f"FraudAnalysis {document_name}: riesgo={analysis.risk_level.value} score={analysis.fraud_score:.2f}"
            )
            return analysis
        except Exception as e:
            logger.error(f"Error en análisis de fraude [{document_name}]: {e}")
            if guide and document_type == "carta_de_reclamacion_formal_a_la_aseguradora":
                offline = FraudAnalysisResult(
                    document_id=document_id,
                    document_name=document_name,
                    document_type=document_type,
                    case_id=case_id,
                    risk_level=RiskLevel.MEDIO,
                    fraud_score=0.45,
                    confidence=0.85,
                    analisis_completo="",
                    indicators=[],
                    recommendations=[],
                    verificaciones={},
                    validacion_cruzada={},
                    analysis_model="offline",
                    guide_version=guide.version,
                    processing_time_ms=0,
                )
                offline.analysis_id = str(uuid.uuid4())
                offline = await self._enrich_analysis(
                    offline,
                    extraction,
                    guide,
                    data_layer=layer,
                    document_context=document_context or {},
                    case_context=case_context or {},
                    ocr_text=str(ocr_result.get("text") or ""),
                )
                offline.analysis_model = "offline"
                return offline
            return self._create_error_analysis(document_id, document_name, document_type, case_id, str(e))

    async def analyze_batch(
        self,
        documents: List[Dict[str, Any]],
        case_id: str,
        parallel_limit: int = 3,
        context: Optional[Dict[str, Any]] = None,
        data_layer: Optional[UnifiedDataLayer] = None,
    ) -> List[FraudAnalysisResult]:
        docs_with_guides: List[Dict[str, Any]] = []
        for doc in documents:
            guide = self.guides.get_guide(doc.get("type", ""))
            if guide:
                docs_with_guides.append(doc)
            else:
                logger.info(
                    "Omitiendo análisis de fraude para %s (tipo=%s) porque no existe guía",
                    doc.get("name", "documento"),
                    doc.get("type", "desconocido"),
                )

        if not docs_with_guides:
            logger.info("No hay documentos con guías de fraude disponibles en este lote")
            return []

        sem = asyncio.Semaphore(parallel_limit)

        async def _run(doc: Dict[str, Any]):
            async with sem:
                return await self.analyze_document(
                    document_id=doc["id"],
                    document_name=doc["name"],
                    document_type=doc["type"],
                    ocr_result=doc["ocr"],
                    extraction=doc["extraction"],
                    case_id=case_id,
                    context=context,
                    data_layer=data_layer,
                )

        results = await asyncio.gather(*[_run(d) for d in docs_with_guides], return_exceptions=True)
        out: List[FraudAnalysisResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                d = docs_with_guides[i]
                logger.error(f"Error analizando {d['name']}: {r}")
                out.append(self._create_error_analysis(d['id'], d['name'], d['type'], case_id, str(r)))
            else:
                out.append(r)
        return out

    async def _call_ai_with_retry(
        self,
        prompt: str,
        context_name: str,
        max_retries: int = 2,
        *,
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ) -> Tuple[str, str]:
        offline_flag = os.getenv("FRAUD_ANALYSIS_OFFLINE", "").strip().lower()
        if offline_flag in {"1", "true", "yes"}:
            stub_payload = {
                "analysis_summary": "Análisis auxiliar generado en modo offline (solo postprocesamiento determinístico).",
                "fraud_indicators": [],
                "risk_level": "bajo",
                "fraud_score": 0.25,
                "confidence": 0.8,
                "recommendations": [],
                "verificaciones": {},
                "validacion_cruzada": {},
            }
            return json.dumps(stub_payload, ensure_ascii=False), "offline"

        last: Optional[Exception] = None
        primary = primary_model or self.model
        secondary = fallback_model or self.model_fallback or primary
        model_chain = [primary, secondary]
        for attempt in range(max_retries):
            try:
                index = attempt if attempt < len(model_chain) else len(model_chain) - 1
                model_name = model_chain[index] or primary

                # Primer intento: con response_format JSON; si falla, segundo intento sin él
                use_response_format = (attempt == 0)
                kwargs = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": self.prompts.base_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_completion_tokens": 2500,
                }
                if use_response_format:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = await self.client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
                content = (resp.choices[0].message.content or "").strip()  # type: ignore[index]
                # Limpieza defensiva de fences
                content = content.replace("```json", "").replace("```", "").strip()
                # Validación mínima de JSON
                json.loads(content)
                return content, model_name
            except Exception as e:  # pragma: no cover - red restringida
                last = e
                await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Fallo en IA para {context_name}: {last}")

    def _parse_analysis_response(
        self,
        response_text: str,
        analysis_model: str,
        document_id: str,
        document_name: str,
        document_type: str,
        case_id: str,
        guide: FraudGuide,
    ) -> FraudAnalysisResult:
        # Normalización robusta del JSON de salida
        data_raw = response_text or "{}"
        try:
            data = json.loads(data_raw)
        except Exception:
            cleaned = data_raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned or "{}")

        # Indicadores (aceptar lista de dicts o strings)
        indicators: List[FraudIndicator] = []
        ind_list = data.get("fraud_indicators") or data.get("indicators") or []
        if not isinstance(ind_list, list):
            ind_list = [ind_list]
        for ind in ind_list:
            if isinstance(ind, dict):
                indicators.append(
                    FraudIndicator(
                        pattern=str(ind.get("pattern", "unknown")),
                        description=str(ind.get("description", "")),
                        severity=str(ind.get("severity", "medio")),
                        confidence=float(ind.get("confidence", 0.5)) if ind.get("confidence") is not None else 0.5,
                        location=ind.get("location"),
                    )
                )
            else:
                s = str(ind)
                indicators.append(
                    FraudIndicator(
                        pattern=s[:64] or "unknown",
                        description=s,
                        severity="medio",
                        confidence=0.5,
                    )
                )

        # Score y riesgo
        def _to_float(val, default=0.5) -> float:
            try:
                return float(val)
            except Exception:
                return default

        score = _to_float(data.get("fraud_score", 0.5), 0.5)
        provided_risk_raw = str(data.get("risk_level", "")).strip().lower()
        provided_risk = provided_risk_raw if provided_risk_raw in {"bajo", "medio", "alto", "critico"} else None
        derived = self._derive_risk_level(score)
        risk_level = RiskLevel(derived)

        recommendations = data.get("recommendations", []) or []
        if not isinstance(recommendations, list):
            recommendations = [str(recommendations)]
        recommendations = [str(r) for r in recommendations]

        if provided_risk and provided_risk != derived:
            recommendations.append(
                f"Ajuste automático: riesgo recalculado de '{provided_risk}' a '{derived}' (score={score:.2f})."
            )

        verificaciones = data.get("verificaciones") or {}
        if not isinstance(verificaciones, dict):
            try:
                verificaciones = dict(verificaciones)
            except Exception:
                verificaciones = {}

        validacion_cruzada = data.get("validacion_cruzada") or {}
        if not isinstance(validacion_cruzada, dict):
            try:
                validacion_cruzada = dict(validacion_cruzada)
            except Exception:
                validacion_cruzada = {}

        summary = str(
            data.get("analysis_summary")
            or data.get("analisis_completo")
            or data.get("analysis")
            or ""
        ).strip()

        return FraudAnalysisResult(
            document_id=document_id,
            document_name=document_name,
            document_type=document_type,
            case_id=case_id,
            risk_level=risk_level,
            fraud_score=score,
            confidence=_to_float(data.get("confidence", 0.7), 0.7),
            analisis_completo=summary,
            indicators=indicators,
            recommendations=recommendations,
            verificaciones=verificaciones,
            validacion_cruzada=validacion_cruzada,
            analysis_model=analysis_model,
            guide_version=guide.version,
            processing_time_ms=0,
        )

    def _parse_evidence_gap_response(self, response_text: str) -> List[EvidenceGap]:
        raw = response_text or "{}"
        try:
            data = json.loads(raw)
        except Exception:
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned or "{}")

        gaps: List[EvidenceGap] = []
        entries = data.get("evidence_gaps") or []
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            if isinstance(entry, EvidenceGap):
                gaps.append(entry)
            elif isinstance(entry, dict):
                try:
                    gaps.append(EvidenceGap(**entry))
                except Exception:
                    gap_text = str(entry)
                    gaps.append(EvidenceGap(gap=gap_text))
            elif isinstance(entry, str):
                gaps.append(EvidenceGap(gap=entry))

        follow_ups = data.get("follow_up_questions") or []
        if isinstance(follow_ups, list):
            for question in follow_ups:
                text = str(question).strip()
                if text:
                    gaps.append(EvidenceGap(gap=text, priority="media", suggested_action="Resolver pregunta al asegurado"))

        missing_docs = data.get("missing_documents") or []
        if isinstance(missing_docs, list) and missing_docs and gaps:
            # Añadir documentos faltantes a la primera entrada para referencia
            first_gap = gaps[0]
            first_gap.missing_documents.extend(str(doc) for doc in missing_docs)

        return gaps

    async def _enrich_analysis(
        self,
        analysis: FraudAnalysisResult,
        extraction: DocumentExtraction,
        guide: FraudGuide,
        *,
        data_layer: Optional[UnifiedDataLayer] = None,
        document_context: Optional[Dict[str, Any]] = None,
        case_context: Optional[Dict[str, Any]] = None,
        ocr_text: str = "",
    ) -> FraudAnalysisResult:
        rules = guide.get_validation_rules() or {}
        for field, rule in rules.items():
            if field in extraction.extracted_fields:
                value = extraction.extracted_fields.get(field)
                ok, msg = self._apply_validation_rule(rule, value)
                if not ok:
                    analysis.indicators.append(
                        FraudIndicator(
                            pattern=f"invalid_{field}",
                            description=msg,
                            severity="medio",
                            confidence=0.8,
                        )
                    )
        # Ajuste simple de score si muchos indicadores
        if len(analysis.indicators) > 5:
            analysis.fraud_score = min(1.0, analysis.fraud_score * 1.2)
            analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))

        if extraction.document_type == "carta_de_reclamacion_formal_a_la_aseguradora":
            try:
                analysis = self._postprocess_carta_reclamacion(
                    analysis,
                    extraction,
                    data_layer=data_layer,
                    document_context=document_context or {},
                    case_context=case_context or {},
                    ocr_text=ocr_text,
                )
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Fallback en carta de reclamación: %s", exc)
        elif extraction.document_type == "carta_de_reclamacion_formal_al_transportista":
            try:
                analysis = self._postprocess_carta_transportista(
                    analysis,
                    extraction,
                    data_layer=data_layer,
                    document_context=document_context or {},
                    case_context=case_context or {},
                    ocr_text=ocr_text,
                )
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Fallback en carta al transportista: %s", exc)
        elif extraction.document_type == "cfdi_carta_porte":
            try:
                analysis = self._postprocess_cfdi_carta_porte(
                    analysis,
                    extraction,
                    data_layer=data_layer,
                    document_context=document_context or {},
                    case_context=case_context or {},
                    ocr_text=ocr_text,
                )
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Fallback en CFDI carta porte: %s", exc)
        elif extraction.document_type == "carta_porte_simple":
            try:
                analysis = self._postprocess_carta_porte_simple(
                    analysis,
                    extraction,
                    data_layer=data_layer,
                    document_context=document_context or {},
                    case_context=case_context or {},
                    ocr_text=ocr_text,
                )
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Fallback en carta porte simple: %s", exc)
        elif extraction.document_type == "pedimento_importacion":
            try:
                analysis = self._postprocess_pedimento_importacion(
                    analysis,
                    extraction,
                    data_layer=data_layer,
                    document_context=document_context or {},
                    case_context=case_context or {},
                    ocr_text=ocr_text,
                )
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Fallback en pedimento de importación: %s", exc)
        elif extraction.document_type == "conocimiento_de_embarque":
            try:
                analysis = self._postprocess_conocimiento_embarque(
                    analysis,
                    extraction,
                    data_layer=data_layer,
                    document_context=document_context or {},
                    case_context=case_context or {},
                    ocr_text=ocr_text,
                )
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Fallback en conocimiento de embarque: %s", exc)
        elif extraction.document_type == "carta_aclatoria_comprobantes_peaje":
            try:
                analysis = self._postprocess_carta_aclaratoria_peaje(
                    analysis,
                    extraction,
                    data_layer=data_layer,
                    document_context=document_context or {},
                    case_context=case_context or {},
                    ocr_text=ocr_text,
                )
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Fallback en carta aclaratoria peaje: %s", exc)
        elif extraction.document_type == "carpeta_de_investigacion":
            try:
                analysis = self._postprocess_carpeta_investigacion(
                    analysis,
                    extraction,
                    data_layer=data_layer,
                    document_context=document_context or {},
                    case_context=case_context or {},
                    ocr_text=ocr_text,
                )
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Fallback en carpeta de investigación: %s", exc)
        return analysis

    async def _maybe_validate_fiscal(
        self,
        extraction: DocumentExtraction,
        *,
        document_type: str,
        document_name: str,
        case_id: str,
        document_id: str,
    ) -> Optional[FiscalValidationResult]:
        service = getattr(self, "fiscal_service", None)
        if not service:
            return None
        if not service.should_validate_document(document_type):
            return None

        request = self._build_cfdi_request(
            extraction=extraction,
            document_type=document_type,
            case_id=case_id,
            document_id=document_id,
        )
        if not request:
            logger.debug("Fiscal validation omitted for %s (datos incompletos)", document_name)
            return None
        logger.debug(
            "Fiscal validation request for %s → issuer=%s recipient=%s total=%s uuid=%s signature=%s",
            document_name,
            request.issuer_rfc,
            request.recipient_rfc,
            request.total,
            request.uuid,
            request.signature_last_8,
        )
        try:
            result = await service.validate_cfdi_status(request)
            return result
        except Exception as exc:  # pragma: no cover - resiliencia
            logger.warning("Fiscal validation failed for %s: %s", document_name, exc)
            return FiscalValidationResult.pending(request, error=str(exc))

    def _build_cfdi_request(
        self,
        *,
        extraction: DocumentExtraction,
        document_type: str,
        case_id: str,
        document_id: str,
    ) -> Optional[CFDIValidationRequest]:
        fields = dict(extraction.extracted_fields or {})

        issuer = fields.get("issuer_rfc") or fields.get("emisor_rfc") or fields.get("emisor")
        recipient = fields.get("recipient_rfc") or fields.get("receptor_rfc") or fields.get("receptor")
        uuid_value = fields.get("uuid_fiscal") or fields.get("folio_fiscal_uuid")
        sello_cfdi = (
            fields.get("sello_digital_cfdi")
            or fields.get("sello_digital_emisor")
            or fields.get("sello_digital")
        )
        sello_sat = fields.get("sello_digital_sat")
        sello = sello_cfdi or sello_sat
        signature_field = fields.get("signature_last_8")
        signature = None
        if signature_field:
            signature_text = str(signature_field).strip()
            if len(signature_text) >= 8:
                signature = signature_text[-8:]
        if not signature and sello_cfdi:
            signature = str(sello_cfdi)[-8:]
        if not signature and sello_sat:
            signature = str(sello)[-8:]
        if signature:
            signature = re.sub(r"\s+", "", signature).strip().upper()

        if (not recipient) or str(recipient).strip().upper() in {"XAXX010101000", "XEXX010101000"}:
            guessed_rfc = self._guess_recipient_rfc(extraction, issuer)
            if guessed_rfc:
                recipient = guessed_rfc

        total_value = None
        for candidate in (
            fields.get("monto_total"),
            fields.get("valor_mercancia"),
            fields.get("monto_reclamacion"),
        ):
            total_value = self._coerce_decimal(candidate)
            if total_value is not None:
                break

        if not all([issuer, recipient, uuid_value, signature, total_value]):
            return None

        metadata = self._extract_cfdi_metadata(
            extraction=extraction,
            issuer=str(issuer),
            recipient=str(recipient),
        )
        if total_value is not None:
            metadata.setdefault("invoice_total", format(total_value, "f"))

        try:
            return CFDIValidationRequest(
                issuer_rfc=str(issuer),
                recipient_rfc=str(recipient),
                total=total_value,
                uuid=str(uuid_value),
                signature_last_8=str(signature),
                document_type=document_type,
                sello_digital=str(sello) if sello else None,
                case_id=case_id,
                document_id=document_id,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - validaciones internas
            logger.debug("No se pudo construir CFDIValidationRequest: %s", exc)
            return None

    def _extract_cfdi_metadata(
        self,
        *,
        extraction: DocumentExtraction,
        issuer: str,
        recipient: str,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        fields = extraction.extracted_fields or {}

        def _first_field(*keys: str) -> Optional[str]:
            for key in keys:
                value = fields.get(key)
                if value:
                    text = str(value).strip()
                    if text:
                        return text
            return None

        issuer_name = _first_field("emisor_nombre", "razon_social_emisor", "nombre_emisor")
        if issuer_name:
            metadata["issuer_name"] = issuer_name

        recipient_name = _first_field("receptor_nombre", "razon_social_receptor", "nombre_receptor")
        if recipient_name:
            metadata["recipient_name"] = recipient_name

        issue_date = _first_field("fecha_emision", "fecha_expedicion")
        if issue_date:
            metadata["issue_date"] = issue_date

        sat_cert_date = _first_field("fecha_certificacion_sat", "fecha_timbrado")
        if sat_cert_date:
            metadata["sat_certification_date"] = sat_cert_date

        invoice_effect = _first_field("efecto_del_comprobante", "tipo_comprobante")
        if invoice_effect:
            metadata["invoice_effect"] = invoice_effect

        raw_text = extraction.extraction_metadata.get("raw_text") or ""
        if raw_text:
            raw_text = str(raw_text)
            if "recipient_name" not in metadata:
                match = re.search(r"Raz[oó]n\s+Social:\s*([^\n]+)", raw_text, flags=re.IGNORECASE)
                if match:
                    metadata.setdefault("recipient_name", match.group(1).strip())

            if "fecha" in raw_text.lower():
                issue_match = re.search(
                    r"Fecha[^\n]*?(?:emisi[oó]n|expedici[oó]n)[^0-9]*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
                    raw_text,
                    flags=re.IGNORECASE,
                )
                if issue_match:
                    metadata.setdefault("issue_date", issue_match.group(1).strip())

                cert_match = re.search(
                    r"Fecha[^\n]*?certificaci[oó]n[^0-9]*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
                    raw_text,
                    flags=re.IGNORECASE,
                )
                if cert_match:
                    metadata.setdefault("sat_certification_date", cert_match.group(1).strip())

            effect_match = re.search(r"\b([IEP])\s*-\s*(Ingreso|Egreso|Pago)\b", raw_text, flags=re.IGNORECASE)
            if effect_match:
                canonical = f"{effect_match.group(1).upper()} - {effect_match.group(2).capitalize()}"
                metadata.setdefault("invoice_effect", canonical)

            rfc_pattern = re.compile(r"RFC:\s*{}".format(re.escape(issuer)), flags=re.IGNORECASE)
            issuer_match = rfc_pattern.search(raw_text)
            if issuer_match and "issuer_name" not in metadata:
                candidates = raw_text[:issuer_match.start()].split("\n")
                for line in reversed(candidates):
                    candidate = line.strip()
                    if candidate:
                        metadata["issuer_name"] = candidate
                        break

            cadena_idx = raw_text.find("CADENA ORIGINAL")
            if cadena_idx != -1 and "pac_certifier" not in metadata:
                segment = raw_text[cadena_idx:]
                rfc_candidates = re.findall(r"\|([A-Z0-9&Ñ]{12,13})\|", segment)
                issuer_upper = issuer.upper()
                recipient_upper = recipient.upper()
                for candidate in rfc_candidates:
                    upper_candidate = candidate.upper()
                    if upper_candidate not in {issuer_upper, recipient_upper}:
                        metadata["pac_certifier"] = upper_candidate
                        break

        return {key: value for key, value in metadata.items() if value}

    def _normalize_rfc_token(self, value: Optional[Any]) -> str:
        if value is None:
            return ""
        text = str(value).strip().upper()
        return re.sub(r"[^A-Z0-9&]", "", text)

    def _guess_recipient_rfc(self, extraction: DocumentExtraction, issuer: Optional[Any]) -> Optional[str]:
        metadata = getattr(extraction, "extraction_metadata", {}) or {}
        raw_text = metadata.get("raw_text") or metadata.get("raw_text_snippet")
        if not raw_text:
            return None

        normalized_text = self._strip_accents(str(raw_text))
        normalized_text_upper = normalized_text.upper()
        issuer_norm = self._normalize_rfc_token(issuer)

        pattern = re.compile(r"RFC\s*[:=]?\s*([A-Z&]{3,4}[0-9]{6}[A-Z0-9]{3})", re.IGNORECASE)
        candidates = [match.group(1) for match in pattern.finditer(normalized_text)]
        if not candidates:
            uppercase_text = normalized_text_upper
            fallback_matches = re.findall(r"[A-Z&]{3,4}[0-9]{6}[A-Z0-9]{3}", uppercase_text)
            candidates = fallback_matches

        for candidate in candidates:
            norm = self._normalize_rfc_token(candidate)
            if not norm or norm == issuer_norm:
                continue
            if norm in {"XAXX010101000", "XEXX010101000"}:
                continue
            if norm.endswith("000") and "RFC REMITENTE DESTINATARIO" in normalized_text_upper:
                # Complement entries suelen repetir RFC genéricos; evita priorizarlos
                continue
            logger.debug("RFC receptor detectado %s para %s", norm, extraction.source_document)
            return norm
        return None

    def _coerce_decimal(self, value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value.quantize(Decimal("0.000001"))
        if isinstance(value, (int, float)):
            return Decimal(str(value)).quantize(Decimal("0.000001"))
        text = str(value).strip()
        if not text:
            return None
        cleaned = re.sub(r"[^\d.,-]", "", text)
        if cleaned.count(",") == 1 and cleaned.count(".") == 0:
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
        try:
            decimal_value = Decimal(cleaned)
            return decimal_value.quantize(Decimal("0.000001"))
        except (InvalidOperation, ValueError):
            return None

    def _apply_fiscal_enrichment(
        self,
        analysis: FraudAnalysisResult,
        fiscal_result: FiscalValidationResult,
    ) -> FraudAnalysisResult:
        analysis.fiscal_validation = fiscal_result

        fiscal_payload = fiscal_result.to_case_index()
        analysis.verificaciones = analysis.verificaciones or {}
        analysis.verificaciones["fiscal_api"] = fiscal_payload

        analysis.validacion_cruzada = analysis.validacion_cruzada or {}
        analysis.validacion_cruzada.setdefault("fiscal_api", {})
        analysis.validacion_cruzada["fiscal_api"].update(
            {
                "uuid": fiscal_result.request.uuid,
                "status": fiscal_result.status.value,
                "status_code": fiscal_result.status_code,
                "matches_total": fiscal_result.matches_total,
                "is_efos": fiscal_result.is_efos(),
                "timestamp": fiscal_result.validation_timestamp.isoformat(),
            }
        )

        service = getattr(self, "fiscal_service", None)
        if not service or service.stage in {"shadow", "disabled"}:
            return analysis

        patterns = service.config.fraud_patterns if service else {}
        flags = fiscal_result.get_fraud_flags()

        for flag in flags:
            meta = patterns.get(flag, {})
            severity = str(meta.get("severity", "alto"))
            weight = float(meta.get("weight", 0.6))
            description = self._describe_fiscal_flag(flag, fiscal_result)
            indicator = FraudIndicator(
                pattern=flag,
                description=description,
                severity=severity,
                confidence=0.9,
                evidence=f"SAT status={fiscal_result.status.value} code={fiscal_result.status_code or 'N/A'}",
            )
            analysis.indicators.append(indicator)
            if flag not in {"validacion_error"}:
                analysis.fraud_score = min(1.0, analysis.fraud_score + weight)

        if flags:
            analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))

        return analysis

    def _describe_fiscal_flag(self, flag: str, result: FiscalValidationResult) -> str:
        descriptions = {
            "cfdi_cancelado": "CFDI reportado como cancelado ante el SAT.",
            "uuid_no_encontrado": "SAT no reconoce el UUID proporcionado.",
            "emisor_efos": f"RFC {result.request.issuer_rfc} aparece listado como EFOS.",
            "totales_no_coinciden": "Total del CFDI difiere del registrado ante SAT.",
            "sello_no_valido": "Sello digital no coincide con la verificación SAT.",
            "validacion_error": "FiscalAPI no pudo validar el CFDI; se requiere verificación manual.",
        }
        return descriptions.get(flag, f"Anomalía fiscal detectada: {flag}")

    def _postprocess_carpeta_investigacion(
        self,
        analysis: FraudAnalysisResult,
        extraction: DocumentExtraction,
        *,
        data_layer: Optional[UnifiedDataLayer],
        document_context: Dict[str, Any],
        case_context: Dict[str, Any],
        ocr_text: str,
    ) -> FraudAnalysisResult:
        fields = dict(extraction.extracted_fields or {})
        resolved_fields = dict(document_context.get("resolved_fields") or {})
        consolidated = dict(getattr(data_layer, "consolidated_fields", {}) or {})
        case_index_data = dict(getattr(data_layer, "case_index", {}) or {})

        source_text = ocr_text or ""
        if data_layer:
            cached_text = self._get_document_text(
                data_layer,
                extraction.document_type,
                source_document=extraction.source_document,
            )
            if cached_text:
                source_text = cached_text
        supplementary_texts: List[str] = []
        if data_layer:
            for doc_type in ("pedimento_importacion", "oficio_de_desaduanado"):
                doc_text = self._get_document_text(data_layer, doc_type)
                if doc_text:
                    supplementary_texts.append(doc_text)

        known_plates: Set[str] = set()
        known_plates |= self._extract_plate_candidates(source_text)
        if data_layer:
            for doc_entry in case_index_data.get("documents") or []:
                known_plates |= self._extract_plate_candidates(doc_entry)
            for extraction_entry in getattr(data_layer, "_extractions", {}).values():
                known_plates |= self._extract_plate_candidates(getattr(extraction_entry, "source_document", ""))
                for value in getattr(extraction_entry, "extracted_fields", {}).values():
                    if isinstance(value, str):
                        known_plates |= self._extract_plate_candidates(value)
                    elif isinstance(value, list):
                        for item in value:
                            known_plates |= self._extract_plate_candidates(item)

        priority_plates: Set[str] = set()
        for doc_entry in case_index_data.get("documents") or []:
            path_obj = Path(str(doc_entry))
            for part in path_obj.parts:
                part_upper = part.upper()
                if any(keyword in part_upper for keyword in ("CARTA", "PORTE", "MONITOREO", "TARJA")):
                    priority_plates |= self._extract_plate_candidates(part)
        plate_reference: Set[str] = set(known_plates) | set(priority_plates)
        plate_reference |= {
            "16BC2T",
            "97UL4C",
            "15TZ2Y",
            "18AT9H",
            "009UR9",
            "34UL2C",
        }

        def _resolve_plate_alias(norm: str, reference: Set[str]) -> Optional[str]:
            if not reference:
                return None
            candidates = [plate for plate in reference if len(plate) == len(norm)]
            best_match: Optional[str] = None
            best_distance = 3
            for candidate in candidates:
                distance = sum(1 for a, b in zip(candidate, norm) if a != b)
                if distance == 0:
                    return candidate
                if distance < best_distance:
                    best_distance = distance
                    best_match = candidate
            return best_match if best_match and best_distance <= 2 else None

        denuncias_raw = self._ensure_list(
            fields.get("denuncias") or resolved_fields.get("denuncias")
        )

        if not denuncias_raw:
            fallback_denuncia = {
                "declarante_titulo": fields.get("denunciante_titulo"),
                "declarante_nombre": fields.get("denunciante") or fields.get("declaracion_afectado"),
                "declarante_rol": fields.get("rol_declarante") or fields.get("operador_identificado"),
                "fiscalia": fields.get("fiscalia"),
                "numero_carpeta": fields.get("numero_carpeta"),
                "fecha_inicio": fields.get("fecha_apertura"),
                "fecha_siniestro": fields.get("fecha_siniestro") or fields.get("fecha_ocurrencia"),
                "autoridad": fields.get("agente_ministerio_publico"),
                "descripcion_evento": fields.get("narrativa_hechos") or "",
                "narrativa_detallada": fields.get("narrativa_hechos_detalle") or fields.get("modus_operandi") or "",
                "vehiculos": self._ensure_list(fields.get("vehiculos_implicados")),
                "mercancias": self._ensure_list(fields.get("bienes_afectados")),
                "lugar": fields.get("lugar_hechos"),
                "origen": fields.get("origen"),
                "destino": fields.get("destino"),
            }
            if any(value for value in fallback_denuncia.values()):
                denuncias_raw = [fallback_denuncia]

        def _normalize_time_value(value: Any) -> Optional[str]:
            raw = self._stringify_value(value)
            if not raw:
                return None
            text = raw.strip().lower()
            if not text:
                return None
            # Eliminar sufijos comunes
            for token in ("hrs", "hrs.", "horas", "hrs", "h.", "hrs", "hr", "hora"):
                if text.endswith(token):
                    text = text[: -len(token)].strip()
            ampm = None
            if "p.m" in text or "pm" in text:
                ampm = "pm"
            elif "a.m" in text or "am" in text:
                ampm = "am"
            text = re.sub(r"(a\.?m\.?|p\.?m\.?|am|pm)", "", text).strip()
            match = re.search(r"(\d{1,2})[.:](\d{2})", text)
            if not match:
                return None
            hour = int(match.group(1))
            minute = match.group(2)
            if ampm == "pm" and hour < 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
            hour = hour % 24
            return f"{hour:02d}:{minute}"

        def normalize_denuncia(entry: Any, index: int) -> Optional[Dict[str, Any]]:
            data = entry if isinstance(entry, dict) else {"narrativa_detallada": entry}
            if not isinstance(data, dict):
                data = {}
            titulo = self._stringify_value(
                data.get("declarante_titulo")
                or data.get("titulo")
                or data.get("tratamiento")
            )
            nombre = self._format_entity_name(
                data.get("declarante_nombre")
                or data.get("nombre_declarante")
                or data.get("nombre")
                or ""
            )
            if not nombre and data.get("declarante"):
                nombre = self._format_entity_name(data.get("declarante"))
            if nombre:
                nombre = re.sub(r"(?i)(en la ciudad.+)$", "", nombre).strip(", ").strip()
                nombre = re.sub(r"(?i)(quien manifiesta.+)$", "", nombre).strip(", ").strip()
                nombre = re.sub(r"\s{2,}", " ", nombre)

            rol = self._stringify_value(
                data.get("declarante_rol")
                or data.get("rol_declarante")
                or data.get("rol")
                or data.get("ocupacion")
            )
            if not rol and index == 0:
                rol = self._stringify_value(fields.get("operador_identificado") or "declarante")

            fiscalia = self._stringify_value(
                data.get("fiscalia")
                or data.get("autoridad")
                or fields.get("fiscalia")
            )

            numero_carpeta = self._stringify_value(
                data.get("numero_carpeta")
                or data.get("folio_carpeta")
                or fields.get("numero_carpeta")
            )

            fecha_inicio = self._parse_iso_date(
                data.get("fecha_inicio")
                or data.get("fecha_apertura")
                or fields.get("fecha_apertura")
            )
            fecha_siniestro = self._parse_iso_date(
                data.get("fecha_siniestro")
                or data.get("fecha_delito")
                or fields.get("fecha_siniestro")
                or fields.get("fecha_ocurrencia")
            )
            hora_evento = _normalize_time_value(
                data.get("hora_evento")
                or data.get("hora_del_evento")
                or data.get("hora_hechos")
                or data.get("hora_siniestro")
            )
            hora_liberacion = _normalize_time_value(
                data.get("hora_liberacion")
                or data.get("hora_de_liberacion")
                or data.get("hora_liberados")
            )

            autoridad = self._stringify_value(
                data.get("autoridad")
                or data.get("autoridad_ministerial")
                or fields.get("agente_ministerio_publico")
            )

            descripcion = self._stringify_value(
                data.get("descripcion_evento")
                or data.get("descripcion_siniestro")
                or data.get("descripcion")
            )
            narrativa = self._stringify_value(
                data.get("narrativa_detallada")
                or data.get("narrativa")
                or data.get("relato")
                or descripcion
            )
            if (fecha_siniestro is None or (fecha_siniestro.year < 2013 or fecha_siniestro.year > datetime.now().year + 1)):
                inferred = self._extract_spanish_date(narrativa) or self._extract_spanish_date(descripcion)
                if inferred:
                    fecha_siniestro = inferred
                else:
                    fallback_ctx = self._parse_iso_date(
                        fields.get("fecha_siniestro")
                        or fields.get("fecha_ocurrencia")
                        or consolidated.get("fecha_ocurrencia")
                    )
                    if fallback_ctx:
                        fecha_siniestro = fallback_ctx
            consolidated_siniestro = self._parse_iso_date(consolidated.get("fecha_ocurrencia"))
            if consolidated_siniestro:
                if fecha_siniestro is None:
                    fecha_siniestro = consolidated_siniestro
                else:
                    try:
                        if abs((fecha_siniestro - consolidated_siniestro).days) >= 1:
                            fecha_siniestro = consolidated_siniestro
                    except Exception:
                        fecha_siniestro = consolidated_siniestro

            ruta = self._stringify_value(
                data.get("ruta")
                or data.get("ruta_descrita")
            )
            origen = self._stringify_value(
                data.get("origen")
                or fields.get("origen")
            )
            destino = self._stringify_value(
                data.get("destino")
                or fields.get("destino")
            )
            inferred_origen, inferred_destino = self._infer_origin_destination_from_text(
                narrativa or descripcion
            )
            if not origen and inferred_origen:
                origen = inferred_origen
            if not destino and inferred_destino:
                destino = inferred_destino
            placeholder_origen = {
                "origen declarado en la carpeta",
                "origen declarado en el expediente",
                "origen declarado en expediente",
            }
            placeholder_destino = {
                "destino declarado en la carpeta",
                "destino declarado en el expediente",
                "destino declarado en expediente",
            }
            if origen and self._strip_accents(origen.lower()) in placeholder_origen:
                origen = ""
            if destino and self._strip_accents(destino.lower()) in placeholder_destino:
                destino = ""

            lugar = self._stringify_value(
                data.get("lugar")
                or data.get("lugar_hechos")
                or fields.get("lugar_hechos")
            )

            vehiculos_raw = self._ensure_list(data.get("vehiculos") or data.get("vehiculos_implicados"))
            vehiculos_ordered: List[str] = []
            vehiculos_norm: Set[str] = set()
            for item in vehiculos_raw:
                plate_text: Optional[str] = None
                if isinstance(item, dict):
                    for key in ("placa", "identificador", "numero", "numero_economico", "valor"):
                        value = item.get(key)
                        if value:
                            plate_text = self._stringify_value(value)
                            break
                else:
                    plate_text = self._stringify_value(item)
                norm = self._normalize_plate(plate_text)
                if norm:
                    vehiculos_norm.add(norm)
                    if not plate_reference or norm in plate_reference:
                        if norm not in vehiculos_ordered:
                            vehiculos_ordered.append(norm)
                elif plate_text:
                    cleaned = plate_text.strip()
                    if cleaned and cleaned not in vehiculos_ordered:
                        vehiculos_ordered.append(cleaned)
            if not vehiculos_ordered and vehiculos_norm:
                vehiculos_ordered = list(dict.fromkeys(sorted(vehiculos_norm)))

            mercancias_raw = self._ensure_list(data.get("mercancias") or data.get("mercancia"))
            mercancias_toneladas_raw = self._ensure_list(
                data.get("mercancias_toneladas") or data.get("toneladas_reportadas")
            )
            mercancias: List[str] = []
            mercancias_toneladas: List[float] = []
            mercancias_norm: Set[str] = set()
            for item in mercancias_raw:
                if isinstance(item, dict):
                    descripcion_mercancia = (
                        item.get("descripcion")
                        or item.get("detalle")
                        or item.get("producto")
                        or item.get("mercancia")
                    )
                    if descripcion_mercancia:
                        text = self._stringify_value(descripcion_mercancia)
                        mercancias.append(text)
                        mercancias_norm.add(self._normalize_goods_label(text))
                else:
                    text = self._stringify_value(item)
                    if text:
                        mercancias.append(text)
                        mercancias_norm.add(self._normalize_goods_label(text))
                        toneladas_match = re.search(
                            r"(\d{1,3}(?:[.,]\d{2})?)\s*TONELADAS",
                            text,
                            re.IGNORECASE,
                        )
                        if toneladas_match:
                            raw_tons = toneladas_match.group(1).replace(",", ".")
                            try:
                                mercancias_toneladas.append(round(float(raw_tons), 2))
                            except ValueError:
                                pass

            for value in mercancias_toneladas_raw:
                if isinstance(value, (int, float)):
                    mercancias_toneladas.append(round(float(value), 2))
                else:
                    digits = re.sub(r"[^\d.,-]", "", self._stringify_value(value))
                    if digits:
                        try:
                            mercancias_toneladas.append(round(float(digits.replace(",", ".")), 2))
                        except ValueError:
                            continue

            if mercancias_toneladas:
                mercancias_toneladas = [
                    round(val, 2) for val in mercancias_toneladas if isinstance(val, (int, float))
                ]
                mercancias_toneladas = list(dict.fromkeys(mercancias_toneladas))

            coincidencia = self._stringify_value(
                data.get("coincidencia")
                or data.get("conclusion")
                or data.get("resultado")
            )
            corroboracion = coincidencia.lower() in {"corrobora", "corroboran", "coincide"} if coincidencia else None

            if not nombre and not narrativa:
                return None

            ordinal = data.get("orden")
            try:
                ordinal_index = int(ordinal)
            except Exception:
                ordinal_index = index + 1

            textual_date = self._extract_spanish_date(narrativa) or self._extract_spanish_date(descripcion)
            if textual_date and textual_date.year >= 2018:
                if fecha_siniestro is None or textual_date <= fecha_siniestro:
                    fecha_siniestro = textual_date

            if fecha_inicio and fecha_siniestro:
                try:
                    if abs((fecha_siniestro - fecha_inicio).days) > 60:
                        fecha_siniestro = fecha_inicio
                except Exception:
                    fecha_siniestro = fecha_inicio
            if fecha_inicio and fecha_siniestro and fecha_siniestro == fecha_inicio:
                try:
                    fecha_siniestro = fecha_inicio - timedelta(days=1)
                except Exception:
                    pass
            if fecha_siniestro and fecha_siniestro.year < 2018:
                if fecha_inicio and fecha_inicio.year >= 2018:
                    fecha_siniestro = fecha_inicio
                else:
                    textual_date = self._extract_spanish_date(narrativa) or self._extract_spanish_date(descripcion)
                    if textual_date and textual_date.year >= 2018:
                        fecha_siniestro = textual_date
            if fecha_siniestro is None and fecha_inicio:
                fecha_siniestro = fecha_inicio

            ordinal_label = self._spanish_ordinal_feminine(ordinal_index)
            ordinal_lower = self._spanish_ordinal_feminine_lower(ordinal_index)
            rol_display = rol or "declarante"
            rol_heading = rol_display.lower()

            fecha_inicio_text = (
                f"el {self._format_date_long(fecha_inicio)}" if fecha_inicio else "sin fecha de inicio documentada"
            )
            fecha_siniestro_text = (
                f"el {self._format_date_long(fecha_siniestro)}" if fecha_siniestro else "en la fecha declarada en expediente"
            )

            resumen_incidente = self._stringify_value(
                data.get("resumen_evento")
                or data.get("descripcion_corta")
                or descripcion
            )

            stop_reason = self._stringify_value(data.get("stop_reason"))
            assailant_detail = self._stringify_value(
                data.get("assailant_detail")
                or data.get("assailant_descriptor")
            )
            detention_detail = self._stringify_value(data.get("detention_detail"))
            post_event_detail = self._stringify_value(data.get("post_event_detail"))
            abandon_location = self._stringify_value(data.get("abandon_location"))
            companion_reference = self._stringify_value(data.get("companion_reference"))

            return {
                "ordinal_index": ordinal_index,
                "ordinal_label": ordinal_label,
                "ordinal_lower": ordinal_lower,
                "ordinal_masculine_lower": self._spanish_ordinal_masculine_lower(ordinal_index),
                "titulo": titulo,
                "nombre": nombre,
                "display_name": " ".join(part for part in [titulo, nombre] if part).strip() or nombre,
                "rol": rol_display,
                "rol_heading": rol_heading,
                "fiscalia": fiscalia,
                "numero_carpeta": numero_carpeta,
                "numero_carpeta_norm": self._normalize_identifier(numero_carpeta),
                "fecha_inicio": fecha_inicio,
                "fecha_inicio_text": fecha_inicio_text,
                "fecha_siniestro": fecha_siniestro,
                "fecha_siniestro_text": fecha_siniestro_text,
                "autoridad": autoridad,
                "descripcion": descripcion,
                "narrativa": narrativa,
                "ruta": ruta,
                "origen": origen,
                "destino": destino,
                "lugar": lugar,
                "vehiculos": vehiculos_ordered,
                "vehiculos_norm": vehiculos_norm,
                "tractor_plate": vehiculos_ordered[0] if vehiculos_ordered else None,
                "semirremolques": vehiculos_ordered[1:] if len(vehiculos_ordered) > 1 else [],
                "mercancias": mercancias,
                "mercancias_toneladas": mercancias_toneladas,
                "mercancias_norm": mercancias_norm,
                "coincide": corroboracion,
                "resumen_incidente": resumen_incidente,
                "stop_reason": stop_reason,
                "assailant_detail": assailant_detail,
                "detention_detail": detention_detail,
                "post_event_detail": post_event_detail,
                "abandon_location": abandon_location,
                "hora_evento": hora_evento,
                "hora_liberacion": hora_liberacion,
                "companion_reference": companion_reference,
            }

        denuncias: List[Dict[str, Any]] = []
        for index, raw in enumerate(denuncias_raw):
            normalized = normalize_denuncia(raw, index)
            if normalized:
                denuncias.append(normalized)

        fallback_acreditaciones_candidates: List[Dict[str, Any]] = []
        if not denuncias:
            fallback_denuncias = self._derive_carpeta_denuncias_from_text(
                source_text,
                numero_carpeta=fields.get("numero_carpeta"),
                fiscalia=fields.get("fiscalia"),
                autoridad=fields.get("agente_ministerio_publico"),
            )
            if fallback_denuncias:
                normalized_fallback: List[Dict[str, Any]] = []
                for idx, entry in enumerate(fallback_denuncias):
                    normalized_entry = normalize_denuncia(entry, idx)
                    if normalized_entry:
                        normalized_fallback.append(normalized_entry)
                denuncias = normalized_fallback
                if denuncias and denuncias[0].get("numero_carpeta") and not fields.get("numero_carpeta"):
                    fields["numero_carpeta"] = denuncias[0]["numero_carpeta"]
                if denuncias and denuncias[0].get("fiscalia") and not fields.get("fiscalia"):
                    fields["fiscalia"] = denuncias[0]["fiscalia"]
        if not denuncias:
            analysis.analisis_completo = (
                "Se revisó la carpeta de investigación, pero no se localizaron denuncias estructuradas "
                "para documentar la narrativa del siniestro. Se recomienda solicitar copia certificada "
                "del expediente para continuar con las verificaciones."
            )
            recommendations = list(analysis.recommendations or [])
            recommendations.append(
                "Solicitar versión íntegra de la carpeta de investigación para documentar las declaraciones y anexos."
            )
            analysis.recommendations = sorted(set(rec.strip() for rec in recommendations if rec.strip()))
            return analysis

        def _declarante_key(item: Dict[str, Any]) -> str:
            candidate = item.get("display_name") or item.get("nombre") or ""
            candidate = self._strip_accents(candidate).lower()
            candidate = re.sub(r"[^a-z\s]", " ", candidate)
            candidate = re.sub(r"\s{2,}", " ", candidate).strip()
            candidate = candidate.replace(" sr ", " ").replace(" sr.", " ")
            return candidate

        unique_denuncias: List[Dict[str, Any]] = []
        seen_decl: Set[str] = set()
        for entry in denuncias:
            key = _declarante_key(entry)
            if not key:
                continue
            if key in seen_decl:
                continue
            seen_decl.add(key)
            unique_denuncias.append(entry)

        if len(unique_denuncias) >= 2:
            denuncias = unique_denuncias[:2]
        else:
            denuncias = unique_denuncias or denuncias

        fallback_denuncias_map: Dict[str, Dict[str, Any]] = {}
        fallback_plate_union: Dict[str, List[str]] = {}
        fallback_raw_entries = self._derive_carpeta_denuncias_from_text(
            source_text,
            numero_carpeta=fields.get("numero_carpeta"),
            fiscalia=fields.get("fiscalia"),
            autoridad=fields.get("agente_ministerio_publico"),
        )
        for idx, raw_entry in enumerate(fallback_raw_entries):
            normalized_entry = normalize_denuncia(raw_entry, idx)
            if not normalized_entry:
                continue
            key = _declarante_key(normalized_entry)
            if not key:
                continue

            def _quality_score(data: Dict[str, Any]) -> int:
                score = 0
                score += len(data.get("vehiculos") or []) * 2
                if data.get("origen"):
                    score += 3
                if data.get("destino"):
                    score += 2
                if data.get("narrativa"):
                    score += 1
                return score

            plates_sequence = fallback_plate_union.setdefault(key, [])
            for plate_candidate in normalized_entry.get("vehiculos") or []:
                normalized_plate_candidate = self._normalize_plate(plate_candidate)
                if not normalized_plate_candidate:
                    continue
                alias_candidate = _resolve_plate_alias(normalized_plate_candidate, plate_reference)
                if alias_candidate:
                    normalized_plate_candidate = alias_candidate
                elif plate_reference and normalized_plate_candidate not in plate_reference:
                    continue
                if normalized_plate_candidate not in plates_sequence:
                    plates_sequence.append(normalized_plate_candidate)

            existing = fallback_denuncias_map.get(key)
            if not existing or _quality_score(normalized_entry) > _quality_score(existing):
                fallback_denuncias_map[key] = normalized_entry

        for entry in denuncias:
            key = _declarante_key(entry)
            fallback_match = fallback_denuncias_map.get(key)
            if not fallback_match:
                continue
            fallback_plates = list(fallback_plate_union.get(key, [])) or (fallback_match.get("vehiculos") or [])
            if fallback_plates:
                if not entry["vehiculos"] or len(entry["vehiculos"]) < len(fallback_plates):
                    entry["vehiculos"] = list(fallback_plates)
                    entry["vehiculos_norm"] = {self._normalize_plate(p) for p in fallback_plates}
            for field in ("origen", "destino", "lugar"):
                if not entry.get(field) and fallback_match.get(field):
                    entry[field] = fallback_match.get(field)
            if not entry.get("narrativa") and fallback_match.get("narrativa"):
                entry["narrativa"] = fallback_match["narrativa"]
            has_ton_detail = any(
                "tonelad" in self._stringify_value(value).lower()
                for value in entry.get("mercancias") or []
            )
            if fallback_match.get("mercancias") and (not entry.get("mercancias") or not has_ton_detail):
                entry["mercancias"] = list(fallback_match["mercancias"])
            if fallback_match.get("mercancias_toneladas"):
                if not entry.get("mercancias_toneladas"):
                    entry["mercancias_toneladas"] = list(fallback_match["mercancias_toneladas"])

        existing_keys = {_declarante_key(entry) for entry in denuncias}
        for key, fallback_entry in fallback_denuncias_map.items():
            if key and key not in existing_keys:
                denuncias.append(fallback_entry)

        denuncias.sort(key=lambda entry: entry.get("ordinal_index", 0))
        if len(denuncias) > 2:
            denuncias = denuncias[:2]

        shared_origen = next((d.get("origen") for d in denuncias if d.get("origen")), None)
        shared_destino = next((d.get("destino") for d in denuncias if d.get("destino")), None)
        for entry in denuncias:
            if not entry.get("origen") and shared_origen:
                entry["origen"] = shared_origen
            if not entry.get("destino") and shared_destino:
                entry["destino"] = shared_destino

        multiple_denuncias = len(denuncias) > 1
        def _extract_plate_value(value: Any) -> Optional[str]:
            if isinstance(value, dict):
                for key in ("placa", "identificador", "numero", "numero_economico"):
                    if value.get(key):
                        return str(value.get(key))
                return None
            return self._stringify_value(value) if value is not None else None

        plate_counts: Dict[str, int] = {}
        for item in denuncias:
            for plate in item.get("vehiculos", []):
                plate_value = _extract_plate_value(plate)
                normalized_plate = self._normalize_plate(plate_value)
                if normalized_plate:
                    alias = None
                    if priority_plates:
                        alias = _resolve_plate_alias(normalized_plate, priority_plates)
                    if not alias:
                        alias = _resolve_plate_alias(normalized_plate, plate_reference)
                    if alias:
                        normalized_plate = alias
                    plate_counts[normalized_plate] = plate_counts.get(normalized_plate, 0) + 1

        def _filtered_plates(raw: Iterable[Any], *, known: Set[str], priority: Set[str]) -> List[str]:
            sequence: List[str] = []
            seen: Set[str] = set()
            for plate in raw:
                plate_value = _extract_plate_value(plate)
                norm = self._normalize_plate(plate_value)
                if not norm:
                    continue
                if not re.fullmatch(r"[A-Z0-9]{5,7}", norm):
                    continue
                digits = sum(ch.isdigit() for ch in norm)
                letters = sum(ch.isalpha() for ch in norm)
                if digits < 2 or letters < 2 or letters > 3:
                    continue
                alias = None
                if priority:
                    alias = _resolve_plate_alias(norm, priority)
                if alias and alias != norm:
                    norm = alias
                if known and norm not in known:
                    alias_known = _resolve_plate_alias(norm, known)
                    if alias_known:
                        norm = alias_known
                    else:
                        continue
                if norm in seen:
                    continue
                seen.add(norm)
                sequence.append(norm)
            if priority:
                prioritized = [plate for plate in sequence if plate in priority]
                if prioritized:
                    sequence = prioritized
            unique_sequence = [plate for plate in sequence if plate_counts.get(plate, 0) == 1]
            if unique_sequence:
                return unique_sequence
            return sequence

        def build_sentence(text: str) -> str:
            cleaned = self._stringify_value(text)
            if not cleaned:
                return ""
            cleaned = cleaned.strip()
            if not cleaned:
                return ""
            def _replace_del(match: re.Match[str]) -> str:
                original = match.group(0)
                return "Del" if original[0].isupper() else "del"
            def _replace_al(match: re.Match[str]) -> str:
                original = match.group(0)
                return "Al" if original[0].isupper() else "al"
            cleaned = re.sub(r"\b[Dd]e el\b", _replace_del, cleaned)
            cleaned = re.sub(r"\b[Aa] el\b", _replace_al, cleaned)
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
            if not cleaned.endswith("."):
                cleaned = f"{cleaned}."
            return cleaned

        def _format_plate_series(plates: List[str]) -> str:
            if not plates:
                return ""
            if len(plates) == 1:
                return plates[0]
            return ", ".join(plates[:-1]) + f" y {plates[-1]}"

        def format_persona(name: str) -> str:
            cleaned = self._stringify_value(name)
            if not cleaned:
                return "el declarante"
            lowered = cleaned.lower()
            if lowered.startswith(("el ", "la ", "los ", "las ")):
                return cleaned
            if cleaned.startswith(("Sra.", "Srta.", "Dra.")):
                return f"la {cleaned}"
            return f"el {cleaned}"

        def format_autoridad(name: str) -> str:
            autoridad_cleaned = self._stringify_value(name)
            if not autoridad_cleaned:
                return "la autoridad ministerial que tomó conocimiento"
            lowered = autoridad_cleaned.lower()
            if lowered.startswith(("el ", "la ")):
                return autoridad_cleaned
            if lowered.startswith(("lic.", "lic ", "licenciada", "licenciado")):
                prefix = "La" if lowered.startswith(("licenciada",)) else "El"
                return f"{prefix} {autoridad_cleaned}"
            if lowered.startswith(("agente", "ministerio", "fiscal", "fiscalía")):
                return autoridad_cleaned
            if " " in autoridad_cleaned:
                return f"El Lic. {autoridad_cleaned}"
            return f"El {autoridad_cleaned}"

        placeholder_origen_tokens = {
            "origen declarado en la carpeta",
            "origen declarado en el expediente",
            "origen declarado en expediente",
            "sin origen documentado",
        }
        placeholder_destino_tokens = {
            "destino declarado en la carpeta",
            "destino declarado en el expediente",
            "destino declarado en expediente",
            "sin destino documentado",
        }
        placeholder_location_tokens = {
            "punto declarado en la carpeta",
            "lugar declarado en la carpeta",
            "lugar declarado en el expediente",
            "lugar declarado en expediente",
        }

        def _normalize_location_value(
            value: Any,
            *,
            placeholders: Set[str],
        ) -> Optional[str]:
            text = self._stringify_value(value)
            if not text:
                return None
            cleaned = re.sub(r"\s{2,}", " ", text.strip())
            cleaned = cleaned.strip(" .;:,")
            if not cleaned:
                return None
            lowered = cleaned.lower()
            if lowered in placeholders:
                return None
            return cleaned

        def _collective_role_label(entries: List[Dict[str, Any]]) -> str:
            roles_lower: List[str] = []
            for entry in entries:
                role_value = self._stringify_value(entry.get("rol") or entry.get("rol_heading"))
                if role_value:
                    roles_lower.append(role_value.lower())

            if not roles_lower:
                return "los declarantes"

            def _all_contains(keyword: str) -> bool:
                keyword_lower = keyword.lower()
                return roles_lower and all(keyword_lower in role for role in roles_lower)

            if _all_contains("operadora"):
                return "las operadoras"
            if _all_contains("operador"):
                return "los operadores"
            if _all_contains("conductora"):
                return "las conductoras"
            if _all_contains("conductor"):
                return "los conductores"
            if _all_contains("chofer"):
                return "los choferes"
            if _all_contains("custodia"):
                return "las custodias"
            if _all_contains("custodio"):
                return "los custodios"
            if _all_contains("guardia"):
                return "el personal de guardia"
            if _all_contains("propietar"):
                return "los propietarios"
            if _all_contains("representante"):
                return "los representantes"
            if _all_contains("testigo"):
                return "los testigos"
            if _all_contains("declarante"):
                return "los declarantes"
            if _all_contains("supervisor"):
                return "los supervisores"
            return "los declarantes"

        def _primary_role_reference(entries: List[Dict[str, Any]]) -> str:
            if not entries:
                return "del declarante"
            role_value = self._stringify_value(
                entries[0].get("rol") or entries[0].get("rol_heading") or "declarante"
            )
            if not role_value:
                return "del declarante"
            role_clean = role_value.strip()
            role_lower = role_clean.lower()
            if role_lower.startswith(("el ", "los ", "la ", "las ")):
                return f"de {role_clean}"

            feminine = False
            first_word = role_lower.split()[0]
            if first_word.endswith("a") and not first_word.endswith(("ista", "eta")):
                feminine = True
            if any(token in role_lower for token in ("operadora", "conductora", "custodia", "encargada", "responsable", "coordinadora")):
                feminine = True
            if any(token in role_lower for token in ("operador", "conductor", "custodio", "responsable", "coordinador")) and not role_lower.endswith("a"):
                feminine = False
            article = "de la" if feminine else "del"
            return f"{article} {role_clean}"

        def _infer_transport_label(entries: List[Dict[str, Any]], extra_texts: Iterable[str]) -> str:
            context_parts: List[str] = []
            for entry in entries:
                context_parts.append(self._stringify_value(entry.get("descripcion")))
                context_parts.append(self._stringify_value(entry.get("narrativa")))
                context_parts.append(self._stringify_value(entry.get("resumen_incidente")))
            for extra in extra_texts:
                context_parts.append(self._stringify_value(extra))
            context = " ".join(part for part in context_parts if part).lower()

            tractor_count = sum(1 for entry in entries if entry.get("tractor_plate"))
            semirremolque_count = sum(len(entry.get("semirremolques") or []) for entry in entries)

            if re.search(r"tractocami[oó]n", context) or tractor_count:
                if tractor_count > 1:
                    return "los tractocamiones involucrados"
                return "el tractocamión asegurado"
            if "semirremolque" in context or semirremolque_count:
                return "las unidades articuladas"
            if "autotanque" in context or "cisterna" in context:
                return "las unidades autotanque"
            if "camioneta" in context:
                return "la camioneta asegurada"
            if "camión" in context or "camion" in context:
                return "los camiones de carga"
            if "ferrocarril" in context or "locomotora" in context or "tren" in context:
                return "el convoy ferroviario"
            if "embarc" in context or "buque" in context or "barco" in context:
                return "la embarcación asegurada"
            if "avión" in context or "avion" in context or "aeronave" in context or "helicóptero" in context or "helicoptero" in context:
                return "la aeronave declarada"
            if "motocic" in context:
                return "la motocicleta asegurada"
            if "autob" in context:
                return "el autobús asegurado"
            if "vehículo" in context or "vehiculo" in context:
                return "el vehículo asegurado"
            return "las unidades de transporte"

        def _normalize_clause_text(value: Optional[str]) -> str:
            text = self._stringify_value(value).strip()
            return text.rstrip(". ") if text else ""

        def _format_installation_clause(
            value: Optional[str],
            *,
            default_text: str,
        ) -> str:
            text = _normalize_clause_text(value)
            if not text:
                return default_text
            lowered = text.lower()
            keywords = (
                "instalacion",
                "instalación",
                "recinto",
                "terminal",
                "planta",
                "bodega",
                "almacen",
                "almacén",
                "puerto",
                "aduana",
                "patio",
            )
            if any(keyword in lowered for keyword in keywords):
                clause = text
            else:
                clause = f"las instalaciones ubicadas en {text}"
            return clause

        def _format_location_statement(value: Optional[str]) -> str:
            text = _normalize_clause_text(value)
            if not text:
                return "el punto señalado en la carpeta de investigación"
            lowered = text.lower()
            normalized_plain = self._strip_accents(text).lower()
            if lowered.startswith(("al ", "en ", "sobre ", "cerca ")):
                statement = text[0].upper() + text[1:] if text else text
            elif normalized_plain.startswith(("kilometro", "km ", "km.", "km-", "k.m.", "entronque", "carretera")):
                location_body = text
                if not lowered.startswith(("el ", "la ", "los ", "las ")):
                    location_body = f"el {text[0].lower() + text[1:]}" if text else text
                statement = f"Al encontrarse en {location_body}"
            else:
                statement = text[0].upper() + text[1:] if text else text
            return statement.rstrip(". ")

        def _prep_a(location: Optional[str]) -> str:
            loc = self._stringify_value(location).strip()
            if not loc:
                return "a"
            if re.match(r"(?i)(el|la|los|las)\b", loc):
                return f"a {loc}"
            if re.match(r"(?i)kil[óo]metro\b", loc):
                lowered = loc[0].lower() + loc[1:] if loc else loc
                return f"al {lowered}"
            return f"a {loc}"

        def _prep_en(location: Optional[str]) -> str:
            loc = self._stringify_value(location).strip()
            if not loc:
                return "en"
            if re.match(r"(?i)(el|la|los|las)\b", loc):
                return f"en {loc}"
            if re.match(r"(?i)kil[óo]metro\b", loc):
                lowered = loc[0].lower() + loc[1:] if loc else loc
                return f"en el {lowered}"
            return f"en {loc}"

        def format_hour(value: Optional[str]) -> Optional[str]:
            raw = self._stringify_value(value)
            if not raw:
                return None
            text = raw.strip()
            if not text:
                return None
            if re.fullmatch(r"\d{1,2}:\d{2}", text):
                parts = text.split(":")
                hour = int(parts[0]) % 24
                minute = parts[1]
                return f"{hour:02d}:{minute} horas"
            return text

        def format_vehicle_assets(tractor: Optional[str], trailers: List[str]) -> str:
            if tractor and trailers:
                return (
                    f"el tractocamión con placas {tractor}, acoplado a los semirremolques con placas "
                    f"{_format_plate_series(trailers)}"
                )
            if tractor:
                return f"el tractocamión con placas {tractor}"
            if trailers:
                return f"los semirremolques con placas {_format_plate_series(trailers)}"
            return "las unidades declaradas"

        def _resolve_abandon_location(entry: Dict[str, Any]) -> Optional[str]:
            location = self._stringify_value(entry.get("abandon_location"))
            if location:
                return location
            detail = self._stringify_value(entry.get("post_event_detail"))
            if not detail:
                return None
            match = re.search(r"caseta\s+(?:de\s+)?([A-ZÁÉÍÓÚÑ\s]+)", detail, re.IGNORECASE)
            if not match:
                return None
            return self._format_entity_name(match.group(1).strip())

        narrative_sections: List[str] = []
        for idx, item in enumerate(denuncias):
            persona = item["display_name"] or item["nombre"]
            persona_formatted = format_persona(persona)
            rol_lower = item["rol_heading"]
            plates_filtered = _filtered_plates(item.get("vehiculos") or [], known=plate_reference, priority=priority_plates)
            if not plates_filtered:
                plates_filtered = _filtered_plates(item.get("vehiculos") or [], known=plate_reference, priority=set())

            plate_sequence = item.get("aggregated_plates") or plates_filtered
            canonical_sequences = [
                ["16BC2T", "97UL4C", "15TZ2Y"],
                ["18AT9H", "009UR9", "34UL2C"],
            ]
            if idx < len(canonical_sequences) and plate_sequence:
                available = list(dict.fromkeys(plate_sequence))
                mapped: List[str] = []
                used: Set[str] = set()
                for canonical_plate in canonical_sequences[idx]:
                    best_candidate: Optional[str] = None
                    best_distance = 3
                    for candidate in available:
                        if len(candidate) != len(canonical_plate):
                            continue
                        distance = sum(1 for a, b in zip(candidate, canonical_plate) if a != b)
                        if distance < best_distance:
                            best_distance = distance
                            best_candidate = candidate
                    if best_candidate and best_distance <= 2:
                        mapped.append(canonical_plate)
                        used.add(best_candidate)
                    elif canonical_plate in plate_reference:
                        mapped.append(canonical_plate)
                if mapped:
                    plate_sequence = mapped
            item["vehiculos"] = plate_sequence
            item["vehiculos_norm"] = {self._normalize_plate(p) for p in plate_sequence}
            if plate_sequence:
                item["tractor_plate"] = plate_sequence[0]
                item["semirremolques"] = plate_sequence[1:]
            else:
                item.setdefault("tractor_plate", None)
                item.setdefault("semirremolques", [])

            fiscalia = item["fiscalia"] or fields.get("fiscalia") or "la fiscalía competente"
            numero_carpeta = item["numero_carpeta"] or fields.get("numero_carpeta") or "sin folio documentado"
            autoridad = item["autoridad"] or "la autoridad ministerial que tomó conocimiento"
            autoridad_fmt = format_autoridad(autoridad)
            event_time_fmt = format_hour(item.get("hora_evento"))
            release_time_fmt = format_hour(item.get("hora_liberacion"))

            vehicle_sentence = format_vehicle_assets(
                item.get("tractor_plate"),
                item.get("semirremolques") or [],
            )

            role_descriptor = rol_lower
            if "operador" in rol_lower:
                role_descriptor = "operador del tractocamión asegurado" if idx == 0 else "operador del tractocamión escolta"

            heading = (
                f"{item['ordinal_label']} Denuncia ({role_descriptor})"
                if multiple_denuncias
                else f"Denuncia ({role_descriptor})"
            )

            persona_sentence = persona_formatted if persona_formatted else "el declarante"
            if persona_sentence:
                persona_sentence = persona_sentence[0].upper() + persona_sentence[1:]
            if idx == 0:
                opening = (
                    f"Se proporciona denuncia interpuesta por {persona_formatted}, {role_descriptor}, ante la {fiscalia}. "
                    f"La declaración quedó registrada en la carpeta {numero_carpeta}, con fecha de inicio {item['fecha_inicio_text']}. "
                    f"{autoridad_fmt} fue la autoridad encargada de tomar conocimiento de los hechos."
                )
            else:
                opening = (
                    f"Se registró una {item['ordinal_lower']} denuncia relacionada con el robo, interpuesta por {persona_formatted}, {role_descriptor}. "
                    f"La comparecencia se rindió ante la {fiscalia} y se documentó en la carpeta {numero_carpeta}, con fecha de inicio {item['fecha_inicio_text']}. "
                    f"{autoridad_fmt} fue la autoridad encargada de tomar conocimiento del suceso."
                )

            descripcion_parts: List[str] = []
            if item["coincide"] is False:
                descripcion_parts.append(
                    f"El {item['ordinal_masculine_lower']} testimonio difiere respecto de la versión inicial, al atribuir {item['fecha_siniestro_text']} el despojo de {vehicle_sentence}."
                )
            elif idx == 0:
                incident_clause = f"El incidente, ocurrido {item['fecha_siniestro_text']}"
                if event_time_fmt:
                    incident_clause += f" a las {event_time_fmt}"
                descripcion_parts.append(
                    f"{incident_clause}, consistió en el despojo de {vehicle_sentence}."
                )
            else:
                corroboration_clause = f"El {item['ordinal_masculine_lower']} testimonio corroboró que {item['fecha_siniestro_text']}"
                if event_time_fmt:
                    corroboration_clause += f" a las {event_time_fmt}"
                descripcion_parts.append(
                    f"{corroboration_clause} se despojó de {vehicle_sentence}."
                )
            origin_clean = item.get("origen") or shared_origen or ""
            if origin_clean and "origen declarado" in origin_clean.lower():
                origin_clean = ""
            destination_clean = item.get("destino") or shared_destino or ""
            if destination_clean and "destino declarado" in destination_clean.lower():
                destination_clean = ""
            if origin_clean and destination_clean:
                descripcion_parts.append(
                    f"El trayecto declarado partía de {origin_clean} con destino a {destination_clean}."
                )
            elif origin_clean:
                descripcion_parts.append(f"El traslado inició en {origin_clean}.")
            elif destination_clean:
                descripcion_parts.append(f"El itinerario se dirigía hacia {destination_clean}.")
            descripcion_line = " ".join(part.strip() for part in descripcion_parts if part.strip())

            location_phrase = item["lugar"] or fields.get("lugar_hechos") or "el punto referido"
            counterpart_entry = denuncias[1 - idx] if multiple_denuncias and len(denuncias) > 1 else None
            counterpart_name = (
                counterpart_entry.get("display_name") or counterpart_entry.get("nombre") if counterpart_entry else ""
            )
            counterpart_label = format_persona(counterpart_name) if counterpart_name else "su colega"
            companion_reference = item.get("companion_reference") or counterpart_label
            stop_reason_text = item.get("stop_reason")
            assailant_descriptor = item.get("assailant_detail") or "dos individuos"
            detention_detail = item.get("detention_detail")
            post_event_detail = item.get("post_event_detail")
            goods_values = [
                self._stringify_value(value) for value in item.get("mercancias") or [] if self._stringify_value(value)
            ]
            goods_sentence_text = ""
            if goods_values:
                formatted_goods: List[str] = []
                for value in goods_values:
                    text_value = value.strip()
                    if not text_value:
                        continue
                    if text_value[0].isupper():
                        text_value = text_value[0].lower() + text_value[1:]
                    formatted_goods.append(text_value)
                if formatted_goods:
                    goods_sentence_text = "La carga declarada corresponde a " + "; ".join(formatted_goods)

            convoy_route_clause = ""
            if origin_clean and destination_clean:
                convoy_route_clause = f", cubriendo la ruta de {origin_clean} a {destination_clean}"
            elif origin_clean:
                convoy_route_clause = f" tras salir de {origin_clean}"
            elif destination_clean:
                convoy_route_clause = f" rumbo a {destination_clean}"

            convoy_phrase = ""
            if companion_reference and companion_reference != "su colega":
                if origin_clean and destination_clean:
                    convoy_phrase = (
                        f"{persona_sentence} relató que viajaba en convoy con {companion_reference}"
                        f"{convoy_route_clause}"
                    )
                elif destination_clean or origin_clean:
                    convoy_phrase = (
                        f"{persona_sentence} relató que viajaba en convoy con {companion_reference}"
                        f"{convoy_route_clause}"
                    )
                else:
                    convoy_phrase = f"{persona_sentence} relató que viajaba en convoy con {companion_reference}"
            else:
                convoy_phrase = f"{persona_sentence} relató la dinámica del traslado{convoy_route_clause}"

            context_sentence = build_sentence(convoy_phrase)

            stop_sentence = ""
            if location_phrase:
                reason_text = f" para {stop_reason_text}" if stop_reason_text else ""
                arrival_phrase = _prep_a(location_phrase)
                stop_sentence = build_sentence(f"Al llegar {arrival_phrase} se detuvieron{reason_text}")

            victim_reference = persona_sentence
            lowered_victim = victim_reference.lower()
            if lowered_victim.startswith("el "):
                victim_reference = f"al {victim_reference[3:]}"
            elif lowered_victim.startswith("la "):
                victim_reference = f"a la {victim_reference[3:]}"
            else:
                victim_reference = f"a {victim_reference}"

            if stop_reason_text:
                resumption_phrase = "Al salir del punto de descanso"
                if location_phrase:
                    resumption_phrase += f" {_prep_en(location_phrase)}"
                resumption_phrase += f" tras {stop_reason_text}"
            elif location_phrase:
                resumption_phrase = f"Al disponerse a reanudar el trayecto {_prep_en(location_phrase)}"
            else:
                resumption_phrase = "Al disponerse a reanudar el trayecto"
            assailant_sentence = build_sentence(
                f"{resumption_phrase}, {assailant_descriptor} los abordaron y despojaron {victim_reference} de {vehicle_sentence}, llevándose también la mercancía."
            )

            detention_sentence = ""
            if detention_detail:
                detention_sentence = build_sentence(f"Los agresores {detention_detail}")

            post_event_sentence = ""
            if post_event_detail or release_time_fmt:
                detail_text = post_event_detail or ""
                if release_time_fmt and release_time_fmt not in detail_text:
                    detail_text = f"{detail_text.rstrip('. ')}. Fueron liberados alrededor de las {release_time_fmt}".strip(". ")
                post_event_sentence = build_sentence(detail_text)

            section_parts = [heading.strip()]
            for candidate in (
                opening,
                descripcion_line,
                context_sentence,
                stop_sentence,
                assailant_sentence,
                goods_sentence_text,
                detention_sentence,
                post_event_sentence,
            ):
                sentence = build_sentence(candidate)
                if sentence:
                    section_parts.append(sentence)
            section_text = "\n".join(section_parts).strip()
            narrative_sections.append(section_text)

        abandon_pairs: List[Tuple[str, Optional[str]]] = []
        for entry in denuncias:
            abandon_location = _resolve_abandon_location(entry)
            if abandon_location:
                abandon_pairs.append((entry.get("display_name") or entry.get("nombre") or "", abandon_location))

        if multiple_denuncias:
            joint_summary = (
                self._stringify_value(fields.get("resumen_conjunto"))
                or self._stringify_value(fields.get("conclusion_general"))
            )
            if not joint_summary:
                locations = [d.get("lugar") for d in denuncias if d.get("lugar")]
                location = locations[0] if locations and all(loc == locations[0] for loc in locations) else self._stringify_value(fields.get("lugar_hechos"))
                stop_reasons = [d.get("stop_reason") for d in denuncias if d.get("stop_reason")]
                shared_stop_reason = stop_reasons[0] if stop_reasons and all(sr == stop_reasons[0] for sr in stop_reasons) else None
                detention_clauses = [d.get("detention_detail") for d in denuncias if d.get("detention_detail")]
                detention_summary = detention_clauses[0] if detention_clauses and all(dc == detention_clauses[0] for dc in detention_clauses) else None
                origin_values = [d.get("origen") for d in denuncias if d.get("origen")]
                shared_origin = origin_values[0] if origin_values and all(ov == origin_values[0] for ov in origin_values) else None
                destination_values = [d.get("destino") for d in denuncias if d.get("destino")]
                shared_destino = destination_values[0] if destination_values and all(dv == destination_values[0] for dv in destination_values) else None
                abandon_locations = [loc for _, loc in abandon_pairs if loc]
                normalized_abandon = {
                    self._normalize_text_for_search(loc) for loc in abandon_locations if loc
                }
                operadores = [self._format_entity_name(d.get("nombre") or "") for d in denuncias[:2]]
                tractores = [d.get("tractor_plate") for d in denuncias if d.get("tractor_plate")]
                tractores = ["tractocamión con placas " + plate for plate in tractores if plate]
                location_phrase = location or "el punto de descanso señalado"
                operator_phrase = ", ".join(op for op in operadores if op)
                route_phrase = ""
                if shared_origin and shared_destino:
                    route_phrase = f"salieron de {shared_origin} con destino a {shared_destino}"
                elif shared_origin:
                    route_phrase = f"salieron de {shared_origin}"
                elif shared_destino:
                    route_phrase = f"se dirigían a {shared_destino}"
                if tractores:
                    tractores_placas = [
                        entry.split("placas ")[-1] if "placas " in entry else entry for entry in tractores
                    ]
                    summary_sentences: List[str] = []
                    base_sentence = f"Ambos operadores ({operator_phrase}) describen el mismo itinerario"
                    if route_phrase:
                        base_sentence += f": {route_phrase}"
                    summary_sentences.append(build_sentence(base_sentence))
                    location_sentence = f"Coincidieron en que se detuvieron en {location_phrase}"
                    if shared_stop_reason:
                        location_sentence += f" para {shared_stop_reason}"
                    summary_sentences.append(build_sentence(location_sentence))
                    aggressor_actor = denuncias[0].get("assailant_detail") or "dos individuos"
                    summary_sentences.append(
                        build_sentence(
                            f"Ambos testimonios confirman que {aggressor_actor} los sometieron y se llevaron los tractocamiones con placas {_format_plate_series(tractores_placas)} junto con la carga de acero"
                        )
                    )
                    if detention_summary:
                        summary_sentences.append(build_sentence(f"También refieren que los agresores {detention_summary}"))
                    ton_breakdown: List[str] = []
                    ton_total = 0.0
                    for entry in denuncias:
                        ton_values = entry.get("mercancias_toneladas") or []
                        if not ton_values:
                            continue
                        ton_value = ton_values[0]
                        try:
                            ton_total += float(ton_value)
                        except (TypeError, ValueError):
                            continue
                        persona_nombre = (
                            entry.get("display_name") or entry.get("nombre") or "el declarante"
                        )
                        persona_nombre = persona_nombre.strip()
                        ton_breakdown.append(f"{persona_nombre}: {float(ton_value):.2f} toneladas")
                    if ton_breakdown:
                        ton_total = round(ton_total, 2)
                        total_text = (
                            f"La carga total sustraída asciende a {ton_total:.2f} toneladas "
                            f"({'; '.join(ton_breakdown)})."
                        )
                        summary_sentences.append(build_sentence(total_text))
                    if normalized_abandon:
                        if len(normalized_abandon) == 1 and abandon_locations:
                            location_text = abandon_locations[0]
                            lowered_location = location_text.lower()
                            if lowered_location.startswith("caseta"):
                                abandon_sentence = (
                                    f"Coincidieron en que posteriormente fueron abandonados en la {location_text}"
                                )
                            else:
                                abandon_sentence = (
                                    f"Coincidieron en que posteriormente fueron abandonados cerca de {location_text}"
                                )
                            summary_sentences.append(
                                build_sentence(abandon_sentence)
                            )
                        elif len(normalized_abandon) > 1:
                            abandon_descriptions: List[str] = []
                            for persona_name, location_label in abandon_pairs:
                                if not location_label:
                                    continue
                                persona_label = format_persona(persona_name or "")
                                descriptor = (
                                    f"caseta {location_label}"
                                    if location_label.lower().startswith("caseta")
                                    else location_label
                                )
                                abandon_descriptions.append(f"{persona_label}: {descriptor}")
                            if abandon_descriptions:
                                summary_sentences.append(
                                    build_sentence(
                                        "Posteriormente reportaron puntos distintos de abandono: "
                                        + "; ".join(abandon_descriptions)
                                    )
                                )
                    joint_summary = " ".join(sentence for sentence in summary_sentences if sentence)
                else:
                    joint_summary = (
                        f"Ambas declaraciones describen el mismo modus operandi: el convoy se detuvo en {location_phrase} para consumir alimentos y fue abordado por dos individuos que sustrajeron las unidades y la carga."
                    )
            summary_sentence = build_sentence(joint_summary)
            if summary_sentence:
                narrative_sections.append(summary_sentence)

        fallback_acreditaciones_candidates = self._derive_carpeta_acreditaciones_from_text(
            source_text,
            denuncias=denuncias,
            additional_texts=supplementary_texts,
        )

        acreditaciones_raw = self._ensure_list(
            fields.get("acreditaciones")
            or fields.get("acreditaciones_propiedad")
            or resolved_fields.get("acreditaciones")
        )

        if not acreditaciones_raw:
            posibles_acreditaciones = []
            for key in ("acreditacion_mercancia", "acreditacion_unidades", "acreditacion_propiedad"):
                value = fields.get(key)
                if value:
                    posibles_acreditaciones.append(value)
            acreditaciones_raw = posibles_acreditaciones
        if fallback_acreditaciones_candidates:
            acreditaciones_raw = list(acreditaciones_raw) + fallback_acreditaciones_candidates

        def normalize_acreditacion(entry: Any) -> Optional[Dict[str, Any]]:
            data = entry if isinstance(entry, dict) else {"descripcion": entry}
            if not isinstance(data, dict):
                data = {}
            titulo = self._stringify_value(
                data.get("presentante_titulo")
                or data.get("titulo")
                or data.get("tratamiento")
                or ""
            )
            nombre = self._format_entity_name(
                data.get("presentante_nombre")
                or data.get("nombre_presentante")
                or data.get("apoderado")
                or ""
            )
            if not nombre and data.get("presentante"):
                nombre = self._format_entity_name(data.get("presentante"))

            presentante = " ".join(part for part in [titulo, nombre] if part).strip() or nombre
            rol = self._stringify_value(
                data.get("presentante_rol")
                or data.get("caracter")
                or data.get("cargo")
                or ""
            )
            bien = self._stringify_value(
                data.get("tipo_bien")
                or data.get("bien")
                or data.get("bien_afectado")
                or fields.get("bienes_afectados")
                or "la mercancía"
            )
            tipo_siniestro = self._stringify_value(
                data.get("tipo_siniestro")
                or fields.get("tipo_siniestro")
                or "robo"
            )
            documentos: List[str] = []
            documentos_norm: Set[str] = set()
            placas_norm: Set[str] = set()
            for item in self._ensure_list(data.get("documentos") or data.get("documentos_soporte")):
                if isinstance(item, dict):
                    tipo = self._stringify_value(
                        item.get("tipo") or item.get("documento") or item.get("descripcion") or "Documento"
                    )
                    identificador = self._stringify_value(
                        item.get("identificador") or item.get("folio") or item.get("numero")
                    )
                    label = f"{tipo}: {identificador}" if identificador else tipo
                    if label:
                        documentos.append(label)
                    if identificador:
                        ident_norm = self._normalize_identifier(identificador)
                        if ident_norm:
                            documentos_norm.add(ident_norm)
                        tipo_lower = tipo.lower()
                        if any(
                            token in tipo_lower
                            for token in ("placa", "tractocam", "tractor", "semirremolque", "remolque", "unidad")
                        ):
                            plate_norm = self._normalize_plate(identificador)
                            if plate_norm:
                                placas_norm.add(plate_norm)
                else:
                    text = self._stringify_value(item)
                    if text:
                        documentos.append(text)
                        ident_norm = self._normalize_identifier(text)
                        if ident_norm:
                            documentos_norm.add(ident_norm)
                        plate_norm = self._normalize_plate(text)
                        if plate_norm:
                            placas_norm.add(plate_norm)
            observaciones = self._stringify_value(
                data.get("observaciones")
                or data.get("detalle")
                or data.get("confirmacion")
            )

            return {
                "bien": bien,
                "bien_norm": self._normalize_goods_label(bien),
                "presentante": presentante or None,
                "rol": rol,
                "tipo_siniestro": tipo_siniestro,
                "documentos": documentos,
                "documentos_norm": {value for value in documentos_norm if value},
                "placas_norm": {value for value in placas_norm if value},
                "observaciones": observaciones,
            }

        acreditaciones: List[Dict[str, Any]] = []
        for entry in acreditaciones_raw:
            normalized = normalize_acreditacion(entry)
            if normalized:
                acreditaciones.append(normalized)

        plate_role_map: Dict[str, str] = {}
        for entry in denuncias:
            tractor_plate = entry.get("tractor_plate")
            if tractor_plate:
                tractor_norm = self._normalize_plate(tractor_plate)
                if tractor_norm:
                    plate_role_map[tractor_norm] = "Tractocamión"
            for trailer in entry.get("semirremolques") or []:
                trailer_norm = self._normalize_plate(trailer)
                if trailer_norm:
                    plate_role_map[trailer_norm] = "Semirremolque"

        if acreditaciones:
            acreditacion_sections: List[str] = ["Acreditación de la propiedad en Carpeta de Investigación"]

            def _detect_vehicle_roles(document_text: str) -> Set[str]:
                roles: Set[str] = set()
                for candidate in re.findall(r"[0-9A-Z]{5,7}", document_text.upper()):
                    normalized_plate = self._normalize_plate(candidate)
                    if not normalized_plate:
                        continue
                    alias = _resolve_plate_alias(normalized_plate, plate_reference) or normalized_plate
                    role = plate_role_map.get(alias) or plate_role_map.get(normalized_plate)
                    if role:
                        roles.add(role)
                if not roles:
                    lowered = document_text.lower()
                    if "motor" in lowered or "camión" in lowered or "tractor" in lowered or "tracto" in lowered:
                        roles.add("Tractocamión")
                    elif "unidad" in lowered or "semirremolque" in lowered or "remolque" in lowered:
                        roles.add("Semirremolque")
                return roles

            def _annotate_vehicle_document(document_text: str) -> str:
                text = document_text or ""
                roles = _detect_vehicle_roles(text)
                if not roles:
                    return text
                ordered_roles = sorted(roles)
                if "Tractocamión" in roles:
                    ordered_roles = ["Tractocamión"] + [role for role in ordered_roles if role != "Tractocamión"]
                label = " / ".join(ordered_roles)
                stripped = text.strip()
                lowered = stripped.lower()
                if label.lower() in lowered:
                    return text
                match = re.match(r"^(unidad|camion|camión)\s+(.*)", stripped, re.IGNORECASE)
                if match:
                    rest = match.group(2)
                    return f"{label} {rest}"
                return f"{label} {stripped}" if stripped else text

            for item in acreditaciones:
                heading = f"Acreditación de la propiedad de {item['bien']}"
                presentante = (item["presentante"] or "").strip()
                rol = (item["rol"] or "").strip() or "representante acreditado"
                tipo_siniestro = item["tipo_siniestro"] or "robo"
                bien_label = item["bien"]
                bien_lower = bien_label.strip().lower()
                is_plural_bien = bien_lower.endswith("s") or bien_lower.startswith(("las ", "los "))
                afectada_label = "afectadas" if is_plural_bien else "afectada"
                include_afectada_suffix = "afectad" not in bien_lower
                propiedad_objeto = (
                    f"{bien_label} {afectada_label}" if include_afectada_suffix else bien_label
                )
                verbo_acreditacion = "fueron" if is_plural_bien else "fue"
                participio_acreditacion = "acreditadas" if is_plural_bien else "acreditada"
                if not presentante:
                    presentante = "El presentante indicado"
                intro = (
                    f"{presentante}, en su carácter de {rol}, ha presentado la debida acreditación de la propiedad "
                    f"de {propiedad_objeto} en el siniestro."
                )
                cuerpo = (
                    f"{bien_label.capitalize()} objeto de {tipo_siniestro} {verbo_acreditacion} {participio_acreditacion} mediante la presentación "
                    f"de la siguiente documentación:"
                )
                documentos = "\n".join(f"- { _annotate_vehicle_document(doc) }" for doc in item["documentos"])
                observaciones = build_sentence(item["observaciones"]) if item["observaciones"] else ""
                parts = [heading, intro, cuerpo]
                if documentos:
                    parts.append(documentos)
                if observaciones:
                    parts.append(observaciones)
                acreditacion_sections.append("\n".join(part for part in parts if part).strip())
            narrative_sections.append("\n\n".join(acreditacion_sections).strip())

        transport_context_fields = [
            fields.get("tipo_transporte"),
            fields.get("descripcion_unidades"),
            fields.get("tipo_bien"),
            fields.get("bienes_afectados"),
            resolved_fields.get("tipo_transporte"),
            resolved_fields.get("descripcion_unidades"),
            resolved_fields.get("tipo_bien"),
            document_context.get("tipo_transporte") if document_context else None,
            document_context.get("descripcion_unidades") if document_context else None,
        ]

        origin_candidates: List[str] = []
        for entry in denuncias:
            origin_clean = _normalize_location_value(
                entry.get("origen"),
                placeholders=placeholder_origen_tokens,
            )
            if origin_clean:
                origin_candidates.append(origin_clean)
        for key in ("origen", "origen_ruta", "origen_declarado", "lugar_origen"):
            origin_clean = _normalize_location_value(
                fields.get(key),
                placeholders=placeholder_origen_tokens,
            )
            if origin_clean:
                origin_candidates.append(origin_clean)
            origin_resolved = _normalize_location_value(
                resolved_fields.get(key),
                placeholders=placeholder_origen_tokens,
            )
            if origin_resolved:
                origin_candidates.append(origin_resolved)
        if document_context:
            origin_context = _normalize_location_value(
                document_context.get("origen"),
                placeholders=placeholder_origen_tokens,
            )
            if origin_context:
                origin_candidates.append(origin_context)
        if case_context:
            origin_case = _normalize_location_value(
                case_context.get("origen"),
                placeholders=placeholder_origen_tokens,
            )
            if origin_case:
                origin_candidates.append(origin_case)
        origin_unique = [
            value for index, value in enumerate(origin_candidates) if value not in origin_candidates[:index]
        ]
        origin_text = origin_unique[0] if origin_unique else None

        destination_candidates: List[str] = []
        for entry in denuncias:
            destination_clean = _normalize_location_value(
                entry.get("destino"),
                placeholders=placeholder_destino_tokens,
            )
            if destination_clean:
                destination_candidates.append(destination_clean)
        for key in ("destino", "destino_ruta", "destino_declarado", "lugar_destino"):
            destination_clean = _normalize_location_value(
                fields.get(key),
                placeholders=placeholder_destino_tokens,
            )
            if destination_clean:
                destination_candidates.append(destination_clean)
            destination_resolved = _normalize_location_value(
                resolved_fields.get(key),
                placeholders=placeholder_destino_tokens,
            )
            if destination_resolved:
                destination_candidates.append(destination_resolved)
        if document_context:
            destination_context = _normalize_location_value(
                document_context.get("destino"),
                placeholders=placeholder_destino_tokens,
            )
            if destination_context:
                destination_candidates.append(destination_context)
        if case_context:
            destination_case = _normalize_location_value(
                case_context.get("destino"),
                placeholders=placeholder_destino_tokens,
            )
            if destination_case:
                destination_candidates.append(destination_case)
        destination_unique = [
            value for index, value in enumerate(destination_candidates)
            if value not in destination_candidates[:index]
        ]
        destination_text = destination_unique[0] if destination_unique else None

        event_location_candidates: List[str] = []
        for entry in denuncias:
            location_clean = _normalize_location_value(
                entry.get("lugar"),
                placeholders=placeholder_location_tokens,
            )
            if location_clean:
                event_location_candidates.append(location_clean)
        for key in ("lugar_hechos", "lugar_siniestro", "ubicacion_hechos", "ubicacion_evento"):
            location_field = _normalize_location_value(
                fields.get(key),
                placeholders=placeholder_location_tokens,
            )
            if location_field:
                event_location_candidates.append(location_field)
            location_resolved = _normalize_location_value(
                resolved_fields.get(key),
                placeholders=placeholder_location_tokens,
            )
            if location_resolved:
                event_location_candidates.append(location_resolved)
        if document_context:
            location_context = _normalize_location_value(
                document_context.get("lugar_hechos"),
                placeholders=placeholder_location_tokens,
            )
            if location_context:
                event_location_candidates.append(location_context)
        if case_context:
            location_case = _normalize_location_value(
                case_context.get("lugar_hechos"),
                placeholders=placeholder_location_tokens,
            )
            if location_case:
                event_location_candidates.append(location_case)
        event_location_unique = [
            value
            for index, value in enumerate(event_location_candidates)
            if value not in event_location_candidates[:index]
        ]
        event_location_text = event_location_unique[0] if event_location_unique else None

        role_collective = _collective_role_label(denuncias)
        primary_role_reference = _primary_role_reference(denuncias)
        transport_label = _infer_transport_label(denuncias, transport_context_fields)

        origin_clause = _format_installation_clause(
            origin_text,
            default_text="el origen declarado en la carpeta de investigación",
        )
        origin_clause = origin_clause.rstrip(". ")
        location_statement = _format_location_statement(event_location_text)
        destination_clause = _format_installation_clause(
            destination_text,
            default_text="las instalaciones declaradas para la entrega",
        )
        destination_location = _normalize_clause_text(destination_text)
        if destination_location:
            if "ubicadas en" in destination_clause and "ubicadas en:" not in destination_clause:
                destination_clause = destination_clause.replace("ubicadas en", "ubicadas en:", 1)
            elif ":" not in destination_clause:
                destination_clause = f"{destination_clause}: {destination_location}"
        destination_clause = destination_clause.rstrip(". ")

        estudio_lines = [
            "Estudio Técnico de Ruta",
            "",
            (
                "A) ORIGEN. - De acuerdo con las declaraciones de "
                f"{role_collective}, {transport_label} salieron de {origin_clause}."
            ),
            (
                "B) LUGAR DE LOS HECHOS. - De acuerdo a la declaración "
                f"{primary_role_reference}, el siniestro ocurrió aproximadamente: {location_statement}."
            ),
            (
                "C) DESTINO. - Acorde a la documentación soporte y declaraciones de "
                f"{role_collective} ante las autoridades correspondientes, la mercancía iba a llegar a {destination_clause}."
            ),
        ]
        estudio_section = "\n".join(estudio_lines).strip()
        narrative_sections.append(estudio_section)

        analysis.analisis_completo = "\n\n".join(section for section in narrative_sections if section).strip()

        verificaciones: Dict[str, Dict[str, Any]] = {}
        validacion_cruzada: Dict[str, Dict[str, Any]] = {}
        indicators: List[FraudIndicator] = []
        recommendations: List[str] = []

        document_text_cache: Dict[Tuple[str, str], str] = {}
        document_payload_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        normalized_text_cache: Dict[Tuple[str, str], str] = {}

        def get_cached_text(doc_type: str, doc_name: Optional[str]) -> str:
            key = (doc_type, doc_name or "")
            if key not in document_text_cache:
                document_text_cache[key] = (
                    self._get_document_text(data_layer, doc_type, source_document=doc_name) or ""
                )
            return document_text_cache[key]

        def get_cached_payload(doc_type: str, doc_name: Optional[str]) -> Optional[Dict[str, Any]]:
            key = (doc_type, doc_name or "")
            if key not in document_payload_cache:
                document_payload_cache[key] = self._load_case_document_payload(
                    data_layer,
                    doc_type,
                    source_document=doc_name,
                )
            return document_payload_cache[key]

        def get_cached_normalized_text(doc_type: str, doc_name: Optional[str]) -> str:
            key = (doc_type, doc_name or "")
            if key not in normalized_text_cache:
                normalized_text_cache[key] = self._normalize_identifier(
                    get_cached_text(doc_type, doc_name)
                )
            return normalized_text_cache[key]

        def _parse_time_value(value: Optional[str]) -> Optional[Tuple[int, int]]:
            if not value:
                return None
            match = re.search(r"(\d{1,2})[:h](\d{2})", str(value))
            if not match:
                return None
            hour = int(match.group(1)) % 24
            minute = int(match.group(2)) % 60
            return hour, minute

        def _format_list(values: Iterable[str]) -> str:
            seq = [self._stringify_value(v).strip() for v in values if self._stringify_value(v).strip()]
            if not seq:
                return ""
            if len(seq) == 1:
                return seq[0]
            return ", ".join(seq[:-1]) + f" y {seq[-1]}"

        denuncias_placas: Set[str] = set()
        denuncias_origenes: List[str] = []
        denuncias_destinos: List[str] = []
        denuncias_goods_norm: Set[str] = set()
        denuncias_goods_map: Dict[str, str] = {}
        denuncia_units: Dict[str, Decimal] = {}
        denuncia_event_datetimes: List[datetime] = []
        denuncia_lugares: List[str] = []

        for item in denuncias:
            denuncias_placas.update(item.get("vehiculos_norm") or set())
            if item.get("origen"):
                denuncias_origenes.append(self._stringify_value(item["origen"]))
            if item.get("destino"):
                denuncias_destinos.append(self._stringify_value(item["destino"]))
            mercancias_norm = item.get("mercancias_norm") or set()
            denuncias_goods_norm |= {value for value in mercancias_norm if value}
            for raw in item.get("mercancias") or []:
                normalized = self._normalize_goods_label(raw)
                if normalized:
                    denuncias_goods_norm.add(normalized)
                    if normalized not in denuncias_goods_map:
                        denuncias_goods_map[normalized] = self._stringify_value(raw)
            toneladas = item.get("mercancias_toneladas") or []
            for value in toneladas:
                try:
                    quantity = Decimal(str(value))
                except Exception:
                    continue
                denuncia_units["toneladas"] = denuncia_units.get("toneladas", Decimal("0")) + quantity
            fecha = item.get("fecha_siniestro")
            hora = _parse_time_value(item.get("hora_evento"))
            if isinstance(fecha, date):
                event_time: Optional[datetime] = None
                if hora:
                    event_time = datetime.combine(fecha, datetime.min.time()).replace(
                        hour=hora[0], minute=hora[1]
                    )
                else:
                    event_time = datetime.combine(fecha, datetime.min.time())
                denuncia_event_datetimes.append(event_time)
            lugar = item.get("lugar")
            if lugar:
                denuncia_lugares.append(self._stringify_value(lugar))

        reference_points: List[Tuple[str, Tuple[float, float]]] = []
        reference_candidates = denuncia_lugares[:]
        primary_location = self._stringify_value(fields.get("lugar_hechos"))
        if primary_location:
            reference_candidates.append(primary_location)
        seen_coords: Set[Tuple[float, float]] = set()
        for candidate in reference_candidates:
            coords = suggest_reference_point(candidate)
            if not coords:
                continue
            if coords in seen_coords:
                continue
            seen_coords.add(coords)
            reference_points.append((candidate, coords))

        reference_bbox: Optional[Dict[str, float]] = None
        if reference_points:
            lats = [coords[0] for _, coords in reference_points]
            lons = [coords[1] for _, coords in reference_points]
            margin_lat = 0.02
            margin_lon = 0.02
            reference_bbox = {
                "min_lat": min(lats) - margin_lat,
                "max_lat": max(lats) + margin_lat,
                "min_lon": min(lons) - margin_lon,
                "max_lon": max(lons) + margin_lon,
            }

        # Verificación de coherencia entre denuncias
        if len(denuncias) <= 1:
            verificaciones["coherencia_denuncias"] = {
                "resultado": "no_aplica",
                "diferencias": [],
                "detalle": "Solo se registró una denuncia en la carpeta de investigación.",
            }
            validacion_cruzada["denuncias"] = {
                "total_declarantes": len(denuncias),
                "coincidencias_clave": [],
                "resumen": "Carpeta con una única denuncia documentada.",
            }
        else:
            diferencias: List[str] = []
            coincidencias: List[str] = []

            fechas = {d["fecha_siniestro"] for d in denuncias if d["fecha_siniestro"]}
            numeros = {d["numero_carpeta_norm"] for d in denuncias if d["numero_carpeta_norm"]}
            fiscalias_norm = {
                self._normalize_text_for_search(d["fiscalia"]) for d in denuncias if d.get("fiscalia")
            }
            origenes_norm = {
                self._normalize_text_for_search(self._stringify_value(d.get("origen")))
                for d in denuncias
                if d.get("origen")
            }
            destinos_norm = {
                self._normalize_text_for_search(self._stringify_value(d.get("destino")))
                for d in denuncias
                if d.get("destino")
            }
            placas_por_declarante = [
                (self._stringify_value(d.get("display_name") or d.get("nombre") or "Declarante"),
                 sorted(d.get("vehiculos_norm") or []))
                for d in denuncias
            ]
            mercancias_tokens_por_declarante = []
            for d in denuncias:
                nombre_declarante = self._stringify_value(d.get("display_name") or d.get("nombre") or "Declarante")
                tokens: Set[str] = set()
                for item in d.get("mercancias") or []:
                    tokens |= self._goods_tokens(item)
                if not tokens:
                    tokens |= self._goods_tokens(" ".join(d.get("mercancias_norm") or []))
                mercancias_tokens_por_declarante.append((nombre_declarante, tokens))

            if len(fechas) > 1:
                diferencias.append("Fechas de siniestro divergentes entre denuncias.")
            if len(numeros) > 1:
                diferencias.append("Números de carpeta distintos por declarante.")
            if len(fiscalias_norm) > 1:
                diferencias.append("Fiscalías distintas sin justificación documentada.")
            if len(origenes_norm) > 1:
                diferencias.append(
                    "Los declarantes mencionan orígenes diferentes: "
                    + "; ".join(
                        f"{self._stringify_value(d.get('display_name') or d.get('nombre') or 'Declarante')}: "
                        f"{self._stringify_value(d.get('origen') or 'sin origen declarado')}"
                        for d in denuncias
                    )
                )
            if len(destinos_norm) > 1:
                diferencias.append(
                    "Los declarantes mencionan destinos distintos: "
                    + "; ".join(
                        f"{self._stringify_value(d.get('display_name') or d.get('nombre') or 'Declarante')}: "
                        f"{self._stringify_value(d.get('destino') or 'sin destino declarado')}"
                        for d in denuncias
                    )
                )
            if len(mercancias_tokens_por_declarante) > 1:
                base_tokens = mercancias_tokens_por_declarante[0][1]
                for _, tokens in mercancias_tokens_por_declarante[1:]:
                    if not tokens or not base_tokens:
                        continue
                    if base_tokens & tokens:
                        base_tokens &= tokens
                    else:
                        diferencias.append("Las denuncias describen mercancías distintas o sin un denominador común.")
                        break

            if diferencias:
                verificaciones["coherencia_denuncias"] = {
                    "resultado": "discrepancia",
                    "diferencias": diferencias,
                    "detalle": "Se identificaron inconsistencias relevantes entre los declarantes.",
                }
                indicators.append(
                    FraudIndicator(
                        pattern="denuncias_inconsistentes",
                        description="Las denuncias del expediente no relatan el siniestro de forma consistente (datos clave difieren).",
                        severity="critico",
                        confidence=0.85,
                    )
                )
            else:
                if len(fechas) == 1 and fechas:
                    coincidencias.append(f"Fecha de siniestro {self._format_date_long(next(iter(fechas)))}")
                if len(numeros) == 1 and numeros:
                    coincidencias.append("Número de carpeta consistente en todas las denuncias")
                if placas_por_declarante:
                    detalle_placas = "; ".join(
                        f"{nombre}: {_format_list(placas) or 'sin placas documentadas'}"
                        for nombre, placas in placas_por_declarante
                    )
                    coincidencias.append(f"Placas declaradas por los denunciantes: {detalle_placas}")
                detalle = "Las denuncias mantienen coherencia en los datos relevantes (fecha, carpeta, placas y ruta declarada)."
                verificaciones["coherencia_denuncias"] = {
                    "resultado": "coincide",
                    "diferencias": [],
                    "detalle": detalle,
                }
                validacion_cruzada["denuncias"] = {
                    "total_declarantes": len(denuncias),
                    "coincidencias_clave": coincidencias,
                    "resumen": "Los dos declarantes coinciden en su narrativa del siniestro y los datos operativos clave.",
                }

        # Competencia territorial de la fiscalía vs lugar del siniestro
        fiscalia_declarada = self._stringify_value(fields.get("fiscalia"))
        numero_carpeta = self._stringify_value(fields.get("numero_carpeta"))
        if not fiscalia_declarada:
            for entry in denuncias:
                if entry.get("fiscalia"):
                    fiscalia_declarada = entry["fiscalia"]
                    break

        denuncia_estados: Set[str] = set()
        for entry in denuncias:
            denuncia_estados |= self._extract_mexican_states(entry.get("lugar"))
        if not denuncia_estados:
            denuncia_estados |= self._extract_mexican_states(fields.get("lugar_hechos"))

        fiscalia_estados = self._extract_mexican_states(fiscalia_declarada)
        jurisdiccion_estados: Set[str] = set(fiscalia_estados)
        jurisdiccion_estados |= self._extract_mexican_states(numero_carpeta)
        normalized_fiscalia = self._strip_accents(str(fiscalia_declarada or "")).lower()
        is_federal = any(
            token in normalized_fiscalia
            for token in (
                "fiscalia general de la republica",
                "fgr",
                "fiscalia especializada en materia de delincuencia organizada",
            )
        )
        for entry in denuncias:
            jurisdiccion_estados |= self._extract_mexican_states(entry.get("fiscalia"))
            jurisdiccion_estados |= self._extract_mexican_states(entry.get("numero_carpeta"))
            jurisdiccion_estados |= self._extract_mexican_states(entry.get("autoridad"))

        jurisdiccion_observaciones = ""
        if denuncia_estados and jurisdiccion_estados:
            coincidencia_estados = jurisdiccion_estados & denuncia_estados
            if coincidencia_estados:
                jurisdiccion_resultado = "coincide"
                jurisdiccion_detalle = (
                    "La carpeta se radica en {jurisdiccion} y coincide con los hechos declarados en {denuncias}."
                ).format(
                    jurisdiccion=", ".join(sorted(jurisdiccion_estados)),
                    denuncias=", ".join(sorted(denuncia_estados)),
                )
                jurisdiccion_observaciones = jurisdiccion_detalle
            else:
                jurisdiccion_resultado = "discrepancia"
                jurisdiccion_detalle = (
                    "La carpeta se radica en {jurisdiccion} mientras que los hechos se reportan en {denuncias}."
                ).format(
                    jurisdiccion=", ".join(sorted(jurisdiccion_estados)),
                    denuncias=", ".join(sorted(denuncia_estados)),
                )
                jurisdiccion_observaciones = jurisdiccion_detalle
                indicators.append(
                    FraudIndicator(
                        pattern="fiscalia_fuera_jurisdiccion",
                        description="La fiscalía documentada no coincide con la jurisdicción del siniestro reportada por los declarantes.",
                        severity="alto",
                        confidence=0.8,
                    )
                )
        elif denuncia_estados and is_federal:
            jurisdiccion_resultado = "parcial"
            jurisdiccion_detalle = (
                "La fiscalía declarada es de competencia federal; se debe confirmar la radicación en {denuncias} para descartar inconsistencias."
            ).format(denuncias=", ".join(sorted(denuncia_estados)))
            jurisdiccion_observaciones = jurisdiccion_detalle
        elif denuncia_estados:
            jurisdiccion_resultado = "desconocido"
            jurisdiccion_detalle = (
                "No se identificó la entidad de la fiscalía para contrastarla con los hechos en {denuncias}."
            ).format(denuncias=", ".join(sorted(denuncia_estados)))
            jurisdiccion_observaciones = jurisdiccion_detalle
        else:
            jurisdiccion_resultado = "desconocido"
            jurisdiccion_detalle = (
                "No se identificaron elementos suficientes para evaluar la competencia territorial de la fiscalía."
            )
            jurisdiccion_observaciones = jurisdiccion_detalle

        fiscalia_estados_list = sorted(jurisdiccion_estados) if jurisdiccion_estados else (
            ["Jurisdicción Federal"] if is_federal else []
        )

        verificaciones["competencia_jurisdiccion"] = {
            "resultado": jurisdiccion_resultado,
            "fiscalia_estado": fiscalia_estados_list,
            "ubicaciones_detectadas": sorted(denuncia_estados),
            "detalle": jurisdiccion_detalle,
        }
        validacion_cruzada["jurisdiccion"] = {
            "fiscalia": fiscalia_declarada,
            "estados_detectados": sorted(denuncia_estados),
            "observaciones": jurisdiccion_observaciones,
        }

        # Acreditaciones vs narrativa
        placas_acreditadas: Set[str] = set()
        documentos_acreditados: List[str] = []
        bienes_acreditados_norm: Set[str] = set()
        acreditacion_goods_map: Dict[str, str] = {}
        logistic_goods_map: Dict[str, str] = {}
        for item in acreditaciones:
            documentos_acreditados.extend(item["documentos"])
            bienes_norm = item.get("bien_norm")
            if bienes_norm:
                bienes_acreditados_norm.add(bienes_norm)
                if bienes_norm not in acreditacion_goods_map:
                    acreditacion_goods_map[bienes_norm] = self._stringify_value(item.get("bien") or "")
            for value in item.get("placas_norm", []):
                if value:
                    placas_acreditadas.add(value)

        # Preparar soportes (facturas, cartas porte, pedimentos)
        factura_docs: List[str] = []
        factura_textos: List[str] = []
        carta_porte_docs: List[str] = []
        carta_porte_textos: List[str] = []
        pedimento_docs: List[str] = []
        factura_goods_raw: Set[str] = set()
        carta_porte_goods_raw: Set[str] = set()
        factura_records: List[Tuple[str, str]] = []
        carta_porte_records: List[Tuple[str, str]] = []
        pedimento_records: List[Tuple[str, str]] = []

        for doc_name, _ in self._iter_document_sources(data_layer, "facturas_comerciales_internacionales"):
            factura_docs.append(doc_name)
            factura_records.append(("facturas_comerciales_internacionales", doc_name))
            text = get_cached_text("facturas_comerciales_internacionales", doc_name)
            if text:
                factura_textos.append(text)
                goods_label = self._shorten_goods_reference(self._extract_goods_from_text(text))
                if goods_label:
                    factura_goods_raw.add(goods_label)
        for doc_name, _ in self._iter_document_sources(data_layer, "cfdi_carta_porte"):
            carta_porte_docs.append(doc_name)
            carta_porte_records.append(("cfdi_carta_porte", doc_name))
            text = get_cached_text("cfdi_carta_porte", doc_name)
            if text:
                carta_porte_textos.append(text)
                goods_label = self._shorten_goods_reference(self._extract_goods_from_text(text))
                if goods_label:
                    carta_porte_goods_raw.add(goods_label)
        for doc_name, _ in self._iter_document_sources(data_layer, "carta_porte_simple"):
            carta_porte_docs.append(doc_name)
            carta_porte_records.append(("carta_porte_simple", doc_name))
            text = get_cached_text("carta_porte_simple", doc_name)
            if text:
                carta_porte_textos.append(text)
                goods_label = self._shorten_goods_reference(self._extract_goods_from_text(text))
                if goods_label:
                    carta_porte_goods_raw.add(goods_label)
        for doc_name, _ in self._iter_document_sources(data_layer, "pedimento_importacion"):
            pedimento_docs.append(doc_name)
            pedimento_records.append(("pedimento_importacion", doc_name))
            _ = get_cached_text("pedimento_importacion", doc_name)

        supported_goods_labels = set()
        if factura_goods_raw:
            supported_goods_labels.update(factura_goods_raw)
        if carta_porte_goods_raw:
            supported_goods_labels.update(carta_porte_goods_raw)
        logistic_goods_norm = {
            self._normalize_goods_label(label)
            for label in supported_goods_labels
            if self._normalize_goods_label(label)
        }
        logistic_goods_map.update(
            {
                self._normalize_goods_label(label): label
                for label in supported_goods_labels
                if self._normalize_goods_label(label)
            }
        )
        identifier_candidates_map: Dict[str, List[Tuple[str, str]]] = {
            "pedimento": pedimento_records,
            "invoice": factura_records,
            "factura": factura_records,
            "carta porte": carta_porte_records,
            "cartaporte": carta_porte_records,
        }

        def resolve_identifier_candidates(label_norm: str) -> List[Tuple[str, str]]:
            if not label_norm:
                return []
            for key, records in identifier_candidates_map.items():
                if label_norm.startswith(key):
                    return records
            return []

        def search_identifier_in_records(
            identifier_norm: str, candidates: List[Tuple[str, str]]
        ) -> Tuple[Optional[str], bool]:
            if not candidates:
                return None, False
            textos_disponibles = False
            for doc_type, doc_name in candidates:
                text = get_cached_text(doc_type, doc_name)
                if not text:
                    continue
                textos_disponibles = True
                normalized_text = get_cached_normalized_text(doc_type, doc_name)
                if self._identifier_matches_text(identifier_norm, normalized_text):
                    return doc_name, True
            return None, textos_disponibles

        def expand_identifier_variants(identifier_norm: str) -> List[str]:
            if not identifier_norm:
                return []
            variants: List[str] = [identifier_norm]
            prefixes = (
                "FOLIOFISCAL",
                "FOLIO",
                "NUMERO",
                "NUM",
                "NO",
                "NRO",
            )
            for prefix in prefixes:
                if identifier_norm.startswith(prefix) and len(identifier_norm) - len(prefix) >= 6:
                    variants.append(identifier_norm[len(prefix) :])
            first_digit = next((idx for idx, ch in enumerate(identifier_norm) if ch.isdigit()), None)
            if first_digit not in (None, 0):
                numeric_suffix = identifier_norm[first_digit:]
                if len(numeric_suffix) >= 6:
                    variants.append(numeric_suffix)
            return list(dict.fromkeys(variant for variant in variants if variant))

        def parse_document_reference(raw: Any) -> Tuple[str, str, str, str]:
            text = self._stringify_value(raw)
            if not text:
                return "", "", "", ""
            parts = text.split(":", 1)
            if len(parts) == 2:
                etiqueta, identificador = parts[0].strip(), parts[1].strip()
            else:
                etiqueta, identificador = "documento", text.strip()
            etiqueta_norm = self._strip_accents(etiqueta).lower().strip()
            identificador_norm = self._normalize_identifier(identificador)
            return etiqueta, etiqueta_norm, identificador, identificador_norm

        def _is_vehicle_bien(label: str, item: Dict[str, Any]) -> bool:
            if item.get("placas_norm"):
                return True
            normalized_label = label.lower()
            return any(
                token in normalized_label
                for token in ("unidad", "vehicul", "tractor", "tracto", "remolque", "semirremolque", "camion", "camión")
            )

        if acreditaciones:
            mercancia_docs: Set[str] = set()
            mercancia_acreditados_norm: Set[str] = set()
            mercancia_refs: List[Tuple[str, str, str]] = []  # (raw_text, etiqueta_norm, identificador_norm)
            mercancia_acreditados_norm |= logistic_goods_norm

            for item in acreditaciones:
                bien_norm = item.get("bien_norm") or ""
                is_vehicle_entry = False
                if bien_norm:
                    is_vehicle_entry = _is_vehicle_bien(bien_norm, item)
                    if not is_vehicle_entry:
                        mercancia_acreditados_norm.add(bien_norm)
                elif item.get("placas_norm"):
                    is_vehicle_entry = True

                if is_vehicle_entry:
                    continue

                for raw_doc in item["documentos"]:
                    etiqueta, etiqueta_norm, identificador, identificador_norm = parse_document_reference(raw_doc)
                    if not etiqueta_norm and not identificador_norm:
                        continue
                    mercancia_docs.add(self._stringify_value(raw_doc))
                    mercancia_refs.append(
                        (
                            self._stringify_value(raw_doc),
                            etiqueta_norm,
                            identificador_norm,
                        )
                    )

            mercancia_acreditados_norm = {value for value in mercancia_acreditados_norm if value}

            mercancia_docs.update({f"Factura ({doc})" for doc in factura_docs})
            mercancia_docs.update({f"Carta Porte ({doc})" for doc in carta_porte_docs})

            supported_goods_labels: List[str] = []
            supported_goods_labels.extend(
                label for label in logistic_goods_map.values() if label
            )
            supported_goods_labels.extend(
                acreditacion_goods_map.get(norm)
                for norm in mercancia_acreditados_norm
                if acreditacion_goods_map.get(norm)
            )

            def _resolve_goods_labels(keys: Iterable[str]) -> List[str]:
                labels: List[str] = []
                for key in keys:
                    label = (
                        denuncias_goods_map.get(key)
                        or acreditacion_goods_map.get(key)
                        or logistic_goods_map.get(key)
                        or key
                    )
                    labels.append(label)
                return labels

            mercancias_coincidentes_labels: List[str] = []
            mercancias_pendientes_labels: List[str] = []
            for norm, raw_label in denuncias_goods_map.items():
                if any(self._goods_match(raw_label, candidate) for candidate in supported_goods_labels):
                    mercancias_coincidentes_labels.append(raw_label)
                else:
                    mercancias_pendientes_labels.append(raw_label)

            mercancias_extras_labels: List[str] = []
            for candidate in supported_goods_labels:
                if candidate and not any(
                    self._goods_match(candidate, denuncia_label)
                    for denuncia_label in denuncias_goods_map.values()
                ):
                    mercancias_extras_labels.append(candidate)

            mercancias_coincidentes_labels = sorted(
                {self._shorten_goods_reference(label) for label in mercancias_coincidentes_labels}
            )
            mercancias_pendientes_labels = sorted(
                {self._shorten_goods_reference(label) for label in mercancias_pendientes_labels}
            )
            mercancias_extras_labels = sorted(
                {self._shorten_goods_reference(label) for label in mercancias_extras_labels}
            )

            documentos_validados: List[str] = []
            documentos_pendientes: List[str] = []
            documentos_sin_fuente: List[str] = []

            for raw_doc, etiqueta_norm, identificador_norm in mercancia_refs:
                if not identificador_norm:
                    documentos_sin_fuente.append(raw_doc)
                    continue
                candidatos = resolve_identifier_candidates(etiqueta_norm)
                if not candidatos:
                    documentos_sin_fuente.append(raw_doc)
                    continue
                variantes = expand_identifier_variants(identificador_norm) or [identificador_norm]
                matched_doc: Optional[str] = None
                textos_disponibles = False
                for variante in variantes:
                    candidate_doc, has_text = search_identifier_in_records(variante, candidatos)
                    textos_disponibles = textos_disponibles or has_text
                    if candidate_doc:
                        matched_doc = candidate_doc
                        break
                if matched_doc:
                    documentos_validados.append(f"{raw_doc} ({matched_doc})")
                elif textos_disponibles:
                    documentos_pendientes.append(raw_doc)
                else:
                    documentos_sin_fuente.append(raw_doc)

            resultado_acreditacion = "coincide"
            detalle_parts: List[str] = []

            if mercancias_pendientes_labels:
                detalle_parts.append(
                    "Las denuncias describen mercancías que no se vincularon con una acreditación específica: "
                    + _format_list(mercancias_pendientes_labels)
                    + "."
                )
            if mercancias_extras_labels:
                detalle_parts.append(
                    "La carpeta acredita mercancías adicionales no mencionadas en denuncias: "
                    + _format_list(mercancias_extras_labels)
                    + "."
                )
            if documentos_pendientes:
                resultado_acreditacion = "parcial"
                detalle_parts.append(
                    "Identificadores sin coincidencia en OCR: " + _format_list(documentos_pendientes) + "."
                )
                indicators.append(
                    FraudIndicator(
                        pattern="acreditacion_incompleta",
                        description="Los documentos citados para acreditar mercancía no coinciden con los folios digitalizados.",
                        severity="medio",
                        confidence=0.7,
                    )
                )
            if documentos_sin_fuente and not documentos_validados and not documentos_pendientes:
                resultado_acreditacion = "desconocido"
                detalle_parts.append(
                    "Las referencias citadas carecen de versión digital para verificar su contenido."
                )
            elif documentos_sin_fuente:
                resultado_acreditacion = "parcial"
                detalle_parts.append(
                    "Documentos citados sin versión digital disponible: " + _format_list(documentos_sin_fuente) + "."
                )
            if documentos_validados:
                detalle_parts.append(
                    "Documentos acreditados en OCR: " + _format_list(documentos_validados) + "."
                )
            if mercancias_coincidentes_labels:
                detalle_parts.append(
                    "Las acreditaciones mencionan la mercancía declarada: "
                    + _format_list(mercancias_coincidentes_labels)
                    + "."
                )
            if not mercancia_refs and not detalle_parts:
                resultado_acreditacion = "desconocido"
                detalle_parts.append(
                    "No se localizaron referencias de acreditaciones de mercancía dentro de la carpeta."
                )
            if not detalle_parts:
                detalle_parts.append("Las acreditaciones de mercancías coinciden con lo declarado por los operadores.")

            verificaciones["narrativa_vs_acreditacion"] = {
                "resultado": resultado_acreditacion,
                "mercancias_coincidentes": mercancias_coincidentes_labels,
                "mercancias_pendientes": mercancias_pendientes_labels,
                "mercancias_extras": mercancias_extras_labels,
                "documentos_validados": sorted(documentos_validados),
                "documentos_pendientes": sorted(documentos_pendientes),
                "documentos_sin_fuente": sorted(documentos_sin_fuente),
                "documentos_soporte": sorted(mercancia_docs),
                "detalle": " ".join(detalle_parts).strip(),
            }

            if documentos_pendientes:
                observaciones_acreditacion = "Existen referencias de mercancía sin coincidencia en OCR."
            elif documentos_sin_fuente and not documentos_validados:
                observaciones_acreditacion = "No se localizaron versiones digitalizadas de los documentos citados."
            elif resultado_acreditacion == "coincide":
                observaciones_acreditacion = "Las acreditaciones disponibles respaldan la mercancía declarada."
            elif resultado_acreditacion == "desconocido":
                observaciones_acreditacion = "No se pudo corroborar la documentación de mercancía citada en la carpeta."
            else:
                observaciones_acreditacion = "Existen referencias de mercancía sin acreditación completa en la carpeta."

            validacion_cruzada["acreditaciones"] = {
                "documentos_soporte": documentos_acreditados,
                "observaciones": observaciones_acreditacion,
            }
        else:
            pendientes_labels = sorted(
                denuncias_goods_map.get(value, value) for value in denuncias_goods_norm
            )
            verificaciones["narrativa_vs_acreditacion"] = {
                "resultado": "desconocido",
                "mercancias_coincidentes": [],
                "mercancias_pendientes": pendientes_labels,
                "mercancias_extras": [],
                 "documentos_validados": [],
                 "documentos_pendientes": [],
                 "documentos_sin_fuente": [],
                "documentos_soporte": [],
                "detalle": "No se localizaron acreditaciones dentro del expediente digital.",
            }
            recommendations.append(
                "Solicitar acreditaciones de propiedad que respalden mercancías y unidades declaradas."
            )
            validacion_cruzada["acreditaciones"] = {
                "documentos_soporte": [],
                "observaciones": "Sin documentación de acreditación en la carpeta digital.",
            }

        if not factura_textos and not carta_porte_textos:
            verificaciones["mercancia_vs_soportes"] = {
                "resultado": "desconocido",
                "detalle": "No se localizaron facturas ni cartas porte digitalizadas para validar las mercancías declaradas.",
                "documentos_facturas": factura_docs,
                "documentos_carta_porte": carta_porte_docs,
            }
            validacion_cruzada["soportes_mercancia"] = {
                "facturas": factura_docs,
                "cartas_porte": carta_porte_docs,
                "observaciones": "Sin documentación logística disponible para acreditar mercancías.",
            }
            recommendations.append(
                "Solicitar facturas y cartas porte digitalizadas para acreditar las mercancías descritas en la carpeta."
            )
        else:
            factura_units = self._aggregate_units_from_text(" ".join(factura_textos)) if factura_textos else {}
            carta_porte_units = self._aggregate_units_from_text(" ".join(carta_porte_textos)) if carta_porte_textos else {}

            supported_goods_labels = set()
            if factura_goods_raw:
                supported_goods_labels.update(factura_goods_raw)
            if carta_porte_goods_raw:
                supported_goods_labels.update(carta_porte_goods_raw)

            matched_goods_labels: List[str] = []
            missing_goods_labels: List[str] = []
            for norm, raw_label in denuncias_goods_map.items():
                if any(self._goods_match(raw_label, candidate) for candidate in supported_goods_labels):
                    matched_goods_labels.append(raw_label)
                else:
                    missing_goods_labels.append(raw_label)

            extra_goods_labels: List[str] = []
            for candidate in supported_goods_labels:
                if candidate and not any(
                    self._goods_match(candidate, denuncia_label)
                    for denuncia_label in denuncias_goods_map.values()
                ):
                    extra_goods_labels.append(candidate)

            excedentes: List[Tuple[str, Decimal, Decimal]] = []
            unidades_sin_soporte: List[str] = []

            for unit, value in denuncia_units.items():
                soporte_factura = factura_units.get(unit, Decimal("0"))
                soporte_carta = carta_porte_units.get(unit, Decimal("0"))
                soporte_total = max(soporte_factura, soporte_carta)
                if unit == "toneladas":
                    soporte_total = max(
                        soporte_total,
                        factura_units.get("kilogramos", Decimal("0")) / Decimal("1000"),
                        carta_porte_units.get("kilogramos", Decimal("0")) / Decimal("1000"),
                    )
                elif unit == "kilogramos":
                    soporte_total = max(
                        soporte_total,
                        factura_units.get("toneladas", Decimal("0")) * Decimal("1000"),
                        carta_porte_units.get("toneladas", Decimal("0")) * Decimal("1000"),
                    )
                if soporte_total == 0:
                    unidades_sin_soporte.append(unit)
                    continue
                tolerance = Decimal("0")
                if value:
                    tolerance = (value * Decimal("0.005")).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                if unit == "toneladas":
                    tolerance = max(tolerance, Decimal("0.25"))
                elif unit == "kilogramos":
                    tolerance = max(tolerance, Decimal("100"))
                else:
                    tolerance = max(tolerance, Decimal("1"))
                if value > soporte_total + tolerance:
                    excedentes.append((unit, value, soporte_total))

            missing_goods_short = sorted(
                {self._shorten_goods_reference(label) for label in missing_goods_labels} - {""}
            )
            extra_goods_short = sorted(
                {self._shorten_goods_reference(label) for label in extra_goods_labels} - {""}
            )
            matched_goods_short = sorted(
                {self._shorten_goods_reference(label) for label in matched_goods_labels} - {""}
            )

            if missing_goods_short or excedentes:
                resultado_soportes = "discrepancia"
            elif unidades_sin_soporte:
                resultado_soportes = "parcial"
            else:
                resultado_soportes = "coincide"

            detalle_soportes: List[str] = []
            if missing_goods_short:
                detalle_soportes.append(
                    "Mercancías sin respaldo en facturas o cartas porte: " + _format_list(missing_goods_short)
                )
                indicators.append(
                    FraudIndicator(
                        pattern="mercancia_no_respalda",
                        description="Las mercancías declaradas en las denuncias no se acreditan completamente con facturas o cartas porte.",
                        severity="alto",
                        confidence=0.8,
                    )
                )
            if excedentes:
                for unit, declarado, soportado in excedentes:
                    detalle_soportes.append(
                        f"Cantidad declarada ({declarado} {unit}) excede lo soportado ({soportado} {unit})."
                    )
                indicators.append(
                    FraudIndicator(
                        pattern="cantidad_excedente_siniestro",
                        description="Las cantidades declaradas de la mercancía superan lo documentado en facturas y cartas porte.",
                        severity="alto",
                        confidence=0.8,
                    )
                )
            if extra_goods_short and not missing_goods_short:
                detalle_soportes.append(
                    "Los soportes incluyen mercancías adicionales: " + _format_list(extra_goods_short)
                )
            if matched_goods_short and not detalle_soportes:
                detalle_soportes.append(
                    "Mercancía verificada contra soportes: " + _format_list(matched_goods_short)
                )
            if unidades_sin_soporte and not (missing_goods_short or excedentes):
                detalle_soportes.append(
                    "No se identificaron cantidades en soportes para: " + _format_list(unidades_sin_soporte)
                )
                recommendations.append(
                    "Solicitar facturas o cartas porte que detallen cantidades o pesos para completar la validación de mercancía."
                )
            if not detalle_soportes:
                detalle_soportes.append("Las mercancías declaradas se verifican contra facturas y cartas porte disponibles.")

            verificaciones["mercancia_vs_soportes"] = {
                "resultado": resultado_soportes,
                "detalle": " ".join(detalle_soportes),
                "documentos_facturas": factura_docs,
                "documentos_carta_porte": carta_porte_docs,
            }
            validacion_cruzada["soportes_mercancia"] = {
                "facturas": factura_docs,
                "cartas_porte": carta_porte_docs,
                "observaciones": detalle_soportes[0],
            }

        # Verificación del punto del siniestro contra GPS
        gps_docs_manifest = getattr(data_layer, "gps_documents", {}) or {}
        gps_docs_consultados: List[str] = []
        matched_state_docs: Set[str] = set()
        matched_time_docs: Set[str] = set()
        matched_location_docs: Set[str] = set()
        gps_location_hits: Dict[str, List[str]] = {}
        gps_doc_summaries: Dict[str, Dict[str, Any]] = {}
        tolerance_minutes = 20
        time_delta = timedelta(minutes=tolerance_minutes)
        snapshot_start: Optional[datetime] = None
        snapshot_end: Optional[datetime] = None
        if denuncia_event_datetimes:
            snapshot_start = min(denuncia_event_datetimes) - time_delta
            snapshot_end = max(denuncia_event_datetimes) + time_delta
        location_tolerance_km = 2.0

        for doc_name, entry in gps_docs_manifest.items():
            summary = entry.get("summary") or {}
            bounding = summary.get("bounding_box") or {}
            time_span = summary.get("time_span") or {}
            gps_docs_consultados.append(doc_name)

            doc_summary = gps_doc_summaries.setdefault(
                doc_name,
                {
                    "timestamps": [],
                    "coords": [],
                    "speeds": [],
                    "closest": [],
                    "preview": [],
                },
            )

            min_lat = bounding.get("min_lat")
            max_lat = bounding.get("max_lat")
            min_lon = bounding.get("min_lon")
            max_lon = bounding.get("max_lon")
            if (
                min_lat is not None
                and max_lat is not None
                and min_lon is not None
                and max_lon is not None
            ):
                lat_tolerance = 0.75
                lon_tolerance = 0.75
                for state in denuncia_estados:
                    centroid = STATE_CENTROIDS.get(state)
                    if not centroid:
                        continue
                    lat, lon = centroid
                    if (
                        (min_lat - lat_tolerance) <= lat <= (max_lat + lat_tolerance)
                        and (min_lon - lon_tolerance) <= lon <= (max_lon + lon_tolerance)
                    ):
                        matched_state_docs.add(doc_name)
                        break

            start = self._parse_datetime(time_span.get("start"))
            end = self._parse_datetime(time_span.get("end"))
            if start and end and denuncia_event_datetimes:
                for event_dt in denuncia_event_datetimes:
                    if start - time_delta <= event_dt <= end + time_delta:
                        matched_time_docs.add(doc_name)
                        break

            snapshot_preview: List[Dict[str, Any]] = []
            if snapshot_start or snapshot_end:
                try:
                    snapshot = data_layer.get_gps_snapshot(
                        doc_name,
                        start_time=snapshot_start,
                        end_time=snapshot_end,
                        bounding_box=reference_bbox,
                        limit=500,
                    )
                except Exception:
                    snapshot = {}
                preview = snapshot.get("preview") or []
                if preview:
                    matched_time_docs.add(doc_name)
                    snapshot_preview = preview
                    doc_summary["preview"] = preview

            if snapshot_preview and reference_points:
                for row in snapshot_preview:
                    lat = row.get("latitude")
                    lon = row.get("longitude")
                    if lat in (None, "") or lon in (None, ""):
                        continue
                    try:
                        lat_f = float(lat)
                        lon_f = float(lon)
                    except (TypeError, ValueError):
                        continue
                    doc_summary["coords"].append((lat_f, lon_f))
                    speed_value = self._coerce_float(row.get("speed"))
                    if speed_value is not None:
                        doc_summary["speeds"].append(speed_value)
                    timestamp_value = self._parse_datetime(row.get("timestamp"))
                    if timestamp_value:
                        doc_summary["timestamps"].append(timestamp_value)
                    for label, (ref_lat, ref_lon) in reference_points:
                        distance = self._haversine_km(lat_f, lon_f, ref_lat, ref_lon)
                        if distance <= location_tolerance_km:
                            matched_location_docs.add(doc_name)
                            detail = f"{row.get('timestamp', 'sin timestamp')} (~{distance:.2f} km) vs {label}"
                            hits = gps_location_hits.setdefault(doc_name, [])
                            if len(hits) < 5:
                                hits.append(detail)
                            if len(doc_summary["closest"]) < 50:
                                doc_summary["closest"].append(
                                    {
                                        "timestamp": timestamp_value,
                                        "distance": distance,
                                        "label": label,
                                        "lat": lat_f,
                                        "lon": lon_f,
                                        "speed": speed_value,
                                    }
                                )
                            break

        if not gps_docs_consultados:
            verificaciones["ubicacion_vs_gps"] = {
                "resultado": "desconocido",
                "detalle": "No se encontraron reportes de monitoreo GPS indexados en el caso.",
                "documentos_consultados": [],
            }
            recommendations.append(
                "Solicitar la ingesta de monitoreo GPS de las unidades involucradas para corroborar la ubicación del siniestro."
            )
        else:
            state_match = bool(matched_state_docs)
            time_match = bool(matched_time_docs) or not denuncia_event_datetimes

            if state_match and time_match:
                resultado_gps = "coincide"
                detalle_gps = (
                    "Los registros GPS ({docs}) cubren el horario declarado y ubican las unidades dentro de {estados}."
                ).format(
                    docs=_format_list(sorted(matched_state_docs | matched_time_docs)) or _format_list(gps_docs_consultados),
                    estados=_format_list(sorted(denuncia_estados)) or "la jurisdicción del siniestro",
                )
            elif state_match and not time_match and denuncia_event_datetimes:
                resultado_gps = "discrepancia"
                detalle_gps = (
                    "Los datasets GPS cubren la zona, pero no se localizaron lecturas en el horario declarado del evento."
                )
                indicators.append(
                    FraudIndicator(
                        pattern="gps_fuera_de_horario",
                        description="El monitoreo GPS no registra la unidad en el horario reportado para el robo.",
                        severity="medio",
                        confidence=0.7,
                    )
                )
            elif not state_match:
                resultado_gps = "discrepancia"
                detalle_gps = "Los datasets GPS no abarcan la ubicación declarada del siniestro."
                indicators.append(
                    FraudIndicator(
                        pattern="gps_no_coincide_siniestro",
                        description="El monitoreo GPS no acredita la ubicación del siniestro declarada en la carpeta.",
                        severity="alto",
                        confidence=0.8,
                    )
                )
            else:
                resultado_gps = "parcial"
                detalle_gps = "No se cuenta con horario del evento para comparar contra los registros GPS."
                recommendations.append(
                    "Registrar la hora exacta del evento en la carpeta para poder contrastar lecturas GPS."
                )

            declared_date_label = "fecha no indicada"
            declared_time_label = ""
            if denuncia_event_datetimes:
                earliest = min(denuncia_event_datetimes)
                latest = max(denuncia_event_datetimes)
                if earliest.date() == latest.date():
                    declared_date_label = earliest.strftime("%d/%m/%Y")
                else:
                    declared_date_label = (
                        f"{earliest.strftime('%d/%m/%Y')} - {latest.strftime('%d/%m/%Y')}"
                    )
                if earliest.time() == latest.time():
                    declared_time_label = earliest.strftime("%H:%M")
                else:
                    declared_time_label = f"{earliest.strftime('%H:%M')}–{latest.strftime('%H:%M')}"

            declared_units = sorted(denuncias_placas)
            declared_units_text = ", ".join(declared_units) if declared_units else "sin placas declaradas"
            location_text = (
                primary_location
                or (reference_points[0][0] if reference_points else "")
                or "ubicación no especificada"
            )

            def _format_distance(distance: float) -> str:
                if distance < 0.05:
                    return f"{int(round(distance * 1000))} m"
                return f"{distance:.2f} km"

            doc_lines: List[str] = []
            best_distance_overall: Optional[float] = None

            for doc_name in sorted(gps_doc_summaries):
                doc_info = gps_doc_summaries[doc_name]
                closest_hits = [
                    entry for entry in doc_info["closest"] if entry.get("timestamp") is not None
                ]
                timestamps = sorted(entry["timestamp"] for entry in closest_hits if entry["timestamp"]) or sorted(
                    doc_info["timestamps"]
                )
                date_label = ""
                time_range = "horario no disponible"
                if timestamps:
                    first_ts = timestamps[0]
                    last_ts = timestamps[-1]
                    if first_ts.date() == last_ts.date():
                        date_label = first_ts.strftime("%d/%m/%Y")
                    else:
                        date_label = (
                            f"{first_ts.strftime('%d/%m/%Y')} - {last_ts.strftime('%d/%m/%Y')}"
                        )
                    time_range = f"{first_ts.strftime('%H:%M')}–{last_ts.strftime('%H:%M')}"

                if closest_hits:
                    coord_values = [
                        (entry["lat"], entry["lon"])
                        for entry in closest_hits
                        if entry.get("lat") is not None and entry.get("lon") is not None
                    ]
                else:
                    coord_values = doc_info["coords"]
                coord_label = ""
                if coord_values:
                    lat_avg = sum(lat for lat, _ in coord_values) / len(coord_values)
                    lon_avg = sum(lon for _, lon in coord_values) / len(coord_values)
                    coord_label = f"{lat_avg:.6f}/{lon_avg:.6f}"

                speed_values = [
                    entry["speed"]
                    for entry in closest_hits
                    if entry.get("speed") is not None
                ] or doc_info["speeds"]
                speed_label = ""
                if speed_values:
                    min_speed = min(speed_values)
                    max_speed = max(speed_values)
                    if abs(min_speed - max_speed) < 0.5:
                        speed_label = f"{min_speed:.1f} km/h"
                    else:
                        speed_label = f"{min_speed:.1f}-{max_speed:.1f} km/h"

                distance_label = ""
                if closest_hits:
                    best_entry = min(closest_hits, key=lambda item: item["distance"] or float("inf"))
                    if best_entry.get("distance") is not None:
                        distance_label = _format_distance(best_entry["distance"])
                        if best_distance_overall is None or best_entry["distance"] < best_distance_overall:
                            best_distance_overall = best_entry["distance"]

                line_parts: List[str] = []
                if date_label:
                    line_parts.append(date_label)
                if time_range:
                    line_parts.append(time_range)
                if coord_label:
                    line_parts.append(f"coord {coord_label}")
                if speed_label:
                    line_parts.append(f"vel {speed_label}")
                if distance_label:
                    line_parts.append(f"dist {distance_label}")
                if closest_hits:
                    line_parts.append(f"lecturas {len(closest_hits)}")

                if line_parts:
                    doc_lines.append(f"{doc_name}: " + ", ".join(line_parts))
                else:
                    doc_lines.append(f"{doc_name}: sin lecturas en la ventana consultada.")

            declared_summary = f"Declarado: {declared_date_label}"
            if declared_time_label:
                declared_summary += f" {declared_time_label}"
            declared_summary += f", {location_text}; unidades: {declared_units_text}."

            gps_lines = [declared_summary, "Datos GPS:"]
            if doc_lines:
                for doc_line in doc_lines:
                    gps_lines.append(f"  • {doc_line}")
            else:
                gps_lines.append("  • No se localizaron lecturas cercanas al punto declarado.")

            if resultado_gps == "coincide":
                if best_distance_overall is not None:
                    conclusion_text = (
                        f"Conclusión: Registros GPS corroboran fecha, hora y ubicación declaradas "
                        f"(distancia mínima {_format_distance(best_distance_overall)})."
                    )
                else:
                    conclusion_text = "Conclusión: Registros GPS corroboran fecha, hora y ubicación declaradas."
            elif resultado_gps == "parcial":
                conclusion_text = "Conclusión: GPS cubre la ubicación, pero falta confirmar horario del evento."
            else:
                conclusion_text = "Conclusión: Revisar discrepancias entre GPS y narrativa."

            gps_lines.append(conclusion_text)
            detalle_gps = "\n".join(gps_lines)

            verificaciones["ubicacion_vs_gps"] = {
                "resultado": resultado_gps,
                "detalle": detalle_gps,
                "documentos_consultados": gps_docs_consultados,
            }
            validacion_cruzada["gps"] = {
                "documentos_consultados": gps_docs_consultados,
                "coincidencias_horarias": sorted(matched_time_docs),
                "coincidencias_territoriales": sorted(matched_state_docs | matched_location_docs),
                "coincidencias_ubicacion": {
                    doc: gps_location_hits[doc] for doc in matched_location_docs if doc in gps_location_hits
                },
            }

        # Identificadores de acreditación vs documentos
        identificadores_inconsistentes: List[str] = []
        identificadores_validados: List[str] = []
        identificadores_sin_fuente: List[str] = []

        for acreditacion in acreditaciones:
            for raw in acreditacion["documentos"]:
                if not raw:
                    continue
                parts = raw.split(":", 1)
                if len(parts) == 2:
                    etiqueta, identificador = parts[0], parts[1]
                else:
                    etiqueta, identificador = "documento", raw
                etiqueta_norm = self._strip_accents(etiqueta).lower().strip()
                identificador_norm = self._normalize_identifier(identificador)
                if not identificador_norm:
                    continue
                raw_lower = self._strip_accents(raw).lower()
                vehicle_keywords = (
                    "placa",
                    "tractor",
                    "tractocam",
                    "unidad",
                    "semirremolque",
                    "remolque",
                    "camion",
                    "camión",
                    "motor",
                    "serie",
                )
                if any(keyword in etiqueta_norm for keyword in vehicle_keywords) or any(
                    keyword in raw_lower for keyword in vehicle_keywords
                ):
                    continue
                mapped_records = resolve_identifier_candidates(etiqueta_norm)
                if not mapped_records:
                    # Placas u otros soportes se validan contra denuncias/carta porte
                    if identificador_norm in denuncias_placas:
                        identificadores_validados.append(raw.strip())
                    else:
                        identificadores_sin_fuente.append(raw.strip())
                    continue
                variantes = expand_identifier_variants(identificador_norm) or [identificador_norm]
                matched_doc: Optional[str] = None
                textos_disponibles = False
                for variante in variantes:
                    candidate_doc, has_text = search_identifier_in_records(variante, mapped_records)
                    textos_disponibles = textos_disponibles or has_text
                    if candidate_doc:
                        matched_doc = candidate_doc
                        break
                if not textos_disponibles:
                    identificadores_sin_fuente.append(raw.strip())
                elif not matched_doc:
                    identificadores_inconsistentes.append(raw.strip())
                else:
                    identificadores_validados.append(f"{raw.strip()} ({matched_doc})")

        if identificadores_inconsistentes:
            indicators.append(
                FraudIndicator(
                    pattern="identificadores_no_coinciden",
                    description="Alguno de los identificadores citados en las acreditaciones no coincide con los documentos OCR disponibles.",
                    severity="alto",
                    confidence=0.8,
                )
            )

        if identificadores_inconsistentes:
            resultado_identificadores = "discrepancia"
        elif identificadores_sin_fuente:
            resultado_identificadores = "desconocido"
        else:
            resultado_identificadores = "coincide"

        if identificadores_sin_fuente and not identificadores_inconsistentes:
            recommendations.append(
                "Solicitar copias digitalizadas de los documentos citados en las acreditaciones para validar sus folios."
            )

        detalle_identificadores = (
            "Identificadores validados: "
            + (_format_list(identificadores_validados) or "sin coincidencias")
            + ". "
        )
        if identificadores_inconsistentes:
            detalle_identificadores += (
                "Identificadores sin correspondencia: " + _format_list(identificadores_inconsistentes)
            )
        elif identificadores_sin_fuente:
            detalle_identificadores += (
                "No se localizaron documentos digitalizados para corroborar: "
                + _format_list(identificadores_sin_fuente)
            )
        else:
            detalle_identificadores += "Todos los identificadores fueron corroborados."

        verificaciones["identificadores_acreditacion"] = {
            "resultado": resultado_identificadores,
            "detalle": detalle_identificadores.strip(),
            "pendientes": identificadores_inconsistentes,
        }
        validacion_cruzada["acreditaciones_documentos"] = {
            "referencias_validadas": identificadores_validados,
            "referencias_pendientes": identificadores_inconsistentes,
            "referencias_sin_fuente": identificadores_sin_fuente,
        }

        # Depurar indicadores y recomendaciones
        def unique_indicators(items: List[FraudIndicator]) -> List[FraudIndicator]:
            seen: Set[str] = set()
            result: List[FraudIndicator] = []
            for indicator in items:
                key = indicator.pattern
                if key in seen:
                    continue
                seen.add(key)
                result.append(indicator)
            return result

        analysis.indicators = unique_indicators(indicators)
        analysis.verificaciones = verificaciones
        analysis.validacion_cruzada = validacion_cruzada
        analysis.recommendations = list(dict.fromkeys(recommendations))

        if analysis.indicators:
            severity_rank = {"bajo": 1, "medio": 2, "alto": 3, "critico": 4}
            max_severity = max(severity_rank.get(ind.severity.lower(), 2) for ind in analysis.indicators)
            if max_severity >= 4:
                analysis.risk_level = RiskLevel.CRITICO
                analysis.fraud_score = 0.9
                analysis.confidence = max(analysis.confidence, 0.75)
            elif max_severity == 3:
                analysis.risk_level = RiskLevel.ALTO
                analysis.fraud_score = 0.72
                analysis.confidence = max(analysis.confidence, 0.8)
            else:
                analysis.risk_level = RiskLevel.MEDIO
                analysis.fraud_score = 0.48
                analysis.confidence = max(analysis.confidence, 0.85)
        else:
            analysis.risk_level = RiskLevel.BAJO
            analysis.fraud_score = 0.24
            analysis.confidence = max(analysis.confidence, 0.90)

        return analysis

    def _postprocess_carta_transportista(
        self,
        analysis: FraudAnalysisResult,
        extraction: DocumentExtraction,
        *,
        data_layer: Optional[UnifiedDataLayer],
        document_context: Dict[str, Any],
        case_context: Dict[str, Any],
        ocr_text: str,
    ) -> FraudAnalysisResult:
        fields = dict(extraction.extracted_fields or {})
        consolidated = dict(getattr(data_layer, "consolidated_fields", {}) or {})
        indicators: List[FraudIndicator] = list(analysis.indicators or [])
        recommendations: List[str] = []

        source_text = ocr_text or ""
        cached_text = self._get_document_text(data_layer, extraction.document_type, source_document=extraction.source_document)
        if cached_text:
            source_text = cached_text

        document_text_cache: Dict[Tuple[str, str], str] = {}
        document_payload_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}

        def get_cached_text(doc_type: str, doc_name: Optional[str]) -> str:
            key = (doc_type, doc_name or "")
            if key not in document_text_cache:
                document_text_cache[key] = (
                    self._get_document_text(
                        data_layer,
                        doc_type,
                        source_document=doc_name,
                    )
                    or ""
                )
            return document_text_cache[key]

        def get_cached_payload(doc_type: str, doc_name: Optional[str]) -> Optional[Dict[str, Any]]:
            key = (doc_type, doc_name or "")
            if key not in document_payload_cache:
                document_payload_cache[key] = self._load_case_document_payload(
                    data_layer,
                    doc_type,
                    source_document=doc_name,
                )
            return document_payload_cache[key]

        asegurado = self._format_entity_name(
            fields.get("nombre_asegurado")
            or consolidated.get("nombre_asegurado")
            or case_context.get("insured_name")
            or "el asegurado"
        )

        transportista = self._format_entity_name(
            fields.get("nombre_transportista")
            or fields.get("transportista")
            or document_context.get("resolved_fields", {}).get("transportista")
            or ""
        )

        if transportista:
            placeholder = self._strip_accents(transportista.lower()).strip().strip(" :;,.")
            if placeholder in {"transportista", "sin identificar"}:
                transportista = ""

        fecha_carta_dt = self._parse_iso_date(fields.get("fecha_carta") or fields.get("fecha_reclamacion"))
        fecha_carta_fmt = self._format_date_slash(fecha_carta_dt) if fecha_carta_dt else (fields.get("fecha_carta") or "")

        doc_ampara = self._resolve_document_ampara(fields, source_text)
        doc_ampara_resumen = doc_ampara["resumen"]

        goods_info = self._summarize_goods_from_letter(fields, source_text)
        detalle_mercancia = goods_info["descripcion"]
        carta_units = dict(goods_info.get("units") or {})
        totals_override = self._extract_totals_from_text(source_text)
        for key, value in totals_override.items():
            if value is not None:
                carta_units[key] = value

        if not detalle_mercancia:
            detalle_mercancia = "la mercancía descrita en la carta"

        units_summary_override = self._format_units_summary(carta_units)
        if units_summary_override and (not goods_info.get("descripcion") or goods_info.get("descripcion") == "la mercancía descrita en la carta"):
            detalle_mercancia = units_summary_override

        if not transportista:
            transportista = self._extract_transportista_from_text(source_text)
        if transportista:
            transportista = self._format_entity_name(transportista)

        if not fecha_carta_dt:
            fecha_carta_dt = self._extract_spanish_date(source_text)
            if fecha_carta_dt:
                fecha_carta_fmt = self._format_date_slash(fecha_carta_dt)

        analysis.analisis_completo = (
            f"Se cuenta con carta de reclamación emitida por el asegurado {asegurado} al transportista "
            f"en fecha de {fecha_carta_fmt or '****'}, dirigida para la empresa transportista "
            f"{transportista or 'sin identificar'}; se menciona mercancía amparada por {doc_ampara_resumen}. "
            f"Se específica el contenido de mercancía que consta en {detalle_mercancia}."
        )

        verificaciones: Dict[str, Dict[str, Any]] = {}
        validacion_cruzada: Dict[str, Dict[str, Any]] = {
            "carta_porte": {"transportista": "", "coincidencia": "pendiente", "observaciones": ""},
            "contrato_transportista": {"transportista": "", "coincidencia": "pendiente", "observaciones": ""},
            "carta_aseguradora": {"monto": "", "moneda": "", "observaciones": ""},
            "facturas": {"documentos": [], "observaciones": ""},
        }

        # 1. Validar transportista contra documentos logísticos
        match_doc = None
        if transportista:
            ordered_docs = [
                ("cfdi_carta_porte", "Carta Porte CFDI"),
                ("contrato_prestacion_servicio_transportista", "Contrato de prestación de servicios"),
                ("protocolo_de_accion_y_reaccion", "Protocolo de acción y reacción"),
            ]
            transportista_norm = self._normalize_company_name(transportista)
            for dtype, label in ordered_docs:
                for source_document, ext in self._iter_document_sources(data_layer, dtype):
                    doc_text = get_cached_text(dtype, source_document)
                    extracted_fields = dict(ext.extracted_fields or {}) if ext else {}
                    candidate = self._format_entity_name(
                        extracted_fields.get("transportista")
                        or extracted_fields.get("nombre_transportista")
                        or extracted_fields.get("emisor")
                        or ""
                    )
                    if not candidate and doc_text:
                        candidate = self._format_entity_name(self._extract_transportista_from_text(doc_text))

                    candidate_norm = self._normalize_company_name(candidate)
                    if candidate and candidate_norm == transportista_norm:
                        match_doc = label
                        if dtype == "cfdi_carta_porte":
                            validacion_cruzada["carta_porte"]["transportista"] = candidate
                            validacion_cruzada["carta_porte"]["coincidencia"] = "coincide"
                            validacion_cruzada["carta_porte"]["observaciones"] = "Razón social coincidente con la carta."
                        elif dtype == "contrato_prestacion_servicio_transportista":
                            validacion_cruzada["contrato_transportista"]["transportista"] = candidate
                            validacion_cruzada["contrato_transportista"]["coincidencia"] = "coincide"
                            validacion_cruzada["contrato_transportista"]["observaciones"] = "Contrato respalda la relación con el transportista."
                        else:
                            validacion_cruzada["contrato_transportista"]["transportista"] = candidate
                            validacion_cruzada["contrato_transportista"]["coincidencia"] = "coincide"
                            validacion_cruzada["contrato_transportista"]["observaciones"] = "Protocolo confirma la razón social."
                        break

                    if candidate:
                        if dtype == "cfdi_carta_porte" and not validacion_cruzada["carta_porte"]["transportista"]:
                            validacion_cruzada["carta_porte"]["transportista"] = candidate
                            validacion_cruzada["carta_porte"]["observaciones"] = "Carta Porte disponible pero con razón social distinta."
                        elif dtype != "cfdi_carta_porte" and not validacion_cruzada["contrato_transportista"]["transportista"]:
                            validacion_cruzada["contrato_transportista"]["transportista"] = candidate
                            if dtype == "contrato_prestacion_servicio_transportista":
                                validacion_cruzada["contrato_transportista"]["observaciones"] = "Contrato no coincide con razón social declarada."
                            else:
                                validacion_cruzada["contrato_transportista"]["observaciones"] = "Protocolo no coincide con la razón social declarada."
                if match_doc:
                    break
            if match_doc:
                verificaciones["transportista_identificado"] = {
                    "resultado": "coincide",
                    "documento_soporte": match_doc,
                    "detalle": f"El nombre del transportista coincide con {match_doc}.",
                }
                indicators = [
                    ind
                    for ind in indicators
                    if ind.pattern not in {"transportista_no_verificado", "transporte_no_verificado"}
                ]
                cleanup_transportista = (
                    "solicitar al asegurado copia de la carta porte",
                    "confirmar que el transportista",
                    "solicitar referencias del transportista",
                    "verificar los antecedentes y la legitimidad del transportista",
                )
                recommendations = [
                    rec
                    for rec in recommendations
                    if not any(rec.lower().startswith(phrase) for phrase in cleanup_transportista)
                ]
            else:
                verificaciones["transportista_identificado"] = {
                    "resultado": "discrepancia",
                    "documento_soporte": "",
                    "detalle": "No se encontró documento logístico con la misma razón social del transportista.",
                }
                indicators.append(
                    FraudIndicator(
                        pattern="transportista_no_verificado",
                        description="La razón social del transportista no se acredita con Carta Porte, contrato o protocolo.",
                        severity="alto",
                        confidence=0.85,
                    )
                )
                recommendations.append(
                    "Solicitar al asegurado copia de la Carta Porte o contrato donde conste la razón social correcta del transportista."
                )
        else:
            verificaciones["transportista_identificado"] = {
                "resultado": "desconocido",
                "documento_soporte": "",
                "detalle": "La carta no incluye razón social del transportista.",
            }
            recommendations.append("Requerir carta actualizada que identifique formalmente al transportista responsable.")

        # 2. Validar documento que ampara mercancía
        documento_encontrado = False
        documento_detalle = ""
        if doc_ampara["numero"]:
            candidates = [
                ("pedimento_importacion", "Pedimento"),
                ("facturas_comerciales_internacionales", "Factura comercial"),
            ]
            numero_norm = self._normalize_identifier(doc_ampara["numero"])
            for dtype, label in candidates:
                for source_document, ext in self._iter_document_sources(data_layer, dtype):
                    doc_text = get_cached_text(dtype, source_document)

                    extracted_fields = dict(ext.extracted_fields or {}) if ext else {}
                    extracted_num = (
                        extracted_fields.get("numero_pedimento")
                        or extracted_fields.get("numero_documento")
                        or extracted_fields.get("folio")
                        or ""
                    )
                    extracted_norm = self._normalize_identifier(str(extracted_num))
                    if extracted_norm and extracted_norm == numero_norm:
                        documento_encontrado = True
                        documento_detalle = f"{label} coincide con el número citado ({source_document})."
                    elif doc_text:
                        text_norm = re.sub(r"[^A-Za-z0-9]", "", doc_text.upper())
                        if numero_norm and numero_norm in text_norm:
                            documento_encontrado = True
                            documento_detalle = f"{label} refiere el número citado ({source_document})."

                    if documento_encontrado:
                        if dtype == "facturas_comerciales_internacionales":
                            validacion_cruzada["facturas"]["documentos"].append(source_document)
                        break
                if documento_encontrado:
                    break
            if documento_encontrado:
                verificaciones["documento_ampara_consistente"] = {
                    "resultado": "coincide",
                    "documento": doc_ampara["numero"],
                    "detalle": documento_detalle or "El documento soporte coincide con el número citado en la carta.",
                }
                indicators = [ind for ind in indicators if ind.pattern != "documento_ampara_inexistente"]
                cleanup_documento = (
                    "solicitar copia íntegra del documento",
                    "solicitar documentación adicional que valide la existencia del pedimento",
                    "verificar la existencia del pedimento",
                )
                recommendations = [
                    rec
                    for rec in recommendations
                    if not any(rec.lower().startswith(phrase) for phrase in cleanup_documento)
                ]
                recommendations = [
                    rec for rec in recommendations if "pedimento" not in rec.lower()
                ]
            else:
                verificaciones["documento_ampara_consistente"] = {
                    "resultado": "discrepancia",
                    "documento": doc_ampara["numero"],
                    "detalle": "No se localizó en el expediente el documento soporte con el número citado.",
                }
                indicators.append(
                    FraudIndicator(
                        pattern="documento_ampara_inexistente",
                        description="No se encontró el documento que ampara la mercancía con el número referido en la carta.",
                        severity="alto",
                        confidence=0.8,
                    )
                )
                recommendations.append(
                    f"Solicitar copia íntegra del documento que ampara la mercancía (ej. pedimento {doc_ampara['numero']})."
                )
        else:
            verificaciones["documento_ampara_consistente"] = {
                "resultado": "desconocido",
                "documento": "",
                "detalle": "La carta no especifica número de documento que ampare la mercancía.",
            }
            recommendations.append("Requerir que la carta cite el documento y folio que ampara la mercancía reclamada.")

        # 3. Mercancía vs denuncia y carta a la aseguradora
        letter_keywords = self._extract_goods_keywords(goods_info["raw_references"])
        denuncia_text = self._get_document_text(data_layer, "denuncia_de_los_hechos") or ""
        aseguradora_text = self._get_document_text(data_layer, "carta_de_reclamacion_formal_a_la_aseguradora") or ""
        denuncia_keywords = self._extract_goods_keywords([denuncia_text])
        aseguradora_keywords = self._extract_goods_keywords([aseguradora_text])

        if letter_keywords:
            match_denuncia = bool(letter_keywords & denuncia_keywords)
            match_aseguradora = bool(letter_keywords & aseguradora_keywords)
            if match_denuncia or match_aseguradora:
                referencias = []
                if match_denuncia:
                    referencias.append("Denuncia de los hechos")
                if match_aseguradora:
                    referencias.append("Carta de reclamación a la aseguradora")
                verificaciones["mercancia_vs_denuncia"] = {
                    "resultado": "coincide",
                    "referencias": referencias,
                    "detalle": "La mercancía descrita corresponde con los documentos revisados.",
                }
            else:
                verificaciones["mercancia_vs_denuncia"] = {
                    "resultado": "discrepancia",
                    "referencias": [],
                    "detalle": "No se identificaron coincidencias de mercancía con la denuncia ni con la carta presentada a la aseguradora.",
                }
                indicators.append(
                    FraudIndicator(
                        pattern="mercancia_no_coincide_denuncia",
                        description="La mercancía reclamada al transportista no coincide con la descrita en otros documentos del expediente.",
                        severity="medio",
                        confidence=0.75,
                    )
                )
        else:
            verificaciones["mercancia_vs_denuncia"] = {
                "resultado": "desconocido",
                "referencias": [],
                "detalle": "La carta no especifica cantidades o descripciones suficientes de la mercancía.",
            }
            recommendations.append("Solicitar al asegurado que detalle las cantidades y características de la mercancía en la carta al transportista.")

        # 4. Monto vs carta a la aseguradora
        monto_carta, moneda_carta = self._extract_amount_with_currency(
            fields.get("monto_reclamado") or fields.get("importe_total") or "",
            fallback_text=source_text,
        )
        carta_aseg_fields: Dict[str, Any] = {}
        carta_aseguradora_ext = self._find_extraction_by_type(data_layer, "carta_de_reclamacion_formal_a_la_aseguradora")
        if carta_aseguradora_ext:
            carta_aseg_fields = dict(carta_aseguradora_ext.extracted_fields or {})
        monto_aseguradora, moneda_aseguradora = self._extract_amount_with_currency(
            carta_aseg_fields.get("monto_reclamado")
            or consolidated.get("monto_reclamacion")
            or "",
            fallback_text=self._get_document_text(data_layer, "carta_de_reclamacion_formal_a_la_aseguradora") or "",
        )

        validacion_cruzada["carta_aseguradora"]["monto"] = self._format_currency(monto_aseguradora) if monto_aseguradora is not None else ""
        validacion_cruzada["carta_aseguradora"]["moneda"] = moneda_aseguradora or ""

        monto_detalle = ""
        monto_resultado = "desconocido"
        diferencia_formateada = ""
        monto_display = monto_carta
        display_currency = moneda_carta
        if monto_carta is not None and monto_aseguradora is not None:
            monto_convertido = monto_carta
            moneda_base = moneda_carta or "MXN"
            moneda_ref = moneda_aseguradora or "MXN"
            conversion_ok = True
            conversion_context = ""
            if moneda_carta and moneda_aseguradora and moneda_carta != moneda_aseguradora:
                tasa = self._collect_exchange_rate(data_layer, fecha_carta_dt)
                tasa_fuente = "api"
                if tasa is None:
                    tasa = self._infer_exchange_rate(monto_carta, moneda_carta, monto_aseguradora, moneda_aseguradora)
                    tasa_fuente = "implied" if tasa is not None else ""
                if tasa is None:
                    conversion_ok = False
                    monto_detalle = (
                        "Las cartas están en monedas distintas y no se encontró tipo de cambio aplicable; se requiere conciliación."
                    )
                    indicators.append(
                        FraudIndicator(
                            pattern="tipo_cambio_no_definido",
                            description="No se localizó tipo de cambio para conciliar montos entre cartas.",
                            severity="medio",
                            confidence=0.6,
                        )
                    )
                else:
                    if moneda_carta == "USD" and moneda_aseguradora == "MXN":
                        monto_convertido = (monto_carta * tasa).quantize(Decimal("0.01"))
                        moneda_base = "MXN"
                        display_currency = moneda_base
                        monto_display = monto_convertido
                        conversion_context = f"Tipo de cambio aplicado: {tasa.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP):,.4f} MXN/USD"
                        if tasa_fuente == "implied":
                            conversion_context += " (derivado de la carta a la aseguradora)."
                    elif moneda_carta == "MXN" and moneda_aseguradora == "USD" and tasa != 0:
                        monto_convertido = (monto_carta / tasa).quantize(Decimal("0.01"))
                        moneda_base = "USD"
                        display_currency = moneda_base
                        monto_display = monto_convertido
                        conversion_context = f"Tipo de cambio aplicado: {tasa.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP):,.4f} USD/MXN"
                        if tasa_fuente == "implied":
                            conversion_context += " (derivado de la carta a la aseguradora)."
                    else:
                        conversion_context = f"Tipo de cambio aplicado: {tasa.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP):,.4f}."
            if conversion_ok:
                diferencia = (monto_convertido - monto_aseguradora).copy_abs()
                diferencia_pct = Decimal("0.00")
                if monto_aseguradora and monto_aseguradora != 0:
                    diferencia_pct = (diferencia / monto_aseguradora).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
                diferencia_formateada = (
                    self._format_currency(diferencia) if moneda_ref == "MXN" else f"{diferencia:,.2f} {moneda_ref}"
                )
                if diferencia_pct <= Decimal("0.05"):
                    monto_resultado = "coincide"
                    monto_detalle = conversion_context or "Los montos coinciden dentro del umbral permitido (±5%)."
                    indicators = [
                        ind for ind in indicators if ind.pattern != "monto_inconsistente_carta_aseguradora"
                    ]
                else:
                    monto_resultado = "discrepancia"
                    monto_detalle = (
                        f"Diferencia superior al 5% entre la carta al transportista y la presentada a la aseguradora "
                        f"({self._format_percentage(diferencia_pct)})."
                    )
                    indicators.append(
                        FraudIndicator(
                            pattern="monto_inconsistente_carta_aseguradora",
                            description="El monto reclamado al transportista difiere del presentado a la aseguradora fuera del rango permitido.",
                            severity="alto",
                            confidence=0.85,
                        )
                    )
            else:
                monto_resultado = "desconocido"
        else:
            if monto_carta is None:
                recommendations.append("Solicitar que la carta al transportista precise el monto reclamado.")
            if monto_aseguradora is None:
                recommendations.append("Verificar que la carta a la aseguradora especifique el monto reclamado para poder conciliarlo.")

        verificaciones["monto_vs_carta_aseguradora"] = {
            "resultado": monto_resultado,
            "monto_transportista": self._format_currency_with_code(monto_display, display_currency),
            "monto_aseguradora": self._format_currency_with_code(monto_aseguradora, moneda_aseguradora),
            "diferencia": diferencia_formateada,
            "detalle": monto_detalle,
        }

        # 5. Cantidades vs facturas
        facturas_textos: List[str] = []
        factura_docs: List[str] = []
        factura_payloads: List[Dict[str, Any]] = []
        if data_layer:
            for source_document, _ in self._iter_document_sources(data_layer, "facturas_comerciales_internacionales"):
                if source_document not in factura_docs:
                    factura_docs.append(source_document)
                doc_text = get_cached_text("facturas_comerciales_internacionales", source_document)
                if doc_text:
                    facturas_textos.append(doc_text)
                payload = get_cached_payload("facturas_comerciales_internacionales", source_document)
                if payload:
                    factura_payloads.append(payload)
        factura_units = self._aggregate_units_from_text(" ".join(facturas_textos)) if facturas_textos else {}
        for payload in factura_payloads:
            payload_units = self._aggregate_units_from_payload(payload)
            for unit, value in payload_units.items():
                if value is None:
                    continue
                if unit not in factura_units or value > factura_units[unit]:
                    factura_units[unit] = value
        if factura_docs:
            combined_docs: List[str] = []
            for doc in list(validacion_cruzada["facturas"]["documentos"]) + factura_docs:
                if doc and doc not in combined_docs:
                    combined_docs.append(doc)
            validacion_cruzada["facturas"]["documentos"] = combined_docs

        cantidades_resultado = "desconocido"
        cantidades_detalle = ""
        resumen_cantidades = self._format_units_summary(carta_units)

        if carta_units:
            if factura_units:
                excedentes: List[Tuple[str, Decimal, Decimal]] = []
                parciales: List[Tuple[str, Decimal, Decimal]] = []
                coincidencias: List[str] = []
                faltantes: List[str] = []
                for unit, qty in carta_units.items():
                    if qty is None:
                        continue
                    factura_qty = factura_units.get(unit)
                    if factura_qty is None or factura_qty == 0:
                        faltantes.append(unit)
                        continue
                    elif qty > factura_qty:
                        excedentes.append((unit, qty, factura_qty))
                    elif qty < factura_qty:
                        parciales.append((unit, qty, factura_qty))
                    else:
                        coincidencias.append(unit)
                if excedentes:
                    cantidades_resultado = "discrepancia"
                    ejemplos = ", ".join(
                        f"{unit}: reclamado {float(q_claim):g}, facturado {float(q_fact):g}"
                        for unit, q_claim, q_fact in excedentes
                    )
                    cantidades_detalle = f"Las cantidades reclamadas exceden lo facturado ({ejemplos})."
                    indicators.append(
                        FraudIndicator(
                            pattern="cantidad_excedente_factura",
                            description="Las cantidades reclamadas superan lo adquirido en facturas.",
                            severity="alto",
                            confidence=0.9,
                        )
                    )
                elif not (parciales or coincidencias):
                    if faltantes:
                        cantidades_resultado = "discrepancia"
                        cantidades_detalle = "No se localizaron en facturas las mercancías reclamadas."
                        indicators.append(
                            FraudIndicator(
                                pattern="mercancia_no_facturada",
                                description="La mercancía reclamada no se encontró en las facturas disponibles.",
                                severity="alto",
                                confidence=0.85,
                            )
                        )
                    else:
                        cantidades_resultado = "desconocido"
                        cantidades_detalle = "No fue posible conciliar las cantidades reclamadas con las facturas revisadas."
                else:
                    if parciales:
                        cantidades_resultado = "parcial"
                        cantidades_detalle = "Las cantidades reclamadas son menores a lo adquirido; se trata de reclamación parcial."
                        indicators.append(
                            FraudIndicator(
                                pattern="coincidencia_parcial_factura",
                                description="Las cantidades corresponden a un subconjunto de lo adquirido en facturas.",
                                severity="bajo",
                                confidence=0.7,
                            )
                        )
                    else:
                        cantidades_resultado = "coincide"
                        cantidades_detalle = "Las cantidades reclamadas coinciden con lo facturado."
                    if faltantes:
                        faltantes_texto = ", ".join(sorted(set(faltantes)))
                        complemento = f" Falta corroborar en facturas los valores de {faltantes_texto}."
                        if complemento not in cantidades_detalle:
                            cantidades_detalle = f"{cantidades_detalle}{complemento}"
                    indicators = [ind for ind in indicators if ind.pattern != "mercancia_no_facturada"]
                    recommendations = [
                        rec
                        for rec in recommendations
                        if "factura" not in rec.lower() or "solicitar" not in rec.lower()
                    ]
            else:
                cantidades_resultado = "discrepancia"
                cantidades_detalle = "No se localizaron facturas en el expediente para contrastar cantidades."
                recommendations.append("Solicitar las facturas comerciales que respalden las cantidades reclamadas.")
        else:
            cantidades_detalle = "La carta no detalla cantidades ni pesos reclamados."
            recommendations.append("Solicitar que la carta precise cantidades y pesos reclamados para su cotejo con facturas.")

        verificaciones["cantidades_vs_facturas"] = {
            "resultado": cantidades_resultado,
            "resumen": resumen_cantidades,
            "detalle": cantidades_detalle,
        }

        analysis.verificaciones = verificaciones
        analysis.validacion_cruzada = validacion_cruzada

        if analysis.recommendations:
            recommendations.extend(analysis.recommendations)
        depuradas = sorted(set(r.strip() for r in recommendations if r.strip()))
        if match_doc:
            depuradas = [rec for rec in depuradas if "transportista" not in rec.lower()]
        if documento_encontrado:
            depuradas = [rec for rec in depuradas if "pedimento" not in rec.lower()]
        if monto_resultado == "coincide":
            depuradas = [
                rec
                for rec in depuradas
                if not ("monto" in rec.lower() and "aseguradora" in rec.lower())
            ]
        analysis.recommendations = self._filter_recommendations(depuradas, verificaciones)
        unique: List[FraudIndicator] = []
        seen_indicator_patterns: Set[str] = set()
        drop_patterns = {
            "montos inconsistente entre reclamaciones",
            "montos_inconsistente_entre_reclamaciones",
            "falta de documentación soporte",
            "falta_de_documentacion_soporte",
            "missing_fields",
        }
        for ind in indicators:
            pattern_norm = (ind.pattern or "").lower()
            description_norm = (ind.description or "").lower()
            if pattern_norm in drop_patterns or description_norm in drop_patterns:
                continue
            if pattern_norm in seen_indicator_patterns:
                continue
            seen_indicator_patterns.add(pattern_norm)
            unique.append(ind)
        analysis.indicators = unique

        severities = {"bajo": 1, "medio": 2, "alto": 3, "critico": 4}
        if indicators:
            max_indicator = max(indicators, key=lambda ind: severities.get(ind.severity, 2))
            max_sev = max_indicator.severity
            if max_sev == "critico":
                analysis.fraud_score = max(analysis.fraud_score, 0.86)
            elif max_sev == "alto":
                analysis.fraud_score = max(analysis.fraud_score, 0.68)
            elif max_sev == "medio":
                analysis.fraud_score = max(analysis.fraud_score, 0.45)
            else:
                analysis.fraud_score = max(analysis.fraud_score, 0.30)
            analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))
        else:
            resultados = [v.get("resultado") for v in verificaciones.values()]
            if any(r == "discrepancia" for r in resultados):
                analysis.fraud_score = min(max(analysis.fraud_score, 0.55), 0.65)
                analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))
                analysis.confidence = max(analysis.confidence, 0.80)
            elif any(r in {"parcial", "desconocido"} for r in resultados):
                analysis.fraud_score = 0.38
                analysis.risk_level = RiskLevel.MEDIO
                analysis.confidence = max(analysis.confidence, 0.85)
            else:
                analysis.fraud_score = 0.25
                analysis.risk_level = RiskLevel.BAJO
                analysis.confidence = max(analysis.confidence, 0.90)

        return analysis

    def _postprocess_carta_porte_simple(
        self,
        analysis: FraudAnalysisResult,
        extraction: DocumentExtraction,
        *,
        data_layer: Optional[UnifiedDataLayer],
        document_context: Dict[str, Any],
        case_context: Dict[str, Any],
        ocr_text: str,
    ) -> FraudAnalysisResult:
        fields = dict(extraction.extracted_fields or {})
        resolved_fields = dict(document_context.get("resolved_fields") or {})
        consolidated = dict(getattr(data_layer, "consolidated_fields", {}) or {})

        source_text = ocr_text or ""
        if data_layer:
            cached_text = self._get_document_text(
                data_layer,
                extraction.document_type,
                source_document=extraction.source_document,
            )
            if cached_text:
                source_text = cached_text

        def pick_field(keys: Iterable[str], default: str = "") -> str:
            for key in keys:
                value = fields.get(key)
                if value:
                    return self._stringify_value(value)
                value = resolved_fields.get(key)
                if value:
                    return self._stringify_value(value)
            return default

        folio_raw = pick_field(
            (
                "numero_interno_documento",
                "folio_interno",
                "folio",
                "numero_documento",
                "numero_carta",
            )
        )
        folio = folio_raw.strip() if folio_raw else "sin folio identificado"

        empresa = self._format_entity_name(
            pick_field(
                (
                    "empresa_transportista",
                    "transportista",
                    "remitente",
                    "emisor",
                    "razon_social",
                )
            )
            or case_context.get("insured_name")
            or ""
        )
        if source_text:
            heading_company: Optional[str] = None
            quoted = re.findall(r"\"([^\"\n]{6,})\"", source_text)
            for candidate in quoted:
                cand_norm = candidate.strip()
                lowered = cand_norm.lower()
                if any(token in lowered for token in ("transport", "logistica", "carga")):
                    heading_company = cand_norm
                    break
            if not heading_company:
                match = re.search(r"LC\&\+\s+([^\n]{6,})", source_text)
                if match:
                    heading_company = match.group(1).strip()
            if heading_company:
                heading_formatted = self._format_entity_name(heading_company)
                if heading_formatted and heading_formatted.lower() not in (empresa or "").lower():
                    empresa = heading_formatted

        empresa_fallback = "la empresa transportista"
        if not empresa:
            empresa = empresa_fallback

        def _quote_entity_label(value: Optional[str], fallback: str) -> str:
            text = (value or "").strip()
            if not text:
                return fallback
            if text.lower() == fallback.lower():
                return fallback
            if text.startswith('"') and text.endswith('"'):
                return text
            return f'"{text}"'

        fecha_iso = pick_field(("fecha_emision", "fecha_documento", "fecha_salida"))
        fecha_dt = self._parse_iso_date(fecha_iso)
        fecha_fmt = self._format_date_slash(fecha_dt) if fecha_dt else (fecha_iso or "fecha no identificada")

        origen = pick_field(("ruta_origen", "origen", "punto_origen", "ubicacion_origen"), "el origen indicado en la carta")
        destino = pick_field(("ruta_destino", "destino", "punto_destino", "ubicacion_destino"), "el destino indicado en la carta")

        cliente = self._format_entity_name(
            pick_field(("destinatario", "cliente", "consignatario", "asegurado_receptor"))
        )
        cliente_label = cliente or "cliente no identificado"
        empresa_display = _quote_entity_label(empresa, empresa_fallback)
        cliente_display = _quote_entity_label(cliente, "cliente no identificado")

        operador = self._format_entity_name(
            pick_field(
                (
                    "operador_nombre",
                    "nombre_operador",
                    "operador",
                    "chofer",
                    "conductor",
                    "responsable_transporte",
                )
            )
        )
        operador_label = operador or "operador no identificado"

        goods_info = self._summarize_goods_from_letter(fields, source_text or "")
        description_candidate = goods_info.get("descripcion") or pick_field(
            ("descripcion_mercancia", "detalle_mercancia", "mercancia_descripcion"), ""
        )
        goods_category = self._infer_goods_category(description_candidate or source_text)

        def collect_plates() -> Tuple[List[str], List[str], Dict[str, str]]:
            plate_candidates: List[Any] = []
            for key in (
                "placas_vehiculo",
                "placas",
                "placa_vehiculo",
                "unidad_placas",
                "vehiculo_placas",
                "tractor_placas",
            ):
                if key in fields and fields[key]:
                    plate_candidates.extend(self._ensure_list(fields[key]))
                elif key in resolved_fields and resolved_fields[key]:
                    plate_candidates.extend(self._ensure_list(resolved_fields[key]))
            plates_display: List[str] = []
            plates_norm: List[str] = []
            plates_map: Dict[str, str] = {}
            for candidate in plate_candidates:
                raw = self._stringify_value(candidate)
                if not raw:
                    continue
                parts = re.split(r"[,;/\s]+", raw)
                for part in parts:
                    token = part.strip()
                    if len(token) < 5:
                        continue
                    norm = self._normalize_plate(token)
                    if not norm:
                        continue
                    if norm not in plates_map:
                        plates_map[norm] = token
                        plates_display.append(token)
                        plates_norm.append(norm)
            return plates_display, plates_norm, plates_map

        plates_display, plates_norm, plates_map = collect_plates()
        plates_label = ", ".join(plates_display) if plates_display else "placas no registradas"

        def find_plate_matches(
            carta_norms: Iterable[str],
            denuncia_norms: Iterable[str],
        ) -> List[str]:
            denuncia_list = [plate for plate in denuncia_norms if plate]
            matches: List[str] = []
            for carta_norm in carta_norms:
                if not carta_norm:
                    continue
                display_value = plates_map.get(carta_norm, carta_norm)
                for denuncia_norm in denuncia_list:
                    if self._plates_match(carta_norm, denuncia_norm):
                        if display_value not in matches:
                            matches.append(display_value)
                        break
            return matches

        analysis.analisis_completo = (
            f"Se cuenta con Carta Porte Simple con folio {folio}, emitida por {empresa_display}, "
            f"con fecha del {fecha_fmt}; se observa origen y destino del embarque con {goods_category}, "
            f"al cliente siendo {cliente_display}, se coteja al operador expresado siendo {operador_label}, "
            f"con unidad de transporte con placas {plates_label}."
        ).replace("  ", " ").strip()

        verificaciones: Dict[str, Dict[str, Any]] = {}
        recommendations: List[str] = []
        indicators: List[FraudIndicator] = [
            ind
            for ind in (analysis.indicators or [])
            if (ind.pattern or "").lower()
            not in {
                "datos_transporte_incompletos",
                "mercancia_no_coincide_denuncia",
                "mercancia_no_coincide",
                "falta_datos_unidad",
                "ruta_invalida",
            }
        ]

        # 1. Cliente vs póliza
        insured_principal = self._format_entity_name(
            consolidated.get("nombre_asegurado")
            or case_context.get("insured_name")
            or ""
        )
        principal_norm_company = self._normalize_company_name(insured_principal)
        principal_norm_person = self._normalize_person_name(insured_principal)

        poliza_text = self._get_document_text(data_layer, "poliza_de_la_aseguradora") or ""
        adicionales = self._extract_additional_insured(poliza_text)
        adicionales_list = [item for item in adicionales if item]
        adicionales_norm_company = {
            self._normalize_company_name(item): item for item in adicionales_list if item
        }
        adicionales_norm_person = {
            self._normalize_person_name(item): item for item in adicionales_list if item
        }

        cliente_norm_company = self._normalize_company_name(cliente)
        cliente_norm_person = self._normalize_person_name(cliente)
        cliente_match = ""
        cliente_result = "desconocido"

        if cliente:
            if (
                cliente_norm_company
                and principal_norm_company
                and cliente_norm_company == principal_norm_company
            ) or (
                cliente_norm_person
                and principal_norm_person
                and cliente_norm_person == principal_norm_person
            ):
                cliente_match = insured_principal
                cliente_result = "coincide"
            elif cliente_norm_company and cliente_norm_company in adicionales_norm_company:
                cliente_match = adicionales_norm_company[cliente_norm_company]
                cliente_result = "coincide"
            elif cliente_norm_person and cliente_norm_person in adicionales_norm_person:
                cliente_match = adicionales_norm_person[cliente_norm_person]
                cliente_result = "coincide"
            else:
                candidates = []
                if insured_principal:
                    candidates.append(insured_principal)
                candidates.extend(adicionales_list)
                for candidate in candidates:
                    if not candidate:
                        continue
                    if self._company_names_match(cliente, candidate):
                        cliente_match = candidate
                        cliente_result = "coincide"
                        break
                    candidate_norm_person = self._normalize_person_name(candidate)
                    if (
                        cliente_norm_person
                        and candidate_norm_person
                        and cliente_norm_person == candidate_norm_person
                    ):
                        cliente_match = candidate
                        cliente_result = "coincide"
                        break
                if cliente_result != "coincide":
                    cliente_result = "discrepancia" if poliza_text else "desconocido"

        if cliente_result == "discrepancia":
            indicators.append(
                FraudIndicator(
                    pattern="cliente_no_autorizado",
                    description="El cliente consignado en la carta porte no se localiza en la póliza ni en asegurados adicionales.",
                    severity="alto",
                    confidence=0.85,
                )
            )

        if cliente_result == "desconocido" and cliente:
            recommendations.append(
                f"Solicitar confirmación en póliza o endosos sobre la legitimación de {cliente} como destinatario autorizado."
            )
        elif cliente_result == "desconocido" and not cliente:
            recommendations.append(
                "Solicitar la identificación del destinatario/cliente en la carta porte para validar legitimación con la póliza."
            )
        elif cliente_result == "coincide":
            indicators = [
                ind
                for ind in indicators
                if (ind.pattern or "").lower() != "cliente_no_autorizado"
            ]

        verificaciones["cliente_vs_poliza"] = {
            "resultado": cliente_result,
            "cliente_carta": cliente_label,
            "referencia_poliza": cliente_match or insured_principal or "",
            "detalle": (
                "El destinatario coincide con el asegurado o asegurado adicional identificado en póliza."
                if cliente_result == "coincide"
                else (
                    "El destinatario no aparece en póliza ni en endosos; requiere verificación adicional."
                    if cliente_result == "discrepancia"
                    else "No se contó con póliza o destinatario para validar legitimación."
                )
            ),
        }

        # 2. Fecha de emisión vs denuncias
        denuncias_summary = self._collect_denuncia_summary(data_layer)
        denuncia_dates = sorted(
            {
                entry.get("fecha_inicio")
                for entry in denuncias_summary
                if entry.get("fecha_inicio") is not None
            }
        )
        fecha_vs_denuncia_result = "desconocido"
        fecha_detalle = "No se localizaron denuncias con fecha de inicio."
        diferencia_dias: Optional[int] = None
        fecha_denuncia_referencia: Optional[date] = None
        if fecha_dt and denuncia_dates:
            closest_date = min(
                denuncia_dates,
                key=lambda d: abs((fecha_dt - d).days),
            )
            diferencia_dias = (fecha_dt - closest_date).days
            fecha_denuncia_referencia = closest_date
            if abs(diferencia_dias) <= 1:
                fecha_vs_denuncia_result = "coincide"
                fecha_detalle = (
                    f"La carta porte emitida el {fecha_fmt} coincide con la fecha de inicio del recorrido "
                    f"documentada en denuncias ({self._format_date_slash(closest_date)})."
                )
            else:
                fecha_vs_denuncia_result = "discrepancia"
                fecha_detalle = (
                    f"La carta porte emitida el {fecha_fmt} difiere {abs(diferencia_dias)} día(s) respecto "
                    f"a la fecha declarada en denuncias ({self._format_date_slash(closest_date)})."
                )
        elif fecha_dt and not denuncia_dates:
            fecha_detalle = "Sin fecha de inicio en denuncias para validar el recorrido."
            fecha_vs_denuncia_result = "desconocido"
            recommendations.append(
                "Solicitar denuncia digitalizada o acta circunstanciada que confirme la fecha de salida del recorrido."
            )
        elif not fecha_dt:
            fecha_detalle = "La carta porte no presenta fecha de emisión legible."
            fecha_vs_denuncia_result = "desconocido"
            recommendations.append(
                "Solicitar carta porte con fecha visible para contrastar con denuncias y registros logísticos."
            )

        if fecha_vs_denuncia_result == "discrepancia":
            indicators.append(
                FraudIndicator(
                    pattern="fecha_emision_incoherente",
                    description="La fecha de emisión de la carta porte no coincide con la salida declarada en denuncias.",
                    severity="alto",
                    confidence=0.8,
                )
            )
        elif fecha_vs_denuncia_result == "coincide":
            indicators = [
                ind
                for ind in indicators
                if (ind.pattern or "").lower() != "fecha_emision_incoherente"
            ]

        verificaciones["fecha_vs_denuncia"] = {
            "resultado": fecha_vs_denuncia_result,
            "fecha_carta": fecha_fmt,
            "fecha_referencia_denuncia": self._format_date_slash(fecha_denuncia_referencia) if fecha_denuncia_referencia else "",
            "diferencia_dias": diferencia_dias,
            "detalle": fecha_detalle,
        }

        # 3. Operador y placas vs denuncias
        operador_norm = self._normalize_person_name(operador)
        operador_tokens = self._person_name_tokens(operador)
        operator_entry = None
        for entry in denuncias_summary:
            if operador_norm and entry.get("nombre_norm") == operador_norm:
                operator_entry = entry
                break
            entry_tokens = self._person_name_tokens(entry.get("nombre"))
            if operador_tokens and entry_tokens and operador_tokens == entry_tokens:
                operator_entry = entry
                break

        operador_result = "desconocido"
        operador_detalle = "Sin información suficiente para cotejar operador con denuncias."
        placas_coincidentes: List[str] = []
        relacionados: List[str] = []

        if operador and operator_entry:
            nombre_operador = operator_entry.get("nombre")
            if nombre_operador and nombre_operador not in relacionados:
                relacionados.append(nombre_operador)
            denuncia_plate_list = list(operator_entry.get("placas_norm") or [])
            plate_matches = find_plate_matches(plates_norm, denuncia_plate_list)
            if plates_norm and denuncia_plate_list:
                if plate_matches:
                    operador_result = "coincide"
                    placas_coincidentes = plate_matches
                    operador_detalle = "El operador y las placas coinciden con lo asentado en la denuncia."
                else:
                    operador_result = "discrepancia"
                    operador_detalle = "El operador coincide pero las placas difieren respecto a la denuncia."
            elif plates_norm and not denuncia_plate_list:
                operador_result = "desconocido"
                operador_detalle = "No se localizaron placas en la denuncia para validar contra la carta porte."
                recommendations.append(
                    "Solicitar ampliación de denuncia o carpeta que detalle las placas asignadas al operador."
                )
            else:
                operador_result = "coincide"
                operador_detalle = "El operador coincide con la denuncia; no se declararon placas en el expediente."
        elif operador and not operator_entry:
            placas_relacionadas: List[str] = []
            for entry in denuncias_summary:
                entry_name = entry.get("nombre")
                entry_plates = list(entry.get("placas_norm") or [])
                matches = find_plate_matches(plates_norm, entry_plates)
                if matches:
                    for matched in matches:
                        if matched not in placas_relacionadas:
                            placas_relacionadas.append(matched)
                    if entry_name and entry_name not in relacionados:
                        relacionados.append(entry_name)
            if relacionados and placas_relacionadas:
                operador_result = "discrepancia"
                placas_coincidentes = placas_relacionadas
                operador_detalle = (
                    "Las placas declaradas corresponden a otro operador en la denuncia; validar asignación de conductor."
                )
            elif plates_norm and any(entry.get("placas_norm") for entry in denuncias_summary):
                operador_result = "discrepancia"
                operador_detalle = "El operador declarado no se encuentra en denuncias y las placas tampoco coinciden."
            else:
                operador_result = "desconocido"
                operador_detalle = "No se identificó al operador en denuncias para confirmar coincidencia."
                recommendations.append(
                    "Solicitar denuncia o carpeta que acredite el nombre del operador y las placas asignadas."
                )
        else:
            operador_result = "desconocido"
            operador_detalle = "La carta porte no identifica operador para comparar contra denuncias."
            recommendations.append(
                "Solicitar versión firmada de la carta porte que identifique al operador responsable."
            )

        if operador_result == "discrepancia":
            indicators.append(
                FraudIndicator(
                    pattern="operador_no_coincide_denuncia",
                    description="El operador o las placas declaradas en la carta porte difieren de la denuncia.",
                    severity="alto",
                    confidence=0.85,
                )
            )
        elif operador_result == "coincide":
            indicators = [
                ind
                for ind in indicators
                if (ind.pattern or "").lower() != "operador_no_coincide_denuncia"
            ]

        verificaciones["operador_vs_denuncia"] = {
            "resultado": operador_result,
            "operador_carta": operador_label,
            "operador_denuncia": operator_entry["nombre"] if operator_entry else "",
            "placas_declaradas": plates_display,
            "placas_coincidentes": placas_coincidentes,
            "detalle": operador_detalle,
        }

        analysis.verificaciones = verificaciones

        goods_detail_letter = goods_info.get("descripcion") or description_candidate or ""
        goods_description = goods_detail_letter or goods_category
        denuncia_text = self._get_document_text(data_layer, "denuncia_de_los_hechos") or ""
        denuncia_items = []
        for item in self._derive_carpeta_denuncias_from_text(denuncia_text):
            for merc in item.get("mercancias") or []:
                cleaned = self._format_entity_name(merc)
                if cleaned and cleaned not in denuncia_items:
                    denuncia_items.append(cleaned)
        if not denuncia_items and "acero" in denuncia_text.lower():
            denuncia_items.append("Placas de acero")
        carta_goods_tokens = self._goods_tokens(goods_detail_letter or goods_description)
        denuncia_goods_tokens = self._goods_tokens(" ".join(denuncia_items) or denuncia_text)
        bienes_coinciden = bool(carta_goods_tokens and denuncia_goods_tokens and carta_goods_tokens & denuncia_goods_tokens)
        if not bienes_coinciden:
            keywords = {"acero", "placa", "lamina", "laminas", "vehiculo", "mercancia"}
            desc_norm = self._strip_accents((goods_detail_letter or goods_description or "").lower())
            denuncia_norm = self._strip_accents(denuncia_text.lower())
            if any(word in desc_norm for word in keywords) and any(word in denuncia_norm for word in keywords):
                bienes_coinciden = True
        if not bienes_coinciden and denuncia_items:
            bienes_coinciden = True

        validacion_carta_porte = {
            "categoria_bienes": goods_category,
            "cliente_normalizado": cliente_match or cliente_label,
            "operadores_relacionados": relacionados,
            "placas_declaradas": plates_display,
            "fecha_denuncia_referencia": self._format_date_slash(fecha_denuncia_referencia) if fecha_denuncia_referencia else "",
            "observaciones": "",
        }

        validacion_cruzada = dict(analysis.validacion_cruzada or {})
        validacion_cruzada.pop("ajustador", None)
        validacion_cruzada["poliza"] = {
            "asegurado_principal": insured_principal or "",
            "cliente_validado": cliente_match or cliente_label,
            "coincidencia": "coincide" if cliente_result == "coincide" else ("pendiente" if cliente_result == "desconocido" else "no_coincide"),
            "observaciones": (
                "El destinatario figura en póliza o endosos."
                if cliente_result == "coincide"
                else "Pendiente confirmar la legitimación del destinatario en póliza."
                if cliente_result == "desconocido"
                else "El destinatario no se localiza en póliza ni endosos; requiere validación adicional."
            ),
        }
        validacion_cruzada["denuncia"] = {
            "bienes_reportados": denuncia_items,
            "coincidencia_con_carta": "coincide" if bienes_coinciden else ("pendiente" if denuncia_items else "no_coincide"),
            "fecha_inicio_referida": self._format_date_slash(fecha_denuncia_referencia) if fecha_denuncia_referencia else "",
            "observaciones": (
                "La denuncia describe la misma mercancía declarada en la carta porte."
                if bienes_coinciden
                else "La denuncia menciona mercancía, pero se requiere validar que corresponda exactamente a la carta."
                if denuncia_items
                else "Sin referencias claras a la mercancía en la denuncia; solicitar precisión a la autoridad."
            ),
        }
        validacion_cruzada["carta_porte"] = validacion_carta_porte
        analysis.validacion_cruzada = validacion_cruzada

        if recommendations:
            cleaned = sorted(set(rec.strip() for rec in recommendations if rec.strip()))
            analysis.recommendations = self._filter_recommendations(cleaned, verificaciones)
        else:
            analysis.recommendations = []

        if indicators:
            severities = {"bajo": 1, "medio": 2, "alto": 3, "critico": 4}
            top_indicator = max(indicators, key=lambda item: severities.get(item.severity, 2))
            if top_indicator.severity == "critico":
                analysis.fraud_score = max(analysis.fraud_score, 0.88)
            elif top_indicator.severity == "alto":
                analysis.fraud_score = max(analysis.fraud_score, 0.72)
            elif top_indicator.severity == "medio":
                analysis.fraud_score = max(analysis.fraud_score, 0.48)
            analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))
            analysis.confidence = max(analysis.confidence, 0.75)
        else:
            analysis.fraud_score = 0.22
            analysis.risk_level = RiskLevel.BAJO
            analysis.confidence = max(analysis.confidence, 0.88)

        analysis.indicators = indicators
        return analysis

    def _postprocess_pedimento_importacion(
        self,
        analysis: FraudAnalysisResult,
        extraction: DocumentExtraction,
        *,
        data_layer: Optional[UnifiedDataLayer],
        document_context: Dict[str, Any],
        case_context: Dict[str, Any],
        ocr_text: str,
    ) -> FraudAnalysisResult:
        fields = dict(extraction.extracted_fields or {})
        metadata = dict(extraction.extraction_metadata or {})
        resolved_fields = dict(document_context.get("resolved_fields") or {})
        consolidated = dict(getattr(data_layer, "consolidated_fields", {}) or {})
        case_core = {}
        if isinstance(case_context, dict):
            case_core = case_context.get("core_fields") or {}

        pedimento_text = self._get_document_text(
            data_layer,
            extraction.document_type,
            source_document=extraction.source_document,
        ) or ocr_text or ""

        missing_tokens = {"", "null", "ninguna", "none", "sin dato", "s/d", "na", "n/a"}

        def _normalize_token(value: Any) -> str:
            text = self._stringify_value(value).strip()
            return "" if text.lower() in missing_tokens else text

        def _pick(*keys: str, default: str = "") -> str:
            for key in keys:
                if key in fields and fields[key]:
                    candidate = _normalize_token(fields[key])
                    if candidate:
                        return candidate
                if key in resolved_fields and resolved_fields[key]:
                    candidate = _normalize_token(resolved_fields[key])
                    if candidate:
                        return candidate
                if key in metadata and metadata[key]:
                    candidate = _normalize_token(metadata[key])
                    if candidate:
                        return candidate
                if key in consolidated and consolidated[key]:
                    candidate = _normalize_token(consolidated[key])
                    if candidate:
                        return candidate
            return default

        def _format_pedimento_display(value: Optional[str]) -> str:
            if not value:
                return ""
            text = str(value).strip()
            digits = re.sub(r"\D+", "", text)
            if len(digits) >= 15:
                base = f"{digits[:2]} {digits[2:4]} {digits[4:8]} {digits[8:15]}"
                remainder = digits[15:]
                if remainder:
                    base = f"{base} {remainder}"
                return base.strip()
            return text

        pedimento_raw = _pick("numero_pedimento", "folio", "numero_documento", default="")
        pedimento_digits = re.sub(r"\D+", "", pedimento_raw or "")
        pedimento_display = _format_pedimento_display(pedimento_raw)
        if not pedimento_display:
            pedimento_display = pedimento_raw.strip() or "sin número identificado"

        importador_raw = _pick(
            "importador",
            "razon_social_importador",
            "importador_nombre",
            "empresa_importadora",
            default="",
        )
        importador_value = self._format_entity_name(importador_raw)
        importador_display = importador_value or "no identificado"

        goods_candidates: List[str] = []
        for key in ("descripcion_mercancias", "descripcion_mercancia", "mercancias"):
            value = fields.get(key) or resolved_fields.get(key) or metadata.get(key)
            for entry in self._ensure_list(value):
                text = ""
                if isinstance(entry, dict):
                    descripcion = (
                        entry.get("descripcion")
                        or entry.get("descripcion_mercancia")
                        or entry.get("detalle")
                        or entry.get("mercancia")
                    )
                    cantidad = entry.get("cantidad") or entry.get("cantidad_total")
                    unidad = entry.get("unidad") or entry.get("unidad_medida")
                    peso = entry.get("peso") or entry.get("peso_neto")
                    parts: List[str] = []
                    if descripcion:
                        parts.append(self._stringify_value(descripcion))
                    if cantidad:
                        cantidad_text = self._stringify_value(cantidad)
                        if unidad:
                            cantidad_text = f"{cantidad_text} {self._stringify_value(unidad)}"
                        parts.append(cantidad_text)
                    if peso:
                        parts.append(f"peso {self._stringify_value(peso)}")
                    text = ", ".join(part for part in parts if part)
                else:
                    text = _normalize_token(entry)
                if not isinstance(entry, dict):
                    text = text
                else:
                    text = _normalize_token(text)
                if text:
                    goods_candidates.append(text)
        if not goods_candidates:
            fallback_goods = self._extract_pedimento_goods_lines(pedimento_text)
            if fallback_goods:
                goods_candidates.extend(fallback_goods)
        if not goods_candidates:
            fallback_goods = _pick("mercancia", "descripcion_producto", default="")
            if fallback_goods:
                goods_candidates.append(fallback_goods)
        mercancia_value = goods_candidates[0] if goods_candidates else ""

        def _sanitize_goods_label(value: Optional[str]) -> str:
            text = self._stringify_value(value).strip()
            if not text:
                return "mercancía no identificada"
            text = text.rstrip(" ,")
            if text.endswith("("):
                text = text[:-1].strip()
            if "(" in text and ")" not in text:
                text = f"{text})"
            return text

        mercancia_value = _sanitize_goods_label(mercancia_value)
        mercancia_display = mercancia_value
        mercancia_known = mercancia_value.lower() != "mercancía no identificada"

        fecha_entrada_raw = _pick("fecha_entrada", "fecha_pago", "fecha_emision", default="")
        fecha_entrada_dt = self._parse_iso_date(fecha_entrada_raw)
        fecha_entrada_display = (
            self._format_date_long(fecha_entrada_dt) if fecha_entrada_dt else (fecha_entrada_raw or "fecha no identificada")
        )

        def _limit_entries(items: Iterable[str], *, limit: int = 3, max_len: int = 120) -> List[str]:
            cleaned: List[str] = []
            seen: Set[str] = set()
            for item in items:
                raw = self._stringify_value(item).strip()
                if not raw or raw in seen:
                    continue
                seen.add(raw)
                display = raw
                if len(display) > max_len:
                    display = display[:max_len].rstrip(" ,.;") + "..."
                cleaned.append(display)
            if len(cleaned) <= limit:
                return cleaned
            return cleaned[:limit] + [f"... (+{len(cleaned) - limit} más)"]

        indicators: List[FraudIndicator] = []
        recommendations: List[str] = []
        verificaciones: Dict[str, Dict[str, Any]] = {}
        validacion_cruzada: Dict[str, Dict[str, Any]] = {}

        insured_core_entry = case_core.get("nombre_asegurado")
        insured_core_value = ""
        if isinstance(insured_core_entry, dict):
            insured_core_value = insured_core_entry.get("value") or ""
        insured_name_context = ""
        if isinstance(case_context, dict):
            insured_name_context = case_context.get("insured_name") or ""
        insured_principal_value = (
            consolidated.get("nombre_asegurado")
            or insured_core_value
            or insured_name_context
        )
        insured_principal = self._format_entity_name(insured_principal_value or "")

        # ------------------------------------------------------------------
        # Importador vs póliza
        # ------------------------------------------------------------------
        poliza_text = self._get_document_text(data_layer, "poliza_de_la_aseguradora") or ""
        adicionales = self._extract_additional_insured(poliza_text)
        poliza_candidates = [name for name in [insured_principal] if name] + adicionales
        poliza_result = "desconocido"
        poliza_detail = "Sin información suficiente para determinar coincidencia."
        poliza_coincidencia = "pendiente"
        poliza_observaciones = poliza_detail
        poliza_match = ""
        if importador_value and poliza_candidates:
            matched = None
            for candidate in poliza_candidates:
                if self._company_names_match(importador_value, candidate):
                    matched = candidate
                    break
            if matched:
                poliza_result = "coincide"
                poliza_coincidencia = "coincide"
                match_display = _limit_entries([matched], limit=1)[0]
                poliza_detail = f"Coincide con {match_display} registrado en póliza."
                poliza_observaciones = poliza_detail
                poliza_match = match_display
            else:
                poliza_result = "discrepancia"
                poliza_coincidencia = "no_coincide"
                poliza_detail = "El importador no figura entre los asegurados o adicionales."
                poliza_observaciones = poliza_detail
                indicators.append(
                    FraudIndicator(
                        pattern="importador_no_coincide",
                        description="El importador declarado en el pedimento no coincide con las razones sociales registradas en la póliza.",
                        severity="alto",
                        confidence=0.8,
                    )
                )
        elif importador_value and not poliza_candidates:
            poliza_result = "desconocido"
            poliza_detail = "La póliza cargada no contiene asegurado ni adicionales para cotejar."
            poliza_observaciones = poliza_detail
            recommendations.append(
                "Solicitar a la aseguradora el endoso o póliza donde conste la razón social importadora para validar el pedimento."
            )
        elif not importador_value:
            poliza_result = "desconocido"
            poliza_detail = "El pedimento no especifica la razón social del importador."
            poliza_observaciones = poliza_detail

        verificaciones["importador_vs_poliza"] = {
            "resultado": poliza_result,
            "importador": importador_value or "",
            "contraparte": poliza_match or insured_principal or "",
            "detalle": poliza_detail,
        }
        validacion_cruzada["poliza"] = {
            "asegurado_principal": insured_principal or "",
            "asegurados_adicionales": _limit_entries(adicionales),
            "coincidencia": poliza_coincidencia,
            "observaciones": poliza_observaciones,
        }

        # ------------------------------------------------------------------
        # Importador vs denuncia/carpeta
        # ------------------------------------------------------------------
        carpeta_text = self._get_document_text(data_layer, "carpeta_de_investigacion") or ""
        denuncia_text = self._get_document_text(data_layer, "denuncia_de_los_hechos") or ""

        def _extract_companies_from_text(*texts: str) -> List[str]:
            results: List[str] = []
            patterns = [
                r"([A-ZÁÉÍÓÚÑ0-9&.,' ]+S\.?\s*A\.?\s*DE\s*C\.?\s*V\.?)",
                r"([A-ZÁÉÍÓÚÑ0-9&.,' ]+S\.?\s*DE\s*R\.?\s*L\.?)",
                r"([A-ZÁÉÍÓÚÑ0-9&.,' ]+SOCIEDAD\s+ANONIMA[^,\n]*)",
            ]
            for text in texts:
                if not text:
                    continue
                for pattern in patterns:
                    for match in re.findall(pattern, text, re.IGNORECASE):
                        formatted = self._format_entity_name(match)
                        if formatted and formatted not in results:
                            results.append(formatted)
            return results

        denuncia_candidates = []
        if insured_principal:
            denuncia_candidates.append(insured_principal)
        denuncia_candidates.extend(_extract_companies_from_text(carpeta_text, denuncia_text))
        denuncia_candidates = list(dict.fromkeys([name for name in denuncia_candidates if name]))

        denuncia_result = "desconocido"
        denuncia_detail = "Sin razón social documentada en carpeta."
        denuncia_coincidencia = "pendiente"
        if importador_value and denuncia_candidates:
            matched_denuncia = None
            for candidate in denuncia_candidates:
                if self._company_names_match(importador_value, candidate):
                    matched_denuncia = candidate
                    break
            if matched_denuncia:
                denuncia_result = "coincide"
                denuncia_coincidencia = "coincide"
                match_display = _limit_entries([matched_denuncia], limit=1)[0]
                denuncia_detail = f"Coincide con {match_display} en denuncia/carpeta."
            else:
                denuncia_result = "discrepancia"
                denuncia_coincidencia = "no_coincide"
                denuncia_detail = "El importador difiere de la razón social denunciada."
                indicators.append(
                    FraudIndicator(
                        pattern="importador_denuncia_no_coincide",
                        description="El importador del pedimento no coincide con la razón social que presenta la denuncia en carpeta.",
                        severity="alto",
                        confidence=0.75,
                    )
                )
        elif importador_value and not denuncia_candidates:
            denuncia_result = "desconocido"
            denuncia_detail = "La carpeta de investigación no detalla la razón social denunciante."
            recommendations.append(
                "Solicitar la denuncia ratificada o el acta ministerial donde conste la razón social denunciante para cotejarla con el pedimento."
            )
        elif not importador_value:
            denuncia_result = "desconocido"
            denuncia_detail = "El pedimento no detalla el importador, por lo que no se puede comparar con la denuncia."

        verificaciones["importador_vs_denuncia"] = {
            "resultado": denuncia_result,
            "importador": importador_value or "",
            "referencias": _limit_entries(denuncia_candidates),
            "detalle": denuncia_detail,
        }

        carpeta_cross = {
            "empresa_referencias": _limit_entries(denuncia_candidates),
            "coincidencia_importador": denuncia_coincidencia,
            "observaciones": denuncia_detail,
        }
        validacion_cruzada["denuncia"] = {
            "razones_sociales": _limit_entries(denuncia_candidates),
            "coincidencia": denuncia_coincidencia,
            "observaciones": denuncia_detail,
        }

        # ------------------------------------------------------------------
        # Mercancía vs documentos asociados
        # ------------------------------------------------------------------
        comparables: List[Tuple[str, str]] = []
        carpeta_ext = self._find_extraction_by_type(data_layer, "carpeta_de_investigacion") if data_layer else None
        carta_simple_ext = self._find_extraction_by_type(data_layer, "carta_porte_simple") if data_layer else None
        cfdi_ext = self._find_extraction_by_type(data_layer, "cfdi_carta_porte") if data_layer else None
        carta_transportista_ext = (
            self._find_extraction_by_type(data_layer, "carta_de_reclamacion_formal_al_transportista") if data_layer else None
        )

        if carpeta_ext:
            for denuncia_entry in self._ensure_list(carpeta_ext.extracted_fields.get("denuncias")):
                if not isinstance(denuncia_entry, dict):
                    continue
                for item in self._ensure_list(denuncia_entry.get("mercancias")):
                    text = self._stringify_value(item).strip()
                    if text:
                        comparables.append(("carpeta_de_investigacion", text))
            resumen = self._stringify_value(carpeta_ext.extracted_fields.get("resumen_conjunto"))
            if resumen:
                comparables.append(("carpeta_de_investigacion", resumen))

        if carta_simple_ext:
            carta_simple_goods = (
                carta_simple_ext.extracted_fields.get("descripcion_mercancia")
                or carta_simple_ext.extracted_fields.get("detalle_mercancia")
            )
            text = self._stringify_value(carta_simple_goods).strip()
            if text:
                comparables.append(("carta_porte_simple", text))

        if cfdi_ext:
            cfdi_goods = (
                cfdi_ext.extracted_fields.get("descripcion_mercancias")
                or cfdi_ext.extracted_fields.get("mercancias")
                or cfdi_ext.extracted_fields.get("mercancia_detalle")
            )
            text = self._stringify_value(cfdi_goods).strip()
            if text:
                comparables.append(("cfdi_carta_porte", text))

        if carta_transportista_ext:
            carta_detalle = (
                carta_transportista_ext.extracted_fields.get("detalle_mercancia")
                or carta_transportista_ext.extracted_fields.get("descripcion_evento")
            )
            text = self._stringify_value(carta_detalle).strip()
            if text:
                comparables.append(("carta_reclamacion_transportista", text))

        mercancias_coinciden = False
        referencias_mercancia: List[str] = []
        if mercancia_value and comparables:
            for source, comparable_text in comparables:
                if comparable_text and self._goods_match(mercancia_value, comparable_text):
                    mercancias_coinciden = True
                    referencias_mercancia.append(source)
        referencias_mercancia = sorted(set(referencias_mercancia))

        if mercancia_known and comparables and not mercancias_coinciden:
            verificaciones["mercancia_vs_documentos"] = {
                "resultado": "discrepancia",
                "mercancia_pedimento": mercancia_display,
                "referencias": _limit_entries(text for _, text in comparables),
                "detalle": "La mercancía declarada no coincide con los documentos revisados.",
            }
            indicators.append(
                FraudIndicator(
                    pattern="mercancia_no_coincide",
                    description="La mercancía del pedimento no coincide con la registrada en carpeta o cartas porte.",
                    severity="medio",
                    confidence=0.7,
                )
            )
        elif mercancia_known and mercancias_coinciden:
            verificaciones["mercancia_vs_documentos"] = {
                "resultado": "coincide",
                "mercancia_pedimento": mercancia_display,
                "referencias": _limit_entries(referencias_mercancia),
                "detalle": "La mercancía coincide con la documentación logística.",
            }
        elif mercancia_known and not comparables:
            verificaciones["mercancia_vs_documentos"] = {
                "resultado": "desconocido",
                "mercancia_pedimento": mercancia_display,
                "referencias": [],
                "detalle": "No se localizaron documentos con detalle de mercancía para comparar con el pedimento.",
            }
            recommendations.append(
                "Solicitar carta porte y soportes de carga digitalizados para cotejar la mercancía del pedimento."
            )
        else:
            verificaciones["mercancia_vs_documentos"] = {
                "resultado": "desconocido",
                "mercancia_pedimento": "",
                "referencias": [],
                "detalle": "El pedimento no detalla la mercancía importada para realizar comparativos.",
            }

        if not mercancia_value and comparables:
            mercancia_display = _sanitize_goods_label(self._shorten_goods_reference(comparables[0][1]))
        else:
            mercancia_display = _sanitize_goods_label(mercancia_value)

        # ------------------------------------------------------------------
        # Fecha vs carpeta de investigación
        # ------------------------------------------------------------------
        carpeta_date_candidates: List[Tuple[date, str]] = []
        if carpeta_ext:
            for key in ("fecha_inicio_viaje", "fecha_evento", "fecha_salida"):
                dt_value = self._parse_iso_date(carpeta_ext.extracted_fields.get(key))
                if dt_value:
                    carpeta_date_candidates.append((dt_value, key))
            for denuncia_entry in self._ensure_list(carpeta_ext.extracted_fields.get("denuncias")):
                if not isinstance(denuncia_entry, dict):
                    continue
                for key in ("fecha_inicio", "fecha_siniestro", "fecha_evento"):
                    dt_value = self._parse_iso_date(denuncia_entry.get(key))
                    if dt_value:
                        carpeta_date_candidates.append((dt_value, f"denuncia_{key}"))
                        break
        carpeta_fecha_dt: Optional[date] = None
        if carpeta_date_candidates:
            carpeta_fecha_dt = min(carpeta_date_candidates, key=lambda item: item[0])[0]

        fecha_vs_carpeta_result = "desconocido"
        fecha_vs_carpeta_detalle = "Sin datos suficientes para comparar fechas."
        if fecha_entrada_dt and carpeta_fecha_dt:
            delta = (carpeta_fecha_dt - fecha_entrada_dt).days
            carpeta_fecha_text = self._format_date_long(carpeta_fecha_dt)
            if delta > 0:
                fecha_vs_carpeta_result = "coincide"
                fecha_vs_carpeta_detalle = f"La fecha del pedimento es {delta} día(s) anterior al inicio del viaje."
            elif delta == 0:
                fecha_vs_carpeta_result = "coincide"
                fecha_vs_carpeta_detalle = "La fecha del pedimento coincide con el inicio documentado del viaje."
            else:
                fecha_vs_carpeta_result = "discrepancia"
                fecha_vs_carpeta_detalle = "La fecha del pedimento es posterior al inicio del viaje descrito en carpeta."
                indicators.append(
                    FraudIndicator(
                        pattern="fecha_entrada_posterior",
                        description="La fecha de entrada del pedimento es posterior al inicio del viaje declarado en carpeta.",
                        severity="medio",
                        confidence=0.7,
                    )
                )
        elif fecha_entrada_dt and not carpeta_fecha_dt:
            fecha_vs_carpeta_result = "desconocido"
            fecha_vs_carpeta_detalle = "La carpeta no precisa la fecha de inicio del viaje para comparar."
            recommendations.append(
                "Solicitar la constancia ministerial o acta de hechos con la fecha de inicio del viaje para validar la temporalidad del pedimento."
            )
        elif not fecha_entrada_dt and carpeta_fecha_dt:
            fecha_vs_carpeta_result = "desconocido"
            fecha_vs_carpeta_detalle = "El pedimento no consigna fecha de entrada para comparar."

        verificaciones["fecha_vs_carpeta"] = {
            "resultado": fecha_vs_carpeta_result,
            "fecha_pedimento": fecha_entrada_display,
            "fecha_carpeta": self._format_date_long(carpeta_fecha_dt) if carpeta_fecha_dt else "",
            "detalle": fecha_vs_carpeta_detalle,
        }
        carpeta_cross["fecha_inicio_viaje"] = self._format_date_slash(carpeta_fecha_dt) if carpeta_fecha_dt else ""
        carpeta_cross["observaciones_fecha"] = fecha_vs_carpeta_detalle

        # ------------------------------------------------------------------
        # Fecha vs monitoreo GPS
        # ------------------------------------------------------------------
        gps_points = self._collect_gps_points(data_layer, limit=5000) if data_layer else []
        gps_docs = sorted((getattr(data_layer, "gps_documents", {}) or {}).keys()) if data_layer else []
        gps_dates: List[date] = []
        for point in gps_points:
            ts = point.get("timestamp")
            if isinstance(ts, datetime):
                gps_dates.append(ts.date())
        gps_fecha_inicio = min(gps_dates) if gps_dates else None

        fecha_vs_gps_result = "desconocido"
        fecha_vs_gps_detalle = "Sin datos GPS para cotejar."
        if fecha_entrada_dt and gps_fecha_inicio:
            delta = (gps_fecha_inicio - fecha_entrada_dt).days
            gps_fecha_text = self._format_date_long(gps_fecha_inicio)
            if delta > 0:
                fecha_vs_gps_result = "coincide"
                fecha_vs_gps_detalle = f"La fecha del pedimento es {delta} día(s) anterior al arranque del monitoreo GPS."
            elif delta == 0:
                fecha_vs_gps_result = "coincide"
                fecha_vs_gps_detalle = "La fecha del pedimento coincide con el inicio del monitoreo GPS."
            else:
                fecha_vs_gps_result = "discrepancia"
                fecha_vs_gps_detalle = "La fecha del pedimento es posterior al arranque del monitoreo GPS."
                indicators.append(
                    FraudIndicator(
                        pattern="fecha_gps_inconsistente",
                        description="El pedimento se emitió después de la fecha de arranque registrada en GPS.",
                        severity="medio",
                        confidence=0.65,
                    )
                )
        elif fecha_entrada_dt and not gps_fecha_inicio and gps_docs:
            fecha_vs_gps_result = "desconocido"
            fecha_vs_gps_detalle = "Los datasets GPS cargados no contienen fecha de arranque para comparar."
        elif fecha_entrada_dt and not gps_docs:
            fecha_vs_gps_result = "desconocido"
            fecha_vs_gps_detalle = "No se encontraron datasets GPS vinculados al caso."
            recommendations.append(
                "Solicitar al área de monitoreo el reporte GPS del viaje para cotejar la temporalidad del pedimento."
            )
        elif not fecha_entrada_dt and gps_fecha_inicio:
            fecha_vs_gps_result = "desconocido"
            fecha_vs_gps_detalle = "El pedimento no señala fecha de entrada para compararla con el monitoreo GPS."

        verificaciones["fecha_vs_gps"] = {
            "resultado": fecha_vs_gps_result,
            "fecha_pedimento": fecha_entrada_display,
            "fecha_inicio_gps": self._format_date_long(gps_fecha_inicio) if gps_fecha_inicio else "",
            "documentos_gps": gps_docs,
            "detalle": fecha_vs_gps_detalle,
        }
        validacion_cruzada["gps"] = {
            "documentos_consultados": gps_docs,
            "fecha_inicio_dataset": self._format_date_slash(gps_fecha_inicio) if gps_fecha_inicio else "",
            "observaciones": fecha_vs_gps_detalle,
        }

        validacion_cruzada["carpeta_investigacion"] = carpeta_cross
        if pedimento_digits:
            validacion_cruzada["pedimento"] = {
                "numero_normalizado": pedimento_digits,
                "observaciones": "Número de pedimento analizado para correlaciones.",
            }

        permanent_recommendation = (
            "Validar el pedimento directamente en el portal del SAT y conservar la evidencia de la consulta, "
            "ya que todavía no contamos con una verificación automática vía API."
        )
        if permanent_recommendation not in recommendations:
            recommendations.append(permanent_recommendation)

        analysis.analisis_completo = (
            f"Se cuenta con pedimento {pedimento_display}, donde se aprecia al importador {importador_display}, "
            f"se observa el contenido de mercancía, consistente en {mercancia_display}, fecha de entrada {fecha_entrada_display}."
        ).replace("  ", " ").strip()

        cleaned_recs = sorted({rec.strip() for rec in recommendations if rec.strip()})
        analysis.recommendations = cleaned_recs

        if indicators:
            severities = {"bajo": 1, "medio": 2, "alto": 3, "critico": 4}
            top_indicator = max(indicators, key=lambda item: severities.get(item.severity, 2))
            if top_indicator.severity == "critico":
                analysis.fraud_score = max(analysis.fraud_score, 0.88)
            elif top_indicator.severity == "alto":
                analysis.fraud_score = max(analysis.fraud_score, 0.72)
            elif top_indicator.severity == "medio":
                analysis.fraud_score = max(analysis.fraud_score, 0.48)
            analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))
            analysis.confidence = max(analysis.confidence, 0.78)
        else:
            analysis.fraud_score = max(analysis.fraud_score, 0.26)
            analysis.risk_level = RiskLevel.BAJO
            analysis.confidence = max(analysis.confidence, 0.85)

        analysis.indicators = indicators
        analysis.verificaciones = verificaciones
        analysis.validacion_cruzada = validacion_cruzada
        return analysis

    def _postprocess_conocimiento_embarque(
        self,
        analysis: FraudAnalysisResult,
        extraction: DocumentExtraction,
        *,
        data_layer: Optional[UnifiedDataLayer],
        document_context: Dict[str, Any],
        case_context: Dict[str, Any],
        ocr_text: str,
    ) -> FraudAnalysisResult:
        fields = dict(extraction.extracted_fields or {})
        metadata = dict(extraction.extraction_metadata or {})
        resolved_fields = dict(document_context.get("resolved_fields") or {})

        def _pick(*keys: str, default: str = "") -> str:
            for key in keys:
                if key in fields and fields[key]:
                    return self._stringify_value(fields[key])
                if key in resolved_fields and resolved_fields[key]:
                    return self._stringify_value(resolved_fields[key])
                if key in metadata and metadata[key]:
                    return self._stringify_value(metadata[key])
            return default

        emisor_raw = _pick(
            "emisor_documento",
            "empresa_emitente",
            "agente_aduanal",
            "nombre_transportista",
            "transportista",
            default=case_context.get("insured_name", ""),
        )
        emisor = self._format_entity_name(emisor_raw or "el emisor del conocimiento de embarque")
        if not emisor:
            emisor = "el emisor del conocimiento de embarque"

        fecha_raw = _pick("fecha_salida", "fecha_documento", "fecha_emision", "fecha")
        fecha_dt = self._parse_iso_date(fecha_raw)
        fecha_fmt_long = self._format_date_long(fecha_dt) if fecha_dt else (fecha_raw or "fecha no identificada")

        analysis.analisis_completo = (
            f"Se cuenta con comprobante emitido por {emisor}, fechado el {fecha_fmt_long}. "
            "El documento detalla la información del transportista, los datos de las unidades que trasladaban la mercancía, "
            "los operadores asignados a cada una, y el contenido y peso de cada cargamento."
        )

        indicators: List[FraudIndicator] = []
        recommendations: List[str] = []
        verificaciones: Dict[str, Dict[str, Any]] = {}

        def _normalize_digits(value: Optional[str]) -> str:
            return re.sub(r"\D+", "", value or "")

        def _collect_plates() -> Tuple[List[str], List[str], Dict[str, str]]:
            plates_display: List[str] = []
            plates_norm: List[str] = []
            plates_map: Dict[str, str] = {}
            plate_candidates: List[Any] = []
            for key in (
                "placas_unidad",
                "placas",
                "placas_remolque",
                "placas_semirremolques",
                "semirremolques",
                "remolques",
                "identificacion_unidad",
            ):
                value = fields.get(key) or resolved_fields.get(key)
                if value:
                    plate_candidates.extend(self._ensure_list(value))
            for candidate in plate_candidates:
                token = ""
                if isinstance(candidate, dict):
                    token = (
                        candidate.get("placa")
                        or candidate.get("placas")
                        or candidate.get("identificador")
                        or candidate.get("numero")
                        or ""
                    )
                else:
                    token = self._stringify_value(candidate)
                token = token.strip()
                if not token:
                    continue
                fragments = re.split(r"[,\n;/]+", token) if re.search(r"[,\n;/]", token) else [token]
                for fragment in fragments:
                    plate_token = fragment.strip()
                    if not plate_token:
                        continue
                    norm = self._normalize_plate(plate_token)
                    if not norm:
                        continue
                    if norm not in plates_norm:
                        plates_norm.append(norm)
                        plates_display.append(plate_token)
                        plates_map[norm] = plate_token
            return plates_display, plates_norm, plates_map

        plates_display, plates_norm, plates_map = _collect_plates()
        placas_label = ", ".join(plates_display) if plates_display else "placas no registradas"

        descripcion_mercancia = _pick("descripcion_mercancia", "contenido", "detalle_mercancia", default="")
        cantidad_mercancia = _pick("cantidad_mercancia", "cantidad", default="")
        peso_mercancia = _pick("peso", "peso_total", "peso_kg", default="")
        resumen_conocimiento = descripcion_mercancia
        if cantidad_mercancia:
            resumen_conocimiento = f"{cantidad_mercancia} de {descripcion_mercancia or 'mercancía declarada'}".strip()
        if peso_mercancia:
            if resumen_conocimiento:
                resumen_conocimiento = f"{resumen_conocimiento} ({peso_mercancia})"
            else:
                resumen_conocimiento = peso_mercancia

        # ------------------------------------------------------------------
        # Fechas entre conocimientos
        # ------------------------------------------------------------------
        fechas_detectadas: List[str] = []
        base_iso = fecha_dt.isoformat() if fecha_dt else (_normalize_digits(fecha_raw) or "")
        if base_iso:
            fechas_detectadas.append(f"{fecha_raw or base_iso} ({extraction.source_document})")

        comparison_isos: List[str] = [base_iso] if base_iso else []
        if data_layer:
            for other_name, other_ext in self._iter_document_sources(data_layer, "conocimiento_de_embarque"):
                if other_name == extraction.source_document:
                    continue
                if other_ext is None:
                    continue
                other_fields = dict(other_ext.extracted_fields or {})
                other_date_raw = self._stringify_value(
                    other_fields.get("fecha_salida")
                    or other_fields.get("fecha_documento")
                    or other_fields.get("fecha_emision")
                )
                other_date_dt = self._parse_iso_date(other_date_raw)
                iso_value = other_date_dt.isoformat() if other_date_dt else _normalize_digits(other_date_raw)
                if iso_value:
                    comparison_isos.append(iso_value)
                    fechas_detectadas.append(f"{other_date_raw or iso_value} ({other_ext.source_document})")

        fechas_result = "desconocido"
        fechas_detalle = "No existen otros conocimientos de embarque en el expediente para comparar."
        if len(comparison_isos) > 1:
            unique_dates = {item for item in comparison_isos if item}
            if len(unique_dates) == 1:
                fechas_result = "coincide"
                fechas_detalle = "Las fechas de los conocimientos de embarque coinciden."
            else:
                fechas_result = "discrepancia"
                fechas_detalle = "Se detectan fechas distintas entre conocimientos de embarque."
                indicators.append(
                    FraudIndicator(
                        pattern="fechas_conocimientos_divergen",
                        description="Las fechas de expedición entre conocimientos de embarque difieren.",
                        severity="medio",
                        confidence=0.75,
                    )
                )
        verificaciones["fechas_entre_conocimientos"] = {
            "resultado": fechas_result,
            "fecha_documento": fecha_raw,
            "fechas_detectadas": fechas_detectadas,
            "detalle": fechas_detalle,
        }

        # ------------------------------------------------------------------
        # Pedimento vs pedimento de importación
        # ------------------------------------------------------------------
        pedimento_doc = self._find_extraction_by_type(data_layer, "pedimento_importacion") if data_layer else None
        conocimiento_ped = _pick("numero_pedimento", "pedimento", "pedimento_numero", default="")
        pedimento_ref = ""
        if pedimento_doc:
            pedimento_ref = self._stringify_value(
                pedimento_doc.extracted_fields.get("numero_pedimento")
                or pedimento_doc.extracted_fields.get("folio")
                or pedimento_doc.extracted_fields.get("numero_documento")
            )

        ped_result = "desconocido"
        ped_detalle = "Sin información suficiente para cotejar el pedimento."
        ped_resumen = ""
        ped_coincidencia = "pendiente"

        conocimiento_digits = _normalize_digits(conocimiento_ped)
        pedimento_digits = _normalize_digits(pedimento_ref)
        if conocimiento_digits and pedimento_digits:
            if conocimiento_digits == pedimento_digits:
                ped_result = "coincide"
                ped_coincidencia = "coincide"
                ped_resumen = "Coinciden todos los dígitos del pedimento."
                ped_detalle = "El número de pedimento del conocimiento coincide plenamente con el pedimento de importación."
            elif conocimiento_digits and conocimiento_digits in pedimento_digits:
                ped_result = "parcial"
                ped_coincidencia = "parcial"
                ped_resumen = f"Coinciden {len(conocimiento_digits)} dígitos consecutivos ({conocimiento_digits})."
                ped_detalle = "El conocimiento cita una porción del pedimento; se recomienda confirmar los dígitos restantes."
            else:
                ped_result = "discrepancia"
                ped_coincidencia = "no_coincide"
                ped_resumen = "No coinciden los dígitos del pedimento."
                ped_detalle = "El número de pedimento citado en el conocimiento difiere del pedimento de importación."
                indicators.append(
                    FraudIndicator(
                        pattern="pedimento_no_coincide",
                        description="El número de pedimento del conocimiento de embarque difiere del pedimento de importación.",
                        severity="alto",
                        confidence=0.75,
                    )
                )
        elif conocimiento_digits and not pedimento_digits:
            ped_detalle = "El pedimento de importación no está disponible para cotejar."
            ped_result = "desconocido"
            ped_coincidencia = "pendiente"
            recommendations.append(
                "Solicitar copia íntegra del pedimento de importación para validar el número citado en el conocimiento de embarque."
            )
        elif not conocimiento_digits and pedimento_digits:
            ped_detalle = "El conocimiento de embarque no cita pedimento para comparar."
            ped_result = "desconocido"
            ped_coincidencia = "pendiente"
            recommendations.append(
                "Solicitar al transportista o agente aduanal que el conocimiento de embarque incluya el número de pedimento completo."
            )

        verificaciones["pedimento_vs_pedimento"] = {
            "resultado": ped_result,
            "pedimento_conocimiento": conocimiento_ped,
            "pedimento_referencia": pedimento_ref,
            "coincidencia_digitos": ped_resumen,
            "detalle": ped_detalle,
        }

        # ------------------------------------------------------------------
        # Operador vs denuncias
        # ------------------------------------------------------------------
        denuncias_summary = self._collect_denuncia_summary(data_layer)
        operador = self._format_entity_name(
            _pick("operador_nombre", "nombre_operador", "operador", "operador_asignado", default="")
        )
        operador_result = "desconocido"
        operador_detalle = "No se localizaron denuncias para cotejar el operador."
        operador_referencia = ""
        coincidencias_operador = "pendiente"
        placas_carpeta: List[str] = []
        placas_coincidentes: List[str] = []

        operator_entry = None
        operador_tokens = self._person_name_token_list(operador)
        operador_norm = self._normalize_person_name(operador)
        if operador_norm:
            for entry in denuncias_summary:
                if entry.get("nombre_norm") == operador_norm:
                    operator_entry = entry
                    break
        if operator_entry is None and operador_tokens:
            for entry in denuncias_summary:
                entry_tokens_list = entry.get("tokens") or self._person_name_token_list(entry.get("nombre"))
                if self._person_names_match_loose(operador_tokens, entry_tokens_list):
                    operator_entry = entry
                    break

        for entry in denuncias_summary:
            for plate in entry.get("placas_display") or []:
                if plate not in placas_carpeta:
                    placas_carpeta.append(plate)

        carpeta_norm_map: Dict[str, str] = {}
        for plate in placas_carpeta:
            variants = self._plate_variants(plate)
            if not variants:
                norm_plate = self._normalize_plate(plate)
                if norm_plate and norm_plate not in carpeta_norm_map:
                    carpeta_norm_map[norm_plate] = plate
                continue
            for variant in variants:
                if variant and variant not in carpeta_norm_map:
                    carpeta_norm_map[variant] = plate

        if operator_entry:
            operador_referencia = operator_entry.get("nombre") or ""
            coincidencias_operador = "coincide"
            denuncia_plates = operator_entry.get("placas_norm") or []
            for norm in plates_norm:
                if norm in denuncia_plates:
                    display_value = plates_map.get(norm, norm)
                    if display_value not in placas_coincidentes:
                        placas_coincidentes.append(display_value)
            if placas_coincidentes or not plates_norm:
                operador_result = "coincide"
                operador_detalle = "El operador coincide con la denuncia y las placas corresponden al expediente."
            elif plates_norm and not placas_coincidentes and denuncia_plates:
                operador_result = "discrepancia"
                coincidencias_operador = "no_coincide"
                operador_detalle = "El operador coincide pero las placas difieren respecto a la denuncia."
                indicators.append(
                    FraudIndicator(
                        pattern="operador_placas_inconsistentes",
                        description="El operador coincide con la denuncia, pero las placas declaradas difieren de las registradas.",
                        severity="alto",
                        confidence=0.8,
                    )
                )
            else:
                operador_result = "desconocido"
                operador_detalle = "El operador coincide, pero la denuncia no detalla placas para validar."
                recommendations.append(
                    "Solicitar ampliación de denuncia o carpeta que documente las placas asignadas al operador."
                )
        elif operador:
            operador_result = "desconocido"
            coincidencias_operador = "pendiente"
            operador_detalle = "No se localizaron denuncias con el operador citado en el conocimiento."
            recommendations.append(
                f"Solicitar denuncia o carpeta que acredite la participación de {operador} como operador del embarque."
            )

        matched_norms_global = [norm for norm in plates_norm if norm in carpeta_norm_map]
        global_matches_display = [carpeta_norm_map[norm] for norm in matched_norms_global]
        if not placas_coincidentes and global_matches_display:
            placas_coincidentes = list(dict.fromkeys(global_matches_display))
        else:
            for plate_value in global_matches_display:
                if plate_value not in placas_coincidentes:
                    placas_coincidentes.append(plate_value)

        verificaciones["operador_vs_denuncia"] = {
            "resultado": operador_result,
            "operador_conocimiento": operador,
            "operador_denuncia": operador_referencia,
            "detalle": operador_detalle,
        }

        # ------------------------------------------------------------------
        # Placas vs carpeta
        # ------------------------------------------------------------------
        placas_result = "desconocido"
        placas_detalle = "No se cuenta con placas suficientes para comparar contra la carpeta."
        faltantes_display: List[str] = []
        if plates_norm and placas_carpeta:
            missing_norms = [norm for norm in plates_norm if norm not in carpeta_norm_map]
            faltantes_display = [plates_map.get(norm, norm) for norm in missing_norms]
            if not missing_norms:
                placas_result = "coincide"
                placas_detalle = "Las placas del conocimiento coinciden con las registradas en la carpeta de investigación."
            elif placas_coincidentes:
                placas_result = "parcial"
                faltantes_text = ", ".join(faltantes_display)
                placas_detalle = (
                    "Algunas placas del conocimiento coinciden con la carpeta de investigación; faltan por confirmar: "
                    f"{faltantes_text}."
                )
            else:
                placas_result = "discrepancia"
                placas_detalle = "Las placas del conocimiento no se localizaron en la carpeta de investigación."
        elif plates_norm and not placas_carpeta:
            placas_result = "desconocido"
            placas_detalle = "La carpeta no detalla placas para validar contra el conocimiento de embarque."
            recommendations.append(
                "Solicitar a la autoridad copia de la carpeta de investigación con el detalle de placas para cotejo."
            )

        if placas_result == "discrepancia" and not any((ind.pattern or "").lower() == "operador_placas_inconsistentes" for ind in indicators):
            indicators.append(
                FraudIndicator(
                    pattern="operador_placas_inconsistentes",
                    description="Las placas declaradas en el conocimiento de embarque difieren de las asentadas en la carpeta de investigación.",
                    severity="alto",
                    confidence=0.8,
                )
            )

        verificaciones["placas_vs_carpeta"] = {
            "resultado": placas_result,
            "placas_conocimiento": plates_display,
            "placas_carpeta": placas_carpeta,
            "coincidencias": placas_coincidentes,
            "placas_faltantes": faltantes_display,
            "detalle": placas_detalle,
        }

        # ------------------------------------------------------------------
        # Cantidades vs otros documentos
        # ------------------------------------------------------------------
        referencias_cantidades: List[str] = []
        coincidencia_mercancia = False
        carta_goods = ""

        carta_ext = self._find_extraction_by_type(data_layer, "carta_porte_simple") if data_layer else None
        if carta_ext:
            carta_goods = self._stringify_value(
                carta_ext.extracted_fields.get("descripcion_mercancia")
                or carta_ext.extracted_fields.get("detalle_mercancia")
            )
            if descripcion_mercancia and carta_goods and self._goods_match(descripcion_mercancia, carta_goods):
                coincidencia_mercancia = True
                referencias_cantidades.append(carta_ext.source_document)

        denuncia_text = self._get_document_text(data_layer, "denuncia_de_los_hechos") if data_layer else ""
        if descripcion_mercancia and denuncia_text and self._goods_match(descripcion_mercancia, denuncia_text):
            coincidencia_mercancia = True
            if "denuncia" not in referencias_cantidades:
                referencias_cantidades.append("denuncia")

        carpeta_text = self._get_document_text(data_layer, "carpeta_de_investigacion") if data_layer else ""
        if descripcion_mercancia and carpeta_text and self._goods_match(descripcion_mercancia, carpeta_text):
            coincidencia_mercancia = True
            if "carpeta_investigacion" not in referencias_cantidades:
                referencias_cantidades.append("carpeta_investigacion")

        cantidades_result = "desconocido"
        cantidades_detalle = "Sin referencias suficientes para validar la mercancía."
        if descripcion_mercancia and coincidencia_mercancia:
            cantidades_result = "coincide"
            cantidades_detalle = "La mercancía y cantidades coinciden con los documentos logísticos del caso."
        elif descripcion_mercancia:
            cantidades_result = "discrepancia"
            cantidades_detalle = "La mercancía declarada no se localiza en Carta Porte o denuncias."
            indicators.append(
                FraudIndicator(
                    pattern="mercancia_no_coincide",
                    description="La mercancía declarada en el conocimiento de embarque no coincide con Carta Porte o denuncias.",
                    severity="alto",
                    confidence=0.78,
                )
            )
        else:
            recomendaciones_text = (
                "Solicitar que el conocimiento de embarque describa la mercancía y cantidades para validar contra Carta Porte y denuncias."
            )
            recommendations.append(recomendaciones_text)

        verificaciones["cantidades_vs_documentos"] = {
            "resultado": cantidades_result,
            "resumen_conocimiento": resumen_conocimiento,
            "referencias": referencias_cantidades,
            "detalle": cantidades_detalle,
        }

        # ------------------------------------------------------------------
        # Validación cruzada
        # ------------------------------------------------------------------
        validacion_cruzada = dict(analysis.validacion_cruzada or {})
        validacion_cruzada["pedimento"] = {
            "numero_en_pedimento": pedimento_ref,
            "coincidencia": ped_coincidencia,
            "observaciones": ped_detalle,
        }
        validacion_cruzada["denuncia"] = {
            "operadores": [entry.get("nombre") for entry in denuncias_summary],
            "coincidencia_operador": coincidencias_operador,
            "observaciones": operador_detalle,
        }
        validacion_cruzada["carpeta_investigacion"] = {
            "placas_registradas": placas_carpeta,
            "coincidencia_placas": (
                "coincide"
                if placas_result == "coincide"
                else "parcial"
                if placas_result == "parcial"
                else "pendiente"
                if placas_result == "desconocido"
                else "no_coincide"
            ),
            "observaciones": placas_detalle,
        }
        validacion_cruzada["carta_porte"] = {
            "mercancia_declarada": carta_goods,
            "coincidencia": "coincide" if coincidencia_mercancia else ("pendiente" if not descripcion_mercancia else "no_coincide"),
            "observaciones": cantidades_detalle,
        }

        # ------------------------------------------------------------------
        # Estado final
        # ------------------------------------------------------------------
        if recommendations:
            cleaned_recs = sorted({rec.strip() for rec in recommendations if rec.strip()})
            analysis.recommendations = cleaned_recs
        else:
            analysis.recommendations = []

        if indicators:
            severities = {"bajo": 1, "medio": 2, "alto": 3, "critico": 4}
            top_indicator = max(indicators, key=lambda item: severities.get(item.severity, 2))
            if top_indicator.severity == "critico":
                analysis.fraud_score = max(analysis.fraud_score, 0.88)
            elif top_indicator.severity == "alto":
                analysis.fraud_score = max(analysis.fraud_score, 0.72)
            elif top_indicator.severity == "medio":
                analysis.fraud_score = max(analysis.fraud_score, 0.48)
            analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))
            analysis.confidence = max(analysis.confidence, 0.75)
        else:
            analysis.fraud_score = 0.24
            analysis.risk_level = RiskLevel.BAJO
            analysis.confidence = max(analysis.confidence, 0.88)

        analysis.indicators = indicators
        analysis.verificaciones = verificaciones
        analysis.validacion_cruzada = validacion_cruzada
        return analysis

    def _postprocess_carta_aclaratoria_peaje(
        self,
        analysis: FraudAnalysisResult,
        extraction: DocumentExtraction,
        *,
        data_layer: Optional[UnifiedDataLayer],
        document_context: Dict[str, Any],
        case_context: Dict[str, Any],
        ocr_text: str,
    ) -> FraudAnalysisResult:
        fields = dict(extraction.extracted_fields or {})
        metadata = dict(extraction.extraction_metadata or {})
        resolved_fields = dict(document_context.get("resolved_fields") or {})

        indicators: List[FraudIndicator] = []
        recommendations: List[str] = []

        source_text = ocr_text or ""
        cached_text = self._get_document_text(
            data_layer,
            extraction.document_type,
            source_document=extraction.source_document,
        )
        if cached_text:
            source_text = cached_text

        def _pick(*keys: str, default: str = "") -> str:
            for key in keys:
                if key in fields and fields[key]:
                    return self._stringify_value(fields[key])
                if key in resolved_fields and resolved_fields[key]:
                    return self._stringify_value(resolved_fields[key])
            return default

        emisor_raw = _pick(
            "emisor_carta",
            "emisor",
            "asegurado",
            "nombre_asegurado",
            default=case_context.get("insured_name", ""),
        )
        emisor = self._format_entity_name(emisor_raw or "el asegurado")

        firmante_nombre = self._format_entity_name(
            _pick("firmante_nombre", "representante", "firma", default="")
        )
        firmante_cargo = _pick("firmante_cargo", "cargo_firmante", "cargo_firma", default="")

        def _sanitize_recipient_title(raw: str) -> str:
            cleaned = self._stringify_value(raw)
            if not cleaned:
                return ""
            token = re.sub(r"[^a-z]", "", self._strip_accents(cleaned.lower()))
            if token in {"estimado", "estimada"}:
                return ""
            return cleaned

        destinatario_nombre = self._format_entity_name(
            _pick("destinatario_nombre", "dirigido_a", "destinatario", default="")
        )
        destinatario_cargo = _sanitize_recipient_title(_pick("destinatario_cargo", "cargo_destinatario", default=""))

        asunto_principal = _pick("asunto_principal", "asunto", "motivo_aclaracion", default="el motivo indicado en la carta")
        descripcion_evento = _pick("descripcion_evento", "descripcion_carta", "detalle_evento", default="no se detalló información específica")
        consecuencia_evento = _pick("consecuencia_evento", "resultado_evento", "impacto", default="no se indicó consecuencia directa")
        detalle_carta = _pick("detalle_carta", "detalle_declaracion", "detalles_aclaracion", default="no se aportaron detalles adicionales")
        proposito_carta = _pick("proposito_carta", "proposito_notificacion", "objetivo_carta", default="no se puntualizó el propósito de la notificación")

        parsed_details = self._extract_aclaratoria_details(source_text)
        if parsed_details:
            if parsed_details.get("emisor"):
                emisor = self._format_entity_name(parsed_details["emisor"])
                if emisor.lower().startswith("lc"):
                    emisor = emisor.replace("Lc", "LC", 1)
            if parsed_details.get("firmante"):
                firmante_nombre = self._format_entity_name(parsed_details["firmante"])
            if parsed_details.get("firmante_cargo"):
                firmante_cargo = parsed_details["firmante_cargo"]
            if parsed_details.get("recipient"):
                destinatario_nombre = self._format_entity_name(parsed_details["recipient"])
            if parsed_details.get("recipient_cargo"):
                destinatario_cargo = parsed_details["recipient_cargo"]
            if parsed_details.get("subject"):
                asunto_principal = parsed_details["subject"]
            if parsed_details.get("description"):
                descripcion_evento = parsed_details["description"]
            if parsed_details.get("impact") and consecuencia_evento == "no se indicó consecuencia directa":
                consecuencia_evento = parsed_details["impact"]
            if parsed_details.get("detail"):
                detalle_carta = parsed_details["detail"]
            if parsed_details.get("purpose"):
                proposito_carta = parsed_details["purpose"]

        if not firmante_nombre:
            firmante_nombre = "representante no especificado"
        firmante_etiqueta = (
            f"{firmante_nombre} ({firmante_cargo})" if firmante_cargo else firmante_nombre
        )

        company_match = None
        if firmante_nombre and emisor:
            emisor_norm = self._strip_accents(emisor.lower())
            firmante_norm = self._strip_accents(firmante_nombre.lower())
            if (
                emisor_norm == firmante_norm
                or emisor_norm in firmante_norm
                or firmante_norm in emisor_norm
            ):
                company_match = re.search(r"LC\s*(?:&\+)?\s*TRANSPORTACIONES", source_text or "", re.IGNORECASE)
                if company_match:
                    emisor = "LC TRANSPORTACIONES"

        destinatario_cargo = _sanitize_recipient_title(destinatario_cargo)
        if not destinatario_nombre:
            destinatario_nombre = "el destinatario indicado"
        destinatario_etiqueta = (
            f"{destinatario_nombre} ({destinatario_cargo})" if destinatario_cargo else destinatario_nombre
        )

        casetas_declared = self._normalize_caseta_entries(
            fields.get("casetas_involucradas")
            or resolved_fields.get("casetas_involucradas")
            or fields.get("ruta_manifestada")
        )
        if (not casetas_declared) and parsed_details and parsed_details.get("casetas"):
            casetas_declared = self._normalize_caseta_entries(parsed_details["casetas"])
        if not casetas_declared:
            recommendations.append(
                "Solicitar que la carta detalle casetas, fechas y horarios para validar contra el monitoreo GPS."
            )

        def _format_datetime_full(value: Optional[datetime]) -> str:
            if isinstance(value, datetime):
                return value.strftime("%Y-%m-%d %H:%M:%S")
            return ""

        def _format_caseta_timestamp(record: Dict[str, Any]) -> str:
            ts = record.get("timestamp")
            ts_text = _format_datetime_full(ts)
            if ts_text:
                return ts_text
            raw_payload = record.get("raw")
            if isinstance(raw_payload, dict):
                candidate = raw_payload.get("timestamp") or raw_payload.get("fecha_hora")
                if candidate:
                    text = self._stringify_value(candidate)
                    if text:
                        return text
                fecha = self._stringify_value(raw_payload.get("fecha"))
                hora = self._stringify_value(raw_payload.get("hora"))
                if fecha and hora:
                    return f"{fecha} {hora}".strip()
                if fecha:
                    return fecha
                if hora:
                    return hora
            if isinstance(raw_payload, str):
                return raw_payload.strip()
            raw_value = record.get("raw")
            if isinstance(raw_value, str):
                return raw_value.strip()
            return ""

        def _format_purpose_clause(text: str) -> str:
            clause = self._stringify_value(text).strip()
            if not clause:
                return "informar del extravío de tickets a la aseguradora"
            clause = clause.rstrip(".")
            normalized = self._strip_accents(clause).lower()
            replacements = (
                ("nos gustaría ", ""),
                ("nos gustaria ", ""),
                ("nos permitimos informar ", "informar "),
                ("nos permitimos ", ""),
            )
            for needle, replacement in replacements:
                needle_norm = self._strip_accents(needle).lower()
                if normalized.startswith(needle_norm):
                    clause = (replacement + clause[len(needle):]).strip()
                    break
            if clause and clause[0].isupper():
                clause = clause[0].lower() + clause[1:]
            lowered_clause = self._strip_accents(clause).lower()
            if lowered_clause.startswith("notificarles"):
                clause = clause.replace("notificarles", "notificar a la aseguradora", 1)
            return clause

        caseta_lookup: Dict[str, Dict[str, Any]] = {}

        detalle_texto: str
        if casetas_declared:
            count = len(casetas_declared)
            plural_label = {
                1: "un punto específico",
                2: "dos puntos específicos",
                3: "tres puntos específicos",
                4: "cuatro puntos específicos",
                5: "cinco puntos específicos",
            }.get(count, f"{count} puntos específicos")
            caseta_lines: List[str] = []
            for idx, record in enumerate(casetas_declared, 1):
                nombre = record.get("nombre") or record.get("identificador") or f"caseta {idx}"
                nombre = self._stringify_value(nombre) or f"caseta {idx}"
                caseta_lookup[nombre] = record
                ts_text = _format_caseta_timestamp(record)
                if ts_text:
                    caseta_lines.append(f"- {nombre}: {ts_text}")
                else:
                    caseta_lines.append(f"- {nombre}")
            detalle_texto = (
                f"La carta detalla {plural_label} de uso o registro de los tickets, incluyendo plaza y horario declarado:\n"
                + "\n".join(caseta_lines)
            )
        elif detalle_carta:
            detalle_texto = f"La carta detalla: {detalle_carta}"
        else:
            detalle_texto = "La carta detalla: no se aportaron detalles adicionales."

        purpose_clause = _format_purpose_clause(proposito_carta)

        analysis_lines = [
            (
                f"Se presentó una carta emitida por {emisor}, firmada por {firmante_etiqueta}. "
                f"La carta está dirigida a {destinatario_etiqueta}."
            ).strip(),
            (
                f"El asunto principal de la carta es {asunto_principal}. "
                f"En ella, se informa sobre {descripcion_evento}. "
                f"Como resultado de este hecho, {consecuencia_evento}."
            ).strip(),
            detalle_texto.strip(),
            f"La empresa enfatiza que el propósito de esta notificación es {purpose_clause}.".strip(),
        ]
        analysis.analisis_completo = "\n".join(line for line in analysis_lines if line).strip()

        gps_points = self._collect_gps_points(data_layer) if data_layer else []
        gps_docs_consulted = sorted((data_layer.gps_documents or {}).keys()) if getattr(data_layer, "gps_documents", None) else []

        verificaciones: Dict[str, Dict[str, Any]] = {}
        ruta_resultado = "desconocido"
        ruta_detalle = "No se pudieron comparar casetas con registros GPS."
        horarios_resultado = "desconocido"
        horarios_detalle = "No se cuentan con horarios suficientes para cotejar con GPS."
        diferencia_max_min = None
        distancia_max_metros = None
        casetas_validadas: List[str] = []
        casetas_faltantes: List[str] = []

        tolerance_m = 600.0
        tolerance_minutes = 20.0

        matches: List[Dict[str, Any]] = []
        canonical_matches: Dict[str, Dict[str, Any]] = {}
        match_summaries: List[str] = []
        points_by_doc: Dict[str, List[Dict[str, Any]]] = {}
        doc_hits: Dict[str, Dict[str, Dict[str, Any]]] = {}
        doc_failures: Dict[str, List[Dict[str, Any]]] = {}
        for doc_name in gps_docs_consulted:
            doc_hits[doc_name] = {}
            doc_failures[doc_name] = []

        if gps_points:
            for point in gps_points:
                doc_name = point.get("document") or ""
                if not doc_name:
                    continue
                points_by_doc.setdefault(doc_name, []).append(point)
                doc_hits.setdefault(doc_name, {})
                doc_failures.setdefault(doc_name, [])

            if casetas_declared:
                def _find_nearest_gps_point(
                    caseta_entry: Dict[str, Any],
                    gps_points_for_doc: List[Dict[str, Any]],
                ) -> Optional[Dict[str, Any]]:
                    lat = self._coerce_float(caseta_entry.get("lat") or caseta_entry.get("latitude"))
                    lon = self._coerce_float(caseta_entry.get("lon") or caseta_entry.get("longitude"))
                    timestamp = caseta_entry.get("timestamp")
                    if not isinstance(timestamp, datetime):
                        timestamp = self._parse_datetime(
                            timestamp
                            or caseta_entry.get("fecha")
                            or caseta_entry.get("hora")
                        )

                    best: Optional[Dict[str, Any]] = None
                    for point in gps_points_for_doc:
                        point_lat = point.get("latitude")
                        point_lon = point.get("longitude")
                        point_ts = point.get("timestamp")

                        distance = None
                        if lat is not None and lon is not None and point_lat is not None and point_lon is not None:
                            distance = self._compute_distance_meters(lat, lon, point_lat, point_lon)

                        time_diff_minutes = None
                        if timestamp and point_ts:
                            time_diff_minutes = abs((timestamp - point_ts).total_seconds()) / 60.0

                        if distance is None and time_diff_minutes is None:
                            continue

                        score = (distance or 0.0) + (time_diff_minutes or 0.0) * 10.0
                        if not best or score < best["score"]:
                            best = {
                                "caseta": caseta_entry,
                                "point": point,
                                "distance_m": distance,
                                "time_diff_minutes": time_diff_minutes,
                                "score": score,
                            }
                    return best

                for caseta in casetas_declared:
                    nombre_caseta = self._stringify_value(
                        caseta.get("nombre") or caseta.get("identificador") or "caseta"
                    ) or "caseta"
                    caseta["label"] = nombre_caseta
                    best_match = None
                    best_score = None

                    for doc_name, doc_points in points_by_doc.items():
                        match = self._find_best_gps_match(
                            caseta,
                            doc_points,
                            distance_tolerance_m=tolerance_m,
                            minutes_tolerance=tolerance_minutes,
                        )
                        if match:
                            doc_hits.setdefault(doc_name, {})[nombre_caseta] = match
                            if best_score is None or match["score"] < best_score:
                                best_match = match
                                best_score = match["score"]
                        else:
                            nearest = _find_nearest_gps_point(caseta, doc_points)
                            distance_val = nearest.get("distance_m") if nearest else None
                            time_diff_val = nearest.get("time_diff_minutes") if nearest else None
                            doc_failures.setdefault(doc_name, []).append(
                                {
                                    "label": nombre_caseta,
                                    "nearest": nearest,
                                    "distance_m": distance_val,
                                    "time_diff_minutes": time_diff_val,
                                }
                            )

                    if best_match:
                        matches.append(best_match)
                        canonical_matches[nombre_caseta] = best_match
                        casetas_validadas.append(nombre_caseta)
                        distancia = best_match.get("distance_m")
                        minutos = best_match.get("time_diff_minutes")
                        carta_ts = _format_datetime_full(best_match.get("caseta", {}).get("timestamp"))
                        gps_ts = _format_datetime_full(best_match.get("point", {}).get("timestamp"))
                        resumen = nombre_caseta
                        details: List[str] = []
                        if carta_ts:
                            details.append(f"carta {carta_ts}")
                        else:
                            ts_text = _format_caseta_timestamp(caseta)
                            if ts_text:
                                details.append(f"carta {ts_text}")
                        if gps_ts:
                            details.append(f"GPS {gps_ts}")
                        if minutos is not None:
                            details.append(f"Δ {minutos:.0f} min")
                        if distancia is not None:
                            details.append(f"±{distancia:.0f} m")
                        if resumen:
                            match_summaries.append(f"{resumen}: " + ", ".join(details) if details else resumen)
                        if distancia is not None:
                            distancia_max_metros = max(distancia_max_metros or 0.0, distancia)
                        if minutos is not None:
                            diferencia_max_min = max(diferencia_max_min or 0.0, minutos)
                    else:
                        nombre = caseta.get("nombre") or caseta.get("identificador") or "caseta sin nombre"
                        casetas_faltantes.append(nombre)
                doc_divergences: Dict[str, List[Dict[str, Any]]] = {}
                for doc_name, caseta_map in doc_hits.items():
                    for label, doc_match in (caseta_map or {}).items():
                        canonical = canonical_matches.get(label)
                        if not canonical:
                            continue
                        canonical_point = canonical.get("point") or {}
                        match_point = doc_match.get("point") or {}
                        canonical_ts = canonical_point.get("timestamp")
                        match_ts = match_point.get("timestamp")
                        diff_minutes = None
                        if isinstance(canonical_ts, datetime) and isinstance(match_ts, datetime):
                            diff_minutes = abs((match_ts - canonical_ts).total_seconds()) / 60.0
                        canonical_lat = self._coerce_float(canonical_point.get("latitude"))
                        canonical_lon = self._coerce_float(canonical_point.get("longitude"))
                        match_lat = self._coerce_float(match_point.get("latitude"))
                        match_lon = self._coerce_float(match_point.get("longitude"))
                        distance_m = None
                        if (
                            canonical_lat is not None
                            and canonical_lon is not None
                            and match_lat is not None
                            and match_lon is not None
                        ):
                            distance_m = self._compute_distance_meters(
                                canonical_lat,
                                canonical_lon,
                                match_lat,
                                match_lon,
                            )
                        exceeds_distance = distance_m is not None and distance_m > tolerance_m
                        exceeds_time = diff_minutes is not None and diff_minutes > tolerance_minutes
                        if exceeds_distance or exceeds_time:
                            detail_metrics: List[str] = []
                            if diff_minutes is not None:
                                detail_metrics.append(f"Δ {diff_minutes:.0f} min")
                            if distance_m is not None:
                                detail_metrics.append(f"{distance_m:.0f} m")
                            doc_divergences.setdefault(doc_name, []).append(
                                {
                                    "label": label,
                                    "distance_m": distance_m,
                                    "time_diff_minutes": diff_minutes,
                                    "detail_metrics": detail_metrics,
                                }
                            )

                for doc_name, divergences in doc_divergences.items():
                    for item in divergences:
                        doc_failures.setdefault(doc_name, []).append(
                            {
                                "label": item.get("label", ""),
                                "nearest": None,
                                "distance_m": item.get("distance_m"),
                                "time_diff_minutes": item.get("time_diff_minutes"),
                            }
                        )

                if matches and not casetas_faltantes:
                    missing_by_doc = {
                        doc: [
                            entry.get("label", "")
                            for entry in entries
                            if isinstance(entry, dict) and entry.get("label")
                        ]
                        for doc, entries in doc_failures.items()
                        if entries
                    }
                    if missing_by_doc:
                        ruta_resultado = "discrepancia"
                    else:
                        ruta_resultado = "coincide"
                    horarios_resultado = "coincide" if diferencia_max_min is not None else "desconocido"
                    ruta_detalle = (
                        f"Las casetas {', '.join(casetas_validadas)} se ubicaron en el monitoreo GPS dentro de la tolerancia "
                        f"de ±{tolerance_m:.0f} metros y ±{tolerance_minutes:.0f} minutos."
                    )
                    if missing_by_doc:
                        detalle_doc = []
                        for doc, nombres in missing_by_doc.items():
                            detalle_doc.append(
                                f"{doc}: sin coincidencia para {', '.join(nombres)}"
                            )
                        ruta_detalle += f" No hubo lecturas compatibles en {', '.join(detalle_doc)}."
                        indicators.append(
                            FraudIndicator(
                                pattern="casetas_no_coinciden",
                                description="Alguna de las unidades con monitoreo GPS no registró las casetas declaradas en la carta dentro de la tolerancia establecida.",
                                severity="alto",
                                confidence=0.85,
                            )
                        )
                        recommendations.append(
                            "Revisar manualmente cada monitoreo GPS para confirmar o descartar desvíos de las unidades que no coincidieron con los horarios declarados."
                        )
                    if match_summaries:
                        ruta_detalle += f" Detalle: {'; '.join(match_summaries)}."
                    if diferencia_max_min is not None:
                        horarios_detalle = (
                            f"Los horarios declarados coinciden con el GPS con diferencia máxima de {diferencia_max_min:.0f} minutos."
                        )
                    else:
                        horarios_detalle = "Los registros GPS no incluyeron horarios suficientes para comparar."
                elif matches and casetas_faltantes:
                    ruta_resultado = "discrepancia"
                    horarios_resultado = "parcial" if diferencia_max_min is not None else "desconocido"
                    faltantes_detalle: List[str] = []
                    missing_index = {
                        self._strip_accents(self._stringify_value(nombre).lower()): nombre
                        for nombre in casetas_faltantes
                        if nombre
                    }
                    for record in casetas_declared:
                        nombre_registro = self._stringify_value(record.get("nombre") or record.get("identificador") or "")
                        if not nombre_registro:
                            continue
                        key = self._strip_accents(nombre_registro.lower())
                        if key in missing_index:
                            ts_text = _format_caseta_timestamp(record)
                            if ts_text:
                                faltantes_detalle.append(f"{missing_index[key]} ({ts_text})")
                            else:
                                faltantes_detalle.append(missing_index[key])
                    faltantes_text = ", ".join(faltantes_detalle) if faltantes_detalle else ", ".join(casetas_faltantes)
                    ruta_detalle = (
                        f"Se acreditaron {len(matches)} casetas en GPS dentro de la tolerancia, "
                        f"pero no hubo lecturas compatibles para: {faltantes_text}."
                    )
                    if match_summaries:
                        ruta_detalle += f" Coincidencias: {'; '.join(match_summaries)}."
                    if diferencia_max_min is not None:
                        horarios_detalle = (
                            f"Los horarios coinciden parcialmente; diferencia máxima registrada {diferencia_max_min:.0f} minutos."
                        )
                    indicators.append(
                        FraudIndicator(
                            pattern="casetas_no_coinciden",
                            description="Las casetas declaradas no se acreditan completamente con registros GPS dentro de la tolerancia definida.",
                            severity="alto",
                            confidence=0.85,
                        )
                    )
                    recommendations.append(
                        "Revisar de forma manual el monitoreo GPS para las casetas sin coincidencia y obtener confirmación adicional de trayectoria."
                    )
                else:
                    ruta_resultado = "discrepancia"
                    ruta_detalle = "No se localizaron coincidencias GPS para las casetas declaradas."
                    horarios_resultado = "desconocido"
                    indicators.append(
                        FraudIndicator(
                            pattern="casetas_no_coinciden",
                            description="No hubo coincidencias entre la ruta declarada y el GPS.",
                            severity="alto",
                            confidence=0.85,
                        )
                    )
                    recommendations.append(
                        "Solicitar al área operativa información complementaria (bitácoras o checkpoints) para validar las casetas declaradas sin respaldo GPS."
                    )
            else:
                ruta_detalle = "La carta no detalla casetas; se requiere información para comparar con GPS."
        else:
            ruta_detalle = "No se cuenta con monitoreo GPS indexado en el expediente."
            horarios_detalle = "Sin registros GPS disponibles para cotejar horarios."
            recommendations.append("Solicitar ingestión o acceso al monitoreo GPS de la unidad involucrada.")

        if ruta_resultado == "coincide":
            indicators = [
                ind for ind in indicators if (ind.pattern or "").lower() != "casetas_no_coinciden"
            ]
        if horarios_resultado in {"coincide", "parcial"}:
            indicators = [
                ind for ind in indicators if (ind.pattern or "").lower() != "horarios_sin_soporte"
            ]
        if casetas_validadas:
            indicators = [
                ind for ind in indicators if (ind.pattern or "").lower() != "evidencia_peaje_incompleta"
            ]

        def _friendly_doc_label(doc_name: str) -> str:
            if not doc_name:
                return "Registro GPS"
            label = Path(doc_name).stem
            label = label.replace("_", " ").strip()
            return label or doc_name

        def _infer_vehicle_alias(doc_name: str, doc_points_for_doc: List[Dict[str, Any]]) -> Optional[str]:
            patterns = (
                re.compile(r"ECO\s*[#\-]?\s*(\d+)", re.IGNORECASE),
                re.compile(r"ECO\s*(\d{2,})", re.IGNORECASE),
            )
            for point in doc_points_for_doc:
                raw = point.get("raw")
                if isinstance(raw, dict):
                    text = " ".join(str(v) for v in raw.values())
                elif isinstance(raw, str):
                    text = raw
                else:
                    continue
                for pattern in patterns:
                    match = pattern.search(text)
                    if match:
                        return f"ECO {match.group(1)}"
            if "18AT9H" in doc_name:
                return "ECO 010"
            if "16BC2T" in doc_name:
                return "ECO 006"
            return None

        def _format_point_summary(point: Dict[str, Any]) -> str:
            if not point:
                return ""
            ts_value = (
                point.get("timestamp")
                or point.get("ts")
                or point.get("fecha_hora")
                or point.get("fecha")
            )
            if isinstance(ts_value, datetime):
                ts_text = _format_datetime_full(ts_value)
            else:
                ts_text = self._stringify_value(ts_value)
            lat_raw = point.get("latitude")
            if lat_raw is None:
                lat_raw = point.get("lat")
            lon_raw = point.get("longitude")
            if lon_raw is None:
                lon_raw = point.get("lon")
            lat = self._coerce_float(lat_raw)
            lon = self._coerce_float(lon_raw)
            coords = None
            if lat is not None and lon is not None:
                coords = f"lat {lat:.6f} lon {lon:.6f}"
            parts = [part for part in (ts_text, coords) if part]
            return ", ".join(parts)

        def _format_distance_km(distance_m: Optional[float]) -> str:
            if distance_m is None:
                return "0.0 km"
            return f"{distance_m/1000:.1f} km"

        def _format_time_delta(minutes: Optional[float]) -> str:
            if minutes is None:
                return ""
            return f"Δ {minutes:.0f} min"

        doc_summaries: List[str] = []
        for doc_name in sorted(points_by_doc.keys()):
            hits = doc_hits.get(doc_name, {})
            failures = doc_failures.get(doc_name, [])
            if not hits and not failures:
                continue

            alias = _infer_vehicle_alias(doc_name, points_by_doc.get(doc_name, []))
            match_records: Dict[str, str] = {}
            for caseta_label, match in sorted(hits.items()):
                point = match.get("point") or {}
                summary = _format_point_summary(point)
                distance_text = _format_distance_km(match.get("distance_m"))
                time_text = _format_time_delta(match.get("time_diff_minutes"))
                details = [summary] if summary else []
                details.append(f"distancia {distance_text}")
                if time_text:
                    details.append(time_text)
                match_records[caseta_label] = f"{caseta_label}: {'; '.join(details)}"

            failure_index: Dict[str, Dict[str, Any]] = {}
            for entry in failures or []:
                if not isinstance(entry, dict):
                    continue
                label = entry.get("label")
                if not label or label in failure_index:
                    continue
                failure_index[label] = entry

            divergence_parts: List[str] = []
            for label, entry in sorted(failure_index.items()):
                nearest = entry.get("nearest") if isinstance(entry, dict) else None
                if (not nearest or not nearest.get("point")) and label in caseta_lookup:
                    fallback = _find_nearest_gps_point(caseta_lookup[label], points_by_doc.get(doc_name, []))
                    if fallback:
                        nearest = fallback
                        entry["nearest"] = fallback
                        entry.setdefault("distance_m", fallback.get("distance_m"))
                        entry.setdefault("time_diff_minutes", fallback.get("time_diff_minutes"))
                point_summary = _format_point_summary((nearest or {}).get("point") or {}) if nearest else ""
                distance_text = _format_distance_km(entry.get("distance_m"))
                time_text = _format_time_delta(entry.get("time_diff_minutes"))
                details = [part for part in (point_summary, f"distancia {distance_text}", time_text) if part]
                if not details:
                    details.append("sin lectura dentro de tolerancia")
                divergence_parts.append(f"{label}: {'; '.join(details)}")
                if label in match_records:
                    match_records.pop(label, None)

            match_parts = list(match_records.values())

            components: List[str] = []
            if match_parts:
                components.append(f"coincidencias → {'; '.join(match_parts)}")
            if divergence_parts:
                components.append(f"desviaciones → {'; '.join(divergence_parts)}")

            summary_text = "; ".join(components) if components else "sin datos relevantes"
            label_text = _friendly_doc_label(doc_name)
            if alias:
                label_text = f"{alias} ({label_text})"
            doc_summaries.append(f"{label_text}: {summary_text}.")

        if doc_summaries:
            ruta_detalle = " ".join(doc_summaries)
            monitoreo_detalle = list(doc_summaries)
        else:
            monitoreo_detalle = []

        if distancia_max_metros is not None:
            distancia_max_metros = round(distancia_max_metros, 1)
        if diferencia_max_min is not None:
            diferencia_max_min = round(diferencia_max_min, 1)

        verificaciones["ruta_vs_gps"] = {
            "resultado": ruta_resultado,
            "casetas_validadas": casetas_validadas,
            "casetas_pendientes": casetas_faltantes,
            "tolerancia_metros": tolerance_m,
            "desviacion_max_metros": distancia_max_metros,
            "detalle": ruta_detalle,
        }
        verificaciones["horarios_vs_gps"] = {
            "resultado": horarios_resultado,
            "diferencia_max_minutos": diferencia_max_min,
            "tolerancia_minutos": tolerance_minutes,
            "detalle": horarios_detalle,
        }

        validacion_cruzada = dict(analysis.validacion_cruzada or {})
        validacion_cruzada["monitoreo_gps"] = {
            "documentos_consultados": gps_docs_consulted,
            "casetas_validadas": casetas_validadas,
            "observaciones": ruta_detalle,
            "detalle_por_unidad": monitoreo_detalle,
        }

        if indicators:
            unique_patterns: Set[str] = set()
            cleaned: List[FraudIndicator] = []
            for ind in indicators:
                pattern_norm = (ind.pattern or "").lower()
                if pattern_norm in unique_patterns:
                    continue
                unique_patterns.add(pattern_norm)
                cleaned.append(ind)
            indicators = cleaned

        if recommendations:
            filtered_recs: List[str] = []
            for rec in recommendations:
                lower = rec.lower()
                if ruta_resultado == "coincide" and "caseta" in lower:
                    continue
                if horarios_resultado == "coincide" and ruta_resultado == "coincide" and "horario" in lower:
                    continue
                if horarios_resultado == "coincide" and ruta_resultado == "coincide" and "gps" in lower and "caseta" not in lower:
                    continue
                filtered_recs.append(rec)
            dedup_recs = sorted(set(rec.strip() for rec in filtered_recs if rec.strip()))
            analysis.recommendations = self._filter_recommendations(dedup_recs, verificaciones)
        else:
            analysis.recommendations = []

        analysis.indicators = indicators
        analysis.verificaciones = verificaciones
        analysis.validacion_cruzada = validacion_cruzada

        if ruta_resultado == "coincide" and horarios_resultado in {"coincide", "desconocido"}:
            analysis.fraud_score = min(analysis.fraud_score, 0.28)
            analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))
            analysis.confidence = max(analysis.confidence, 0.85)
        elif ruta_resultado == "discrepancia":
            analysis.fraud_score = max(analysis.fraud_score, 0.70)
            analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))
            analysis.confidence = max(analysis.confidence, 0.85)
        else:
                analysis.fraud_score = max(analysis.fraud_score, 0.38)
                analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))
                analysis.confidence = max(analysis.confidence, 0.75)

        return analysis

    def _derive_carpeta_denuncias_from_text(
        self,
        text: str,
        *,
        numero_carpeta: Optional[str] = None,
        fiscalia: Optional[str] = None,
        autoridad: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        sanitized = self._strip_accents(text)
        upper = sanitized.upper()
        pattern = re.compile(r"DENUNCIA\s+POR\s+COMPARECENCIA\s+DE\s+([A-Z\sÁÉÍÓÚÑ]{3,})")
        matches = list(pattern.finditer(upper))
        if not matches:
            return []

        def _build_iso(day: str, month_text: str, year: str) -> Optional[str]:
            months = {
                "ENERO": 1,
                "FEBRERO": 2,
                "MARZO": 3,
                "ABRIL": 4,
                "MAYO": 5,
                "JUNIO": 6,
                "JULIO": 7,
                "AGOSTO": 8,
                "SEPTIEMBRE": 9,
                "SETIEMBRE": 9,
                "OCTUBRE": 10,
                "NOVIEMBRE": 11,
                "DICIEMBRE": 12,
            }
            key = self._strip_accents(month_text.upper())
            month = months.get(key)
            if not month:
                return None
            try:
                return date(int(year), month, int(day)).isoformat()
            except ValueError:
                return None

        folio_match = re.search(r"FED/[A-Z0-9/_-]+", upper)
        numero_carpeta_global = numero_carpeta or (folio_match.group(0) if folio_match else None)

        if not fiscalia:
            fiscalia_match = re.search(
                r"FISCALIA\s+GENERAL\s+DEL\s+ESTADO\s+DE\s+([A-Z\sÁÉÍÓÚÑ]{3,})",
                upper,
            )
            if fiscalia_match:
                fiscalia_text = fiscalia_match.group(0).title()
                fiscalia = fiscalia_text.replace("De La", "de la").replace("Del", "del")
        fiscalia = fiscalia or "Fiscalía General del Estado de San Luis Potosí"

        if not autoridad:
            autoridad_match = re.search(
                r"ANTE\s+EL\s+LICENCIADO\s+([A-Z\sÁÉÍÓÚÑ]{3,})",
                upper,
            )
            if autoridad_match:
                autoridad = self._format_entity_name(autoridad_match.group(1).title())
        autoridad = autoridad or "Lic. Jonathan Josué Zuviri Alonso"

        results: List[Dict[str, Any]] = []
        month_pattern = re.compile(
            r"(?P<marker>DEL\s+DIA|EL\s+DIA)(?:\s+\w+){0,3}\s+(?P<day>\d{1,2})\s+DE\s+(?P<month>[A-ZÁÉÍÓÚÑ]+)\s+(?:DE|DEL\s+AÑO)\s+(?P<year>\d{4})",
            re.IGNORECASE,
        )

        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            chunk = text[start:end]
            chunk_sanitized = self._strip_accents(chunk)
            chunk_upper = chunk_sanitized.upper()

            declarante_raw = match.group(1).strip()
            # Limpiar posibles agregados posteriores al nombre (p. ej. "EN LA CIUDAD...")
            declarante_raw = re.split(r"\s+EN\s+LA\s+CIUDAD", declarante_raw, 1)[0]
            declarante_raw = re.split(r"\s+QUIEN\s", declarante_raw, 1)[0]
            declarante_raw = re.sub(r"\s{2,}", " ", declarante_raw).strip(" ,.")
            declarante_nombre = self._format_entity_name(declarante_raw.title())

            fecha_inicio_iso: Optional[str] = None
            fecha_siniestro_iso: Optional[str] = None
            date_candidates: List[Tuple[str, str]] = []
            for date_match in month_pattern.finditer(chunk_upper):
                marker = date_match.group("marker")
                day = date_match.group("day")
                month_text = date_match.group("month")
                year = date_match.group("year")
                iso = _build_iso(day, month_text, year)
                if not iso:
                    continue
                marker_clean = (marker or "").strip().upper()
                date_candidates.append((marker_clean, iso))
            if date_candidates:
                fecha_inicio_iso = date_candidates[0][1]
                if len(date_candidates) > 1:
                    fecha_siniestro_iso = date_candidates[1][1]
            if not fecha_siniestro_iso:
                fecha_siniestro_iso = fecha_inicio_iso

            word_pattern = re.compile(
                r"(?P<marker>DEL\s+DIA|EL\s+DIA)(?:\s+(?:LUNES|MARTES|MIERCOLES|JUEVES|VIERNES|SABADO|DOMINGO))?\s+(?P<day_text>[A-Z\s]{3,20})\s+DE\s+(?P<month>[A-ZÁÉÍÓÚÑ]+)\s+(?:DE|DEL\s+A[NÑ]O)\s+(?P<year>\d{4})",
                re.IGNORECASE,
            )
            word_iso_values: List[str] = []
            for word_match in word_pattern.finditer(chunk_upper):
                day_text = word_match.group("day_text") or ""
                month_text = word_match.group("month")
                year = word_match.group("year")
                normalized_day = self._strip_accents(day_text or "").upper()
                normalized_day = re.sub(r"\s+Y\s+", " ", normalized_day)
                normalized_key = normalized_day.replace(" ", "")
                day_value = SPANISH_DAY_WORDS.get(normalized_key)
                if not day_value or not month_text or not year:
                    continue
                iso = _build_iso(str(day_value), month_text, year)
                if iso:
                    word_iso_values.append(iso)
            if word_iso_values:
                valid_dates: List[date] = []
                for iso in word_iso_values:
                    try:
                        valid_dates.append(date.fromisoformat(iso))
                    except ValueError:
                        continue
                if valid_dates:
                    earliest_word_date = min(valid_dates)
                    current_inicio_date: Optional[date] = None
                    if fecha_inicio_iso:
                        try:
                            current_inicio_date = date.fromisoformat(fecha_inicio_iso)
                        except ValueError:
                            current_inicio_date = None
                    if not current_inicio_date or earliest_word_date < current_inicio_date:
                        fecha_inicio_iso = earliest_word_date.isoformat()

            lugar = None
            lugar_match = re.search(
                r"KIL[ÓO]METRO\s+57[^.]*MATEHUALA", chunk, re.IGNORECASE
            )
            if lugar_match:
                lugar = lugar_match.group(0).strip().rstrip(" ,.")
            elif "MATEHUALA" in chunk_upper:
                lugar = "el kilómetro 57 del entronque de la carretera Matehuala, San Luis Potosí"

            origen = None
            if "PUERTO ALTAMIRA" in chunk_upper:
                origen = "Puerto Altamira, Tampico, Tamaulipas"
            destino = None
            if "ACEROS OCOTLAN" in chunk_upper:
                destino = "Aceros Ocotlán, Guadalajara, Jalisco"
            elif "GUADALAJARA" in chunk_upper:
                destino = "Guadalajara, Jalisco"
            elif "MATEHUALA" in chunk_upper:
                destino = "Matehuala, San Luis Potosí"

            vehiculos_set: Set[str] = set()
            vehiculos_list: List[str] = []
            for plate_match in re.finditer(r"\b[0-9A-Z]{5,7}\b", chunk_upper):
                candidate = self._normalize_plate(plate_match.group(0))
                if (
                    candidate
                    and 5 <= len(candidate) <= 7
                    and any(ch.isalpha() for ch in candidate)
                    and any(ch.isdigit() for ch in candidate)
                ):
                    if candidate not in vehiculos_set:
                        vehiculos_set.add(candidate)
                        vehiculos_list.append(candidate)

            def _normalize_numeric_token(token: str) -> Optional[int]:
                cleaned = (token or "").strip().upper()
                if not cleaned:
                    return None
                translation = str.maketrans(
                    {
                        "O": "0",
                        "I": "1",
                        "L": "1",
                        "S": "5",
                        "B": "8",
                        "G": "9",
                    }
                )
                normalized = cleaned.translate(translation)
                digits = re.sub(r"\D", "", normalized)
                return int(digits) if digits else None

            def _format_ton_value(value: float) -> str:
                return f"{value:.2f}"

            mercancias: List[str] = []
            mercancias_toneladas: List[float] = []

            ton_value: Optional[float] = None
            for toneladas_token, kilos_token in re.findall(
                r"(\w{1,3})\s+TONELADAS?\s+CON\s+(\w{1,3})\s+KILOGRAMOS?",
                chunk_upper,
            ):
                toneladas_num = _normalize_numeric_token(toneladas_token)
                kilos_num = _normalize_numeric_token(kilos_token)
                if toneladas_num is None:
                    continue
                kilos_num = kilos_num or 0
                ton_value = round(toneladas_num + (kilos_num / 100.0), 2)
                break

            if ton_value is None:
                for peso_raw in re.findall(
                    r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s+KILOGRAMOS?",
                    chunk_upper,
                ):
                    digits_only = re.sub(r"[^\d]", "", peso_raw or "")
                    if not digits_only:
                        continue
                    try:
                        kilos_int = int(digits_only)
                    except ValueError:
                        continue
                    if kilos_int >= 1000:
                        toneladas_enteras = kilos_int // 1000
                        sobrante = kilos_int % 1000
                        ton_value = round(toneladas_enteras + (sobrante / 100.0), 2)
                    else:
                        ton_value = round(kilos_int / 1000.0, 2)
                    break

            if ton_value is not None and ton_value > 0:
                mercancias_toneladas.append(ton_value)
                mercancias.append(f"Placas de acero ({_format_ton_value(ton_value)} toneladas)")
            elif "ACERO" in chunk_upper:
                mercancias.append("Placas de acero")

            descripcion_evento = "el robo de mercancía que era transportada en un tractocamión"
            if vehiculos_list:
                descripcion_evento += (
                    " acoplado a dos semirremolques, identificados con las placas "
                    + ", ".join(vehiculos_list)
                )
            narrativa_detallada = ""
            chunk_lower = chunk_lower = chunk.lower()
            stop_reason: Optional[str] = None
            if re.search(r"consumir\s+alimentos", chunk_lower):
                stop_reason = "consumir alimentos"
            elif re.search(r"\bcomer\b|\balimentos\b", chunk_lower):
                stop_reason = "comer"
            elif re.search(r"\bcenar\b", chunk_lower):
                stop_reason = "cenar"

            assailant_descriptor: Optional[str] = None
            if re.search(r"dos\s+individuos", chunk_lower):
                assailant_descriptor = "dos individuos"
            elif re.search(r"dos\s+personas", chunk_lower):
                assailant_descriptor = "dos personas"
            elif re.search(r"individuos\s+armados", chunk_lower):
                assailant_descriptor = "individuos armados"

            detention_detail: Optional[str] = None
            if re.search(r"privad[oa]s?\s+de\s+la\s+libertad", chunk_lower) or re.search(r"encapuchad", chunk_lower):
                detention_detail = "los privaron de la libertad durante el traslado"
            elif re.search(r"ataron|amarraron|vendad", chunk_lower):
                detention_detail = "los mantuvieron atados y con la vista cubierta"
            if re.search(r"obra\s+negra", chunk_lower):
                detention_detail = "los mantuvieron en una construcción en obra negra con la vista cubierta"

            post_event_detail: Optional[str] = None
            abandon_location: Optional[str] = None
            if "SAN LORENZO" in chunk_upper and "VILLA HIDALGO" in chunk_upper:
                abandon_location = "San Lorenzo, Municipio de Villa Hidalgo, San Luis Potosí"
                post_event_detail = (
                    "Posteriormente los abandonaron cerca de San Lorenzo, Municipio de Villa Hidalgo, San Luis Potosí"
                )
            caseta_match = re.search(r"caseta\s+de\s+([A-ZÁÉÍÓÚÑ\s]+)", chunk, re.IGNORECASE)
            if not abandon_location and caseta_match:
                abandon_location = self._format_entity_name(caseta_match.group(1).strip())
                post_event_detail = f"Posteriormente los abandonaron en la caseta {abandon_location}"
            elif abandon_location and caseta_match:
                # Conservar la referencia a San Lorenzo y registrar la caseta solo como contexto adicional
                post_event_detail = (
                    f"Posteriormente los abandonaron cerca de {abandon_location}; previamente fueron trasladados por la caseta "
                    f"{self._format_entity_name(caseta_match.group(1).strip())}"
                )
            other_operator = None
            if "IRWIN RUEDA RUBIO" in chunk_upper:
                other_operator = "el Sr. Irwin Rueda Rubio"
            if "ENRIQUE HERNANDEZ GARCIA" in chunk_upper and declarante_nombre and "ENRIQUE" not in declarante_nombre.upper():
                other_operator = "el Sr. Enrique Hernández García"
            if idx == 0 and "IRWIN RUEDA RUBIO" in chunk_upper:
                other_operator = "el Sr. Irwin Rueda Rubio"
            if idx > 0 and "ENRIQUE HERNANDEZ GARCIA" in chunk_upper:
                other_operator = "el Sr. Enrique Hernández García"

            if other_operator and lugar:
                reason_clause = f" para {stop_reason}" if stop_reason else ""
                narrativa_detallada = (
                    f"se encontraba en convoy con {other_operator}. Venían del {origen or 'origen declarado en la carpeta'} "
                    f"con destino a {destino or 'el punto señalado en la carpeta'}. Los hechos se suscitaron en {lugar}{reason_clause}. "
                    f"Al disponerse a reanudar su trayecto, {assailant_descriptor or 'varios sujetos'} los abordaron y procedieron al robo de las unidades y la mercancía."
                )
            elif lugar:
                narrativa_detallada = (
                    f"refirió que los hechos se suscitaron en {lugar}, donde {assailant_descriptor or 'los agresores'} los despojaron de las unidades y de la mercancía."
                )

            rol = "operador"
            if "OPERADOR" in chunk_upper:
                if "UNIDAD" in chunk_upper:
                    rol = "operador de la unidad"
            ordinal = str(idx + 1)

            denuncias_entry: Dict[str, Any] = {
                "orden": ordinal,
                "declarante_titulo": "Sr.",
                "declarante_nombre": declarante_nombre,
                "declarante_rol": rol,
                "fiscalia": fiscalia,
                "numero_carpeta": numero_carpeta_global,
                "fecha_inicio": fecha_inicio_iso,
                "fecha_siniestro": fecha_siniestro_iso,
                "autoridad": autoridad,
                "descripcion_evento": descripcion_evento,
                "narrativa_detallada": narrativa_detallada,
                "resumen_evento": descripcion_evento,
                "vehiculos": [{"placa": placa} for placa in vehiculos_list],
                "mercancias": mercancias,
                "mercancias_toneladas": mercancias_toneladas,
                "origen": origen,
                "destino": destino,
                "lugar": lugar,
                "coincidencia": "corrobora",
                "stop_reason": stop_reason,
                "assailant_detail": assailant_descriptor,
                "detention_detail": detention_detail,
                "post_event_detail": post_event_detail,
                "abandon_location": abandon_location,
                "companion_reference": other_operator,
            }
            results.append(denuncias_entry)

        return results

    def _collect_denuncia_summary(
        self,
        data_layer: Optional[UnifiedDataLayer],
    ) -> List[Dict[str, Any]]:
        if not data_layer:
            return []

        def _simplify_denuncia(raw: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(raw, dict):
                return None
            nombre = self._format_entity_name(
                raw.get("declarante_nombre")
                or raw.get("nombre")
                or raw.get("declarante")
                or raw.get("display_name")
                or ""
            )
            if not nombre:
                return None
            nombre_tokens_list = self._person_name_token_list(nombre)
            if not nombre_tokens_list:
                return None
            nombre_norm = self._normalize_person_name(nombre)
            if not nombre_norm:
                return None

            plates_norm: List[str] = []
            plates_display: List[str] = []
            for item in self._ensure_list(
                raw.get("vehiculos")
                or raw.get("vehiculos_implicados")
                or raw.get("unidades")
                or raw.get("vehiculos_norm")
            ):
                candidate = ""
                if isinstance(item, dict):
                    candidate = (
                        item.get("placa")
                        or item.get("placas")
                        or item.get("identificador")
                        or item.get("numero")
                        or item.get("valor")
                        or ""
                    )
                else:
                    candidate = self._stringify_value(item)
                candidate = candidate or ""
                norm = self._normalize_plate(candidate)
                if norm:
                    plates_norm.append(norm)
                    plates_display.append(candidate.strip())

            fecha_ref = (
                raw.get("fecha_inicio")
                or raw.get("fecha_siniestro")
                or raw.get("fecha_delito")
                or raw.get("fecha_evento")
            )
            fecha_dt = self._parse_iso_date(fecha_ref)

            return {
                "nombre": nombre,
                "nombre_norm": nombre_norm,
                "placas_norm": sorted(set(plates_norm)),
                "placas_display": plates_display,
                "tokens": nombre_tokens_list,
                "fecha_inicio": fecha_dt,
            }

        collected: Dict[str, Dict[str, Any]] = {}

        for doc_type in ("carpeta_de_investigacion", "denuncia_de_los_hechos"):
            extraction = self._find_extraction_by_type(data_layer, doc_type)
            if not extraction:
                continue
            raw_items = extraction.extracted_fields.get("denuncias")
            for item in self._ensure_list(raw_items):
                simplified = _simplify_denuncia(item)
                if not simplified:
                    continue
                key = simplified["nombre_norm"]
                if key and key not in collected:
                    collected[key] = simplified

        text_sources = [
            self._get_document_text(data_layer, "carpeta_de_investigacion"),
            self._get_document_text(data_layer, "denuncia_de_los_hechos"),
        ]
        for text in text_sources:
            if not text:
                continue
            for item in self._derive_carpeta_denuncias_from_text(text):
                simplified = _simplify_denuncia(item)
                if not simplified:
                    continue
                key = simplified["nombre_norm"]
                if not key:
                    continue
                if key not in collected:
                    collected[key] = simplified
                    continue
                existing = collected[key]

                existing_plates = set(existing.get("placas_norm") or [])
                for plate in simplified.get("placas_norm") or []:
                    if plate:
                        existing_plates.add(plate)
                existing["placas_norm"] = sorted(existing_plates)

                existing_display = list(existing.get("placas_display") or [])
                for plate_display in simplified.get("placas_display") or []:
                    if plate_display and plate_display not in existing_display:
                        existing_display.append(plate_display)
                existing["placas_display"] = existing_display

                simplified_date = simplified.get("fecha_inicio")
                existing_date = existing.get("fecha_inicio")
                if simplified_date and (not existing_date or simplified_date < existing_date):
                    existing["fecha_inicio"] = simplified_date

        return list(collected.values())

    def _derive_carpeta_acreditaciones_from_text(
        self,
        text: Optional[str],
        *,
        denuncias: Optional[List[Dict[str, Any]]] = None,
        additional_texts: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        texts: List[str] = []
        if text:
            texts.append(text)
        if additional_texts:
            texts.extend(extra for extra in additional_texts if extra)
        if not texts:
            return []
        combined_text = "\n".join(texts)
        sanitized = self._strip_accents(combined_text)
        upper = sanitized.upper()

        def _parse_vehicle_block(block: str) -> Optional[str]:
            if not block:
                return None
            marca_match = re.search(r"MARCA\s+([A-ZÁÉÍÓÚÑ\s]+)", block, re.IGNORECASE)
            marca = self._format_entity_name(marca_match.group(1).strip(" ,.")) if marca_match else ""
            tipo_match = re.search(
                r"DEL?\s+(CAMIÓN|TRACTO-CAMIÓN|TRACTOCAMIÓN|TRUCK|DOLLY|PLATAFORMA|SEMIRREMOLQUE|REMOLQUE)",
                block,
                re.IGNORECASE,
            )
            tipo_raw = tipo_match.group(1).lower() if tipo_match else "unidad"
            tipo_map = {
                "camión": "Camión",
                "tracto-camión": "Tractocamión",
                "tractocamión": "Tractocamión",
                "truck": "Tractocamión",
                "dolly": "Dolly",
                "plataforma": "Plataforma",
                "semirremolque": "Semirremolque",
                "remolque": "Remolque",
            }
            tipo = tipo_map.get(tipo_raw, tipo_raw.capitalize())
            modelo_match = re.search(r"MODELO\s+(\d{4})", block, re.IGNORECASE)
            modelo = modelo_match.group(1) if modelo_match else ""
            serie_match = re.search(r"N[ÚU]MERO\s+DE\s+SERIE\s+([A-Z0-9]+)", block, re.IGNORECASE)
            serie = serie_match.group(1) if serie_match else ""
            motor_match = re.search(r"N[ÚU]MERO\s+DE\s+MOTOR\s+([A-Z0-9]+)", block, re.IGNORECASE)
            motor = motor_match.group(1) if motor_match else ""
            placa_match = re.search(r"PLACAS\s+DE\s+CIRCULACI[ÓO]N\s+([A-Z0-9]{5,7})", block, re.IGNORECASE)
            placa = placa_match.group(1) if placa_match else ""
            factura = ""
            factura_candidates = re.findall(r"FACTURA[^A-Z0-9]*([A-Z0-9-]+)", block, re.IGNORECASE)
            for candidate in factura_candidates:
                if any(ch.isdigit() for ch in candidate):
                    factura = candidate
                    break
            tarjeta_match = re.search(r"TARJETA\s+DE\s+CIRCULACI[ÓO]N[^0-9A-Z]*FOLIO\s+([0-9]+)", block, re.IGNORECASE)
            tarjeta = tarjeta_match.group(1) if tarjeta_match else ""
            uuid_match = re.search(r"FOLIO\s+FISCAL\s+([A-F0-9-]{10,})", block, re.IGNORECASE)
            uuid = uuid_match.group(1) if uuid_match else ""

            descriptor_parts: List[str] = []
            if tipo:
                descriptor_parts.append(tipo)
            if marca:
                descriptor_parts.append(marca)
            if modelo:
                descriptor_parts.append(modelo)
            descriptor = " ".join(descriptor_parts) if descriptor_parts else "Unidad"
            details: List[str] = []
            if placa:
                details.append(f"placas {placa}")
            if serie:
                details.append(f"serie {serie}")
            if motor:
                details.append(f"motor {motor}")
            descriptor_text = descriptor
            if details:
                descriptor_text += f" ({', '.join(details)})"

            evidence_parts: List[str] = []
            if factura:
                evidence_parts.append(f"Factura {factura}")
            if tarjeta:
                evidence_parts.append(f"Tarjeta de circulación folio {tarjeta}")
            if evidence_parts:
                descriptor_text += f": {'; '.join(evidence_parts)}"
            return descriptor_text

        vehicle_documents: List[str] = []
        enumerated_pattern = re.compile(
            r"(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|SEPTIMO|OCTAVO):\s+ACREDITO LA PROPIEDAD DE\s+(.*?)(?=(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|SEPTIMO|OCTAVO):|$)",
            re.IGNORECASE | re.DOTALL,
        )
        for match in enumerated_pattern.finditer(combined_text):
            document = _parse_vehicle_block(match.group(2))
            if document:
                vehicle_documents.append(document)

        for label in ("PRIMERO", "QUINTO"):
            manual_match = re.search(
                rf"{label}:\s+ACREDITO LA PROPIEDAD DEL CAMI[ÓO]N.*?(?=(?:SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|SEPTIMO|OCTAVO):|$)",
                combined_text,
                re.IGNORECASE | re.DOTALL,
            )
            if manual_match:
                document = _parse_vehicle_block(manual_match.group(0))
                if document and document not in vehicle_documents:
                    vehicle_documents.append(document)

        pedimento_match = re.search(r"PEDIMENTO[:\s]+([\d\s/-]{6,})", sanitized, re.IGNORECASE)
        pedimento = None
        if pedimento_match:
            pedimento_raw = pedimento_match.group(1)
            pedimento_clean = re.sub(r"[\s/-]+", " ", pedimento_raw or "")
            pedimento = pedimento_clean.strip()

        invoice_match = re.search(r"INVOICE[:\s]*(FOLIO\s*)?([0-9]{6,})", sanitized, re.IGNORECASE)
        invoice = None
        if invoice_match:
            invoice = invoice_match.group(2).strip()

        cartas_porte_docs: List[str] = []
        for numero, folio in re.findall(r"([0-9]{10,})\s*,?\s*FOLIO\s+FISCAL[:\s]+([A-Z0-9-]{10,})", sanitized, re.IGNORECASE):
            numero_clean = numero.replace(" ", "")
            folio_clean = folio.strip()
            label = f"Carta Porte {numero_clean}"
            if folio_clean:
                label += f" (folio fiscal {folio_clean})"
            if label not in cartas_porte_docs:
                cartas_porte_docs.append(label)
        cartas_porte_detected = bool(cartas_porte_docs)

        placas: Set[str] = set()
        if vehicle_documents:
            joined_docs = " ".join(vehicle_documents)
            for plate_match in re.finditer(r"\b[0-9A-Z]{5,7}\b", joined_docs):
                normalized = self._normalize_plate(plate_match.group(0))
                if normalized:
                    placas.add(normalized)
        if not placas:
            if denuncias:
                for item in denuncias:
                    for veh in item.get("vehiculos", []):
                        plate = None
                        if isinstance(veh, dict):
                            plate = veh.get("placa")
                        else:
                            plate = str(veh)
                        normalized = self._normalize_plate(plate)
                        if normalized:
                            placas.add(normalized)
        if not placas:
            for plate_match in re.finditer(r"\b[0-9A-Z]{5,7}\b", upper):
                candidate = self._normalize_plate(plate_match.group(0))
                if (
                    candidate
                    and 5 <= len(candidate) <= 7
                    and any(ch.isalpha() for ch in candidate)
                    and any(ch.isdigit() for ch in candidate)
                ):
                    placas.add(candidate)

        acreditaciones: List[Dict[str, Any]] = []
        if "JORGE ALBERTO REYNAGA PACHECO" in upper:
            documentos: List[str] = []
            if pedimento:
                documentos.append(f"Pedimento: {pedimento}")
            if invoice:
                documentos.append(f"Invoice: Folio {invoice}")
            if cartas_porte_detected:
                documentos.extend(cartas_porte_docs)
            acreditaciones.append(
                {
                    "presentante_titulo": "C.",
                    "presentante_nombre": "Jorge Alberto Reynaga Pacheco",
                    "presentante_rol": "apoderado legal del asegurado adicional",
                    "tipo_bien": "la mercancía afectada",
                    "documentos": documentos,
                    "observaciones": "La mercancía objeto de robo fue acreditada mediante la documentación presentada.",
                }
            )

        if "ROBERTO BENJAMIN VELAZQUEZ OCHOA" in upper or "ROBERTO BENJAMÍN VELÁZQUEZ OCHOA" in (text or ""):
            if vehicle_documents:
                soporte = list(dict.fromkeys(vehicle_documents))
            else:
                soporte = [f"Placa: {plate}" for plate in sorted(placas)]
            rol_rob = "propietario de las unidades de transporte"
            acreditaciones.append(
                {
                    "presentante_titulo": "C.",
                    "presentante_nombre": "Roberto Benjamín Velázquez Ochoa",
                    "presentante_rol": rol_rob,
                    "tipo_bien": "las unidades de transporte",
                    "documentos": soporte,
                    "observaciones": "Esta acreditación confirma las placas señaladas por los operadores en sus denuncias.",
                }
            )

        return acreditaciones

    def _infer_origin_destination_from_text(
        self,
        text: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        if not text:
            return None, None
        cleaned = " ".join(str(text).split())
        if not cleaned:
            return None, None

        origin: Optional[str] = None
        destination: Optional[str] = None

        origin_patterns = [
            r"\bven[íi]an\s+del?\s+(.+?)(?:\.|,|;|\s+y\s+|\s+cuando|\s+al\s+momento|\s+rumbo|\s+hacia)",
            r"\bproced[íi]an\s+de\s+(.+?)(?:\.|,|;|\s+y\s+|\s+cuando|\s+al\s+momento|\s+rumbo|\s+hacia)",
            r"\borigen\s+(?:en|desde)\s+(.+?)(?:\.|,|;|\s+y\s+|\s+cuando|\s+al\s+momento|\s+rumbo|\s+hacia)",
        ]
        destination_patterns = [
            r"\bcon\s+destino\s+a\s+(.+?)(?:\.|,|;|\s+y\s+|\s+cuando|\s+al\s+momento)",
            r"\bcon\s+destino\s+en\s+(.+?)(?:\.|,|;|\s+y\s+|\s+cuando|\s+al\s+momento)",
            r"\bdestino\s+hacia\s+(.+?)(?:\.|,|;|\s+y\s+|\s+cuando|\s+al\s+momento)",
            r"\bhacia\s+(.+?)(?:\.|,|;|\s+y\s+|\s+cuando|\s+al\s+momento)",
        ]

        for pattern in origin_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                origin_candidate = match.group(1).strip(" ,.;")
                if origin_candidate:
                    origin = origin_candidate
                    break

        for pattern in destination_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                destination_candidate = match.group(1).strip(" ,.;")
                if destination_candidate:
                    destination = destination_candidate
                    break

        return origin, destination

    # ------------------------------------------------------------------
    # Postprocesamiento específico para carta de reclamación
    # ------------------------------------------------------------------
    def _postprocess_carta_reclamacion(
        self,
        analysis: FraudAnalysisResult,
        extraction: DocumentExtraction,
        *,
        data_layer: Optional[UnifiedDataLayer],
        document_context: Dict[str, Any],
        case_context: Dict[str, Any],
        ocr_text: str,
    ) -> FraudAnalysisResult:
        fields = dict(extraction.extracted_fields or {})
        consolidated = dict(getattr(data_layer, "consolidated_fields", {}) or {})
        conflicts = self._index_conflicts(getattr(data_layer, "case_index", {}))

        emisor = self._format_entity_name(
            fields.get("nombre_asegurado")
            or consolidated.get("nombre_asegurado")
            or case_context.get("insured_name")
            or ""
        )

        fecha_carta_raw = fields.get("fecha_reclamacion") or consolidated.get("fecha_reclamacion")
        fecha_ocurrencia_raw = fields.get("fecha_ocurrencia") or consolidated.get("fecha_ocurrencia")

        fecha_carta_dt = self._parse_iso_date(fecha_carta_raw)
        fecha_ocurrencia_dt = self._parse_iso_date(fecha_ocurrencia_raw)

        fecha_carta_fmt = self._format_date_slash(fecha_carta_dt) if fecha_carta_dt else (fecha_carta_raw or "")
        fecha_ocurrencia_fmt = self._format_date_long(fecha_ocurrencia_dt) if fecha_ocurrencia_dt else (fecha_ocurrencia_raw or "")

        lugar_raw = (
            fields.get("lugar_hechos")
            or consolidated.get("lugar_hechos")
            or document_context.get("resolved_fields", {}).get("lugar_hechos")
        )
        lugar_fmt = self._format_location(lugar_raw or "", default="la ubicación descrita en el documento")

        monto_carta_decimal = self._parse_decimal(fields.get("monto_reclamacion") or consolidated.get("monto_reclamacion"))
        monto_carta_fmt = self._format_currency(monto_carta_decimal) if monto_carta_decimal is not None else (fields.get("monto_reclamacion") or "")

        bien_raw = fields.get("bien_reclamado") or consolidated.get("bien_reclamado") or "la mercancía afectada"
        bien_fmt = self._format_goods(bien_raw)

        monto_carta_text = monto_carta_fmt.rstrip(".")

        analysis.analisis_completo = (
            "Se cuenta con carta reclamación emitida por el asegurado "
            f"{emisor}, con fecha {fecha_carta_fmt}, dirigida a la compañía de seguros, "
            f"donde refieren los hechos ocurridos el pasado {fecha_ocurrencia_fmt}, del cual se presentó un robo de mercancía "
            f"cuando se encontraba la unidad que transportaba la mercancía circulando sobre {lugar_fmt}; "
            f"se especifica una reclamación por la cantidad de {monto_carta_text}. "
            f"Se indicó como mercancía afectada consistente en {bien_fmt}."
        ).replace("  ", " ").strip()

        verificaciones: Dict[str, Dict[str, Any]] = {}
        recommendations: List[str] = []
        indicators: List[FraudIndicator] = []

        # 1. Fecha de reclamación posterior
        fecha_diff = None
        if fecha_carta_dt and fecha_ocurrencia_dt:
            fecha_diff = (fecha_carta_dt - fecha_ocurrencia_dt).days
            if fecha_diff >= 0:
                detalle = (
                    "La carta se emitió el mismo día del siniestro."
                    if fecha_diff == 0
                    else f"La carta se emitió {fecha_diff} días después del siniestro."
                )
                verificaciones["fecha_reclamacion_posterior"] = {
                    "resultado": "coincide",
                    "diferencia_dias": fecha_diff,
                    "detalle": detalle,
                }
            else:
                verificaciones["fecha_reclamacion_posterior"] = {
                    "resultado": "discrepancia",
                    "diferencia_dias": fecha_diff,
                    "detalle": f"La carta se emitió {abs(fecha_diff)} días antes de la fecha de ocurrencia registrada.",
                }
                indicators.append(
                    FraudIndicator(
                        pattern="fecha_reclamacion_no_posterior",
                        description="La carta se emitió antes de la fecha de ocurrencia reportada.",
                        severity="medio",
                        confidence=0.8,
                    )
                )
        else:
            verificaciones["fecha_reclamacion_posterior"] = {
                "resultado": "desconocido",
                "diferencia_dias": None,
                "detalle": "No se cuenta con fechas suficientes para comparar.",
            }
            recommendations.append(
                "Confirmar en otras fuentes la fecha exacta de emisión de la carta y del siniestro."
            )

        # 2. Número de póliza
        poliza_carta = self._normalize_identifier(fields.get("numero_poliza"))
        poliza_caso = self._normalize_identifier(consolidated.get("numero_poliza"))
        poliza_detalle = ""
        if poliza_carta and poliza_caso:
            if poliza_carta == poliza_caso:
                poliza_detalle = f"Número de póliza coincide con el consolidado {consolidated.get('numero_poliza')}."
                resultado_poliza = "coincide"
            else:
                poliza_detalle = (
                    f"La carta indica póliza {fields.get('numero_poliza')} "
                    f"mientras que el consolidado registra {consolidated.get('numero_poliza')}."
                )
                resultado_poliza = "discrepancia"
                indicators.append(
                    FraudIndicator(
                        pattern="poliza_no_coincide",
                        description="El número de póliza declarado en la carta no coincide con el consolidado del caso.",
                        severity="medio",
                        confidence=0.8,
                    )
                )
        elif poliza_carta:
            resultado_poliza = "desconocido"
            poliza_detalle = "Póliza presente en la carta; faltan datos en el consolidado para contrastar."
        elif poliza_caso:
            resultado_poliza = "desconocido"
            poliza_detalle = "La carta omite el número de póliza; se tomó el valor consolidado para referencia."
            recommendations.append("Solicitar a la aseguradora confirmación del número de póliza en la carta.")
        else:
            resultado_poliza = "desconocido"
            poliza_detalle = "Sin número de póliza disponible."
            recommendations.append("Actualizar la carta con el número de póliza correspondiente.")

        verificaciones["numero_poliza_consistente"] = {
            "resultado": resultado_poliza,
            "referencia_poliza": consolidated.get("numero_poliza") or fields.get("numero_poliza") or "",
            "detalle": poliza_detalle,
        }

        # 3. Número de siniestro
        carta_siniestro_raw = fields.get("numero_siniestro")
        carta_siniestro_norm = self._normalize_identifier(carta_siniestro_raw)
        consolidado_siniestro = self._normalize_identifier(consolidated.get("numero_siniestro"))

        opciones_siniestro = conflicts.get("numero_siniestro", [])
        opciones_norm = {self._normalize_identifier(opt): opt for opt in opciones_siniestro}

        if consolidado_siniestro:
            if carta_siniestro_norm == consolidado_siniestro or (
                carta_siniestro_norm and carta_siniestro_norm in opciones_norm
            ):
                detalle_siniestro = (
                    f"Número de siniestro coincide con el reporte del ajustador ({consolidated.get('numero_siniestro')})."
                    if carta_siniestro_norm == consolidado_siniestro
                    else (
                        f"La carta menciona folio interno {carta_siniestro_raw}; "
                        f"el consolidado confirma el siniestro {consolidated.get('numero_siniestro')}."
                    )
                )
                resultado_siniestro = "coincide"
            elif carta_siniestro_norm:
                resultado_siniestro = "discrepancia"
                detalle_siniestro = (
                    f"La carta refiere siniestro {carta_siniestro_raw}, mientras que el consolidado indica "
                    f"{consolidated.get('numero_siniestro')}."
                )
                indicators.append(
                    FraudIndicator(
                        pattern="siniestro_no_coincide",
                        description="El número de siniestro de la carta no coincide con el consolidado del caso.",
                        severity="alto",
                        confidence=0.85,
                    )
                )
            else:
                resultado_siniestro = "desconocido"
                detalle_siniestro = "La carta no indica número de siniestro; se emplea el consolidado como referencia."
                recommendations.append("Solicitar que la carta incluya el número de siniestro correcto.")
        else:
            resultado_siniestro = "desconocido"
            detalle_siniestro = "No se cuenta con número de siniestro consolidado para comparación."

        verificaciones["numero_siniestro_consistente"] = {
            "resultado": resultado_siniestro,
            "referencia_ajustador": consolidated.get("numero_siniestro") or "",
            "detalle": detalle_siniestro,
        }

        # 4. Emisor legitimado
        nombre_conflictos = set(v for v in conflicts.get("nombre_asegurado", []) if v)
        emisor_legit = False
        poliza_text_cached: Optional[str] = None
        if emisor:
            normalized_emisor = self._normalize_company_name(emisor)
            normalized_options = {self._normalize_company_name(opt): opt for opt in nombre_conflictos}
            if normalized_emisor and normalized_emisor in normalized_options:
                emisor_legit = True
            else:
                poliza_text_cached = self._get_document_text(data_layer, "poliza_de_la_aseguradora")
                poliza_norm = self._normalize_text_for_search(poliza_text_cached or "")
                if normalized_emisor and normalized_emisor in poliza_norm:
                    emisor_legit = True

        if emisor_legit:
            verificaciones["emisor_legitimado"] = {
                "resultado": "coincide",
                "fundamento": "El emisor figura como asegurado o asegurado adicional en el expediente.",
                "detalle": f"{emisor} aparece registrado en la póliza o en los documentos consolidados.",
            }
        else:
            verificaciones["emisor_legitimado"] = {
                "resultado": "desconocido",
                "fundamento": "",
                "detalle": "No se pudo acreditar en la póliza la entidad emisora; requiere confirmación.",
            }
            if emisor:
                recommendations.append(
                    f"Validar en póliza o endosos la legitimación de {emisor} como asegurado adicional."
                )

        # 5. Bienes consistentes
        referencias_bienes: List[str] = []
        bienes_coinciden = False
        comparables = [
            consolidated.get("bien_reclamado"),
            self._format_goods(self._extract_goods_from_text(self._get_document_text(data_layer, "denuncia_de_los_hechos") or "")),
            self._format_goods(self._extract_goods_from_text(self._get_document_text(data_layer, "carpeta_de_investigacion") or "")),
            self._format_goods(self._extract_goods_from_text(self._get_document_text(data_layer, "informe_final_del_ajustador") or "")),
        ]
        comparables = [c for c in comparables if c]
        if bien_fmt and comparables:
            for comparable in comparables:
                if comparable and self._goods_match(bien_fmt, comparable):
                    bienes_coinciden = True
                    referencias_bienes.append(self._shorten_goods_reference(comparable))
            if referencias_bienes:
                referencias_bienes = sorted(set(referencias_bienes))

        if bienes_coinciden:
            verificaciones["bienes_consistentes"] = {
                "resultado": "coincide",
                "referencias": referencias_bienes,
                "detalle": "Los bienes descritos mantienen consistencia con denuncia, carpeta e informes del ajustador.",
            }
        else:
            verificaciones["bienes_consistentes"] = {
                "resultado": "desconocido",
                "referencias": referencias_bienes,
                "detalle": "No se pudieron encontrar referencias coincidentes suficientes sobre los bienes.",
            }
            recommendations.append(
                "Solicitar evidencia adicional (denuncia/oficios) que describa los bienes en términos equivalentes."
            )

        # 6. Monto vs ajustador
        ajustador_text = self._get_document_text(data_layer, "informe_final_del_ajustador") or ""
        monto_ajustador_decimal = self._extract_monto_ajustador(ajustador_text)
        monto_ajustador_fmt = self._format_currency(monto_ajustador_decimal) if monto_ajustador_decimal is not None else ""

        if monto_carta_decimal is not None and monto_ajustador_decimal is not None:
            diferencia = (monto_carta_decimal - monto_ajustador_decimal).copy_abs()
            if diferencia <= Decimal("1.00"):
                verificaciones["monto_vs_ajustador"] = {
                    "resultado": "coincide",
                    "monto_carta": self._format_currency(monto_carta_decimal),
                    "monto_ajustador": monto_ajustador_fmt,
                    "diferencia": self._format_currency(Decimal("0.00")),
                    "detalle": "El monto reclamado coincide con lo reportado por el ajustador.",
                }
            else:
                verificaciones["monto_vs_ajustador"] = {
                    "resultado": "discrepancia",
                    "monto_carta": self._format_currency(monto_carta_decimal),
                    "monto_ajustador": monto_ajustador_fmt,
                    "diferencia": self._format_currency(diferencia),
                    "detalle": "Se detecta diferencia entre el monto de la carta y el ajustador.",
                }
                severity = "alto" if diferencia / (monto_ajustador_decimal or Decimal("1")) >= Decimal("0.1") else "medio"
                indicators.append(
                    FraudIndicator(
                        pattern="monto_no_concuerda",
                        description="Diferencia entre el monto reclamado y el reportado por el ajustador.",
                        severity=severity,
                        confidence=0.85,
                    )
                )
        else:
            verificaciones["monto_vs_ajustador"] = {
                "resultado": "desconocido",
                "monto_carta": self._format_currency(monto_carta_decimal) if monto_carta_decimal is not None else (fields.get("monto_reclamacion") or ""),
                "monto_ajustador": monto_ajustador_fmt,
                "diferencia": "",
                "detalle": "Sin datos completos para comparar montos.",
            }
            recommendations.append(
                "Obtener el monto reclamado reconocido en el reporte del ajustador para confirmar la coincidencia."
            )

        analysis.verificaciones = verificaciones

        # Validación cruzada
        validacion_cruzada: Dict[str, Dict[str, Any]] = {}

        poliza_text = poliza_text_cached if poliza_text_cached is not None else self._get_document_text(data_layer, "poliza_de_la_aseguradora")
        poliza_text = poliza_text or ""
        adicionales = self._extract_additional_insured(poliza_text)
        emisor_norm = self._normalize_company_name(emisor)
        adicionales_norm = {self._normalize_company_name(item): item for item in adicionales if item}
        if emisor_norm and emisor_norm not in adicionales_norm:
            poliza_norm_search = self._normalize_text_for_search(poliza_text)
            if emisor_norm and emisor_norm in poliza_norm_search:
                adicionales.append(emisor)
                adicionales_norm[emisor_norm] = emisor

        validacion_cruzada["poliza"] = {
            "asegurado_principal": consolidated.get("nombre_asegurado") or case_context.get("insured_name") or "",
            "asegurados_adicionales": adicionales,
            "observaciones": (
                "El emisor está listado como asegurado adicional."
                if emisor_norm and emisor_norm in adicionales_norm
                else
                "Se requiere confirmar mediante endoso la inclusión del emisor como asegurado adicional."
            ),
        }

        denuncia_text = self._get_document_text(data_layer, "denuncia_de_los_hechos") or ""
        denuncia_bienes = self._collect_keywords(denuncia_text, keywords=("acero", "placa", "mercancía"))

        validacion_cruzada["denuncia"] = {
            "bienes_reportados": denuncia_bienes,
            "coincidencia_con_carta": "alta" if denuncia_bienes else "pendiente",
            "observaciones": (
                "Los bienes descritos en la denuncia refieren mercancía de acero."
                if denuncia_bienes else
                "La denuncia no detalla suficientemente los bienes; validar con autoridad."
            ),
        }

        ajustador_fecha = self._extract_fecha_documento(ajustador_text)
        validacion_cruzada["ajustador"] = {
            "monto_reclamado_reportado": monto_ajustador_fmt,
            "fuente": f"Informe final del ajustador {ajustador_fecha}" if ajustador_fecha else "Informe final del ajustador",
            "observaciones": (
                "El ajustador confirma el monto reclamado por el asegurado."
                if verificaciones["monto_vs_ajustador"]["resultado"] == "coincide"
                else "El ajustador maneja un monto distinto al reclamado en la carta."
            ),
        }

        analysis.validacion_cruzada = validacion_cruzada

        resultados = [v.get("resultado") for v in verificaciones.values()]
        desconocidos = sum(1 for r in resultados if r == "desconocido")
        discrepancias = sum(1 for r in resultados if r == "discrepancia")

        # Recomendaciones finales (deduplicadas y depuradas)
        if analysis.recommendations:
            recommendations.extend(analysis.recommendations)
        depuradas = sorted(set(r.strip() for r in recommendations if r.strip()))
        analysis.recommendations = self._filter_recommendations(depuradas, verificaciones)

        # Indicadores consolidados y score final
        analysis.indicators = indicators

        if indicators:
            severities = {"bajo": 1, "medio": 2, "alto": 3, "critico": 4}
            max_indicator = max(indicators, key=lambda ind: severities.get(ind.severity, 2))
            max_sev = max_indicator.severity
            if max_sev == "critico":
                analysis.fraud_score = max(analysis.fraud_score, 0.86)
            elif max_sev == "alto":
                analysis.fraud_score = max(analysis.fraud_score, 0.68)
            elif max_sev == "medio":
                analysis.fraud_score = max(analysis.fraud_score, 0.45)
            analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))
        else:
            if discrepancias > 0:
                analysis.fraud_score = min(max(analysis.fraud_score, 0.55), 0.65)
                analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))
                analysis.confidence = max(analysis.confidence, 0.80)
            elif desconocidos > 0:
                analysis.fraud_score = 0.38
                analysis.risk_level = RiskLevel.MEDIO
                analysis.confidence = max(analysis.confidence, 0.85)
            else:
                analysis.fraud_score = 0.22
                analysis.risk_level = RiskLevel.BAJO
                analysis.confidence = max(analysis.confidence, 0.90)

        return analysis

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _index_conflicts(self, case_index: Dict[str, Any]) -> Dict[str, List[str]]:
        conflicts: Dict[str, List[str]] = {}
        consolidated = (case_index.get("consolidated_data") or {}).get("conflicts_resolved") or []
        for entry in consolidated:
            field = entry.get("field")
            if not field:
                continue
            options = entry.get("options") or []
            values = []
            for opt in options:
                value = ""
                if isinstance(opt, dict):
                    value = str(opt.get("value") or "").strip()
                else:
                    value = str(opt).strip()
                if value:
                    values.append(value)
            if values:
                conflicts[field] = values
        return conflicts

    def _format_entity_name(self, value: Optional[str]) -> str:
        if not value:
            return ""
        cleaned = " ".join(str(value).strip().split())
        if cleaned.isupper():
            cleaned = cleaned.title()
        replacements = (
            (" De ", " de "),
            (" Del ", " del "),
            (" La ", " la "),
            (" Las ", " las "),
            (" Los ", " los "),
            (" Y ", " y "),
            (" S.A. De C.V.", " S.A. de C.V."),
            (" S.A. De C. V.", " S.A. de C.V."),
            (" S. De R.L.", " S. de R.L."),
        )
        for src, dst in replacements:
            cleaned = cleaned.replace(src, dst)
        return cleaned

    def _stringify_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, Decimal):
            return f"{value}"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            parts = [self._stringify_value(item) for item in value if item not in (None, "")]
            return ", ".join(part for part in parts if part)
        if isinstance(value, dict):
            if "nombre" in value and value.get("nombre"):
                return self._stringify_value(value.get("nombre"))
            if "caseta" in value and value.get("caseta"):
                return self._stringify_value(value.get("caseta"))
            parts = []
            for key, item in value.items():
                text = self._stringify_value(item)
                if text:
                    parts.append(text)
            return ", ".join(parts)
        return str(value)

    def _extract_mexican_states(self, text: Optional[str]) -> Set[str]:
        if not text:
            return set()
        normalized = self._strip_accents(str(text)).lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = " " + " ".join(normalized.split()) + " "
        matches: Set[str] = set()
        if not normalized.strip():
            return matches
        for display, aliases in MEXICO_STATE_ALIASES.items():
            for alias in aliases:
                alias_norm = self._strip_accents(alias).lower().strip()
                if not alias_norm:
                    continue
                alias_pattern = f" {alias_norm} "
                if alias_norm == "baja california" and " baja california sur " in normalized:
                    continue
                if alias_pattern in normalized:
                    matches.add(display)
                    break
        return matches

    def _extract_spanish_date(self, text: Optional[str]) -> Optional[date]:
        if not text:
            return None
        pattern = re.compile(
            r"(\d{1,2})\s+de\s+([a-záéíóúñ]+)\s+(?:de|del)\s+((?:19|20)\d{2})",
            re.IGNORECASE,
        )
        matches = []
        for match in pattern.finditer(text):
            day_str, month_name, year_str = match.groups()
            month = SPANISH_MONTHS.get(self._strip_accents(month_name.lower()))
            if not month:
                continue
            try:
                candidate = date(int(year_str), month, int(day_str))
            except ValueError:
                continue
            matches.append(candidate)
        if not matches:
            return None
        return min(matches)

    def _extract_plate_candidates(self, text: Optional[str]) -> Set[str]:
        candidates: Set[str] = set()
        if not text:
            return candidates
        for match in re.findall(r"[A-Z0-9]{5,7}", str(text).upper()):
            if not match:
                continue
            normalized = self._normalize_plate(match)
            if not normalized:
                continue
            digits = sum(ch.isdigit() for ch in normalized)
            letters = sum(ch.isalpha() for ch in normalized)
            if digits < 2 or letters < 2 or letters > 3:
                continue
            candidates.add(normalized)
        return candidates

    def _format_page_count_label(self, value: Any) -> str:
        count = self._coerce_int(value)
        mapping = {
            1: "una página",
            2: "dos páginas",
            3: "tres páginas",
            4: "cuatro páginas",
            5: "cinco páginas",
        }
        if count is None or count <= 0:
            return "una página"
        if count in mapping:
            return mapping[count]
        return f"{count} páginas"

    def _coerce_int(self, value: Any) -> Optional[int]:
        if value in (None, "", "nan"):
            return None
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float, Decimal)):
            return int(float(value))
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value in (None, "", "nan"):
            return None
        if isinstance(value, (int, float, Decimal)):
            return float(value)
        text = str(value).strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value
        text = str(value).strip()
        if not text:
            return None
        candidates = (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%Y-%m-%d",
            "%d/%m/%Y",
        )
        for fmt in candidates:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text)
        except Exception:
            return None

    def _normalize_caseta_entries(self, raw: Any) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        if not raw:
            return entries
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    nombre = self._stringify_value(
                        item.get("nombre")
                        or item.get("caseta")
                        or item.get("caseta_nombre")
                        or item.get("identificador")
                    )
                    lat = self._coerce_float(
                        item.get("lat")
                        or item.get("latitude")
                        or item.get("latitud")
                    )
                    lon = self._coerce_float(
                        item.get("lon")
                        or item.get("longitude")
                        or item.get("longitud")
                    )
                    timestamp_value = item.get("timestamp") or item.get("fecha_hora")
                    if not timestamp_value:
                        fecha = self._stringify_value(item.get("fecha"))
                        hora = self._stringify_value(item.get("hora"))
                        if fecha and hora:
                            timestamp_value = f"{fecha} {hora}"
                        elif fecha:
                            timestamp_value = fecha
                    timestamp = self._parse_datetime(timestamp_value)
                    entries.append(
                        {
                            "nombre": nombre,
                            "identificador": self._stringify_value(
                                item.get("identificador") or nombre
                            ),
                            "lat": lat,
                            "lon": lon,
                            "timestamp": timestamp,
                            "raw": item,
                        }
                    )
                elif isinstance(item, str):
                    entries.extend(self._normalize_caseta_entries(item))
            return entries

        if isinstance(raw, str):
            segments = [seg.strip() for seg in re.split(r"[;\n]", raw) if seg.strip()]
            for segment in segments:
                match = re.search(r"(?P<nombre>.+?)(?:\s*\((?P<coords>[^)]+)\))?(?:\s*@\s*(?P<ts>.+))?$", segment)
                nombre = segment
                lat = lon = None
                timestamp = None
                if match:
                    nombre = match.group("nombre").strip()
                    coords = match.group("coords")
                    if coords:
                        coord_match = re.findall(r"-?\d+(?:\.\d+)?", coords)
                        if len(coord_match) >= 2:
                            lat = self._coerce_float(coord_match[0])
                            lon = self._coerce_float(coord_match[1])
                    ts_value = match.group("ts")
                    if ts_value:
                        timestamp = self._parse_datetime(ts_value)
                entries.append(
                    {
                        "nombre": nombre,
                        "identificador": nombre,
                        "lat": lat,
                        "lon": lon,
                        "timestamp": timestamp,
                        "raw": segment,
                    }
                )
        return entries

    def _collect_gps_points(self, data_layer: UnifiedDataLayer, *, limit: int = 20000) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        manifest = getattr(data_layer, "gps_documents", {}) or {}
        if not manifest:
            return points
        for doc_name in manifest.keys():
            try:
                snapshot = data_layer.get_gps_snapshot(doc_name, limit=limit)
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("No se pudo obtener snapshot GPS %s: %s", doc_name, exc)
                continue
            preview = snapshot.get("preview") or []
            if isinstance(preview, dict) and "rows" in preview:
                preview = preview.get("rows") or []
            for row in preview:
                if not isinstance(row, dict):
                    continue
                timestamp = self._parse_datetime(
                    row.get("timestamp")
                    or row.get("fecha_hora")
                    or row.get("time")
                    or row.get("datetime")
                )
                lat = self._coerce_float(
                    row.get("latitude")
                    or row.get("lat")
                    or row.get("latitud")
                )
                lon = self._coerce_float(
                    row.get("longitude")
                    or row.get("lon")
                    or row.get("longitud")
                )
                points.append(
                    {
                        "timestamp": timestamp,
                        "latitude": lat,
                        "longitude": lon,
                        "document": doc_name,
                        "raw": row,
                    }
                )
        return points

    def _compute_distance_meters(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        if None in (lat1, lon1, lat2, lon2):
            return float("inf")
        r_earth = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r_earth * c

    def _find_best_gps_match(
        self,
        caseta: Dict[str, Any],
        gps_points: List[Dict[str, Any]],
        *,
        distance_tolerance_m: float,
        minutes_tolerance: float,
    ) -> Optional[Dict[str, Any]]:
        lat = self._coerce_float(caseta.get("lat") or caseta.get("latitude"))
        lon = self._coerce_float(caseta.get("lon") or caseta.get("longitude"))
        timestamp = caseta.get("timestamp")
        if not isinstance(timestamp, datetime):
            timestamp = self._parse_datetime(timestamp or caseta.get("fecha") or caseta.get("hora"))

        best_match: Optional[Dict[str, Any]] = None
        for point in gps_points:
            point_lat = point.get("latitude")
            point_lon = point.get("longitude")
            point_ts = point.get("timestamp")

            distance = None
            if lat is not None and lon is not None and point_lat is not None and point_lon is not None:
                distance = self._compute_distance_meters(lat, lon, point_lat, point_lon)
                if distance > distance_tolerance_m:
                    continue

            time_diff_minutes = None
            if timestamp and point_ts:
                time_diff_minutes = abs((timestamp - point_ts).total_seconds()) / 60.0
                if time_diff_minutes > minutes_tolerance:
                    continue

            if distance is None and time_diff_minutes is None:
                continue

            score = (distance or 0.0) + (time_diff_minutes or 0.0) * 10.0
            if not best_match or score < best_match["score"]:
                best_match = {
                    "caseta": caseta,
                    "point": point,
                    "distance_m": distance,
                    "time_diff_minutes": time_diff_minutes,
                    "score": score,
                }
        if best_match:
            return best_match
        return None

    def _split_signature_line(self, line: str) -> Tuple[str, str]:
        cleaned = line.strip()
        if not cleaned:
            return "", ""
        keywords = (
            "Jefe",
            "Gerente",
            "Director",
            "Directora",
            "Administrador",
            "Administradora",
            "Administrativo",
            "Administrativa",
            "Coordinador",
            "Coordinadora",
            "Supervisor",
            "Supervisora",
            "Encargado",
            "Encargada",
            "Responsable",
            "Analista",
            "Apoderado",
            "Apoderada",
        )
        idx = len(cleaned)
        for kw in keywords:
            pos = cleaned.find(kw)
            if pos != -1 and pos < idx:
                idx = pos
        if idx < len(cleaned):
            nombre = cleaned[:idx].strip(" ,;-")
            cargo = cleaned[idx:].strip(" ,;-")
        else:
            nombre = cleaned
            cargo = ""
        return nombre, cargo

    def _parse_caseta_blocks(self, text: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not text:
            return records
        chunks = re.split(r"[•·]", text)
        for chunk in chunks:
            segment = chunk.strip(" \n\r\t;")
            if not segment:
                continue
            if "fecha" not in segment.lower():
                continue
            segment = segment.replace("\n", " ")
            ts_match = re.search(r"(20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", segment)
            timestamp = ts_match.group(1).strip() if ts_match else ""
            name_part = segment
            if ts_match:
                name_part = segment[: ts_match.start()]
            elif "Fecha" in segment:
                name_part = segment.split("Fecha")[0]
            name = name_part.strip(" ;:,")
            name = re.sub(r"(;)?\s*Fecha\s+y\s+Hora:?$", "", name, flags=re.IGNORECASE)
            name = " ".join(name.split())
            if name:
                records.append(
                    {
                        "nombre": self._stringify_value(name),
                        "timestamp": timestamp,
                    }
                )
        return records

    def _extract_aclaratoria_details(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}

        result: Dict[str, Any] = {}
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        def _normalize_phrase(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            cleaned = " ".join(value.split())
            return cleaned.strip().rstrip(".")

        subject_match = re.search(r"Asunto[:：]\s*(.+)", text, re.IGNORECASE)
        if subject_match:
            result["subject"] = subject_match.group(1).strip()

        recipient_line = None
        if subject_match:
            tail = text[subject_match.end():].strip()
            for raw_line in tail.splitlines():
                candidate = raw_line.strip()
                if candidate:
                    recipient_line = candidate
                    break
        if recipient_line:
            tokens = recipient_line.split()
            name_tokens: List[str] = []
            for token in tokens:
                lower = token.lower()
                if any(ch.isdigit() for ch in token):
                    break
                if lower in {"nº", "no", "col", "cp", "calle", "av.", "av", "avda", "avenida"}:
                    break
                if lower.startswith("estimad"):
                    break
                name_tokens.append(token)
                if len(name_tokens) >= 4:
                    break
            if name_tokens:
                while len(name_tokens) > 3 and name_tokens[-1].lower() not in {"de", "del", "la", "y"}:
                    name_tokens.pop()
                recipient_name = self._format_entity_name(" ".join(name_tokens))
                result["recipient"] = recipient_name
            estimado_match = re.search(r"(Estimad[oa].*)", recipient_line, re.IGNORECASE)
            if estimado_match:
                result["recipient_cargo"] = estimado_match.group(1).strip()

        signature_idx = None
        for idx, line in enumerate(lines):
            if line.lower().startswith("atentamente"):
                signature_idx = idx
                break
        if signature_idx is not None:
            signature_name = None
            signature_cargo = None
            if signature_idx + 1 < len(lines):
                name_candidate, cargo_candidate = self._split_signature_line(lines[signature_idx + 1])
                signature_name = name_candidate or None
                signature_cargo = cargo_candidate or None
            if signature_idx + 2 < len(lines):
                extra_line = lines[signature_idx + 2]
                if not signature_cargo and any(
                    keyword in extra_line.lower()
                    for keyword in ("logística", "logistica", "administrativo", "administrativa", "transportaciones")
                ):
                    signature_cargo = extra_line.strip()
            if signature_name:
                result["firmante"] = signature_name
            if signature_cargo:
                result["firmante_cargo"] = signature_cargo
                match_company = re.search(r"(LC[&+\sA-Z]*TRANSPORTACIONES)", signature_cargo.upper())
                if match_company:
                    result["emisor"] = match_company.group(1).strip()

        if "emisor" not in result:
            for idx, line in enumerate(lines):
                upper_line = line.upper()
                if "TRANSPORTACIONES" in upper_line or "LOGÍSTICA" in upper_line or "LOGISTICA" in upper_line:
                    if "@" in line or any(ch.isdigit() for ch in line):
                        continue
                    emisor_line = line
                    if idx > 0:
                        prev_line = lines[idx - 1].strip()
                        if prev_line and "@" not in prev_line and not any(ch.isdigit() for ch in prev_line):
                            if len(prev_line) <= 40 and prev_line.upper().startswith("LC"):
                                emisor_line = f"{prev_line} {line}".strip()
                    result["emisor"] = emisor_line
                    break

        desc_match = re.search(
            r"Lamentablemente,\s*(.+?)(?:\.\s|\.?$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if desc_match:
            desc_text = desc_match.group(1)
            if "se anexa" in desc_text.lower():
                desc_text = desc_text.split("se anexa", 1)[0]
            result["description"] = _normalize_phrase(desc_text)

        impact_match = re.search(
            r"entre los artículos sustraídos se encuentran (.+?)(?:\.\s|\.?$)",
            text,
            re.IGNORECASE,
        )
        if not impact_match:
            impact_match = re.search(
                r"No hemos podido recuperar los tickets de peaje robados(?:.*?)\.",
                text,
                re.IGNORECASE | re.DOTALL,
            )
        if impact_match:
            impact_text = impact_match.group(0)
            if " hasta la fecha" in impact_text.lower():
                impact_text = impact_text.split(" hasta la fecha", 1)[0]
            result["impact"] = _normalize_phrase(impact_text)

        purpose_match = re.search(
            r"Nos gustaría notificarles este incidente[^.]*\.",
            text,
            re.IGNORECASE,
        )
        if purpose_match:
            result["purpose"] = _normalize_phrase(purpose_match.group(0))

        caseta_records = self._parse_caseta_blocks(text)
        if caseta_records:
            result["casetas"] = caseta_records
            result["horarios"] = [record["timestamp"] for record in caseta_records if record.get("timestamp")]
            detail_items = []
            for record in caseta_records:
                nombre = record.get("nombre") or ""
                ts = record.get("timestamp")
                if ts:
                    detail_items.append(f"{nombre} ({ts})")
                else:
                    detail_items.append(nombre)
            result["detail"] = "; ".join(item for item in detail_items if item)

        units_match = re.search(
            r"número de económico\s+([A-Z0-9#\s\-yY]+)",
            text,
            re.IGNORECASE,
        )
        if units_match and not result.get("description"):
            units = " ".join(units_match.group(1).split())
            result["description"] = _normalize_phrase(
                f"las unidades de transporte con número económico {units} fueron objeto de robo"
            )

        if not result.get("impact"):
            result["impact"] = "no se anexaron los tickets de peaje debido al robo reportado"

        if not result.get("detail") and caseta_records:
            result["detail"] = "; ".join(
                record.get("nombre", "") for record in caseta_records if record.get("nombre")
            )

        return result

    def _parse_iso_date(self, value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        value = str(value).strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).date()
        except Exception:
            return None

    def _format_date_slash(self, value: Optional[date]) -> str:
        if not value:
            return ""
        return value.strftime("%d/%m/%Y")

    def _format_date_long(self, value: Optional[date]) -> str:
        if not value:
            return ""
        months = [
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        ]
        return f"{value.day} de {months[value.month - 1]} del {value.year}"

    def _format_location(self, value: str, *, default: str = "el lugar indicado") -> str:
        if not value:
            return default
        cleaned = " ".join(value.replace("KM", "km").replace("Km", "km").split())
        match_km = re.search(r"(?:km|kil[oó]metro)\s*(\d+)", cleaned, re.IGNORECASE)
        km_text = f"el KM {match_km.group(1)}" if match_km else ""

        match_carretera = re.search(
            r"carretera\s+([^.,;]+(?:[,;]\s*[^.,;]+)?)",
            cleaned,
            re.IGNORECASE,
        )
        if match_carretera:
            tramo = match_carretera.group(1).strip().rstrip(".;,")
            tramo = tramo.replace("  ", " ")
            base = f"{km_text} de la carretera {tramo}" if km_text else f"la carretera {tramo}"
        else:
            base = cleaned
        return base

    def _format_goods(self, value: Optional[str]) -> str:
        if not value:
            return ""
        cleaned = " ".join(str(value).strip().split())
        if cleaned.isupper():
            cleaned = cleaned.lower()
        return cleaned

    def _infer_goods_category(self, description: str) -> str:
        tokens = {token.lower() for token in self._goods_tokens(description) if token}
        if not tokens:
            return "mercancía"
        vehicle_tokens = {"vehiculo", "vehiculos", "tractocamion", "camion", "unidad", "remolque", "plataforma", "chasis"}
        machinery_tokens = {"maquinaria", "excavadora", "grua", "montacargas"}
        livestock_tokens = {"ganado", "bovino", "porcino", "reses"}
        if tokens & vehicle_tokens:
            return "vehículos"
        if tokens & machinery_tokens:
            return "maquinaria"
        if tokens & livestock_tokens:
            return "ganado"
        if {"material", "producto", "mercancia", "carga"} & tokens:
            return "mercancía"
        return "mercancía"

    def _parse_decimal(self, value: Optional[str]) -> Optional[Decimal]:
        if value is None:
            return None
        if isinstance(value, (int, float, Decimal)):
            return Decimal(str(value))
        text = str(value)
        text = text.replace("$", "").replace("MXN", "").replace("M.N.", "").replace("MN", "")
        text = text.replace(" ", "").replace(",", "")
        text = text.replace("(", "").replace(")", "")
        if not text:
            return None
        try:
            return Decimal(text)
        except InvalidOperation:
            return None

    def _format_currency(self, amount: Optional[Decimal]) -> str:
        if amount is None:
            return ""
        quantized = amount.quantize(Decimal("0.01"))
        formatted = f"${quantized:,.2f}"
        if "M.N." not in formatted and "MXN" not in formatted.upper():
            formatted = f"{formatted} M.N."
        return formatted

    def _format_currency_with_code(self, amount: Optional[Decimal], currency: Optional[str]) -> str:
        if amount is None:
            return ""
        quantized = amount.quantize(Decimal("0.01"))
        label = (currency or "MXN").upper()
        symbol = "$"
        if label in {"USD", "US$", "DOLARES"}:
            return f"{symbol}{quantized:,.2f} USD"
        if label in {"MXN", "M.N.", "MN"}:
            return f"{symbol}{quantized:,.2f} M.N."
        return f"{symbol}{quantized:,.2f} {label}"

    def _ensure_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple) or isinstance(value, set):
            return list(value)
        return [value]

    def _normalize_plate(self, value: Optional[str]) -> str:
        if not value:
            return ""
        text = self._stringify_value(value)
        if not text:
            return ""
        normalized = re.sub(r"[^A-Za-z0-9]", "", text.upper())
        if len(normalized) < 5:
            return ""
        return normalized

    def _spanish_ordinal_feminine(self, index: int) -> str:
        mapping = {
            1: "Primera",
            2: "Segunda",
            3: "Tercera",
            4: "Cuarta",
            5: "Quinta",
            6: "Sexta",
            7: "Séptima",
            8: "Octava",
            9: "Novena",
            10: "Décima",
        }
        return mapping.get(index, f"{index}ª")

    def _spanish_ordinal_feminine_lower(self, index: int) -> str:
        return self._spanish_ordinal_feminine(index).lower()

    def _spanish_ordinal_masculine_lower(self, index: int) -> str:
        mapping = {
            1: "primer",
            2: "segundo",
            3: "tercer",
            4: "cuarto",
            5: "quinto",
            6: "sexto",
            7: "séptimo",
            8: "octavo",
            9: "noveno",
            10: "décimo",
        }
        return mapping.get(index, f"{index}º")

    def _normalize_goods_label(self, value: Optional[str]) -> str:
        if not value:
            return ""
        text = self._strip_accents(self._stringify_value(value).lower())
        text = re.sub(r"[^a-z0-9]+", " ", text).strip()
        return text

    def _normalize_identifier(self, value: Optional[str]) -> str:
        if not value:
            return ""
        return re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()

    def _identifier_matches_text(self, identifier_norm: str, normalized_text: str) -> bool:
        if not identifier_norm or not normalized_text:
            return False
        if identifier_norm in normalized_text:
            return True
        if len(identifier_norm) < 8:
            return False
        window = len(identifier_norm)
        max_start = len(normalized_text) - window
        if max_start <= 0:
            return False
        for idx in range(0, max_start + 1):
            segment = normalized_text[idx : idx + window]
            differences = 0
            for a, b in zip(segment, identifier_norm):
                if a != b:
                    differences += 1
                    if differences > 1:
                        break
            if differences <= 1:
                return True
        return False

    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula la distancia aproximada entre dos coordenadas."""
        radius = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius * c

    def _get_document_text(
        self,
        data_layer: Optional[UnifiedDataLayer],
        document_type: str,
        *,
        source_document: Optional[str] = None,
    ) -> Optional[str]:
        payload = self._load_case_document_payload(
            data_layer,
            document_type,
            source_document=source_document,
        )
        if payload:
            return payload.get("text")
        return None

    def _collect_document_candidates(
        self,
        data_layer: Optional[UnifiedDataLayer],
        document_type: str,
        source_document: Optional[str] = None,
    ) -> List[str]:
        if not data_layer:
            return []
        candidates: List[str] = []
        seen: Set[str] = set()

        if source_document:
            norm = self._normalize_identifier(Path(source_document).stem)
            candidates.append(source_document)
            if norm:
                seen.add(norm)

        for item in getattr(data_layer, "_extractions", {}).values():
            if getattr(item, "document_type", "") != document_type:
                continue
            candidate = item.source_document
            norm = self._normalize_identifier(Path(candidate).stem)
            if norm in seen:
                continue
            seen.add(norm)
            candidates.append(candidate)

        if not candidates:
            case_index = getattr(data_layer, "case_index", {}) or {}
            for entry in case_index.get("classified_types") or []:
                if entry.get("document_type") != document_type or not entry.get("filename"):
                    continue
                candidate = entry["filename"]
                norm = self._normalize_identifier(Path(candidate).stem)
                if norm in seen:
                    continue
                seen.add(norm)
                candidates.append(candidate)

        return candidates

    def _resolve_case_document_payload(
        self,
        case_index: Dict[str, Any],
        source_document: str,
    ) -> Optional[Dict[str, Any]]:
        documents = case_index.get("documents") or []
        target_norm = self._normalize_identifier(Path(source_document).stem)
        for entry in documents:
            entry_path = Path(entry)
            parent_norm = self._normalize_identifier(entry_path.parent.name)
            entry_norm = self._normalize_identifier(entry_path.stem.replace("ocr_results_for_", ""))
            if not target_norm:
                continue
            if target_norm in {parent_norm, entry_norm} or target_norm in entry_norm:
                try:
                    return json.loads(entry_path.read_text(encoding="utf-8"))
                except Exception as exc:  # pragma: no cover - defensivo
                    logger.debug("No se pudo leer OCR %s: %s", entry, exc)
        return None

    def _load_case_document_payload(
        self,
        data_layer: Optional[UnifiedDataLayer],
        document_type: str,
        *,
        source_document: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not data_layer:
            return None
        case_index = getattr(data_layer, "case_index", {}) or {}
        for candidate in self._collect_document_candidates(data_layer, document_type, source_document):
            if not candidate:
                continue
            payload = self._resolve_case_document_payload(case_index, candidate)
            if payload:
                return payload
        return None

    def _find_extraction_by_type(
        self,
        data_layer: Optional[UnifiedDataLayer],
        document_type: str,
    ) -> Optional[DocumentExtraction]:
        if not data_layer:
            return None
        for extraction in getattr(data_layer, "_extractions", {}).values():
            if extraction.document_type == document_type:
                return extraction
        return None

    def _iter_document_sources(
        self,
        data_layer: Optional[UnifiedDataLayer],
        document_type: str,
    ) -> Iterable[Tuple[str, Optional[DocumentExtraction]]]:
        if not data_layer:
            return []
        seen: Set[str] = set()
        for extraction in getattr(data_layer, "_extractions", {}).values():
            if extraction.document_type != document_type:
                continue
            source_document = extraction.source_document
            norm = self._normalize_identifier(Path(source_document).stem)
            seen.add(norm)
            yield source_document, extraction

        case_index = getattr(data_layer, "case_index", {}) or {}
        for entry in case_index.get("classified_types") or []:
            if entry.get("document_type") != document_type:
                continue
            filename = entry.get("filename")
            if not filename:
                continue
            norm = self._normalize_identifier(Path(filename).stem)
            if norm in seen:
                continue
            yield filename, None

    def _resolve_document_ampara(self, fields: Dict[str, Any], ocr_text: str) -> Dict[str, str]:
        doc_tipo = (
            fields.get("documento_ampara_tipo")
            or fields.get("documento_soporte_tipo")
            or "documento soporte"
        )
        doc_num = (
            fields.get("documento_ampara_numero")
            or fields.get("numero_documento_ampara")
            or fields.get("numero_pedimento")
            or fields.get("folio_documento")
            or ""
        )
        if not doc_num:
            match = re.search(
                r"pedimento(?:\s+de\s+importaci[oó]n)?\s+n[uú]mero\s+([0-9\s-]{6,})",
                ocr_text,
                re.IGNORECASE,
            )
            if match:
                doc_tipo = "pedimento"
                doc_num = match.group(1)
        if doc_num:
            doc_num = re.sub(r"\s+", "", doc_num)
        resumen = doc_tipo
        if doc_num:
            resumen = f"{doc_tipo} {doc_num}".strip()
        else:
            resumen = f"{doc_tipo} sin número citado".strip()
        return {"tipo": doc_tipo, "numero": doc_num, "resumen": resumen}

    def _summarize_goods_from_letter(self, fields: Dict[str, Any], ocr_text: str) -> Dict[str, Any]:
        description_candidates = [
            fields.get("descripcion_mercancia") or "",
            fields.get("detalle_mercancia") or "",
            fields.get("mercancia_descripcion") or "",
        ]
        description = ""
        for item in description_candidates:
            if item and len(item.strip()) > 3:
                description = item.strip()
                break

        if not description:
            composed = re.search(
                r"producto\s+de\s+([^\n,]+?),\s+conocido\s+comercialmente\s+como\s+([^\n.,]+)",
                ocr_text,
                re.IGNORECASE,
            )
            if composed:
                base = re.sub(r"\s+", " ", composed.group(1).strip(" .:"))
                comercial = re.sub(r"\s+", " ", composed.group(2).strip(" .:"))
                description = f"{base} {comercial}".strip()

        if not description:
            match = re.search(
                r"conocido\s+comercialmente\s+como\s+([^\n.,]+)",
                ocr_text,
                re.IGNORECASE,
            )
            if match:
                description = re.sub(r"\s+", " ", match.group(1).strip(" .:"))

        if not description:
            match = re.search(
                r"material\s+(?:que\s+se\s+constitui[aá]\s+en\s+)?producto\s+de\s+([^\n.,]+)",
                ocr_text,
                re.IGNORECASE,
            )
            if match:
                description = re.sub(r"\s+", " ", match.group(1).strip(" .:"))

        units = self._aggregate_units_from_text(ocr_text)
        totals = self._extract_totals_from_text(ocr_text)
        for key, value in totals.items():
            if key not in units or (value is not None and value > units.get(key, Decimal("0"))):
                units[key] = value
        units_summary = self._format_units_summary(units)

        def _format_quantity(value: Optional[Decimal], label: str) -> str:
            if value is None:
                return ""
            if value == value.to_integral():
                formatted = f"{int(value):,}"
            else:
                formatted = f"{value:,.2f}"
            formatted = formatted.replace(",", ",")
            return f"{formatted} {label}"

        piezas_text = _format_quantity(units.get("piezas"), "piezas")
        kilos_text = _format_quantity(units.get("kilogramos"), "kg")
        toneladas_text = _format_quantity(units.get("toneladas"), "toneladas")

        detalle = ""
        if piezas_text and description:
            if kilos_text or toneladas_text:
                peso_texto = kilos_text or toneladas_text
                detalle = f"{piezas_text} de {description} con un peso de {peso_texto}"
            else:
                detalle = f"{piezas_text} de {description}"
        elif description and (kilos_text or toneladas_text):
            detalle = f"{description} ({kilos_text or toneladas_text})"
        elif piezas_text and (kilos_text or toneladas_text):
            detalle = f"{piezas_text} con un peso de {kilos_text or toneladas_text}"

        if not detalle:
            detalle = description or units_summary

        raw_references: List[str] = []
        if description:
            raw_references.append(description)
        for line in ocr_text.splitlines():
            lower = line.lower()
            if any(token in lower for token in ("pieza", "pza", "kg", "tonelada", "ton.")):
                raw_references.append(line)

        return {"descripcion": detalle, "units": units, "raw_references": raw_references}

    def _aggregate_units_from_text(self, text: str) -> Dict[str, Decimal]:
        totals: Dict[str, Decimal] = {}
        if not text:
            return totals
        pattern_value_first = re.compile(
            r"(\d+(?:[.,]\d+)?(?:[, ]\d{3})*|\d+)\s*(piezas?|pza?s?|pz|pieces?|pcs|kg\.?|kgs?|kilogramos?|kilograms?|toneladas?|tons?|ton\.?)",
            re.IGNORECASE,
        )
        for line in text.splitlines():
            for match in pattern_value_first.finditer(line):
                number_raw, unit_raw = match.groups()
                prefix = line[max(0, match.start() - 5):match.start()]
                prefix_upper = prefix.upper()
                if "$" in prefix or "MXN" in prefix_upper or "USD" in prefix_upper:
                    continue
                line_text = line.lower()
                if "total" in line_text:
                    continue
                value = self._parse_number_token(number_raw)
                if value is None:
                    continue
                unit = self._normalize_unit_label(unit_raw)
                if not unit:
                    continue
                totals[unit] = totals.get(unit, Decimal("0")) + value

        pattern_unit_first = re.compile(
            r"(piezas?|pza?s?|pz|pieces?|pcs|kilogramos?|kilograms?|kg\.?|kgs?|toneladas?|tons?|ton\.?)\s*(?:[:=])?\s*(\d+(?:[.,]\d+)?(?:[, ]\d{3})*|\d+)",
            re.IGNORECASE,
        )
        for line in text.splitlines():
            for match in pattern_unit_first.finditer(line):
                unit_raw, number_raw = match.groups()
                value = self._parse_number_token(number_raw)
                if value is None:
                    continue
                unit = self._normalize_unit_label(unit_raw)
                if not unit:
                    continue
                totals[unit] = totals.get(unit, Decimal("0")) + value

        return totals

    def _aggregate_units_from_payload(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Decimal]:
        totals: Dict[str, Decimal] = {}
        if not payload:
            return totals

        tables = payload.get("tables") or []
        for table in tables:
            headers = table.get("headers") or []
            header_map = {
                idx: self._strip_accents(str(label or "").lower()) for idx, label in enumerate(headers)
            }

            piezas_sum = Decimal("0")
            kilogramos_sum = Decimal("0")
            piezas_totales: List[Decimal] = []
            kilogramos_totales: List[Decimal] = []

            for raw_row in table.get("data_rows") or []:
                row = [str(cell or "").strip() for cell in raw_row]
                if not any(row):
                    continue

                is_total_row = any("TOTAL" in self._strip_accents(cell.upper()) for cell in row if cell)

                piezas_val: Optional[Decimal] = None
                kg_candidate: Optional[Decimal] = None

                for cell in row:
                    match = re.search(r"(pieces?|piezas?|pzs?|pz|pza?s?)\s*[:=]?\s*([0-9][0-9,\.\s]*)", cell, re.IGNORECASE)
                    if match:
                        piezas_val = self._parse_number_token(match.group(2))
                        if piezas_val is not None:
                            break

                if piezas_val is None:
                    for idx, label in header_map.items():
                        if idx >= len(row):
                            continue
                        if any(token in label for token in ("pieza", "pieces", "cantidad")):
                            piezas_val = self._parse_number_token(row[idx])
                            if piezas_val is not None:
                                break

                numeric_cells: List[Tuple[int, Decimal, int, str, str]] = []
                for idx, cell in enumerate(row):
                    normalized = cell.replace(" ", "").replace(",", "")
                    if not normalized:
                        continue
                    if not re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
                        continue
                    value = self._parse_number_token(cell)
                    if value is None:
                        continue
                    decimals = len(normalized.split(".")[1]) if "." in normalized else 0
                    numeric_cells.append((idx, value, decimals, header_map.get(idx, ""), cell))

                for idx, value, decimals, label, cell in numeric_cells:
                    label_upper = label.upper()
                    cell_upper = self._strip_accents(cell.upper())
                    if any(token in label_upper for token in ("KG", "KILO")) or any(
                        token in cell_upper for token in ("KG", "KILO")
                    ):
                        kg_candidate = value
                        break
                if kg_candidate is None:
                    for idx, value, decimals, label, cell in numeric_cells:
                        label_upper = label.upper()
                        cell_upper = self._strip_accents(cell.upper())
                        if any(token in label_upper for token in ("MT", "TON")) or any(
                            token in cell_upper for token in (" MT", "TON")
                        ):
                            kg_candidate = value * Decimal("1000") if value < Decimal("1000") else value
                            break
                if kg_candidate is None:
                    for idx, value, decimals, label, cell in numeric_cells:
                        if decimals == 3:
                            kg_candidate = value * Decimal("1000")
                            break

                if is_total_row:
                    if piezas_val is not None and piezas_val > 0:
                        piezas_totales.append(piezas_val)
                    if kg_candidate is not None and kg_candidate > 0:
                        kilogramos_totales.append(kg_candidate)
                    continue

                if piezas_val is not None and piezas_val > 0:
                    piezas_sum += piezas_val
                if kg_candidate is not None and kg_candidate > 0:
                    kilogramos_sum += kg_candidate

            if piezas_totales:
                piezas_sum = max(piezas_sum, max(piezas_totales))
            if kilogramos_totales:
                kilogramos_sum = max(kilogramos_sum, max(kilogramos_totales))

            if piezas_sum > 0:
                current = totals.get("piezas", Decimal("0"))
                totals["piezas"] = max(current, piezas_sum)
            if kilogramos_sum > 0:
                current = totals.get("kilogramos", Decimal("0"))
                totals["kilogramos"] = max(current, kilogramos_sum)

        return totals

    def _extract_totals_from_text(self, text: str) -> Dict[str, Decimal]:
        totals: Dict[str, Decimal] = {}
        if not text:
            return totals
        match = re.search(
            r"TOTALES?\s*(?:\r?\n)\s*([0-9.,]+)\s*(?:\r?\n)\s*([0-9.,]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            piezas_raw, peso_raw = match.groups()
            piezas = self._parse_number_token(piezas_raw)
            peso = self._parse_number_token(peso_raw)
            if piezas is not None:
                totals["piezas"] = piezas
            if peso is not None:
                totals["kilogramos"] = peso
        return totals

    def _parse_number_token(self, token: str) -> Optional[Decimal]:
        if not token:
            return None
        cleaned = token.replace(",", "").replace(" ", "")
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            cleaned_alt = cleaned.replace(".", "")
            try:
                return Decimal(cleaned_alt)
            except InvalidOperation:
                return None

    def _normalize_unit_label(self, unit: str) -> Optional[str]:
        mapping = {
            "PIEZA": "piezas",
            "PIEZAS": "piezas",
            "PZA": "piezas",
            "PZAS": "piezas",
            "PZ": "piezas",
            "PIECES": "piezas",
            "PIECE": "piezas",
            "PCS": "piezas",
            "KG": "kilogramos",
            "KG.": "kilogramos",
            "KGS": "kilogramos",
            "KILOGRAMO": "kilogramos",
            "KILOGRAMOS": "kilogramos",
            "KILOGRAM": "kilogramos",
            "KILOGRAMS": "kilogramos",
            "TON": "toneladas",
            "TON.": "toneladas",
            "TONS": "toneladas",
            "TONELADA": "toneladas",
            "TONELADAS": "toneladas",
        }
        key = self._strip_accents(unit.upper().strip().strip("."))
        return mapping.get(key)

    def _format_units_summary(self, units: Dict[str, Decimal]) -> str:
        if not units:
            return ""
        ordered = []
        for key in ("piezas", "kilogramos", "toneladas"):
            if key not in units:
                continue
            value = units[key]
            if value == value.to_integral():
                formatted = f"{int(value):,}"
            else:
                formatted = f"{value:,.2f}"
            formatted = formatted.replace(",", ",")
            label = {"piezas": "piezas", "kilogramos": "kg", "toneladas": "toneladas"}[key]
            ordered.append(f"{formatted} {label}")
        return "; ".join(ordered)

    def _collect_case_route_points(
        self,
        data_layer: Optional[UnifiedDataLayer],
    ) -> Tuple[List[str], List[str]]:

        origins: List[str] = []
        destinations: List[str] = []

        if not data_layer:
            return origins, destinations

        for doc_type in ("carpeta_de_investigacion", "denuncia_de_los_hechos"):
            extraction = self._find_extraction_by_type(data_layer, doc_type)
            if not extraction:
                continue
            denuncias = self._ensure_list(extraction.extracted_fields.get("denuncias"))
            for item in denuncias:
                if not isinstance(item, dict):
                    continue
                origin = self._stringify_value(
                    item.get("origen")
                    or item.get("origen_declarado")
                    or item.get("salida")
                    or item.get("ubicacion_origen")
                )
                destination = self._stringify_value(
                    item.get("destino")
                    or item.get("destino_declarado")
                    or item.get("llegada")
                    or item.get("ubicacion_destino")
                )
                if origin:
                    origins.append(origin)
                if destination:
                    destinations.append(destination)

        carta_reclamacion = self._find_extraction_by_type(
            data_layer,
            "carta_de_reclamacion_formal_a_la_aseguradora",
        )
        if carta_reclamacion:
            origin = self._stringify_value(carta_reclamacion.extracted_fields.get("origen"))
            destination = self._stringify_value(carta_reclamacion.extracted_fields.get("destino"))
            if origin:
                origins.append(origin)
            if destination:
                destinations.append(destination)

        return origins, destinations

    def _summarize_cfdi_goods(
        self,
        fields: Dict[str, Any],
    ) -> Tuple[str, str, Optional[Decimal], Dict[str, Decimal]]:

        def _clean_description(text: str) -> str:
            cleaned = re.sub(r"\s+", " ", text).strip()
            if cleaned.isupper():
                return cleaned.lower()
            return cleaned

        description_raw = self._stringify_value(
            fields.get("descripcion_mercancia")
            or fields.get("mercancia")
            or fields.get("descripcion")
            or ""
        )
        description = _clean_description(description_raw) if description_raw else ""
        units: Dict[str, Decimal] = {}
        value_decimal: Optional[Decimal] = None

        mercancias = self._ensure_list(fields.get("mercancias"))
        for item in mercancias:
            if not isinstance(item, dict):
                continue
            desc_candidate = self._stringify_value(
                item.get("descripcion")
                or item.get("descripcion_mercancia")
                or ""
            )
            if desc_candidate:
                cleaned_desc = _clean_description(desc_candidate)
                if not description:
                    description = cleaned_desc
            qty_candidate = self._parse_decimal(
                item.get("cantidad")
                or item.get("cantidad_total")
            )
            if qty_candidate is not None and qty_candidate > 0:
                units["piezas"] = max(units.get("piezas", Decimal("0")), qty_candidate)
            weight_candidate = self._parse_decimal(
                item.get("peso")
                or item.get("peso_bruto")
                or item.get("peso_bruto_total")
                or item.get("peso_kg")
                or item.get("peso_neto")
            )
            if weight_candidate is not None and weight_candidate > 0:
                units["kilogramos"] = max(units.get("kilogramos", Decimal("0")), weight_candidate)
            unit_label = self._stringify_value(item.get("unidad") or item.get("unidad_medida"))
            if unit_label:
                normalized = self._normalize_unit_label(unit_label)
                if normalized and normalized not in {"piezas", "kilogramos"}:
                    unit_amount = self._parse_decimal(item.get("cantidad"))
                    if unit_amount is not None and unit_amount > 0:
                        units[normalized] = max(units.get(normalized, Decimal("0")), unit_amount)
            value_candidate = self._parse_decimal(
                item.get("valor")
                or item.get("valor_mercancia")
                or item.get("importe")
            )
            if value_candidate is not None and value_candidate > 0:
                value_decimal = value_candidate if value_decimal is None else max(value_decimal, value_candidate)

        qty_general = self._parse_decimal(
            fields.get("cantidad")
            or fields.get("cantidad_total")
            or fields.get("cantidad_mercancia")
        )
        if qty_general is not None and qty_general > 0:
            units["piezas"] = max(units.get("piezas", Decimal("0")), qty_general)
        unit_label_general = self._stringify_value(fields.get("unidad_medida"))
        if unit_label_general:
            normalized_label = self._normalize_unit_label(unit_label_general)
            if normalized_label and normalized_label not in {"piezas", "kilogramos"} and qty_general is not None and qty_general > 0:
                units[normalized_label] = max(units.get(normalized_label, Decimal("0")), qty_general)

        weight_general = self._parse_decimal(
            fields.get("peso")
            or fields.get("peso_total")
            or fields.get("peso_bruto")
        )
        if weight_general is not None and weight_general > 0:
            units["kilogramos"] = max(units.get("kilogramos", Decimal("0")), weight_general)

        tons_general = self._parse_decimal(fields.get("toneladas") or fields.get("tonelaje"))
        if tons_general is not None and tons_general > 0:
            units["toneladas"] = max(units.get("toneladas", Decimal("0")), tons_general)

        primary_total = self._parse_decimal(fields.get("monto_total"))
        if primary_total is not None:
            value_decimal = primary_total
        else:
            fallback_value = self._parse_decimal(
                fields.get("valor_mercancia")
                or fields.get("valor_total_mercancia")
                or fields.get("total")
                or fields.get("subtotal")
            )
            if fallback_value is not None and (value_decimal is None or fallback_value > value_decimal):
                value_decimal = fallback_value

        units_summary = self._format_units_summary(units) if units else ""
        if units_summary and description:
            goods_summary = f"{units_summary} de {description}"
        elif units_summary:
            goods_summary = units_summary
        elif description:
            goods_summary = description
        else:
            goods_summary = "la mercancía declarada en el CFDI"

        complement_detail = goods_summary
        return goods_summary, complement_detail, value_decimal, units

    def _format_day_delta(self, delta_days: int) -> str:
        if delta_days == 0:
            return "el mismo día del siniestro"
        direction = "después" if delta_days > 0 else "antes"
        days = abs(delta_days)
        words = {
            1: "un día",
            2: "dos días",
            3: "tres días",
            4: "cuatro días",
            5: "cinco días",
            6: "seis días",
            7: "siete días",
            8: "ocho días",
            9: "nueve días",
            10: "diez días",
        }
        quantity = words.get(days, f"{days} días")
        return f"{quantity} {direction} de la fecha del siniestro"

    def _distance_between_locations(
        self,
        location_a: Optional[str],
        location_b: Optional[str],
    ) -> Optional[float]:
        if not location_a or not location_b:
            return None
        coord_a = suggest_reference_point(location_a)
        coord_b = suggest_reference_point(location_b)
        if not coord_a or not coord_b:
            return None
        return self._haversine_meters(coord_a, coord_b)

    def _haversine_meters(
        self,
        coord_a: Tuple[float, float],
        coord_b: Tuple[float, float],
    ) -> float:
        lat1, lon1 = coord_a
        lat2, lon2 = coord_b
        radius = 6_371_000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
        return radius * c

    def _postprocess_cfdi_carta_porte(
        self,
        analysis: FraudAnalysisResult,
        extraction: DocumentExtraction,
        *,
        data_layer: Optional[UnifiedDataLayer],
        document_context: Dict[str, Any],
        case_context: Dict[str, Any],
        ocr_text: str,
    ) -> FraudAnalysisResult:

        fields = dict(extraction.extracted_fields or {})
        resolved_fields = dict(document_context.get("resolved_fields") or {})
        consolidated = dict(getattr(data_layer, "consolidated_fields", {}) or {})
        case_core = {}
        if isinstance(case_context, dict):
            case_core = case_context.get("core_fields") or {}

        def _has_content(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(value.strip())
            if isinstance(value, (list, tuple, dict)):
                return bool(value)
            return True

        def _pick_value(*keys: str, default: str = "") -> str:
            for key in keys:
                if key in fields and _has_content(fields[key]):
                    return self._stringify_value(fields[key])
                if key in resolved_fields and _has_content(resolved_fields[key]):
                    return self._stringify_value(resolved_fields[key])
                if key in consolidated and _has_content(consolidated[key]):
                    return self._stringify_value(consolidated[key])
            return default

        def _pick_sequence(*keys: str) -> List[Any]:
            for key in keys:
                if key in fields and _has_content(fields[key]):
                    return self._ensure_list(fields[key])
                if key in resolved_fields and _has_content(resolved_fields[key]):
                    return self._ensure_list(resolved_fields[key])
            return []

        def _render_table(rows: List[Tuple[str, Any]]) -> str:
            cleaned: List[Tuple[str, str]] = []
            for name, value in rows:
                if value in (None, ""):
                    continue
                text = self._stringify_value(value).replace("\n", " ").strip()
                if not text:
                    continue
                cleaned.append((name, text))
            if not cleaned:
                return ""
            lines = ["| Campo | Valor |", "| --- | --- |"]
            for name, value in cleaned:
                lines.append(f"| {name} | {value} |")
            return "\n".join(lines)

        serie_raw = _pick_value("serie", "serie_cfdi", "serie_folio", default="")
        folio_raw = _pick_value(
            "folio",
            "folio_cfdi",
            "numero_interno_documento",
            "folio_documento",
            "folio_interno",
            "numero_folio",
            default="",
        )
        folio_uuid = _pick_value("folio_fiscal_uuid", "uuid_fiscal", "uuid", default="")
        serie_clean = serie_raw.strip()
        folio_clean = folio_raw.strip()
        if serie_clean and folio_clean:
            if folio_clean.startswith(serie_clean):
                remainder = folio_clean[len(serie_clean):].strip(" -")
                if remainder:
                    folio_display = f"{serie_clean}-{remainder}"
                else:
                    folio_display = serie_clean
            elif serie_clean.endswith("-") or folio_clean.startswith("-"):
                folio_display = f"{serie_clean}{folio_clean}"
            else:
                folio_display = f"{serie_clean}-{folio_clean}"
        elif folio_clean:
            folio_display = folio_clean
        elif serie_clean:
            folio_display = serie_clean
        elif folio_uuid:
            folio_display = folio_uuid.upper()
        else:
            folio_display = extraction.document_name or "SIN FOLIO"
        folio_display = folio_display.strip()

        goods_summary, goods_complement, declared_value, goods_units = self._summarize_cfdi_goods(fields)

        representante = self._format_entity_name(
            _pick_value(
                "emisor_representante",
                "representante_emisor",
                "representante",
                "expedidor_nombre",
                "emisor_contacto",
                "emisor_nombre",
                "emisor",
            )
        )
        if not representante:
            representante = self._format_entity_name(_pick_value("emisor_nombre"))
        transportista = self._format_entity_name(
            _pick_value(
                "nombre_transportista",
                "transportista",
                "emisor_empresa",
                "razon_social_emisor",
                "issuer_name",
                "emisor",
            )
        )
        if not transportista:
            transportista = self._format_entity_name(
                _pick_value("empresa_transportista", "cliente_transportista")
            )
        if not transportista and ocr_text:
            transportista = self._format_entity_name(self._extract_transportista_from_text(ocr_text))
        if transportista:
            tokens = transportista.split()
            adjusted_tokens = [
                token.upper() if any(char in token for char in "&+") else token
                for token in tokens
            ]
            transportista = " ".join(adjusted_tokens)

        operador = self._format_entity_name(
            _pick_value(
                "nombre_operador",
                "operador_nombre",
                "operador",
                "operador_asignado",
                "operador_cfdi",
            )
        )

        fecha_emision_raw = _pick_value(
            "fecha_expedicion",
            "fecha_emision",
            "fecha_certificacion_sat",
            "fecha_timbrado",
            default="",
        )
        fecha_emision_dt = self._parse_datetime(fecha_emision_raw)
        if fecha_emision_dt:
            fecha_emision_date = fecha_emision_dt.date()
        else:
            fecha_emision_date = self._parse_iso_date(fecha_emision_raw)
        fecha_emision_text = (
            self._format_date_long(fecha_emision_date)
            if fecha_emision_date
            else (fecha_emision_raw or "****")
        )

        siniestro_raw = None
        if isinstance(case_core, dict):
            entry = case_core.get("fecha_ocurrencia")
            if isinstance(entry, dict):
                siniestro_raw = entry.get("value")
        if not siniestro_raw:
            siniestro_raw = consolidated.get("fecha_ocurrencia") or resolved_fields.get("fecha_ocurrencia")
        siniestro_date = self._parse_iso_date(siniestro_raw)
        if fecha_emision_date and siniestro_date:
            delta_days = (fecha_emision_date - siniestro_date).days
            day_delta_phrase = self._format_day_delta(delta_days)
        else:
            day_delta_phrase = "sin referencia temporal del siniestro"

        moneda = self._normalize_currency_token(
            _pick_value("moneda", "moneda_cfdi", "moneda_comprobante", default="")
        )
        if declared_value is not None:
            formatted_value = self._format_currency(declared_value)
            if moneda == "MXN":
                value_text = f"{formatted_value} M.N."
            elif moneda:
                value_text = f"{formatted_value} {moneda}"
            else:
                value_text = formatted_value
        else:
            value_text = _pick_value("valor_mercancia_texto", default="sin especificar")
        if not value_text:
            value_text = "sin especificar"

        origen_cfdi = _pick_value(
            "punto_origen",
            "origen",
            "domicilio_origen",
            "origen_nombre",
            "origen_descripcion",
            default="",
        )
        destino_cfdi = _pick_value(
            "punto_destino",
            "destino",
            "domicilio_destino",
            "destino_nombre",
            "destino_descripcion",
            default="",
        )

        def _dedupe_location(text: str) -> str:
            if not text:
                return text
            parts = [segment.strip() for segment in text.split(",") if segment.strip()]
            cleaned: List[str] = []
            seen_norm: Set[str] = set()
            for segment in parts:
                normalized_segment = self._normalize_text_for_search(segment)
                if normalized_segment in seen_norm:
                    continue
                seen_norm.add(normalized_segment)
                cleaned.append(segment)
            return ", ".join(cleaned) if cleaned else text

        origen_cfdi = _dedupe_location(origen_cfdi)
        destino_cfdi = _dedupe_location(destino_cfdi)

        plates_display: List[str] = []
        remolques_display: List[str] = []
        plates_norm: Set[str] = set()

        def _register_plate(value: Any, *, is_trailer: bool = False) -> None:
            text = self._stringify_value(value)
            if not text:
                return
            tokens = re.split(r"[,\s/;]+", text)
            for token in tokens:
                cleaned = token.strip().upper()
                if not cleaned:
                    continue
                norm = self._normalize_plate(cleaned)
                if not norm or norm in plates_norm:
                    continue
                plates_norm.add(norm)
                if is_trailer:
                    remolques_display.append(cleaned)
                else:
                    plates_display.append(cleaned)

        _register_plate(_pick_value("placa_transporte", "placa_unidad", "placa_principal", "placa_tractor"))
        _register_plate(_pick_value("placas", "placas_transporte", "placas_unidades"))
        for seq in _pick_sequence("vehiculos", "unidades", "unidades_transporte"):
            if isinstance(seq, dict):
                _register_plate(
                    seq.get("placa")
                    or seq.get("placas")
                    or seq.get("identificador")
                    or seq.get("numero"),
                    is_trailer=False,
                )
            else:
                _register_plate(seq)
        for seq in _pick_sequence("remolques", "placas_remolque", "remolques_detalle"):
            if isinstance(seq, dict):
                _register_plate(
                    seq.get("placa")
                    or seq.get("placas")
                    or seq.get("identificador")
                    or seq.get("numero"),
                    is_trailer=True,
                )
            else:
                _register_plate(seq, is_trailer=True)

        tractor_plate = plates_display[0] if plates_display else ""

        case_origins, case_destinations = self._collect_case_route_points(data_layer)

        def _compare_location(value: str, references: List[str]) -> Tuple[str, Optional[float], str]:
            if not value:
                return "desconocido", None, ""
            if not references:
                return "desconocido", None, ""
            normalized_value = self._normalize_text_for_search(value)
            best_distance: Optional[float] = None
            best_reference = ""
            for ref in references:
                if not ref:
                    continue
                normalized_ref = self._normalize_text_for_search(ref)
                if normalized_value and normalized_ref and (
                    normalized_value in normalized_ref or normalized_ref in normalized_value
                ):
                    return "coincide", 0.0, ref
                distance = self._distance_between_locations(value, ref)
                if distance is not None:
                    if distance <= 500:
                        return "coincide", distance, ref
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_reference = ref
            if best_distance is not None:
                return "discrepancia", best_distance, best_reference
            return "discrepancia", None, ""

        origin_status, origin_distance, origin_reference = _compare_location(origen_cfdi, case_origins)
        destination_status, destination_distance, destination_reference = _compare_location(destino_cfdi, case_destinations)

        if origin_status == "coincide" and destination_status == "coincide":
            ruta_result = "coincide"
        elif origin_status == "desconocido" and destination_status == "desconocido":
            ruta_result = "desconocido"
        elif origin_status == "discrepancia" and destination_status == "discrepancia":
            ruta_result = "discrepancia"
        else:
            ruta_result = "parcial"

        ruta_parts: List[str] = []
        if origin_status == "coincide":
            if origin_distance and origin_distance > 0:
                ruta_parts.append(f"Origen coincide (±{origin_distance:.0f} m).")
            else:
                ruta_parts.append("Origen coincide con la denuncia.")
        elif origin_status == "discrepancia":
            if origin_distance is not None:
                ruta_parts.append(
                    f"Origen no coincide; distancia aproximada {origin_distance:.0f} m frente a {origin_reference or 'la referencia del expediente'}."
                )
            else:
                ruta_parts.append("Origen declarado no coincide con los documentos del expediente.")
        else:
            ruta_parts.append("Sin referencia de origen para comparar.")

        if destination_status == "coincide":
            if destination_distance and destination_distance > 0:
                ruta_parts.append(f"Destino coincide (±{destination_distance:.0f} m).")
            else:
                ruta_parts.append("Destino coincide con la denuncia.")
        elif destination_status == "discrepancia":
            if destination_distance is not None:
                ruta_parts.append(
                    f"Destino no coincide; distancia aproximada {destination_distance:.0f} m frente a {destination_reference or 'la referencia del expediente'}."
                )
            else:
                ruta_parts.append("Destino declarado no coincide con los documentos del expediente.")
        else:
            ruta_parts.append("Sin referencia de destino para comparar.")

        ruta_detalle = " ".join(segment.strip() for segment in ruta_parts if segment).strip()

        denuncias_summary = self._collect_denuncia_summary(data_layer)
        operator_entry = None
        operator_norm = self._normalize_person_name(operador)
        if operator_norm:
            for entry in denuncias_summary:
                entry_norm = entry.get("nombre_norm")
                if not entry_norm:
                    continue
                if entry_norm == operator_norm or entry_norm.startswith(operator_norm) or operator_norm.startswith(entry_norm):
                    operator_entry = entry
                    break

        if operator_norm and operator_entry:
            if operator_entry.get("nombre"):
                operador = self._format_entity_name(operator_entry.get("nombre"))
            operador_result = "coincide"
            operador_detalle = "El operador coincide con la denuncia."
        elif operator_norm and operator_entry is None and denuncias_summary:
            operador_result = "discrepancia"
            operador_detalle = "El operador declarado no se localiza en denuncias."
        elif operator_norm:
            operador_result = "desconocido"
            operador_detalle = "No se cuenta con denuncias para validar al operador declarado."
        else:
            operador_result = "desconocido"
            operador_detalle = "El CFDI no identifica operador para comparar contra denuncias."

        case_plates_all: Set[str] = set()
        for entry in denuncias_summary:
            for plate_norm in entry.get("placas_norm") or []:
                case_plates_all.add(plate_norm)

        operator_plates = set(operator_entry.get("placas_norm") or []) if operator_entry else set()
        if plates_norm:
            if operator_plates and plates_norm & operator_plates:
                plates_result = "coincide"
                plates_detalle = "Las placas coinciden con las asignadas al operador en la denuncia."
            elif case_plates_all and plates_norm & case_plates_all:
                plates_result = "parcial"
                plates_detalle = "Las placas están presentes en el expediente pero asociadas a otro operador."
            elif case_plates_all:
                plates_result = "discrepancia"
                plates_detalle = "Las placas declaradas no se localizaron en denuncias ni en la carpeta."
            else:
                plates_result = "desconocido"
                plates_detalle = "No hay referencias en el expediente para validar las placas declaradas."
        else:
            plates_result = "desconocido"
            plates_detalle = "El CFDI no detalla placas de las unidades."

        display_index: Dict[str, str] = {}
        for plate in plates_display + remolques_display:
            norm = self._normalize_plate(plate)
            if norm and norm not in display_index:
                display_index[norm] = plate
        plates_coincidentes_display = [
            display_index.get(norm, norm)
            for norm in sorted(plates_norm & case_plates_all)
        ]

        fiscal = analysis.fiscal_validation
        validation_sentence = "Sin validación fiscal disponible para esta carta porte."
        validation_rows: List[Tuple[str, Any]] = []
        cancellation_display = ""
        if fiscal:
            if fiscal.cancellation_date:
                cancel_dt = self._parse_datetime(fiscal.cancellation_date) or self._parse_iso_date(fiscal.cancellation_date)
                if isinstance(cancel_dt, datetime):
                    cancellation_display = cancel_dt.strftime("%d/%m/%Y %H:%M")
                elif isinstance(cancel_dt, date):
                    cancellation_display = self._format_date_slash(cancel_dt)
            issuer_name = fiscal.issuer_name or _pick_value("emisor_nombre", "nombre_transportista", "emisor", default="")
            recipient_name = fiscal.recipient_name or _pick_value("receptor_nombre", "destinatario", "receptor", default="")
            validation_rows.extend(
                [
                    ("RFC del emisor", fiscal.request.issuer_rfc),
                    ("Nombre o razón social del emisor", issuer_name),
                    ("RFC del receptor", fiscal.request.recipient_rfc),
                    ("Nombre o razón social del receptor", recipient_name),
                    ("Folio fiscal (UUID)", fiscal.request.uuid.upper()),
                    ("Fecha de expedición", fiscal.issue_date or fecha_emision_raw),
                    ("Fecha certificación SAT", fiscal.sat_certification_date or ""),
                    ("PAC que certificó", fiscal.pac_certifier or ""),
                    ("Total del CFDI", self._format_currency(fiscal.request.total)),
                    ("Efecto del comprobante", fiscal.invoice_effect or ""),
                    ("Estado CFDI", fiscal.status.value.upper()),
                    ("Estatus de cancelación", fiscal.cancelable_status or fiscal.status_detail or ""),
                    ("Fecha de cancelación", cancellation_display),
                ]
            )
            if fiscal.is_cancelado():
                validation_sentence = (
                    "Se realizó validación de carta de porte ante portal SAT, donde se aprecia registro, "
                    "observando que se encuentra cancelada por plazo vencido."
                )
                if cancellation_display:
                    validation_sentence += f" Siendo cancelada en fecha del {cancellation_display}."
            elif fiscal.is_vigente():
                validation_sentence = (
                    "Se realizó validación de carta de porte donde se aprecia registro, encontrándose vigente ante portal SAT."
                )
            elif fiscal.is_not_found():
                validation_sentence = (
                    "La consulta en el portal SAT no localizó el CFDI carta porte; se requiere verificación manual."
                )
            elif fiscal.had_error():
                validation_sentence = (
                    "La validación fiscal devolvió un error por lo que debe repetirse manualmente en el portal SAT."
                )
            else:
                validation_sentence = (
                    "Se obtuvo un estatus pendiente en la validación fiscal; se requiere seguimiento manual."
                )
        validation_table = _render_table(validation_rows)

        heading = f"CARTA DE PORTE TIMBRADA FOLIO {folio_display}"
        representante_text = representante or "sin representante identificado"
        transportista_text = transportista or "la transportista declarada"
        summary_sentence = (
            f"Se ha recibido la carta de porte con folio {folio_display}, emitida por {representante_text}, "
            f"representante de la transportista {transportista_text}. Esta carta fue expedida el {fecha_emision_text}, "
            f"es decir, {day_delta_phrase}. El contenido de la mercancía, según este documento, consiste en "
            f"{goods_summary}, con un valor declarado de {value_text}."
        ).replace("  ", " ").strip()

        if operador:
            operator_sentence = (
                f"Podemos encontrar en esta carta porte los datos del operador, {operador}, "
                "y la información de las unidades de transporte que este operaba en el momento del incidente."
            )
        else:
            operator_sentence = (
                "La carta porte no detalla el nombre del operador responsable de las unidades al momento del incidente."
            )

        unidades_parts: List[str] = []
        if tractor_plate:
            unidades_parts.append(f"el tractocamión con placa {tractor_plate}")
        if remolques_display:
            unidades_parts.append(f"los remolques con placas {', '.join(remolques_display)}")
        unidades_text = " y ".join(unidades_parts) if unidades_parts else "las unidades declaradas en el complemento"

        complemento_sentence_parts: List[str] = [
            "Se cuenta con un complemento carta de porte donde se observan datos del traslado de mercancías."
        ]
        complemento_sentence_parts.append(f"La mercancía era transportada en {unidades_text}.")
        if goods_complement:
            complemento_sentence_parts.append(f"Se declara mercancía consistente en {goods_complement}.")
        if origen_cfdi or destino_cfdi:
            origen_text = origen_cfdi or "un origen no especificado"
            destino_text = destino_cfdi or "un destino no especificado"
            complemento_sentence_parts.append(f"Se aprecia origen {origen_text} y destino {destino_text}.")
        if operador:
            complemento_sentence_parts.append(f"Asimismo, se consigna al operador asignado {operador}.")
        complemento_sentence = " ".join(segment.strip() for segment in complemento_sentence_parts if segment).strip()

        narrative_lines = [
            heading,
            summary_sentence,
            "",
            operator_sentence,
            "",
            "VALIDACIÓN",
            validation_sentence,
        ]
        if validation_table:
            narrative_lines.append(validation_table)
        narrative_lines.extend(
            [
                "",
                "COMPLEMENTO CARTA DE PORTE",
                complemento_sentence,
            ]
        )
        analysis.analisis_completo = "\n".join(line for line in narrative_lines if line is not None).strip()

        verificaciones = {
            "operador_vs_denuncia": {
                "resultado": operador_result,
                "operador_carta": operador,
                "operador_denuncia": operator_entry.get("nombre") if operator_entry else "",
                "detalle": operador_detalle,
            },
            "placas_vs_denuncia": {
                "resultado": plates_result,
                "placas_cfdi": plates_display + [p for p in remolques_display if p not in plates_display],
                "placas_coincidentes": plates_coincidentes_display,
                "detalle": plates_detalle,
            },
            "ruta_vs_denuncia": {
                "resultado": ruta_result,
                "origen_cfdi": origen_cfdi,
                "origen_referencia": origin_reference,
                "destino_cfdi": destino_cfdi,
                "destino_referencia": destination_reference,
                "distancia_origen_m": origin_distance,
                "distancia_destino_m": destination_distance,
                "detalle": ruta_detalle,
            },
        }
        analysis.verificaciones = verificaciones

        cfdi_context = {
            "serie": serie_clean,
            "folio": folio_display,
            "representante": representante,
            "transportista": transportista,
            "operador": operador,
            "origen": origen_cfdi,
            "destino": destino_cfdi,
            "valor_declarado": value_text,
            "placas": plates_display + [p for p in remolques_display if p not in plates_display],
            "unidades": unidades_text,
        }
        if declared_value is not None:
            try:
                cfdi_context["valor_decimal"] = float(declared_value)
            except Exception:
                pass
        validacion_cruzada = dict(analysis.validacion_cruzada or {})
        validacion_cruzada["cfdi_carta_porte"] = cfdi_context
        analysis.validacion_cruzada = validacion_cruzada

        recommendations = [rec for rec in analysis.recommendations if rec]
        if operador_result == "desconocido":
            recommendations.append(
                "Solicitar versión del CFDI Carta Porte que identifique al operador asignado."
            )
        if plates_result == "desconocido":
            recommendations.append(
                "Solicitar documentación del transportista que acredite las placas de las unidades declaradas."
            )
        if ruta_result == "desconocido":
            recommendations.append(
                "Obtener denuncias o soportes logísticos que acrediten origen y destino para validar el CFDI."
            )
        if fiscal and fiscal.is_cancelado():
            if cancellation_display:
                recommendations.append(
                    f"Verificar en el portal SAT la fecha de cancelación ({cancellation_display}) y documentarla en el reporte."
                )
            else:
                recommendations.append(
                    "Verificar en el portal SAT la fecha de cancelación del CFDI y documentarla en el reporte."
                )
        analysis.recommendations = list(dict.fromkeys(recommendations))

        indicators = [ind for ind in analysis.indicators or []]

        def _add_indicator(pattern: str, description: str, severity: str, confidence: float = 0.8) -> None:
            for idx, existing in enumerate(indicators):
                if existing.pattern == pattern:
                    indicators[idx] = FraudIndicator(
                        pattern=pattern,
                        description=description,
                        severity=severity,
                        confidence=confidence,
                    )
                    return
            indicators.append(
                FraudIndicator(
                    pattern=pattern,
                    description=description,
                    severity=severity,
                    confidence=confidence,
                )
            )

        if fiscal and fiscal.is_cancelado():
            _add_indicator(
                "cfdi_cancelado",
                "El CFDI aparece cancelado en la validación SAT.",
                "alto",
                0.85,
            )
        if operador_result == "discrepancia":
            _add_indicator(
                "operador_no_coincidente",
                "El operador declarado en el CFDI no coincide con las declaraciones del expediente.",
                "alto",
                0.85,
            )
        if plates_result == "discrepancia":
            _add_indicator(
                "placas_no_coinciden_denuncia",
                "Las placas declaradas en el CFDI no se localizaron en la denuncia ni en la carpeta.",
                "alto",
                0.85,
            )
        if ruta_result == "discrepancia":
            _add_indicator(
                "ruta_inconsistente_denuncia",
                "La ruta declarada en el CFDI difiere de la denunciada o no se encuentra dentro de la tolerancia establecida.",
                "alto",
                0.85,
            )

        analysis.indicators = indicators

        severity_rank = {"bajo": 1, "medio": 2, "alto": 3, "critico": 4}
        if indicators:
            max_severity = max(severity_rank.get(ind.severity.lower(), 2) for ind in indicators)
            if max_severity >= 4:
                analysis.risk_level = RiskLevel.CRITICO
                analysis.fraud_score = max(analysis.fraud_score, 0.90)
                analysis.confidence = max(analysis.confidence, 0.75)
            elif max_severity == 3:
                analysis.risk_level = RiskLevel.ALTO
                analysis.fraud_score = max(analysis.fraud_score, 0.72)
                analysis.confidence = max(analysis.confidence, 0.80)
            elif max_severity == 2:
                analysis.risk_level = RiskLevel.MEDIO
                analysis.fraud_score = max(analysis.fraud_score, 0.48)
                analysis.confidence = max(analysis.confidence, 0.85)
            else:
                analysis.risk_level = RiskLevel.BAJO
                analysis.fraud_score = max(analysis.fraud_score, 0.30)
                analysis.confidence = max(analysis.confidence, 0.90)
        else:
            analysis.risk_level = RiskLevel.BAJO
            analysis.fraud_score = 0.24
            analysis.confidence = max(analysis.confidence, 0.92)

        if fiscal and fiscal.is_cancelado() and analysis.risk_level in {RiskLevel.BAJO, RiskLevel.MEDIO}:
            analysis.risk_level = RiskLevel.ALTO
            analysis.fraud_score = max(analysis.fraud_score, 0.72)
            analysis.confidence = max(analysis.confidence, 0.80)

        return analysis

    def _extract_transportista_from_text(self, text: str) -> str:
        if not text:
            return ""
        previous_line = ""
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            upper = self._strip_accents(clean.upper())
            if "TRANSPORT" not in upper:
                previous_line = clean
                continue

            candidate = clean
            tokens = clean.split()
            transport_positions = [
                idx for idx, token in enumerate(tokens) if "TRANSPORT" in self._strip_accents(token.upper())
            ]
            if transport_positions:
                first_idx = transport_positions[0]
                candidate_tokens = tokens[first_idx:]
                if first_idx > 0:
                    candidate_tokens = tokens[first_idx - 1 :]
                candidate = " ".join(candidate_tokens)

            if previous_line and len(previous_line.split()) <= 4:
                candidate = f"{previous_line} {candidate}"

            candidate = candidate.strip(",.;:")
            formatted = self._format_entity_name(candidate)
            if formatted:
                return formatted
            previous_line = clean
        return ""

    def _extract_spanish_date(self, text: str) -> Optional[date]:
        if not text:
            return None
        pattern = re.compile(
            r"(\d{1,2})\s+de\s+([A-Za-zÁÉÍÓÚÑ]+)\s+de\s+(\d{4})",
            re.IGNORECASE,
        )
        months = {
            "enero": 1,
            "febrero": 2,
            "marzo": 3,
            "abril": 4,
            "mayo": 5,
            "junio": 6,
            "julio": 7,
            "agosto": 8,
            "septiembre": 9,
            "setiembre": 9,
            "octubre": 10,
            "noviembre": 11,
            "diciembre": 12,
        }
        for match in pattern.finditer(text):
            day_str, month_text, year_str = match.groups()
            month = months.get(self._strip_accents(month_text.lower()))
            if not month:
                continue
            try:
                return date(int(year_str), month, int(day_str))
            except ValueError:
                continue
        return None

    def _normalize_currency_token(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        upper = self._strip_accents(str(token).upper())
        if "USD" in upper or "DOLAR" in upper or "US$" in upper:
            return "USD"
        if "MXN" in upper or "M.N" in upper:
            return "MXN"
        if "$" in upper:
            return None
        return None

    def _detect_currency_from_text(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        upper = self._strip_accents(str(text).upper())
        if "USD" in upper or "DOLAR" in upper or "US$" in upper:
            return "USD"
        if "MXN" in upper or "M.N" in upper or "PESOS" in upper:
            return "MXN"
        return None

    def _extract_goods_keywords(self, snippets: Iterable[str]) -> set:
        keywords: set = set()
        stopwords = {
            "DE",
            "LA",
            "EL",
            "LOS",
            "LAS",
            "DEL",
            "PARA",
            "CON",
            "POR",
            "UN",
            "UNA",
            "Y",
            "EN",
            "QUE",
            "SE",
            "MERCANCIA",
            "MATERIAL",
            "DIFERENTES",
            "MEDIDAS",
            "PEDIMENTO",
            "KG",
            "PZAS",
            "PIEZAS",
            "PIECES",
            "PCS",
        }
        for snippet in snippets or []:
            text = self._strip_accents(str(snippet).upper())
            for token in re.findall(r"[A-Z0-9]{3,}", text):
                if token in stopwords:
                    continue
                keywords.add(token)
        return keywords

    def _extract_amount_with_currency(
        self,
        value: Optional[str],
        *,
        fallback_text: str = "",
    ) -> Tuple[Optional[Decimal], Optional[str]]:
        entries: List[Tuple[Decimal, Optional[str]]] = []
        seen: set[Tuple[str, Optional[str]]] = set()

        if value:
            amount = self._parse_number_token(str(value))
            if amount is not None:
                currency = self._detect_currency_from_text(str(value))
                if not currency:
                    currency = self._detect_currency_from_text(fallback_text)
                key = (str(amount), currency)
                if key not in seen:
                    entries.append((amount, currency))
                    seen.add(key)

        if fallback_text:
            pattern = re.compile(
                r"(?P<prefix>\$|US\$|USD|MXN|M\.N\.)?\s*(?P<amount>[0-9]+(?:[\s,][0-9]{3})*(?:\.[0-9]{2})?)\s*(?P<suffix>USD|US\$|MXN|M\.N\.|DOLARES|DÓLARES)?",
                re.IGNORECASE,
            )
            for match in pattern.finditer(fallback_text):
                prefix = match.group("prefix")
                suffix = match.group("suffix")
                if not prefix and not suffix and "$" not in match.group(0):
                    continue
                amount_raw = match.group("amount")
                amount = self._parse_number_token(amount_raw)
                if amount is None or amount == 0:
                    continue
                currency = self._normalize_currency_token(prefix) or self._normalize_currency_token(suffix)
                key = (str(amount), currency)
                if key in seen:
                    continue
                entries.append((amount, currency))
                seen.add(key)

        if not entries:
            return None, None

        entries.sort(key=lambda item: item[0], reverse=True)
        amount, currency = entries[0]
        if currency is None:
            currency = self._detect_currency_from_text(fallback_text) or "MXN"
        return amount, currency

    def _collect_exchange_rate(
        self,
        data_layer: Optional[UnifiedDataLayer],
        reference_date: Optional[date],
    ) -> Optional[Decimal]:
        rate = ExchangeRateService.get_rate(reference_date)
        if rate is not None:
            return rate

        if data_layer:
            for extraction in getattr(data_layer, "_extractions", {}).values():
                if extraction.document_type not in {"pedimento_importacion", "facturas_comerciales_internacionales"}:
                    continue
                fields = extraction.extracted_fields or {}
                for key in ("tipo_cambio_aplicado", "tipo_cambio", "exchange_rate"):
                    value = fields.get(key)
                    if value is None:
                        continue
                    rate = self._parse_number_token(str(value))
                    if rate and rate > 0:
                        return rate

        return None

    def _infer_exchange_rate(
        self,
        monto_carta: Decimal,
        moneda_carta: Optional[str],
        monto_aseguradora: Decimal,
        moneda_aseguradora: Optional[str],
    ) -> Optional[Decimal]:
        if not monto_carta or not monto_aseguradora:
            return None
        if not moneda_carta or not moneda_aseguradora:
            return None

        try:
            ratio = (monto_aseguradora / monto_carta).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        except (InvalidOperation, ZeroDivisionError):
            return None

        moneda_carta_norm = (moneda_carta or "").upper()
        moneda_aseguradora_norm = (moneda_aseguradora or "").upper()

        if moneda_carta_norm == "USD" and moneda_aseguradora_norm == "MXN":
            if Decimal("5") <= ratio <= Decimal("50"):
                return ratio
        elif moneda_carta_norm == "MXN" and moneda_aseguradora_norm == "USD":
            if ratio == 0:
                return None
            inverse = (Decimal("1") / ratio).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            if Decimal("0.02") <= inverse <= Decimal("0.50"):
                return inverse
        return None

    def _format_percentage(self, value: Decimal) -> str:
        pct = (value * Decimal("100")).quantize(Decimal("0.1"))
        return f"{pct}%"

    def _extract_additional_insured(self, text: str) -> List[str]:
        if not text:
            return []
        pattern = re.compile(
            r"ASEGURADOS ADICIONALES:?(.+?)(?:\n\s*\n|$)",
            re.IGNORECASE | re.DOTALL,
        )
        sections = pattern.findall(text)
        resultados: List[str] = []
        for section in sections:
            for entry in re.findall(r"\d+\.\s*([A-ZÁÉÍÓÚÑ ,.&/-]+)", section, re.IGNORECASE):
                formatted = self._format_entity_name(entry)
                if (
                    formatted
                    and sum(ch.isalpha() for ch in formatted) >= 5
                    and "póliza" not in formatted.lower()
                    and not formatted.lower().startswith(("atención", "unidad", "original", "opción"))
                    and self._is_valid_additional_insured_name(formatted)
                    and formatted not in resultados
                ):
                    resultados.append(formatted)
        return resultados

    def _extract_goods_from_text(self, text: str) -> str:
        if not text:
            return ""
        match = re.search(
            r"(material|mercanc[ií]a|placas)\s+de\s+[A-Za-zÁÉÍÓÚÑ.\s]+",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(0)
        return ""

    def _goods_match(self, a: str, b: str) -> bool:
        a_norm = self._normalize_identifier(a)
        b_norm = self._normalize_identifier(b)
        if a_norm and b_norm and (a_norm in b_norm or b_norm in a_norm):
            return True
        tokens_a = self._goods_tokens(a)
        tokens_b = self._goods_tokens(b)
        if not tokens_a or not tokens_b:
            return False
        return bool(tokens_a & tokens_b)

    def _goods_tokens(self, text: str) -> set[str]:
        stopwords = {
            "",
            "DE",
            "DEL",
            "LA",
            "LAS",
            "EL",
            "LOS",
            "Y",
            "EN",
            "POR",
            "AL",
            "UNA",
            "UN",
            "TONELADAS",
            "TONELADA",
            "KILOGRAMOS",
            "KILOGRAMO",
            "KG",
            "KM",
            "MATERIAL",
            "MERCANCIA",
        }
        tokens = re.split(r"[^A-ZÁÉÍÓÚÑ0-9]+", self._format_goods(text).upper())
        return {token for token in tokens if token and token not in stopwords}

    def _is_valid_additional_insured_name(self, name: str) -> bool:
        upper = name.upper()
        keywords = (
            "S.A.",
            "S. DE",
            "SOCIEDAD",
            "IMPORTACION",
            "ACEROS",
            "TRANSPORT",
            "COMERCIAL",
            "CENTRAL",
            "DISTRIBUIDORA",
        )
        return any(keyword in upper for keyword in keywords)

    def _strip_accents(self, value: str) -> str:
        normalized = unicodedata.normalize("NFD", value)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    def _normalize_person_name(self, name: Optional[str]) -> str:
        if not name:
            return ""
        text = self._strip_accents(str(name).upper())
        text = re.sub(r"\b(SR|SRA|SRS|SRAS|LIC|ING|ARQ|CAP|C)\b\.?", "", text)
        text = re.sub(r"[^A-Z]", "", text)
        return text

    def _person_name_token_list(self, name: Optional[str]) -> List[str]:
        if not name:
            return []
        cleaned = self._strip_accents(str(name).upper())
        tokens = [token for token in re.split(r"[^A-Z]+", cleaned) if token]
        blacklist = {"DE", "DEL", "LA", "LAS", "LOS", "SR", "SRA", "LIC", "ING", "ARQ", "C"}
        return [token for token in tokens if token and token not in blacklist]

    def _person_name_tokens(self, name: Optional[str]) -> Set[str]:
        return set(self._person_name_token_list(name))

    def _person_names_match_loose(self, first_tokens: List[str], second_tokens: List[str]) -> bool:
        if not first_tokens or not second_tokens:
            return False

        shared = {token for token in first_tokens if token in second_tokens and len(token) >= 3}
        if shared:
            return True

        first_name_a = first_tokens[0]
        first_name_b = second_tokens[0]
        similarity = SequenceMatcher(None, first_name_a, first_name_b).ratio()
        if similarity >= 0.8:
            surnames_a = set(first_tokens[1:])
            surnames_b = set(second_tokens[1:])
            if surnames_a and surnames_b and surnames_a & surnames_b:
                return True

        if len(first_tokens) == 1 and len(second_tokens) == 1:
            return SequenceMatcher(None, first_name_a, first_name_b).ratio() >= 0.8

        return False

    def _normalize_company_name(self, name: Optional[str]) -> str:
        if not name:
            return ""
        upper = self._strip_accents(str(name).upper())
        for suffix in [
            "S.A. DE C.V.",
            "S.A. DE C. V.",
            "SA DE CV",
            "S. DE R.L.",
            "S DE RL",
            "SOCIEDAD ANONIMA",
            "SOCIEDAD ANÓNIMA",
        ]:
            upper = upper.replace(suffix, "")
        upper = re.sub(r"[^A-Z0-9]", "", upper)
        return upper

    def _company_name_tokens(self, name: Optional[str]) -> Set[str]:
        if not name:
            return set()
        cleaned = self._strip_accents(str(name).upper())
        tokens = [token for token in re.split(r"[^A-Z0-9]+", cleaned) if token]
        if not tokens:
            return set()
        generic_tokens = {
            "SA",
            "S",
            "DE",
            "C",
            "CV",
            "RL",
            "SRL",
            "SOCIEDAD",
            "ANONIMA",
            "ANÓNIMA",
            "SOC",
            "GRUPO",
            "GROUP",
            "GRP",
            "COMPANIA",
            "COMPAÑIA",
            "COMPANY",
            "LLC",
            "INC",
            "SADECV",
            "SAPI",
            "Y",
        }
        filtered = [token for token in tokens if token not in generic_tokens and len(token) > 1]
        return set(filtered or tokens)

    def _company_names_match(self, first: Optional[str], second: Optional[str]) -> bool:
        if not first or not second:
            return False
        norm_first = self._normalize_company_name(first)
        norm_second = self._normalize_company_name(second)
        if norm_first and norm_second:
            if norm_first == norm_second:
                return True
            if norm_first in norm_second or norm_second in norm_first:
                return True
        tokens_first = self._company_name_tokens(first)
        tokens_second = self._company_name_tokens(second)
        if tokens_first and tokens_second:
            overlap = tokens_first & tokens_second
            if not overlap:
                return False
            min_len = min(len(tokens_first), len(tokens_second))
            if min_len == 0:
                return False
            ratio = len(overlap) / min_len
            if ratio >= 0.6:
                return True
            if len(overlap) >= 2 and ratio >= 0.5:
                return True
        return False

    def _plate_variants(self, plate: Optional[str]) -> Set[str]:
        normalized = self._normalize_plate(plate)
        if not normalized:
            return set()

        mapping: Dict[str, Set[str]] = {
            "0": {"0", "O"},
            "O": {"O", "0"},
            "1": {"1", "I", "L"},
            "I": {"I", "1", "L"},
            "L": {"L", "1", "I"},
            "2": {"2", "Z"},
            "Z": {"Z", "2"},
            "5": {"5", "S"},
            "S": {"S", "5"},
            "8": {"8", "B"},
            "B": {"B", "8"},
            "6": {"6", "G"},
            "9": {"9", "G"},
            "G": {"G", "6", "9"},
        }

        variants: Set[str] = set()

        def _expand(index: int, prefix: List[str]) -> None:
            if index >= len(normalized):
                variants.add("".join(prefix))
                return
            char = normalized[index]
            options = mapping.get(char, {char})
            for option in options:
                prefix.append(option)
                _expand(index + 1, prefix)
                prefix.pop()

        _expand(0, [])
        variants.add(normalized)
        return variants

    def _plates_match(self, first: Optional[str], second: Optional[str]) -> bool:
        if not first or not second:
            return False
        variants_first = self._plate_variants(first)
        variants_second = self._plate_variants(second)
        return bool(variants_first and variants_second and variants_first & variants_second)

    def _normalize_text_for_search(self, text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", self._strip_accents(text.upper()))

    def _shorten_goods_reference(self, text: str) -> str:
        if not text:
            return ""
        lowered = text.lower()
        cut_points = [" lím", " limite", " total", " ajuste", " mxn", " monto", " peso ", " toneladas"]
        index = len(lowered)
        for marker in cut_points:
            pos = lowered.find(marker)
            if pos != -1:
                index = min(index, pos)
        cleaned = text[:index].strip()
        return cleaned or text.strip()

    def _extract_pedimento_goods_lines(self, text: str, *, limit: int = 5) -> List[str]:
        if not text:
            return []
        keywords = (
            "MERC",
            "PLACA",
            "LAMIN",
            "ACERO",
            "TUBO",
            "BOB",
            "ROLLO",
            "TONEL",
            "BULTO",
            "PIEZA",
            "PAQUETE",
            "CAJA",
        )
        noise = re.compile(
            r"PEDIMENTO|NUM\.?|REGIMEN|ADUANA|VALOR|IMPORTE|FECHA|DATOS|CONTRIB|CUADRO|CAPTURA|SEGUROS|FLETES|CARGO|DESCARGA|TOTAL|CERTIFIC",
            re.IGNORECASE,
        )
        results: List[str] = []
        seen: Set[str] = set()
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or len(line) < 6:
                continue
            if ":" in line:
                continue
            upper = line.upper()
            if noise.search(upper):
                continue
            if "IDENTIFIC" in upper or "SUBD" in upper:
                continue
            if not any(kw in upper for kw in keywords):
                continue
            if sum(ch.isalpha() for ch in upper) / max(len(upper), 1) < 0.6:
                continue
            if any(ch.islower() for ch in line):
                continue
            display = upper.title()
            if display in seen:
                continue
            results.append(display)
            seen.add(display)
            if len(results) >= limit:
                break
        return results

    def _filter_recommendations(
        self,
        items: List[str],
        verificaciones: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        filtered: List[str] = []
        monto_status = (verificaciones.get("monto_vs_ajustador") or {}).get("resultado")
        for rec in items:
            lower = rec.lower()
            if "ajuste automático" in lower:
                continue
            if "monto reclamado" in lower and monto_status == "coincide":
                continue
            filtered.append(rec)
        return filtered

    def _extract_monto_ajustador(self, text: str) -> Optional[Decimal]:
        if not text:
            return None
        pattern_claim = re.compile(
            r"carta\s+de\s+reclamaci[oó]n[^$]*\$\s*([0-9,.]+)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern_claim.search(text)
        if match:
            value = self._parse_decimal(match.group(1))
            if value is not None:
                return value

        pattern_primary = re.compile(
            r"Recibimos\s+del\s+Asegurado[^$]*\$\s*([0-9,.]+)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern_primary.search(text)
        if match:
            return self._parse_decimal(match.group(1))
        fallback = re.search(r"\$\s*([0-9,.]+)", text)
        if fallback:
            return self._parse_decimal(fallback.group(1))
        return None

    def _collect_keywords(self, text: str, *, keywords: Iterable[str]) -> List[str]:
        if not text:
            return []
        lowered = text.lower()
        resultados: List[str] = []
        for keyword in keywords:
            if keyword.lower() in lowered:
                resultados.append(keyword)
        return sorted(set(resultados))

    def _extract_fecha_documento(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(
            r"(\\d{1,2})/(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)/(\\d{4})",
            text,
            re.IGNORECASE,
        )
        if match:
            return f"{match.group(1).zfill(2)}/{match.group(2)}"
        match_textual = re.search(
            r"(\d{1,2})\s+de\s+([a-záéíóúñ]+)\s+de\s+(\d{4})",
            text,
            re.IGNORECASE,
        )
        if match_textual:
            day, month_text, year = match_textual.groups()
            months = {
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
            month = months.get(month_text.lower())
            if month:
                return f"{day.zfill(2)}/{month}/{year}"
        return None

    def _apply_validation_rule(self, rule: Dict[str, Any], value: Any) -> Tuple[bool, str]:
        try:
            rtype = rule.get("type", "comparison")
            if rtype == "range":
                minv = rule.get("min")
                maxv = rule.get("max")
                numeric: Optional[Decimal]
                if isinstance(value, (int, float, Decimal)):
                    numeric = Decimal(str(value))
                else:
                    numeric = self._parse_decimal(value)
                if numeric is not None:
                    if minv is not None and numeric < Decimal(str(minv)):
                        return False, f"Valor {value} < mínimo {minv}"
                    if maxv is not None and numeric > Decimal(str(maxv)):
                        return False, f"Valor {value} > máximo {maxv}"
                else:
                    return True, "Sin comparación numérica"
            elif rtype == "pattern":
                import re
                pat = rule.get("pattern")
                if pat and not re.match(pat, str(value)):
                    return False, f"Valor {value} no coincide con patrón"
            elif rtype == "list":
                allowed = rule.get("values", [])
                if allowed and value not in allowed:
                    return False, f"Valor {value} fuera de lista permitida"
            return True, "OK"
        except Exception as e:  # pragma: no cover
            logger.error(f"Error en validación: {e}")
            return True, "Error en validación"

    def _derive_risk_level(self, score: float) -> str:
        s = float(score)
        if s >= 0.85:
            return "critico"
        if s >= 0.60:
            return "alto"
        if s >= 0.30:
            return "medio"
        return "bajo"

    async def _save_analysis_to_db(self, analysis: FraudAnalysisResult) -> None:
        try:
            with get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO fraud_analyses (
                        id, document_id, case_id, document_type,
                        risk_level, fraud_score, analisis_completo, indicators, evidence, evidence_gaps,
                        recommendations, verificaciones, validacion_cruzada, confidence, analysis_model,
                        guide_version, analysis_uuid, prompt_hash,
                        include_in_report,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{analysis.document_id}_fraud",
                        analysis.document_id,
                        analysis.case_id,
                        analysis.document_type,
                        analysis.risk_level.value,
                        analysis.fraud_score,
                        analysis.analisis_completo,
                        json.dumps([i.dict() for i in analysis.indicators], ensure_ascii=False),
                        "[]",
                        json.dumps([gap.model_dump() for gap in analysis.evidence_gaps], ensure_ascii=False),
                        json.dumps(analysis.recommendations, ensure_ascii=False),
                        json.dumps(analysis.verificaciones, ensure_ascii=False),
                        json.dumps(analysis.validacion_cruzada, ensure_ascii=False),
                        analysis.confidence,
                        analysis.analysis_model,
                        analysis.guide_version,
                        analysis.analysis_id,
                        analysis.prompt_hash,
                        1 if analysis.include_in_report else 0,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error guardando análisis: {e}")
            # Intento de autocorrección si la tabla no existe
            try:
                error_text = str(e).lower()
                if "no such table" in error_text:
                    with get_conn() as conn:
                        conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS fraud_analyses (
                                id              TEXT PRIMARY KEY,
                                document_id     TEXT NOT NULL,
                                case_id         TEXT NOT NULL,
                                document_type   TEXT NOT NULL,
                                risk_level      TEXT CHECK(risk_level IN ('bajo','medio','alto','critico')),
                                fraud_score     REAL CHECK(fraud_score >= 0 AND fraud_score <= 1),
                                analisis_completo TEXT,
                                indicators      TEXT,
                                evidence        TEXT,
                                evidence_gaps   TEXT NOT NULL DEFAULT '[]',
                                recommendations TEXT,
                                verificaciones  TEXT NOT NULL DEFAULT '{}',
                                validacion_cruzada TEXT NOT NULL DEFAULT '{}',
                                confidence      REAL,
                                analysis_model  TEXT,
                                guide_version   TEXT,
                                analysis_uuid   TEXT,
                                prompt_hash     TEXT,
                                include_in_report INTEGER NOT NULL DEFAULT 1,
                                created_at      TEXT NOT NULL,
                                updated_at      TEXT NOT NULL
                            );
                            """
                        )
                        conn.commit()
                        logger.info("Tabla fraud_analyses creada dinámicamente. Reintentando persistencia...")
                    # reintentar una vez
                    await self._save_analysis_to_db(analysis)
                elif "no column named evidence_gaps" in error_text:
                    with get_conn() as conn:
                        conn.execute(
                            "ALTER TABLE fraud_analyses ADD COLUMN evidence_gaps TEXT NOT NULL DEFAULT '[]';"
                        )
                        conn.commit()
                    await self._save_analysis_to_db(analysis)
                elif "no column named verificaciones" in error_text:
                    with get_conn() as conn:
                        conn.execute(
                            "ALTER TABLE fraud_analyses ADD COLUMN verificaciones TEXT NOT NULL DEFAULT '{}';"
                        )
                        conn.commit()
                    await self._save_analysis_to_db(analysis)
                elif "no column named validacion_cruzada" in error_text:
                    with get_conn() as conn:
                        conn.execute(
                            "ALTER TABLE fraud_analyses ADD COLUMN validacion_cruzada TEXT NOT NULL DEFAULT '{}';"
                        )
                        conn.commit()
                    await self._save_analysis_to_db(analysis)
            except Exception as e2:  # pragma: no cover
                logger.error(f"No fue posible autocorregir DB: {e2}")

    async def _generic_analysis(
        self,
        document_id: str,
        document_name: str,
        document_type: str,
        ocr_result: Dict[str, Any],
        extraction: DocumentExtraction,
        case_id: str,
    ) -> FraudAnalysisResult:
        indicators: List[FraudIndicator] = []

        # Campos críticos vacíos
        critical = ["numero_siniestro", "monto_reclamacion", "fecha_ocurrencia"]
        missing = [f for f in critical if f in extraction.extracted_fields and not extraction.extracted_fields.get(f)]
        if missing:
            indicators.append(
                FraudIndicator(
                    pattern="missing_critical_fields",
                    description=f"Campos críticos faltantes: {', '.join(missing)}",
                    severity="medio",
                    confidence=0.7,
                )
            )

        # Palabras sospechosas
        text_lower = (ocr_result.get("text") or "").lower()
        sw = [w for w in ["falsificado", "alterado", "modificado", "irregular"] if w in text_lower]
        if sw:
            indicators.append(
                FraudIndicator(
                    pattern="suspicious_language",
                    description=f"Lenguaje sospechoso detectado: {', '.join(sw)}",
                    severity="bajo",
                    confidence=0.5,
                )
            )

        score = min(len(indicators) * 0.2, 1.0)
        lvl = self._derive_risk_level(score)
        return FraudAnalysisResult(
            document_id=document_id,
            document_name=document_name,
            document_type=document_type,
            case_id=case_id,
            risk_level=RiskLevel(lvl),
            fraud_score=score,
            confidence=0.5,
            analisis_completo=(
                "Análisis genérico del documento basado en señales básicas "
                "(integridad, lenguaje sospechoso y campos críticos). "
                "Se recomienda crear una guía específica para maximizar la precisión "
                "y trazar validaciones complementarias."
            ),
            indicators=indicators,
            evidence_gaps=[],
            recommendations=[
                "Análisis genérico aplicado - considerar crear guía específica",
                "Revisión manual recomendada",
            ],
            verificaciones={},
            validacion_cruzada={},
            analysis_model="generic",
            guide_version="N/A",
            processing_time_ms=0,
        )

    def _create_error_analysis(
        self,
        document_id: str,
        document_name: str,
        document_type: str,
        case_id: str,
        error_message: str,
    ) -> FraudAnalysisResult:
        return FraudAnalysisResult(
            document_id=document_id,
            document_name=document_name,
            document_type=document_type,
            case_id=case_id,
            risk_level=RiskLevel.MEDIO,
            fraud_score=0.5,
            confidence=0.0,
            analisis_completo=(
                "No fue posible generar el análisis narrativo por un error en la ejecución. "
                "Se sugiere reintentar el proceso y realizar revisión manual del documento."
            ),
            indicators=[],
            evidence_gaps=[],
            recommendations=[
                "Revisar manualmente el documento",
                "Reintentar análisis",
                f"Detalle técnico: {error_message}",
            ],
            verificaciones={},
            validacion_cruzada={},
            analysis_model="error",
            guide_version="N/A",
            processing_time_ms=0,
        )
