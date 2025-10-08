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
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

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

logger = logging.getLogger(__name__)


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
            indicator_response, model_used = await self._call_ai_with_retry(
                indicator_prompt, context_name=document_name
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

            analysis = await self._enrich_analysis(analysis, extraction, guide)

            gap_prompt = self.prompts.build_evidence_gap_prompt(
                document_type=document_type,
                document_name=document_name,
                ocr_content=ocr_result,
                extracted_fields=extraction.extracted_fields,
                guide=guide._data,  # type: ignore[attr-defined]
                case_context=case_context,
                document_context=document_context,
            )
            combined_prompt = f"{indicator_prompt}\n\n---\n\n{gap_prompt}"
            analysis.prompt_hash = hashlib.sha256(combined_prompt.encode("utf-8")).hexdigest()
            try:
                gap_response, gap_model = await self._call_ai_with_retry(
                    gap_prompt, context_name=f"{document_name}_gaps"
                )
                gaps = self._parse_evidence_gap_response(gap_response)
                analysis.evidence_gaps = gaps
                if gap_model != model_used:
                    analysis.analysis_model = f"{model_used}|gaps:{gap_model}"
            except Exception as gap_exc:  # pragma: no cover - fallo tolerable
                logger.warning("No fue posible obtener brechas de evidencia para %s: %s", document_name, gap_exc)

            analysis.processing_time_ms = int((datetime.now() - start).total_seconds() * 1000)

            await self._save_analysis_to_db(analysis)
            logger.info(
                f"FraudAnalysis {document_name}: riesgo={analysis.risk_level.value} score={analysis.fraud_score:.2f}"
            )
            return analysis
        except Exception as e:
            logger.error(f"Error en análisis de fraude [{document_name}]: {e}")
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
        self, prompt: str, context_name: str, max_retries: int = 2
    ) -> Tuple[str, str]:
        last: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                model_name = self.model if attempt == 0 else self.model_fallback

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
                        evidence=ind.get("evidence"),
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

        # Evidencia y recomendaciones (a listas de strings)
        evidence = data.get("supporting_evidence") or data.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        evidence = [str(e) for e in evidence]

        recommendations = data.get("recommendations", []) or []
        if not isinstance(recommendations, list):
            recommendations = [str(recommendations)]
        recommendations = [str(r) for r in recommendations]

        if provided_risk and provided_risk != derived:
            evidence.append(
                f"Riesgo ajustado de '{provided_risk}' a '{derived}' (score={score:.2f})"
            )

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
            evidence=evidence,
            recommendations=recommendations,
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
                            evidence=f"{field}={value}",
                        )
                    )
        # Ajuste simple de score si muchos indicadores
        if len(analysis.indicators) > 5:
            analysis.fraud_score = min(1.0, analysis.fraud_score * 1.2)
            analysis.risk_level = RiskLevel(self._derive_risk_level(analysis.fraud_score))
        return analysis

    def _apply_validation_rule(self, rule: Dict[str, Any], value: Any) -> Tuple[bool, str]:
        try:
            rtype = rule.get("type", "comparison")
            if rtype == "range":
                minv = rule.get("min")
                maxv = rule.get("max")
                if minv is not None and value < minv:
                    return False, f"Valor {value} < mínimo {minv}"
                if maxv is not None and value > maxv:
                    return False, f"Valor {value} > máximo {maxv}"
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
                        recommendations, confidence, analysis_model,
                        guide_version, analysis_uuid, prompt_hash,
                        include_in_report,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        json.dumps(analysis.evidence, ensure_ascii=False),
                        json.dumps([gap.model_dump() for gap in analysis.evidence_gaps], ensure_ascii=False),
                        json.dumps(analysis.recommendations, ensure_ascii=False),
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
        evidence: List[str] = []

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
            evidence.append("Se detectaron palabras sospechosas en el texto OCR")

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
            evidence=evidence,
            evidence_gaps=[],
            recommendations=[
                "Análisis genérico aplicado - considerar crear guía específica",
                "Revisión manual recomendada",
            ],
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
            evidence=[f"Error en análisis: {error_message}"],
            evidence_gaps=[],
            recommendations=["Revisar manualmente el documento", "Reintentar análisis"],
            analysis_model="error",
            guide_version="N/A",
            processing_time_ms=0,
        )
