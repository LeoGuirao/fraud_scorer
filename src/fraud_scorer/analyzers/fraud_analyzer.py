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
    FraudAnalysisResult,
    FraudIndicator,
    RiskLevel,
)
from fraud_scorer.models.extraction import DocumentExtraction
from fraud_scorer.analyzers.fraud_guide_manager import FraudGuideManager, FraudGuide
from fraud_scorer.prompts.fraud_prompts import FraudPromptBuilder
from fraud_scorer.storage.db import get_conn
from fraud_scorer.settings import get_model_for_task

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
    ) -> FraudAnalysisResult:
        guide = self.guides.get_guide(document_type)
        if not guide:
            logger.warning(f"No hay guía para {document_type}. Ejecutando análisis genérico")
            return await self._generic_analysis(
                document_id, document_name, document_type, ocr_result, extraction, case_id
            )

        prompt = self.prompts.build_fraud_analysis_prompt(
            document_type=document_type,
            document_name=document_name,
            ocr_content=ocr_result,
            extracted_fields=extraction.extracted_fields,
            guide=guide._data,  # type: ignore[attr-defined]
            context=context or {},
        )

        start = datetime.now()
        try:
            response_text = await self._call_ai_with_retry(prompt, context_name=document_name)
            analysis = self._parse_analysis_response(
                response_text,
                document_id=document_id,
                document_name=document_name,
                document_type=document_type,
                case_id=case_id,
                guide=guide,
            )
            # Trazabilidad
            analysis.analysis_id = str(uuid.uuid4())
            analysis.prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

            analysis = await self._enrich_analysis(analysis, extraction, guide)
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
    ) -> List[FraudAnalysisResult]:
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
                )

        results = await asyncio.gather(*[_run(d) for d in documents], return_exceptions=True)
        out: List[FraudAnalysisResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                d = documents[i]
                logger.error(f"Error analizando {d['name']}: {r}")
                out.append(self._create_error_analysis(d['id'], d['name'], d['type'], case_id, str(r)))
            else:
                out.append(r)
        return out

    async def _call_ai_with_retry(self, prompt: str, context_name: str, max_retries: int = 2) -> str:
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
                return content
            except Exception as e:  # pragma: no cover - red restringida
                last = e
                await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Fallo en IA para {context_name}: {last}")

    def _parse_analysis_response(
        self,
        response_text: str,
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
        ind_list = data.get("indicators", []) or []
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
        provided_risk = str(data.get("risk_level", "")).strip().lower() or "medio"
        derived = self._derive_risk_level(score)
        risk_str = provided_risk if provided_risk in {"bajo", "medio", "alto", "critico"} else derived
        risk_level = RiskLevel(risk_str)

        # Evidencia y recomendaciones (a listas de strings)
        evidence = data.get("evidence", []) or []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        evidence = [str(e) for e in evidence]

        recommendations = data.get("recommendations", []) or []
        if not isinstance(recommendations, list):
            recommendations = [str(recommendations)]
        recommendations = [str(r) for r in recommendations]

        if risk_str != derived:
            evidence.append(
                f"Riesgo ajustado de '{risk_str}' a '{derived}' (score={score:.2f})"
            )

        return FraudAnalysisResult(
            document_id=document_id,
            document_name=document_name,
            document_type=document_type,
            case_id=case_id,
            risk_level=risk_level,
            fraud_score=score,
            confidence=_to_float(data.get("confidence", 0.7), 0.7),
            analisis_completo=str(data.get("analisis_completo", "")).strip(),
            indicators=indicators,
            evidence=evidence,
            recommendations=recommendations,
            analysis_model=self.model,
            guide_version=guide.version,
            processing_time_ms=0,
        )

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
                        risk_level, fraud_score, analisis_completo, indicators, evidence,
                        recommendations, confidence, analysis_model,
                        guide_version, analysis_uuid, prompt_hash,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        json.dumps(analysis.recommendations, ensure_ascii=False),
                        analysis.confidence,
                        analysis.analysis_model,
                        analysis.guide_version,
                        analysis.analysis_id,
                        analysis.prompt_hash,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error guardando análisis: {e}")
            # Intento de autocorrección si la tabla no existe
            try:
                if "no such table" in str(e).lower():
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
                                recommendations TEXT,
                                confidence      REAL,
                                analysis_model  TEXT,
                                guide_version   TEXT,
                                analysis_uuid   TEXT,
                                prompt_hash     TEXT,
                                created_at      TEXT NOT NULL,
                                updated_at      TEXT NOT NULL
                            );
                            """
                        )
                        conn.commit()
                        logger.info("Tabla fraud_analyses creada dinámicamente. Reintentando persistencia...")
                    # reintentar una vez
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
            recommendations=["Revisar manualmente el documento", "Reintentar análisis"],
            analysis_model="error",
            guide_version="N/A",
            processing_time_ms=0,
        )
