from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from fraud_scorer.analyzers.correlation.orchestrator import CorrelationEngine
from fraud_scorer.analyzers.correlation.models.correlation_result import CorrelationReport
from fraud_scorer.analyzers.fraud_analyzer import FraudAnalyzer
from fraud_scorer.analyzers.unified_data_layer import UnifiedDataLayer
from fraud_scorer.analyzers.fraud_guide_manager import FraudGuideManager
from fraud_scorer.models.extraction import (
    ConsolidatedExtraction,
    ConsolidatedFields,
    DocumentExtraction,
)
from fraud_scorer.models.fraud_analysis import EvidenceGap, FraudAnalysisResult, RiskLevel
from fraud_scorer.storage.db import get_conn, save_correlation_findings
from fraud_scorer.storage.ocr_cache import OCRCacheManager
from fraud_scorer.templates.fraud_report_generator import FraudReportGenerator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _jload(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        if isinstance(value, (dict, list)):
            return value
        return json.loads(value)
    except Exception:
        return default


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sanitize_filename(value: Optional[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (value or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "SIN_NOMBRE"


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


@dataclass
class FraudDocumentContext:
    document_id: str
    document_name: str
    document_type: str
    ocr_result: Dict[str, Any]
    extraction: DocumentExtraction
    include_in_report: bool


@dataclass
class ReportArtifacts:
    html_path: Optional[Path]
    pdf_path: Optional[Path]


class FraudDocumentCatalog:
    """Helpers to read and persist fraud analyses for editor use."""

    def __init__(self, cache_manager: Optional[OCRCacheManager] = None) -> None:
        self.cache_manager = cache_manager or OCRCacheManager()

    def load_case_index(self, case_id: str) -> Dict[str, Any]:
        case_index = self.cache_manager.get_case_index(case_id, auto_reconstruct=True)
        if not case_index:
            raise ValueError("Caso no encontrado")
        return case_index

    def list_results(self, case_id: str) -> Tuple[List[FraudAnalysisResult], Dict[str, Any]]:
        case_index = self.load_case_index(case_id)
        results = self._hydrate_from_case_index(case_index)
        if not results:
            results = self._hydrate_from_db(case_id)
        meta = self._load_db_metadata(case_id)
        for result in results:
            db_meta = meta.get(result.document_id)
            if not db_meta:
                continue
            include_flag = db_meta.get("include_in_report")
            if include_flag is not None:
                result.include_in_report = bool(include_flag)
        return results, case_index

    def build_preview(self, case_id: str) -> List[Dict[str, Any]]:
        results, _ = self.list_results(case_id)
        return self.build_preview_from_results(results)

    def get_manual_report_html(self, case_id: str) -> Optional[str]:
        case_index = self.load_case_index(case_id)
        html = case_index.get("report_override_html")
        if isinstance(html, str) and html.strip():
            return html
        return None

    def set_manual_report_html(self, case_id: str, html_content: str) -> str:
        sanitized = self._sanitize_html_override(html_content)
        case_index = self.load_case_index(case_id)
        case_index.pop("report_overrides", None)
        case_index["report_override_html"] = sanitized
        case_index["report_override_updated_at"] = _iso_now()
        case_index["updated_at"] = _iso_now()
        self.cache_manager.save_case_index(case_id, case_index)
        return sanitized

    def clear_manual_report_html(self, case_id: str) -> None:
        case_index = self.load_case_index(case_id)
        if "report_override_html" in case_index:
            case_index.pop("report_override_html", None)
            case_index.pop("report_override_updated_at", None)
            case_index.pop("report_overrides", None)
            case_index["updated_at"] = _iso_now()
            self.cache_manager.save_case_index(case_id, case_index)

    def build_catalog(self, case_id: str) -> Dict[str, Any]:
        results, _ = self.list_results(case_id)
        metadata = self._load_db_metadata(case_id)
        documents: List[Dict[str, Any]] = []
        for result in results:
            meta = metadata.get(result.document_id, {})
            documents.append(
                {
                    "analysis": result.model_dump(),
                    "metadata": {
                        "last_run": meta.get("updated_at") or meta.get("created_at"),
                        "analysis_uuid": meta.get("analysis_uuid"),
                        "include_in_report": result.include_in_report,
                        "processing_time_ms": result.processing_time_ms,
                    },
                }
            )
        return {
            "case_id": case_id,
            "count": len(documents),
            "documents": documents,
            "preview": self.build_preview_from_results(results),
            "generated_at": _iso_now(),
        }

    def update_visibility(
        self, case_id: str, document_id: str, include: bool
    ) -> Tuple[FraudAnalysisResult, List[Dict[str, Any]]]:
        results, case_index = self.list_results(case_id)
        target: Optional[FraudAnalysisResult] = None
        for result in results:
            if result.document_id == document_id:
                result.include_in_report = include
                target = result
                break
        if target is None:
            raise LookupError("Documento no encontrado")

        updated_at = self._update_db_visibility(case_id, document_id, include)
        case_index["fraud_analyses"] = [res.model_dump() for res in results]
        case_index["updated_at"] = updated_at
        self.cache_manager.save_case_index(case_id, case_index)
        return target, self.build_preview_from_results(results)

    def build_preview_from_results(
        self, results: Sequence[FraudAnalysisResult]
    ) -> List[Dict[str, Any]]:
        preview: List[Dict[str, Any]] = []
        for res in results:
            preview.append(
                {
                    "document_id": res.document_id,
                    "document_name": res.document_name,
                    "document_type": res.document_type,
                    "risk_level": res.risk_level.value,
                    "fraud_score": res.fraud_score,
                    "include_in_report": res.include_in_report,
                }
            )
        return preview

    def _update_db_visibility(self, case_id: str, document_id: str, include: bool) -> str:
        now = _iso_now()
        with get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE fraud_analyses
                   SET include_in_report = ?, updated_at = ?
                 WHERE case_id = ? AND document_id = ?
                """,
                (1 if include else 0, now, case_id, document_id),
            )
            if cur.rowcount == 0:
                raise LookupError("Documento no encontrado en base de datos")
            conn.commit()
        return now

    def _hydrate_from_case_index(self, case_index: Dict[str, Any]) -> List[FraudAnalysisResult]:
        hydrated: List[FraudAnalysisResult] = []
        raw_items = case_index.get("fraud_analyses") or []
        for item in raw_items:
            if isinstance(item, FraudAnalysisResult):
                hydrated.append(item)
                continue
            if isinstance(item, dict):
                payload = dict(item)
                payload.pop("evidence", None)
                payload.pop("evidence_gaps", None)
                payload.setdefault("include_in_report", True)
                try:
                    hydrated.append(FraudAnalysisResult.model_validate(payload))
                except Exception as exc:
                    logger.warning("No se pudo rehidratar análisis de fraude: %s", exc)
        return hydrated

    def _hydrate_from_db(self, case_id: str) -> List[FraudAnalysisResult]:
        results: List[FraudAnalysisResult] = []
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT fa.document_id,
                       fa.document_type,
                       fa.case_id,
                       fa.risk_level,
                       fa.fraud_score,
                       fa.analisis_completo,
                       fa.indicators,
                       COALESCE(fa.evidence_gaps, '[]') AS evidence_gaps,
                       fa.recommendations,
                       COALESCE(fa.verificaciones, '{}') AS verificaciones,
                       COALESCE(fa.validacion_cruzada, '{}') AS validacion_cruzada,
                       fa.confidence,
                       fa.analysis_model,
                       fa.guide_version,
                       fa.analysis_uuid,
                       fa.prompt_hash,
                       fa.include_in_report,
                       fa.created_at,
                       fa.updated_at,
                       d.filename AS document_name
                  FROM fraud_analyses fa
             LEFT JOIN documents d ON d.id = fa.document_id
                 WHERE fa.case_id = ?
              ORDER BY fa.created_at ASC
                """,
                (case_id,),
            ).fetchall()
        for row in rows:
            try:
                risk_value = str(row["risk_level"] or "medio").lower()
                risk_enum = RiskLevel(risk_value) if risk_value in RiskLevel._value2member_map_ else RiskLevel.MEDIO
                indicators = _jload(row["indicators"], [])
                gaps: List[EvidenceGap] = []
                recommendations = _jload(row["recommendations"], [])
                verificaciones = _jload(row.get("verificaciones"), {})
                if not isinstance(verificaciones, dict):
                    try:
                        verificaciones = dict(verificaciones)
                    except Exception:
                        verificaciones = {}
                validacion_cruzada = _jload(row.get("validacion_cruzada"), {})
                if not isinstance(validacion_cruzada, dict):
                    try:
                        validacion_cruzada = dict(validacion_cruzada)
                    except Exception:
                        validacion_cruzada = {}
                timestamp = _parse_timestamp(row["updated_at"]) or _parse_timestamp(row["created_at"]) or datetime.now()
                include_raw = row["include_in_report"]
                include_flag = True if include_raw is None else bool(include_raw)

                result = FraudAnalysisResult(
                    document_id=row["document_id"],
                    document_name=row.get("document_name") or row["document_id"],
                    document_type=row["document_type"] or "otro",
                    case_id=row["case_id"],
                    analysis_id=row["analysis_uuid"],
                    prompt_hash=row["prompt_hash"],
                    risk_level=risk_enum,
                    fraud_score=float(row["fraud_score"] or 0.0),
                    confidence=float(row["confidence"] or 0.0),
                    analisis_completo=row["analisis_completo"] or "",
                    indicators=indicators,
                    evidence_gaps=gaps,
                    recommendations=recommendations,
                    verificaciones=verificaciones,
                    validacion_cruzada=validacion_cruzada,
                    analysis_model=row["analysis_model"] or "unknown",
                    guide_version=row["guide_version"] or "N/A",
                    processing_time_ms=0,
                    timestamp=timestamp,
                    include_in_report=include_flag,
                )
                results.append(result)
            except Exception as exc:
                logger.warning("No se pudo reconstruir análisis desde DB: %s", exc)
        return results

    def _load_db_metadata(self, case_id: str) -> Dict[str, Dict[str, Any]]:
        payload: Dict[str, Dict[str, Any]] = {}
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT document_id, include_in_report, analysis_uuid, created_at, updated_at
                  FROM fraud_analyses
                 WHERE case_id = ?
                """,
                (case_id,),
            ).fetchall()
        for row in rows:
            include_raw = row["include_in_report"]
            payload[row["document_id"]] = {
                "include_in_report": True if include_raw is None else bool(include_raw),
                "analysis_uuid": row["analysis_uuid"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        return payload

    def _sanitize_html_override(self, content: str) -> str:
        cleaned = str(content or "")
        try:
            cleaned = re.sub(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r"on[a-zA-Z]+\s*=\s*'[^']*'", "", cleaned)
            cleaned = re.sub(r"on[a-zA-Z]+\s*=\s*\"[^\"]*\"", "", cleaned)
            cleaned = re.sub(r"javascript:\s*", "", cleaned, flags=re.IGNORECASE)
        except Exception:
            pass
        return cleaned


class FraudDocumentReprocessService:
    """Runs fraud document re-analysis, correlation refresh and report regeneration."""

    def __init__(
        self,
        catalog: FraudDocumentCatalog,
        reports_dir: Path,
        templates_dir: Path,
    ) -> None:
        self.catalog = catalog
        self.cache_manager = catalog.cache_manager
        self.reports_dir = reports_dir
        self.templates_dir = templates_dir
        self.guide_manager = FraudGuideManager()
        self.correlation_engine = CorrelationEngine()
        self._docless_system = None

    async def reprocess(
        self,
        case_id: str,
        document_id: str,
        *,
        progress_callback: Optional[Callable[[str, str, int], None]] = None,
    ) -> Tuple[FraudAnalysisResult, ReportArtifacts, List[Dict[str, Any]]]:
        results, case_index = self.catalog.list_results(case_id)
        current_map = {res.document_id: res for res in results}
        previous = current_map.get(document_id)
        include_flag = previous.include_in_report if previous else True

        context = self._build_document_context(case_id, document_id, case_index, include_flag)
        if progress_callback:
            progress_callback("re-analyzing", "Reanalizando documento…", 35)

        analyzer = FraudAnalyzer()
        data_layer = UnifiedDataLayer.from_case_index(case_index)
        analysis = await analyzer.analyze_document(
            document_id=context.document_id,
            document_name=context.document_name,
            document_type=context.document_type,
            ocr_result=context.ocr_result,
            extraction=context.extraction,
            case_id=case_id,
            context=self._build_analysis_context(case_index, data_layer=data_layer),
            data_layer=data_layer,
        )

        if include_flag is not None:
            analysis.include_in_report = include_flag
            self.catalog._update_db_visibility(case_id, document_id, include_flag)

        updated_results = self._merge_results(results, analysis)
        case_index["fraud_analyses"] = [res.model_dump() for res in updated_results]
        case_index["updated_at"] = _iso_now()

        corr_report = None
        if progress_callback:
            progress_callback("refreshing-report", "Actualizando correlaciones…", 65)
        corr_report = self._refresh_correlations(
            case_id, case_index, updated_results, context.extraction
        )
        if corr_report:
            case_index["fraud_correlations"] = corr_report.model_dump()
            try:
                save_correlation_findings(case_id, corr_report.findings)
            except Exception as exc:
                logger.warning("No se pudieron persistir correlaciones: %s", exc)

        if progress_callback:
            progress_callback("refreshing-report", "Regenerando reporte…", 80)
        artifacts = self._regenerate_report(
            case_id,
            case_index,
            updated_results,
            corr_report,
            manual_html=self.catalog.get_manual_report_html(case_id),
        )
        if artifacts.html_path:
            case_index["report_path"] = str(artifacts.html_path)
        if artifacts.pdf_path:
            case_index["pdf_path"] = str(artifacts.pdf_path)

        self.cache_manager.save_case_index(case_id, case_index)
        preview = self.catalog.build_preview_from_results(updated_results)
        return analysis, artifacts, preview

    def refresh_report(self, case_id: str) -> Tuple[ReportArtifacts, List[Dict[str, Any]]]:
        """Regenerate HTML/PDF after visibility changes without re-analysing."""
        results, case_index = self.catalog.list_results(case_id)
        case_index["fraud_analyses"] = [res.model_dump() for res in results]
        case_index["updated_at"] = _iso_now()

        corr_report = None
        raw_corr = case_index.get("fraud_correlations")
        if raw_corr:
            try:
                corr_report = (
                    raw_corr
                    if isinstance(raw_corr, CorrelationReport)
                    else CorrelationReport.model_validate(raw_corr)
                )
            except Exception as exc:
                logger.warning("No se pudo reconstruir correlación previa para %s: %s", case_id, exc)

        artifacts = self._regenerate_report(
            case_id,
            case_index,
            results,
            corr_report,
            generate_pdf=False,
            manual_html=self.catalog.get_manual_report_html(case_id),
        )
        if artifacts.html_path:
            case_index["report_path"] = str(artifacts.html_path)
        if artifacts.pdf_path:
            case_index["pdf_path"] = str(artifacts.pdf_path)

        self.cache_manager.save_case_index(case_id, case_index)
        preview = self.catalog.build_preview_from_results(results)
        return artifacts, preview

    def _merge_results(
        self,
        existing: Iterable[FraudAnalysisResult],
        updated: FraudAnalysisResult,
    ) -> List[FraudAnalysisResult]:
        out: List[FraudAnalysisResult] = []
        replaced = False
        for res in existing:
            if res.document_id == updated.document_id:
                out.append(updated)
                replaced = True
            else:
                out.append(res)
        if not replaced:
            out.append(updated)
        return out

    def _build_document_context(
        self,
        case_id: str,
        document_id: str,
        case_index: Dict[str, Any],
        include_flag: bool,
    ) -> FraudDocumentContext:
        with get_conn() as conn:
            row = conn.execute(
                """
                SELECT d.id,
                       d.filename,
                       d.filepath,
                       d.language,
                       d.page_count,
                       e.document_type,
                       e.entities AS e_entities,
                       e.key_value_pairs AS e_kv,
                       e.extra,
                       o.raw_text,
                       o.key_value_pairs,
                       o.tables,
                       o.entities,
                       o.confidence,
                       o.metadata,
                       o.errors
                  FROM documents d
             LEFT JOIN extracted_data e ON e.document_id = d.id
             LEFT JOIN ocr_results o ON o.document_id = d.id
                 WHERE d.id = ? AND d.case_id = ?
                """,
                (document_id, case_id),
            ).fetchone()
        if not row:
            raise LookupError("Documento no encontrado en la base de datos")

        document_name = row["filename"] or document_id
        ocr_result = {
            "text": row["raw_text"] or "",
            "tables": _jload(row["tables"], []),
            "entities": _jload(row["entities"], []),
            "key_value_pairs": _jload(row["key_value_pairs"], {}),
            "confidence_scores": _jload(row["confidence"], {}),
            "metadata": _jload(row["metadata"], {}),
            "errors": _jload(row["errors"], []),
            "page_count": row["page_count"],
            "language": row["language"],
        }

        if not ocr_result.get("text"):
            docless = self._load_docless_payload(case_id, case_index, document_name)
            if docless:
                ocr_result = docless

        raw_fields = _jload(row["e_kv"], {})
        extracted_fields = raw_fields if isinstance(raw_fields, dict) else {}
        raw_entities = _jload(row["e_entities"], {})
        extracted_entities = raw_entities if isinstance(raw_entities, dict) else {}
        raw_extra = _jload(row["extra"], {})
        extra_payload = raw_extra if isinstance(raw_extra, dict) else {}
        document_type = self._resolve_document_type(
            row["document_type"], case_index, document_name
        )
        merged_fields = dict(extracted_fields)
        if extra_payload:
            merged_fields.update(extra_payload)
        extraction = DocumentExtraction(
            source_document=document_name,
            document_type=document_type or "otro",
            extracted_fields=merged_fields,
            extraction_metadata={"entities": extracted_entities},
        )
        return FraudDocumentContext(
            document_id=document_id,
            document_name=document_name,
            document_type=document_type or "otro",
            ocr_result=ocr_result,
            extraction=extraction,
            include_in_report=include_flag,
        )

    def _resolve_document_type(
        self,
        fallback_type: Optional[str],
        case_index: Dict[str, Any],
        document_name: str,
    ) -> str:
        manual = case_index.get("manual_classifications") or {}
        if isinstance(manual, dict):
            override = manual.get(document_name)
            if override:
                return self.guide_manager.normalize_type(str(override))
        classified = case_index.get("classified_types") or []
        for item in classified:
            if not isinstance(item, dict):
                continue
            if str(item.get("filename")) == document_name and item.get("document_type"):
                return self.guide_manager.normalize_type(str(item.get("document_type")))
        if fallback_type:
            return self.guide_manager.normalize_type(str(fallback_type))
        return "otro"

    def _load_docless_payload(
        self,
        case_id: str,
        case_index: Dict[str, Any],
        document_name: str,
    ) -> Optional[Dict[str, Any]]:
        helper = self._ensure_docless_helper()
        try:
            base_folder = self.cache_manager.get_case_folder_path(case_id, case_index)
            payload = helper._prepare_docless_ocr(case_id, case_index, base_folder)
            for item in payload.get("ocr_results", []):
                if item.get("filename") == document_name:
                    return item.get("ocr_result") or {}
        except Exception as exc:
            logger.warning("No fue posible reconstruir OCR docless: %s", exc)
        return None

    def _ensure_docless_helper(self):
        if self._docless_system is not None:
            return self._docless_system
        scripts_path = PROJECT_ROOT / "scripts"
        if str(scripts_path) not in sys.path:
            sys.path.insert(0, str(scripts_path))
        spec = importlib.util.spec_from_file_location("run_report", scripts_path / "run_report.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        system = module.FraudAnalysisSystemV2()
        system.cache_manager = self.cache_manager
        self._docless_system = system
        return self._docless_system

    def _build_analysis_context(
        self,
        case_index: Dict[str, Any],
        *,
        data_layer: Optional[UnifiedDataLayer] = None,
    ) -> Dict[str, Any]:
        layer = data_layer or UnifiedDataLayer.from_case_index(case_index)
        return layer.build_case_context()

    def _refresh_correlations(
        self,
        case_id: str,
        case_index: Dict[str, Any],
        fraud_results: Sequence[FraudAnalysisResult],
        updated_extraction: DocumentExtraction,
    ):
        consolidated_raw = case_index.get("consolidated_data") or {}
        consolidated = self._build_consolidated(consolidated_raw, case_id)

        raw_extractions = case_index.get("extraction_results") or []
        extractions: List[DocumentExtraction] = []
        seen: set[str] = set()
        for item in raw_extractions:
            extraction = None
            if isinstance(item, DocumentExtraction):
                extraction = item
            elif isinstance(item, dict):
                payload = dict(item)
                payload.setdefault("source_document", payload.get("document_name") or payload.get("file_name"))
                payload.setdefault("document_type", payload.get("document_type") or "otro")
                try:
                    extraction = DocumentExtraction.model_validate(payload)
                except Exception:
                    try:
                        extraction = DocumentExtraction(
                            source_document=str(payload.get("source_document") or "documento"),
                            document_type=str(payload.get("document_type") or "otro"),
                            extracted_fields=payload.get("extracted_fields") or {},
                            extraction_metadata=payload.get("extraction_metadata") or {},
                        )
                    except Exception:
                        extraction = None
            if extraction:
                seen.add(extraction.source_document)
                extractions.append(extraction)
        if updated_extraction.source_document not in seen:
            extractions.append(updated_extraction)
        else:
            extractions = [
                updated_extraction if ex.source_document == updated_extraction.source_document else ex
                for ex in extractions
            ]

        try:
            return self.correlation_engine.run(
                case_id=case_id,
                consolidated=consolidated,
                extractions=extractions,
                fraud_results=fraud_results,
                case_index=case_index,
                cache_manager=self.cache_manager,
            )
        except Exception as exc:
            logger.warning("Correlación degradada para %s: %s", case_id, exc)
            return None

    def _build_consolidated(self, raw: Dict[str, Any], case_id: str) -> ConsolidatedExtraction:
        fields_raw = raw.get("consolidated_fields") or {}
        try:
            consolidated = ConsolidatedExtraction.model_validate(
                {
                    "case_id": raw.get("case_id") or case_id,
                    "consolidated_fields": fields_raw,
                    "consolidation_sources": raw.get("consolidation_sources") or {},
                    "conflicts_resolved": raw.get("conflicts_resolved") or [],
                    "confidence_scores": raw.get("confidence_scores") or {},
                }
            )
            return consolidated
        except Exception:
            fields = ConsolidatedFields(**fields_raw) if isinstance(fields_raw, dict) else ConsolidatedFields()
            return ConsolidatedExtraction(
                case_id=case_id,
                consolidated_fields=fields,
                consolidation_sources=raw.get("consolidation_sources") or {},
                conflicts_resolved=raw.get("conflicts_resolved") or [],
                confidence_scores=raw.get("confidence_scores") or {},
            )

    def _regenerate_report(
        self,
        case_id: str,
        case_index: Dict[str, Any],
        fraud_results: Sequence[FraudAnalysisResult],
        correlation_report,
        *,
        generate_pdf: bool = True,
        manual_html: Optional[str] = None,
    ) -> ReportArtifacts:
        fields = (case_index.get("consolidated_data") or {}).get("consolidated_fields") or {}
        insured = fields.get("nombre_asegurado") or case_index.get("insured_name") or "SIN_NOMBRE"
        claim = fields.get("numero_siniestro") or case_index.get("claim_number") or case_id
        s_insured = _sanitize_filename(str(insured))
        s_claim = _sanitize_filename(str(claim))

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        html_path = self.reports_dir / f"{s_insured}_{s_claim}_INFORME.html"
        pdf_path = html_path.with_suffix(".pdf")
        try:
            if html_path.exists():
                html_path.unlink()
        except Exception:
            pass
        if generate_pdf:
            try:
                if pdf_path.exists():
                    pdf_path.unlink()
            except Exception:
                pass

        fraud_gen = FraudReportGenerator(template_dir=self.templates_dir)
        docs_meta = [
            {"name": res.document_name, "type": res.document_type}
            for res in fraud_results
        ]
        consolidated = self._build_consolidated(case_index.get("consolidated_data") or {}, case_id)
        report_payload = fraud_gen.prepare_fraud_report_data(
            consolidated_data=consolidated,
            fraud_analyses=list(fraud_results),
            documents_metadata=docs_meta,
            correlation_report=correlation_report,
        )
        auto_html_content = fraud_gen.render_html_template("report_template.html", report_payload)
        final_html = manual_html if manual_html else auto_html_content
        html_path.write_text(final_html, encoding="utf-8")

        if generate_pdf:
            try:
                fraud_gen.generate_pdf(final_html, pdf_path)
            except Exception as exc:
                logger.warning("No se pudo generar PDF actualizado: %s", exc)
                pdf_path = None
        else:
            if not pdf_path.exists():
                pdf_path = None

        return ReportArtifacts(html_path=html_path, pdf_path=pdf_path)
