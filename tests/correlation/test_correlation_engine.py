import json
from pathlib import Path
from typing import Any, List, Optional

import pytest

from fraud_scorer.analyzers.correlation.models import (
    CaseContext,
    CorrelationFinding,
    FindingSeverity,
    FindingStatus,
)
from fraud_scorer.analyzers.correlation.engines.rule_engine import RuleEngine
from fraud_scorer.analyzers.correlation.engines.statistical_correlator import StatisticalCorrelator
from fraud_scorer.analyzers.correlation.engines.rag_evidence_builder import RAGEvidenceBuilder
from fraud_scorer.analyzers.correlation.orchestrator import CorrelationEngine
from fraud_scorer.models.extraction import (
    ConsolidatedExtraction,
    ConsolidatedFields,
    DocumentExtraction,
)
from fraud_scorer.models.fraud_analysis import (
    FraudAnalysisResult,
    FraudIndicator,
    RiskLevel,
)
from fraud_scorer.ai.orchestration.agente_rick import RickQueryResult


@pytest.fixture
def correlation_inputs(monkeypatch):
    def _build(
        monto_poliza: float,
        monto_factura: float,
        *,
        suma_asegurada: float | None = None,
        fecha_ocurrencia: str = "2025-01-10",
        fecha_reclamacion: str = "2025-01-15",
        extra_docs: Optional[List[DocumentExtraction]] = None,
    ):
        consolidated = ConsolidatedExtraction(
            case_id="CASE-1",
            consolidated_fields=ConsolidatedFields(
                monto_reclamacion=monto_poliza,
                suma_asegurada=suma_asegurada,
                fecha_ocurrencia=fecha_ocurrencia,
                fecha_reclamacion=fecha_reclamacion,
            ),
            consolidation_sources={},
            conflicts_resolved=[],
            confidence_scores={},
        )
        base_docs = [
            DocumentExtraction(
                source_document="factura.pdf",
                document_type="facturas_comerciales_internacionales",
                extracted_fields={"monto_total": monto_factura},
                extraction_metadata={},
            )
        ]
        if extra_docs:
            base_docs.extend(extra_docs)
        fraud_result = FraudAnalysisResult(
            document_id="doc-poliza",
            document_name="poliza.pdf",
            document_type="poliza_de_la_aseguradora",
            case_id="CASE-1",
            risk_level=RiskLevel.MEDIO,
            fraud_score=0.45,
            confidence=0.9,
            analisis_completo="",
            indicators=[
                FraudIndicator(
                    pattern="consistency_check",
                    description="Montos consistentes",
                    severity="medio",
                    confidence=0.8,
                )
            ],
            evidence=[],
            recommendations=[],
            analysis_model="test-model",
            guide_version="1.0",
        )
        monkeypatch.setattr(
            CaseContext,
            "_load_document_metadata",
            staticmethod(lambda case_id, cache_manager=None, case_index=None: ({}, {})),
        )
        context = CaseContext.from_case(
            case_id="CASE-1",
            consolidated=consolidated,
            extractions=base_docs,
            fraud_results=[fraud_result],
            case_index={},
        )
        return context, consolidated, base_docs, [fraud_result]

    return _build


@pytest.fixture
def multiple_invoices_context(monkeypatch):
    consolidated = ConsolidatedExtraction(
        case_id="CASE-OUTLIER",
        consolidated_fields=ConsolidatedFields(
            monto_reclamacion="120000",
            suma_asegurada="110000",
            fecha_ocurrencia="2025-01-01",
            fecha_reclamacion="2025-01-25",
            vigencia_fin="2025-01-20",
        ),
        consolidation_sources={},
        conflicts_resolved=[],
        confidence_scores={},
    )
    base_values = [10000, 11000, 11200, 50000]
    extractions = [
        DocumentExtraction(
            source_document=f"factura_{idx}.pdf",
            document_type="facturas_comerciales_internacionales",
            extracted_fields={
                "monto_total": value,
                "peso_total": value / 10 + idx * 3,
            },
            extraction_metadata={},
        )
        for idx, value in enumerate(base_values, start=1)
    ]
    fraud_result = FraudAnalysisResult(
        document_id="doc-poliza",
        document_name="poliza.pdf",
        document_type="poliza_de_la_aseguradora",
        case_id="CASE-OUTLIER",
        risk_level=RiskLevel.MEDIO,
        fraud_score=0.5,
        confidence=0.9,
        analisis_completo="",
        indicators=[],
        evidence=[],
        recommendations=[],
        analysis_model="test-model",
        guide_version="1.0",
    )
    monkeypatch.setattr(
        CaseContext,
        "_load_document_metadata",
        staticmethod(lambda case_id, cache_manager=None, case_index=None: ({}, {})),
    )
    context = CaseContext.from_case(
        case_id="CASE-OUTLIER",
        consolidated=consolidated,
        extractions=extractions,
        fraud_results=[fraud_result],
        case_index={},
    )
    return context, consolidated, extractions, [fraud_result]


@pytest.fixture
def rule_engine(tmp_path: Path) -> RuleEngine:
    rules_payload = {
        "meta": {"version": "vtest"},
        "rules": [
            {
                "id": "policy_vs_invoice_amount",
                "version": "1.0",
                "severity": "high",
                "description": "Montos entre póliza y facturas deben coincidir",
                "documents": [
                    "poliza_de_la_aseguradora",
                    "facturas_comerciales_internacionales",
                ],
                "entities": ["monto_reclamacion", "monto_total"],
                "condition": {
                    "type": "equality",
                    "source": "consolidated.consolidated_fields.monto_reclamacion",
                    "target": "aggregates.facturas_comerciales_internacionales.monto_total",
                    "tolerance": 0.01,
                },
                "on_fail": {
                    "recommendation": "Revisar montos con el área de siniestros",
                },
                "missing_summary": "Sin datos para comparar montos",
            },
            {
                "id": "claim_vs_sum_insured_ceiling",
                "version": "1.0",
                "severity": "critical",
                "description": "El monto reclamado no debe exceder la suma asegurada declarada",
                "documents": ["poliza_de_la_aseguradora"],
                "entities": ["monto_reclamacion", "suma_asegurada"],
                "condition": {
                    "type": "numeric_order",
                    "lhs": "consolidated.consolidated_fields.monto_reclamacion",
                    "rhs": "consolidated.consolidated_fields.suma_asegurada",
                    "operator": "<=",
                    "tolerance": 0.02,
                },
                "missing_summary": "Faltan montos de reclamación o suma asegurada",
            },
            {
                "id": "occurrence_vs_denuncia_date",
                "version": "1.0",
                "severity": "medium",
                "description": "La denuncia debe registrarse después del siniestro",
                "documents": [
                    "denuncia_de_los_hechos",
                    "poliza_de_la_aseguradora",
                ],
                "entities": ["fecha_ocurrencia", "fecha_denuncia"],
                "condition": {
                    "type": "temporal_order",
                    "earlier": "consolidated.consolidated_fields.fecha_ocurrencia",
                    "later": "documents.denuncia_de_los_hechos[0].extracted_fields.fecha_denuncia",
                    "allow_equal": False,
                },
                "missing_summary": "Faltan fechas de ocurrencia o denuncia",
            },
            {
                "id": "gps_window_covers_occurrence_start",
                "version": "1.0",
                "severity": "medium",
                "description": "El monitoreo GPS debe iniciar antes del siniestro",
                "documents": ["reporte_gps"],
                "entities": ["fecha_inicio", "fecha_ocurrencia"],
                "condition": {
                    "type": "temporal_order",
                    "earlier": "documents.reporte_gps[0].extracted_fields.fecha_inicio",
                    "later": "consolidated.consolidated_fields.fecha_ocurrencia",
                    "allow_equal": True,
                },
                "missing_summary": "Faltan fechas de GPS o siniestro",
            },
            {
                "id": "gps_window_covers_occurrence_end",
                "version": "1.0",
                "severity": "medium",
                "description": "El monitoreo GPS debe cubrir la fecha del siniestro",
                "documents": ["reporte_gps"],
                "entities": ["fecha_fin", "fecha_ocurrencia"],
                "condition": {
                    "type": "temporal_order",
                    "earlier": "consolidated.consolidated_fields.fecha_ocurrencia",
                    "later": "documents.reporte_gps[0].extracted_fields.fecha_fin",
                    "allow_equal": True,
                },
                "missing_summary": "Faltan fechas de GPS o siniestro",
            },
            {
                "id": "route_overlap_gps_vs_carta_porte",
                "version": "1.0",
                "severity": "high",
                "description": "Ruta GPS debe coincidir con Carta Porte",
                "documents": ["cfdi_carta_porte", "reporte_gps"],
                "entities": ["trayecto", "ruta_planeada"],
                "condition": {
                    "type": "set_overlap",
                    "source": "documents.reporte_gps[0].extracted_fields.trayecto",
                    "target": "documents.cfdi_carta_porte[0].extracted_fields.ruta_planeada",
                    "min_overlap": 1,
                },
                "missing_summary": "No hay datos de ruta",
            },
            {
                "id": "plate_overlap_carta_porte_investigacion",
                "version": "1.0",
                "severity": "high",
                "description": "Las placas deben coincidir entre Carta Porte y carpeta de investigación",
                "documents": ["cfdi_carta_porte", "carpeta_de_investigacion"],
                "entities": ["placas"],
                "condition": {
                    "type": "set_overlap",
                    "source": "documents.cfdi_carta_porte[0].extracted_fields.placas",
                    "target": "documents.carpeta_de_investigacion[0].extracted_fields.placas",
                    "min_overlap": 1,
                },
                "missing_summary": "No se localizaron placas para comparar",
            },
            {
                "id": "gps_trayecto_existe",
                "version": "1.0",
                "severity": "low",
                "description": "El reporte GPS debe contener trayecto registrado",
                "documents": ["reporte_gps"],
                "condition": {
                    "type": "exists",
                    "path": "documents.reporte_gps[0].extracted_fields.trayecto",
                },
                "missing_summary": "No se encontró trayecto en el reporte GPS",
            },
        ],
    }
    rules_path = tmp_path / "rules.yaml"
    entity_path = tmp_path / "entities.yaml"
    rules_path.write_text(json.dumps(rules_payload), encoding="utf-8")
    entity_data = {
        "version": "vtest",
        "field_aliases": {
            "monto_reclamo": {"canonical": "monto_reclamacion"},
            "numero_placas": {"canonical": "placas"},
        },
    }
    entity_path.write_text(json.dumps(entity_data), encoding="utf-8")
    return RuleEngine(rules_path=rules_path, entity_mappings_path=entity_path)



def test_case_context_builds_aggregates(correlation_inputs):
    context, _consolidated, _extractions, _fraud = correlation_inputs(100.0, 45.5)
    aggregates = context.aggregates["facturas_comerciales_internacionales"]
    assert pytest.approx(45.5) == aggregates["monto_total"]
    assert context.entities["monto_reclamacion"] == [100.0]

def test_case_context_normalizes_aliases(monkeypatch):
    monkeypatch.setattr(
        CaseContext,
        "_load_document_metadata",
        staticmethod(lambda case_id, cache_manager=None, case_index=None: ({}, {})),
    )
    alias_doc = DocumentExtraction(
        source_document="carta_porte_alias.pdf",
        document_type="carta_porte_simple",
        extracted_fields={"numero_placas": ["XYZ123"]},
        extraction_metadata={},
    )
    context = CaseContext.from_case(
        case_id="CASE-ALIAS",
        consolidated=None,
        extractions=[alias_doc],
        fraud_results=[],
        case_index={},
    )
    assert "cfdi_carta_porte" in context.documents_by_type
    alias_docs = context.resolve("documents.carta_porte_simple")
    assert isinstance(alias_docs, list) and alias_docs
    canonical_docs = context.resolve("documents.cfdi_carta_porte")
    assert canonical_docs[0]["document_type"] == "cfdi_carta_porte"
    assert canonical_docs[0]["extracted_fields"]["placas"] == ["XYZ123"]
    assert canonical_docs[0]["extracted_fields"]["numero_placas"] == ["XYZ123"]
    assert context.resolve("entities.placas") == [["XYZ123"]]
    assert context.resolve("entities.numero_placas") == [["XYZ123"]]

def test_numeric_order_rule_detects_overclaim(rule_engine: RuleEngine, correlation_inputs):
    context, *_ = correlation_inputs(150.0, 150.0, suma_asegurada=100.0)
    findings = rule_engine.evaluate(context)
    target = next((f for f in findings if f.rule_id == "claim_vs_sum_insured_ceiling"), None)
    assert target is not None
    assert target.status == FindingStatus.FAIL
    assert target.metadata.get("threshold")


def test_occurrence_vs_denuncia_sequence(rule_engine: RuleEngine, correlation_inputs):
    denuncia_doc = DocumentExtraction(
        source_document="denuncia.pdf",
        document_type="denuncia_de_los_hechos",
        extracted_fields={"fecha_denuncia": "2025-01-05"},
        extraction_metadata={},
    )
    context, *_ = correlation_inputs(120.0, 120.0, fecha_ocurrencia="2025-01-10", extra_docs=[denuncia_doc])
    findings = rule_engine.evaluate(context)
    target = next((f for f in findings if f.rule_id == "occurrence_vs_denuncia_date"), None)
    assert target is not None
    assert target.status == FindingStatus.FAIL


def test_gps_window_rules(rule_engine: RuleEngine, correlation_inputs):
    gps_doc = DocumentExtraction(
        source_document="gps.json",
        document_type="reporte_gps",
        extracted_fields={
            "fecha_inicio": "2025-01-08",
            "fecha_fin": "2025-01-12",
            "trayecto": ["origen", "punto_robo"],
        },
        extraction_metadata={},
    )
    carta_porte_doc = DocumentExtraction(
        source_document="carta_porte.xml",
        document_type="cfdi_carta_porte",
        extracted_fields={
            "ruta_planeada": ["origen", "destino"],
            "placas": ["XYZ123"]
        },
        extraction_metadata={},
    )
    carpeta_doc = DocumentExtraction(
        source_document="carpeta.pdf",
        document_type="carpeta_de_investigacion",
        extracted_fields={"placas": ["XYZ123"]},
        extraction_metadata={},
    )
    context, *_ = correlation_inputs(120.0, 120.0, fecha_ocurrencia="2025-01-10", extra_docs=[gps_doc, carta_porte_doc, carpeta_doc])
    findings = rule_engine.evaluate(context)
    gps_start = next((f for f in findings if f.rule_id == "gps_window_covers_occurrence_start"), None)
    gps_end = next((f for f in findings if f.rule_id == "gps_window_covers_occurrence_end"), None)
    route = next((f for f in findings if f.rule_id == "route_overlap_gps_vs_carta_porte"), None)
    plate = next((f for f in findings if f.rule_id == "plate_overlap_carta_porte_investigacion"), None)
    assert gps_start is not None and gps_start.status == FindingStatus.PASS
    assert gps_end is not None and gps_end.status == FindingStatus.PASS
    assert route is not None and route.status == FindingStatus.PASS
    assert plate is not None and plate.status == FindingStatus.PASS


def test_route_overlap_detects_deviation(rule_engine: RuleEngine, correlation_inputs):
    gps_doc = DocumentExtraction(
        source_document="gps.json",
        document_type="reporte_gps",
        extracted_fields={"trayecto": ["desvio_no_autorizado", "ruta_alterna"]},
        extraction_metadata={},
    )
    carta_doc = DocumentExtraction(
        source_document="carta.xml",
        document_type="cfdi_carta_porte",
        extracted_fields={"ruta_planeada": ["terminal_autorizada", "destino"]},
        extraction_metadata={},
    )
    context, *_ = correlation_inputs(100.0, 100.0, extra_docs=[gps_doc, carta_doc])
    findings = rule_engine.evaluate(context)
    target = next((f for f in findings if f.rule_id == "route_overlap_gps_vs_carta_porte"), None)
    assert target is not None
    assert target.status == FindingStatus.FAIL
    assert target.metadata.get("source_values")



def test_rule_engine_detects_mismatch(rule_engine: RuleEngine, correlation_inputs):
    context, *_ = correlation_inputs(100.0, 70.0)
    findings = rule_engine.evaluate(context)
    target = next((f for f in findings if f.rule_id == "policy_vs_invoice_amount"), None)
    assert target is not None
    assert target.status == FindingStatus.FAIL
    assert target.recommendation is not None


def test_correlation_engine_report_counts(rule_engine: RuleEngine, correlation_inputs):
    context, consolidated, extractions, fraud_results = correlation_inputs(150.0, 120.0)
    engine = CorrelationEngine(rule_engine=rule_engine)
    report = engine.run(
        case_id=context.case_id,
        consolidated=consolidated,
        extractions=extractions,
        fraud_results=fraud_results,
        case_index={},
        enable_rag=False,
    )
    assert report.case_id == context.case_id
    assert report.findings
    assert report.status_counts[FindingStatus.PASS] >= 0



def test_statistical_correlator_detects_outliers(tmp_path: Path, multiple_invoices_context):
    config = {
        "version": "vtest",
        "numeric_anomalies": [
            {
                "id": "invoices_outlier",
                "description": "Montos con comportamiento atípico",
                "document_type": "facturas_comerciales_internacionales",
                "field": "monto_total",
                "min_samples": 3,
                "max_zscore": 1.5,
                "severity": "medium",
            }
        ],
        "gap_anomalies": [],
    }
    cfg_path = tmp_path / "stat.yaml"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    correlator = StatisticalCorrelator(config_path=cfg_path)
    context, *_ = multiple_invoices_context
    findings = correlator.analyze(context)
    assert findings, "Se esperaba al menos un hallazgo estadístico"
    assert any(f.status == FindingStatus.FAIL for f in findings)
    assert findings[0].metadata.get("outliers")

def test_statistical_gap_rule_flags_violation(tmp_path: Path, correlation_inputs):
    config = {
        "version": "vtest",
        "numeric_anomalies": [],
        "gap_anomalies": [
            {
                "id": "delayed_claim_test",
                "description": "Reclamación demasiado tardía",
                "start": "consolidated.consolidated_fields.fecha_ocurrencia",
                "end": "consolidated.consolidated_fields.fecha_reclamacion",
                "max_days": 5,
                "severity": "medium",
            }
        ],
    }
    cfg_path = tmp_path / "gap.yaml"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    correlator = StatisticalCorrelator(config_path=cfg_path)
    context, consolidated, extractions, fraud_results = correlation_inputs(
        100.0,
        80.0,
        fecha_ocurrencia="2025-01-01",
        fecha_reclamacion="2025-02-20",
    )
    findings = correlator.analyze(context)
    target = next((f for f in findings if f.rule_id == "delayed_claim_test"), None)
    assert target is not None and target.status == FindingStatus.FAIL
    assert target.metadata.get("difference_days") > 5



def test_statistical_ratio_requires_context(tmp_path: Path, correlation_inputs):
    config = {
        "version": "vtest",
        "numeric_anomalies": [
            {
                "id": "ratio_rule",
                "description": "Monto vs número de póliza",
                "source_path": "consolidated.consolidated_fields.monto_reclamacion",
                "reference_path": "consolidated.consolidated_fields.numero_poliza",  # Este campo será None
                "max_ratio": 1.0,
                "severity": "critical",
            }
        ],
    }
    cfg_path = tmp_path / "stat_ratio.yaml"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    correlator = StatisticalCorrelator(config_path=cfg_path)
    context, consolidated, extractions, fraud_results = correlation_inputs(100.0, 50.0)
    # numero_poliza será None por defecto, lo que debe generar INSUFFICIENT_DATA
    findings = correlator.analyze(context)
    assert any(f.status == FindingStatus.INSUFFICIENT_DATA for f in findings)

def test_statistical_correlation_flags_high_alignment(tmp_path: Path, multiple_invoices_context):
    config = {
        "version": "vtest",
        "numeric_anomalies": [],
        "gap_anomalies": [],
        "correlation_checks": [
            {
                "id": "invoice_weight_correlation",
                "description": "Correlación inesperadamente alta entre monto y peso reportados",
                "method": "pearson",
                "pair_mode": "document",
                "left": {
                    "document_type": "facturas_comerciales_internacionales",
                    "field": "monto_total",
                },
                "right": {
                    "document_type": "facturas_comerciales_internacionales",
                    "field": "peso_total",
                },
                "min_samples": 3,
                "max_abs_correlation": 0.5,
                "expected_sign": "positive",
                "severity": "medium",
            }
        ],
    }
    cfg_path = tmp_path / "stat_corr.yaml"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    correlator = StatisticalCorrelator(config_path=cfg_path)
    context, *_ = multiple_invoices_context
    findings = correlator.analyze(context)
    target = next((f for f in findings if f.rule_id == "invoice_weight_correlation"), None)
    assert target is not None and target.status == FindingStatus.FAIL
    assert target.metadata.get("violations")
    assert target.metadata.get("sample_size") >= 3



def test_rag_evidence_builder_enriches(monkeypatch, correlation_inputs):
    context, consolidated, extractions, fraud_results = correlation_inputs(100.0, 50.0)
    finding = CorrelationFinding(
        id="finding-1",
        rule_id="sample_rule",
        rule_version="1.0",
        status=FindingStatus.NEEDS_CONTEXT,
        severity=FindingSeverity.MEDIUM,
        summary="Datos insuficientes",
        documents_involved=["poliza.pdf"],
        entities_involved=[],
        evidence=[],
        metadata={"source_value": 100, "target_value": None},
        tags=[],
    )

    captured_events: list[tuple[str, Optional[float]]] = []

    def _record(status: str, latency: Optional[float]) -> None:
        captured_events.append((status, latency))

    monkeypatch.setattr(
        "fraud_scorer.analyzers.correlation.engines.rag_evidence_builder.record_rag_event",
        _record,
    )

    class _FakeRick:
        def query(
            self,
            *,
            case_id: str,
            question: str,
            scope: str = "case",
            module: Optional[str] = None,
            **_: str,
        ) -> RickQueryResult:  # type: ignore[override]
            assert module == "correlation"
            return RickQueryResult(
                answer="Se confirma discrepancia entre póliza y facturas.",
                sources=[{"source_document": "poliza.pdf", "similarity": 0.91}],
                latency_ms=120,
            )

    builder = RAGEvidenceBuilder(enabled=True, service=_FakeRick())
    builder.build([finding], case_id="CASE-1", context=context)

    assert finding.status == FindingStatus.FAIL
    assert finding.evidence
    rag_meta = finding.metadata.get("rag", {})
    assert rag_meta.get("status") == "answered"
    assert rag_meta.get("module") == "correlation"
    assert captured_events and captured_events[0][0] == "answered"
    assert captured_events[0][1] == 120
