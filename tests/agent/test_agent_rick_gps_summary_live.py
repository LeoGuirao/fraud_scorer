from __future__ import annotations

import os

import pytest

from fraud_scorer.ai.config import load_config
from fraud_scorer.ai.orchestration import AgenteRickService

CASE_ID = "CASE-2025-0001"


requires_live = pytest.mark.skipif(
    os.getenv("AGENT_RICK_LIVE_TEST") != "1",
    reason="Set AGENT_RICK_LIVE_TEST=1 to run live Rick agent integration tests.",
)


@requires_live
def test_agent_rick_live_gps_summary():
    """Smoke test manual: requiere red y credenciales (ver guía Better Practices sección GPS)."""
    overrides = {
        "agent_mode_enabled": True,
        "agent_verbose": False,
        "max_results": 25,
        "agent_max_iterations": 4,
    }
    config = load_config(overrides=overrides)
    service = AgenteRickService(config=config)

    service.index_case(CASE_ID, rebuild=False)

    question = (
        "Rick, valida con el monitoreo GPS del CASE-2025-0001 cómo se corroboran las coordenadas del robo, "
        "los peajes recorridos, la distancia acumulada y la pausa nocturna."
    )

    result = service.query(case_id=CASE_ID, question=question)

    assert result.answer, "El agente no generó respuesta."
    assert service._agent_last_gps_queries, "El agente no invocó query_gps_location."
    assert any(source.get("document_type") == "gps_dataset" for source in result.sources), (
        "La respuesta no incluyó fuentes GPS."
    )

    print("\n=== Respuesta del Agente Rick ===\n")
    print(result.answer.strip())

    print("\n=== Consultas GPS Ejecutadas ===\n")
    for idx, query in enumerate(service._agent_last_gps_queries, start=1):
        print(f"Consulta #{idx}:")
        print(query)
        print()

    assert any(query.get("row_count", 0) > 0 for query in service._agent_last_gps_queries), (
        "Las consultas GPS ejecutadas no devolvieron filas; revisa filtros start/end o document_name."
    )

    answer_lower = result.answer.lower()
    assert "gps" in answer_lower or "coorden" in answer_lower, "La respuesta no hace referencia explícita al GPS."
    assert "km" in answer_lower or "distancia" in answer_lower, "No se mencionan distancias en la respuesta."
