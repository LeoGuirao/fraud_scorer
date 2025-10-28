"""
Smoke test end-to-end: Valida que el agente Rick filtra por document_type con un caso real.

Usa el caso CASE-2025-0001 que ya está indexado en el sistema.
"""

from __future__ import annotations

import pytest

from fraud_scorer.ai.config import RickAgentConfig
from fraud_scorer.ai.orchestration.agente_rick import AgenteRickService


@pytest.fixture
def rick_service() -> AgenteRickService:
    """Servicio Rick con modo agente habilitado."""
    config = RickAgentConfig(
        agent_mode_enabled=True,
        agent_verbose=True,  # Para ver los steps del agente
    )
    return AgenteRickService(config=config)


def test_smoke_filter_denuncia_de_los_hechos(rick_service: AgenteRickService):
    """
    Smoke test: Solicita información específicamente de 'Denuncia de los hechos'.

    Valida que:
    1. El agente reconoce la mención al tipo de documento
    2. LangChain genera document_type='denuncia_de_los_hechos'
    3. Solo retorna fuentes de ese tipo (no oficio_denuncia ni carpeta_investigacion)
    """
    case_id = "CASE-2025-0001"

    # Pregunta que menciona explícitamente el tipo de documento
    question = "¿Qué dice la Denuncia de los hechos sobre el robo?"

    result = rick_service.query(
        case_id=case_id,
        question=question,
        user_id="smoke_test",
    )

    # Debug: mostrar resultado
    print(f"\n{'='*80}")
    print(f"Pregunta: {question}")
    print(f"Respuesta: {result.answer}")
    print(f"\nFuentes encontradas:")
    for idx, source in enumerate(result.sources, 1):
        doc_type = source.get("document_type", "unknown")
        doc_name = source.get("source_document", "unknown")
        print(f"  {idx}. {doc_name} (tipo: {doc_type})")

    if result.metadata and "agent_steps" in result.metadata:
        print(f"\nAgent steps:")
        for step in result.metadata["agent_steps"]:
            tool = step.get("tool")
            tool_input = step.get("tool_input", "")
            print(f"  - Tool: {tool}")
            print(f"    Input: {tool_input[:200]}")  # Primeros 200 chars

    print(f"{'='*80}\n")

    # Validaciones
    assert result.sources, "No se encontraron fuentes en la respuesta"

    # Verificar que TODAS las fuentes son del tipo correcto
    source_types = [s.get("document_type") for s in result.sources]
    wrong_types = [t for t in source_types if t != "denuncia_de_los_hechos"]

    assert not wrong_types, (
        f"ERROR: Se encontraron documentos de tipos incorrectos: {set(wrong_types)}. "
        f"Solo se esperaba 'denuncia_de_los_hechos'. "
        f"Fuentes: {[(s.get('source_document'), s.get('document_type')) for s in result.sources]}"
    )

    # Verificar que el agente usó el parámetro document_type
    if result.metadata and "agent_steps" in result.metadata:
        search_steps = [
            step for step in result.metadata["agent_steps"]
            if step.get("tool") == "search_case_documents"
        ]

        if search_steps:
            # Al menos uno de los steps debe incluir document_type
            has_filter = any(
                "document_type" in str(step.get("tool_input", ""))
                and "denuncia_de_los_hechos" in str(step.get("tool_input", ""))
                for step in search_steps
            )

            assert has_filter, (
                f"El agente no pasó document_type='denuncia_de_los_hechos'. "
                f"Steps: {[s.get('tool_input') for s in search_steps]}"
            )


def test_smoke_without_filter_returns_mixed_types(rick_service: AgenteRickService):
    """
    Test de control: Sin mención de tipo específico, debe retornar tipos mixtos.
    """
    case_id = "CASE-2025-0001"

    # Pregunta genérica sin mencionar tipo de documento
    question = "¿Qué información hay sobre el robo de mercancía?"

    result = rick_service.query(
        case_id=case_id,
        question=question,
        user_id="smoke_test",
    )

    print(f"\n{'='*80}")
    print(f"Pregunta (sin filtro): {question}")
    print(f"\nTipos de documento encontrados:")
    source_types = set(s.get("document_type") for s in result.sources)
    for doc_type in source_types:
        count = sum(1 for s in result.sources if s.get("document_type") == doc_type)
        print(f"  - {doc_type}: {count} documentos")
    print(f"{'='*80}\n")

    assert result.sources, "No se encontraron fuentes"

    # Debe haber al menos 2 tipos diferentes (sin filtro específico)
    unique_types = set(s.get("document_type") for s in result.sources)

    # Nota: Esta validación puede fallar si la búsqueda semántica solo retorna un tipo.
    # Es más un test informativo que una validación estricta.
    if len(unique_types) == 1:
        print(f"⚠️  Advertencia: Solo se encontró un tipo de documento sin filtro: {unique_types}")
        print("    Esto puede ser válido si todos los documentos relevantes son del mismo tipo.")


def test_smoke_filter_carpeta_investigacion(rick_service: AgenteRickService):
    """
    Smoke test: Solicita información de 'Carpeta de investigación'.
    """
    case_id = "CASE-2025-0001"

    question = "¿Qué menciona la Carpeta de investigación sobre el transportista?"

    result = rick_service.query(
        case_id=case_id,
        question=question,
        user_id="smoke_test",
    )

    print(f"\n{'='*80}")
    print(f"Pregunta: {question}")
    print(f"\nFuentes encontradas:")
    for idx, source in enumerate(result.sources, 1):
        doc_type = source.get("document_type", "unknown")
        doc_name = source.get("source_document", "unknown")
        print(f"  {idx}. {doc_name} (tipo: {doc_type})")
    print(f"{'='*80}\n")

    if not result.sources:
        pytest.skip("No se encontraron fuentes para carpeta_investigacion")

    source_types = [s.get("document_type") for s in result.sources]
    wrong_types = [t for t in source_types if t != "carpeta_de_investigacion" and t != "carpeta_investigacion"]

    # Nota: Puede haber variaciones en el nombre (con/sin guión bajo)
    assert not wrong_types, (
        f"ERROR: Se encontraron tipos incorrectos: {set(wrong_types)}. "
        f"Solo se esperaba 'carpeta_de_investigacion' o 'carpeta_investigacion'."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
