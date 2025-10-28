from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable, Tuple

import pytest

from fraud_scorer.ai.config import RickAgentConfig, load_config
from fraud_scorer.ai.orchestration import AgenteRickService
from fraud_scorer.ai.retrieval import RetrievedDocument

CASE_ID = "CASE-2025-0001"


class _StubLLM:
    """Stub que simula el cliente ChatOpenAI usado por AgenteRickService."""

    def __init__(self) -> None:
        self.messages: list[list] = []

    def invoke(self, messages: list) -> object:  # pragma: no cover - proxy mínimo
        self.messages.append(messages)

        class _Response:
            def __init__(self) -> None:
                self.content = "respuesta_stub"
                self.response_metadata = {
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}
                }

        return _Response()


@dataclass(frozen=True)
class QuestionnaireItem:
    identifier: str
    question: str
    expected_types: set[str]
    expected_answers: Tuple[str, ...]
    top_k: int = 15
    require_all: bool = True


SHORT_QUESTIONNAIRE: tuple[QuestionnaireItem, ...] = (
    QuestionnaireItem(
        identifier="numero_siniestro",
        question="Número del siniestro?",
        expected_types={
            "structured_card",
            "informe_final_del_ajustador",
            "carpeta_de_investigacion",
        },
        expected_answers=("20240000001361",),
        top_k=20,
    ),
    QuestionnaireItem(
        identifier="nombre_asegurado",
        question="Nombre del asegurado?",
        expected_types={
            "structured_card",
            "poliza_de_la_aseguradora",
            "carpeta_de_investigacion",
        },
        expected_answers=("Grupo Aceros Ocotlán",),
        top_k=12,
        require_all=False,
    ),
    QuestionnaireItem(
        identifier="numero_poliza",
        question="Número de póliza?",
        expected_types={"poliza_de_la_aseguradora", "informe_final_del_ajustador"},
        expected_answers=("76 - 238",),
        top_k=12,
    ),
    QuestionnaireItem(
        identifier="domicilio_poliza",
        question="Domicilio en la póliza?",
        expected_types={"poliza_de_la_aseguradora"},
        expected_answers=("Av Lazaro Cardenas No. Ext. 2257",),
        top_k=15,
    ),
    QuestionnaireItem(
        identifier="bien_reclamado",
        question="Bien reclamado?",
        expected_types={"structured_card", "informe_final_del_ajustador"},
        expected_answers=("Embarque consistente en varillas de acero", "placas de acero"),
        top_k=15,
        require_all=False,
    ),
    QuestionnaireItem(
        identifier="monto_reclamacion",
        question="Monto reclamado?",
        expected_types={
            "informe_final_del_ajustador",
            "carta_de_reclamacion_formal_a_la_aseguradora",
        },
        expected_answers=("3,145,997.60", "3,155,679.10"),
        top_k=15,
        require_all=False,
    ),
    QuestionnaireItem(
        identifier="tipo_siniestro",
        question="Tipo de siniestro?",
        expected_types={"poliza_de_la_aseguradora", "structured_card"},
        expected_answers=("Robo de bulto por entero",),
        top_k=18,
        require_all=False,
    ),
    QuestionnaireItem(
        identifier="fecha_ocurrencia",
        question="Fecha del siniestro?",
        expected_types={
            "structured_card",
            "informe_final_del_ajustador",
            "carpeta_de_investigacion",
        },
        expected_answers=("13 de febrero de 2024", "2024-02-13"),
        top_k=15,
        require_all=False,
    ),
    QuestionnaireItem(
        identifier="fecha_reclamacion",
        question="Fecha de reclamación?",
        expected_types={
            "carta_de_reclamacion_formal_a_la_aseguradora",
            "structured_card",
        },
        expected_answers=("07 de marzo de 2024", "2024-03-07", "13 de febrero de 2024"),
        top_k=15,
        require_all=False,
    ),
    QuestionnaireItem(
        identifier="lugar_hechos",
        question="Lugar del siniestro?",
        expected_types={"carpeta_de_investigacion", "denuncia_de_los_hechos"},
        expected_answers=("carretera matehuala", "carretera federal 57", "kilometro 57"),
        top_k=15,
        require_all=False,
    ),
    QuestionnaireItem(
        identifier="operadores",
        question="Operadores involucrados?",
        expected_types={
            "informe_final_del_ajustador",
            "oficio_denuncia",
            "carpeta_de_investigacion",
        },
        expected_answers=("Enrique Hernández García", "Irwin Rueda Rubio"),
        top_k=18,
    ),
    QuestionnaireItem(
        identifier="apoderado",
        question="Apoderado legal?",
        expected_types={
            "informe_final_del_ajustador",
            "carpeta_de_investigacion",
            "oficio_denuncia",
            "acreditacion_de_propiedad_y_representacion",
        },
        expected_answers=("Reynaga", "Jorge Alberto Reynaga"),
        top_k=20,
        require_all=False,
    ),
    QuestionnaireItem(
        identifier="placas_camiones",
        question="Placas de camiones?",
        expected_types={
            "cfdi_carta_porte",
            "informe_final_del_ajustador",
            "carpeta_de_investigacion",
        },
        expected_answers=("16BC2T", "18AT9H"),
        top_k=15,
        require_all=False,
    ),
    QuestionnaireItem(
        identifier="placas_semirremolques",
        question="Placas de semirremolques?",
        expected_types={
            "carpeta_de_investigacion",
            "cfdi_carta_porte",
            "conocimiento_de_embarque",
        },
        expected_answers=("97UL4C", "15TZ2Y", "009UR9", "34UL2C"),
        top_k=18,
        require_all=False,
    ),
    QuestionnaireItem(
        identifier="camion_enrique",
        question="Unidad de Enrique Hernández García?",
        expected_types={"carpeta_de_investigacion", "informe_final_del_ajustador"},
        expected_answers=("16BC2T", "97UL4C", "15TZ2Y"),
        top_k=18,
    ),
    QuestionnaireItem(
        identifier="camion_irwin",
        question="Unidad de Irwin Rueda Rubio?",
        expected_types={"carpeta_de_investigacion", "informe_final_del_ajustador"},
        expected_answers=("18AT9H", "009UR9", "34UL2C"),
        top_k=18,
    ),
    QuestionnaireItem(
        identifier="ministerio_publico",
        question="Ministerio público responsable?",
        expected_types={"denuncia_de_los_hechos", "carpeta_de_investigacion"},
        expected_answers=("Jonathan Josué Zuviri Alonso",),
        top_k=12,
    ),
)


def _normalize(text: str) -> str:
    """Remueve diacríticos y normaliza mayúsculas/minúsculas para coincidencias robustas."""

    normalized = unicodedata.normalize("NFD", text or "")
    return "".join(char for char in normalized.lower() if not unicodedata.combining(char))


def _collect_text(chunks: Iterable[RetrievedDocument], limit: int) -> str:
    texts: list[str] = []
    for item in chunks:
        if len(texts) >= limit:
            break
        texts.append(item.document.page_content)
    return "\n".join(texts)


@pytest.fixture(scope="module")
def rick_service() -> AgenteRickService:
    """Instancia AgenteRickService con LLM simulado y límites más bajos de resultados."""

    overrides: dict[str, object] = {
        "agent_mode_enabled": False,
        "max_results": 20,
    }
    config: RickAgentConfig = load_config(overrides=overrides)
    return AgenteRickService(config=config, llm=_StubLLM())


@pytest.mark.parametrize("item", SHORT_QUESTIONNAIRE, ids=lambda item: item.identifier)
def test_questionnaire_short_queries_surface_expected_sources(
    rick_service: AgenteRickService, item: QuestionnaireItem
) -> None:
    retrieved = rick_service._retriever.retrieve(  # type: ignore[attr-defined]
        CASE_ID,
        item.question,
        k=rick_service.config.max_results,  # type: ignore[attr-defined]
    )

    assert retrieved, "El retriever no devolvió resultados para la pregunta abreviada."

    top_docs = retrieved[: item.top_k]
    document_types = {
        (doc.document.metadata or {}).get("document_type") for doc in top_docs
    }
    document_types.discard(None)

    assert (
        document_types & item.expected_types
    ), (
        f"La pregunta corta '{item.identifier}' no recuperó ninguno de los tipos de documento "
        f"esperados: {item.expected_types}. Tipos obtenidos: {document_types}."
    )

    aggregated_text = _normalize(_collect_text(top_docs, item.top_k))
    present = [
        answer for answer in item.expected_answers if _normalize(answer) in aggregated_text
    ]

    if item.require_all:
        missing = [
            answer
            for answer in item.expected_answers
            if _normalize(answer) not in aggregated_text
        ]
        assert not missing, (
            f"Faltan fragmentos esperados para '{item.identifier}' con la consulta abreviada: {missing}. "
            f"Contexto evaluado (primeros 500 caracteres normalizados): "
            f"{aggregated_text[:500]}"
        )
    else:
        assert present, (
            f"Ninguna de las opciones {item.expected_answers} apareció en el contexto "
            f"recuperado para '{item.identifier}' con la pregunta corta."
        )
