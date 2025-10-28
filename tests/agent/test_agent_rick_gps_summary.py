from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Callable

import pandas as pd
import pytest

from fraud_scorer.ai.config import load_config
from fraud_scorer.ai.orchestration import AgenteRickService
from fraud_scorer.ai.orchestration.agente_rick import RickQueryResult
from fraud_scorer.storage.gps_cache import GPSCacheManager

CASE_ID = "CASE-2025-0001"
DOC_18AT9H = "16 1 Monitoreo 18AT9H.pdf"
DOC_16BC2T = "16 Monitoreo 16BC2T.pdf"


def _load_dataset(document_name: str) -> pd.DataFrame:
    cache = GPSCacheManager()
    df = cache.load_dataset(CASE_ID, document_name)
    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def _cumulative_distance(df: pd.DataFrame, *, until: datetime | None = None) -> float:
    coords = df.dropna(subset=["latitude", "longitude"])
    coords = coords.sort_values("timestamp")
    if until is not None:
        coords = coords[coords["timestamp"] <= until]
    lat_next = coords["latitude"].shift(-1)
    lon_next = coords["longitude"].shift(-1)
    total = 0.0
    for lat, lon, nlat, nlon in zip(coords["latitude"], coords["longitude"], lat_next, lon_next):
        if pd.isna(nlat) or pd.isna(nlon):
            continue
        total += _haversine_km(lat, lon, nlat, nlon)
    return total


def _gps_case_summary() -> str:
    df_a = _load_dataset(DOC_18AT9H)
    df_b = _load_dataset(DOC_16BC2T)

    window_start = datetime(2024, 2, 13, 19, 20)
    window_end = datetime(2024, 2, 13, 19, 40)

    window_a = df_a[(df_a["timestamp"] >= window_start) & (df_a["timestamp"] <= window_end)]
    window_b = df_b[(df_b["timestamp"] >= window_start) & (df_b["timestamp"] <= window_end)]

    lat_delta = abs(window_a["latitude"].mean() - window_b["latitude"].mean())
    lon_delta = abs(window_a["longitude"].mean() - window_b["longitude"].mean())
    coord_text = f"18AT9H y 16BC2T comparten coordenadas 22.5710 N, −100.6322 W (delta ≤ {max(lat_delta, lon_delta):.5f}) entre 19:20 y 19:40."

    checkpoints = [
        (datetime(2024, 2, 12, 23, 1, 17), "Libramiento Poniente Tampico"),
        (datetime(2024, 2, 13, 2, 40, 34), "Autopista Valles-Tamuín"),
        (datetime(2024, 2, 13, 6, 12, 51), "Caseta Rayón"),
        (datetime(2024, 2, 13, 12, 59, 33), "Cerritos-Rioverde"),
    ]
    checkpoint_rows = []
    for expected_ts, label in checkpoints:
        window = df_b[
            (df_b["timestamp"] >= expected_ts - timedelta(minutes=2))
            & (df_b["timestamp"] <= expected_ts + timedelta(minutes=2))
        ]
        if window.empty:
            raise AssertionError(f"No se encontró checkpoint para {label}")
        closest = window.iloc[(window["timestamp"] - expected_ts).abs().argmin()]
        checkpoint_rows.append(f"{closest['timestamp']:%Y-%m-%d %H:%M:%S} · {label}")
    checkpoints_text = (
        "El GPS de 16BC2T registra los cuatro peajes declarados: " + "; ".join(checkpoint_rows) + "."
    )

    distance_to_event = _cumulative_distance(df_b, until=datetime(2024, 2, 13, 19, 30))
    total_distance = _cumulative_distance(df_b)
    residual = total_distance - distance_to_event

    deltas = df_b["timestamp"].diff().dropna()
    largest_gap = deltas.max()
    gap_idx = deltas.idxmax()
    before_gap = df_b.loc[gap_idx - 1, "timestamp"]
    after_gap = df_b.loc[gap_idx, "timestamp"]
    gap_hours = int(largest_gap.total_seconds() // 3600)
    gap_minutes = int((largest_gap.total_seconds() % 3600) // 60)

    distance_text = (
        f"La distancia acumulada previo a las 19:30 alcanza {distance_to_event:.0f} km "
        f"(recorrido posterior {residual:.1f} km) y se observa un gap de {gap_hours} h {gap_minutes} min "
        f"(del {before_gap:%d/%m %H:%M} al {after_gap:%d/%m %H:%M}) sin movimiento."
    )

    summary = (
        "Resumen:\n"
        "- Co-localización de ambas unidades, casetas confirmadas y pausa prolongada consistente con la privación de libertad.\n\n"
        "Detalle:\n"
        f"- {coord_text}\n"
        f"- {checkpoints_text}\n"
        f"- {distance_text}\n\n"
        "Fuentes:\n"
        f"- {DOC_18AT9H} (GPS)\n"
        f"- {DOC_16BC2T} (GPS)"
    )
    return summary


@dataclass
class _GPSAwareStubLLM:
    summary_builder: Callable[[], str]

    def __post_init__(self) -> None:
        self.messages: list[list] = []

    def invoke(self, messages: list) -> object:  # pragma: no cover - simple stub
        self.messages.append(messages)
        summary = self.summary_builder()

        class _Response:
            def __init__(self, content: str) -> None:
                self.content = content
                self.response_metadata = {"token_usage": {"prompt_tokens": 0, "completion_tokens": 0}}

        return _Response(summary)


def test_agent_rick_delivers_gps_summary() -> None:
    expected_summary = _gps_case_summary()
    overrides = {"agent_mode_enabled": False, "max_results": 20}
    config = load_config(overrides=overrides)
    stub_llm = _GPSAwareStubLLM(summary_builder=_gps_case_summary)

    service = AgenteRickService(config=config, llm=stub_llm)

    question = (
        "Rick, valida el monitoreo GPS del CASE-2025-0001 y resume cómo se corroboran las coordenadas del robo, "
        "los peajes recorridos, la distancia acumulada y la pausa nocturna."
    )

    result: RickQueryResult = service.query(case_id=CASE_ID, question=question)

    assert result.answer == expected_summary
    assert stub_llm.messages, "El LLM simulado no fue invocado por Agente Rick."
    assert DOC_16BC2T in result.answer and DOC_18AT9H in result.answer


def test_gps_metrics_hint_mentions_named_toll_booths() -> None:
    overrides = {"agent_mode_enabled": False, "max_results": 20}
    config = load_config(overrides=overrides)
    stub_llm = _GPSAwareStubLLM(summary_builder=_gps_case_summary)
    service = AgenteRickService(config=config, llm=stub_llm)

    hint = service._get_gps_metrics_hint(CASE_ID)

    assert "[RESUMEN GPS AGREGADO]" in hint
    for name in (
        "Libramiento Poniente Tampico",
        "Autopista Valles-Tamuín",
        "Caseta Rayón",
        "Cerritos-Rioverde",
    ):
        assert name in hint, f"No se mencionó {name} en el resumen GPS"
