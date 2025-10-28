from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from fraud_scorer.storage.gps_cache import GPSCacheManager


CASE_ID = "CASE-2025-0001"
FULL_DOC_18AT9H = "16 1 Monitoreo 18AT9H.pdf"
FULL_DOC_16BC2T = "16 Monitoreo 16BC2T.pdf"


def _load_dataset(document_name: str):
    cache = GPSCacheManager()
    df = cache.load_dataset(CASE_ID, document_name)
    if df.empty:
        pytest.fail(f"El dataset de {document_name} no debería estar vacío")
    df = df.dropna(subset=["timestamp"])
    assert not df.empty, f"{document_name} carece de timestamps normalizados"
    return df.sort_values("timestamp").reset_index(drop=True)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia aproximada en km entre dos coordenadas."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def _cumulative_distance_km(df, end_time: datetime | None = None) -> float:
    coords = df.dropna(subset=["latitude", "longitude"]).sort_values("timestamp").reset_index(drop=True)
    if end_time is not None:
        coords = coords[coords["timestamp"] <= end_time]
    lat_shift = coords["latitude"].shift(-1)
    lon_shift = coords["longitude"].shift(-1)
    total = 0.0
    for lat, lon, next_lat, next_lon in zip(coords["latitude"], coords["longitude"], lat_shift, lon_shift):
        if next_lat is None or next_lon is None:
            continue
        if isinstance(next_lat, float) and math.isnan(next_lat):
            continue
        if isinstance(next_lon, float) and math.isnan(next_lon):
            continue
        total += _haversine_km(lat, lon, next_lat, next_lon)
    return total


def test_monitoreo_18at9h_dataset_covers_declared_period():
    df = _load_dataset(FULL_DOC_18AT9H)
    time_series = df["timestamp"]

    assert time_series.min() == datetime(2024, 2, 12, 0, 0, 5)
    assert time_series.max() == datetime(2024, 2, 14, 0, 17, 38)

    window = df[
        (df["timestamp"] >= datetime(2024, 2, 13, 19, 0))
        & (df["timestamp"] <= datetime(2024, 2, 13, 20, 0))
    ].dropna(subset=["latitude", "longitude"])

    assert len(window) >= 300, "Se esperaban lecturas continuas en la hora del siniestro"
    assert window["latitude"].round(6).unique().tolist() == [22.571049]
    assert window["longitude"].round(6).unique().tolist() == [-100.632222]


def test_monitoreo_16bc2t_dataset_has_valid_coordinates():
    df = _load_dataset(FULL_DOC_16BC2T)
    time_series = df["timestamp"]

    assert time_series.min() == datetime(2024, 2, 12, 0, 0, 1)
    assert time_series.max() == datetime(2024, 2, 14, 15, 39, 56)

    coord_series = df.dropna(subset=["latitude", "longitude"])
    assert len(coord_series) == len(df), "Se perdieron coordenadas durante la normalización"
    assert coord_series["latitude"].between(-90, 90).all()
    assert coord_series["longitude"].between(-180, 180).all()


def test_monitoreo_ambas_unidades_coinciden_en_evento():
    df_a = _load_dataset(FULL_DOC_18AT9H)
    df_b = _load_dataset(FULL_DOC_16BC2T)

    start = datetime(2024, 2, 13, 19, 20)
    end = start + timedelta(minutes=20)

    window_a = df_a[(df_a["timestamp"] >= start) & (df_a["timestamp"] <= end)].dropna(
        subset=["latitude", "longitude"]
    )
    window_b = df_b[(df_b["timestamp"] >= start) & (df_b["timestamp"] <= end)].dropna(
        subset=["latitude", "longitude"]
    )

    assert not window_a.empty, "18AT9H no reportó datos en la ventana del siniestro"
    assert not window_b.empty, "16BC2T no reportó datos en la ventana del siniestro"

    lat_diff = abs(window_a["latitude"].mean() - window_b["latitude"].mean())
    lon_diff = abs(window_a["longitude"].mean() - window_b["longitude"].mean())

    assert lat_diff <= 1e-4, f"Diferencia de latitud inesperada entre unidades: {lat_diff}"
    assert lon_diff <= 1e-4, f"Diferencia de longitud inesperada entre unidades: {lon_diff}"


def test_monitoreo_16bc2t_pasa_por_puntos_clave_de_la_ruta():
    df = _load_dataset(FULL_DOC_16BC2T)

    expected_checkpoints = [
        # (timestamp, lat, lon, tolerancia grados)
        (datetime(2024, 2, 12, 23, 1, 17), 22.2753, -97.8935, 0.02),  # Libramiento Poniente Tampico
        (datetime(2024, 2, 13, 2, 40, 34), 21.9898, -99.1021, 0.02),  # Autopista Valles-Tamuín
        (datetime(2024, 2, 13, 6, 12, 51), 21.8644, -99.6182, 0.02),  # Caseta Rayón
        (datetime(2024, 2, 13, 12, 59, 33), 22.4321, -100.3241, 0.03),  # Cerritos-Rioverde
    ]

    for expected_time, lat_ref, lon_ref, tolerance in expected_checkpoints:
        window = df[
            (df["timestamp"] >= expected_time - timedelta(minutes=2))
            & (df["timestamp"] <= expected_time + timedelta(minutes=2))
        ]
        assert not window.empty, f"No se encontró monitoreo alrededor de {expected_time}"
        closest = window.iloc[(window["timestamp"] - expected_time).abs().argmin()]
        lat_diff = abs(closest["latitude"] - lat_ref)
        lon_diff = abs(closest["longitude"] - lon_ref)
        assert lat_diff <= tolerance, f"Latitud fuera de rango en {expected_time}: diff={lat_diff}"
        assert lon_diff <= tolerance, f"Longitud fuera de rango en {expected_time}: diff={lon_diff}"

    # Origen esperado cercano al Puerto de Altamira
    origin_lat, origin_lon = df.iloc[0][["latitude", "longitude"]]
    assert _haversine_km(origin_lat, origin_lon, 22.4637, -97.9054) <= 5.0, "El origen no coincide con Altamira"


def test_monitoreo_distancia_y_detencion_posterior_al_siniestro():
    df = _load_dataset(FULL_DOC_16BC2T)

    distance_to_event = _cumulative_distance_km(df, datetime(2024, 2, 13, 19, 30))
    total_distance = _cumulative_distance_km(df)

    assert 420 <= distance_to_event <= 450, f"Distancia previa al siniestro fuera de rango: {distance_to_event:.1f} km"
    assert total_distance - distance_to_event <= 1.0, "El tracto recorrió distancia significativa tras el siniestro"


def test_monitoreo_gap_posterior_corrobora_privacion_operadores():
    df = _load_dataset(FULL_DOC_16BC2T).sort_values("timestamp").reset_index(drop=True)
    time_deltas = df["timestamp"].diff().dropna()

    largest_gap = time_deltas.max()
    assert largest_gap >= timedelta(hours=10), f"El gap más largo ({largest_gap}) no refleja la noche del 14/02"

    gap_index = time_deltas.idxmax()
    before_gap = df.loc[gap_index - 1]
    after_gap = df.loc[gap_index]

    assert before_gap["timestamp"] == datetime(2024, 2, 14, 0, 34, 9)
    assert after_gap["timestamp"] == datetime(2024, 2, 14, 11, 2, 6)
    assert abs(before_gap["latitude"] - after_gap["latitude"]) <= 0.0001
    assert abs(before_gap["longitude"] - after_gap["longitude"]) <= 0.0001
