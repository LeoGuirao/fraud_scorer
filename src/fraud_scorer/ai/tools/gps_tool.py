"""Herramienta GPS para el modo agentic del Agente Rick."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from fraud_scorer.services.gps_query_service import GPSDirectQueryService

logger = logging.getLogger(__name__)

_ISO_Z_SUFFIX = "Z"


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    candidate = trimmed
    if candidate.endswith(_ISO_Z_SUFFIX):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"Timestamp inválido: {value}") from exc
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _validate_bbox(bbox: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if bbox is None:
        return None
    required_keys = {"min_lat", "max_lat", "min_lon", "max_lon"}
    if not required_keys.issubset(bbox.keys()):
        missing = required_keys.difference(bbox.keys())
        raise ValueError(f"bounding_box incompleto; faltan: {', '.join(sorted(missing))}")

    min_lat = float(bbox["min_lat"])
    max_lat = float(bbox["max_lat"])
    min_lon = float(bbox["min_lon"])
    max_lon = float(bbox["max_lon"])

    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValueError("Latitude fuera de rango [-90, 90].")
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValueError("Longitude fuera de rango [-180, 180].")
    if min_lat > max_lat:
        raise ValueError("min_lat no puede ser mayor que max_lat.")
    if min_lon > max_lon:
        raise ValueError("min_lon no puede ser mayor que max_lon.")

    return {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
    }


def _normalize_reference_point(point: Optional[Dict[str, float]]) -> Optional[Tuple[float, float]]:
    if point is None:
        return None
    if not isinstance(point, dict) or not {"lat", "lon"}.issubset(point.keys()):
        raise ValueError("reference_point debe incluir llaves 'lat' y 'lon'.")
    lat = float(point["lat"])
    lon = float(point["lon"])
    if not (-90.0 <= lat <= 90.0):
        raise ValueError("reference_point.lat fuera de rango [-90, 90].")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError("reference_point.lon fuera de rango [-180, 180].")
    return lat, lon


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0  # Radio terrestre aproximado en km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


class GPSQueryInput(BaseModel):
    """Esquema de entrada para la herramienta GPS."""

    case_id: str = Field(..., description="ID del caso (ej. CASE-2025-0001)")
    document_name: str = Field(..., description="Nombre del documento GPS normalizado")
    timestamp: Optional[str] = Field(
        None,
        description="Timestamp ISO para buscar un punto exacto (Z o +00:00 permitidos)",
    )
    start_time: Optional[str] = Field(
        None,
        description="Inicio de ventana temporal en formato ISO 8601",
    )
    end_time: Optional[str] = Field(
        None,
        description="Fin de la ventana temporal en formato ISO 8601",
    )
    bounding_box: Optional[Dict[str, float]] = Field(
        None,
        description="Cuadro delimitador con llaves min_lat, max_lat, min_lon, max_lon",
    )
    event_labels: Optional[Sequence[str]] = Field(
        None,
        description="Filtro opcional por etiquetas de evento normalizadas",
    )
    limit: Optional[int] = Field(
        None,
        description="Máximo de filas para el preview (por defecto, 50)",
        ge=1,
        le=200,
    )
    reference_point: Optional[Dict[str, float]] = Field(
        None,
        description="Coordenadas objetivo {'lat': float, 'lon': float} para calcular distancias.",
    )

    @field_validator("case_id", "document_name")
    def _not_empty(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("El valor no puede ser vacío.")
        return cleaned

    @field_validator("reference_point")
    def _validate_reference_point(cls, value: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        _normalize_reference_point(value)
        return value


def build_gps_query_tool(
    *,
    service: GPSDirectQueryService,
    default_limit: int = 50,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> StructuredTool:
    """Crea un StructuredTool que permite consultar datasets GPS de forma segura."""

    if default_limit <= 0:
        raise ValueError("default_limit debe ser mayor a 0.")

    def _query(
        case_id: str,
        document_name: str,
        timestamp: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        bounding_box: Optional[Dict[str, float]] = None,
        event_labels: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        reference_point: Optional[Dict[str, float]] = None,
    ) -> str:
        payload = GPSQueryInput(
            case_id=case_id,
            document_name=document_name,
            timestamp=timestamp,
            start_time=start_time,
            end_time=end_time,
            bounding_box=bounding_box,
            event_labels=event_labels,
            limit=limit,
            reference_point=reference_point,
        )

        request_limit = payload.limit or default_limit

        ts_exact = _parse_datetime(payload.timestamp)
        start_dt = _parse_datetime(payload.start_time)
        end_dt = _parse_datetime(payload.end_time)

        if ts_exact and (start_dt or end_dt):
            raise ValueError("No combines timestamp exacto con rangos start_time/end_time.")
        if start_dt and end_dt and start_dt > end_dt:
            raise ValueError("start_time debe ser menor o igual a end_time.")

        bbox = _validate_bbox(payload.bounding_box)
        reference_point = _normalize_reference_point(payload.reference_point)
        filters = {
            "start_time": ts_exact or start_dt,
            "end_time": ts_exact or end_dt,
            "event_labels": payload.event_labels,
            "bounding_box": bbox,
            "limit": request_limit,
        }

        logger.info(
            "Tool query_gps_location llamada para caso=%s documento=%s limit=%s",
            payload.case_id,
            payload.document_name,
            request_limit,
        )

        result = service.query_dataset(
            payload.case_id,
            payload.document_name,
            start_time=filters["start_time"],
            end_time=filters["end_time"],
            event_labels=filters["event_labels"],
            bounding_box=filters["bounding_box"],
            limit=filters["limit"],
        )

        safe_result: Dict[str, Any] = {
            "row_count": result.get("row_count", 0),
            "summary": result.get("summary"),
            "preview": result.get("preview") or [],
        }

        if on_result is not None:
            try:
                on_result(
                    {
                        "case_id": payload.case_id,
                        "document_name": payload.document_name,
                        "row_count": safe_result["row_count"],
                        "filters": {
                            "start_time": filters["start_time"].isoformat()
                            if isinstance(filters["start_time"], datetime)
                            else filters["start_time"],
                            "end_time": filters["end_time"].isoformat()
                            if isinstance(filters["end_time"], datetime)
                            else filters["end_time"],
                            "event_labels": list(filters["event_labels"]) if filters["event_labels"] else None,
                            "bounding_box": bbox,
                            "limit": filters["limit"],
                        },
                        "reference_point": {
                            "lat": reference_point[0],
                            "lon": reference_point[1],
                        }
                        if reference_point
                        else None,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensivo
                logger.warning("Callback on_result falló: %s", exc)

        preview_rows = result.get("preview") or []
        if reference_point:
            safe_result["reference_point"] = {"lat": reference_point[0], "lon": reference_point[1]}
            distances = []
            enriched_preview = []
            for row in preview_rows:
                row_copy = dict(row)
                lat = row_copy.get("latitude") or row_copy.get("lat")
                lon = row_copy.get("longitude") or row_copy.get("long") or row_copy.get("lon")
                try:
                    lat_f = float(lat) if lat is not None else None
                    lon_f = float(lon) if lon is not None else None
                except (TypeError, ValueError):
                    lat_f = lon_f = None
                if lat_f is not None and lon_f is not None:
                    distance = _haversine_km(reference_point[0], reference_point[1], lat_f, lon_f)
                    row_copy["distance_km"] = round(distance, 3)
                    distances.append(distance)
                enriched_preview.append(row_copy)
            preview_rows = enriched_preview
            if distances:
                safe_result["distance_stats"] = {
                    "min_km": round(min(distances), 3),
                    "max_km": round(max(distances), 3),
                    "avg_km": round(sum(distances) / len(distances), 3),
                }
        safe_result["preview"] = preview_rows

        return json.dumps(safe_result, ensure_ascii=False, indent=2)

    tool = StructuredTool.from_function(
        func=_query,
        name="query_gps_location",
        description=(
            "Consulta datos GPS normalizados del caso. Utiliza esta herramienta cuando necesites "
            "saber ubicaciones, rutas o validar declaraciones frente a telemetría. "
            "Puedes proporcionar reference_point={'lat': ..., 'lon': ...} para obtener distancias en km."
        ),
        args_schema=GPSQueryInput,
    )
    return tool


__all__ = ["build_gps_query_tool", "GPSQueryInput"]
