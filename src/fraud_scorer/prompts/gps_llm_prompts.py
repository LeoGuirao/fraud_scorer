"""Prompt builders for GPS LLM summaries."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from langchain.schema import HumanMessage, SystemMessage

DEFAULT_QUESTION_ES = (
    "Analiza el monitoreo GPS y entrega un resumen ejecutivo con anomalías,"
    " gaps relevantes y recomendaciones operativas."
)

DEFAULT_QUESTION_EN = (
    "Analyse the GPS monitoring data and provide an executive summary,"
    " highlighting anomalies, relevant gaps, and recommended actions."
)


def build_gps_summary_messages(
    *,
    summary: Dict[str, Any],
    preview_rows: Sequence[Dict[str, Any]],
    question: str | None,
    language: str = "es",
) -> list:
    system_prompt = _system_prompt(language)
    human_prompt = _human_prompt(summary, preview_rows, question, language)
    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]


def _system_prompt(language: str) -> str:
    if language.lower().startswith("en"):
        return (
            "You are an insurance fraud analyst specialised in logistics and GPS telemetry. "
            "Produce concise, verifiable insights, and always flag data quality issues."
        )
    return (
        "Eres un analista de fraude especializado en logística y telemetría GPS. "
        "Entrega un resumen ejecutable, destacando hallazgos verificables y riesgos operativos."
    )


def _human_prompt(
    summary: Dict[str, Any],
    preview_rows: Sequence[Dict[str, Any]],
    question: str | None,
    language: str,
) -> str:
    context_lines: list[str] = []
    gaps = summary.get("time_gaps") or []
    if gaps:
        context_lines.append("Gaps detectados:")
        for gap in gaps[:5]:
            context_lines.append(
                f"- {gap.get('start')} → {gap.get('end')} ({gap.get('gap_minutes')} minutos)"
            )
    else:
        context_lines.append("Sin gaps relevantes reportados.")

    bbox = summary.get("bounding_box") or {}
    if bbox:
        context_lines.append(
            "Área recorrida: "
            f"lat [{bbox.get('min_lat')}, {bbox.get('max_lat')}], "
            f"lon [{bbox.get('min_lon')}, {bbox.get('max_lon')}]"
        )

    speed_stats = summary.get("speed_stats") or {}
    if speed_stats:
        context_lines.append(
            "Velocidad promedio: {avg:.1f} km/h · p95={p95:.1f} km/h · máxima={max:.1f} km/h".format(
                avg=speed_stats.get("avg", 0.0),
                p95=speed_stats.get("p95", 0.0),
                max=speed_stats.get("max", 0.0),
            )
        )

    if preview_rows:
        context_lines.append("Muestras de eventos:")
        for row in preview_rows[:8]:
            pieces = []
            ts = row.get("timestamp")
            if ts:
                pieces.append(f"ts={ts}")
            if row.get("event_label"):
                pieces.append(f"event={row['event_label']}")
            if row.get("speed"):
                pieces.append(f"speed={row['speed']}")
            lat, lon = row.get("latitude"), row.get("longitude")
            if lat and lon:
                pieces.append(f"lat={lat}, lon={lon}")
            context_lines.append(" - " + ", ".join(pieces))

    prompt_body = "\n".join(context_lines)

    if not question:
        question = DEFAULT_QUESTION_EN if language.lower().startswith("en") else DEFAULT_QUESTION_ES

    if language.lower().startswith("en"):
        instructions = (
            "Provide a bullet list with: (1) key route findings, (2) anomalies or data quality risks, "
            "and (3) recommended next steps. Keep it under 220 words."
        )
    else:
        instructions = (
            "Responde en viñetas con: (1) hallazgos clave de la ruta, (2) anomalías o riesgos de datos y "
            "(3) siguientes pasos recomendados. Máximo 220 palabras."
        )

    return f"{prompt_body}\n\nPregunta: {question}\n\n{instructions}"


__all__ = ["build_gps_summary_messages"]
