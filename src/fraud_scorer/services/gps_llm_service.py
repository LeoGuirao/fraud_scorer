"""Selective LLM summaries for GPS artifacts with cost control."""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from fraud_scorer.services.gps_query_service import GPSDirectQueryService
from fraud_scorer.prompts.gps_llm_prompts import build_gps_summary_messages

logger = logging.getLogger(__name__)


DEFAULT_MODEL = os.getenv("GPS_LLM_MODEL", "gpt-4o-mini")
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}
DEFAULT_COMPLETION_TOKENS = 350


class GPSLLMQueryService:
    """Generates targeted summaries for GPS data using a small LLM budget."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        model: Optional[str] = None,
        cost_threshold: Optional[float] = None,
        query_service: Optional[GPSDirectQueryService] = None,
        llm: Optional[ChatOpenAI] = None,
        max_preview_rows: Optional[int] = None,
        max_context_chars: Optional[int] = None,
    ) -> None:
        self._enabled = enabled if enabled is not None else _read_bool("GPS_LLM_ENABLED", False)
        self.model = model or os.getenv("GPS_LLM_MODEL", DEFAULT_MODEL)
        self.cost_threshold = (
            float(os.getenv("GPS_COST_THRESHOLD", "1.0"))
            if cost_threshold is None
            else float(cost_threshold)
        )
        self.max_preview_rows = (
            int(os.getenv("GPS_LLM_PREVIEW_ROWS", "120"))
            if max_preview_rows is None
            else int(max_preview_rows)
        )
        self.max_context_chars = (
            int(os.getenv("GPS_LLM_MAX_CONTEXT_CHARS", "8000"))
            if max_context_chars is None
            else int(max_context_chars)
        )
        self._query_service = query_service or GPSDirectQueryService()
        self._llm = llm

    @property
    def enabled(self) -> bool:
        return self._enabled

    def summarise_route(
        self,
        *,
        case_id: str,
        document_name: str,
        question: Optional[str] = None,
        language: str = "es",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_labels: Optional[Sequence[str]] = None,
        bounding_box: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "reason": "GPS_LLM_ENABLED=false"}

        dataset = self._query_service.query_dataset(
            case_id,
            document_name,
            start_time=start_time,
            end_time=end_time,
            event_labels=event_labels,
            bounding_box=bounding_box,
            limit=self.max_preview_rows,
        )

        summary = dataset.get("summary") or {}
        preview_rows = dataset.get("preview") or []
        trimmed_preview = preview_rows[: self.max_preview_rows]

        context_text = _build_context_block(summary, trimmed_preview)
        if len(context_text) > self.max_context_chars:
            context_text = context_text[: self.max_context_chars] + "\n[Contexto truncado]"

        prompt_tokens = self._estimate_prompt_tokens(context_text)
        cost_estimate = self._estimate_cost(prompt_tokens, DEFAULT_COMPLETION_TOKENS)
        if cost_estimate > self.cost_threshold:
            return {
                "skipped": True,
                "reason": "estimated_cost_exceeds_threshold",
                "estimated_cost": cost_estimate,
                "estimated_prompt_tokens": prompt_tokens,
                "model": self.model,
            }

        messages = build_gps_summary_messages(
            summary=summary,
            preview_rows=trimmed_preview,
            question=question,
            language=language,
        )

        llm = self._ensure_llm()
        try:
            response = llm.invoke(messages)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - depends on provider/network
            logger.error(
                "LLM invocation failed for GPS summary %s/%s: %s",
                case_id,
                document_name,
                exc,
            )
            return {
                "error": "llm_error",
                "message": "No se pudo generar el resumen GPS en este momento.",
            }

        metadata = getattr(response, "response_metadata", {}) or {}
        token_usage = metadata.get("token_usage") or {}

        logger.info(
            "GPS LLM summary generated",
            extra={
                "case_id": case_id,
                "document_name": document_name,
                "model": self.model,
                "estimated_cost": cost_estimate,
                "estimated_prompt_tokens": prompt_tokens,
                "token_usage": token_usage,
            },
        )

        return {
            "answer": response.content.strip(),
            "model": self.model,
            "estimated_cost": cost_estimate,
            "estimated_prompt_tokens": prompt_tokens,
            "token_usage": token_usage,
            "summary": summary,
            "preview_rows": trimmed_preview,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(model=self.model, temperature=0)
        return self._llm

    def _estimate_prompt_tokens(self, context: str) -> int:
        # Simple heuristic: 1 token ≈ 4 characters
        return max(1, math.ceil(len(context) / 4))

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["gpt-4o-mini"])
        prompt_cost = (prompt_tokens / 1000) * pricing["input"]
        completion_cost = (completion_tokens / 1000) * pricing["output"]
        return round(prompt_cost + completion_cost, 6)


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_context_block(summary: Dict[str, Any], preview_rows: Sequence[Dict[str, Any]]) -> str:
    lines: list[str] = []
    time_span = summary.get("time_span") or {}
    if time_span:
        lines.append(
            f"Ventana temporal: {time_span.get('start')} → {time_span.get('end')}"
            f" ({time_span.get('duration_minutes')} minutos)"
        )
    if summary.get("time_gaps"):
        for gap in summary["time_gaps"][:5]:
            lines.append(
                f"Gap detectado: {gap.get('start')} → {gap.get('end')} ({gap.get('gap_minutes')} minutos)"
            )
    if summary.get("speed_stats"):
        stats = summary["speed_stats"]
        lines.append(
            f"Velocidad promedio: {stats.get('avg'):.2f} km/h; pico: {stats.get('max'):.2f} km/h; p95: {stats.get('p95'):.2f} km/h"
        )
    bbox = summary.get("bounding_box")
    if bbox:
        lines.append(
            "Bounding box: "
            f"lat [{bbox.get('min_lat')}, {bbox.get('max_lat')}], "
            f"lon [{bbox.get('min_lon')}, {bbox.get('max_lon')}]"
        )

    if preview_rows:
        lines.append("\nMuestras representativas:")
        for row in preview_rows[:10]:
            pieces = []
            timestamp = row.get("timestamp")
            if timestamp:
                pieces.append(f"ts={timestamp}")
            lat = row.get("latitude") or row.get("lat")
            lon = row.get("longitude") or row.get("long")
            if lat is not None and lon is not None:
                pieces.append(f"lat={lat}, lon={lon}")
            speed = row.get("speed")
            if speed is not None:
                pieces.append(f"speed={speed}")
            label = row.get("event_label") or row.get("event")
            if label:
                pieces.append(f"event={label}")
            lines.append(" - " + ", ".join(str(piece) for piece in pieces if piece))

    return "\n".join(lines)


__all__ = ["GPSLLMQueryService"]
