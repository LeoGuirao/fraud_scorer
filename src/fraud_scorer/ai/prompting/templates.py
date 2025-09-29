"""Utilidades para construir prompts del Agente Rick."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from langchain_core.messages import SystemMessage, HumanMessage

from . import SYSTEM_PROMPT_PATH


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_messages(context_segments: Iterable[str], question: str) -> list:
    context = "\n\n".join(segment.strip() for segment in context_segments if segment)
    user_content = f"Contexto:\n{context}\n\nPregunta:\n{question.strip()}"
    return [
        SystemMessage(content=load_system_prompt()),
        HumanMessage(content=user_content),
    ]


__all__ = ["build_messages", "load_system_prompt"]
