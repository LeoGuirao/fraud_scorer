"""Prompt helpers for Agente Rick."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "agente_rick_system.txt"

__all__ = ["PROMPTS_DIR", "SYSTEM_PROMPT_PATH"]
