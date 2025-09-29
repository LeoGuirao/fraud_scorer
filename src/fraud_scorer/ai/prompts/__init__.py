"""Prompt templates for Agente Rick."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "agente_rick_system.txt"

__all__ = ["SYSTEM_PROMPT_PATH"]
