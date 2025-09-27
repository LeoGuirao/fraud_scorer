#!/usr/bin/env python3
"""Herramienta para validar las guías de fraude (YAML/JSON).

Carga cada guía, verifica campos obligatorios y compila expresiones
regulares declaradas en la sección `validation_rules`. Se usa en CI para
atrapar errores de formato (p. ej. escapes inválidos) antes de ejecutar
el pipeline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

GUIDES_DIR = Path(__file__).resolve().parents[1] / "src" / "fraud_scorer" / "guides"


@dataclass
class GuideIssue:
    path: Path
    message: str
    section: Optional[str] = None

    def format(self) -> str:
        location = f"[{self.section}]" if self.section else ""
        return f"- {self.path.name}{location}: {self.message}"


def iter_guides(paths: Iterable[Path]) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for path in paths:
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML no está disponible; instálalo para validar guías YAML")
            with path.open("r", encoding="utf-8") as handle:
                yield path, yaml.safe_load(handle)
        elif path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                yield path, json.load(handle)


def validate_patterns(data: Dict[str, Any], path: Path) -> List[GuideIssue]:
    issues: List[GuideIssue] = []
    rules = (data.get("methodology") or {}).get("validation_rules", {})
    for field_name, definition in rules.items():
        if not isinstance(definition, dict):
            continue
        if definition.get("type") != "pattern":
            continue
        expr = definition.get("pattern")
        if not isinstance(expr, str) or not expr:
            issues.append(GuideIssue(path, "Regla de patrón sin expresión definida", field_name))
            continue
        try:
            re.compile(expr)
        except re.error as exc:
            issues.append(GuideIssue(path, f"Regex inválido: {exc}", field_name))
    return issues


def validate_metadata(data: Dict[str, Any], path: Path) -> List[GuideIssue]:
    issues: List[GuideIssue] = []
    metadata = data.get("metadata") or {}
    if not metadata.get("type"):
        issues.append(GuideIssue(path, "metadata.type ausente o vacío"))
    if not metadata.get("version"):
        issues.append(GuideIssue(path, "metadata.version ausente o vacío"))
    return issues


def run_validation(directory: Path) -> List[GuideIssue]:
    guide_paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".yaml", ".yml", ".json"})
    if not guide_paths:
        raise RuntimeError(f"No se encontraron guías en {directory}")

    issues: List[GuideIssue] = []
    for path, data in iter_guides(guide_paths):
        if not isinstance(data, dict):
            issues.append(GuideIssue(path, "Guía no es un objeto JSON/YAML"))
            continue
        issues.extend(validate_metadata(data, path))
        issues.extend(validate_patterns(data, path))
    return issues


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Valida las guías de fraude")
    parser.add_argument(
        "--guides-dir",
        type=Path,
        default=GUIDES_DIR,
        help="Directorio de guías a validar (default: src/fraud_scorer/guides)",
    )
    args = parser.parse_args(argv)

    try:
        issues = run_validation(args.guides_dir)
    except Exception as exc:
        print(f"❌ Error ejecutando validación: {exc}", file=sys.stderr)
        return 2

    if issues:
        print("❌ Se encontraron problemas en las guías:")
        for issue in issues:
            print(issue.format())
        return 1

    print("✅ Todas las guías se cargaron y validaron correctamente")
    return 0


if __name__ == "__main__":
    sys.exit(main())
