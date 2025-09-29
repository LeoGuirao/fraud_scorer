"""Herramientas para validar reglas de correlación y configuración estadística."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

_ALLOWED_SEVERITIES = {"low", "medium", "high", "critical"}
_ALLOWED_CONDITIONS = {"equality", "temporal_order", "set_overlap", "exists", "numeric_order"}


def _load_yaml(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def _validate_rule(rule: Dict[str, Any], index: int, source: Path) -> List[str]:
    errors: List[str] = []
    rule_id = rule.get("id") or f"rule_{index}"

    for key in ("id", "version", "severity", "condition"):
        if key not in rule:
            errors.append(f"{source.name} :: Regla {rule_id} carece de campo obligatorio '{key}'")

    severity = str(rule.get("severity") or "").lower()
    if severity and severity not in _ALLOWED_SEVERITIES:
        errors.append(f"{source.name} :: Regla {rule_id} usa severidad desconocida '{severity}'")

    documents = rule.get("documents")
    if not isinstance(documents, Sequence) or not documents:
        errors.append(f"{source.name} :: Regla {rule_id} debe definir al menos un documento en 'documents'")

    entities = rule.get("entities")
    if not isinstance(entities, Sequence) or not entities:
        errors.append(f"{source.name} :: Regla {rule_id} debe definir entidades a vincular")

    condition = rule.get("condition") or {}
    cond_type = str(condition.get("type") or "").lower()
    if cond_type not in _ALLOWED_CONDITIONS:
        errors.append(f"{source.name} :: Regla {rule_id} usa condición desconocida '{cond_type}'")
    else:
        if cond_type == "equality":
            for field in ("source", "target"):
                if field not in condition:
                    errors.append(f"{source.name} :: Regla {rule_id} condición equality requiere '{field}'")
        if cond_type == "temporal_order":
            for field in ("earlier", "later"):
                if field not in condition:
                    errors.append(f"{source.name} :: Regla {rule_id} condición temporal_order requiere '{field}'")
        if cond_type == "set_overlap":
            for field in ("source", "target"):
                if field not in condition:
                    errors.append(f"{source.name} :: Regla {rule_id} condición set_overlap requiere '{field}'")
        if cond_type == "exists":
            if "path" not in condition:
                errors.append(f"{source.name} :: Regla {rule_id} condición exists requiere 'path'")
        if cond_type == "numeric_order":
            for field in ("lhs", "rhs"):
                if field not in condition:
                    errors.append(f"{source.name} :: Regla {rule_id} condición numeric_order requiere '{field}'")

    return errors


def _validate_unique_rule_ids(rules: Sequence[Dict[str, Any]], source: Path) -> List[str]:
    errors: List[str] = []
    seen: Dict[str, int] = {}
    for idx, rule in enumerate(rules):
        rule_id = rule.get("id") or f"rule_{idx}"
        if rule_id in seen:
            errors.append(
                f"{source.name} :: Reglas duplicadas con id '{rule_id}' (índices {seen[rule_id]} y {idx})"
            )
        seen[rule_id] = idx
    return errors


def _validate_entity_mappings(payload: Dict[str, Any], source: Path) -> List[str]:
    errors: List[str] = []
    if "version" not in payload:
        errors.append(f"{source.name} :: Debe declarar 'version'")
    if not payload.get("document_aliases") and not payload.get("field_aliases"):
        errors.append(f"{source.name} :: Debe definir al menos document_aliases o field_aliases")
    return errors


def _validate_statistical_config(payload: Dict[str, Any], source: Path) -> List[str]:
    errors: List[str] = []
    if "version" not in payload:
        errors.append(f"{source.name} :: Debe declarar 'version'")
    numeric = payload.get("numeric_anomalies", [])
    gap = payload.get("gap_anomalies", [])
    if not isinstance(numeric, list):
        errors.append(f"{source.name} :: 'numeric_anomalies' debe ser una lista")
    if not isinstance(gap, list):
        errors.append(f"{source.name} :: 'gap_anomalies' debe ser una lista")
    for collection, label in ((numeric, "numeric"), (gap, "gap")):
        if not collection:
            continue
        for idx, item in enumerate(collection):
            if not isinstance(item, dict):
                errors.append(f"{source.name} :: entrada {label}[{idx}] debe ser un objeto")
                continue
            if "id" not in item:
                errors.append(f"{source.name} :: entrada {label}[{idx}] carece de 'id'")
            if "description" not in item:
                errors.append(f"{source.name} :: entrada {label}[{idx}] carece de 'description'")
    return errors


def lint_rules(
    rules_path: Path,
    entity_path: Path,
    statistical_path: Path,
) -> List[str]:
    errors: List[str] = []

    rules_payload = _load_yaml(rules_path)
    rules = rules_payload.get("rules") or []
    if not isinstance(rules, list):
        return [f"{rules_path.name} :: la clave 'rules' debe ser una lista"]

    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"{rules_path.name} :: entrada de reglas en índice {idx} no es un objeto")
            continue
        errors.extend(_validate_rule(rule, idx, rules_path))

    errors.extend(_validate_unique_rule_ids(rules, rules_path))

    entity_payload = _load_yaml(entity_path)
    if isinstance(entity_payload, dict):
        errors.extend(_validate_entity_mappings(entity_payload, entity_path))
    else:
        errors.append(f"{entity_path.name} :: contenido inválido")

    statistical_payload = _load_yaml(statistical_path)
    if isinstance(statistical_payload, dict):
        errors.extend(_validate_statistical_config(statistical_payload, statistical_path))
    else:
        errors.append(f"{statistical_path.name} :: contenido inválido")

    return errors


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    rules_path = base_dir / 'correlation_rules.yaml'
    entity_path = base_dir / 'entity_mappings.yaml'
    statistical_path = base_dir / 'statistical_config.yaml'

    errors = lint_rules(rules_path, entity_path, statistical_path)
    if errors:
        for error in errors:
            print(f"❌ {error}")
        print(
            f"\nSe encontraron {len(errors)} problemas en la configuración de correlación."
        )
        return 1

    print("✅ Reglas de correlación y configuración estadística válidas.")
    return 0


if __name__ == '__main__':  # pragma: no cover - entrada CLI
    raise SystemExit(main())
