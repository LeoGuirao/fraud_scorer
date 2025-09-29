#!/usr/bin/env python3
"""Valida reglas de correlación y configuración estadística."""
from __future__ import annotations

import argparse
from pathlib import Path

from fraud_scorer.analyzers.correlation.rules.validator import lint_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lint para reglas de correlación")
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("src/fraud_scorer/analyzers/correlation/rules/correlation_rules.yaml"),
        help="Ruta al archivo correlation_rules.yaml",
    )
    parser.add_argument(
        "--entities",
        type=Path,
        default=Path("src/fraud_scorer/analyzers/correlation/rules/entity_mappings.yaml"),
        help="Ruta al archivo entity_mappings.yaml",
    )
    parser.add_argument(
        "--statistical",
        type=Path,
        default=Path("src/fraud_scorer/analyzers/correlation/rules/statistical_config.yaml"),
        help="Ruta al archivo statistical_config.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = lint_rules(args.rules, args.entities, args.statistical)
    if errors:
        for error in errors:
            print(f"❌ {error}")
        print(f"\nSe identificaron {len(errors)} problemas en las reglas de correlación.")
        return 1

    print("✅ Reglas y configuración estadística sin problemas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
