"""Normalización de entidades y documentos para el motor de correlación."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from fraud_scorer.settings import DOCUMENT_TYPE_ALIASES


class EntityNormalizer:
    """Resuelve alias de documentos y campos hacia sus nombres canónicos."""

    _DEFAULT_PATH = Path(__file__).resolve().parents[1] / "rules" / "entity_mappings.yaml"

    def __init__(
        self,
        *,
        mappings: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
    ) -> None:
        self.version = str(version or (mappings or {}).get("version") or "v0")

        self._doc_alias_to_canonical: Dict[str, str] = {}
        self._doc_canonical_to_aliases: Dict[str, set[str]] = defaultdict(set)
        self._field_alias_to_canonical: Dict[str, str] = {}
        self._field_canonical_to_aliases: Dict[str, set[str]] = defaultdict(set)

        self._populate_from_settings()
        if mappings:
            self._populate_from_mappings(mappings)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "EntityNormalizer":
        config_path = path or cls._DEFAULT_PATH
        raw: Dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if isinstance(loaded, dict):
                raw = loaded
        version = raw.get("version") or raw.get("meta", {}).get("version")
        return cls(mappings=raw, version=version)

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "EntityNormalizer":
        return cls.from_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def canonical_document_type(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        return self._doc_alias_to_canonical.get(value, value)

    def canonical_field_name(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        return self._field_alias_to_canonical.get(value, value)

    def aliases_for_document(self, canonical: str) -> List[str]:
        aliases = self._doc_canonical_to_aliases.get(canonical, set())
        return sorted(alias for alias in aliases if alias != canonical)

    def aliases_for_field(self, canonical: str) -> List[str]:
        aliases = self._field_canonical_to_aliases.get(canonical, set())
        return sorted(alias for alias in aliases if alias != canonical)

    def normalize_fields(self, fields: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Normaliza un diccionario de campos devolviendo mapa alias→canónico."""
        if not fields:
            return {}, {}
        normalised: Dict[str, Any] = {}
        alias_map: Dict[str, str] = {}
        for key, value in fields.items():
            canonical = self.canonical_field_name(key) or key
            if canonical not in normalised:
                normalised[canonical] = value
            else:
                # Si el canónico ya existe pero el actual es el mismo valor, ignorar.
                # Si difiere, preferir el primer valor visto para estabilidad.
                prev = normalised[canonical]
                if prev is None and value is not None:
                    normalised[canonical] = value
            if canonical != key:
                alias_map[key] = canonical
        return normalised, alias_map

    def document_alias_index(self) -> Dict[str, List[str]]:
        return {canonical: self.aliases_for_document(canonical) for canonical in self._doc_canonical_to_aliases}

    def field_alias_index(self) -> Dict[str, List[str]]:
        return {canonical: self.aliases_for_field(canonical) for canonical in self._field_canonical_to_aliases}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _populate_from_settings(self) -> None:
        for alias, canonical in (DOCUMENT_TYPE_ALIASES or {}).items():
            if not canonical:
                continue
            self._register_document_alias(alias, canonical)

    def _populate_from_mappings(self, mappings: Dict[str, Any]) -> None:
        doc_aliases = mappings.get("document_aliases") or {}
        if isinstance(doc_aliases, dict):
            for group, aliases in doc_aliases.items():
                canonical = self._resolve_canonical_from_aliases(aliases)
                if canonical:
                    self._register_document_alias(group, canonical)
                    for alias in self._iter_aliases(aliases):
                        self._register_document_alias(alias, canonical)

        field_aliases = mappings.get("field_aliases") or {}
        if isinstance(field_aliases, dict):
            for alias, payload in field_aliases.items():
                canonical, extra_aliases = self._extract_field_alias_payload(alias, payload)
                self._register_field_alias(alias, canonical)
                for extra in extra_aliases:
                    self._register_field_alias(extra, canonical)

    @staticmethod
    def _iter_aliases(aliases: Any) -> Iterable[str]:
        if isinstance(aliases, str):
            yield aliases
        elif isinstance(aliases, Iterable):
            for item in aliases:
                if isinstance(item, str):
                    yield item

    @staticmethod
    def _resolve_canonical_from_aliases(aliases: Any) -> Optional[str]:
        if isinstance(aliases, list) and aliases:
            first = aliases[0]
            return first if isinstance(first, str) else None
        if isinstance(aliases, str):
            return aliases
        return None

    @staticmethod
    def _extract_field_alias_payload(alias: str, payload: Any) -> Tuple[str, List[str]]:
        if isinstance(payload, dict):
            canonical = payload.get("canonical") or alias
            extra = [item for item in payload.get("aliases", []) if isinstance(item, str)]
            return canonical, extra
        if isinstance(payload, str):
            return payload, []
        return alias, []

    def _register_document_alias(self, alias: str, canonical: str) -> None:
        alias = str(alias)
        canonical = str(canonical)
        self._doc_alias_to_canonical[alias] = canonical
        self._doc_alias_to_canonical.setdefault(canonical, canonical)
        self._doc_canonical_to_aliases[canonical].add(alias)
        self._doc_canonical_to_aliases[canonical].add(canonical)

    def _register_field_alias(self, alias: str, canonical: str) -> None:
        alias = str(alias)
        canonical = str(canonical)
        self._field_alias_to_canonical[alias] = canonical
        self._field_alias_to_canonical.setdefault(canonical, canonical)
        self._field_canonical_to_aliases[canonical].add(alias)
        self._field_canonical_to_aliases[canonical].add(canonical)
