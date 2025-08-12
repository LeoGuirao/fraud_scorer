# src/fraud_scorer/pipelines/field_map.py
from pathlib import Path
from typing import Dict, List
import yaml, unicodedata, re, os

def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", (s or "").lower().strip())
                if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", s)

class FieldMap:
    def __init__(self, path: str | Path | None = None):
        path = path or os.getenv("FIELD_MAP_PATH", "src/fraud_scorer/config/field_map.yaml")
        self.path = Path(path)
        self.cfg = yaml.safe_load(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self._rev = self._build_reverse()

    def _build_reverse(self) -> Dict[str, str]:
        rev: Dict[str, str] = {}
        for scope, fields in (self.cfg or {}).items():
            for canon, aliases in (fields or {}).items():
                for a in (aliases or []):
                    rev[_norm(a)] = canon
        return rev

    def lookup(self, doc_type: str, key: str) -> str | None:
        nk = _norm(key)
        if nk in self._rev:
            return self._rev[nk]
        # candidatos por contains: global y doc_type
        for scope in ("global", doc_type):
            for canon, aliases in (self.cfg.get(scope) or {}).items():
                if any(_norm(a) in nk or nk in _norm(a) for a in (aliases or [])):
                    return canon
        return None

    def vocab(self) -> Dict[str, List[str]]:
        return {canon: list(set(aliases or [])) for _, fields in (self.cfg or {}).items() for canon, aliases in (fields or {}).items()}
