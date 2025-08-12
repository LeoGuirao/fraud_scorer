from __future__ import annotations
import os
from typing import Dict, List, Tuple

# Flags de backends
_USE_EMB = True
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    _USE_EMB = False

_USE_FUZZ = False
if not _USE_EMB:
    try:
        from rapidfuzz import fuzz
        _USE_FUZZ = True
    except Exception:
        pass

_SIM_TH = float(os.getenv("FIELDS_SIM_THRESHOLD", "0.72"))

class SemanticAliasMatcher:
    """
    Empareja llaves observadas -> campo canónico usando embeddings o fuzzy.
    Seguro ante:
      - vocabularios vacíos
      - tamaños de vector inconsistentes
    """
    _MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, canon_aliases: Dict[str, List[str]]):
        self.canon_aliases = canon_aliases or {}
        self.alias_texts: List[str] = []
        self.alias_owner: List[str] = []

        for canon, aliases in self.canon_aliases.items():
            items = list(aliases or [])
            items.append(canon)  # el propio nombre canónico también puntúa
            for t in items:
                t = (t or "").strip()
                if not t:
                    continue
                self.alias_texts.append(t)
                self.alias_owner.append(canon)

        self.ready = False
        self.model = None
        self.alias_emb = None

        if _USE_EMB and self.alias_texts:
            try:
                self.model = SentenceTransformer(self._MODEL_ID)
                # encode devuelve (N, D). Si N=0, evitamos setup.
                self.alias_emb = self.model.encode(self.alias_texts, normalize_embeddings=True)
                if isinstance(self.alias_emb, list):
                    self.alias_emb = np.asarray(self.alias_emb)
                self.ready = self.alias_emb is not None and self.alias_emb.size > 0
            except Exception:
                self.model = None
                self.alias_emb = None
                self.ready = False

        if not _USE_EMB and _USE_FUZZ and self.alias_texts:
            self.ready = True  # fuzzy no requiere pre-cálculo

    def best(self, query: str) -> Tuple[str | None, float]:
        q = (query or "").strip()
        if not q or not self.alias_texts or not self.ready:
            return None, 0.0

        # Embeddings
        if _USE_EMB and self.model is not None and self.alias_emb is not None:
            try:
                qv = self.model.encode([q], normalize_embeddings=True)
                # qv shape (1, D); alias_emb shape (N, D)
                if hasattr(qv, "ndim") and qv.ndim == 2:
                    qv = qv[0]
                sims = self.alias_emb @ qv  # cosinas (por normalización)
                i = int(sims.argmax())
                score = float(sims[i])
                return (self.alias_owner[i], score) if score >= _SIM_TH else (None, score)
            except Exception:
                # si algo falla, caída blanda a fuzzy (si existe)
                pass

        # Fuzzy
        if _USE_FUZZ:
            best_c, best_s = None, 0.0
            for txt, owner in zip(self.alias_texts, self.alias_owner):
                s = fuzz.token_set_ratio(q, txt) / 100.0
                if s > best_s:
                    best_s, best_c = s, owner
            return (best_c, best_s) if best_s >= _SIM_TH else (None, best_s)

        return None, 0.0
