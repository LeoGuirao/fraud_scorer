"""
Gestor de Guías Especializadas de Fraude (YAML/JSON)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # Permite fallback y evita romper en entornos sin PyYAML

logger = logging.getLogger(__name__)


class FraudGuide:
    def __init__(self, data: Dict[str, Any]):
        self._data = data or {}

    @property
    def document_type(self) -> str:
        return self._data.get("metadata", {}).get("type", "")

    @property
    def version(self) -> str:
        return self._data.get("metadata", {}).get("version", "1.0")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get_fraud_indicators(self, risk_level: Optional[str] = None) -> List[Dict[str, Any]]:
        indicators = self._data.get("methodology", {}).get("fraud_indicators", {})
        if risk_level:
            return indicators.get(risk_level, [])
        out: List[Dict[str, Any]] = []
        for lvl in ("high_risk", "medium_risk", "low_risk"):
            out.extend(indicators.get(lvl, []))
        return out

    def get_validation_rules(self) -> Dict[str, Any]:
        return self._data.get("methodology", {}).get("validation_rules", {})

    def get_cross_references(self) -> List[str]:
        return self._data.get("methodology", {}).get("cross_reference_documents", [])

    def response_template(self) -> Dict[str, Any]:
        return self._data.get("response_template", {})


class FraudGuideManager:
    ALIASES: Dict[str, str] = {
        'carta_reclamacion': 'carta_de_reclamacion_formal_a_la_aseguradora',
        'carta_respuesta': 'carta_de_reclamacion_formal_a_la_aseguradora',
        'carpeta': 'carpeta_de_investigacion',
        'denuncia': 'denuncia_de_los_hechos',
        'denuncia_hechos': 'denuncia_de_los_hechos',
        'poliza': 'poliza_de_la_aseguradora',
        'factura': 'facturas_comerciales_internacionales',
        'factura_compra': 'facturas_comerciales_internacionales',
        'tarjeta_circulacion': 'tarjeta_de_circulacion_vehiculo',
        'carta_porte': 'cfdi_carta_porte',
    }

    def __init__(self, guides_dir: Optional[Path] = None):
        if guides_dir is None:
            guides_dir = Path(__file__).resolve().parent.parent / "guides"
        self.guides_dir = guides_dir
        self._guides: Dict[str, FraudGuide] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.guides_dir.exists():
            logger.warning(f"Directorio de guías no existe: {self.guides_dir}")
            return
        for p in self.guides_dir.iterdir():
            if p.suffix.lower() in (".yaml", ".yml", ".json"):
                try:
                    guide = self._load_file(p)
                    if guide and guide.document_type:
                        self._guides[guide.document_type] = guide
                        logger.info(f"Guía cargada: {guide.document_type} v{guide.version}")
                except Exception as e:
                    logger.error(f"Error cargando guía {p.name}: {e}")

    def _load_file(self, path: Path) -> Optional[FraudGuide]:
        data: Dict[str, Any]
        if path.suffix.lower() in (".yaml", ".yml"):
            if yaml is None:
                logger.warning("PyYAML no disponible; omitiendo guía YAML: %s", path.name)
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)  # type: ignore
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return FraudGuide(data)

    def get_guide(self, document_type: str) -> Optional[FraudGuide]:
        canonical = self.normalize_type(document_type)
        guide = self._guides.get(canonical)
        if guide:
            return guide
        logger.warning(f"No se encontró guía para tipo: {document_type}")
        return None

    def normalize_type(self, document_type: str) -> str:
        doc_type = (document_type or "").strip()
        if not doc_type:
            return ""
        if doc_type in self._guides:
            return doc_type
        lower = doc_type.lower()
        alias = self.ALIASES.get(lower)
        if alias:
            return alias
        return doc_type
