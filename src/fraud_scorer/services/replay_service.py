# src/fraud_scorer/services/replay_service.py

import logging
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import json
import unicodedata
from collections import defaultdict

from ..storage.ocr_cache import OCRCacheManager
from ..storage.post_process_verifier import verify_case_artifacts
from ..storage.db import get_conn, get_correlation_findings
from ..storage.cases import get_case_by_id, get_total_savings
from ..pipelines.data_flow import build_docs_for_template_from_db
from ..processors.ai.ai_field_extractor import AIFieldExtractor
from ..processors.ai.ai_consolidator import AIConsolidator
from ..templates.ai_report_generator import AIReportGenerator
from ..settings import ExtractionConfig

logger = logging.getLogger(__name__)

def _summarize_correlation_counts(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = defaultdict(int)
    severity: Dict[str, int] = defaultdict(int)
    for item in findings or []:
        status = str(item.get("status") or "").lower()
        severity_level = str(item.get("severity") or "").lower()
        if status:
            counts[status] += 1
        if severity_level:
            severity[severity_level] += 1
    counts["total"] = len(findings or [])
    if severity:
        counts["by_severity"] = dict(severity)
    return dict(counts)

class ReplayService:
    """
    Contiene toda la lógica de negocio para el sistema de Replay.
    Es utilizado tanto por la API web como podría serlo por la CLI.
    """
    def __init__(self):
        self.cache_manager = OCRCacheManager()
        self._chroma_base_path: Optional[Path] = None
        self._rick_audit_log: Optional[Path] = None

        try:
            from ..ai.config import load_config

            config = load_config()
            self._chroma_base_path = config.chroma_base_path
            self._rick_audit_log = config.audit_log_path
        except Exception as exc:  # pragma: no cover - defensivo
            logger.debug("No se pudo cargar configuración del Agente Rick: %s", exc)
            chroma_env = os.getenv("AGENTE_RICK_CHROMA_PATH")
            audit_env = os.getenv("AGENTE_RICK_AUDIT_PATH")
            if chroma_env:
                self._chroma_base_path = Path(chroma_env)
            if audit_env:
                self._rick_audit_log = Path(audit_env)

        if self._chroma_base_path is None:
            self._chroma_base_path = Path(os.getenv("AGENTE_RICK_CHROMA_PATH", "data/chroma"))
        if self._rick_audit_log is None:
            self._rick_audit_log = Path(os.getenv("AGENTE_RICK_AUDIT_PATH", "data/logs/agent_rick_audit.jsonl"))

    @staticmethod
    def _parse_numeric_token(token: str) -> Optional[float]:
        token = token.strip()
        if not token:
            return None

        # Normalize thousand and decimal separators.
        if "," in token and "." in token:
            if token.rfind(",") > token.rfind("."):
                token = token.replace(".", "")
                token = token.replace(",", ".")
            else:
                token = token.replace(",", "")
        elif "," in token:
            token = token.replace(",", ".")
        else:
            token = token.replace(",", "")

        token = token.replace(" ", "")

        try:
            return float(token)
        except ValueError:
            return None

    @classmethod
    def _coerce_amount(cls, value: Any) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            return numeric if math.isfinite(numeric) else None

        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None

            cleaned = re.sub(r"[\$€£]|MXN|USD|EUR|MX\$", "", cleaned, flags=re.IGNORECASE)
            matches = re.findall(r"-?\d[\d.,]*", cleaned)
            for token in matches:
                parsed = cls._parse_numeric_token(token)
                if parsed is not None:
                    return parsed

        return None

    def compute_claim_amount(self, case_data: Dict[str, Any]) -> Optional[float]:
        """Calcula el monto de reclamación usando heurísticas consolidadas."""
        return self._extract_claim_amount(case_data)

    def _extract_claim_amount(self, case_data: Dict[str, Any]) -> Optional[float]:
        consolidated_fields = (
            (case_data.get("consolidated_data") or {}).get("consolidated_fields")
            or {}
        )
        amount = self._coerce_amount(consolidated_fields.get("monto_reclamacion"))
        if amount is not None:
            return amount

        extraction_results = case_data.get("extraction_results") or []
        candidates: List[float] = []
        for entry in extraction_results:
            fields = entry.get("extracted_fields") or {}
            candidate = self._coerce_amount(fields.get("monto_reclamacion"))
            if candidate is not None:
                candidates.append(candidate)

        return max(candidates) if candidates else None

    def _detect_tentative_fraud(self, case_data: Dict[str, Any]) -> bool:
        fraud_entries = case_data.get("fraud_analyses") or []
        for entry in fraud_entries:
            score = entry.get("fraud_score")
            score_value = self._coerce_amount(score)
            if score_value is not None and score_value >= 0.5:
                return True

            risk_level = entry.get("risk_level")
            if isinstance(risk_level, str) and risk_level.strip().lower() in {
                "alto",
                "high",
                "elevado",
            }:
                return True

        return False

    def _collect_name_variations(self, name: str) -> set[str]:
        """Genera variantes sanitizadas de un nombre para matching flexible."""
        variations: set[str] = set()
        if not name:
            return variations

        candidates = {name, name.strip()}
        candidates.add(name.replace(' ', '_'))
        candidates.add(name.replace(' ', ''))

        try:
            candidates.add(self.cache_manager._sanitize_filename(name))
        except Exception:
            pass

        normalized = unicodedata.normalize('NFKD', name)
        ascii_variant = ''.join(c for c in normalized if not unicodedata.combining(c))
        if ascii_variant:
            candidates.add(ascii_variant)
            candidates.add(ascii_variant.replace(' ', '_'))
            candidates.add(ascii_variant.replace(' ', ''))

        for candidate in candidates:
            if candidate:
                variations.add(candidate)
                variations.add(candidate.lower())

        return variations

    @staticmethod
    def _normalize_case_filename(name: str) -> str:
        """Normaliza nombres de archivo para hacer matching robusto."""
        if not name:
            return ""

        candidate = str(name).strip()
        candidate = candidate.replace("\\", "/").split("/")[-1]
        if candidate.startswith("ocr_results_for_"):
            candidate = candidate[len("ocr_results_for_") :]
        if candidate.endswith(".json"):
            candidate = candidate[:-5]
        candidate = candidate.replace("__", "_")
        candidate = candidate.replace("_", " ")
        candidate = re.sub(r"\s+\.", ".", candidate)
        candidate = re.sub(r"\s+", " ", candidate)
        candidate = candidate.strip()
        normalized = unicodedata.normalize("NFKD", candidate)
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return normalized.lower()

    def _map_case_document_type(self, doc_type: Optional[str]) -> Optional[str]:
        """Convierte tipos almacenados (AI/manual) a canónicos admitidos por extracción."""
        if not doc_type:
            return None

        alias_map = getattr(ExtractionConfig, "DOCUMENT_TYPE_ALIASES", {})
        normalized = unicodedata.normalize("NFKD", doc_type.strip()).lower()
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))

        canonical = alias_map.get(normalized, normalized)
        if canonical in ExtractionConfig.DOCUMENT_FIELD_MAPPING:
            return canonical

        logger.debug("Tipo de documento no reconocido en cache: %s", doc_type)
        return None

    def _build_case_document_types(self, case_index: Dict[str, Any]) -> Dict[str, str]:
        """Construye un mapeo filename → document_type usando clasificaciones previas."""

        doc_types: Dict[str, str] = {}

        def register(name: Optional[str], doc_type: Optional[str]) -> None:
            canonical = self._map_case_document_type(doc_type)
            if not name or not canonical:
                return
            key = self._normalize_case_filename(name)
            if key:
                doc_types[key] = canonical
                simplified = re.sub(r"[^a-z0-9.]+", "", key)
                if simplified:
                    doc_types.setdefault(simplified, canonical)

        for entry in case_index.get("classified_types") or []:
            register(entry.get("filename"), entry.get("document_type") or entry.get("ai_document_type"))

        for name, doc_type in (case_index.get("manual_classifications") or {}).items():
            register(name, doc_type)

        for name, doc_type in (case_index.get("ai_classifications") or {}).items():
            register(name, doc_type)

        return doc_types

    def _build_cleanup_tokens(
        self,
        *,
        case_id: str,
        insured_name: Optional[str],
        claim_number: Optional[str],
    ) -> Set[str]:
        tokens: Set[str] = set()
        candidates = {
            case_id,
            case_id.lower(),
            case_id.replace('-', ''),
            case_id.replace('-', '_'),
        }
        if insured_name:
            candidates.update(self._collect_name_variations(insured_name))
        if claim_number:
            candidates.add(str(claim_number))
            candidates.add(str(claim_number).replace('-', ''))
        try:
            candidates.add(self.cache_manager._sanitize_filename(case_id))
        except Exception:
            pass
        for value in (insured_name, claim_number):
            if not value:
                continue
            try:
                candidates.add(self.cache_manager._sanitize_filename(str(value)))
            except Exception:
                pass
        tokens.update({c.lower() for c in candidates if c})
        return tokens

    def _remove_report_family(self, report_file: Path) -> int:
        """Elimina los artefactos (json/html/pdf) asociados a un reporte."""
        removed = 0
        parent = report_file.parent
        stem = report_file.stem
        variants = {stem}
        variants.add(stem.replace('_RESULTADOS', ''))
        variants.add(stem.replace('_INFORME', ''))
        # Agregar más variantes comunes
        variants.add(stem.replace('_REPORTE', ''))
        variants.add(stem.replace('_FINAL', ''))

        for variant in {v for v in variants if v}:
            for ext in ('.json', '.html', '.pdf'):
                candidate = parent / f"{variant}{ext}"
                if candidate.exists():
                    try:
                        candidate.unlink()
                        removed += 1
                    except Exception:
                        pass
        return removed

    def _remove_reports_by_case_id(self, case_id: str, tokens: Optional[Set[str]] = None) -> int:
        """Busca reportes por contenido de case_id y elimina la familia completa."""
        if not case_id:
            return 0

        total_removed = 0
        reports_dir = Path("data/reports")
        if reports_dir.exists():
            for json_file in reports_dir.glob("*.json"):
                try:
                    content = json_file.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                payload = content.lower()
                match = case_id.lower() in payload
                if not match and tokens:
                    match = any(token in payload for token in tokens)
                if not match:
                    continue
                total_removed += self._remove_report_family(json_file)
        return total_removed
    def _remove_report_artifacts(self, tokens: Set[str]) -> int:
        reports_dir = Path("data/reports")
        if not reports_dir.exists() or not tokens:
            return 0

        tokens_lower = {token for token in tokens if token}
        if not tokens_lower:
            return 0

        removed = 0
        processed_stems: Set[str] = set()
        for report_file in list(reports_dir.iterdir()):
            if not report_file.is_file():
                continue
            stem = report_file.stem
            name_lower = report_file.name.lower()
            should_remove = any(token in name_lower for token in tokens_lower)

            if not should_remove and report_file.suffix.lower() == ".json":
                try:
                    content = report_file.read_text(encoding="utf-8", errors="ignore").lower()
                    should_remove = any(token in content for token in tokens_lower)
                except Exception:
                    should_remove = False

            if should_remove and stem not in processed_stems:
                removed += self._remove_report_family(report_file)
                processed_stems.add(stem)

        if removed:
            logger.info(f"  ✓ Reportes eliminados (tokens): {removed}")
        return removed

    def _remove_pipeline_cache_artifacts(self, tokens: Set[str]) -> int:
        base = Path(os.getenv("FS_DATA_DIR", "data")) / "temp" / "pipeline_cache"
        if not base.exists():
            return 0

        tokens_lower = {token for token in tokens if token}
        if not tokens_lower:
            return 0

        removed = 0
        candidates = sorted(base.glob("**/*"), key=lambda p: len(p.parts), reverse=True)
        for path in candidates:
            if not path.exists():
                continue
            target = path.name.lower()
            if not any(token in target for token in tokens_lower):
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed += 1
            except Exception as exc:
                logger.debug("No se pudo eliminar artefacto de pipeline %s: %s", path, exc)

        if removed:
            logger.info(f"  ✓ Pipeline cache limpiado (tokens): {removed} artefactos")
        return removed

    def _remove_vector_indexes(self, tokens: Set[str]) -> int:
        base = self._chroma_base_path
        if not base or not base.exists():
            return 0

        tokens_lower = {token for token in tokens if token}
        if not tokens_lower:
            return 0

        removed = 0
        for path in list(base.iterdir()):
            if not path.is_dir():
                continue
            name_lower = path.name.lower()
            match = any(token in name_lower for token in tokens_lower)
            if not match:
                manifest = path / "index_manifest.json"
                if manifest.exists():
                    try:
                        payload = manifest.read_text(encoding="utf-8", errors="ignore").lower()
                        match = any(token in payload for token in tokens_lower)
                    except Exception:
                        match = False
            if not match:
                continue
            try:
                shutil.rmtree(path)
                removed += 1
                logger.info(f"  ✓ Índice vectorial eliminado: {path.name}")
            except Exception as exc:
                logger.warning("  ⚠️ No se pudo eliminar índice vectorial %s: %s", path.name, exc)
        return removed

    def _remove_gps_artifacts(self, tokens: Set[str]) -> int:
        gps_dir = Path("data/gps")
        if not gps_dir.exists():
            return 0

        tokens_lower = {token for token in tokens if token}
        if not tokens_lower:
            return 0

        removed = 0
        for case_folder in gps_dir.iterdir():
            if not case_folder.is_dir():
                continue
            name_lower = case_folder.name.lower()
            match = any(token in name_lower for token in tokens_lower)

            if not match:
                metadata_candidates = [
                    case_folder / "_metadata.json",
                    case_folder / "_manifest.json",
                ]
                for metadata_file in metadata_candidates:
                    if not metadata_file.exists():
                        continue
                    try:
                        content = metadata_file.read_text(encoding="utf-8", errors="ignore").lower()
                    except Exception:
                        continue
                    if any(token in content for token in tokens_lower):
                        match = True
                        break

            if not match:
                # Buscamos _metadata.json dentro de subcarpetas específicas
                for nested_meta in case_folder.glob("*/_metadata.json"):
                    try:
                        content = nested_meta.read_text(encoding="utf-8", errors="ignore").lower()
                        if any(token in content for token in tokens_lower):
                            match = True
                            break
                    except Exception:
                        continue

            if not match:
                continue

            try:
                shutil.rmtree(case_folder)
                removed += 1
                logger.info(f"  ✓ Artefactos GPS eliminados: {case_folder.name}")
            except Exception as exc:
                logger.warning(f"  ⚠️ No se pudo eliminar GPS {case_folder.name}: {exc}")
        return removed

    def _purge_rick_audit_log(self, tokens: Set[str]) -> int:
        log_path = self._rick_audit_log
        if not log_path or not log_path.exists():
            return 0

        tokens_lower = {token for token in tokens if token}
        if not tokens_lower:
            return 0

        tmp_path: Optional[Path] = None
        removed = 0
        try:
            tmp_path = log_path.with_suffix(log_path.suffix + ".tmp")
            with log_path.open("r", encoding="utf-8", errors="ignore") as src, tmp_path.open("w", encoding="utf-8") as dst:
                for line in src:
                    raw_lower = line.lower()
                    try:
                        payload = json.loads(line)
                    except Exception:
                        payload = None

                    case_match = False
                    if payload and isinstance(payload, dict):
                        value = str(payload.get("case_id") or "").lower()
                        if value:
                            case_match = any(token in value for token in tokens_lower)

                    if not case_match:
                        case_match = any(token in raw_lower for token in tokens_lower)

                    if case_match:
                        removed += 1
                        continue
                    dst.write(line)

            tmp_path.replace(log_path)
            if removed:
                logger.info(f"  ✓ Bitácora Rick depurada: {removed} eventos")
        except FileNotFoundError:
            return 0
        except Exception as exc:
            logger.warning("  ⚠️ No se pudo depurar bitácora Rick: %s", exc)
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            return 0

        return removed

    def _reset_vector_store(self) -> bool:
        base = self._chroma_base_path
        if not base:
            return False

        try:
            if base.exists():
                shutil.rmtree(base)
            base.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as exc:
            logger.warning("  ⚠️ No se pudo reiniciar índices vectoriales: %s", exc)
            return False

    def _reset_rick_audit_log(self) -> bool:
        log_path = self._rick_audit_log
        if not log_path:
            return False
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if log_path.exists():
                log_path.unlink()
            log_path.touch()
            return True
        except Exception as exc:
            logger.warning("  ⚠️ No se pudo reiniciar bitácora Rick: %s", exc)
            return False

    def _collect_db_cleanup_snapshot(self, case_id: str) -> Dict[str, int]:
        """Registra conteos residuales en tablas dependientes del caso."""
        checks = {
            "cases": "SELECT COUNT(*) FROM cases WHERE case_id = ?",
            "documents": "SELECT COUNT(*) FROM documents WHERE case_id = ?",
            "fraud_analyses": "SELECT COUNT(*) FROM fraud_analyses WHERE case_id = ?",
            "fraud_correlations": "SELECT COUNT(*) FROM fraud_correlations WHERE case_id = ?",
            "runs": "SELECT COUNT(*) FROM runs WHERE case_id = ?",
            "cache_stats": "SELECT COUNT(*) FROM cache_stats WHERE scope = ?",
            "ocr_results": "SELECT COUNT(*) FROM ocr_results WHERE document_id IN (SELECT id FROM documents WHERE case_id = ?)",
            "extracted_data": "SELECT COUNT(*) FROM extracted_data WHERE document_id IN (SELECT id FROM documents WHERE case_id = ?)",
            "ai_analyses": "SELECT COUNT(*) FROM ai_analyses WHERE document_id IN (SELECT id FROM documents WHERE case_id = ?)"
        }

        summary: Dict[str, int] = {}
        with get_conn() as conn:
            for label, query in checks.items():
                summary[label] = int(conn.execute(query, (case_id,)).fetchone()[0])

        residual = {k: v for k, v in summary.items() if v}
        if residual:
            logger.warning("  ⚠️ Residuos en BD tras eliminar %s: %s", case_id, residual)
        else:
            logger.info(f"  ✓ BD verificada sin residuos para {case_id}")
        return summary

    def _cleanup_empty_cache_shards(self) -> int:
        cleaned = 0
        base = self.cache_manager.cache_dir
        if not base.exists():
            return cleaned

        for shard in base.iterdir():
            if shard.name == "case_index" or not shard.is_dir():
                continue
            try:
                next(shard.iterdir())
            except StopIteration:
                try:
                    shard.rmdir()
                    cleaned += 1
                except Exception:
                    pass
        if cleaned:
            logger.info(f"  ✓ Shards vacíos eliminados: {cleaned}")
        return cleaned

    def _verify_filesystem_cleanup(self, tokens: Set[str]) -> Dict[str, List[str]]:
        tokens_lower = {token for token in tokens if token}
        residuals: Dict[str, List[str]] = {}

        def _collect_matches(root: Path) -> List[str]:
            matches: List[str] = []
            if not root.exists():
                return matches
            for item in root.glob("**/*"):
                if not item.exists():
                    continue
                name_lower = item.name.lower()
                if any(token in name_lower for token in tokens_lower):
                    matches.append(str(item))
            return matches

        if tokens_lower:
            reports_dir = Path("data/reports")
            report_matches = _collect_matches(reports_dir)
            if report_matches:
                residuals['reports'] = report_matches

            gps_dir = Path("data/gps")
            gps_matches = _collect_matches(gps_dir)
            if gps_matches:
                residuals['gps'] = gps_matches

            pipeline_dir = Path(os.getenv("FS_DATA_DIR", "data")) / "temp" / "pipeline_cache"
            pipeline_matches = _collect_matches(pipeline_dir)
            if pipeline_matches:
                residuals['pipeline_cache'] = pipeline_matches

            base = self._chroma_base_path
            if base and base.exists():
                chroma_matches = []
                for path in base.iterdir():
                    if not path.exists():
                        continue
                    name_lower = path.name.lower()
                    if path.is_dir() and any(token in name_lower for token in tokens_lower):
                        chroma_matches.append(str(path))
                if chroma_matches:
                    residuals['vector_store'] = chroma_matches

            log_path = self._rick_audit_log
            if log_path and log_path.exists():
                try:
                    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
                        matches = [line.strip() for line in fh if any(token in line.lower() for token in tokens_lower)]
                        if matches:
                            residuals['rick_audit'] = matches[:10]
                except Exception:
                    pass

        if residuals:
            logger.warning("  ⚠️ Verificación FS encontró residuos: %s", {k: len(v) for k, v in residuals.items()})
        else:
            logger.info("  ✓ FS verificado sin residuos relevantes")
        return residuals

    def _comprehensive_cleanup_verification(self, case_id: str, tokens: Set[str]) -> Dict[str, List[str]]:
        tokens_lower = {token for token in tokens if token}
        residuals: Dict[str, List[str]] = {}

        def _collect(root: Path, include_dirs: bool = False) -> List[str]:
            matches: List[str] = []
            if not root.exists() or not tokens_lower:
                return matches
            for item in root.rglob("*"):
                if not item.exists():
                    continue
                if item.is_dir() and not include_dirs:
                    continue
                name_lower = item.name.lower()
                if any(token in name_lower for token in tokens_lower):
                    matches.append(str(item))
            return matches

        scan_dirs = {
            "reports": Path("data/reports"),
            "pipeline_cache": Path(os.getenv("FS_DATA_DIR", "data")) / "temp" / "pipeline_cache",
            "gps": Path("data/gps"),
            "ocr_cache": Path("data/ocr_cache"),
            "uploads": Path("data/uploads"),
            "temp": Path("data/temp"),
        }

        for label, directory in scan_dirs.items():
            include_dirs = label in {"gps", "ocr_cache", "temp"}
            matches = _collect(directory, include_dirs=include_dirs)
            if matches:
                residuals[label] = matches

        base = self._chroma_base_path
        if base and base.exists():
            chroma_matches = _collect(base, include_dirs=True)
            if chroma_matches:
                residuals["vector_store"] = chroma_matches

        log_path = self._rick_audit_log
        if log_path and log_path.exists():
            try:
                with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
                    lines = [
                        line.strip()
                        for line in fh
                        if any(token in line.lower() for token in tokens_lower)
                    ]
                if lines:
                    residuals["rick_audit"] = lines[:25]
            except Exception:
                pass

        return residuals

    def verify_complete_deletion(self, case_id: str) -> Dict[str, Any]:
        issues: List[Dict[str, Any]] = []

        db_snapshot = self._collect_db_cleanup_snapshot(case_id)
        if any(db_snapshot.values()):
            issues.append({"type": "database", "details": db_snapshot})

        index_path = self.cache_manager.index_dir / f"{case_id}.json"
        if index_path.exists():
            issues.append({"type": "case_index", "path": str(index_path)})

        case_index = self.cache_manager.get_case_index(case_id)
        insured = (case_index or {}).get("insured_name")
        claim = (case_index or {}).get("claim_number")
        tokens = self._build_cleanup_tokens(case_id=case_id, insured_name=insured, claim_number=claim)

        fs_residuals = self._comprehensive_cleanup_verification(case_id, tokens)
        if fs_residuals:
            issues.append({"type": "filesystem", "details": fs_residuals})

        return {
            "case_id": case_id,
            "clean": len(issues) == 0,
            "issues": issues,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché (FS) y métricas DB complementarias.
        - FS: refleja borrados manuales (usa OCRCacheManager.get_cache_stats())
        - DB: agrega conteos de tablas para diagnóstico (no rompe UI)
        """
        try:
            fs_stats = self.cache_manager.get_cache_stats()  # {'total_cases','total_cached_files','cache_size_mb', ...}
            # Mapear a los nombres usados por la UI para no romper nada
            out = {
                "case_count": fs_stats.get("total_cases", 0),
                "file_count": fs_stats.get("total_cached_files", 0),
                "total_size_mb": fs_stats.get("cache_size_mb", 0.0),
                "cache_directory": fs_stats.get("cache_directory", str(self.cache_manager.cache_dir))
            }

            # Métricas DB (complemento)
            try:
                with get_conn() as conn:
                    out["db_cases"] = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                    out["db_documents"] = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                    out["db_ocr_results"] = conn.execute("SELECT COUNT(*) FROM ocr_results").fetchone()[0]
                    out["db_extracted_data"] = conn.execute("SELECT COUNT(*) FROM extracted_data").fetchone()[0]
                    # Agregar métricas globales de cache_stats si existen
                    row = conn.execute("SELECT ocr_hits, ocr_misses, bytes_saved, ms_saved FROM cache_stats WHERE scope='global'").fetchone()
                    if row:
                        out["ocr_hits"] = row["ocr_hits"]
                        out["ocr_misses"] = row["ocr_misses"]
                        out["bytes_saved"] = row["bytes_saved"]
                        out["ms_saved"] = row["ms_saved"]
            except Exception:
                pass
            try:
                out["total_savings"] = float(get_total_savings())
            except Exception:
                out["total_savings"] = 0.0
            return out
        except Exception as e:
            logger.error(f"Error al calcular estadísticas del caché: {e}")
            return {"case_count": 0, "file_count": 0, "total_size_mb": 0}

    def list_available_cases(self) -> List[Dict[str, Any]]:
        """Lista los casos disponibles desde la base de datos o el índice de caché."""
        # Usar el método existente del cache_manager que ya funciona
        cases = self.cache_manager.list_cached_cases()
        result = []
        
        for case in cases:
            case_id = case.get("case_id")
            if not case_id:
                continue

            case_index = self.cache_manager.get_case_index(case_id, auto_reconstruct=True) or {}
            correlation_counts: Dict[str, Any] = {}
            correlation_report = case_index.get("fraud_correlations") or {}
            findings_payload = correlation_report.get("findings") or []
            if not findings_payload:
                try:
                    findings_payload = get_correlation_findings(case_id)
                except Exception:
                    findings_payload = []
            if findings_payload:
                correlation_counts = _summarize_correlation_counts(findings_payload)
            consolidated_fields = (
                (case_index.get("consolidated_data") or {}).get("consolidated_fields")
                or {}
            )

            insured_name = (
                consolidated_fields.get("nombre_asegurado")
                or case.get("insured_name")
                or ""
            )
            claim_number = (
                consolidated_fields.get("numero_siniestro")
                or case.get("claim_number")
                or case_id
            )

            document_count = case_index.get("total_documents")
            if document_count is None:
                document_count = case.get("total_documents")
            if document_count is None:
                documents = case_index.get("documents") or []
                document_count = len(documents)
            document_count = document_count or 0

            processed_at = (
                case_index.get("processed_at")
                or case.get("processed_at")
                or ""
            )

            claim_amount = self._extract_claim_amount(case_index)
            tentative_fraud = self._detect_tentative_fraud(case_index)
            case_row = get_case_by_id(case_id)
            tentative_decision = case_row["tentative_decision"] if case_row else None
            savings_amount = float(case_row["savings_amount"] or 0.0) if case_row else 0.0

            entry = {
                "case_id": case_id,
                "title": case.get("case_title", case_id),
                "case_title": case.get("case_title", case_id),
                "claim_number": claim_number,
                "insured_name": insured_name,
                "document_count": document_count,
                "processed_at": processed_at,
                "created_at": processed_at,
                "is_processed": bool(processed_at),
                "claim_amount": claim_amount,
                "tentative_fraud": tentative_fraud,
                "fraud_score": None,
                "tentative_decision": tentative_decision,
                "savings_amount": savings_amount,
                "correlation_counts": correlation_counts,
                "has_correlations": bool(correlation_counts.get("total")),
            }

            result.append(entry)
        
        return result

    async def process_replay(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Función principal que ejecuta el pipeline de replay.
        Delega a _core_replay_processing para la lógica centralizada.
        """
        return await self._core_replay_processing(config)

    def cleanup_db_orphans(self) -> dict:
        """Elimina orfandad en DB para mantener consistencia tras cambios manuales en FS."""
        from ..storage.db import get_conn, get_correlation_findings
        stats = {}
        with get_conn() as conn:
            stats['orphan_extracted_before'] = conn.execute(
                "SELECT COUNT(*) FROM extracted_data WHERE document_id NOT IN (SELECT id FROM documents)"
            ).fetchone()[0]
            stats['orphan_runs_before'] = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE case_id NOT IN (SELECT case_id FROM cases)"
            ).fetchone()[0]
            stats['orphan_ocr_before'] = conn.execute(
                "SELECT COUNT(*) FROM ocr_results WHERE document_id NOT IN (SELECT id FROM documents)"
            ).fetchone()[0]
            stats['orphan_correlations_before'] = conn.execute(
                "SELECT COUNT(*) FROM fraud_correlations WHERE case_id NOT IN (SELECT case_id FROM cases)"
            ).fetchone()[0]

            conn.execute("DELETE FROM extracted_data WHERE document_id NOT IN (SELECT id FROM documents)")
            conn.execute("DELETE FROM runs WHERE case_id NOT IN (SELECT case_id FROM cases)")
            conn.execute("DELETE FROM ocr_results WHERE document_id NOT IN (SELECT id FROM documents)")
            conn.execute("DELETE FROM fraud_correlations WHERE case_id NOT IN (SELECT case_id FROM cases)")

            stats['orphan_extracted_after'] = conn.execute(
                "SELECT COUNT(*) FROM extracted_data WHERE document_id NOT IN (SELECT id FROM documents)"
            ).fetchone()[0]
            stats['orphan_runs_after'] = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE case_id NOT IN (SELECT case_id FROM cases)"
            ).fetchone()[0]
            stats['orphan_ocr_after'] = conn.execute(
                "SELECT COUNT(*) FROM ocr_results WHERE document_id NOT IN (SELECT id FROM documents)"
            ).fetchone()[0]
            stats['orphan_correlations_after'] = conn.execute(
                "SELECT COUNT(*) FROM fraud_correlations WHERE case_id NOT IN (SELECT case_id FROM cases)"
            ).fetchone()[0]

        return stats

    async def deep_purge_case(self, case_id: str) -> bool:
        """
        Limpieza profunda de un caso: artefactos de FS + filas en DB (cases y cascada).
        """
        try:
            # 1) Limpiar artefactos de FS
            await self.purge_case(case_id)

            # 2) Eliminar caso en DB (cascada elimina documentos, ocr_results, extracted_data, runs)
            from ..storage.db import get_conn, get_correlation_findings
            with get_conn() as conn:
                deleted_corr = conn.execute(
                    "DELETE FROM fraud_correlations WHERE case_id = ?",
                    (case_id,)
                ).rowcount
                deleted_cases = conn.execute(
                    "DELETE FROM cases WHERE case_id = ?",
                    (case_id,)
                ).rowcount
                conn.commit()
                if deleted_corr:
                    logger.info(f"  ✓ Correlaciones eliminadas: {deleted_corr}")
                logger.info(f"  ✓ Eliminado de BD ({deleted_cases} fila(s) en cases)")

            # 3) Limpiar métricas de cache del caso (ya lo hace purge_case, pero reforzamos)
            from ..storage.db import reset_cache_stats
            reset_cache_stats(case_id)

            self._collect_db_cleanup_snapshot(case_id)

            logger.info(f"✅ Deep purge completado para {case_id}")
            return True
        except Exception as e:
            logger.error(f"Error en deep purge {case_id}: {e}")
            return False
    
    async def _core_replay_processing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lógica centralizada del sistema de replay.
        Esta función es usada tanto por la interfaz web como por la terminal.
        
        Args:
            config: Diccionario con las opciones del replay incluyendo:
                - case_id: ID del caso a procesar
                - use_ai: Boolean para usar AI o sistema legacy
                - output_dir: Directorio de salida para reportes
                - regenerate_report: Boolean para generar reportes
                - api_key: API key de OpenAI (opcional)
                - model: Modelo de AI a usar
                - temperature: Temperatura del modelo
                - per_doc: Boolean para análisis por documento
        
        Returns:
            Dict con los resultados del procesamiento
        """
        case_id = config["case_id"]
        logger.info(f"Iniciando replay para el caso: {case_id} con config: {config}")

        # Obtener información del caso
        case_index = self.cache_manager.get_case_index(case_id)
        if not case_index:
            raise ValueError(f"No se encontró información del caso {case_id}")

        cached_doc_types = self._build_case_document_types(case_index)

        # Cargar resultados OCR del cache
        ocr_results = []
        for doc_path in case_index.get('documents', []):
            doc_path = Path(doc_path)
            if self.cache_manager.has_cache(doc_path, case_id=case_id):
                ocr_result = self.cache_manager.get_cache(doc_path, case_id=case_id)
                normalized_name = self._normalize_case_filename(doc_path.name)
                cached_type = cached_doc_types.get(normalized_name)
                if not cached_type:
                    simplified = re.sub(r"[^a-z0-9.]+", "", normalized_name)
                    cached_type = cached_doc_types.get(simplified)
                if cached_type:
                    logger.debug(
                        "Tipo de documento recuperado del cache: %s → %s",
                        doc_path.name,
                        cached_type,
                    )
                ocr_results.append({
                    'filename': doc_path.name,
                    'ocr_result': ocr_result,
                    'document_type': cached_type
                })
            else:
                logger.warning(f"No hay cache para {doc_path.name}")

        if not ocr_results:
            raise RuntimeError("No se encontraron resultados OCR en cache")

        # Procesar según las opciones
        if config.get('use_ai'):
            return await self._process_with_ai(
                ocr_results=ocr_results,
                case_id=case_id,
                options=config
            )
        else:
            return await self._process_legacy(
                ocr_results=ocr_results,
                case_id=case_id,
                options=config
            )

    def _clean_existing_files(self, case_id: str, output_path: Path) -> None:
        """
        Elimina archivos existentes para un case_id antes de generar nuevos.
        
        Args:
            case_id: ID del caso
            output_path: Directorio donde buscar archivos existentes
        """
        patterns_to_clean = [
            f"INF-{case_id}.html",  # Formato antiguo
            f"INF-{case_id}.pdf",   # Formato antiguo
            f"INF-*-*.html",        # Formato nuevo
            f"INF-*-*.pdf",         # Formato nuevo
            f"replay_{case_id}_*.json",
            f"{case_id}_*.html",
            f"{case_id}_*.pdf"
        ]
        
        for pattern in patterns_to_clean:
            for file in output_path.glob(pattern):
                try:
                    file.unlink()
                    logger.info(f"Archivo existente eliminado: {file.name}")
                except Exception as e:
                    logger.warning(f"No se pudo eliminar {file.name}: {e}")
    
    async def _process_with_ai(
        self,
        ocr_results: List[Dict],
        case_id: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa con el sistema de IA usando los datos cacheados
        """
        # Asegurar que el directorio de salida existe
        output_dir = options.get('output_dir')
        if output_dir is None:
            output_dir = 'data/reports'
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio de salida: {output_path.absolute()}")
        
        # Obtener información del caso para nomenclatura
        case_index = self.cache_manager.get_case_index(case_id)
        if case_index:
            insured_name = case_index.get('insured_name', 'DESCONOCIDO')
            claim_number = case_index.get('claim_number', case_id)
        else:
            insured_name = 'DESCONOCIDO'
            claim_number = case_id
        
        # Limpiar archivos existentes antes de generar nuevos
        self._clean_existing_files(case_id, output_path)

        # 1) Resolver API key (UI/CLI > entorno)
        api_key = (options.get("api_key") or os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "No se encontró OPENAI_API_KEY. Cárgala desde .env o introdúcela en la UI/CLI."
            )
        # Exportar al entorno por si algún componente la lee de os.getenv
        if os.getenv("OPENAI_API_KEY") != api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # 2) Resolver config de IA (modelo/temperatura)
        model = options.get("model") or os.getenv("OPENAI_MODEL", "gpt-5")
        try:
            temperature = float(options.get("temperature", 0.1))
        except Exception:
            temperature = 0.1

        # 3) Inicializar componentes de IA
        extractor = AIFieldExtractor(api_key=api_key)
        consolidator = AIConsolidator(api_key=api_key)
        report_generator = AIReportGenerator()

        # Fase 1: Extracción
        logger.info("Extrayendo campos con IA...")
        extractions = await extractor.extract_from_documents_batch(
            documents=ocr_results,
            parallel_limit=3
        )

        # Fase 2: Consolidación
        logger.info("Consolidando datos...")
        consolidated = await consolidator.consolidate_extractions(
            extractions=extractions,
            case_id=case_id,
            use_advanced_reasoning=True
        )

        # Fase de análisis de fraude eliminada

        # Fase 4: Generar reporte si se solicita
        if options.get('regenerate_report', True):
            logger.info("Generando reporte...")
            
            # Sanitizar nombres para uso en archivos
            import re
            def sanitize_filename(name):
                if not name:
                    return "SIN_NOMBRE"
                return re.sub(r'[^a-zA-Z0-9_.-]+', '_', name).strip('_')
            
            s_insured = sanitize_filename(insured_name)
            s_claim = sanitize_filename(claim_number)
            
            html_path = output_path / f"INF-{s_insured}-{s_claim}.html"
            html_content = report_generator.generate_report(
                consolidated_data=consolidated,
                output_path=html_path
            )

            # Intentar generar PDF
            pdf_path = output_path / f"INF-{s_insured}-{s_claim}.pdf"
            report_generator.generate_pdf(html_content, pdf_path)

        # Preparar resultados
        try:
            consolidated_dict = consolidated.model_dump()
        except Exception:
            try:
                consolidated_dict = consolidated.dict()
            except Exception:
                consolidated_dict = consolidated

        try:
            extractions_list = [e.model_dump() for e in extractions]
        except Exception:
            try:
                extractions_list = [e.dict() for e in extractions]
            except Exception:
                extractions_list = extractions

        results = {
            "case_id": case_id,
            "replay_date": datetime.now().isoformat(),
            "options_used": {**options, "api_key": "***redacted***"},
            "extraction_results": extractions_list,
            "consolidated_data": consolidated_dict,
            "output_path": str(output_path)
        }

        verification_payload = verify_case_artifacts(case_id)
        results["post_process_verification"] = verification_payload

        # Guardar JSON de resultados
        json_path = output_path / f"replay_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Replay completado. Resultados en: {output_path}")
        return results

    async def _process_legacy(
        self,
        ocr_results: List[Dict],
        case_id: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa con el sistema legacy
        """
        logger.info("Procesamiento legacy no implementado aún")
        return {"status": "legacy_not_implemented"}

    def clear_cache(self, cases_to_delete: List[str]) -> Dict[str, Any]:
        """Limpia el caché para una lista de case_id o para todos si 'all' está en la lista."""
        from fraud_scorer.storage.cases import get_conn

        if "all" in cases_to_delete:
            # LIMPIEZA TOTAL DEL SISTEMA
            logger.info("🧹 Iniciando LIMPIEZA TOTAL del sistema...")
            try:
                # 1. Limpiar toda la base de datos
                logger.info("  → Limpiando base de datos...")
                with get_conn() as conn:
                    # Obtener todos los case_ids antes de limpiar
                    all_cases = conn.execute("SELECT case_id FROM cases").fetchall()
                    deleted_rows = {}

                    # Limpiar todas las tablas (ordenado dependencias primero)
                    deleted_rows['fraud_correlations'] = conn.execute("DELETE FROM fraud_correlations").rowcount
                    deleted_rows['fraud_analyses'] = conn.execute("DELETE FROM fraud_analyses").rowcount
                    deleted_rows['ai_analyses'] = conn.execute("DELETE FROM ai_analyses").rowcount
                    deleted_rows['extracted_data'] = conn.execute("DELETE FROM extracted_data").rowcount
                    deleted_rows['ocr_results'] = conn.execute("DELETE FROM ocr_results").rowcount
                    deleted_rows['documents'] = conn.execute("DELETE FROM documents").rowcount
                    deleted_rows['runs'] = conn.execute("DELETE FROM runs").rowcount
                    deleted_rows['cases'] = conn.execute("DELETE FROM cases").rowcount
                    deleted_rows['cache_stats'] = conn.execute("DELETE FROM cache_stats").rowcount
                    conn.commit()
                    logger.info(f"    ✓ {len(all_cases)} casos eliminados de la BD")
                    for table_name, count in deleted_rows.items():
                        logger.info(f"    • {table_name}: {count} filas eliminadas")

                    # Recrear registro global en cache_stats para mantener métricas coherentes
                    conn.execute(
                        "INSERT OR IGNORE INTO cache_stats(scope, updated_at) VALUES (?, CURRENT_TIMESTAMP)",
                        ('global',)
                    )
                    conn.commit()
                    remaining_cases = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
                    remaining_corr = conn.execute("SELECT COUNT(*) FROM fraud_correlations").fetchone()[0]
                    logger.info(
                        "    ✓ Verificación post-limpieza: %s casos en BD, %s correlaciones",
                        remaining_cases,
                        remaining_corr,
                    )

                # 2. Limpiar carpeta de cache OCR
                cache_dir = Path(getattr(self.cache_manager, "cache_dir", "data/ocr_cache"))
                if cache_dir.exists():
                    logger.info("  → Limpiando cache OCR...")
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("    ✓ Cache OCR limpiado")
                self._cleanup_empty_cache_shards()

                logger.info("  → Reiniciando índices vectoriales de Agente Rick...")
                if self._reset_vector_store():
                    logger.info("    ✓ Índices vectoriales reiniciados")

                logger.info("  → Reiniciando bitácora del Agente Rick...")
                if self._reset_rick_audit_log():
                    logger.info("    ✓ Bitácora limpiada")

                # 3. Limpiar carpetas temporales
                temp_dir = Path("data/temp")
                if temp_dir.exists():
                    logger.info("  → Limpiando carpetas temporales...")
                    for folder in temp_dir.iterdir():
                        if folder.is_dir():
                            shutil.rmtree(folder)
                    logger.info("    ✓ Carpetas temporales limpiadas")

                # 4. Limpiar reportes
                reports_dir = Path("data/reports")
                if reports_dir.exists():
                    logger.info("  → Limpiando reportes...")
                    for file in reports_dir.iterdir():
                        if file.is_file():
                            file.unlink()
                    logger.info("    ✓ Reportes limpiados")

                # 5. Limpiar uploads
                uploads_dir = Path("data/uploads")
                if uploads_dir.exists():
                    logger.info("  → Limpiando uploads...")
                    for item in uploads_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    logger.info("    ✓ Uploads limpiados")

                logger.info("🎆 LIMPIEZA TOTAL COMPLETADA - Sistema reiniciado")
                return {
                    "status": "success",
                    "message": "Sistema completamente limpiado. Todos los casos, cachés y archivos han sido eliminados."
                }
            except Exception as e:
                logger.error(f"Error durante limpieza total: {e}")
                return {"status": "error", "message": f"Error limpiando todo el sistema: {e}"}
        
        # LIMPIEZA DE CASOS INDIVIDUALES
        from fraud_scorer.storage.cases import get_conn
        cleared_cases = []
        errors = []

        for case_id in cases_to_delete:
            try:
                logger.info(f"🗑️ Eliminando caso {case_id} completamente...")

                # 1. Cargar información del caso antes de eliminarlo
                case_index = self.cache_manager.get_case_index(case_id)
                base_path = None
                insured_name = ''
                claim_number = ''

                # Obtener base_path de la BD
                with get_conn() as conn:
                    case_row = conn.execute(
                        "SELECT base_path FROM cases WHERE case_id = ?",
                        (case_id,)
                    ).fetchone()
                    if case_row:
                        base_path = case_row['base_path']

                if case_index:
                    insured_name = case_index.get('insured_name', '') or ''
                    claim_number = case_index.get('claim_number', '') or ''

                # 2. Eliminar de la base de datos (cascada elimina todo)
                with get_conn() as conn:
                    deleted_corr = conn.execute(
                        "DELETE FROM fraud_correlations WHERE case_id = ?",
                        (case_id,)
                    ).rowcount
                    deleted_cases = conn.execute(
                        "DELETE FROM cases WHERE case_id = ?",
                        (case_id,)
                    ).rowcount
                    conn.commit()
                    if deleted_corr:
                        logger.info(f"  ✓ Correlaciones eliminadas: {deleted_corr}")
                    logger.info(f"  ✓ Eliminado de BD ({deleted_cases} fila(s) en cases)")

                # 3. Limpiar carpeta base del caso (temp o reorganizada)
                base_folder_name = ''
                if base_path:
                    base_folder = Path(base_path)
                    base_folder_name = base_folder.name
                    if base_folder.exists():
                        if "temp" in str(base_folder):
                            # Es una carpeta temporal, eliminar completamente
                            shutil.rmtree(base_folder)
                            logger.info(f"  ✓ Carpeta temporal eliminada: {base_folder.name}")

                # Intentar inferir datos desde la carpeta si no estaban en el índice
                if base_folder_name and ' - ' in base_folder_name:
                    folder_insured, folder_claim = base_folder_name.split(' - ', 1)
                    if folder_insured and not insured_name:
                        insured_name = folder_insured.replace('_', ' ')
                    if folder_claim and not claim_number:
                        claim_number = folder_claim

                cleanup_tokens = self._build_cleanup_tokens(
                    case_id=case_id,
                    insured_name=insured_name,
                    claim_number=claim_number,
                )

                # 4. Limpiar archivos de cache OCR (shards)
                if case_index and "cache_files" in case_index:
                    for doc_path_str in case_index["cache_files"]:
                        try:
                            cache_path = self.cache_manager._get_cache_path(Path(doc_path_str))
                            if cache_path.exists():
                                cache_path.unlink()
                        except Exception:
                            pass
                    logger.info(f"  ✓ Cache OCR limpiado")

                # 5. Limpiar carpeta reorganizada
                try:
                    if insured_name or claim_number:
                        s_insured = self.cache_manager._sanitize_filename(insured_name or "")
                        s_claim = self.cache_manager._sanitize_filename(claim_number or case_id)
                        case_folder = self.cache_manager.cache_dir / f"{s_insured} - {s_claim}"
                        if case_folder.exists():
                            shutil.rmtree(case_folder)
                            logger.info(f"  ✓ Carpeta reorganizada eliminada")
                except Exception:
                    pass

                # 6. Limpiar índice del caso
                index_path = self.cache_manager.index_dir / f"{case_id}.json"
                if index_path.exists():
                    index_path.unlink()
                    logger.info(f"  ✓ Índice eliminado")

                for backup_path in self.cache_manager.index_dir.glob(f"{case_id}.json.backup_*"):
                    try:
                        backup_path.unlink()
                        logger.info(f"  ✓ Backup eliminado: {backup_path.name}")
                    except Exception:
                        pass

                # 7. Limpiar reportes del caso
                # Buscar reportes con el nombre del asegurado/reclamo si existe
                reports_to_delete = []

                # Patrones básicos con case_id y variaciones del nombre
                reports_patterns = {
                    f"*{case_id}*",
                    f"INF-{case_id}*",
                    f"replay_{case_id}*"
                }

                for variant in self._collect_name_variations(insured_name):
                    reports_patterns.add(f"*{variant}*")

                if claim_number:
                    reports_patterns.add(f"*{claim_number}*")

                # Buscar y eliminar reportes
                for pattern in reports_patterns:
                    reports_dir = Path("data/reports")
                    if reports_dir.exists():
                        for report in reports_dir.glob(pattern):
                            reports_to_delete.append(report)

                # Eliminar reportes únicos
                deleted_reports = set()
                for report in reports_to_delete:
                    if report not in deleted_reports:
                        try:
                            report.unlink()
                            deleted_reports.add(report)
                            logger.info(f"    - Eliminado: {report.name}")
                        except Exception as e:
                            logger.debug(f"    - No se pudo eliminar {report.name}: {e}")

                if deleted_reports:
                    logger.info(f"  ✓ {len(deleted_reports)} reportes eliminados")

                fallback_removed = self._remove_reports_by_case_id(case_id, cleanup_tokens)
                if fallback_removed:
                    logger.info(f"  ✓ Reportes eliminados por búsqueda directa: {fallback_removed}")

                residual_reports = self._remove_report_artifacts(cleanup_tokens)
                if residual_reports:
                    logger.info(f"  ✓ Reportes eliminados (escaneo): {residual_reports}")

                # 8. Limpiar archivos de pipeline_cache
                pipeline_cache_dir = Path("data/temp/pipeline_cache")
                if pipeline_cache_dir.exists():
                    pipeline_patterns = {
                        f"{case_id}*",
                        f"*{case_id}.status.jsonl"
                    }

                    if claim_number:
                        pipeline_patterns.add(f"*{claim_number}*")

                    deleted_pipeline = []
                    for pattern in pipeline_patterns:
                        for file in pipeline_cache_dir.glob(pattern):
                            try:
                                file.unlink()
                                deleted_pipeline.append(file.name)
                            except Exception:
                                pass

                    if deleted_pipeline:
                        logger.info(f"  ✓ Pipeline cache limpiado: {len(deleted_pipeline)} archivos")

                extra_pipeline = self._remove_pipeline_cache_artifacts(cleanup_tokens)
                if extra_pipeline:
                    logger.info(f"  ✓ Pipeline cache (escaneo) eliminado: {extra_pipeline}")

                self._remove_vector_indexes(cleanup_tokens)
                self._remove_gps_artifacts(cleanup_tokens)
                self._purge_rick_audit_log(cleanup_tokens)
                self._cleanup_empty_cache_shards()
                residuals = self._verify_filesystem_cleanup(cleanup_tokens)
                if residuals:
                    for category, items in residuals.items():
                        if category == 'rick_audit':
                            self._purge_rick_audit_log(cleanup_tokens)
                            continue
                        for item in items:
                            path = Path(item)
                            try:
                                if path.is_dir():
                                    shutil.rmtree(path)
                                elif path.exists():
                                    path.unlink()
                            except Exception as exc:
                                logger.debug("No se pudo eliminar residuo %s: %s", path, exc)
                    self._verify_filesystem_cleanup(cleanup_tokens)
                self._collect_db_cleanup_snapshot(case_id)
                logger.info(f"✅ Caso {case_id} completamente eliminado")
                cleared_cases.append(case_id)

            except Exception as e:
                logger.error(f"Error eliminando caso {case_id}: {e}")
                errors.append({"case_id": case_id, "error": str(e)})

        return {
            "status": "success",
            "message": f"Limpieza completada. Casos limpiados: {len(cleared_cases)}.",
            "cleared": cleared_cases,
            "errors": errors
        }
    
    async def purge_case(self, case_id: str) -> bool:
        """
        Limpia artefactos del caso especificado sin tocar métricas globales.
        """
        try:
            # Limpiar archivos del caso en el caché
            case_index = self.cache_manager.get_case_index(case_id)
            insured = (case_index or {}).get('insured_name') or ""
            claim = (case_index or {}).get('claim_number') or ""
            with get_conn() as conn:
                deleted_corr = conn.execute(
                    "DELETE FROM fraud_correlations WHERE case_id = ?",
                    (case_id,)
                ).rowcount
                if deleted_corr:
                    logger.info(f"  ✓ Correlaciones eliminadas (purge): {deleted_corr}")
            if case_index and "cache_files" in case_index:
                for doc_path_str in case_index["cache_files"]:
                    cache_path = self.cache_manager._get_cache_path(Path(doc_path_str))
                    if cache_path.exists():
                        cache_path.unlink()

            # Limpiar índice del caso
            index_path = self.cache_manager.index_dir / f"{case_id}.json"
            if index_path.exists():
                index_path.unlink()

            for backup_path in self.cache_manager.index_dir.glob(f"{case_id}.json.backup_*"):
                try:
                    backup_path.unlink()
                except Exception:
                    pass

            # Limpiar carpeta reorganizada si existe (Nombre - Reclamo)
            try:
                insured = (case_index or {}).get('insured_name') or ""
                claim = (case_index or {}).get('claim_number') or ""
                if insured or claim:
                    s_insured = self.cache_manager._sanitize_filename(insured)
                    s_claim = self.cache_manager._sanitize_filename(claim or case_id)
                    case_folder = self.cache_manager.cache_dir / f"{s_insured} - {s_claim}"
                    if case_folder.exists():
                        shutil.rmtree(case_folder)
            except Exception:
                pass

            # Limpiar archivos de status/progress
            base = os.getenv("FS_DATA_DIR", "data")
            status_file = Path(base) / "temp" / "pipeline_cache" / f"{case_id}.status.jsonl"
            if status_file.exists():
                status_file.unlink()

            tokens = self._build_cleanup_tokens(case_id=case_id, insured_name=insured, claim_number=claim)
            self._remove_report_artifacts(tokens)
            self._remove_reports_by_case_id(case_id, tokens)
            self._remove_pipeline_cache_artifacts(tokens)
            self._remove_vector_indexes(tokens)
            self._remove_gps_artifacts(tokens)
            self._purge_rick_audit_log(tokens)
            self._cleanup_empty_cache_shards()
            residuals = self._verify_filesystem_cleanup(tokens)
            if residuals:
                for category, items in residuals.items():
                    if category == 'rick_audit':
                        self._purge_rick_audit_log(tokens)
                        continue
                    for item in items:
                        path = Path(item)
                        try:
                            if path.is_dir():
                                shutil.rmtree(path)
                            elif path.exists():
                                path.unlink()
                        except Exception as exc:
                            logger.debug("No se pudo eliminar residuo %s: %s", path, exc)
                self._verify_filesystem_cleanup(tokens)

            # Resetear métricas del caso (pero no las globales)
            from ..storage.db import reset_cache_stats
            reset_cache_stats(case_id)

            logger.info(f"✅ Caso {case_id} purgado exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error purgando caso {case_id}: {e}")
            return False
    
    async def purge_orphans(self) -> int:
        """
        Elimina archivos de caché sin entrada correspondiente en la DB.
        """
        orphan_count = 0
        
        try:
            # Obtener todos los casos conocidos de la DB
            with get_conn() as conn:
                rows = conn.execute("SELECT DISTINCT case_id FROM cases").fetchall()
                valid_cases = {row["case_id"] for row in rows}

            # Mapear case_id -> nombre de carpeta reorganizada esperado
            case_folder_names: Dict[str, str] = {}
            for index_file in self.cache_manager.index_dir.glob("*.json"):
                case_id = index_file.stem
                try:
                    with open(index_file, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)
                except Exception:
                    continue

                case_folder = case_data.get('case_folder')
                if not case_folder:
                    insured = case_data.get('insured_name') or ""
                    claim = case_data.get('claim_number') or case_id
                    if insured or claim:
                        s_insured = self.cache_manager._sanitize_filename(insured)
                        s_claim = self.cache_manager._sanitize_filename(claim or case_id)
                        case_folder = f"{s_insured} - {s_claim}"

                if case_folder:
                    case_folder_names[case_id] = case_folder

            # Revisar archivos de índice
            for index_file in self.cache_manager.index_dir.glob("*.json"):
                case_id = index_file.stem
                if case_id not in valid_cases:
                    logger.info(f"Eliminando índice huérfano: {case_id}")
                    # Leer primero para conocer archivos asociados, luego eliminar el índice
                    case_data = None
                    try:
                        with open(index_file, 'r', encoding='utf-8') as f:
                            case_data = json.load(f)
                    except Exception:
                        case_data = None
                    
                    # Eliminar archivos de caché asociados (shards)
                    try:
                        for doc_path_str in (case_data or {}).get("cache_files", []):
                            cache_path = self.cache_manager._get_cache_path(Path(doc_path_str))
                            if cache_path.exists():
                                cache_path.unlink()
                                orphan_count += 1
                    except Exception:
                        pass
                    
                    # Eliminar carpeta reorganizada si existiera (Nombre - Reclamo)
                    try:
                        insured = (case_data or {}).get('insured_name') or ""
                        claim = (case_data or {}).get('claim_number') or ""
                        if insured or claim:
                            s_insured = self.cache_manager._sanitize_filename(insured)
                            s_claim = self.cache_manager._sanitize_filename(claim or case_id)
                            case_folder = self.cache_manager.cache_dir / f"{s_insured} - {s_claim}"
                            if case_folder.exists():
                                shutil.rmtree(case_folder)
                                orphan_count += 1
                    except Exception:
                        pass
                    
                    # Eliminar índice al final
                    try:
                        index_file.unlink()
                        orphan_count += 1
                    except Exception:
                        pass
            
            # Revisar carpetas reorganizadas
            valid_folder_names = set(case_folder_names.values())
            for folder in self.cache_manager.cache_dir.iterdir():
                if folder.is_dir() and folder.name != "case_index":
                    if folder.name in valid_folder_names:
                        continue

                    # Extraer case_id del nombre de la carpeta si es posible
                    folder_has_valid_case = False
                    for case_id, expected_name in case_folder_names.items():
                        if folder.name == expected_name:
                            folder_has_valid_case = True
                            break
                        if case_id in folder.name:
                            folder_has_valid_case = True
                            break

                    if not folder_has_valid_case and "-" in folder.name:
                        logger.info(f"Eliminando carpeta huérfana: {folder.name}")
                        shutil.rmtree(folder)
                        orphan_count += 1
            
            logger.info(f"✅ {orphan_count} archivos huérfanos eliminados")
            return orphan_count
            
        except Exception as e:
            logger.error(f"Error eliminando huérfanos: {e}")
            return orphan_count
