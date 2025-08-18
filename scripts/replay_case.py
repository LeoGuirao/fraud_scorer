#!/usr/bin/env python3
import sys
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fraud_scorer.replay_case")

# ----------------------------
# Imports con fallback a src/
# ----------------------------
try:
    from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
    # 👇 forzamos el procesador segmentado y traemos el loader de config
    from fraud_scorer.pipelines.segmented_processor import (
        SegmentedTemplateProcessor,
        load_pipeline_config,  # solo lo importamos; el TP leerá el YAML si se lo pasas
    )
    from fraud_scorer.storage.db import (
        get_conn,
        create_run,
        save_ai_analysis,
    )
    from fraud_scorer.pipelines.utils import (
        load_case_header,
        build_docs_for_template_from_db,
        save_json_snapshot,
    )
except ImportError:
    from pathlib import Path as _Path
    _REPO_ROOT = _Path(__file__).resolve().parents[1]
    _SRC_PATH = _REPO_ROOT / "src"
    if _SRC_PATH.exists():
        sys.path.insert(0, str(_SRC_PATH))
    from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
    from fraud_scorer.pipelines.segmented_processor import (
        SegmentedTemplateProcessor,
        load_pipeline_config,
    )
    from fraud_scorer.storage.db import (
        get_conn,
        create_run,
        save_ai_analysis,
    )
    from fraud_scorer.pipelines.utils import (
        load_case_header,
        build_docs_for_template_from_db,
        save_json_snapshot,
    )


# ----------------------------
# Helpers (resumenes ligeros)
# ----------------------------
def _summarize_docs_for_results(processed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Crea un resumen compacto por documento para el JSON final (evita guardar OCR completo).
    """
    summary = []
    for d in processed_docs:
        ex = d.get("extracted_data") or {}
        kv_sf = ex.get("specific_fields") or {}
        doc_type = (
            d.get("document_type")
            or ex.get("document_type")
            or ((d.get("ocr_data") or {}).get("metadata") or {}).get("document_type")
            or "unknown"
        )
        file_name = d.get("file_name") or ((d.get("ocr_data") or {}).get("metadata") or {}).get("source_name")
        summary.append(
            {
                "file_name": file_name,
                "document_type": doc_type,
                "extracted_fields_count": len(kv_sf),
            }
        )
    return summary


def _consolidated_summary(consolidated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce el tamaño del consolidado a lo útil para auditoría.
    """
    return {
        "case_info": consolidated.get("case_info", {}),
        "validation_summary": consolidated.get("validation_summary", {}),
        "processing_stats": consolidated.get("processing_stats", {}),
    }


class ReplaySystem:
    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.3,
        config_path: Optional[str] = None,
    ):
        self.ai = AIDocumentAnalyzer()
        # 👇 siempre segmentado; si pasas YAML se inyecta al orquestador
        self.template = SegmentedTemplateProcessor(config_path=config_path)
        self.model = model
        self.temperature = float(temperature)
        self.config_path = config_path

        if config_path:
            try:
                cfg = load_pipeline_config(config_path)
                logger.info(f"Config de pipeline cargada desde {config_path}: {list(cfg.keys())}")
            except Exception as e:
                logger.warning(f"No se pudo cargar config YAML ({config_path}): {e}")

    async def _build_ai_case_from_consolidated(
        self, consolidated: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        En lugar de pasar TODOS los docs a la IA, pasamos
        un único 'documento' sintético con el resumen consolidado.
        """
        synthetic_doc = {
            "document_type": "consolidated_summary",
            "raw_text": json.dumps(
                {
                    "case_info": consolidated.get("case_info", {}),
                    "validation_summary": consolidated.get("validation_summary", {}),
                    "processing_stats": consolidated.get("processing_stats", {}),
                },
                ensure_ascii=False,
            ),
            "entities": [],
            "key_value_pairs": consolidated.get("case_info", {}),
            "specific_fields": consolidated.get("case_info", {}),
            "ocr_metadata": {"source_name": "consolidated_summary.json"},
        }
        return await self.ai.analyze_claim_documents([synthetic_doc])

    def _inject_document_ids(
        self,
        docs_for_template: List[Dict[str, Any]],
        processed_docs: List[Dict[str, Any]],
    ) -> None:
        """
        Intenta adjuntar `_document_id` a cada doc normalizado para que
        `--per-doc` pueda guardar IA por documento. Empata por filename
        (metadata.source_name) y, si falla, usa fallback por índice.
        """
        # Mapa por nombre de archivo
        name_to_id: Dict[str, Any] = {}
        for pd in processed_docs:
            fname = (pd.get("file_name") or "").strip().lower()
            if fname and pd.get("document_id"):
                name_to_id[fname] = pd["document_id"]

        # Inyección por filename
        matched = 0
        for d in docs_for_template:
            meta = d.get("ocr_metadata", {}) or {}
            source_name = (meta.get("source_name") or meta.get("file_name") or "").strip().lower()
            if source_name and source_name in name_to_id:
                d["_document_id"] = name_to_id[source_name]
                matched += 1

        # Fallback por índice si nada coincidió
        if matched == 0 and len(docs_for_template) == len(processed_docs):
            for i, d in enumerate(docs_for_template):
                d["_document_id"] = processed_docs[i].get("document_id")

    async def replay(
        self,
        case_id: str,
        out_dir: Path,
        no_save: bool = False,
        per_doc: bool = False,
        clobber: bool = False,
    ) -> Dict[str, Any]:
        # 1) Cargar metadatos del caso
        header = load_case_header(case_id)

        # 2) Obtener documentos (normalizados) desde la DB
        docs_for_template, processed_docs = build_docs_for_template_from_db(case_id)
        if not docs_for_template:
            raise RuntimeError("No hay documentos OCR/extraídos para este caso.")

        # 2.1) Inyectar document_id a docs normalizados para IA por-doc
        self._inject_document_ids(docs_for_template, processed_docs)

        # 3) (Opcional) Limpiar salidas previas
        if clobber:
            out_dir.mkdir(parents=True, exist_ok=True)
            targets = list(out_dir.glob("INF-*.html")) + list(out_dir.glob("INF-*.pdf"))
            targets += list(out_dir.glob(f"resultados_{case_id}.json"))
            removed = 0
            for pth in targets:
                try:
                    pth.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
            if removed:
                logger.info(f"--clobber: {removed} archivo(s) previos eliminados en {out_dir}")

        # 4) Procesamiento segmentado (aislado) y consolidación
        logger.info("Procesando documentos en modo segmentado/aislado…")
        consolidated = await self.template.orchestrator.process_documents(docs_for_template)

        # 5) Análisis IA liviano usando SOLO el consolidado
        print("\n🤖 IA (resumen consolidado, sin OCR)…")
        ai_case = await self._build_ai_case_from_consolidated(consolidated)

        # 6) Guardar corrida IA en DB (opcional)
        run_id: Optional[str] = None
        if not no_save:
            run_id = create_run(
                case_id=case_id,
                purpose="replay_case_segmented",
                llm_model=self.model,
                params={
                    "temperature": self.temperature,
                    "from": "replay_segmented",
                    **({"config": self.config_path} if self.config_path else {}),
                },
            )

        # 7) (Opcional) IA por documento (con aislamiento ya cubierto por el pipeline)
        if per_doc and not no_save:
            print("🧪 Guardando análisis IA por-documento…")
            for d in docs_for_template:
                doc_struct = {
                    "document_type": d.get("document_type"),
                    "raw_text": d.get("raw_text", ""),
                    "entities": d.get("entities", []),
                    "key_value_pairs": d.get("key_value_pairs", {}),
                    "specific_fields": d.get("specific_fields", {}),
                    "ocr_metadata": d.get("ocr_metadata", {}),
                }
                try:
                    ai_doc = await self.ai.analyze_document(doc_struct)
                except Exception as e:
                    logger.warning(f"IA por-doc falló: {e}")
                    continue
                doc_id = d.get("_document_id")
                if doc_id:
                    try:
                        save_ai_analysis(
                            document_id=doc_id,
                            run_id=run_id,
                            ai=ai_doc,
                            model=self.model,
                            temperature=self.temperature,
                        )
                    except Exception as e:
                        logger.warning(f"No se pudo guardar análisis IA de documento {doc_id}: {e}")

        # 8) Preparar resumen técnico (JSON ligero)
        fraud_score = float(ai_case.get("fraud_score", 0.0))
        risk_level = "bajo" if fraud_score < 0.3 else ("medio" if fraud_score < 0.6 else "alto")
        results = {
            "case_id": case_id,
            "case_name": header.get("name"),
            "processing_date": datetime.now().isoformat(),
            "documents_processed": len(processed_docs),
            # 👇 Solo resumen, NO OCR completo
            "documents_summary": _summarize_docs_for_results(processed_docs),
            "consolidated_summary": _consolidated_summary(consolidated),
            "fraud_analysis": {
                "fraud_score": round(fraud_score * 100, 2),
                "risk_level": risk_level,
                "indicators": (ai_case.get("fraud_indicators", []) or [])[:5],
                "inconsistencies": (ai_case.get("inconsistencies", []) or [])[:5],
                "external_validations": ai_case.get("external_validations", []),
                "route_analysis": ai_case.get("route_analysis", {}),
            },
            "ai_analysis_raw": ai_case,
        }

        # 9) Construir Informe (a partir del pipeline segmentado)
        print("\n📝 Generando informe (segmentado)…")
        informe = self.template._build_informe_from_consolidated(consolidated, ai_case)
        # ✅ Número de siniestro estable: usa el case_id (o déjalo vacío si prefieres)
        informe.numero_siniestro = case_id  # o: case_id.split('-')[-1]

        # 10) Generar HTML/PDF directamente
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / f"INF-{informe.numero_siniestro}.html"
        self.template.generate_report(informe, str(html_path))

        pdf_path = None
        try:
            from weasyprint import HTML  # opcional
            pdf_path = html_path.with_suffix(".pdf")
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        except ImportError:
            logger.warning("WeasyPrint no instalado. PDF omitido.")
        except Exception as e:
            logger.warning(f"Error generando PDF: {e}")

        logger.info(f"HTML: {html_path}" + (f" | PDF: {pdf_path}" if pdf_path else " | PDF omitido"))

        # 11) Guardar snapshot JSON técnico (ligero)
        out_json = out_dir / f"resultados_{case_id}.json"
        save_json_snapshot(results, out_json)
        print(f"✓ Resultados JSON guardados: {out_json}")

        return results


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reanaliza un caso usando SOLO los datos en la DB (modo segmentado, sin OCR)."
    )
    p.add_argument("--case-id", required=True, help="ID del caso (ej. CASE-2025-0001)")
    p.add_argument("--out", default="data/reports", help="Carpeta de salida")
    p.add_argument("--no-save", action="store_true", help="No guarda corridas/análisis en la DB")
    p.add_argument("--per-doc", action="store_true", help="(Opcional) ejecutar y guardar análisis IA por documento")
    p.add_argument("--model", default="gpt-4-turbo-preview", help="Modelo LLM para IA")
    p.add_argument("--temperature", type=float, default=0.3, help="Temperatura del modelo")
    p.add_argument(
        "--clobber",
        action="store_true",
        help="Borra INF-*.html/pdf previos en --out antes de generar nuevos",
    )
    # 👇 NUEVO: config YAML opcional para el pipeline segmentado
    p.add_argument("--config", dest="config", default=None, help="Ruta a pipeline_config.yaml")
    return p.parse_args(argv)


async def main(argv: List[str]) -> None:
    args = parse_args(argv)
    system = ReplaySystem(
        model=args.model,
        temperature=args.temperature,
        config_path=args.config,  # 👈 pasamos el YAML al TP
    )
    out_dir = Path(args.out or "data/reports")

    # Limpieza temprana si se pide --clobber
    if args.clobber:
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = list(out_dir.glob("INF-*.html")) + list(out_dir.glob("INF-*.pdf"))
        if args.case_id:
            targets += list(out_dir.glob(f"resultados_{args.case_id}.json"))
        for pth in targets:
            try:
                pth.unlink()
            except FileNotFoundError:
                pass

    await system.replay(
        case_id=args.case_id,
        out_dir=out_dir,
        no_save=args.no_save,
        per_doc=args.per_doc,
        clobber=False,  # ya limpiamos arriba si se solicitó
    )
    print("\n✅ Replay terminado")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
