#!/usr/bin/env python3
import sys
import asyncio
import argparse
from pathlib import Path
import logging
from typing import Dict, Any
import json
from datetime import datetime

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
    from fraud_scorer.templates.template_processor import TemplateProcessor
    from fraud_scorer.storage.db import (
        get_conn,
        create_run,
        save_ai_analysis,
    )
    from fraud_scorer.pipelines.utils import (
        load_case_header,
        build_docs_for_template_from_db,
        render_report,
        save_json_snapshot,
    )
except ImportError:
    from pathlib import Path as _Path
    _REPO_ROOT = _Path(__file__).resolve().parents[1]
    _SRC_PATH = _REPO_ROOT / "src"
    if _SRC_PATH.exists():
        sys.path.insert(0, str(_SRC_PATH))
    from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
    from fraud_scorer.templates.template_processor import TemplateProcessor
    from fraud_scorer.storage.db import (
        get_conn,
        create_run,
        save_ai_analysis,
    )
    from fraud_scorer.pipelines.utils import (
        load_case_header,
        build_docs_for_template_from_db,
        render_report,
        save_json_snapshot,
    )


class ReplaySystem:
    def __init__(self, model: str = "gpt-4-turbo-preview", temperature: float = 0.3):
        self.ai = AIDocumentAnalyzer()
        self.template = TemplateProcessor()
        self.model = model
        self.temperature = float(temperature)

    async def replay(
        self,
        case_id: str,
        out_dir: Path,
        no_save: bool = False,
        per_doc: bool = False
    ) -> Dict[str, Any]:
        # 1. Cargar metadatos del caso
        header = load_case_header(case_id)

        # 2. Obtener documentos y estructura para IA / plantilla
        docs_for_template, processed_docs = build_docs_for_template_from_db(case_id)
        if not docs_for_template:
            raise RuntimeError("No hay documentos OCR/extraídos para este caso.")

        # 3. Análisis IA del caso completo
        print("\n🤖 IA (caso completo, sin OCR)...")
        ai_case = await self.ai.analyze_claim_documents(docs_for_template)

        # 4. Guardar corrida IA en DB (opcional)
        run_id = None
        if not no_save:
            run_id = create_run(
                case_id=case_id,
                purpose="replay_case",
                llm_model=self.model,
                params={"temperature": self.temperature, "from": "replay"},
            )

        # 5. (Opcional) Análisis por documento
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
                ai_doc = await self.ai.analyze_document(doc_struct)
                save_ai_analysis(
                    document_id=d.get("_document_id"),
                    run_id=run_id,
                    ai=ai_doc,
                    model=self.model,
                    temperature=self.temperature,
                )

        # 6. Preparar resumen
        fraud_score = float(ai_case.get("fraud_score", 0.0))
        risk_level = "bajo" if fraud_score < 0.3 else ("medio" if fraud_score < 0.6 else "alto")
        results = {
            "case_id": case_id,
            "case_name": header.get("name"),
            "processing_date": datetime.now().isoformat(),
            "documents_processed": len(processed_docs),
            "documents": processed_docs,
            "fraud_analysis": {
                "fraud_score": round(fraud_score * 100, 2),
                "risk_level": risk_level,
                "indicators": ai_case.get("fraud_indicators", []),
                "inconsistencies": ai_case.get("inconsistencies", []),
                "external_validations": ai_case.get("external_validations", []),
                "route_analysis": ai_case.get("route_analysis", {}),
            },
            "ai_analysis_raw": ai_case,
        }

        # 7. Renderizar informe HTML/PDF
        numero_siniestro = getattr(
            self.template.extract_from_documents(docs_for_template, ai_case),
            "numero_siniestro",
            case_id
        )
        render_report(self.template, docs_for_template, ai_case, out_dir, numero_siniestro)

        # 8. Guardar snapshot JSON
        save_json_snapshot(results, out_dir / f"resultados_{case_id}.json")

        return results


def parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reanaliza un caso usando SOLO los datos en la DB (sin OCR).")
    p.add_argument("--case-id", required=True, help="ID del caso (ej. CASE-2025-0001)")
    p.add_argument("--out", default="data/reports", help="Carpeta de salida")
    p.add_argument("--no-save", action="store_true", help="No guarda corridas/análisis en la DB")
    p.add_argument("--per-doc", action="store_true", help="(Opcional) ejecutar y guardar análisis IA por documento")
    p.add_argument("--model", default="gpt-4-turbo-preview", help="Modelo LLM para IA")
    p.add_argument("--temperature", type=float, default=0.3, help="Temperatura del modelo")
    return p.parse_args(argv)


async def main(argv: list) -> None:
    args = parse_args(argv)
    system = ReplaySystem(model=args.model, temperature=args.temperature)
    out_dir = Path(args.out)
    await system.replay(case_id=args.case_id, out_dir=out_dir, no_save=args.no_save, per_doc=args.per_doc)
    print("\n✅ Replay terminado")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
