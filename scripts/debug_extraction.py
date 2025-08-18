#!/usr/bin/env python3
import sys
import json
import logging
from pathlib import Path
from collections import Counter
import re
import sqlite3

# Asegura que `src/` está en el path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_extraction")

from fraud_scorer.storage.db import get_conn  # noqa: E402


# ---------------- utils de introspección ----------------
def get_table_names(conn) -> set:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {r[0] for r in cur.fetchall()}


def get_table_columns(conn, table_name: str) -> set:
    cur = conn.cursor()
    try:
        cur.execute(f"PRAGMA table_info({table_name})")
        cols = [r[1] for r in cur.fetchall()]
        return set(cols)
    except sqlite3.OperationalError:
        return set()


def find_first(cols: set, candidates: list) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def json_safe_load(s):
    try:
        return json.loads(s) if isinstance(s, str) else (s or {})
    except Exception:
        return {}


def infer_doc_type(extracted_dict: dict) -> str:
    """Intenta deducir el tipo de documento desde el JSON de extracción."""
    if not extracted_dict:
        return ""
    for path in [
        ("document_type",),
        ("document_kind",),
        ("type",),
        ("specific_fields", "document_type"),
        ("key_value_pairs", "document_type"),
    ]:
        d = extracted_dict
        ok = True
        for k in path:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                ok = False
                break
        if ok and isinstance(d, str):
            return d
    return ""


# ---------------- carga de documentos con tipos ----------------
def fetch_documents_with_types(case_id: str):
    """
    Devuelve lista de dicts:
      {document_id, file_name, doc_type, extracted_json, extracted}
    - Se adapta a columnas reales (id/document_id/doc_id, file_name/filename/path, etc.).
    - Si no existe case_id en 'documents', intenta traer todo y filtrar nada.
    """
    conn = get_conn()
    cur = conn.cursor()

    tables = get_table_names(conn)
    if "documents" not in tables:
        raise RuntimeError("No existe la tabla 'documents' en la base de datos.")

    doc_cols = get_table_columns(conn, "documents")
    # Detectar columnas clave en documents
    doc_id_col = find_first(doc_cols, ["document_id", "id", "doc_id"])
    file_name_col = find_first(doc_cols, ["file_name", "filename", "name", "source_name", "path", "filepath"])
    doc_type_col = find_first(doc_cols, ["document_type", "type"])
    case_id_col = find_first(doc_cols, ["case_id", "case", "caseId"])

    if not doc_id_col:
        raise RuntimeError("No se encontró columna de id en 'documents' (busqué: document_id, id, doc_id).")
    if not file_name_col:
        # Aún así intentaremos continuar con un nombre sintético
        logger.warning("No se encontró columna de nombre de archivo en 'documents'. Usaré 'desconocido'.")
    if not case_id_col:
        logger.warning("No encontré columna case_id en 'documents'. Cargaré TODOS los documentos.")

    # Construir SELECT dinámico
    select_cols = [f"d.{doc_id_col} as doc_id"]
    if file_name_col:
        select_cols.append(f"d.{file_name_col} as file_name")
    else:
        select_cols.append(f"'' as file_name")
    if doc_type_col:
        select_cols.append(f"d.{doc_type_col} as doc_type")
    else:
        select_cols.append(f"'' as doc_type")

    # Vamos a traer extracted_data aparte y mapear en Python (más robusto)
    sql_docs = f"""
        SELECT {", ".join(select_cols)}
        FROM documents d
        {"WHERE d." + case_id_col + " = ?" if case_id_col else ""}
        ORDER BY d.{doc_id_col}
    """
    params = (case_id,) if case_id_col else ()
    cur.execute(sql_docs, params)
    doc_rows = cur.fetchall()

    # --- Cargar extracted_data (si existe) y mapear por id ---
    extracted_map = {}
    if "extracted_data" in tables:
        ed_cols = get_table_columns(conn, "extracted_data")
        ed_id_col = find_first(ed_cols, ["document_id", "doc_id", "id"])
        ed_json_col = find_first(ed_cols, ["extracted_data", "data", "json", "payload", "content"])
        if ed_id_col and ed_json_col:
            # Traemos todo y mapeamos (base local, suele ser pequeña)
            cur.execute(f"SELECT {ed_id_col}, {ed_json_col} FROM extracted_data")
            for rid, rjson in cur.fetchall():
                extracted_map[rid] = rjson
        else:
            logger.warning(
                "Tabla 'extracted_data' no tiene columnas esperadas "
                "(id: document_id/doc_id/id, json: extracted_data/data/json). "
                "Continuaré sin extracción."
            )
    else:
        logger.warning("No existe tabla 'extracted_data'. Continuaré sin extracción.")

    docs = []
    for row in doc_rows:
        # row -> (doc_id, file_name, doc_type)
        doc_id = row[0]
        file_name = row[1] if len(row) > 1 else "desconocido"
        doc_type = (row[2] if len(row) > 2 else "") or ""

        extracted_json = extracted_map.get(doc_id, None)
        extracted = json_safe_load(extracted_json)

        if not doc_type:
            doc_type = infer_doc_type(extracted) or ""

        docs.append(
            {
                "document_id": doc_id,
                "file_name": file_name,
                "doc_type": (doc_type or "").lower(),
                "extracted_json": extracted_json,
                "extracted": extracted,
            }
        )

    conn.close()
    return docs


# ---------------- reportes / debug ----------------
def check_document_types(case_id: str):
    """Imprime la distribución de tipos de documento para el caso."""
    docs = fetch_documents_with_types(case_id)
    cnt = Counter((d["doc_type"] or "desconocido") for d in docs)
    print("\n📊 Distribución de tipos de documento:")
    for t, c in cnt.most_common():
        print(f"   {t}: {c}")


def debug_poliza_extraction(case_id: str):
    """Debug detallado de extracción de pólizas."""
    docs = fetch_documents_with_types(case_id)
    polizas = [d for d in docs if d["doc_type"] in ("poliza", "poliza_seguro")]

    print("\n" + "=" * 60)
    print(f"ANÁLISIS DE PÓLIZAS - {case_id}")
    print("=" * 60)
    print(f"Encontradas: {len(polizas)} pólizas\n")

    # Regex útiles
    re_poliza = re.compile(r"p[óo]liza\s*[:№#]?\s*([A-Z0-9\-/]+)", re.IGNORECASE)
    re_asegurado = re.compile(r"asegurado\s*:\s*([^\n]{5,80})", re.IGNORECASE)

    for d in polizas:
        file_name = d["file_name"]
        doc_type = d["doc_type"]
        extracted = d["extracted"] or {}

        print(f"\n📄 {file_name}")
        print(f"   Tipo: {doc_type}")

        if extracted:
            sf = extracted.get("specific_fields", {}) or {}
            kv = extracted.get("key_value_pairs", {}) or {}
            raw_text = extracted.get("raw_text", "") or ""

            print(f"\n   specific_fields ({len(sf)} campos):")
            shown = 0
            for k, v in sf.items():
                if v:
                    print(f"      {k}: {str(v)[:80]}")
                    shown += 1
                if shown >= 10:
                    break

            print(f"\n   key_value_pairs ({len(kv)} pares):")
            relevant_keys = [
                "poliza",
                "póliza",
                "numero_poliza",
                "número de póliza",
                "asegurado",
                "nombre_asegurado",
                "contratante",
                "vigencia",
                "desde",
                "hasta",
                "del",
                "al",
                "domicilio",
                "direccion",
                "dirección",
            ]
            shown = 0
            for k, v in kv.items():
                k_lower = k.lower()
                if any(rel in k_lower for rel in relevant_keys):
                    print(f"      ✓ {k}: {str(v)[:100]}")
                    shown += 1
                if shown >= 20:
                    break

            if raw_text:
                preview = raw_text[:500]
                print(f"\n   raw_text preview (500 chars):")
                print("      " + preview.replace("\n", " ")[:500])

                pol_match = re_poliza.search(raw_text)
                if pol_match:
                    print(f"      🔍 Posible póliza en texto: {pol_match.group(1)}")

                aseg_match = re_asegurado.search(raw_text)
                if aseg_match:
                    print(f"      🔍 Posible asegurado: {aseg_match.group(1).strip()}")
        else:
            print("   ⚠️ Sin datos extraídos")


# ---------------- main ----------------
if __name__ == "__main__":
    case_id = sys.argv[1] if len(sys.argv) > 1 else "CASE-2025-0005"
    check_document_types(case_id)
    debug_poliza_extraction(case_id)
