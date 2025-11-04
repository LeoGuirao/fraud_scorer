#!/usr/bin/env python3
"""
Extrae campos clave de un CFDI XML sin depender de la IA.
Útil para validar la presencia de sello digital y UUID antes de enviar al pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional
import xml.etree.ElementTree as ET


def parse_cfdi(path: Path) -> Dict[str, Optional[str]]:
    tree = ET.parse(path)
    root = tree.getroot()

    # Normalizar namespaces
    ns_map: Dict[str, str] = {"cfdi": "http://www.sat.gob.mx/cfd/4"}
    for key, value in root.attrib.items():
        if key.startswith("{http://www.w3.org/2000/xmlns/}"):
            prefix = key.split("}", 1)[1]
            ns_map[prefix] = value

    def find(tag: str) -> Optional[ET.Element]:
        for prefix, uri in ns_map.items():
            element = root.find(f".//{{{uri}}}{tag}")
            if element is not None:
                return element
        return None

    comprobante = root
    emisor = find("Emisor")
    receptor = find("Receptor")
    timbre = find("TimbreFiscalDigital")

    sello = comprobante.attrib.get("Sello") if comprobante is not None else None
    sello_clean = "".join((sello or "").strip().split()) or None
    signature_last_8 = sello_clean[-8:] if sello_clean and len(sello_clean) >= 8 else None

    return {
        "issuer_rfc": (emisor.attrib.get("Rfc") if emisor is not None else None),
        "recipient_rfc": (receptor.attrib.get("Rfc") if receptor is not None else None),
        "total": comprobante.attrib.get("Total") if comprobante is not None else None,
        "uuid": timbre.attrib.get("UUID") if timbre is not None else None,
        "fecha_emision": comprobante.attrib.get("Fecha") if comprobante is not None else None,
        "sello_digital_sat": sello_clean,
        "signature_last_8": signature_last_8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae campos críticos de un CFDI (XML).")
    parser.add_argument("cfdi_path", type=Path, help="Ruta al CFDI XML.")
    parser.add_argument("--json", action="store_true", help="Devuelve la salida en JSON.")
    args = parser.parse_args()

    if not args.cfdi_path.exists():
        print(f"❌ Archivo no encontrado: {args.cfdi_path}", file=sys.stderr)
        sys.exit(1)

    try:
        payload = parse_cfdi(args.cfdi_path)
    except Exception as exc:
        print(f"❌ Error procesando CFDI: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=== CFDI Extraction Preview ===")
        for key, value in payload.items():
            print(f"{key:18}: {value or 'NO DISPONIBLE'}")

        if payload["sello_digital_sat"] and len(payload["sello_digital_sat"]) < 100:
            print("⚠️  Sello digital parece incompleto (<100 caracteres)")


if __name__ == "__main__":
    main()
