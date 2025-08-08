#!/usr/bin/env python3
import sys
import asyncio
from pathlib import Path
from jinja2 import exceptions as j2exc

# importa tu función principal:
from processors.templates.template_processor import process_claim
async def main():
    if len(sys.argv) != 3:
        print("Uso: run_report.py <carpeta_docs> <output_pdf>")
        sys.exit(1)

    folder_in = Path(sys.argv[1])
    out_pdf  = Path(sys.argv[2])

    if not folder_in.exists() or not folder_in.is_dir():
        print(f"❌ carpeta no existe: {folder_in}")
        sys.exit(1)

    # 1. Genera primero el HTML:
    html_path = out_pdf.with_suffix(".html")
    print(f"📝 Generando HTML en {html_path} …")
    await process_claim(str(folder_in), str(html_path.parent))

    if not html_path.exists():
        print("❌ no se creó el HTML, revisa errores anteriores.")
        sys.exit(1)

    # 2. Convierte a PDF con WeasyPrint
    from weasyprint import HTML
    print(f"📦 Convirtiendo a PDF en {out_pdf} …")
    HTML(filename=str(html_path)).write_pdf(str(out_pdf))
    print("✅ Informe PDF listo:", out_pdf)

if __name__ == "__main__":
    asyncio.run(main())
