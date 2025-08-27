#!/usr/bin/env python3
"""
Script de inicio rápido para el servidor web de Fraud Scorer
"""
import os
import sys
import socket
from pathlib import Path
from rich.console import Console
import uvicorn

console = Console()

# --- Asegurar layout "src" en sys.path y para procesos hijo (reload) ---
ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Importante: para el proceso del reloader
os.environ.setdefault("PYTHONPATH", str(SRC))

def get_local_ip() -> str:
    """Obtiene la IP local de la máquina (mejor heurística LAN)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def main():
    host = os.getenv("FS_HOST", "0.0.0.0")
    port = int(os.getenv("FS_PORT", "8000"))
    reload_flag = os.getenv("FS_RELOAD", "0") == "1"
    workers = int(os.getenv("FS_WORKERS", "1"))

    # Con reload solo 1 worker
    if reload_flag and workers != 1:
        console.print("[yellow]⚠️  FS_RELOAD=1 requiere FS_WORKERS=1; ajustando automáticamente.[/yellow]")
        workers = 1

    local_url = f"http://localhost:{port}"
    lan_url = f"http://{get_local_ip()}:{port}"

    console.print("\n" + "=" * 60)
    console.print("🚀 FRAUD SCORER - SERVIDOR WEB")
    console.print("=" * 60)
    console.print(f"\n📍 El servidor está iniciando...")
    console.print(f"\n🌐 URLs de acceso:")
    console.print(f"   • Local:    {local_url}")
    console.print(f"   • Red:      {lan_url}")
    console.print(f"\n📝 Comparte la URL de red con tus compañeros")
    console.print(f"   (deben estar en la misma red WiFi/LAN)")
    console.print(f"\n⚠️  Para detener el servidor: CTRL+C")
    console.print("=" * 60 + "\n")

    # MUY IMPORTANTE: pasar la app como import string (NO el objeto)
    uvicorn.run(
        "fraud_scorer.api.web_interface:app",
        host=host,
        port=port,
        reload=reload_flag,
        workers=workers,
        log_level="info",
    )

if __name__ == "__main__":
    main()
