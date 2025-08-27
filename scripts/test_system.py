#!/usr/bin/env python3
"""
Script de Prueba y Chequeo de Salud para Fraud Scorer v2.0

Verifica el entorno, las dependencias, las credenciales y la integridad
de los componentes clave del nuevo pipeline de IA.
"""
import os
import sys
import asyncio
from pathlib import Path

# --- Configuración Inicial ---
# Añadir la raíz del proyecto al path de Python para encontrar los módulos de 'src'
project_root = Path(__file__).resolve().parents[1]
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    # Fallback si colorama no está instalado
    class Fore:
        CYAN = GREEN = RED = YELLOW = BLUE = ""
    class Style:
        RESET_ALL = ""

# --- Funciones de Verificación ---

def check_environment():
    """Verifica el entorno, las dependencias y las variables de entorno."""
    print(f"\n{Fore.CYAN}🔍 Verificando Entorno de Desarrollo...{Style.RESET_ALL}\n")
    all_good = True

    # 1. Verificar Versión de Python
    py_version = sys.version_info
    if py_version.major == 3 and py_version.minor >= 10:
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Python {py_version.major}.{py_version.minor} instalado.")
    else:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Se requiere Python 3.10+ (tienes {py_version.major}.{py_version.minor}).")
        all_good = False

    # 2. Verificar Dependencias Críticas
    imports = [
        ("azure.ai.formrecognizer", "Azure Form Recognizer"),
        ("openai", "OpenAI"),
        ("jinja2", "Jinja2"),
        ("weasyprint", "WeasyPrint (para PDFs)"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("pandas", "Pandas (para .xlsx/.csv)"),
        ("docx", "python-docx (para .docx)"),
    ]
    for module, name in imports:
        try:
            __import__(module)
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} Dependencia '{name}' instalada.")
        except ImportError:
            print(f"{Fore.RED}✗{Style.RESET_ALL} Dependencia '{name}' NO instalada. Ejecuta 'pip install -r requirements.txt'.")
            all_good = False

    # 3. Verificar Variables de Entorno
    print(f"\n{Fore.CYAN}🔑 Verificando Credenciales (.env)...{Style.RESET_ALL}\n")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print(f"{Fore.YELLOW}⚠{Style.RESET_ALL} 'python-dotenv' no instalado. No se pudo cargar .env.")

    env_vars = [
        ("AZURE_ENDPOINT", "Azure Endpoint"),
        ("AZURE_OCR_KEY", "Azure OCR Key"),
        ("OPENAI_API_KEY", "OpenAI API Key"),
    ]
    for var, name in env_vars:
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." if len(value) > 8 else value
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} Credencial '{name}' encontrada: {masked}")
        else:
            print(f"{Fore.RED}✗{Style.RESET_ALL} Credencial '{name}' NO encontrada en el entorno.")
            all_good = False

    # 4. Verificar Estructura de Directorios de la v2
    print(f"\n{Fore.CYAN}📁 Verificando Estructura de Directorios Clave...{Style.RESET_ALL}\n")
    dirs = [
        "data/raw", "data/reports", "data/temp",
        "src/fraud_scorer/api", "src/fraud_scorer/parsers",
        "src/fraud_scorer/processors/ai", "src/fraud_scorer/storage",
        "src/fraud_scorer/templates",
    ]
    for dir_path in dirs:
        full_path = project_root / dir_path
        if full_path.is_dir():
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} Directorio: {dir_path}")
        else:
            print(f"{Fore.RED}✗{Style.RESET_ALL} Directorio NO encontrado: {dir_path}")
            all_good = False

    return all_good

async def test_azure_connection():
    """Prueba la conexión con el servicio de Azure Document Intelligence."""
    print(f"\n{Fore.CYAN}🔌 Probando Conexión con Azure...{Style.RESET_ALL}\n")
    try:
        from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
        AzureOCRProcessor()
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Conexión con Azure configurada correctamente.")
        return True
    except Exception as e:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Error al inicializar el cliente de Azure: {e}")
        return False

async def test_openai_connection():
    """Prueba la conexión con la API de OpenAI."""
    print(f"\n{Fore.CYAN}🤖 Probando Conexión con OpenAI...{Style.RESET_ALL}\n")
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=1, timeout=10)
        await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5,
        )
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Conexión con OpenAI y la API key funcionan.")
        return True
    except Exception as e:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Error conectando con OpenAI: {e}")
        return False

async def test_v2_pipeline():
    """Crea y procesa un documento de prueba usando el pipeline de la v2."""
    print(f"\n{Fore.CYAN}📄 Probando Pipeline de Procesamiento v2...{Style.RESET_ALL}\n")
    
    # 1. Crear documento de prueba
    test_dir = project_root / "data/temp/test_docs"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "documento_prueba_v2.txt"
    test_content = "Número de Póliza: 123-ABC-789. Monto: $1,234.56"
    test_file.write_text(test_content, encoding="utf-8")
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} Documento de prueba creado en: {test_file}")

    try:
        # 2. Probar el DocumentParser (Punto de entrada de la v2)
        from fraud_scorer.parsers.document_parser import DocumentParser
        from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor

        # (No se llamará a Azure para un .txt, se usará el parser nativo)
        parser = DocumentParser(AzureOCRProcessor())
        parsed_doc = parser.parse_document(test_file)

        assert parsed_doc is not None, "El parser devolvió None."
        assert "text" in parsed_doc, "La salida del parser no tiene la clave 'text'."
        assert parsed_doc["text"] == test_content, "El texto parseado no coincide."
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} DocumentParser procesó el archivo correctamente.")

        # 3. Prueba de humo (smoke test) del AIFieldExtractor
        from fraud_scorer.processors.ai.ai_field_extractor import AIFieldExtractor
        extractor = AIFieldExtractor()
        # Preparamos un payload similar al que recibiría el extractor
        doc_payload = {
            "filename": test_file.name,
            "ocr_result": parsed_doc,
            "document_type": "otro"
        }
        extraction_result = await extractor.extract_from_documents_batch([doc_payload])
        assert isinstance(extraction_result, list), "El extractor no devolvió una lista."
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} AIFieldExtractor se ejecutó sin errores (smoke test).")
        
        print(f"\n{Fore.GREEN}✓{Style.RESET_ALL} Pipeline de procesamiento v2 parece estar funcionando correctamente.")
        return True

    except Exception as e:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Falló la prueba del pipeline v2: {e}")
        return False
    finally:
        # Limpieza
        if test_file.exists():
            test_file.unlink()

async def main():
    """Función principal de prueba"""
    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   FRAUD SCORER v2.0 - SCRIPT DE CHEQUEO DEL SISTEMA{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")

    env_ok = check_environment()
    if not env_ok:
        print(f"\n{Fore.YELLOW}⚠️  Hay problemas críticos con el entorno. Por favor, corrígelos antes de continuar.{Style.RESET_ALL}")
        print(f"   Asegúrate de haber ejecutado 'pip install -r requirements.txt' y de tener un archivo .env con las credenciales.")
        return

    # Ejecutar pruebas de conexión y pipeline
    azure_ok = await test_azure_connection()
    openai_ok = await test_openai_connection()
    pipeline_ok = await test_v2_pipeline()

    # Resumen
    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   RESUMEN DE PRUEBAS{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}\n")
    
    results = [
        ("Entorno y Dependencias", env_ok),
        ("Conexión a Azure", azure_ok),
        ("Conexión a OpenAI", openai_ok),
        ("Pipeline de Procesamiento v2", pipeline_ok)
    ]
    
    all_passed = all(r[1] for r in results)

    for name, passed in results:
        status = f"{Fore.GREEN}✓ PASÓ{Style.RESET_ALL}" if passed else f"{Fore.RED}✗ FALLÓ{Style.RESET_ALL}"
        print(f"  {name}: {status}")
    
    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    
    if all_passed:
        print(f"{Fore.GREEN}🎉 ¡ENHORABUENA! Todas las pruebas pasaron.{Style.RESET_ALL}")
        print(f"\nEl sistema está listo para ser utilizado. Próximos pasos sugeridos:")
        print(f"  - Iniciar la infraestructura: {Fore.CYAN}./start.sh{Style.RESET_ALL}")
        print(f"  - Procesar un caso: {Fore.CYAN}python scripts/run_report.py /ruta/a/tus/documentos{Style.RESET_ALL}")
        print(f"  - Usar el modo Replay: {Fore.CYAN}python scripts/replay_case.py{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️  Algunas pruebas fallaron. Revisa los mensajes de error para solucionar los problemas.{Style.RESET_ALL}")
    
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    asyncio.run(main())