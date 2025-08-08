#!/usr/bin/env python3
"""
Script de prueba para verificar que todo funciona correctamente
"""
import os
import sys
from pathlib import Path
from colorama import Fore, Style, init

# Inicializar colorama
init()

def check_environment():
    """Verifica el entorno y las dependencias"""
    print(f"\n{Fore.CYAN}🔍 Verificando entorno...{Style.RESET_ALL}\n")
    
    all_good = True
    
    # Verificar Python
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 10:
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Python {python_version.major}.{python_version.minor} instalado")
    else:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Python 3.10+ requerido (actual: {python_version.major}.{python_version.minor})")
        all_good = False
    
    # Verificar imports críticos
    imports_to_check = [
        ("azure.ai.formrecognizer", "Azure Form Recognizer"),
        ("openai", "OpenAI"),
        ("jinja2", "Jinja2"),
        ("weasyprint", "WeasyPrint"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic")
    ]
    
    for module_name, display_name in imports_to_check:
        try:
            __import__(module_name)
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} {display_name} instalado")
        except ImportError:
            print(f"{Fore.RED}✗{Style.RESET_ALL} {display_name} NO instalado")
            all_good = False
    
    # Verificar variables de entorno
    print(f"\n{Fore.CYAN}🔑 Verificando credenciales...{Style.RESET_ALL}\n")
    
    env_vars = [
        ("AZURE_ENDPOINT", "Azure Endpoint"),
        ("AZURE_OCR_KEY", "Azure OCR Key"),
        ("OPENAI_API_KEY", "OpenAI API Key")
    ]
    
    from dotenv import load_dotenv
    load_dotenv()
    
    for var_name, display_name in env_vars:
        value = os.getenv(var_name)
        if value:
            # Mostrar solo los primeros caracteres por seguridad
            masked_value = value[:10] + "..." if len(value) > 10 else value
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} {display_name}: {masked_value}")
        else:
            print(f"{Fore.RED}✗{Style.RESET_ALL} {display_name}: NO CONFIGURADO")
            all_good = False
    
    # Verificar directorios
    print(f"\n{Fore.CYAN}📁 Verificando estructura de directorios...{Style.RESET_ALL}\n")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/reports",
        "data/temp",
        "templates",
        "processors/ocr",
        "processors/ai"
    ]
    
    for dir_path in directories:
        if Path(dir_path).exists():
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} {dir_path}")
        else:
            print(f"{Fore.YELLOW}⚠{Style.RESET_ALL} {dir_path} - Creando...")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return all_good

def test_azure_connection():
    """Prueba la conexión con Azure OCR"""
    print(f"\n{Fore.CYAN}🔌 Probando conexión con Azure...{Style.RESET_ALL}\n")
    
    try:
        from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor
        processor = AzureOCRProcessor()
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Conexión con Azure establecida")
        return True
    except Exception as e:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Error conectando con Azure: {e}")
        return False

def test_openai_connection():
    """Prueba la conexión con OpenAI"""
    print(f"\n{Fore.CYAN}🤖 Probando conexión con OpenAI...{Style.RESET_ALL}\n")
    
    try:
        import openai
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Hacer una llamada simple de prueba
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Di 'OK' si funciona"}],
            max_tokens=10
        )
        
        if response.choices[0].message.content:
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} Conexión con OpenAI establecida")
            return True
    except Exception as e:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Error conectando con OpenAI: {e}")
        return False

def test_sample_document():
    """Crea y procesa un documento de prueba"""
    print(f"\n{Fore.CYAN}📄 Creando documento de prueba...{Style.RESET_ALL}\n")
    
    # Crear un archivo de texto simple de prueba
    test_dir = Path("data/temp/test_docs")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / "documento_prueba.txt"
    test_content = """
    CARTA DE RECLAMACIÓN
    
    Fecha: 15 de Enero de 2025
    Número de Siniestro: TEST-2025-001
    Nombre del Asegurado: Juan Pérez González
    RFC: PEGJ850215XXX
    Número de Póliza: 123-456-789
    
    Por medio de la presente, solicito la indemnización por el siniestro
    ocurrido el día 10 de enero de 2025, consistente en el robo de mercancía
    con un valor de $50,000.00 MXN.
    
    Placa del vehículo: ABC-123-D
    
    Atentamente,
    Juan Pérez González
    """
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} Documento de prueba creado: {test_file}")
    
    # Intentar procesar el documento
    try:
        print(f"\n{Fore.CYAN}🔍 Procesando documento de prueba...{Style.RESET_ALL}\n")
        
        from fraud_scorer.processors.ocr.document_extractor import UniversalDocumentExtractor
        
        extractor = UniversalDocumentExtractor()
        
        # Simular resultado OCR
        mock_ocr_result = {
            'text': test_content,
            'tables': [],
            'key_value_pairs': {},
            'entities': []
        }
        
        result = extractor.extract_structured_data(mock_ocr_result)
        
        print(f"  Tipo de documento detectado: {result.get('document_type', 'No detectado')}")
        
        if 'entities' in result:
            print(f"  Entidades encontradas:")
            for entity_type, values in result['entities'].items():
                print(f"    - {entity_type}: {values}")
        
        print(f"\n{Fore.GREEN}✓{Style.RESET_ALL} Procesamiento exitoso")
        return True
        
    except Exception as e:
        print(f"{Fore.RED}✗{Style.RESET_ALL} Error procesando documento: {e}")
        return False

def main():
    """Función principal de prueba"""
    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   FRAUD SCORER - PRUEBA DEL SISTEMA{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    
    # Verificar entorno
    env_ok = check_environment()
    
    if not env_ok:
        print(f"\n{Fore.YELLOW}⚠️  Hay problemas con el entorno.{Style.RESET_ALL}")
        print(f"Por favor, ejecuta: {Fore.CYAN}bash scripts/setup_macos.sh{Style.RESET_ALL}")
        return
    
    # Probar conexiones
    azure_ok = test_azure_connection()
    openai_ok = test_openai_connection()
    
    # Probar procesamiento
    processing_ok = test_sample_document()
    
    # Resumen
    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   RESUMEN DE PRUEBAS{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}\n")
    
    results = [
        ("Entorno", env_ok),
        ("Azure OCR", azure_ok),
        ("OpenAI", openai_ok),
        ("Procesamiento", processing_ok)
    ]
    
    all_passed = all(r[1] for r in results)
    
    for name, passed in results:
        status = f"{Fore.GREEN}✓ PASÓ{Style.RESET_ALL}" if passed else f"{Fore.RED}✗ FALLÓ{Style.RESET_ALL}"
        print(f"  {name}: {status}")
    
    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    
    if all_passed:
        print(f"{Fore.GREEN}🎉 ¡TODAS LAS PRUEBAS PASARON!{Style.RESET_ALL}")
        print(f"\nEl sistema está listo para usar:")
        print(f"  {Fore.CYAN}python scripts/run_report.py /ruta/a/carpeta/documentos{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️  Algunas pruebas fallaron.{Style.RESET_ALL}")
        print(f"Revisa los errores arriba y verifica tu configuración.")
    
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()