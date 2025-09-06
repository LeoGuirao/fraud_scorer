#!/usr/bin/env python3
"""
Test para verificar que los modelos óptimos se seleccionan correctamente según la investigación
"""

import sys
from pathlib import Path

# Añadir el src al path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from fraud_scorer.settings import get_model_for_task, ModelType


def test_optimal_model_selection():
    """Test que verifica la selección óptima de modelos según investigación 2025"""
    print("🧪 TESTEO DE SELECCIÓN ÓPTIMA DE MODELOS GPT-5")
    print("=" * 70)
    
    # Test cases basados en la investigación
    test_cases = [
        # Extracción con AI Directo (visión) - debe usar GPT-5 Mini
        ("extraction", "direct_ai", "gpt-5-mini", "Visión: GPT-5 Mini (95% más económico, optimizado para extraction)"),
        
        # Extracción con OCR + AI (texto) - debe usar GPT-5 completo  
        ("extraction", "ocr_text", "gpt-5", "Texto: GPT-5 (272K tokens, ideal para documentos complejos)"),
        
        # Consolidación - debe usar GPT-5 completo
        ("consolidation", "ocr_text", "gpt-5", "Consolidación: GPT-5 (razonamiento complejo)"),
        
        # Generación - debe usar GPT-5 Mini
        ("generation", "ocr_text", "gpt-5-mini", "Generación: GPT-5 Mini (eficiente)"),
        
        # Fallback para tarea desconocida
        ("unknown_task", "ocr_text", "gpt-4o-mini", "Fallback: GPT-4o Mini (compatibilidad)"),
    ]
    
    print("Verificando selección de modelos:")
    success_count = 0
    
    for task, route, expected_model, description in test_cases:
        try:
            actual_model = get_model_for_task(task, route)
            
            if actual_model == expected_model:
                success_count += 1
                print(f"✅ {description}")
                print(f"   {task} + {route} → {actual_model}")
            else:
                print(f"❌ {description}")
                print(f"   {task} + {route} → {actual_model} (esperaba {expected_model})")
        except Exception as e:
            print(f"❌ {description}")
            print(f"   ERROR: {e}")
        
        print()
    
    # Resumen de modelos disponibles
    print("=" * 70)
    print("MODELOS DISPONIBLES EN EL SISTEMA:")
    print("=" * 70)
    
    models = [
        ("GPT-5 (Visión)", ModelType.GPT5_VISION.value, "Para AI directo con visión"),
        ("GPT-5 Mini (Visión)", ModelType.GPT5_VISION_MINI.value, "RECOMENDADO para extraction + visión"),
        ("GPT-5 Nano (Visión)", ModelType.GPT5_VISION_NANO.value, "Para tareas simples de clasificación"),
        ("GPT-5", ModelType.GPT5.value, "Para texto + 272K context"),
        ("GPT-5 Mini", ModelType.GPT5_MINI.value, "Para generación eficiente"),
        ("GPT-4o Mini (Fallback)", ModelType.EXTRACTOR.value, "Compatibilidad"),
    ]
    
    for name, model_id, usage in models:
        print(f"• {name:25} | {model_id:15} | {usage}")
    
    # Resultados finales
    print("\n" + "=" * 70)
    print("VENTAJAS DE LA CONFIGURACIÓN OPTIMIZADA:")
    print("=" * 70)
    print("• 🔥 AI Directo: GPT-5 Mini es 95% MÁS ECONÓMICO que GPT-5 estándar")
    print("• 📄 OCR + AI: GPT-5 con 272K tokens (2x más contexto que GPT-4o)")
    print("• 🧠 Consolidación: GPT-5 completo para razonamiento avanzado")
    print("• ⚡ Generación: GPT-5 Mini para eficiencia en reportes")
    
    print(f"\n📊 Resultado: {success_count}/{len(test_cases)} configuraciones correctas")
    
    if success_count == len(test_cases):
        print("✅ CONFIGURACIÓN ÓPTIMA VERIFICADA")
        print("✨ El sistema usará los mejores modelos GPT-5 para cada tarea")
        return True
    else:
        print("❌ Hay configuraciones subóptimas")
        return False


if __name__ == "__main__":
    success = test_optimal_model_selection()
    sys.exit(0 if success else 1)