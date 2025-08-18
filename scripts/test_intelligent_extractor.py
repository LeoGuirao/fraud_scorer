# test_intelligent_extractor.py

#!/usr/bin/env python3
"""
Script para probar el IntelligentFieldExtractor con documentos reales
"""

import json
import logging
from pathlib import Path
from fraud_scorer.extractors.intelligent_extractor import IntelligentFieldExtractor

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_extractor():
    """Prueba el extractor con un documento de ejemplo"""
    
    # Documento de prueba (basado en tu JSON)
    test_document = {
        'document_type': 'poliza',
        'raw_text': """
        Asunto: Reclamación formal de siniestro - Póliza No .: 141 - 2346 Inciso:1
        Nombre: MODA YKT, S.A. de C.V.
        Domicilio Fiscal: EMILIANO ZAPATA 19 PISO No. Ext. 2 LOC. B Y No. Int: PISO 3, 
        SAN JERÓNIMO TEPETLACALCO. C.P 54090. TLALNEPANTLA DE BAZ. ESTADO DE MEXICO
        Vigencia: 12:00 HRS. DEL 26/JUL/2024 AL 26/JUL/2025
        RFC: HEAP5602278N7
        Total: $1,008,399.46
        Fecha del siniestro: 15 de enero de 2025
        """,
        'key_value_pairs': {
            'asunto': 'Reclamación formal de siniestro - Póliza No .: 141 - 2346 Inciso:1',
            'nombre': 'MODA YKT, S.A. de C.V.'
        }
    }
    
    # Inicializar extractor
    extractor = IntelligentFieldExtractor()
    
    # Extraer campos
    print("\n" + "="*60)
    print("PROBANDO EXTRACTOR INTELIGENTE")
    print("="*60)
    
    results = extractor.extract_all_fields(test_document, debug=True)
    
    print("\n" + "="*60)
    print("RESULTADOS DE EXTRACCIÓN")
    print("="*60)
    
    for field_name, result in results.items():
        print(f"\n📋 {field_name}:")
        print(f"   Valor: {result.value}")
        print(f"   Confianza: {result.confidence:.2%}")
        print(f"   Estrategia: {result.strategy}")
        if result.source:
            print(f"   Fuente: {result.source}")
    
    # Mostrar estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE EXTRACCIÓN")
    print("="*60)
    
    stats = extractor.get_stats_report()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    return results

if __name__ == "__main__":
    test_extractor()