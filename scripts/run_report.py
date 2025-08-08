#!/usr/bin/env python3
"""
Script para generar informes de siniestros a partir de una carpeta de documentos
Uso: python scripts/run_report.py /ruta/a/carpeta/documentos
"""
import sys
import os
import asyncio
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging
import argparse
from tqdm import tqdm
import colorama
from colorama import Fore, Style

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos del proyecto
from fraud_scorer.processors.ocr.azure_ocr import AzureOCRProcessor, OCRResult
from fraud_scorer.processors.ocr.document_extractor import UniversalDocumentExtractor
from fraud_scorer.processors.ai.document_analyzer import AIDocumentAnalyzer
from templates.template_processor import TemplateProcessor
from dotenv import load_dotenv

# Inicializar colorama para colores en terminal
colorama.init()

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FraudScorerRunner:
    """
    Clase principal para ejecutar el análisis completo de siniestros
    """
    
    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar credenciales
        self._verify_credentials()
        
        # Inicializar procesadores
        print(f"{Fore.CYAN}Inicializando procesadores...{Style.RESET_ALL}")
        self.ocr_processor = AzureOCRProcessor()
        self.document_extractor = UniversalDocumentExtractor()
        self.ai_analyzer = AIDocumentAnalyzer()
        self.template_processor = TemplateProcessor()
        print(f"{Fore.GREEN}✓ Procesadores inicializados{Style.RESET_ALL}")
    
    def _verify_credentials(self):
        """Verifica que las credenciales estén configuradas"""
        required_vars = ['AZURE_ENDPOINT', 'AZURE_OCR_KEY', 'OPENAI_API_KEY']
        missing = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)
        
        if missing:
            print(f"{Fore.RED}❌ Faltan variables de entorno: {', '.join(missing)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Por favor configura estas variables en el archivo .env{Style.RESET_ALL}")
            sys.exit(1)
    
    async def process_folder(self, folder_path: str) -> str:
        """
        Procesa todos los documentos de una carpeta y genera el informe
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"La carpeta no existe: {folder_path}")
        
        print(f"\n{Fore.BLUE}═══════════════════════════════════════════════════════{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📁 Procesando carpeta: {folder.name}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}═══════════════════════════════════════════════════════{Style.RESET_ALL}\n")
        
        # Buscar archivos de documentos
        document_files = self._find_documents(folder)
        
        if not document_files:
            print(f"{Fore.RED}❌ No se encontraron documentos válidos en la carpeta{Style.RESET_ALL}")
            return None
        
        print(f"{Fore.GREEN}✓ Encontrados {len(document_files)} documentos{Style.RESET_ALL}")
        
        # Procesar cada documento
        ocr_results = []
        extracted_data = []
        
        print(f"\n{Fore.CYAN}🔍 Ejecutando OCR en documentos...{Style.RESET_ALL}")
        
        with tqdm(total=len(document_files), desc="Procesando", unit="doc") as pbar:
            for doc_file in document_files:
                try:
                    # OCR
                    pbar.set_description(f"OCR: {doc_file.name[:30]}...")
                    ocr_result = await self._process_document_ocr(doc_file)
                    ocr_results.append(ocr_result)
                    
                    # Extracción estructurada
                    structured = self.document_extractor.extract_structured_data(ocr_result)
                    structured['file_path'] = str(doc_file)
                    structured['file_name'] = doc_file.name
                    extracted_data.append(structured)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n{Fore.YELLOW}⚠️  Error procesando {doc_file.name}: {e}{Style.RESET_ALL}")
                    pbar.update(1)
                    continue
        
        print(f"\n{Fore.GREEN}✓ OCR completado para {len(ocr_results)} documentos{Style.RESET_ALL}")
        
        # Análisis con AI
        print(f"\n{Fore.CYAN}🤖 Analizando con AI...{Style.RESET_ALL}")
        ai_analysis = await self._analyze_with_ai(extracted_data)
        
        # Generar informe
        print(f"\n{Fore.CYAN}📝 Generando informe...{Style.RESET_ALL}")
        report_path = await self._generate_report(extracted_data, ai_analysis, folder.name)
        
        print(f"\n{Fore.GREEN}═══════════════════════════════════════════════════════{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ INFORME GENERADO EXITOSAMENTE{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📄 Archivo: {report_path}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}═══════════════════════════════════════════════════════{Style.RESET_ALL}\n")
        
        return report_path
    
    def _find_documents(self, folder: Path) -> List[Path]:
        """Encuentra todos los documentos válidos en la carpeta"""
        valid_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.docx'}
        documents = []
        
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in valid_extensions:
                # Ignorar archivos temporales o del sistema
                if not file.name.startswith('.') and not file.name.startswith('~'):
                    documents.append(file)
        
        # Ordenar por nombre para procesamiento consistente
        return sorted(documents)
    
    async def _process_document_ocr(self, file_path: Path) -> Dict[str, Any]:
        """Procesa un documento con OCR"""
        # Si es PDF, convertir primera página a imagen para análisis visual
        if file_path.suffix.lower() == '.pdf':
            # Por ahora usar el PDF directamente
            # TODO: Implementar conversión PDF a imagen si es necesario
            pass
        
        # Ejecutar OCR
        ocr_result = self.ocr_processor.analyze_document(str(file_path))
        
        # Agregar metadata del archivo
        ocr_result['source_file'] = file_path.name
        ocr_result['file_type'] = file_path.suffix.lower()
        
        return ocr_result
    
    async def _analyze_with_ai(self, extracted_data: List[Dict]) -> Dict[str, Any]:
        """Analiza los documentos con AI para detectar fraude e inconsistencias"""
        
        # Preparar contexto para el análisis
        context = {
            "documents_count": len(extracted_data),
            "document_types": list(set(d.get('document_type', 'unknown') for d in extracted_data)),
            "claim_data": {}
        }
        
        # Extraer información clave de todos los documentos
        for doc in extracted_data:
            doc_type = doc.get('document_type', 'unknown')
            
            # Agregar entidades encontradas al contexto
            if 'entities' in doc:
                for entity_type, values in doc['entities'].items():
                    if entity_type not in context['claim_data']:
                        context['claim_data'][entity_type] = []
                    context['claim_data'][entity_type].extend(values)
        
        # Análisis cruzado de documentos
        print("  - Detectando inconsistencias...")
        inconsistencies = self._detect_inconsistencies(extracted_data)
        
        # Análisis de fraude con AI
        print("  - Evaluando indicadores de fraude...")
        fraud_indicators = await self._analyze_fraud_indicators(extracted_data, inconsistencies)
        
        # Compilar análisis completo
        analysis = {
            "context": context,
            "inconsistencies": inconsistencies,
            "fraud_indicators": fraud_indicators,
            "fraud_score": self._calculate_fraud_score(fraud_indicators, inconsistencies),
            "risk_level": "bajo",  # Se calculará basado en el score
            "recommendations": [],
            "external_validations": []
        }
        
        # Determinar nivel de riesgo
        if analysis['fraud_score'] > 0.7:
            analysis['risk_level'] = "crítico"
            analysis['recommendations'].append("Rechazar reclamación por indicios claros de fraude")
        elif analysis['fraud_score'] > 0.5:
            analysis['risk_level'] = "alto"
            analysis['recommendations'].append("Investigación profunda requerida antes de proceder")
        elif analysis['fraud_score'] > 0.3:
            analysis['risk_level'] = "medio"
            analysis['recommendations'].append("Verificar documentación adicional")
        else:
            analysis['risk_level'] = "bajo"
            analysis['recommendations'].append("Proceder con validación estándar")
        
        print(f"  - Score de fraude: {analysis['fraud_score']:.2%}")
        print(f"  - Nivel de riesgo: {analysis['risk_level']}")
        
        return analysis
    
    def _detect_inconsistencies(self, documents: List[Dict]) -> List[Dict]:
        """Detecta inconsistencias entre documentos"""
        inconsistencies = []
        
        # Comparar placas vehiculares si existen
        all_placas = []
        for doc in documents:
            if 'entities' in doc and 'placa' in doc['entities']:
                placas = doc['entities']['placa']
                for placa in placas:
                    all_placas.append({
                        'value': placa,
                        'source': doc['file_name']
                    })
        
        # Si hay múltiples placas diferentes, es una inconsistencia
        unique_placas = set(p['value'] for p in all_placas)
        if len(unique_placas) > 1:
            inconsistencies.append({
                "field": "Placas vehiculares",
                "description": f"Se encontraron {len(unique_placas)} placas diferentes",
                "values": list(unique_placas),
                "severity": "critical",
                "affected_documents": list(set(p['source'] for p in all_placas))
            })
        
        # Comparar RFCs
        all_rfcs = []
        for doc in documents:
            if 'entities' in doc and 'rfc' in doc['entities']:
                rfcs = doc['entities']['rfc']
                for rfc in rfcs:
                    all_rfcs.append({
                        'value': rfc,
                        'source': doc['file_name']
                    })
        
        unique_rfcs = set(r['value'] for r in all_rfcs)
        if len(unique_rfcs) > 2:  # Puede haber RFC del asegurado y beneficiario
            inconsistencies.append({
                "field": "RFC",
                "description": f"Múltiples RFCs detectados: {len(unique_rfcs)}",
                "values": list(unique_rfcs),
                "severity": "medium",
                "affected_documents": list(set(r['source'] for r in all_rfcs))
            })
        
        return inconsistencies
    
    async def _analyze_fraud_indicators(self, documents: List[Dict], inconsistencies: List[Dict]) -> List[Dict]:
        """Analiza indicadores de fraude usando AI"""
        indicators = []
        
        # Indicador por inconsistencias críticas
        critical_inconsistencies = [i for i in inconsistencies if i['severity'] == 'critical']
        if critical_inconsistencies:
            indicators.append({
                "type": "inconsistency",
                "description": "Inconsistencias críticas en documentación",
                "severity": "high",
                "confidence": 0.9,
                "details": critical_inconsistencies
            })
        
        # Buscar patrones sospechosos en fechas
        for doc in documents:
            if doc.get('document_type') == 'denuncia':
                # Verificar si la denuncia fue muy tardía o muy rápida
                if 'entities' in doc and 'fecha' in doc['entities']:
                    # Análisis simplificado de fechas
                    indicators.append({
                        "type": "temporal_pattern",
                        "description": "Patrón temporal a verificar",
                        "severity": "medium",
                        "confidence": 0.5,
                        "details": {"source": doc['file_name']}
                    })
        
        # Verificar montos excesivos
        all_amounts = []
        for doc in documents:
            if 'entities' in doc and 'moneda' in doc['entities']:
                amounts = doc['entities']['moneda']
                all_amounts.extend(amounts)
        
        if all_amounts:
            # Análisis simplificado de montos
            max_amount = max([float(a.replace('$', '').replace(',', '')) for a in all_amounts if a])
            if max_amount > 1000000:  # Más de 1 millón
                indicators.append({
                    "type": "excessive_amount",
                    "description": f"Monto reclamado excesivo: ${max_amount:,.2f}",
                    "severity": "medium",
                    "confidence": 0.6,
                    "details": {"amount": max_amount}
                })
        
        return indicators
    
    def _calculate_fraud_score(self, indicators: List[Dict], inconsistencies: List[Dict]) -> float:
        """Calcula un score de fraude basado en los indicadores"""
        score = 0.0
        weights = {
            'critical': 0.3,
            'high': 0.2,
            'medium': 0.1,
            'low': 0.05
        }
        
        # Puntaje por inconsistencias
        for inconsistency in inconsistencies:
            severity = inconsistency.get('severity', 'low')
            if severity == 'critical':
                score += 0.3
            elif severity == 'high':
                score += 0.2
            elif severity == 'medium':
                score += 0.1
        
        # Puntaje por indicadores de fraude
        for indicator in indicators:
            severity = indicator.get('severity', 'low')
            confidence = indicator.get('confidence', 0.5)
            score += weights.get(severity, 0.05) * confidence
        
        # Normalizar entre 0 y 1
        return min(score, 1.0)
    
    async def _generate_report(self, extracted_data: List[Dict], ai_analysis: Dict, folder_name: str) -> str:
        """Genera el informe final en PDF"""
        
        # Extraer información para el informe
        informe = self.template_processor.extract_from_documents(extracted_data, ai_analysis)
        
        # Si no se pudo extraer número de siniestro, usar el nombre de la carpeta
        if informe.numero_siniestro == "SIN_NUMERO":
            # Intentar extraer del nombre de la carpeta
            parts = folder_name.split(' - ')
            if parts:
                informe.numero_siniestro = parts[0].strip()
        
        # Generar timestamp para el nombre del archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generar HTML
        html_path = self.output_dir / f"INF-{informe.numero_siniestro}_{timestamp}.html"
        html_content = self.template_processor.generate_report(informe, str(html_path))
        
        # Generar PDF
        pdf_path = self.output_dir / f"INF-{informe.numero_siniestro}_{timestamp}.pdf"
        
        print(f"  - Generando PDF...")
        try:
            from weasyprint import HTML, CSS
            
            # Configurar CSS para mejor renderizado en PDF
            css = CSS(string='''
                @page {
                    size: A4;
                    margin: 20mm;
                }
                body {
                    font-size: 10pt;
                }
                .section {
                    page-break-inside: avoid;
                }
                table {
                    font-size: 9pt;
                }
            ''')
            
            HTML(string=html_content).write_pdf(str(pdf_path), stylesheets=[css])
            print(f"  - PDF generado: {pdf_path.name}")
            
        except ImportError:
            print(f"{Fore.YELLOW}⚠️  WeasyPrint no está instalado. Solo se generó HTML{Style.RESET_ALL}")
            return str(html_path)
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Error generando PDF: {e}{Style.RESET_ALL}")
            print(f"  - HTML guardado en: {html_path}")
            return str(html_path)
        
        return str(pdf_path)


async def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Genera informes de análisis de siniestros a partir de documentos'
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Ruta a la carpeta con los documentos del siniestro'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/reports',
        help='Directorio de salida para los informes (default: data/reports)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Activa el modo debug con más información'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Crear runner
    runner = FraudScorerRunner(output_dir=args.output)
    
    try:
        # Procesar carpeta
        report_path = await runner.process_folder(args.folder)
        
        if report_path:
            print(f"\n{Fore.CYAN}🎉 Proceso completado exitosamente!{Style.RESET_ALL}")
            print(f"{Fore.GREEN}📄 Informe disponible en: {report_path}{Style.RESET_ALL}\n")
            
            # Abrir el informe automáticamente (opcional)
            if sys.platform == 'darwin':  # macOS
                os.system(f'open "{report_path}"')
            elif sys.platform == 'win32':  # Windows
                os.startfile(report_path)
            elif sys.platform.startswith('linux'):  # Linux
                os.system(f'xdg-open "{report_path}"')
        else:
            print(f"\n{Fore.RED}❌ No se pudo generar el informe{Style.RESET_ALL}\n")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}\n")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Ejecutar con asyncio
    asyncio.run(main())