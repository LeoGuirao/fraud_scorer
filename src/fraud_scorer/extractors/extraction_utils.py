# src/fraud_scorer/extractors/extraction_utils.py

"""
Utilidades adicionales para el sistema de extracción
"""

from typing import Dict, List, Any, Optional, Tuple
import re
import json
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ExtractionMonitor:
    """Monitor para analizar y mejorar la extracción"""
    
    def __init__(self, log_dir: Path = Path("logs/extraction")):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'documents_processed': 0,
            'fields_extracted': 0,
            'failures': [],
            'success_by_strategy': {}
        }
    
    def log_extraction(self, document_id: str, field: str, result: Any):
        """Registra una extracción para análisis posterior"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'document_id': document_id,
            'field': field,
            'success': result.value is not None if hasattr(result, 'value') else False,
            'strategy': result.strategy if hasattr(result, 'strategy') else 'unknown',
            'confidence': result.confidence if hasattr(result, 'confidence') else 0.0
        }
        
        # Actualizar estadísticas de sesión
        self.session_data['documents_processed'] += 1
        if log_entry['success']:
            self.session_data['fields_extracted'] += 1
            strategy = log_entry['strategy']
            if strategy not in self.session_data['success_by_strategy']:
                self.session_data['success_by_strategy'][strategy] = 0
            self.session_data['success_by_strategy'][strategy] += 1
        else:
            self.session_data['failures'].append({
                'document': document_id,
                'field': field
            })
        
        # Guardar en archivo de log
        log_file = self.log_dir / f"extraction_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def generate_report(self) -> Dict[str, Any]:
        """Genera reporte de la sesión de extracción"""
        
        success_rate = 0
        if self.session_data['documents_processed'] > 0:
            success_rate = (self.session_data['fields_extracted'] / 
                          self.session_data['documents_processed'] * 100)
        
        return {
            'session_duration': str(
                datetime.now() - datetime.fromisoformat(self.session_data['start_time'])
            ),
            'total_documents': self.session_data['documents_processed'],
            'total_fields_extracted': self.session_data['fields_extracted'],
            'success_rate': f"{success_rate:.1f}%",
            'top_strategies': sorted(
                self.session_data['success_by_strategy'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'failure_summary': self._summarize_failures()
        }
    
    def _summarize_failures(self) -> Dict[str, int]:
        """Resume los campos que más fallan"""
        
        failure_counts = {}
        for failure in self.session_data['failures']:
            field = failure['field']
            failure_counts[field] = failure_counts.get(field, 0) + 1
        
        return dict(sorted(failure_counts.items(), key=lambda x: x[1], reverse=True))


class PatternOptimizer:
    """Optimizador de patrones regex basado en resultados"""
    
    def __init__(self):
        self.pattern_performance = {}
    
    def analyze_pattern_performance(self, log_file: Path) -> Dict[str, Any]:
        """Analiza el rendimiento de los patrones desde logs"""
        
        pattern_stats = {}
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('strategy') in ['regex_strict', 'regex_fuzzy']:
                        pattern = entry.get('metadata', {}).get('pattern')
                        if pattern:
                            if pattern not in pattern_stats:
                                pattern_stats[pattern] = {
                                    'attempts': 0,
                                    'successes': 0
                                }
                            pattern_stats[pattern]['attempts'] += 1
                            if entry.get('success'):
                                pattern_stats[pattern]['successes'] += 1
                except:
                    continue
        
        # Calcular tasas de éxito
        for pattern, stats in pattern_stats.items():
            if stats['attempts'] > 0:
                stats['success_rate'] = stats['successes'] / stats['attempts']
        
        # Ordenar por tasa de éxito
        sorted_patterns = sorted(
            pattern_stats.items(),
            key=lambda x: x[1].get('success_rate', 0),
            reverse=True
        )
        
        return {
            'best_patterns': sorted_patterns[:10],
            'worst_patterns': sorted_patterns[-10:] if len(sorted_patterns) > 10 else [],
            'total_patterns_analyzed': len(pattern_stats)
        }
    
    def suggest_pattern_order(self, field_name: str, current_patterns: List[str]) -> List[str]:
        """Sugiere un nuevo orden de patrones basado en el rendimiento"""
        
        # Obtener estadísticas de cada patrón
        pattern_scores = []
        for pattern in current_patterns:
            score = self.pattern_performance.get(pattern, {}).get('success_rate', 0.5)
            pattern_scores.append((pattern, score))
        
        # Ordenar por score descendente
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [pattern for pattern, _ in pattern_scores]


class DataValidator:
    """Validador avanzado de datos extraídos"""
    
    @staticmethod
    def validate_rfc(rfc: str) -> Tuple[bool, Optional[str]]:
        """Valida RFC mexicano con explicación de error"""
        
        if not rfc:
            return False, "RFC vacío"
        
        rfc = rfc.upper().strip()
        
        # Validar formato básico
        pattern = r'^[A-ZÑ&]{3,4}\d{6}[A-Z\d]{3}$'
        if not re.match(pattern, rfc):
            return False, f"Formato inválido: {rfc}"
        
        # Validar que la fecha sea válida
        if len(rfc) >= 10:
            year = rfc[4:6]
            month = rfc[6:8]
            day = rfc[8:10]
            
            try:
                year_int = int(year)
                month_int = int(month)
                day_int = int(day)
                
                # Año entre 00-99 (1900-1999 o 2000-2099)
                if year_int < 0 or year_int > 99:
                    return False, f"Año inválido: {year}"
                
                # Mes entre 01-12
                if month_int < 1 or month_int > 12:
                    return False, f"Mes inválido: {month}"
                
                # Día entre 01-31
                if day_int < 1 or day_int > 31:
                    return False, f"Día inválido: {day}"
                    
            except ValueError:
                return False, "Fecha no numérica en RFC"
        
        return True, None
    
    @staticmethod
    def validate_policy_number(policy: str) -> Tuple[bool, Optional[str]]:
        """Valida número de póliza"""
        
        if not policy:
            return False, "Número de póliza vacío"
        
        policy = policy.strip()
        
        # Debe contener al menos un número
        if not re.search(r'\d', policy):
            return False, "No contiene números"
        
        # No debe ser muy corto ni muy largo
        if len(policy) < 3:
            return False, "Muy corto (< 3 caracteres)"
        
        if len(policy) > 50:
            return False, "Muy largo (> 50 caracteres)"
        
        # No debe contener caracteres extraños
        if re.search(r'[^\w\s\-/]', policy):
            return False, "Contiene caracteres no válidos"
        
        return True, None
    
    @staticmethod
    def validate_amount(amount: Any) -> Tuple[bool, Optional[str]]:
        """Valida montos monetarios"""
        
        if amount is None:
            return False, "Monto nulo"
        
        # Convertir a float si es string
        if isinstance(amount, str):
            try:
                # Limpiar string
                clean = amount.replace(",", "").replace("$", "")
                clean = clean.replace("MXN", "").replace("MN", "").strip()
                amount = float(clean)
            except:
                return False, f"No se pudo convertir a número: {amount}"
        
        # Validar rango
        if amount <= 0:
            return False, "Monto debe ser positivo"
        
        if amount > 1000000000:  # Mil millones
            return False, "Monto excesivamente alto"
        
        return True, None
    
    @staticmethod
    def validate_date(date_str: str, min_year: int = 2000, max_year: int = 2030) -> Tuple[bool, Optional[str]]:
        """Valida fechas con rangos razonables"""
        
        if not date_str:
            return False, "Fecha vacía"
        
        # Intentar parsear la fecha
        from datetime import datetime
        
        formats = [
            "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y",
            "%Y/%m/%d", "%m/%d/%Y"
        ]
        
        parsed_date = None
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                break
            except:
                continue
        
        if not parsed_date:
            return False, f"Formato de fecha no reconocido: {date_str}"
        
        # Validar rango de años
        if parsed_date.year < min_year:
            return False, f"Año muy antiguo: {parsed_date.year}"
        
        if parsed_date.year > max_year:
            return False, f"Año muy futuro: {parsed_date.year}"
        
        return True, None


class FieldMerger:
    """Fusiona campos de múltiples fuentes con resolución de conflictos"""
    
    @staticmethod
    def merge_fields(
        sources: List[Dict[str, Any]],
        strategy: str = "highest_confidence"
    ) -> Dict[str, Any]:
        """
        Fusiona campos de múltiples fuentes
        
        Args:
            sources: Lista de diccionarios con campos extraídos
            strategy: Estrategia de fusión (highest_confidence, most_recent, voting)
        
        Returns:
            Diccionario con campos fusionados
        """
        
        if not sources:
            return {}
        
        if strategy == "highest_confidence":
            return FieldMerger._merge_by_confidence(sources)
        elif strategy == "most_recent":
            return FieldMerger._merge_by_recency(sources)
        elif strategy == "voting":
            return FieldMerger._merge_by_voting(sources)
        else:
            # Por defecto, usar confianza
            return FieldMerger._merge_by_confidence(sources)
    
    @staticmethod
    def _merge_by_confidence(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fusiona tomando el valor con mayor confianza"""
        
        merged = {}
        confidence_map = {}
        
        for source in sources:
            for field, value in source.items():
                if field == 'confidence_scores':
                    continue
                
                # Obtener confianza del campo
                confidence = source.get('confidence_scores', {}).get(field, 0.5)
                
                # Si no existe o tiene menor confianza, reemplazar
                if field not in merged or confidence > confidence_map.get(field, 0):
                    merged[field] = value
                    confidence_map[field] = confidence
        
        return merged
    
    @staticmethod
    def _merge_by_recency(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fusiona tomando el valor más reciente"""
        
        # El último source es el más reciente
        merged = {}
        for source in sources:
            merged.update(source)
        
        return merged
    
    @staticmethod
    def _merge_by_voting(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fusiona por votación (valor más común)"""
        
        from collections import Counter
        
        field_values = {}
        
        # Recolectar todos los valores para cada campo
        for source in sources:
            for field, value in source.items():
                if field not in field_values:
                    field_values[field] = []
                field_values[field].append(str(value))
        
        # Seleccionar el valor más común para cada campo
        merged = {}
        for field, values in field_values.items():
            if values:
                most_common = Counter(values).most_common(1)[0][0]
                merged[field] = most_common
        
        return merged