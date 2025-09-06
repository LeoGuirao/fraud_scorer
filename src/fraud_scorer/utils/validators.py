# utils/validators.py
"""
Validador avanzado de datos extraídos.

Provee utilidades para validar RFC, números de póliza, montos y fechas.
Incluye el nuevo FieldValidator para el Sistema de Extracción Guiada.
"""

from typing import Any, Optional, Tuple, Dict, List
import re
from fraud_scorer.settings import ExtractionConfig


class DataValidator:
    """Validador avanzado de datos extraídos."""

    @staticmethod
    def validate_rfc(rfc: str) -> Tuple[bool, Optional[str]]:
        """Valida RFC mexicano y devuelve (es_válido, error).

        Reglas básicas:
        - Formato: 3 o 4 letras (A–Z, Ñ, &) + 6 dígitos de fecha + 3 alfanum.
        - La fecha (YYMMDD) debe ser plausible.
        """
        if not rfc:
            return False, "RFC vacío"

        rfc = rfc.upper().strip()

        # Formato básico
        pattern = r"^[A-ZÑ&]{3,4}\d{6}[A-Z\d]{3}$"
        if not re.match(pattern, rfc):
            return False, f"Formato inválido: {rfc}"

        # Validar fecha (YYMMDD)
        if len(rfc) >= 10:
            year = rfc[4:6]
            month = rfc[6:8]
            day = rfc[8:10]

            try:
                year_int = int(year)
                month_int = int(month)
                day_int = int(day)

                # Año 00-99 (se permite rango completo por compatibilidad)
                if year_int < 0 or year_int > 99:
                    return False, f"Año inválido: {year}"

                # Mes 01-12
                if month_int < 1 or month_int > 12:
                    return False, f"Mes inválido: {month}"

                # Día 01-31 (validación simple)
                if day_int < 1 or day_int > 31:
                    return False, f"Día inválido: {day}"

            except ValueError:
                return False, "Fecha no numérica en RFC"

        return True, None

    @staticmethod
    def validate_policy_number(policy: str) -> Tuple[bool, Optional[str]]:
        """Valida número de póliza y devuelve (es_válido, error)."""
        if not policy:
            return False, "Número de póliza vacío"

        policy = policy.strip()

        # Debe contener al menos un dígito
        if not re.search(r"\d", policy):
            return False, "No contiene números"

        # Longitud razonable
        if len(policy) < 3:
            return False, "Muy corto (< 3 caracteres)"
        if len(policy) > 50:
            return False, "Muy largo (> 50 caracteres)"

        # Sin caracteres extraños
        if re.search(r"[^\w\s\-/]", policy):
            return False, "Contiene caracteres no válidos"

        return True, None

    @staticmethod
    def validate_amount(amount: Any) -> Tuple[bool, Optional[str]]:
        """Valida montos monetarios y devuelve (es_válido, error)."""
        if amount is None:
            return False, "Monto nulo"

        # Convertir a float si es string
        if isinstance(amount, str):
            try:
                clean = amount.replace(",", "").replace("$", "")
                clean = clean.replace("MXN", "").replace("MN", "").strip()
                amount = float(clean)
            except Exception:
                return False, f"No se pudo convertir a número: {amount}"

        # Validar rango
        try:
            amount_val = float(amount)
        except Exception:
            return False, "Monto no numérico"

        if amount_val <= 0:
            return False, "Monto debe ser positivo"

        if amount_val > 1_000_000_000:  # Mil millones
            return False, "Monto excesivamente alto"

        return True, None

    @staticmethod
    def validate_date(
        date_str: str,
        min_year: int = 2000,
        max_year: int = 2030,
    ) -> Tuple[bool, Optional[str]]:
        """Valida fechas con formatos comunes y rangos razonables.

        Formatos soportados:
        - YYYY-MM-DD
        - DD/MM/YYYY
        - DD-MM-YYYY
        - YYYY/MM/DD
        - MM/DD/YYYY
        """
        if not date_str:
            return False, "Fecha vacía"

        from datetime import datetime

        formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%Y/%m/%d",
            "%m/%d/%Y",
        ]

        parsed_date = None
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                break
            except Exception:
                continue

        if not parsed_date:
            return False, f"Formato de fecha no reconocido: {date_str}"

        # Validar rango de años
        if parsed_date.year < min_year:
            return False, f"Año muy antiguo: {parsed_date.year}"
        if parsed_date.year > max_year:
            return False, f"Año muy futuro: {parsed_date.year}"

        return True, None


class FieldValidator:
    """
    Validador de campos para el Sistema de Extracción Guiada.
    Valida campos según las reglas definidas en la configuración.
    """
    
    def __init__(self):
        self.config = ExtractionConfig()
        self.validation_rules = self.config.FIELD_VALIDATION_RULES
        self.field_mapping = self.config.DOCUMENT_FIELD_MAPPING
        self.siniestro_types = self.config.SINIESTRO_TYPES
    
    def validate_field(self, field_name: str, value: Any) -> Tuple[bool, Any, Optional[str]]:
        """
        Valida un campo específico según sus reglas.
        
        Returns:
            Tuple de (es_válido, valor_transformado, mensaje_error)
        """
        if value is None:
            return True, None, None
        
        # Obtener reglas para este campo
        rules = self.validation_rules.get(field_name, {})
        
        # Aplicar transformación si existe
        if 'transform' in rules and callable(rules['transform']):
            try:
                value = rules['transform'](value)
            except Exception as e:
                return False, value, f"Error en transformación: {e}"
        
        # Validar por tipo
        field_type = rules.get('type')
        
        if field_type == 'date':
            is_valid, error = self._validate_date(value, rules)
            return is_valid, value, error
        
        elif field_type == 'float':
            is_valid, error = self._validate_float(value, rules)
            return is_valid, value, error
        
        # Validar por regex (después de transformación)
        if 'regex' in rules:
            pattern = rules['regex']
            # Si ya se transformó, usar el valor transformado para validación
            test_value = value if 'transform' in rules else str(value)
            if not re.match(pattern, str(test_value)):
                return False, value, f"No cumple formato: {rules.get('format', pattern)}"
        
        # Validar longitud
        if 'min_length' in rules:
            if len(str(value)) < rules['min_length']:
                return False, value, f"Muy corto (mín: {rules['min_length']})"
        
        if 'max_length' in rules:
            if len(str(value)) > rules['max_length']:
                return False, value, f"Muy largo (máx: {rules['max_length']})"
        
        # Validar máximo de palabras
        if 'max_words' in rules:
            words = str(value).split()
            if len(words) > rules['max_words']:
                # Truncar a máximo permitido
                value = ' '.join(words[:rules['max_words']])
        
        return True, value, None
    
    def _validate_date(self, value: Any, rules: Dict) -> Tuple[bool, Optional[str]]:
        """Valida campos de fecha"""
        validator = DataValidator()
        return validator.validate_date(str(value))
    
    def _validate_float(self, value: Any, rules: Dict) -> Tuple[bool, Optional[str]]:
        """Valida campos numéricos"""
        try:
            num_value = float(value)
            
            if 'min' in rules and num_value < rules['min']:
                return False, f"Valor menor al mínimo ({rules['min']})"
            
            if 'max' in rules and num_value > rules['max']:
                return False, f"Valor mayor al máximo ({rules['max']})"
            
            return True, None
        except (ValueError, TypeError):
            return False, "No es un número válido"
    
    def validate_document_fields(
        self, 
        document_type: str, 
        fields: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Valida todos los campos de un documento según su tipo.
        
        Returns:
            Tuple de (campos_validados, lista_de_errores)
        """
        validated_fields = {}
        errors = []
        
        # Obtener campos permitidos para este tipo de documento
        allowed_fields = self.field_mapping.get(document_type, [])
        
        for field_name, value in fields.items():
            # Si el campo no está permitido para este documento
            if allowed_fields and field_name not in allowed_fields:
                validated_fields[field_name] = None
                continue
            
            # Validar el campo
            is_valid, transformed_value, error = self.validate_field(field_name, value)
            
            if is_valid:
                validated_fields[field_name] = transformed_value
            else:
                validated_fields[field_name] = value  # Mantener valor original
                errors.append(f"{field_name}: {error}")
        
        return validated_fields, errors
    
    def validate_tipo_siniestro(self, value: str) -> Optional[str]:
        """
        Valida y normaliza el tipo de siniestro al catálogo oficial.
        """
        if not value:
            return None
        
        value_lower = value.lower()
        
        # Buscar coincidencia en el catálogo
        for category, types in self.siniestro_types.items():
            for siniestro_type in types:
                if siniestro_type.lower() in value_lower or value_lower in siniestro_type.lower():
                    return siniestro_type
        
        # Si no hay coincidencia exacta, intentar mapeo por palabras clave
        mappings = {
            "colision": "Colisión / Choque",
            "choque": "Colisión / Choque",
            "robo": "Robo Total / Parcial",
            "incendio": "Incendio",
            "inundacion": "Inundación",
            "transito": "Riesgos Ordinarios de Tránsito (ROT)"
        }
        
        for keyword, mapped_type in mappings.items():
            if keyword in value_lower:
                return mapped_type
        
        # Si no se puede mapear, retornar valor original
        return value
