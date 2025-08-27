# utils/data_validator.py
"""
Validador avanzado de datos extraídos.

Provee utilidades para validar RFC, números de póliza, montos y fechas.
"""

from typing import Any, Optional, Tuple
import re


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
