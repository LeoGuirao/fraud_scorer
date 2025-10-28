from __future__ import annotations

import os
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Optional

import requests


class ExchangeRateService:
    """Obtiene tipos de cambio históricos de forma resiliente."""

    DEFAULT_BASE = "USD"
    DEFAULT_TARGET = "MXN"
    TIMEOUT_SECONDS = 6
    API_URL_TEMPLATE = "https://api.frankfurter.app/{date}"

    @classmethod
    def get_rate(
        cls,
        reference_date: Optional[date],
        base: str = DEFAULT_BASE,
        target: str = DEFAULT_TARGET,
    ) -> Optional[Decimal]:
        override = cls._get_env_override(base, target)
        if override is not None:
            return override

        if not reference_date:
            return None

        return cls._fetch_rate(reference_date, base, target)

    @classmethod
    def _get_env_override(cls, base: str, target: str) -> Optional[Decimal]:
        env_key = f"FRAUD_EXCHANGE_RATE_{base.upper()}_{target.upper()}"
        value = os.getenv(env_key)
        if not value:
            return None
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None

    @classmethod
    @lru_cache(maxsize=64)
    def _fetch_rate(cls, reference_date: date, base: str, target: str) -> Optional[Decimal]:
        formatted_date = reference_date.strftime("%Y-%m-%d")
        url = cls.API_URL_TEMPLATE.format(date=formatted_date)
        params = {"from": base.upper(), "to": target.upper()}
        try:
            response = requests.get(url, params=params, timeout=cls.TIMEOUT_SECONDS)
            if response.status_code != 200:
                return None
            payload = response.json()
            rates = payload.get("rates") or {}
            rate_value = rates.get(target.upper())
            if rate_value is None:
                return None
            return Decimal(str(rate_value))
        except Exception:
            return None
