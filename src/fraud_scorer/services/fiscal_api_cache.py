"""
Cache con fallback (Redis -> memoria) para respuestas de FiscalAPI.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

try:  # pragma: no cover - dependencia opcional
    import redis  # type: ignore
except Exception:  # pragma: no cover - fallback si redis no está disponible
    redis = None  # type: ignore

try:  # pragma: no cover - métricas opcionales
    from prometheus_client import Counter

    CACHE_COUNTER = Counter(
        "fiscal_validation_cache_events_total",
        "Eventos del cache FiscalAPI",
        labelnames=("event", "backend"),
    )
except Exception:  # pragma: no cover - fallback silencioso

    class _Noop:  # pylint: disable=too-few-public-methods
        def labels(self, *args: Any, **kwargs: Any) -> "_Noop":
            return self

        def inc(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    CACHE_COUNTER = _Noop()

from fraud_scorer.models.fiscal_validation import FiscalValidationResult

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:  # pragma: no cover - utilidad simple
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Tipo no serializable: {type(value)!r}")


class _InMemoryTTLCache:
    """Cache en memoria con TTL y límite simple."""

    def __init__(self, ttl_seconds: int, max_entries: int = 2048) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._store: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            expires_at, payload = entry
            if expires_at <= now:
                self._store.pop(key, None)
                return None
            return payload

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            if len(self._store) >= self.max_entries:
                self._evict_one()
            self._store[key] = (time.time() + self.ttl, payload)

    def _evict_one(self) -> None:
        if not self._store:
            return
        oldest_key = next(iter(self._store.keys()))
        self._store.pop(oldest_key, None)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)


class FiscalValidationCache:
    """Cachea respuestas de FiscalAPI para reducir latencia y costos."""

    def __init__(
        self,
        namespace: str,
        ttl_seconds: int,
        max_local_entries: int = 2048,
        redis_client: Optional[Any] = None,
    ) -> None:
        self.namespace = namespace.strip(":")
        self.ttl_seconds = ttl_seconds
        self._redis = redis_client or self._build_redis_client()
        self._memory = _InMemoryTTLCache(ttl_seconds=ttl_seconds, max_entries=max_local_entries)

    def _build_redis_client(self) -> Optional[Any]:
        if redis is None:
            return None

        if os.getenv("FISCAL_CACHE_USE_REDIS", "true").lower() in {"0", "false", "off", "no"}:
            logger.debug("FiscalValidationCache: Redis deshabilitado por configuración")
            return None

        redis_url = os.getenv("REDIS_URL")
        redis_host_env = os.getenv("REDIS_HOST")

        if not redis_url and not redis_host_env:
            logger.debug("FiscalValidationCache: Redis no configurado; usando cache en memoria")
            return None

        redis_port_env = os.getenv("REDIS_PORT")
        redis_password = os.getenv("REDIS_PASSWORD")

        host = redis_host_env or "localhost"
        port = int(redis_port_env or "6379")
        password = redis_password or None
        try:
            if redis_url:
                client = redis.from_url(redis_url, socket_timeout=1.5, socket_connect_timeout=1.5)
            else:
                client = redis.Redis(
                    host=host,
                    port=port,
                    password=password,
                    socket_timeout=1.5,
                    socket_connect_timeout=1.5,
                )
            # Ping suave para validar conectividad
            client.ping()
            logger.info("FiscalValidationCache: conectado a Redis en %s", redis_url or f"{host}:{port}")
            return client
        except Exception as exc:
            logger.warning("FiscalValidationCache: no se pudo conectar a Redis (%s). Fallback a memoria.", exc)
            return None

    def _namespaced(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Optional[FiscalValidationResult]:
        namespaced = self._namespaced(key)
        if self._redis:
            try:
                raw = self._redis.get(namespaced)
                if raw:
                    CACHE_COUNTER.labels(event="hit", backend="redis").inc()
                    data = json.loads(raw)
                    return FiscalValidationResult.model_validate(data)
            except Exception as exc:  # pragma: no cover - errores raros
                logger.debug("FiscalValidationCache: error leyendo Redis (%s)", exc)

        payload = self._memory.get(namespaced)
        if payload:
            CACHE_COUNTER.labels(event="hit", backend="memory").inc()
            return FiscalValidationResult.model_validate(payload)

        CACHE_COUNTER.labels(event="miss", backend="memory").inc()
        return None

    def set(self, key: str, result: FiscalValidationResult) -> None:
        data = result.model_dump()
        namespaced = self._namespaced(key)

        self._memory.set(namespaced, data)
        CACHE_COUNTER.labels(event="store", backend="memory").inc()

        if not self._redis:
            return

        try:
            json_payload = json.dumps(data, default=_json_default)
            self._redis.setex(namespaced, self.ttl_seconds, json_payload)
            CACHE_COUNTER.labels(event="store", backend="redis").inc()
        except Exception as exc:  # pragma: no cover - errores raros
            logger.debug("FiscalValidationCache: error guardando en Redis (%s)", exc)

    def invalidate(self, key: str) -> None:
        namespaced = self._namespaced(key)
        self._memory.delete(namespaced)
        if self._redis:
            try:
                self._redis.delete(namespaced)
            except Exception:  # pragma: no cover - no crítico
                pass
