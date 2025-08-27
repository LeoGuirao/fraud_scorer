"""
Sistema de cache para optimizar las llamadas a la API
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any

class AIExtractionCache:
    """Cache persistente para extracciones"""
    
    def __init__(self, cache_dir: Path = Path("data/cache/ai_extractions")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = 24  # Cache válido por 24 horas
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        # Verificar TTL
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age > timedelta(hours=self.ttl_hours):
            cache_file.unlink()  # Eliminar cache expirado
            return None
        
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    def set(self, key: str, value: Any):
        """Guarda valor en cache"""
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    
    def generate_key(self, *args) -> str:
        """Genera clave única para cache"""
        content = json.dumps(args, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()