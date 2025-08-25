"""
Gestor de Cache para resultados OCR
Evita llamadas repetidas a Azure OCR
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class OCRCacheManager:
    """
    Gestiona el cache de resultados OCR para evitar re-procesar documentos
    """
    
    def __init__(self, cache_dir: Path = None):
        """
        Inicializa el gestor de cache
        
        Args:
            cache_dir: Directorio donde guardar el cache
        """
        if cache_dir is None:
            cache_dir = Path("data/ocr_cache")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Directorio para índices de casos
        self.index_dir = self.cache_dir / "case_index"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"OCR Cache inicializado en: {self.cache_dir}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """
        Genera un hash único para un archivo basado en su contenido
        """
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Leer en chunks para archivos grandes
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        
        # Incluir metadata del archivo
        file_stats = file_path.stat()
        metadata = f"{file_path.name}_{file_stats.st_size}_{file_stats.st_mtime}"
        hasher.update(metadata.encode())
        
        return hasher.hexdigest()
    
    def _get_cache_path(self, file_hash: str) -> Path:
        """
        Obtiene la ruta del archivo de cache para un hash dado
        """
        # Usar los primeros 2 caracteres del hash para crear subdirectorios
        # Esto evita tener miles de archivos en un solo directorio
        subdir = self.cache_dir / file_hash[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{file_hash}.json"
    
    def has_cache(self, file_path: Path) -> bool:
        """
        Verifica si existe cache para un archivo
        """
        file_hash = self._get_file_hash(file_path)
        cache_path = self._get_cache_path(file_hash)
        return cache_path.exists()
    
    def get_cache(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Obtiene el resultado OCR cacheado para un archivo
        
        Returns:
            Dict con el resultado OCR o None si no existe
        """
        file_hash = self._get_file_hash(file_path)
        cache_path = self._get_cache_path(file_hash)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            logger.info(f"✓ Cache OCR encontrado para: {file_path.name}")
            return cache_data['ocr_result']
            
        except Exception as e:
            logger.error(f"Error leyendo cache para {file_path.name}: {e}")
            return None
    
    def save_cache(self, file_path: Path, ocr_result: Dict[str, Any]) -> bool:
        """
        Guarda el resultado OCR en cache
        
        Returns:
            True si se guardó exitosamente
        """
        try:
            file_hash = self._get_file_hash(file_path)
            cache_path = self._get_cache_path(file_hash)
            
            cache_data = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_hash': file_hash,
                'cached_at': datetime.now().isoformat(),
                'ocr_result': ocr_result
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ Cache OCR guardado para: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando cache para {file_path.name}: {e}")
            return False
    
    def save_case_index(self, case_id: str, case_data: Dict[str, Any]) -> bool:
        """
        Guarda un índice del caso para facilitar el replay
        """
        try:
            index_path = self.index_dir / f"{case_id}.json"
            
            index_data = {
                'case_id': case_id,
                'case_title': case_data.get('case_title', 'Sin título'),
                'processed_at': datetime.now().isoformat(),
                'documents': case_data.get('documents', []),
                'total_documents': len(case_data.get('documents', [])),
                'folder_path': case_data.get('folder_path', ''),
                'cache_files': case_data.get('cache_files', [])
            }
            
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ Índice de caso guardado: {case_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando índice del caso {case_id}: {e}")
            return False
    
    def get_case_index(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el índice de un caso
        """
        index_path = self.index_dir / f"{case_id}.json"
        
        if not index_path.exists():
            return None
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error leyendo índice del caso {case_id}: {e}")
            return None
    
    def list_cached_cases(self) -> List[Dict[str, Any]]:
        """
        Lista todos los casos que tienen cache disponible
        """
        cases = []
        
        for index_file in self.index_dir.glob("*.json"):
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                    cases.append({
                        'case_id': case_data['case_id'],
                        'case_title': case_data['case_title'],
                        'processed_at': case_data['processed_at'],
                        'total_documents': case_data['total_documents']
                    })
            except Exception as e:
                logger.warning(f"Error leyendo índice {index_file}: {e}")
                continue
        
        # Ordenar por fecha de procesamiento (más reciente primero)
        cases.sort(key=lambda x: x['processed_at'], reverse=True)
        return cases
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del cache
        """
        total_files = sum(1 for _ in self.cache_dir.rglob("*.json"))
        total_cases = len(list(self.index_dir.glob("*.json")))
        
        # Calcular tamaño total
        total_size = 0
        for file in self.cache_dir.rglob("*.json"):
            total_size += file.stat().st_size
        
        return {
            'total_cached_files': total_files,
            'total_cases': total_cases,
            'cache_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_directory': str(self.cache_dir)
        }