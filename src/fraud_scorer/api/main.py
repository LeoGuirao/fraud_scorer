"""
Aplicación principal de FastAPI para el Fraud Scorer v2.
Este archivo se encarga de crear la aplicación, configurar middleware
e incluir todos los routers de los endpoints.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# 1. Importar los routers desde la carpeta de endpoints
from .endpoints import general, documents, reports

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Crear la instancia de la aplicación
app = FastAPI(
    title="Sistema de Análisis de Siniestros v2.0",
    description="API para análisis automático de documentos de siniestros con OCR y detección de fraude mediante IA.",
    version="2.0.0"  # Actualizamos la versión
)

# 3. Añadir el middleware (esto ya lo tenías bien)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501", "http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Incluir los routers en la aplicación principal (¡Este es el paso clave!)
logger.info("Incluyendo routers de la API...")
app.include_router(general.router)
app.include_router(documents.router)
app.include_router(reports.router)
logger.info("Routers incluidos exitosamente.")


# 5. (Opcional pero recomendado) Bloque para ejecutar en desarrollo
if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando servidor Uvicorn para desarrollo...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)