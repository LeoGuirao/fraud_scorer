# Fraud Scorer - Sistema de Análisis de Siniestros

Sistema inteligente para análisis y detección de fraude en siniestros de seguros.

## 🚀 Quick Start

# Activar entorno
source venv/bin/activate

# Levantar servicios
docker-compose up -d

# Iniciar API
cd api && python main.py

📁 **Estructura del Proyecto**

fraud_scorer/
├── api/                 # FastAPI backend
├── processors/          # OCR, LLM, Fraude
├── templates/           # Plantillas de informes
├── ui/                  # Interfaz Streamlit
├── tests/               # Tests
├── docs/                # Documentación
├── scripts/             # Scripts útiles
├── data/                # Datos (raw/processed)
└── infra/               # Infraestructura Docker


🔧 **Servicios**
- API: http://localhost:8000  
- n8n: http://localhost:5678 (admin/fraudscorer123)  
- MinIO: http://localhost:9001  
- PostgreSQL: localhost:5432  
- Redis: localhost:6379  