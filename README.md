# 🔍 Fraud Scorer v2.0  
**Sistema Inteligente de Análisis de Siniestros con Detección de Fraude**  

Un sistema empresarial que combina OCR avanzado con IA (GPT-4) para analizar documentos de siniestros, extraer información clave y detectar posibles indicadores de fraude de manera automatizada.  

---

## 🎯 Características Principales
- **OCR Inteligente**: Procesamiento de documentos con Azure Document Intelligence  
- **Extracción con IA**: Uso de GPT-4 para extracción semántica de campos  
- **Detección de Fraude**: Análisis automático de inconsistencias y patrones sospechosos  
- **Sistema de Cache**: Reutilización inteligente de resultados OCR  
- **Modo Replay**: Reprocesamiento de casos sin re-escanear documentos  
- **Generación de Reportes**: Informes HTML/PDF profesionales con análisis detallado  

---

## 📋 Requisitos
- Python 3.10+  
- Docker y Docker Compose  
- Credenciales de Azure Document Intelligence  
- API Key de OpenAI (GPT-4)  

---

## 🚀 Instalación Rápida

### 1. Clonar el repositorio
```bash
git clone https://github.com/tuusuario/fraud-scorer.git
cd fraud-scorer
```

### 2. Configurar entorno (macOS/Linux)
```bash
# Dependencias del sistema y entorno virtual
./scripts/setup_macos.sh   # Para macOS

# O manualmente:
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar credenciales
Crear archivo **.env** en la raíz del proyecto:
```env
# Azure Document Intelligence
AZURE_ENDPOINT=https://tu-recurso.cognitiveservices.azure.com/
AZURE_OCR_KEY=tu-clave-azure

# OpenAI
OPENAI_API_KEY=sk-tu-api-key
OPENAI_MODEL=gpt-4o-mini

# Base de datos
POSTGRES_USER=fraud_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=fraud_scorer

# Redis
REDIS_PASSWORD=redis_password

# MinIO (S3 local)
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
```

### 4. Iniciar servicios
```bash
# Levantar infraestructura (PostgreSQL, Redis, MinIO)
docker-compose up -d

# Verificar instalación
python scripts/test_system.py
```

---

## 📖 Uso

### Procesamiento de Casos

**Opción 1: Línea de comandos**
```bash
# Procesar documentos
python scripts/run_report.py /ruta/a/documentos --out data/reports

# Con título personalizado
python scripts/run_report.py /ruta/a/documentos --title "Siniestro ABC-123"
```

**Opción 2: Interfaz Web**
```bash
python scripts/start_web_server.py
```
Acceder en: [http://localhost:8000](http://localhost:8000)

---

### Sistema de Replay
```bash
# Modo interactivo
python scripts/replay_case.py

# Listar casos
python scripts/replay_case.py --list

# Replay directo
python scripts/replay_case.py --case-id CASE-2025-0001
```

---

## 🏗️ Arquitectura
```
fraud_scorer/
├── src/fraud_scorer/
│   ├── api/              # FastAPI endpoints
│   ├── parsers/          # Procesamiento de documentos
│   ├── processors/       
│   │   ├── ocr/          # Azure OCR
│   │   └── ai/           # GPT-4 extractors
│   ├── storage/          # Cache y base de datos
│   ├── templates/        # Generación de reportes
│   └── models/           # Modelos de datos
├── scripts/              # Scripts de ejecución
├── data/                 # Datos y reportes
└── docker-compose.yml    # Servicios
```

### Flujo de Procesamiento
1. 📄 **Ingesta** → Carga de documentos  
2. 🔍 **OCR** → Extracción de texto  
3. 🧠 **Extracción IA** → GPT-4 identifica campos clave  
4. 🔄 **Consolidación** → Resolución de conflictos  
5. ⚠️ **Análisis** → Detección de inconsistencias y fraude  
6. 📊 **Reporte** → Informe HTML/PDF  

---

## 🔧 Configuración Avanzada
### Modelos IA
- `gpt-4o`: Precisión máxima  
- `gpt-4o-mini`: Más rápido y económico  
- `gpt-3.5-turbo`: Legacy (no recomendado)  

### Clasificación con visión (opcional)
- Pipeline (`run_report.py`) puede usar visión para PDF/imagenes si `CLASSIFICATION_ENGINE.use_vision=true` en `src/fraud_scorer/settings.py`.
- Requisitos: instalar `PyMuPDF` (`pip install pymupdf`).  
- Consideraciones: mayor costo/latencia (se envían imágenes al LLM). Para mantener costos bajos, déjalo desactivado.

### Tipos de Documentos
- Pólizas, facturas, denuncias, cartas porte, GPS, peritajes, IDs  

---

## 📊 API REST
[http://localhost:8000/docs](http://localhost:8000/docs)

- `POST /documents/upload` → Subir y procesar  
- `GET  /reports/generate` → Generar reporte  
- `GET  /health` → Estado del sistema  

---

## 🧪 Testing
```bash
# Verificar instalación
python scripts/test_system.py

# Tests unitarios
pytest tests/

# Test de integración
python scripts/run_report.py data/test_cases/ejemplo --debug
```

---

## 🐛 Troubleshooting
**OPENAI_API_KEY no definido**  
```bash
export OPENAI_API_KEY=sk-...
```

**Azure OCR falla**  
- Verificar `.env`  
- Revisar endpoint y rate limits  

**WeasyPrint no genera PDFs**  
```bash
# macOS
brew install cairo pango gdk-pixbuf

# Linux
apt-get install python3-cffi python3-brotli libpango-1.0-0 libpangoft2-1.0-0
```

---

## 📈 Métricas
- Tiempo: **30–60s/caso** (5–10 docs)  
- Precisión: **>95%**  
- Cache OCR: **-80% tiempo reprocesamiento**  
- Tokens: ~2000/doc (GPT-4o-mini)  

---

## 🤝 Contribuir
```bash
git checkout -b feature/NuevaCaracteristica
git commit -m 'Agregar nueva característica'
git push origin feature/NuevaCaracteristica
```

---

## 📄 Licencia
Propietario – Todos los derechos reservados  

---

## 👥 Equipo
- **Desarrollador Principal:** Leo Guirao  
- 📧 [guiraoleo.2000@gmail.com](mailto:guiraoleo.2000@gmail.com)  
