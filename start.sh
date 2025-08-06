#!/bin/bash
echo "🚀 Iniciando Fraud Scorer..."
source venv/bin/activate
if ! docker info > /dev/null 2>&1; then
  echo "❌ Inicia Docker Desktop"; exit 1
fi
docker-compose up -d
sleep 10; docker-compose ps
echo "✅ Servicios listos!"
echo "- API:   http://localhost:8000"
echo "- Docs:  http://localhost:8000/docs"
echo "- n8n:   http://localhost:5678 (admin/fraudscorer123)"
echo "- MinIO: http://localhost:9001 (leo_fraude/best.fraud482)"
echo "Para correr la API: cd api && python main.py"
