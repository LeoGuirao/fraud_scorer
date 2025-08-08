#!/bin/bash
# Script de instalación para macOS

echo "🚀 Instalando Fraud Scorer - Sistema de Análisis de Siniestros"
echo "============================================================"

# Verificar macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
  echo "❌ Este script es solo para macOS"
  exit 1
fi

# Homebrew
if ! command -v brew &> /dev/null; then
  echo "📦 Instalando Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "📦 Instalando dependencias del sistema para WeasyPrint…"
brew install python@3.11 cairo pango gdk-pixbuf libffi poppler pkg-config

echo "📦 Instalando GLib y GObject-Introspection…"
brew install glib gobject-introspection

# — OMITIDO: wkhtmltopdf (ya no está disponible en Homebrew) —

# Crear/activar venv
if [ ! -d "venv" ]; then
  echo "🐍 Creando entorno virtual…"
  python3.11 -m venv venv
fi
source venv/bin/activate

echo "📦 Actualizando pip…"
pip install --upgrade pip setuptools wheel

echo "📦 Instalando dependencias de Python…"
pip install -r requirements.txt

echo "📦 Instalando Azure Form Recognizer…"
pip install azure-ai-formrecognizer azure-core

echo "📦 Instalando WeasyPrint…"
pip install WeasyPrint

echo ""
echo "✅ Verificando instalación…"
python -c "import azure.ai.formrecognizer; print('✓ Azure OCR instalado')"
python -c "import openai; print('✓ OpenAI instalado')"
python -c "import weasyprint; print('✓ WeasyPrint instalado')"
python -c "from jinja2 import Template; print('✓ Jinja2 instalado')"

echo "📁 Creando estructura de directorios…"
mkdir -p data/{raw,processed,reports,temp}
mkdir -p templates logs

# .env
if [ ! -f ".env" ]; then
  echo "⚠️  No se encontró .env; creando .env.example…"
  cat > .env.example << EOF
# (tu .env aquí)
EOF
  echo "📝 Rellena .env con tus credenciales"
else
  echo "✓ .env encontrado"
fi

echo ""
echo "✅ Instalación completada!"
echo ""
echo "📋 Próximos pasos:"
echo "1. Asegúrate de que el archivo .env tiene las credenciales correctas"
echo "2. Ejecuta: source venv/bin/activate"
echo "3. Para procesar documentos: python scripts/run_report.py /ruta/a/carpeta"
echo ""
echo "🎉 ¡Listo para usar!"