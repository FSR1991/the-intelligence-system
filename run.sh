#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_EXE="$PROJECT_ROOT/.venv/Scripts/python.exe"

if [[ ! -f "$PYTHON_EXE" ]]; then
    echo "ERROR: No se encontró el entorno virtual."
    echo "Se esperaba: .venv/Scripts/python.exe"
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    echo "ERROR: No se encontró el archivo .env."
    exit 1
fi

if ! command -v cygpath >/dev/null 2>&1; then
    echo "ERROR: Este script debe ejecutarse desde Git Bash."
    exit 1
fi

export PYTHONPATH="$(cygpath -w "$PROJECT_ROOT/src")"

"$PYTHON_EXE" -c "import uvicorn; import intelligence_system" || {
    echo "ERROR: No se pudieron importar las dependencias o el paquete."
    exit 1
}

echo "Iniciando The Intelligence System..."
echo "API: http://127.0.0.1:8000"
echo "Documentación: http://127.0.0.1:8000/docs"

exec "$PYTHON_EXE" -m uvicorn \
    intelligence_system.api.main:app \
    --host 127.0.0.1 \
    --port 8000