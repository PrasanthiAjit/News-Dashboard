#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$ROOT_DIR/app.py"
VENV="$ROOT_DIR/../.venv"

if [ -x "$VENV/bin/streamlit" ]; then
  STREAMLIT_CMD="$VENV/bin/streamlit"
elif command -v streamlit >/dev/null 2>&1; then
  STREAMLIT_CMD="$(command -v streamlit)"
elif [ -x "$VENV/bin/python" ]; then
  STREAMLIT_CMD="$VENV/bin/python -m streamlit"
else
  STREAMLIT_CMD="python -m streamlit"
fi

echo "Starting Streamlit with: $STREAMLIT_CMD run $APP $*"
eval "$STREAMLIT_CMD run \"$APP\" $*"
