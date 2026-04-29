#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install --upgrade --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install --upgrade --no-cache-dir -r requirements-training.txt

python scripts/check_training_env.py

echo "RunPod training environment is ready."
