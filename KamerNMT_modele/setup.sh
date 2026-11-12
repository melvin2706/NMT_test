#!/usr/bin/env bash
# KamerNMT - one-command setup for Linux / macOS.
# Creates .venv in this folder, installs PyTorch (CPU) and all libraries,
# and installs the `kamernmt` command.
# Usage:  ./setup.sh          (CPU, recommended)
#         ./setup.sh gpu      (PyTorch with NVIDIA CUDA, for faster training)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
command -v "$PYTHON" >/dev/null || { echo "Python 3.10+ not found (set PYTHON=/path/to/python)"; exit 1; }

[ -d .venv ] || { echo "Creating the virtual environment .venv ..."; "$PYTHON" -m venv .venv; }
PY=.venv/bin/python
"$PY" -m pip install --upgrade pip

if [ "${1:-}" = "gpu" ]; then
  echo "Installing PyTorch with CUDA ..."
  "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
else
  echo "Installing PyTorch for CPU ..."
  "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installing the other libraries ..."
"$PY" -m pip install -r requirements.txt
echo "Installing the kamernmt command ..."
"$PY" -m pip install -e .

cat <<'EOF'

Done. Activate the environment with:   source .venv/bin/activate
Then:
  python main.py --demo                     check everything works (a few minutes)
  python main.py                            train the model
  kamernmt                                  translate interactively
  kamernmt --tgt dua "Je vais au marché."   translate a phrase
EOF
