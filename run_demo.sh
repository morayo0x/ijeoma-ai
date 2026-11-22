#!/usr/bin/env bash
# Set up and launch the Ijeoma AI demo on a Linux machine with an NVIDIA GPU.
#
#   bash run_demo.sh            # local network + public gradio.live link
#
# Creates .venv on first run, installs dependencies, asks for a Hugging Face
# token if HF_TOKEN is not set, and starts the Workbench, Chat and Compare pages.
set -euo pipefail
cd "$(dirname "$0")"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU memory (used / total):"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
else
    echo "warning: nvidia-smi not found; the model will run on CPU and be slow" >&2
fi

if [ ! -d .venv ]; then
    echo "Creating virtual environment and installing dependencies (first run only)..."
    python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -U "jax[cuda12]"
    .venv/bin/pip install -q -r app/requirements.txt
fi

if [ -z "${HF_TOKEN:-}" ]; then
    read -r -s -p "Hugging Face token (input hidden): " HF_TOKEN
    echo
    export HF_TOKEN
fi

exec .venv/bin/python app/ijeoma_app.py --share "$@"
