#!/usr/bin/env bash
# Set up and launch the Ijeoma AI demo on a Linux machine with an NVIDIA GPU.
#
#   bash run_demo.sh            # local network + public gradio.live link
#   bash run_demo.sh --dtype float32   # full precision, as evaluated (needs ~4 GB)
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

if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment and installing dependencies (first run only)..."
    rm -rf .venv
    if ! python3 -m venv .venv; then
        echo "Install the venv module first:  sudo apt install -y python3-venv  (or python3.X-venv), then re-run." >&2
        rm -rf .venv
        exit 1
    fi
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -U "jax[cuda12]"
    .venv/bin/pip install -q -r app/requirements.txt
fi

if [ -z "${HF_TOKEN:-}" ]; then
    read -r -s -p "Hugging Face token (input hidden): " HF_TOKEN
    echo
    export HF_TOKEN
fi

# Half precision halves GPU memory (about 2.5 GB instead of 4 GB per model).
exec .venv/bin/python app/ijeoma_app.py --share --dtype bfloat16 "$@"
