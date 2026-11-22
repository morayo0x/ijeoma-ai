# Ijeoma AI — test pages

Three pages served by one Gradio app:

| Page | URL | What it does |
|---|---|---|
| **Workbench** | `/` | Pick a destination (or paste a passage), choose a task — Classify, Ask a question, Traveller digest, Summarise — and see the output plus the exact prompt sent |
| **Chat** | `/chat` | A conversation with the guide about one destination; every reply sees the conversation so far. A checkbox hides the passage to show what the model does with no source |
| **Compare** | `/compare` | The same prompt to base Gemma 3 1B and to Ijeoma AI, side by side |

The 40 sample passages (`sample_passages.json`) come from the held-out test split,
so the model never saw them in training. Prompts are imported from the data
pipeline, so the app sends exactly the formats the model was trained on.

## Run on a Colab GPU (public link)

Open [`notebooks/ijeoma_demo_colab.ipynb`](../notebooks/ijeoma_demo_colab.ipynb)
in Colab, choose **T4 GPU**, make sure `HF_TOKEN` is in Colab Secrets, set
`REPO_URL` to this repository, and **Run all**. It prints a
`https://….gradio.live` link that works on any device for up to 72 hours.

## Run on your own GPU machine

Needs an NVIDIA GPU with ≥ 12 GB memory (≥ 6 GB without the Compare page),
Python 3.10+, and a Hugging Face token for an account that has accepted the
[Gemma licence](https://huggingface.co/google/gemma-3-1b-it).

```bash
git clone <this repository> ijeoma-ai
cd ijeoma-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -U "jax[cuda12]"
pip install -r app/requirements.txt
export HF_TOKEN=hf_...            # or: huggingface-cli login
python3 app/ijeoma_app.py
```

First start downloads Gemma 3 1B (~2 GB) and takes a few minutes. Then open
`http://localhost:7860` on the machine, or `http://<machine-ip>:7860` from
another computer on the same network. Add `--share` for a public link.

| Option | Default | |
|---|---|---|
| `--adapter PATH` | `models/ijeoma_lora_…e1.npz` | LoRA weights to load |
| `--port N` | 7860 | |
| `--host ADDR` | `0.0.0.0` | `127.0.0.1` to keep it to this machine |
| `--share` | off | public gradio.live link |
| `--preload-base` | off | load the base model at start-up, not on first Compare |
| `--fake` | off | no model at all — check the pages work |

## Memory and speed

Ijeoma AI and the base model are separate copies of Gemma 3 1B (~4 GB each in
float32). The base copy loads only when someone first presses **Compare**. Each
answer takes ~3–6 s on a T4; the first request after start-up is slower while
the generate function compiles. Requests are served one at a time.
