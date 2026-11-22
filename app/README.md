# Demo application

A Gradio application for interactive evaluation of Ijeoma AI.

| Page | Route | Function |
|---|---|---|
| Workbench | `/` | Run classification, question answering, digest or summary on a sample or user-supplied passage; shows the exact prompt sent to the model |
| Chat | `/chat` | Multi-turn conversation about one destination; each reply is conditioned on the dialogue history. The passage can be withheld to observe behaviour without a source |
| Compare | `/compare` | The same prompt sent to the baseline and the fine-tuned model, shown side by side |

The 40 sample passages in `sample_passages.json` are drawn from the test split
(two to three per country) and were not seen in training. Prompt wording is
imported from the dataset pipeline, so the application sends exactly the
formats used in training.

## Running on Colab

Open [`notebooks/ijeoma_demo_colab.ipynb`](../notebooks/ijeoma_demo_colab.ipynb),
select a T4 GPU runtime, add `HF_TOKEN` to Colab Secrets, set `REPO_URL` to this
repository and run all cells. The notebook prints a public `gradio.live` URL,
valid for up to 72 hours or until the session ends.

## Running on a GPU host

On Linux, `bash run_demo.sh` performs every step below: it creates a virtual
environment, installs dependencies, prompts for the Hugging Face token and starts
the application with a public link.


Requires an NVIDIA GPU with at least 12 GB of memory (6 GB without the Compare
page), Python 3.10+, and a Hugging Face token for an account that has accepted
the [Gemma licence](https://huggingface.co/google/gemma-3-1b-it).

```bash
git clone https://github.com/morayo0x/ijeoma-ai.git
cd ijeoma-ai
python -m venv .venv && source .venv/bin/activate
pip install -U "jax[cuda12]"
pip install -r app/requirements.txt
export HF_TOKEN=...
python app/ijeoma_app.py
```

The first start downloads Gemma 3 1B (about 2 GB). The application listens on
port 7860 on all interfaces; open `http://localhost:7860`, or
`http://<host-address>:7860` from another machine on the network.

| Option | Default | Description |
|---|---|---|
| `--adapter PATH` | `models/ijeoma_lora_gemma-3-1b-it-r8-lr0.0002-e1.npz` | LoRA weights |
| `--preset NAME` | `hf://google/gemma-3-1b-it` | Base model |
| `--host ADDR` | `0.0.0.0` | Bind address; `127.0.0.1` restricts access to the local machine |
| `--port N` | `7860` | Port |
| `--share` | off | Create a public `gradio.live` URL |
| `--preload-base` | off | Load the baseline model at start-up rather than on first comparison |
| `--dtype` | `float32` | `bfloat16` halves GPU memory; the reported evaluation used `float32` |
| `--stub` | off | Run without a model, for interface testing |

## Resource use

The fine-tuned and baseline models are held as separate instances of about 4 GB
each in float32; the baseline is loaded on first use. Generation takes about
3–6 s per request on a T4, longer for the first request while the generate
function compiles. Requests are processed sequentially.

## Modules

| File | Contents |
|---|---|
| `ijeoma_app.py` | Page layout, event handlers and command-line interface |
| `model.py` | Model loading, adapter restoration and generation (`IjeomaModel`, `StubModel`) |
| `prompts.py` | Prompt builders for each task |
