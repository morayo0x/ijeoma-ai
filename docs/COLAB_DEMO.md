# Running the Ijeoma AI demo on Colab (T4 GPU)

This guide starts the demo (Workbench, Chat and Compare pages) on a free Colab
T4 GPU and gives a public `gradio.live` link. It takes about five minutes.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/morayo0x/ijeoma-ai/blob/main/notebooks/ijeoma_demo_colab.ipynb)

## Before you start

You need a Hugging Face account that has accepted the Gemma licence on
[google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it), and a
Hugging Face access token for that account (Settings → Access Tokens, type
**Read**).

## Steps

1. **Open the notebook.** Use the badge above, or open
   [`notebooks/ijeoma_demo_colab.ipynb`](../notebooks/ijeoma_demo_colab.ipynb) in
   Colab (File → Open notebook → GitHub → `morayo0x/ijeoma-ai`).
2. **Select the GPU.** Runtime → Change runtime type → **T4 GPU** → Save.
3. **Add the token as a secret.** Click the key icon in the left sidebar → Add new
   secret. Name: `HF_TOKEN`, value: your token. Switch on **Notebook access**.
   Do not paste the token into a code cell.
4. **Run all cells.** Runtime → Run all. The cells:
   - check that a GPU is attached and set Keras to the JAX backend,
   - install `keras`, `keras-hub` and `gradio==5.49.1`,
   - read `HF_TOKEN` from Colab Secrets,
   - clone this repository (`REPO_URL`) into `/content/ijeoma-ai`,
   - start the app with the LoRA adapter, `--share` and `--preload-base`.
5. **Open the link.** After three to five minutes the last cell prints
   `Running on public URL: https://….gradio.live`. The pages are:
   - `/` — Workbench
   - `/chat` — Chat
   - `/compare` — Compare (baseline Gemma 3 1B vs Ijeoma AI)

The link stays valid for up to 72 hours, or until the Colab session ends or the
last cell is stopped. The first request on each page is slower while the
generate function compiles; after that, replies take about 3–6 s on a T4.

## Resource use on a T4

The T4 has 15 GB of memory. The notebook loads the fine-tuned model and the
baseline (`--preload-base`) in float32, about 4 GB each, which is the precision
used for the reported evaluation. No `--dtype` change is needed on Colab.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `NO GPU - Runtime > Change runtime type > T4` | No GPU attached. Select the T4 runtime and run all cells again. |
| `Keras already imported - Runtime > Restart session` | Keras was imported before the backend was set. Runtime → Restart session, then Run all. |
| `No Hugging Face token in Colab Secrets` | Add the `HF_TOKEN` secret and enable Notebook access (step 3). |
| `401` / `403` / `GatedRepoError` while downloading Gemma | The token's account has not accepted the Gemma licence, or the token is invalid. |
| `adapter present: False` | The clone failed or `REPO_URL` is wrong. Check `REPO_URL` in the fourth cell. |
| Out of memory | Another notebook is using the GPU. Runtime → Disconnect and delete runtime, then start again. |
| Link stops working | The session ended or timed out. Run all cells again for a new link. |

## Without GitHub access

If `REPO_URL` is set to an empty string, the notebook mounts Google Drive and
unpacks `ijeoma_app_bundle.zip` from `MyDrive/Ijeoma/` instead of cloning.

To run on your own Linux GPU machine instead of Colab, see
[`app/README.md`](../app/README.md).
