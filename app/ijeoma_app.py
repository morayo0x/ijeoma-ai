"""Ijeoma AI test pages.

Three pages, one server:
  /         Workbench  - run any task on a destination passage
  /chat     Chat       - a traveller conversation about one destination
  /compare  Compare    - base Gemma 3 1B and Ijeoma AI side by side

    python app/ijeoma_app.py --adapter models/ijeoma_lora_gemma-3-1b-it-r8-lr0.0002-e1.npz
    python app/ijeoma_app.py --share          # public gradio.live link (Colab)
    python app/ijeoma_app.py --fake           # no model: test the pages only
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gradio as gr  # noqa: E402

import prompts as P  # noqa: E402

DEFAULT_ADAPTER = os.path.join(HERE, "..", "models",
                               "ijeoma_lora_gemma-3-1b-it-r8-lr0.0002-e1.npz")
SAMPLES = json.load(open(os.path.join(HERE, "sample_passages.json"), encoding="utf-8"))
CHOICES = [f"{s['name']} ({s['country']})" for s in SAMPLES]
BY_LABEL = dict(zip(CHOICES, SAMPLES))

NOTICE = (
    "Ijeoma AI answers **only from the passage you give it**. It is a research "
    "prototype: it can still be wrong, and it must never be used for visa, "
    "health, safety, security or border advice. Sample passages come from "
    "Wikipedia, Wikivoyage and OpenStreetMap (CC BY-SA 4.0 / ODbL) and were held "
    "out of training."
)



def passage_for(label):
    return BY_LABEL[label]["passage"] if label in BY_LABEL else ""


def build(engine):
    def run(task, passage, question, style, which="ijeoma"):
        if task == "Ask a question" and not (question or "").strip():
            return "Type a question first.", "", ""
        prompt = P.TASKS[task](passage, question or "", style)
        out, dt = engine.generate(prompt, which=which)
        if task == "Classify":
            out = f"```json\n{out}\n```"
        return out, prompt, f"{dt:.1f}s"

    def passage_inputs():
        pick = gr.Dropdown(CHOICES, value=CHOICES[0], label="Destination (held-out sample)")
        passage = gr.Textbox(passage_for(CHOICES[0]), lines=9, max_lines=20,
                             label="Source passage - edit it, paste your own, or clear it")
        pick.change(passage_for, pick, passage)
        return pick, passage

    def task_inputs():
        task = gr.Radio(list(P.TASKS), value="Ask a question", label="Task")
        question = gr.Textbox(label="Question", placeholder="e.g. What can visitors see there?")
        style = gr.Dropdown(list(P.SUMMARY_STYLES), value=list(P.SUMMARY_STYLES)[0],
                            label="Summary style", visible=False)

        def toggle(t):
            return gr.update(visible=t == "Ask a question"), gr.update(visible=t == "Summarise")

        task.change(toggle, task, [question, style])
        return task, question, style

    with gr.Blocks(title="Ijeoma AI", theme=gr.themes.Soft()) as demo:
        # ---------------------------------------------------------- Workbench
        gr.Markdown(f"# Ijeoma AI · Workbench\n\nRun any task on a destination passage. "
                    "Use the menu at the top right for Chat and Compare.\n\n{NOTICE}")
        with gr.Row():
            with gr.Column():
                _, passage = passage_inputs()
                task, question, style = task_inputs()
                go = gr.Button("Run", variant="primary")
            with gr.Column():
                out = gr.Markdown(label="Output")
                took = gr.Textbox(label="Time", interactive=False, max_lines=1)
                with gr.Accordion("Exact prompt sent to the model", open=False):
                    sent = gr.Textbox(lines=12, show_label=False, interactive=False)
        go.click(run, [task, passage, question, style], [out, sent, took])
        gr.Examples(
            [["Ask a question", "How much does it cost to visit?"],
             ["Ask a question", "What can visitors see there?"],
             ["Ask a question", "Is it safe to travel there right now?"]],
            [task, question], label="Try")

    # -------------------------------------------------------------- Chat
    with demo.route("Chat", "/chat"):
        gr.Markdown(f"# Ijeoma AI · Chat\n\nTalk to the guide about one "
                    "destination. Each reply sees the whole conversation so far, "
                    "as in training.\n\n" + NOTICE)
        with gr.Row():
            with gr.Column(scale=2):
                pick_c, passage_c = passage_inputs()
                no_src = gr.Checkbox(False, label="Hide the passage from the model "
                                     "(shows what it does with no source)")
            with gr.Column(scale=3):
                chat = gr.Chatbot(type="messages", height=430, label="Guide")
                msg = gr.Textbox(placeholder="Ask the guide something and press Enter",
                                 show_label=False)
                clear = gr.Button("New conversation")

        def respond(message, history, passage, hide):
            if not (message or "").strip():
                return "", history
            pairs, q = [], None
            for turn in history:
                if turn["role"] == "user":
                    q = turn["content"]
                elif q is not None:
                    pairs.append((q, turn["content"]))
                    q = None
            prompt = P.dialogue(message, "" if hide else passage, pairs)
            reply, _ = engine.generate(prompt)
            history = history + [{"role": "user", "content": message},
                                 {"role": "assistant", "content": reply}]
            return "", history

        msg.submit(respond, [msg, chat, passage_c, no_src], [msg, chat])
        clear.click(lambda: [], None, chat)
        pick_c.change(lambda: [], None, chat)

    # -------------------------------------------------------------- Compare
    with demo.route("Compare", "/compare"):
        gr.Markdown(f"# Ijeoma AI · Compare\n\nThe same prompt to the base "
                    "model (Gemma 3 1B, before fine-tuning) and to Ijeoma AI. The "
                    "base model loads the first time you press Compare, which takes "
                    "a minute.\n\n" + NOTICE)
        with gr.Row():
            with gr.Column():
                _, passage_x = passage_inputs()
            with gr.Column():
                task_x, question_x, style_x = task_inputs()
                go_x = gr.Button("Compare", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base Gemma 3 1B")
                out_base = gr.Markdown()
                t_base = gr.Textbox(label="Time", interactive=False, max_lines=1)
            with gr.Column():
                gr.Markdown("### Ijeoma AI")
                out_ij = gr.Markdown()
                t_ij = gr.Textbox(label="Time", interactive=False, max_lines=1)

        def compare(task, passage, question, style):
            b, _, tb = run(task, passage, question, style, which="base")
            i, _, ti = run(task, passage, question, style, which="ijeoma")
            return b, tb, i, ti

        go_x.click(compare, [task_x, passage_x, question_x, style_x],
                   [out_base, t_base, out_ij, t_ij])
        gr.Examples(
            [["Ask a question", "What is the entry fee?"],
             ["Ask a question", "Who built it, and when?"],
             ["Classify", ""],
             ["Traveller digest", ""]],
            [task_x, question_x], label="Try")

    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--preset", default="hf://google/gemma-3-1b-it")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable on the local network")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    ap.add_argument("--preload-base", action="store_true",
                    help="load the base model at start-up instead of on first Compare")
    ap.add_argument("--fake", action="store_true", help="no model; test the pages only")
    a = ap.parse_args()

    if a.fake:
        from model import FakeIjeoma as Engine
    else:
        if not os.path.exists(a.adapter):
            sys.exit(f"adapter not found: {a.adapter}")
        from model import Ijeoma as Engine
    engine = Engine(a.preset, a.adapter)
    if a.preload_base and not a.fake:
        engine.ensure("base")

    build(engine).queue(default_concurrency_limit=1).launch(
        server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
