"""Ijeoma AI demo: a Gradio app with three pages.

    /          Workbench   run any task on a destination passage
    /chat      Chat        multi-turn conversation about one destination
    /compare   Compare     base Gemma 3 1B and Ijeoma AI side by side

Usage:
    python app/ijeoma_app.py                 # serve on http://0.0.0.0:7860
    python app/ijeoma_app.py --share         # also create a public gradio.live link
    python app/ijeoma_app.py --stub          # no model; for testing the interface
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gradio as gr

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))  # allow running as a script from any directory

import prompts  # noqa: E402  (needs the path set above)

DEFAULT_ADAPTER = APP_DIR.parent / "models" / "ijeoma_lora_gemma-3-1b-it-r8-lr0.0002-e1.npz"
DEFAULT_PRESET = "hf://google/gemma-3-1b-it"

with open(APP_DIR / "sample_passages.json", encoding="utf-8") as fh:
    SAMPLES = json.load(fh)
SAMPLE_LABELS = [f"{s['name']} ({s['country']})" for s in SAMPLES]
SAMPLE_BY_LABEL = dict(zip(SAMPLE_LABELS, SAMPLES, strict=True))

NOTICE = (
    "Ijeoma AI answers **only from the passage it is given**. It is a research "
    "prototype and can be wrong; do not use it for visa, health, safety, security "
    "or border advice. Sample passages are from Wikipedia, Wikivoyage and "
    "OpenStreetMap (CC BY-SA 4.0 / ODbL) and were held out of training."
)


def sample_passage(label: str) -> str:
    sample = SAMPLE_BY_LABEL.get(label)
    return sample["passage"] if sample else ""


def build_app(model) -> gr.Blocks:
    """Assemble the three pages around a model exposing ``generate``."""

    def run_task(task, passage, question, style, variant="ijeoma"):
        if task == "Ask a question" and not (question or "").strip():
            return "Enter a question first.", "", ""
        prompt = prompts.TASKS[task](passage, question or "", style)
        output, seconds = model.generate(prompt, variant=variant)
        if task == "Classify":
            output = f"```json\n{output}\n```"
        return output, prompt, f"{seconds:.1f} s"

    def passage_controls():
        picker = gr.Dropdown(
            SAMPLE_LABELS, value=SAMPLE_LABELS[0], label="Destination (held-out sample)"
        )
        passage = gr.Textbox(
            sample_passage(SAMPLE_LABELS[0]),
            lines=9,
            max_lines=20,
            label="Source passage (edit, replace or clear)",
        )
        picker.change(sample_passage, picker, passage)
        return picker, passage

    def task_controls():
        task = gr.Radio(list(prompts.TASKS), value="Ask a question", label="Task")
        question = gr.Textbox(label="Question", placeholder="e.g. What can visitors see there?")
        style = gr.Dropdown(
            list(prompts.SUMMARY_STYLES),
            value=next(iter(prompts.SUMMARY_STYLES)),
            label="Summary style",
            visible=False,
        )

        def show_relevant(selected):
            return (
                gr.update(visible=selected == "Ask a question"),
                gr.update(visible=selected == "Summarise"),
            )

        task.change(show_relevant, task, [question, style])
        return task, question, style

    with gr.Blocks(title="Ijeoma AI", theme=gr.themes.Soft()) as app:
        # Workbench --------------------------------------------------------
        gr.Markdown(
            "# Ijeoma AI · Workbench\n\nRun any task on a destination passage. "
            f"Chat and Compare are in the menu at the top right.\n\n{NOTICE}"
        )
        with gr.Row():
            with gr.Column():
                _, passage = passage_controls()
                task, question, style = task_controls()
                run_button = gr.Button("Run", variant="primary")
            with gr.Column():
                output = gr.Markdown(label="Output")
                elapsed = gr.Textbox(label="Generation time", interactive=False, max_lines=1)
                with gr.Accordion("Prompt sent to the model", open=False):
                    prompt_view = gr.Textbox(lines=12, show_label=False, interactive=False)
        run_button.click(run_task, [task, passage, question, style], [output, prompt_view, elapsed])
        gr.Examples(
            [
                ["Ask a question", "How much does it cost to visit?"],
                ["Ask a question", "What can visitors see there?"],
                ["Ask a question", "Is it safe to travel there right now?"],
            ],
            [task, question],
            label="Examples",
        )

    # Chat -----------------------------------------------------------------
    with app.route("Chat", "/chat"):
        gr.Markdown(
            "# Ijeoma AI · Chat\n\nA conversation with the guide about one destination. "
            f"Each reply is conditioned on the conversation so far, as in training.\n\n{NOTICE}"
        )
        with gr.Row():
            with gr.Column(scale=2):
                chat_picker, chat_passage = passage_controls()
                hide_passage = gr.Checkbox(
                    False, label="Withhold the passage (shows behaviour with no source)"
                )
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=430, label="Guide")
                message = gr.Textbox(placeholder="Ask the guide and press Enter", show_label=False)
                reset = gr.Button("New conversation")

        def respond(text, history, passage, withhold):
            if not (text or "").strip():
                return "", history
            exchanges, pending = [], None
            for turn in history:
                if turn["role"] == "user":
                    pending = turn["content"]
                elif pending is not None:
                    exchanges.append((pending, turn["content"]))
                    pending = None
            prompt = prompts.dialogue(text, "" if withhold else passage, exchanges)
            reply, _ = model.generate(prompt)
            return "", [
                *history,
                {"role": "user", "content": text},
                {"role": "assistant", "content": reply},
            ]

        message.submit(respond, [message, chatbot, chat_passage, hide_passage], [message, chatbot])
        reset.click(lambda: [], None, chatbot)
        chat_picker.change(lambda: [], None, chatbot)

    # Compare --------------------------------------------------------------
    with app.route("Compare", "/compare"):
        gr.Markdown(
            "# Ijeoma AI · Compare\n\nThe same prompt sent to Gemma 3 1B before "
            "fine-tuning and to Ijeoma AI. The base model loads on first use, "
            f"which takes about a minute.\n\n{NOTICE}"
        )
        with gr.Row():
            with gr.Column():
                _, cmp_passage = passage_controls()
            with gr.Column():
                cmp_task, cmp_question, cmp_style = task_controls()
                compare_button = gr.Button("Compare", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base Gemma 3 1B")
                base_output = gr.Markdown()
                base_time = gr.Textbox(label="Generation time", interactive=False, max_lines=1)
            with gr.Column():
                gr.Markdown("### Ijeoma AI")
                tuned_output = gr.Markdown()
                tuned_time = gr.Textbox(label="Generation time", interactive=False, max_lines=1)

        def compare(task, passage, question, style):
            base, _, base_s = run_task(task, passage, question, style, variant="base")
            tuned, _, tuned_s = run_task(task, passage, question, style, variant="ijeoma")
            return base, base_s, tuned, tuned_s

        compare_button.click(
            compare,
            [cmp_task, cmp_passage, cmp_question, cmp_style],
            [base_output, base_time, tuned_output, tuned_time],
        )
        gr.Examples(
            [
                ["Ask a question", "What is the entry fee?"],
                ["Ask a question", "Who built it, and when?"],
                ["Classify", ""],
                ["Traveller digest", ""],
            ],
            [cmp_task, cmp_question],
            label="Examples",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER, help="LoRA weights (.npz)")
    parser.add_argument("--preset", default=DEFAULT_PRESET, help="base model preset")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (127.0.0.1 = local only)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument(
        "--preload-base", action="store_true", help="load the base model at start-up"
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="model precision; bfloat16 halves GPU memory (evaluation used float32)",
    )
    parser.add_argument("--stub", action="store_true", help="run without a model")
    args = parser.parse_args()

    if args.stub:
        from model import StubModel

        model = StubModel()
    else:
        if not args.adapter.exists():
            parser.error(f"adapter not found: {args.adapter}")
        from model import IjeomaModel

        model = IjeomaModel(args.preset, str(args.adapter), dtype=args.dtype)
        if args.preload_base:
            model.ensure_loaded("base")

    build_app(model).queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
