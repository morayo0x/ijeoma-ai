"""Ijeoma AI chat: a single conversational interface with automatic passage retrieval.

Users type questions about West African destinations. Each message is routed to
a destination passage (see ``retrieval.py``), and the fine-tuned model answers
from that passage, conditioned on the conversation so far. Every answer cites
its source.

Usage:
    python app/chat_app.py                 # serve on http://0.0.0.0:7860
    python app/chat_app.py --share         # also create a public gradio.live link
    python app/chat_app.py --stub          # no model; for testing the interface
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
from ijeoma_build import RE_REFUSAL  # noqa: E402  (on the path via prompts)
from retrieval import Destination, PassageIndex, Router, answer_style  # noqa: E402

DEFAULT_ADAPTER = APP_DIR.parent / "models" / "ijeoma_lora_gemma-3-1b-it-r8-lr0.0002-e1.npz"
DEFAULT_PASSAGES = APP_DIR.parent / "pipeline" / "out" / "passages.jsonl"
DEFAULT_PRESET = "hf://google/gemma-3-1b-it"
MAX_HISTORY = 3  # previous exchanges passed to the model; training used up to two

WELCOME = (
    "Hello! I'm **Ijeoma**, a guide to places to visit in West Africa. Ask me about "
    "a town, park, museum or historic site, for example *Kakum National Park*, "
    "*Sukur* or *Elmina Castle*, or ask for ideas, like *where should I visit in Ghana?*"
)
NOT_FOUND = (
    "I couldn't find that place among the West African destinations I know about. "
    "Try the name of a town, park, museum or historic site, for example "
    "*Kakum National Park*, *Sukur* or *Elmina Castle*."
)
OUT_OF_SCOPE = (
    "I can't advise on visas, safety, health requirements, prices or bookings: that "
    "information changes often and is not in my sources. Please check official "
    "government travel advice. I'm happy to tell you about places to visit."
)
NOTICE = (
    "Ijeoma answers only from its source texts (Wikipedia, Wikivoyage, "
    "OpenStreetMap) and can make mistakes. Do not rely on it for visa, safety, "
    "health or border advice."
)
EXAMPLES = [
    "Tell me about Kakum National Park",
    "What is Sukur known for?",
    "Why should I visit Elmina Castle?",
    "Where should I visit in Ghana?",
    "Do I need a visa for Ghana?",
]


SUMMARY_STYLE_KEYS = {
    "one_line": "One sentence",
    "notable": "What makes it notable",
    "summary": "Short summary (2-3 sentences)",
}


def footer(destination: Destination) -> str:
    parts = [f"Source: {destination.citation}"]
    if destination.map_url:
        parts.append(f"[View on map]({destination.map_url})")
    return " · ".join(parts)


def describe_labels(destination: Destination, raw: str) -> str:
    """Render the model's classification JSON as a readable line.

    Falls back to the dataset's own labels if the model output does not parse.
    """
    try:
        text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        labels = json.loads(text[text.index("{") : text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        labels = {
            "theme": destination.theme,
            "terrain": destination.terrain or None,
            "traveler_profile": destination.traveler_profile,
            "world_heritage": destination.world_heritage,
        }
    lines = [f"**{destination.name}** ({destination.country})"]
    if labels.get("theme"):
        lines.append(f"- **Type of place:** {labels['theme']}")
    if labels.get("terrain"):
        lines.append(f"- **Setting:** {labels['terrain']}")
    if labels.get("traveler_profile"):
        lines.append(f"- **Best for:** {labels['traveler_profile']}")
    heritage = labels.get("world_heritage")
    if isinstance(heritage, str):
        heritage = heritage.strip().lower() == "true"
    lines.append(
        "- **UNESCO World Heritage:** " + ("yes" if heritage else "not listed in the source")
    )
    return "\n".join(lines)


def new_state() -> dict:
    return {"topic": None, "history": [], "pending": []}


def build_app(model, router: Router) -> gr.Blocks:
    by_id = {d.passage_id: d for d in router.index.destinations}

    def reply(message: str, state: dict) -> str:
        topic: Destination | None = by_id.get(state["topic"]) if state["topic"] else None
        pending = [by_id[i] for i in state["pending"]]
        route = router.route(message, topic, pending)
        state["pending"] = []

        if route.kind == "reset":
            state.update(new_state())
            return "Sure, let's start again. " + WELCOME.split("! ", 1)[1]
        if route.kind == "greeting":
            return WELCOME
        if route.kind == "thanks":
            return "You're welcome! Ask me about another place whenever you like."
        if route.kind == "not_found":
            return NOT_FOUND
        if route.kind == "recommend":
            # A new kind of place was requested: let go of the previous destination.
            state["topic"], state["history"] = None, []
            state["pending"] = [d.passage_id for d in route.choices]
            options = "\n".join(f"{i}. {d.label}" for i, d in enumerate(route.choices, 1))
            return (
                f"Here are some places worth exploring:\n\n{options}\n\n"
                "Reply with a number or a name to hear more, or ask about a country or a "
                "kind of place (beaches, wildlife, history, museums)."
            )
        if route.kind == "choose":
            state["pending"] = [d.passage_id for d in route.choices]
            options = "\n".join(f"{i}. {d.label}" for i, d in enumerate(route.choices, 1))
            return f"I found a few places that might match:\n\n{options}\n\nReply with a number or a name."
        if route.kind == "out_of_scope":
            answer, _ = model.generate(prompts.answer(message, ""))
            # The model was trained to decline these; fall back to a fixed reply if it does not.
            return answer if RE_REFUSAL.search(answer) else OUT_OF_SCOPE

        destination = route.destination
        if pending and destination in pending:
            # The user picked from a list ("2"); ask about the chosen place itself.
            message = f"Tell me about {destination.name}."
        if route.new_topic or state["topic"] != destination.passage_id:
            state["topic"], state["history"] = destination.passage_id, []

        style = answer_style(message)
        if style == "classify":
            raw, _ = model.generate(prompts.classify(destination.passage))
            answer = describe_labels(destination, raw)
        elif style == "digest":
            answer, _ = model.generate(prompts.digest(destination.passage))
        elif style in SUMMARY_STYLE_KEYS:
            answer, _ = model.generate(
                prompts.summarize(destination.passage, SUMMARY_STYLE_KEYS[style])
            )
        else:
            history = [tuple(pair) for pair in state["history"][-MAX_HISTORY:]]
            answer, _ = model.generate(prompts.dialogue(message, destination.passage, history))
        state["history"].append([message, answer])
        return f"{answer}\n\n<sub>{footer(destination)}</sub>"

    def respond(message, chat, state):
        if not (message or "").strip():
            return "", chat, state
        text = reply(message.strip(), state)
        chat = [*chat, {"role": "user", "content": message}, {"role": "assistant", "content": text}]
        return "", chat, state

    with gr.Blocks(title="Ijeoma AI", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Ijeoma AI\nA guide to West African destinations.")
        state = gr.State(new_state())
        chat = gr.Chatbot(
            value=[{"role": "assistant", "content": WELCOME}],
            type="messages",
            height=480,
            show_label=False,
        )
        with gr.Row():
            message = gr.Textbox(
                placeholder="Ask about a place in West Africa…", show_label=False, scale=5
            )
            send = gr.Button("Send", variant="primary", scale=1)
        gr.Examples(EXAMPLES, message, label="Try")
        reset = gr.Button("New conversation", size="sm")
        gr.Markdown(f"<sub>{NOTICE}</sub>")

        message.submit(respond, [message, chat, state], [message, chat, state])
        send.click(respond, [message, chat, state], [message, chat, state])
        reset.click(
            lambda: ([{"role": "assistant", "content": WELCOME}], new_state()),
            None,
            [chat, state],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER, help="LoRA weights (.npz)")
    parser.add_argument(
        "--passages", type=Path, default=DEFAULT_PASSAGES, help="destination passages"
    )
    parser.add_argument("--preset", default=DEFAULT_PRESET, help="base model preset")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (127.0.0.1 = local only)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="model precision; bfloat16 halves GPU memory (evaluation used float32)",
    )
    parser.add_argument("--stub", action="store_true", help="run without a model")
    args = parser.parse_args()

    router = Router(PassageIndex.from_jsonl(args.passages))
    if args.stub:
        from model import StubModel

        model = StubModel()
    else:
        if not args.adapter.exists():
            parser.error(f"adapter not found: {args.adapter}")
        from model import IjeomaModel

        model = IjeomaModel(args.preset, str(args.adapter), dtype=args.dtype)

    build_app(model, router).queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
