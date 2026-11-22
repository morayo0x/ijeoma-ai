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
from retrieval import (  # noqa: E402
    Destination,
    PassageIndex,
    Router,
    answer_style,
    describe_custom,
    distance_km,
    load_country_guides,
    looks_like_description,
    preview,
)

DEFAULT_ADAPTER = APP_DIR.parent / "models" / "ijeoma_lora_gemma-3-1b-it-r8-lr0.0002-e1.npz"
DEFAULT_PASSAGES = APP_DIR.parent / "pipeline" / "out" / "passages.jsonl"
DEFAULT_GUIDES = APP_DIR.parent / "pipeline" / "out" / "country_guides.jsonl"
DEFAULT_PRESET = "hf://google/gemma-3-1b-it"
MAX_HISTORY = 3  # previous exchanges passed to the model; training used up to two

WELCOME = (
    "Hello! I'm **Ijeoma**, a guide to places to visit in West Africa. Ask me about "
    "a town, park, museum or historic site, for example *Kakum National Park*, "
    "*Sukur* or *Elmina Castle*, or ask for ideas, like *where should I visit in Ghana?* "
    "You can also paste a description of a place I don't know, and I'll work from that."
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


FACET_TEXT = {
    "language": "the languages spoken",
    "climate": "the weather and seasons",
    "food": "local food",
    "festivals": "festivals and holidays",
    "etiquette": "local customs and etiquette",
}
HELP_TEXT = (
    "I'm **Ijeoma**, a guide to places to visit in West Africa. I answer from travel guides "
    "and encyclopaedia articles about 1,600 destinations in 16 countries, and I cite my "
    "source each time. You can ask me to:\n\n"
    "- **describe a place**: *Tell me about Sukur*, *Why should I visit Elmina Castle?*\n"
    "- **suggest places**: *Where should I visit in Ghana?*, *waterfalls in Togo*, *places for kids*\n"
    "- **find what's nearby**: *What is near Kakum National Park?*\n"
    "- **compare**: *Kakum or Mole?*\n"
    "- **answer practical questions**: languages, climate, food, festivals, local customs\n"
    "- **read your own description** of a place and answer questions about it\n\n"
    "I can't help with visas, safety, health, prices, bookings or live information such as "
    "weather forecasts and opening hours; please check official sources for those."
)
SMALL_TALK_TEXT = (
    "I'm only a travel guide, so I'll leave that to others! Ask me about a place in West "
    "Africa, or for ideas, like *where should I visit in Ghana?*"
)
NON_ENGLISH_TEXT = (
    "Sorry, I can only chat in English for now. Please ask in English, for example: "
    "*Where should I visit in Nigeria?*"
)

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
    lines = [f"**{destination.label}**"]
    if destination.source == "user" and labels.get("country"):
        lines = [f"**{destination.name}** ({labels['country']})"]
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
    return {"topic": None, "history": [], "pending": [], "custom": None, "user_name": None}


def build_app(model, router: Router) -> gr.Blocks:
    by_id = {d.passage_id: d for d in router.index.destinations}
    by_id.update({d.passage_id: d for d in router.guides.values()})

    def current_topic(state: dict) -> Destination | None:
        if state["topic"] == "custom" and state["custom"]:
            return describe_custom(state["custom"])
        return by_id.get(state["topic"]) if state["topic"] else None

    def format_list(route) -> str:
        """Numbered list; each item gets a one-sentence description.

        By default the description is the passage's opening sentence (instant). When
        the user asks for a description of each, the model writes one per place.
        """
        items = []
        for i, d in enumerate(route.choices, 1):
            if route.describe:
                text, _ = model.generate(prompts.summarize(d.passage, "One sentence"))
            else:
                text = preview(d)
            kind = f" · *{route.labels[i - 1]}*" if i <= len(route.labels) else ""
            items.append(f"{i}. **{d.label}**{kind}: {text}")
        return "\n".join(items)

    def reply(message: str, state: dict) -> str:
        if looks_like_description(message):
            return read_description(message, state)
        topic = current_topic(state)
        pending = [by_id[i] for i in state["pending"]]
        route = router.route(message, topic, pending)
        state["pending"] = []

        if route.kind == "introduce":
            state["user_name"] = route.place
            return (
                f"Nice to meet you, {route.place}! Ask me about a place in West Africa, or ask "
                "for ideas, like *where should I visit in Ghana?*"
            )
        if route.kind == "reset":
            name = state.get("user_name")
            state.update(new_state())
            state["user_name"] = name
            return "Sure, let's start again. " + WELCOME.split("! ", 1)[1]
        if route.kind == "greeting":
            if state.get("user_name"):
                return f"Hello again, {state['user_name']}! " + WELCOME.split("! ", 1)[1]
            return WELCOME
        if route.kind == "thanks":
            return "You're welcome! Ask me about another place whenever you like."
        if route.kind == "not_found":
            return NOT_FOUND
        if route.kind == "recommend":
            # A new kind of place was requested: let go of the previous destination.
            state["topic"], state["history"] = None, []
            state["pending"] = [d.passage_id for d in route.choices]
            return (
                f"{route.heading or 'Here are some places worth exploring:'}\n\n"
                f"{format_list(route)}\n\n"
                "Reply with a number or a name to hear more, or ask about a country or a "
                "kind of place (beaches, wildlife, history, museums)."
            )
        if route.kind == "choose":
            state["pending"] = [d.passage_id for d in route.choices]
            return (
                f"I found a few places that might match:\n\n{format_list(route)}\n\n"
                "Reply with a number or a name."
            )
        if route.kind == "describe_list":
            state["pending"] = [d.passage_id for d in route.choices]
            return f"{format_list(route)}\n\nReply with a number or a name to hear more."
        if route.kind == "help":
            return HELP_TEXT
        if route.kind == "smalltalk":
            return SMALL_TALK_TEXT
        if route.kind == "non_english":
            return NON_ENGLISH_TEXT
        if route.kind == "plan":
            state["topic"], state["history"] = None, []
            state["pending"] = [d.passage_id for d in route.choices]
            return (
                f"{route.heading}\n\n{format_list(route)}\n\nReply with a number or a name to "
                "hear more, or ask *how far is X from Y?* to see distances between them."
            )
        if route.kind == "nearby":
            state["pending"] = [d.passage_id for d in route.choices]
            return (
                f"{route.heading}\n\n{format_list(route)}\n\nDistances are straight-line "
                "estimates, not road distances. Reply with a number or a name to hear more."
            )
        if route.kind == "distance":
            a, b = route.choices
            if a.latitude is None or b.latitude is None:
                return (
                    f"I don't have map coordinates for both {a.name} and {b.name}, so I can't "
                    "estimate the distance. A map app will give you routes and travel times."
                )
            km = distance_km(a, b)
            return (
                f"**{a.name}** and **{b.name}** are about **{km:.0f} km** apart in a straight "
                "line. I don't have road routes, travel times or transport options; a map app "
                "or local transport service can help with those."
            )
        if route.kind == "transport":
            place = route.destination
            return (
                f"I don't have directions, routes or transport options for **{place.name}**. "
                "A map app or local transport service can help with getting there. I can tell "
                f"you about the place itself, or what's near it: *What is near {place.name}?*"
            )
        if route.kind == "compare":
            state["topic"], state["history"] = None, []
            state["pending"] = [d.passage_id for d in route.choices if d.passage_id in by_id]
            parts = []
            for d in route.choices:
                summary, _ = model.generate(prompts.summarize(d.passage, "One sentence"))
                kind = "" if d.theme in ("", "Country Guide") else f" · *{d.theme}*"
                parts.append(f"**{d.label}**{kind}\n{summary}\n<sub>{footer(d)}</sub>")
            return (
                "Here's how they compare:\n\n" + "\n\n".join(parts) + "\n\nWhich suits you "
                "depends on what you enjoy. Ask me more about either one."
            )
        if route.kind == "live":
            return (
                "I don't have live weather. For today's conditions or a forecast, please check "
                "a weather service. I can tell you about the usual climate and seasons of a place."
            )
        if route.kind == "which_place":
            return f"Which country or place would you like to know {FACET_TEXT[route.facet]} for?"
        if route.kind == "fact_missing":
            return (
                f"My sources don't describe {FACET_TEXT[route.facet]} for {route.place}. "
                "Try asking about a nearby town or region."
            )
        if route.kind == "fact":
            # Answered from the passage that covers the question best; the current
            # destination is kept, so the conversation can carry on about it.
            source = route.destination
            answer, _ = model.generate(prompts.answer(message, source.passage))
            return f"From the guide to **{source.name}**: {answer}\n\n<sub>{footer(source)}</sub>"
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
            if destination.passage_id != "custom":
                state["custom"] = None

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

    def read_description(text: str, state: dict) -> str:
        """A pasted description of a new place: classify it and write a digest."""
        destination = describe_custom(text)
        state.update(new_state())
        state["topic"], state["custom"] = "custom", destination.passage
        raw, _ = model.generate(prompts.classify(destination.passage))
        digest, _ = model.generate(prompts.digest(destination.passage))
        state["history"].append(["Tell me about this place.", digest])
        note = ""
        if len(text.strip()) > len(destination.passage):
            note = "\n\n<sub>Only the first part of a long description was used.</sub>"
        return (
            "Here's what I make of the place you described.\n\n"
            f"{describe_labels(destination, raw)}\n\n**Why visit:** {digest}\n\n"
            "Ask me anything about it, or ask about another place."
            f"{note}\n\n<sub>{footer(destination)}</sub>"
        )

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
    parser.add_argument(
        "--guides", type=Path, default=DEFAULT_GUIDES, help="country guide passages"
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

    router = Router(PassageIndex.from_jsonl(args.passages), load_country_guides(args.guides))
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
