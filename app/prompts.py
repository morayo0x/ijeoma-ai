"""Prompt builders for the Ijeoma AI demo.

Instruction wording is imported from the data pipeline rather than restated, so
the demo sends exactly the prompt formats the model was trained on. A fine-tuned
1B model is sensitive to paraphrase; a reworded instruction is effectively an
unseen input.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))

from ijeoma_pairs import (
    ANSWER_INSTRUCTION,
    CLASSIFY_INSTRUCTIONS,
    LABEL_MENU,
    TASK_TAGS,
)
from ijeoma_summarize import INSTR_NOTABLE, INSTR_ONE_LINE, INSTR_SUMMARY
from ijeoma_teacher import DIALOGUE_INSTRUCTION, DIGEST_INSTRUCTIONS

NO_PASSAGE = "(none available)"

SUMMARY_STYLES = {
    "Short summary (2-3 sentences)": INSTR_SUMMARY[1],
    "One sentence": INSTR_ONE_LINE[0],
    "What makes it notable": INSTR_NOTABLE[0],
}


def _passage(text: str | None) -> str:
    text = (text or "").strip()
    return text or NO_PASSAGE


def classify(passage: str) -> str:
    return (
        f"{TASK_TAGS['classify']} {CLASSIFY_INSTRUCTIONS[0]}\n{LABEL_MENU}\n\n"
        f"PASSAGE:\n{_passage(passage)}"
    )


def answer(question: str, passage: str) -> str:
    return (
        f"{TASK_TAGS['answer']} {ANSWER_INSTRUCTION}\n\n"
        f"QUESTION: {question.strip()}\n\nPASSAGE:\n{_passage(passage)}"
    )


def dialogue(question: str, passage: str, history: list[tuple[str, str]]) -> str:
    """Build a conversation-turn prompt. ``history`` is (traveller, guide) pairs, oldest first."""
    lines = []
    for q, a in history:
        lines += [f"Traveller: {q}", f"Guide: {a}"]
    convo = "\n".join(lines) if lines else "(start of conversation)"
    return (
        f"{TASK_TAGS['answer']} {DIALOGUE_INSTRUCTION}\n\n"
        f"CONVERSATION SO FAR:\n{convo}\n\n"
        f"QUESTION: {question.strip()}\n\nPASSAGE:\n{_passage(passage)}"
    )


def digest(passage: str) -> str:
    return f"{TASK_TAGS['digest']} {DIGEST_INSTRUCTIONS[2]}\n\nPASSAGE:\n{_passage(passage)}"


def summarize(passage: str, style: str) -> str:
    return f"[SUMMARIZE] {SUMMARY_STYLES[style]}\n\nPASSAGE:\n{_passage(passage)}"


TASKS = {
    "Classify": lambda p, q, s: classify(p),
    "Ask a question": lambda p, q, s: answer(q, p),
    "Traveller digest": lambda p, q, s: digest(p),
    "Summarise": lambda p, q, s: summarize(p, s),
}
