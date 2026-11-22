"""Prompt builders for the Ijeoma AI test pages.

The wording is imported from the data pipeline, not copied, so the app sends
exactly the prompts the model was trained on. A paraphrased instruction is a
different input to a fine-tuned 1B model and scores noticeably worse.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))

from ijeoma_pairs import ANSWER_INSTRUCTION, CLASSIFY_INSTRUCTIONS, LABEL_MENU, TASK_TAGS  # noqa: E402
from ijeoma_summarize import INSTR_NOTABLE, INSTR_ONE_LINE, INSTR_SUMMARY  # noqa: E402
from ijeoma_teacher import DIALOGUE_INSTRUCTION, DIGEST_INSTRUCTIONS  # noqa: E402

NO_PASSAGE = "(none available)"

SUMMARY_STYLES = {
    "Short summary (2-3 sentences)": INSTR_SUMMARY[1],
    "One sentence": INSTR_ONE_LINE[0],
    "What makes it notable": INSTR_NOTABLE[0],
}


def _passage(p):
    p = (p or "").strip()
    return p if p else NO_PASSAGE


def classify(passage):
    return (f"{TASK_TAGS['classify']} {CLASSIFY_INSTRUCTIONS[0]}\n{LABEL_MENU}\n\n"
            f"PASSAGE:\n{_passage(passage)}")


def answer(question, passage):
    return (f"{TASK_TAGS['answer']} {ANSWER_INSTRUCTION}\n\n"
            f"QUESTION: {question.strip()}\n\nPASSAGE:\n{_passage(passage)}")


def dialogue(question, passage, history):
    """history: list of (traveller, guide) pairs, oldest first."""
    lines = []
    for q, a in history:
        lines += [f"Traveller: {q}", f"Guide: {a}"]
    convo = "\n".join(lines) if lines else "(start of conversation)"
    return (f"{TASK_TAGS['answer']} {DIALOGUE_INSTRUCTION}\n\n"
            f"CONVERSATION SO FAR:\n{convo}\n\n"
            f"QUESTION: {question.strip()}\n\nPASSAGE:\n{_passage(passage)}")


def digest(passage):
    return f"{TASK_TAGS['digest']} {DIGEST_INSTRUCTIONS[2]}\n\nPASSAGE:\n{_passage(passage)}"


def summarize(passage, style):
    return f"[SUMMARIZE] {SUMMARY_STYLES[style]}\n\nPASSAGE:\n{_passage(passage)}"


TASKS = {
    "Classify": lambda p, q, s: classify(p),
    "Ask a question": lambda p, q, s: answer(q, p),
    "Traveller digest": lambda p, q, s: digest(p),
    "Summarise": lambda p, q, s: summarize(p, s),
}
