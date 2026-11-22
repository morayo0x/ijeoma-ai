"""Smoke tests: the chat application's command line and interface build correctly."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import chat_app  # noqa: E402
from model import StubModel  # noqa: E402
from retrieval import PassageIndex, Router, load_country_guides  # noqa: E402


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "app" / "chat_app.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--guides" in result.stdout


def test_app_builds_with_defaults():
    router = Router(
        PassageIndex.from_jsonl(chat_app.DEFAULT_PASSAGES),
        load_country_guides(chat_app.DEFAULT_GUIDES),
    )
    assert router.guides
    chat_app.build_app(StubModel(), router)
