"""Routing tests for the chat interface (run with: python -m pytest tests)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from retrieval import PassageIndex, Router  # noqa: E402


@pytest.fixture(scope="module")
def router():
    return Router(PassageIndex.from_jsonl(ROOT / "pipeline" / "out" / "passages.jsonl"))


@pytest.fixture(scope="module")
def kakum(router):
    return router.route("Tell me about Kakum National Park", None, []).destination


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Tell me about Sukur", "Sukur"),
        ("What is Kakum?", "Kakum National Park"),
        ("Elmina", "Elmina"),
        ("Tell me about Elmina Castle", "Elmina Castle"),
        ("Osun-Osogbo sacred grove", "Osun Osogbo Sacred Grove"),
        ("What is the National Museum of Ghana?", "National Museum of Ghana"),
    ],
)
def test_destination_found_by_name(router, message, expected):
    route = router.route(message, None, [])
    assert route.kind == "answer"
    assert route.destination.name == expected


@pytest.mark.parametrize(
    "message",
    [
        "What is Angkor Wat?",
        "Tell me about the Eiffel Tower",
        "Kanjiro Salt Flats",
        "Olumbe Rock Shelters",
        "Zarenga Cliff Tombs",
        "asdfgh",
    ],
)
def test_unknown_places_are_not_guessed(router, message):
    assert router.route(message, None, []).kind == "not_found"


@pytest.mark.parametrize("message", ["Do I need a visa for Nigeria?", "Is Ghana safe right now?"])
def test_logistics_without_destination_is_out_of_scope(router, message):
    assert router.route(message, None, []).kind == "out_of_scope"


def test_broad_request_offers_choices_in_that_country(router):
    route = router.route("beaches in Senegal", None, [])
    assert route.kind == "choose"
    assert 2 <= len(route.choices) <= 3
    assert all("Senegal" in d.country for d in route.choices)


def test_numbered_choice_selects_destination(router):
    choices = router.route("beaches in Senegal", None, []).choices
    route = router.route("2", None, choices)
    assert route.kind == "answer"
    assert route.destination == choices[1]


@pytest.mark.parametrize(
    "message",
    [
        "When was it created?",
        "Is it in the Central Region?",
        "How do I get there from Accra?",
        "Is it safe to visit?",
        "Does the park have a canopy walkway?",
    ],
)
def test_follow_ups_stay_on_topic(router, kakum, message):
    route = router.route(message, kakum, [])
    assert route.kind == "answer"
    assert route.destination == kakum


@pytest.mark.parametrize(
    ("message", "expected_kind"),
    [
        ("Tell me about Elmina Castle", "answer"),
        ("Tell me about Sukur", "answer"),
        ("Any beaches in Senegal?", "choose"),
        ("What is the Eiffel Tower?", "not_found"),
    ],
)
def test_new_place_leaves_topic(router, kakum, message, expected_kind):
    route = router.route(message, kakum, [])
    assert route.kind == expected_kind
    assert route.destination != kakum


def test_greeting(router):
    assert router.route("hello", None, []).kind == "greeting"
