"""Routing tests for the chat interface (run with: python -m pytest tests)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from retrieval import (  # noqa: E402
    PassageIndex,
    Router,
    answer_style,
    describe_custom,
    load_country_guides,
    looks_like_description,
)


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
    assert route.kind in ("choose", "recommend")
    assert 2 <= len(route.choices) <= 5
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
        ("Any beaches in Senegal?", "recommend"),
        ("What is the Eiffel Tower?", "not_found"),
    ],
)
def test_new_place_leaves_topic(router, kakum, message, expected_kind):
    route = router.route(message, kakum, [])
    assert route.kind == expected_kind
    assert route.destination != kakum


def test_greeting(router):
    assert router.route("hello", None, []).kind == "greeting"


@pytest.mark.parametrize(
    "message",
    [
        "Where are interesting destinations to visit?",
        "What are the top attractions?",
        "Where can I learn about history?",
    ],
)
def test_general_questions_recommend_a_spread(router, message):
    route = router.route(message, None, [])
    assert route.kind == "recommend"
    assert len({d.country for d in route.choices}) == len(route.choices) >= 3


def test_recommendations_respect_country_and_theme(router):
    route = router.route("Where can I see wildlife in Ghana?", None, [])
    assert route.kind == "recommend"
    assert all(d.country == "Ghana" and d.theme == "Nature & Wildlife" for d in route.choices)


def test_recommendations_exclude_people(router):
    route = router.route("Recommend some places in Senegal", None, [])
    names = {d.name for d in route.choices}
    assert not names & {"Anne Pépin", "Signare", "Caty Louette", "Victoria Albis"}


def test_anything_else_nearby_uses_current_country(router, kakum):
    route = router.route("What else is interesting nearby?", kakum, [])
    assert route.kind == "recommend"
    assert all(d.country == "Ghana" for d in route.choices)


def test_short_names_need_deliberate_mention(router):
    assert router.route("oh nice, tell me more", None, []).kind == "not_found"
    route = router.route("Tell me about Bia National Park", None, [])
    assert route.kind == "answer" and route.destination.name == "Bia National Park"


@pytest.mark.parametrize("message", ["thanks", "ok thanks", "thank you!", "bye"])
def test_thanks(router, message):
    assert router.route(message, None, []).kind == "thanks"


def test_recommendation_mid_conversation_switches_country(router):
    sukur = router.route("Tell me about Sukur", None, []).destination
    route = router.route("Where can I see wildlife in Ghana?", sukur, [])
    assert route.kind == "recommend"
    assert all(d.country == "Ghana" and d.theme == "Nature & Wildlife" for d in route.choices)


def _replay(router, messages):
    """Route a conversation the way the chat app does; return the routes."""
    topic, pending, routes = None, [], []
    for message in messages:
        route = router.route(message, topic, pending)
        routes.append(route)
        pending = route.choices if route.kind in ("choose", "recommend") else []
        if route.kind == "answer":
            topic = route.destination
        if route.kind in ("recommend", "reset"):
            topic = None
    return routes


@pytest.mark.parametrize(
    "message", ["let us start again", "let's start over", "forget that", "reset", "start afresh"]
)
def test_reset(router, message):
    assert router.route(message, None, []).kind == "reset"


def test_specific_subject_is_honoured(router):
    route = router.route("i want to visit waterfall", None, [])
    assert route.kind == "recommend"
    assert all(
        any(w in d.name.lower() for w in ("falls", "waterfall", "cascade")) for d in route.choices
    )


def test_changing_the_request_lets_go_of_the_previous_place(router):
    routes = _replay(
        router,
        [
            "i want to visit waterfall",
            "1",
            "is there a waterfall there?",
            "actually I want to see ruins",
        ],
    )
    waterfall = routes[1].destination
    assert routes[2].kind == "answer" and routes[2].destination == waterfall
    assert routes[3].kind == "recommend"
    assert waterfall not in routes[3].choices


def test_ruins_replace_waterfall_list(router):
    routes = _replay(router, ["i want to visit waterfall", "what about ruins?"])
    assert routes[1].kind == "recommend"
    first = {d.passage_id for d in routes[0].choices}
    assert not first & {d.passage_id for d in routes[1].choices}


@pytest.mark.parametrize(
    ("message", "style"),
    [
        ("Why should I visit?", "digest"),
        ("Give me an overview", "digest"),
        ("Sum it up in one sentence", "one_line"),
        ("What is Sukur known for?", "notable"),
        ("Summarise it", "summary"),
        ("What kind of place is it?", "classify"),
        ("Is it a World Heritage site?", "classify"),
        ("When was it built?", "dialogue"),
    ],
)
def test_answer_style(message, style):
    assert answer_style(message) == style


@pytest.mark.parametrize(
    "message",
    ["Why should I visit?", "Give me an overview", "Summarise it", "What kind of place is it?"],
)
def test_style_requests_stay_on_current_place(router, kakum, message):
    route = router.route(message, kakum, [])
    assert route.kind == "answer" and route.destination == kakum


def test_map_link(kakum):
    assert kakum.map_url and "openstreetmap.org" in kakum.map_url


LEKKI = (
    "The Lekki Conservation Centre is a nature reserve in Lagos, Nigeria, run by the Nigerian "
    "Conservation Foundation. It covers 78 hectares of wetland and forest along the Lekki "
    "Peninsula and protects mangroves, monkeys, crocodiles and many bird species. Visitors walk "
    "along raised boardwalks through the forest, and the centre has a canopy walkway about "
    "401 metres long."
)


def test_pasted_description_is_detected():
    assert looks_like_description(LEKKI)
    assert not looks_like_description("Tell me about Kakum National Park")
    assert not looks_like_description(
        "What is the longest canopy walkway in Africa and where can I find it and how high is "
        "it above the forest floor in metres, and is it open to visitors every day of the week?"
    )


def test_pasted_description_becomes_the_topic(router):
    place = describe_custom(LEKKI)
    assert place.name == "Lekki Conservation Centre"
    assert place.citation == "your description"
    for question in ("How long is the canopy walkway?", "Why should I visit?"):
        route = router.route(question, place, [])
        assert route.kind == "answer" and route.destination.passage_id == "custom"
    assert router.route("Tell me about Sukur", place, []).destination.name == "Sukur"


@pytest.mark.parametrize(
    "message", ["What's the weather in Lagos today?", "What's the forecast for Accra this week?"]
)
def test_live_weather_is_declined(router, message):
    assert router.route(message, None, []).kind == "live"


def test_facet_without_place_asks_which(router):
    assert router.route("What's the weather like?", None, []).kind == "which_place"


@pytest.mark.parametrize(
    "message",
    ["What language do people speak in Ghana?", "When is the rainy season in Nigeria?"],
)
def test_country_facet_uses_a_passage_from_that_country(router, message):
    route = router.route(message, None, [])
    assert route.kind == "fact"
    assert route.destination.country in message


def test_facet_falls_back_to_a_nearby_passage(router, kakum):
    route = router.route("What language do they speak there?", kakum, [])
    assert route.kind == "fact" and route.destination.country == "Ghana"


def test_facet_uses_the_place_itself_when_it_covers_it(router):
    sukur = router.route("Tell me about Sukur", None, []).destination
    route = router.route("What language do they speak there?", sukur, [])
    assert route.kind == "answer" and route.destination == sukur


def test_best_time_is_not_a_recommendation(router, kakum):
    assert router.route("When is the best time to visit?", kakum, []).kind == "fact"


@pytest.fixture(scope="module")
def guided():
    return Router(
        PassageIndex.from_jsonl(ROOT / "pipeline" / "out" / "passages.jsonl"),
        load_country_guides(ROOT / "pipeline" / "out" / "country_guides.jsonl"),
    )


@pytest.mark.parametrize(
    ("message", "passage_id"),
    [
        ("Tell me about Ghana", "country-ghana-overview"),
        ("Tell me about the Ivory Coast", "country-cote-d-ivoire-overview"),
        ("Tell me about Guinea-Bissau", "country-guinea-bissau-overview"),
        ("What language do they speak in Mali?", "country-mali-language"),
        ("What language is spoken in Liberia?", "country-liberia-language"),
        ("When is the rainy season in Nigeria?", "country-nigeria-climate"),
    ],
)
def test_country_questions_use_country_guides(guided, message, passage_id):
    assert guided.route(message, None, []).destination.passage_id == passage_id


@pytest.mark.parametrize(
    ("message", "name"),
    [("Tell me about Bissau", "Bissau"), ("Tell me about Cape Coast", "Cape Coast")],
)
def test_cities_named_like_countries_still_resolve(guided, message, name):
    assert guided.route(message, None, []).destination.name == name


def test_niger_is_not_nigeria(guided):
    route = guided.route("What is the weather like in Niger?", None, [])
    assert route.destination.country == "Niger"


def test_place_without_facet_uses_its_country_guide(guided):
    kakum = guided.route("Tell me about Kakum National Park", None, []).destination
    route = guided.route("What language do they speak there?", kakum, [])
    assert route.kind == "fact" and route.destination.passage_id == "country-ghana-language"


@pytest.mark.parametrize(
    ("message", "name"),
    [("my name morayo", "Morayo"), ("My name is Morayo", "Morayo"), ("call me Tunde", "Tunde")],
)
def test_introductions(router, message, name):
    route = router.route(message, None, [])
    assert route.kind == "introduce" and route.place == name


def test_description_of_each_is_honoured(router):
    route = router.route("give me ruins to visit and a description of each", None, [])
    assert route.kind == "recommend" and route.describe
    follow_up = router.route("describe them", None, route.choices)
    assert follow_up.kind == "describe_list" and follow_up.choices == route.choices


def test_plain_lists_are_not_described_by_the_model(router):
    assert not router.route("where are interesting places to visit?", None, []).describe


def test_general_request_gives_a_labelled_mix(router):
    route = router.route("where are interesting places to visit?", None, [])
    assert route.heading.startswith("Here's a mix of places")
    assert len(route.labels) == len(route.choices) >= 5
    assert {"Ruins", "Wildlife & safari", "Waterfall"} <= set(route.labels)


@pytest.mark.parametrize(
    ("message", "heading_words"),
    [
        ("give me ruins to visit", "ruins"),
        ("waterfalls in Togo", "waterfalls worth visiting in Togo"),
        ("safari in Senegal", "wildlife and safari spots worth visiting in Senegal"),
    ],
)
def test_heading_names_the_request(router, message, heading_words):
    assert heading_words in router.route(message, None, []).heading


def test_waterfall_towns_are_not_waterfalls(router):
    names = {d.name for d in router.route("waterfalls in Togo", None, []).choices}
    assert names and all("Falls" in n or "falls" in n for n in names)
