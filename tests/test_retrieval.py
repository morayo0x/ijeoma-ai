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


def test_anything_else_nearby_lists_closest_places(router, kakum):
    route = router.route("What else is interesting nearby?", kakum, [])
    assert route.kind == "nearby" and route.destination == kakum
    assert route.choices and all(label.endswith("km away") for label in route.labels)


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


# ---- request types added for common traveller questions ----


@pytest.mark.parametrize(
    "message",
    [
        "Is it safe to travel to northern Nigeria?",
        "Is Kakum safe?",
        "Can you recommend a hotel in Accra?",
        "Where can I stay near Kakum?",
        "How much is the entrance fee at Kakum?",
    ],
)
def test_safety_prices_and_lodging_are_declined_even_with_a_place(router, message):
    assert router.route(message, None, []).kind == "out_of_scope"


def test_safety_follow_up_is_declined(router, kakum):
    assert router.route("Is it safe to visit?", kakum, []).kind == "out_of_scope"


@pytest.mark.parametrize("message", ["Who are you?", "What can you do?", "help"])
def test_help(router, message):
    assert router.route(message, None, []).kind == "help"


@pytest.mark.parametrize("message", ["Tell me a joke", "How are you?"])
def test_small_talk(router, message):
    assert router.route(message, None, []).kind == "smalltalk"


@pytest.mark.parametrize(
    "message",
    [
        "Bawo ni, ibo lo dara ju lati lo?",
        "Abeg wetin una dey do?",
        "Bonjour, quels sont les endroits à visiter?",
    ],
)
def test_non_english(router, message):
    assert router.route(message, None, []).kind == "non_english"


@pytest.mark.parametrize(
    "message",
    [
        "Plan a 3-day trip to Ghana",
        "I have a week in Senegal, what should I do?",
        "How many days do I need in Mali?",
    ],
)
def test_trip_planning_offers_highlights(router, message):
    route = router.route(message, None, [])
    assert route.kind == "plan" and "itineraries" in route.heading and route.choices


def test_nearby_named_place(router):
    route = router.route("What is near Elmina Castle?", None, [])
    assert route.kind == "nearby" and route.destination.name == "Elmina Castle"
    assert len(route.choices) == len(route.labels) > 0


def test_distance_between_two_places(router):
    route = router.route("How far is Elmina from Cape Coast?", None, [])
    assert route.kind == "distance"
    assert {d.name for d in route.choices} == {"Elmina", "Cape Coast"}


def test_distance_from_current_place(router, kakum):
    route = router.route("How do I get there from Accra?", kakum, [])
    assert route.kind == "distance" and kakum in route.choices


def test_transport_to_one_place(router):
    route = router.route("How do I get to Mole National Park?", None, [])
    assert route.kind == "transport" and route.destination.name == "Mole National Park"


def test_compare_two_places(router):
    route = router.route("Which is better, Kakum or Mole?", None, [])
    assert route.kind == "compare"
    assert [d.name for d in route.choices] == ["Kakum National Park", "Mole National Park"]


def test_compare_two_countries(guided):
    route = guided.route("Is Ghana better than Nigeria?", None, [])
    assert route.kind == "compare"
    assert [d.passage_id for d in route.choices] == [
        "country-ghana-overview",
        "country-nigeria-overview",
    ]


def test_family_places(router):
    route = router.route("Good places for kids in Ghana", None, [])
    assert route.kind == "recommend" and "family" in route.heading
    assert all(d.country == "Ghana" for d in route.choices)


def test_romantic_and_budget_requests(router):
    romantic = router.route("Romantic getaway in Cape Verde", None, [])
    assert romantic.kind == "recommend" and "romantic" in romantic.heading
    budget = router.route("Cheap places to visit", None, [])
    assert budget.kind == "recommend" and "prices" in budget.heading


@pytest.mark.parametrize(
    ("message", "passage_id"),
    [
        ("What food should I try in Nigeria?", "country-nigeria-food"),
        ("What should I wear when visiting a mosque in Mali?", "country-mali-etiquette"),
    ],
)
def test_country_guide_facets(guided, message, passage_id):
    route = guided.route(message, None, [])
    assert route.kind == "fact" and route.destination.passage_id == passage_id


def test_food_question_without_country_finds_the_right_guide(guided):
    route = guided.route("What is jollof rice?", None, [])
    assert route.kind == "fact" and route.destination.passage_id.endswith("-food")


def test_phrase_question_finds_the_language_guide(guided):
    route = guided.route("How do you say thank you in Twi?", None, [])
    assert route.kind == "fact" and route.destination.passage_id == "country-ghana-language"


def test_list_names_do_not_hijack_other_questions(router):
    plan = router.route("Plan a 3-day trip to Ghana", None, [])
    elmina = next(d for d in plan.choices if d.name == "Elmina Castle")
    route = router.route("What is near Elmina Castle?", None, plan.choices)
    assert route.kind == "nearby" and route.destination == elmina
    assert router.route("Elmina Castle", None, plan.choices).kind == "answer"


def test_transport_does_not_assume_the_current_place(router, kakum):
    assert router.route("How do I get to Mole National Park?", kakum, []).kind == "transport"
    assert router.route("How far is Mole from here?", kakum, []).kind == "distance"


@pytest.mark.parametrize(
    ("message", "passage_id"),
    [
        ("where is Liberia located", "country-liberia-overview"),
        ("How big is Mali?", "country-mali-overview"),
        ("What countries border Togo?", "country-togo-overview"),
        ("Is Burkina Faso a big country?", "country-burkina-faso-overview"),
        ("Which region of Africa is Guinea in?", "country-guinea-overview"),
    ],
)
def test_general_country_questions_use_the_overview(guided, message, passage_id):
    route = guided.route(message, None, [])
    assert route.kind == "answer" and route.destination.passage_id == passage_id


def test_border_crossings_are_still_declined(guided):
    assert guided.route("Is the border crossing to Togo open?", None, []).kind == "out_of_scope"


@pytest.fixture(scope="module")
def ruins(router):
    return router.route("i want to visit interesting ruins", None, []).choices


def test_compare_list_items_by_number(router, ruins):
    route = router.route("compare 1 and 2", None, ruins)
    assert route.kind == "compare" and route.choices == ruins[:2]


def test_compare_list_items_by_description(router, ruins):
    route = router.route("compare ruins in burkina faso and kano city walls", None, ruins)
    assert route.kind == "compare"
    assert [d.name for d in route.choices] == [
        "Ancient Ferrous Metallurgy Sites of Burkina Faso",
        "Ancient Kano City Walls",
    ]


@pytest.mark.parametrize(
    ("message", "index"),
    [("tell me about 2", 1), ("tell me more about the last one", -1), ("what about the third", 2)],
)
def test_tell_me_about_a_list_item(router, ruins, message, index):
    route = router.route(message, None, ruins)
    assert route.kind == "answer" and route.destination == ruins[index]


def test_describe_selected_items(router, ruins):
    route = router.route("describe 1 and 4", None, ruins)
    assert route.kind == "describe_list" and route.choices == [ruins[0], ruins[3]]


def test_distance_between_list_items(router, ruins):
    assert router.route("how far is 1 from 3?", None, ruins).kind == "distance"


def test_which_one_is_not_a_list_reference(router, ruins):
    assert router.route("which one is the oldest?", None, ruins).kind != "compare"


@pytest.mark.parametrize(
    ("message", "name"),
    [
        ("my name is john hicks", "John Hicks"),
        ("I'm Usman Kolawole", "Usman Kolawole"),
        ("my name is Ada", "Ada"),
    ],
)
def test_multi_word_and_place_like_names(router, message, name):
    route = router.route(message, None, [])
    assert route.kind == "introduce" and route.place == name


@pytest.mark.parametrize(
    "message", ["do you remember my name", "what's my name?", "Who am I?", "do you know me"]
)
def test_asking_for_my_name(router, kakum, message):
    # Even mid-conversation about a place, this is about the user, not the place.
    assert router.route(message, kakum, []).kind == "ask_name"


def test_introduction_mid_conversation_is_not_a_follow_up(router, kakum):
    assert router.route("my name is john hicks", kakum, []).kind == "introduce"


@pytest.mark.parametrize(
    "message",
    [
        "give me countries in west-africa, and tell me what specific thing they are popular for?",
        "Which countries are in West Africa?",
        "What countries do you cover?",
    ],
)
def test_country_list(guided, message):
    route = guided.route(message, None, [])
    assert route.kind == "countries" and len(route.choices) == 16
    assert all(route.labels)


def test_country_list_items_can_be_picked_and_compared(guided):
    countries = guided.route("Which countries are in West Africa?", None, []).choices
    ghana = next(i for i, g in enumerate(countries, 1) if g.name == "Ghana")
    assert guided.route(str(ghana), None, countries).destination.name == "Ghana"
    route = guided.route("compare Ghana and Senegal", None, countries)
    assert route.kind == "compare" and [g.name for g in route.choices] == ["Ghana", "Senegal"]
