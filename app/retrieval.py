"""Passage retrieval and conversation routing for the chat-only interface.

The fine-tuned model answers reliably only when it is given the source passage,
so every user message is first routed to a destination passage:

1. **Name match** - the message names a destination (generic words such as
   "national", "park" or country names are not enough on their own).
2. **Follow-up** - no new destination is named, so the current one is kept.
3. **Search** - a vague request ("beaches in Senegal") is matched against the
   passages with BM25 and the best candidates are offered as numbered choices.
4. **Out of scope** - visa, safety, health, money or booking questions with no
   destination are answered without a passage, which the model was trained to
   decline.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

COUNTRIES = {
    "benin", "burkina", "faso", "cape", "verde", "cabo", "ivoire", "cote", "gambia",
    "ghana", "guinea", "bissau", "liberia", "mali", "mauritania", "niger", "nigeria",
    "senegal", "sierra", "leone", "togo", "africa", "african", "west",
    "nigerian", "ghanaian", "senegalese", "malian", "beninese", "togolese", "ivory",
    "ivorian", "gambian", "liberian", "guinean", "mauritanian", "nigerien", "burkinabe",
}  # fmt: skip

GENERIC = COUNTRIES | {
    "the", "of", "and", "a", "an", "de", "la", "le", "les", "du", "des", "da", "do", "dos",
    "national", "park", "reserve", "forest", "museum", "fort", "castle", "state", "mosque",
    "beach", "palace", "lake", "river", "city", "town", "village", "sanctuary", "waterfalls",
    "falls", "monument", "memorial", "art", "nature", "natural", "wildlife", "hills", "hill",
    "game", "house", "rock", "island", "centre", "center", "cemetery", "east", "west",
    "north", "south", "classified", "church", "cathedral", "market", "site", "area", "region",
    "district", "grove", "sacred", "gallery", "university", "stadium", "zoo", "garden",
    "gardens", "botanical", "mount", "mountain", "lagoon", "bay", "old", "new", "great",
    "royal", "historic", "colonial", "slave", "museu", "musee", "parc", "praia", "monte",
    "history", "heritage", "culture", "cultural",
}  # fmt: skip

OUT_OF_SCOPE = re.compile(
    r"\b(visa|visas|vaccin\w*|malaria|yellow fever|safe|safety|security|danger\w*|"
    r"border (crossings?|closures?|posts?|status|open|closed)|cross(ing)? the border|exchange rate|currency|flight\w*|airline\w*|book|booking|hotel\w*|"
    r"accommodation|lodging|guesthouse\w*|where (can|should|to) (i )?(stay|sleep)|"
    r"how much (does|do|is|are|will|would)|cost|costs|price|prices|fee|fees|entrance charge)\b",
    re.I,
)
REFERS_BACK = re.compile(r"\b(it|its|it's|there|here|this place|that place|the place)\b", re.I)
GREETING = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b", re.I)
THANKS = re.compile(r"\b(thanks?|thank you|cheers|bye|goodbye)\b", re.I)

# Common words that may be absent from the passages but are not place names.
QUERY_WORDS = {
    "tell", "about", "what", "whats", "where", "when", "which", "who", "how", "why", "is",
    "are", "was", "were", "can", "could", "would", "should", "do", "does", "did", "me", "my",
    "i", "you", "your", "please", "from", "by", "visit", "visiting", "see", "go", "going", "know", "show",
    "any", "some", "there", "here", "it", "its", "this", "that", "in", "at", "to", "for",
    "with", "on", "near", "around", "best", "good", "nice", "interesting", "places", "place",
    "things", "thing", "recommend", "suggest", "want", "like", "travel", "trip", "tourist",
    "tourists", "attractions", "attraction", "destinations", "destination", "anything",
    "oh", "ok", "okay", "cool", "great", "wow", "yes", "no", "sure", "more", "also",
    "else", "other", "another", "really", "very", "much", "lot", "lots", "sounds", "the",
    "a", "an", "and", "or", "but", "so", "then", "thats", "that's", "be", "of",
}  # fmt: skip

INTRODUCTION = re.compile(
    r"^\s*(?:hi|hello|hey)?[,!.]?\s*(?:my name is|my name's|my name|call me|i am|i'm|im|this is)\s+"
    r"([A-Za-zÀ-ÿ][\w'-]{1,30}(?:\s+[A-Za-zÀ-ÿ][\w'-]{1,30}){0,2})\s*[.!]?\s*$",
    re.I,
)
COUNTRY_LIST = re.compile(
    r"\b(which|what|list|name|give me|show me|all)\b.{0,40}\b(countries|nations)\b|"
    r"\b(countries|nations)\b.{0,30}\b(in|of) west[- ]?africa\b|"
    r"\b(countries|nations)\b.{0,30}\b(do you|you) (cover|know)\b",
    re.I,
)
ASK_NAME = re.compile(
    r"\b(what'?s my name|what is my name|do you (remember|know|recall) (my name|me|who i am)|"
    r"who am i|remember me|my name\?)",
    re.I,
)
DESCRIBE_EACH = re.compile(
    r"\b(descri\w*\s+(?:of\s+)?(?:each|them|all|these|those|every)|"
    r"(?:a|some)?\s*descriptions?\s+(?:of|for)\s+(?:each|them|all|these|those)|"
    r"with\s+(?:a\s+|short\s+)?descriptions?|tell me (?:more )?about (?:each|them|all of them|these|those)|"
    r"(?:each|every) (?:one|of them))\b",
    re.I,
)
COUNTRY_FACT = re.compile(
    r"\b(where|located|location|situated|capital|population|size|how (big|large|small)|"
    r"big|large|small|area|borders?|bordering|neighbou?r\w*|region|part of|continent|"
    r"independen\w*|official|government|president|economy|history|known for|famous for|"
    r"people|religion\w*)\b",
    re.I,
)
ABOUT_COUNTRY = re.compile(
    r"\b(tell me (about|more about)|what is|what's|describe|overview|introduc\w*|information|info|"
    r"learn about|know about)\b",
    re.I,
)
RECOMMEND = re.compile(
    r"\b(interesting|recommend\w*|suggest\w*|best|top|must[- ]see|worth (visiting|seeing)|"
    r"where (should|can|could) i (go|travel|visit|see)|places to (visit|go|see)|things to (do|see)|"
    r"what (should|can) i (see|do|visit)|destinations|attractions|ideas)\b",
    re.I,
)

RESET = re.compile(
    r"\b(start (again|over|afresh)|restart|reset|begin again|new (conversation|search|chat)|"
    r"forget (that|it|this|everything)|clear (that|it|this|the chat)|from (the )?(start|beginning|scratch))\b",
    re.I,
)
# A change of request: "actually", "instead", "rather", "something else".
CHANGE = re.compile(
    r"\b(actually|instead|rather|something else|change|different|now i want)\b", re.I
)

HELP = re.compile(
    r"\b(who are you|what are you|what can you (do|tell)|what do you (do|know)|"
    r"how (do|does) (you|this|it) work|what (can|should) i ask|are you (a bot|human|real|an ai)|"
    r"help me|^help\??$)",
    re.I,
)
SMALL_TALK = re.compile(
    r"\b(joke|how are you|what'?s up|how'?s it going|i love you|you'?re (great|funny|smart|stupid|"
    r"amazing)|are you (married|single)|favou?rite (colou?r|food|song)|sing|tell me something funny)\b",
    re.I,
)
# Words in Nigerian and West African languages and French, for a polite "English only" reply.
NON_ENGLISH_WORDS = {
    "bawo", "nibo", "ibo", "dara", "jowo", "ese", "emi", "lati", "abeg", "wetin", "una", "dey",
    "wahala", "sabi", "oya", "comot", "sannu", "yaya", "nagode", "lafiya", "kedu", "nnoo", "biko",
    "daalu", "ndewo", "akwaaba", "medaase", "bonjour", "merci", "voudrais", "visiter", "quels",
    "sont", "endroits", "pourquoi", "comment", "beaucoup", "ndi", "nna", "ekaaro", "ekasan",
}  # fmt: skip
PLAN = re.compile(
    r"\b(plan|planning|itinerar\w*|\d+[- ]?days?|\d+[- ]?weeks?|"
    r"(one|two|three|four|five|six|seven|a|few|couple of) (days?|weeks?)|weekend|"
    r"how (many|much) (days|time)|how long (should|do) i (stay|spend|need))\b",
    re.I,
)
NEARBY = re.compile(
    r"\b(near|nearby|close to|close by|surrounding|in the (area|vicinity)|next to|"
    r"neighbou?ring|not far from|around (here|it|there))\b",
    re.I,
)
DISTANCE = re.compile(
    r"\b(how far|distance|how many (km|kilometres|kilometers|miles)|"
    r"how (do|can|would) i (get|go|travel) (to|from|there)|get (to|from)|travel (to|from)|"
    r"drive (to|from))\b",
    re.I,
)
COMPARE = re.compile(
    r"\b(better|best of|compare|comparison|versus|vs\.?|difference between|which (one )?"
    r"(should|is|would)|or)\b",
    re.I,
)
# Words that frame a phrase question ("how do you say thank you in Twi?") rather than
# identify what it is about.
PHRASE_WORDS = {
    "say", "thank", "thanks", "hello", "goodbye", "please", "phrase", "phrases", "word",
    "words", "mean", "means", "language", "languages", "speak", "spoken",
}  # fmt: skip
ROMANTIC = re.compile(r"\b(romantic|romance|honeymoon|couples?|getaway)\b", re.I)
BUDGET = re.compile(r"\b(cheap|budget|affordable|inexpensive|low[- ]cost)\b", re.I)

# Specific things a traveller asks to see. A recommendation must mention one of
# these words in the destination's name or description, not merely share a theme.
SUBJECTS = {
    "waterfall": ("waterfall", "waterfalls", "falls", "cascade", "cascades", "chutes"),
    "ruins": ("ruins", "ruin", "archaeological", "excavation", "excavations", "ancient"),
    "beach": ("beach", "beaches", "plage"),
    "lake": ("lake", "lakes", "lagoon"),
    "mountain": ("mountain", "mountains", "mount", "mont", "monte", "peak", "plateau", "hills"),
    "cave": ("cave", "caves", "grotto"),
    "island": ("island", "islands", "islet", "ilha", "ilheu", "ile"),
    "museum": ("museum", "museums", "musee", "museu", "gallery"),
    "mosque": ("mosque", "mosques"),
    "fort": ("fort", "forts", "fortress", "castle", "castles", "forte", "fortaleza"),
    "palace": ("palace", "palaces"),
    "river": ("river", "rivers"),
    "forest": ("forest", "forests", "rainforest"),
    "zoo": ("zoo",),
    "market": ("market", "markets"),
    "wildlife": ("park", "reserve", "sanctuary", "game", "wildlife", "biosphere"),
    "family": ("zoo", "amusement", "garden", "gardens", "botanical", "beach", "leisure",
               "recreation", "resort", "fun", "waterpark"),
}  # fmt: skip
SUBJECT_TRIGGERS = {
    "waterfall": ("waterfall", "waterfalls", "falls", "cascade"),
    "ruins": ("ruins", "ruin", "archaeology", "archaeological", "ancient"),
    "beach": ("beach", "beaches", "seaside"),
    "lake": ("lake", "lakes", "lagoon"),
    "mountain": ("mountain", "mountains", "hills", "hiking", "climb", "climbing"),
    "cave": ("cave", "caves"),
    "island": ("island", "islands"),
    "museum": ("museum", "museums", "gallery", "galleries"),
    "mosque": ("mosque", "mosques"),
    "fort": ("fort", "forts", "castle", "castles"),
    "palace": ("palace", "palaces"),
    "river": ("river", "rivers"),
    "forest": ("forest", "forests", "rainforest"),
    "zoo": ("zoo",),
    "market": ("market", "markets"),
    "wildlife": ("wildlife", "safari", "animals", "elephants", "lions", "game"),
    "family": ("kids", "children", "child", "family", "families", "toddlers"),
}  # fmt: skip
SUBJECT_TITLES_EXTRA = {"wildlife": "wildlife and safari spots", "family": "family-friendly places"}
NAME_NEEDS_THEME = {"wildlife", "family"}
# Themes under which a description-only mention counts (a waterfall *town* does not).
SUBJECT_THEMES = {
    "waterfall": {"Landscape & Scenery", "Nature & Wildlife"},
    "ruins": {"Archaeological Site", "Historic Site", "Monument & Memorial"},
    "beach": {"Coast & Beach", "Town or Area Guide", "Nature & Wildlife"},
    "lake": {"Landscape & Scenery", "Nature & Wildlife"},
    "mountain": {"Landscape & Scenery", "Nature & Wildlife"},
    "cave": {"Landscape & Scenery", "Nature & Wildlife", "Sacred & Ceremonial"},
    "island": {"Coast & Beach", "Nature & Wildlife", "Town or Area Guide", "Historic Site"},
    "museum": {"Museum & Gallery"},
    "mosque": {"Sacred & Ceremonial"},
    "fort": {"Fort & Castle"},
    "palace": {"Palace & Royal Site"},
    "river": {"Nature & Wildlife", "Landscape & Scenery"},
    "forest": {"Nature & Wildlife"},
    "zoo": {"Family & Leisure", "Nature & Wildlife"},
    "market": {"Town or Area Guide", "General Attraction"},
    "wildlife": {"Nature & Wildlife"},
    "family": {"Family & Leisure", "Coast & Beach", "Nature & Wildlife", "Landscape & Scenery"},
}

# Kinds of place offered when someone asks for "interesting places" with no preference.
VARIETY = (
    ("Ruins", set(), {"ruins"}),
    ("Wildlife & safari", set(), {"wildlife"}),
    ("Waterfall", set(), {"waterfall"}),
    ("Beach", set(), {"beach"}),
    ("Fort & castle", set(), {"fort"}),
    ("Sacred site", {"Sacred & Ceremonial"}, set()),
    ("Historic town", {"Historic Site"}, set()),
    ("Museum", set(), {"museum"}),
)
SUBJECT_TITLES = {
    "waterfall": "waterfalls", "ruins": "ruins and archaeological sites", "beach": "beaches",
    "lake": "lakes", "mountain": "mountains and hills", "cave": "caves", "island": "islands",
    "museum": "museums", "mosque": "mosques", "fort": "forts and castles", "palace": "palaces",
    "river": "rivers", "forest": "forests", "zoo": "zoos", "market": "markets",
}  # fmt: skip
THEME_TITLES = {
    "Coast & Beach": "beaches and coastal spots", "Nature & Wildlife": "nature and wildlife spots",
    "Landscape & Scenery": "scenic places", "Museum & Gallery": "museums and galleries",
    "Fort & Castle": "forts and castles", "Historic Site": "historic places",
    "Archaeological Site": "archaeological sites", "Monument & Memorial": "monuments",
    "Palace & Royal Site": "palaces and royal sites", "Sacred & Ceremonial": "sacred sites",
    "Family & Leisure": "family-friendly places", "Town or Area Guide": "towns and cities",
}  # fmt: skip


def list_heading(country: str | None, themes: set[str], subjects: set[str]) -> str:
    where = f"in {country_title(country)}" if country else "across West Africa"
    if subjects:
        titles = {**SUBJECT_TITLES, **SUBJECT_TITLES_EXTRA}
        kinds = " and ".join(titles[s] for s in sorted(subjects))
    elif themes and len(themes) <= 2:
        kinds = " and ".join(THEME_TITLES.get(t, t.lower()) for t in sorted(themes))
    elif themes:
        kinds = "places"
        kinds = "historic and cultural places" if themes & set(THEME_TITLES) else kinds
    else:
        return f"Here's a mix of places worth visiting {where}, one of each kind:"
    return f"Here are some {kinds} worth visiting {where}:"


def country_title(country: str) -> str:
    special = {
        "cote d'ivoire": "Côte d'Ivoire",
        "gambia": "The Gambia",
        "guinea-bissau": "Guinea-Bissau",
    }
    return special.get(country, country.title())


# Words in a request mapped to the dataset's theme labels.
THEME_KEYWORDS = {
    "Coast & Beach": ("beach", "beaches", "coast", "coastal", "sea", "seaside", "ocean"),
    "Nature & Wildlife": (
        "nature", "wildlife", "animals", "safari", "forest", "forests", "birds",
        "birdwatching", "national park", "national parks", "rainforest", "elephants",
    ),
    "Landscape & Scenery": (
        "waterfall", "waterfalls", "mountain", "mountains", "hills", "scenery", "landscape",
        "views", "hiking", "lake", "lakes",
    ),
    "Museum & Gallery": ("museum", "museums", "gallery", "galleries", "art"),
    "Fort & Castle": ("fort", "forts", "castle", "castles", "slave trade"),
    "Historic Site": ("history", "historic", "historical", "heritage", "old town"),
    "Archaeological Site": ("ruins", "archaeology", "archaeological", "ancient"),
    "Monument & Memorial": ("monument", "monuments", "memorial"),
    "Palace & Royal Site": ("palace", "palaces", "royal", "kings", "kingdom"),
    "Sacred & Ceremonial": ("sacred", "shrine", "shrines", "spiritual", "religious", "mosque", "mosques", "grove"),
    "Family & Leisure": ("family", "kids", "children", "amusement", "leisure"),
    "Town or Area Guide": ("city", "cities", "town", "towns"),
}  # fmt: skip
THEME_GROUPS = {
    "history": ("Historic Site", "Fort & Castle", "Archaeological Site", "Monument & Memorial", "Palace & Royal Site"),
    "culture": ("Sacred & Ceremonial", "Museum & Gallery", "Palace & Royal Site", "Historic Site"),
}  # fmt: skip
PERSON = re.compile(
    r"\bwas an? [\w\s\"-]{0,40}\b(woman|man|women|merchant|trader|signare|signara|"
    r"businesswoman|businessperson|slaver|historian|businessman|ruler|queen|king|cleric|"
    r"strategist|founder|emperor|leader|chief|warrior|politician|writer|poet|artist|"
    r"musician|soldier|general|explorer|missionary|scholar)\b"
    # Lifespan right after the name, then "was" (possibly after aliases):
    # "X (c. 1615 – after 1670) was ...", "X (c. 1830 – June 2, 1900), also known as Y, was ..."
    r"|^[^.(]{0,80}\((c\.\s*)?\d{3,4}\s*[–-][^)]*\d{3,4}\)[^.]{0,160}\bwas\b"
    r"|^[^.(]{0,80}\(\s*[bd]\.[^)]*\),?\s+(was|is)\b"
    r"|^\w+s were [\w\s-]{0,40}\bwomen\b",
    re.I,
)

SOURCE_LABELS = {"wikipedia": "Wikipedia", "wikivoyage": "Wikivoyage", "osm": "OpenStreetMap"}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalise(text).replace("'", " "))


def _label(value) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in ("", "none", "nan", "null") else text


def _coordinate(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


# How the user wants a destination described. Checked in order; the default is
# a conversational answer.
ANSWER_STYLES = (
    ("one_line", re.compile(
        r"\b(in )?(one|a single|1) sentence\b|one[- ]liner|in a nutshell|\bbriefly\b|\btl;?dr\b", re.I)),
    ("classify", re.compile(
        r"what (kind|type|sort) of (place|destination|attraction|site)|"
        r"who (is it|would it be|is this) (good|suitable|best|ideal) for|"
        r"(is it|is this)( a)? (unesco|world heritage)|\bcategor(y|ise|ize)\b|\btags?\b|\blabels?\b", re.I)),
    ("notable", re.compile(
        r"(known|famous|notable|significant|special|remembered) for|what makes it|"
        r"why is it (famous|important|significant|special)|\bsignificance\b", re.I)),
    ("digest", re.compile(
        r"why (should|would) (i|we|someone|anyone) (visit|go)|why (visit|go there)|"
        r"\boverview\b|\bintroduc(e|tion)\b|\brundown\b|sell (it|this place) to me|"
        r"(travel|traveller'?s?|visitor'?s?) (guide|digest|summary)", re.I)),
    ("summary", re.compile(r"\bsummar(y|ise|ize)\b|\bsum (it|this|that) up\b|\brecap\b|short version", re.I)),
)  # fmt: skip


# Practical questions answered from whichever passage covers them best.
FACETS = {
    "language": (
        re.compile(
            r"\b(language|languages|speak|spoken|dialect|dialects|lingua franca|"
            r"how (do|would|can) (you|i) say|phrases?|thank you in|hello in)\b",
            re.I,
        ),
        re.compile(r"\b(language|languages|spoken|speak|lingua franca|dialect)\b", re.I),
    ),
    "climate": (
        re.compile(
            r"\b(weather|forecast|climate|rainy|rains?|dry season|wet season|seasons?|harmattan|"
            r"temperatures?|humid|humidity|best time (to|for) (visit|go|travel)|"
            r"when (should|to) (i )?(visit|go))\b",
            re.I,
        ),
        re.compile(
            r"\b(climate|rainy season|dry season|wet season|harmattan|rainfall|temperatures?|"
            r"humid|humidity|best time to visit|rains)\b",
            re.I,
        ),
    ),
    "food": (
        re.compile(
            r"\b(food|foods|eat|eating|dish|dishes|cuisine|meals?|restaurants?|drinks?|street food|"
            r"jollof|fufu|suya|banku|kenkey|egusi|thieboudienne|ceebu jen|attieke|yassa|mafe|"
            r"kelewele|waakye|garri|akara|moi moi|pepper soup|palm wine|bissap|plantain)\b",
            re.I,
        ),
        re.compile(
            r"\b(dish|dishes|cuisine|food|restaurant|jollof|fufu|suya|delicacy|delicacies|"
            r"street food|palm wine)\b",
            re.I,
        ),
    ),
    "festivals": (
        re.compile(
            r"\b(festivals?|holidays?|celebrat\w*|carnivals?|events?|ceremon(y|ies)|"
            r"durbar|homowo|panafest|odwira|fetu)\b",
            re.I,
        ),
        re.compile(
            r"\b(festival|festivals|celebrat\w*|carnival|ceremony|ceremonies|durbar)\b", re.I
        ),
    ),
    "etiquette": (
        re.compile(
            r"\b(wear|dress|dress code|etiquette|customs?|respect\w*|polite|rude|taboos?|"
            r"photos?|photograph\w*|pictures?|tipping|tip|behave|behaviou?r|offend\w*)\b",
            re.I,
        ),
        re.compile(
            r"\b(dress modestly|etiquette|respect|taboo|customs|photograph\w*|permission|"
            r"modest|remove (your )?shoes)\b",
            re.I,
        ),
    ),
}
LIVE = re.compile(
    r"\b(today|tonight|tomorrow|right now|currently|this (week|weekend|month)|forecast|at the moment)\b",
    re.I,
)


REGION_WORDS = {
    "central", "north", "northern", "south", "southern", "east", "eastern", "west", "western",
    "coast", "coastal", "region", "plain", "plateau", "delta", "highlands", "savannah",
    "upper", "lower", "interior", "sahel",
}  # fmt: skip


def is_region_guide(destination: Destination) -> bool:
    """A Wikivoyage guide to a region or country rather than to a single site."""
    return (
        destination.source == "wikivoyage"
        and destination.theme == "Town or Area Guide"
        and bool(set(destination.name_tokens) & (REGION_WORDS | COUNTRIES))
    )


def distance_km(a: Destination, b: Destination) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a.latitude, a.longitude, b.latitude, b.longitude))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371 * math.asin(math.sqrt(h))


def question_facet(message: str) -> str | None:
    for facet, (question, _) in FACETS.items():
        if question.search(message):
            return facet
    return None


def facet_coverage(destination: Destination, facet: str) -> int:
    return len(FACETS[facet][1].findall(destination.passage))


MAX_DESCRIPTION_CHARS = 5000  # keeps prompt plus answer within the model's context


def looks_like_description(message: str) -> bool:
    """True for a pasted description of a place: long, in sentences, not a question."""
    text = message.strip()
    sentences = [x for x in re.split(r"(?<=[.!])\s+", text) if len(x.split()) >= 4]
    return len(text.split()) >= 40 and len(sentences) >= 2 and not text.endswith("?")


def describe_custom(text: str) -> Destination:
    """Wrap a user-supplied description as a destination the model can answer from."""
    text = text.strip()[:MAX_DESCRIPTION_CHARS]
    first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    match = re.match(r"^(?:The )?(.{3,60}?)\s+(?:is|are|was|were|lies|sits|stands)\b", first)
    name = match.group(1).strip(" ,") if match else "this place"
    name_toks = tuple(tokens(name))
    return Destination(
        passage_id="custom",
        name=name,
        country="",
        source="user",
        url="",
        passage=text,
        name_tokens=name_toks,
        distinctive=frozenset(t for t in name_toks if t not in GENERIC) or frozenset(name_toks),
    )


def answer_style(message: str) -> str:
    """'one_line', 'classify', 'notable', 'digest', 'summary' or 'dialogue'."""
    for style, pattern in ANSWER_STYLES:
        if pattern.search(message):
            return style
    return "dialogue"


@dataclass
class Destination:
    passage_id: str
    name: str
    country: str
    source: str
    url: str
    passage: str
    theme: str = ""
    world_heritage: bool = False
    terrain: str = ""
    traveler_profile: str = ""
    latitude: float | None = None
    longitude: float | None = None
    name_tokens: tuple[str, ...] = ()
    distinctive: frozenset[str] = frozenset()

    @property
    def label(self) -> str:
        return f"{self.name} ({self.country})" if self.country else self.name

    @property
    def map_url(self) -> str | None:
        if self.latitude is None or self.longitude is None:
            return None
        lat, lon = self.latitude, self.longitude
        return f"https://www.openstreetmap.org/?mlat={lat:.5f}&mlon={lon:.5f}#map=13/{lat:.5f}/{lon:.5f}"

    @property
    def citation(self) -> str:
        if self.source == "user":
            return "your description"
        source = SOURCE_LABELS.get(self.source, self.source.title())
        return f"[{source}: {self.name}]({self.url})" if self.url else f"{source}: {self.name}"


@dataclass
class Route:
    """How to answer one message."""

    kind: str  # answer | fact | fact_missing | which_place | choose | recommend | describe_list
    #             introduce | ask_name | countries
    #             out_of_scope | live | greeting | thanks | reset | not_found
    destination: Destination | None = None
    choices: list[Destination] = field(default_factory=list)
    new_topic: bool = False
    facet: str | None = None  # for "fact" and "fact_missing": language or climate
    place: str = ""  # "fact_missing": the country or place asked; "introduce": the user's name
    describe: bool = False  # for lists: the user asked for a description of each item
    heading: str = ""  # for lists: what the list is ("waterfalls worth visiting in Togo")
    labels: list[str] = field(default_factory=list)  # for lists: each item's kind of place
    picked: bool = False  # the message only picked an item from the previous list


class PassageIndex:
    """Name lookup plus BM25 search over destination passages."""

    def __init__(self, destinations: list[Destination], k1: float = 1.5, b: float = 0.75):
        self.destinations = destinations
        self.k1, self.b = k1, b
        self._docs = [Counter(tokens(f"{d.name} {d.name} {d.passage}")) for d in destinations]
        self._lengths = [sum(c.values()) for c in self._docs]
        self._avg_len = sum(self._lengths) / max(1, len(self._lengths))
        df = Counter(t for doc in self._docs for t in doc)
        n = len(destinations)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> PassageIndex:
        destinations = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if ";" in str(r["country"]):
                    country = str(r["country"]).replace(";", " /")
                else:
                    country = str(r["country"])
                name_toks = tuple(tokens(r["name"]))
                # Words that identify the place; generic and common words never count alone.
                distinctive = frozenset(
                    t for t in name_toks if t not in GENERIC and t not in QUERY_WORDS
                ) or frozenset(name_toks)
                destinations.append(
                    Destination(
                        passage_id=r["passage_id"],
                        name=r["name"],
                        country=country,
                        source=str(r.get("source") or ""),
                        url=str(r.get("url") or ""),
                        passage=r["passage"],
                        theme=str(r.get("theme") or ""),
                        world_heritage=str(r.get("world_heritage")) == "True",
                        terrain=_label(r.get("terrain")),
                        traveler_profile=_label(r.get("traveler_profile")),
                        latitude=_coordinate(r.get("latitude")),
                        longitude=_coordinate(r.get("longitude")),
                        name_tokens=name_toks,
                        distinctive=distinctive,
                    )
                )
        return cls(destinations)

    def match_names(self, message: str) -> list[Destination]:
        """Destinations whose distinctive name words all appear in the message, best first."""
        msg = set(tokens(message))
        # Words written with a capital letter after the first word: deliberate names.
        capitalised = {
            t for w in re.findall(r"\S+", message)[1:] if w[:1].isupper() for t in tokens(w)
        }
        whole_message_is_name = len(msg) <= 3
        scored = []
        for d in self.destinations:
            if not d.distinctive or not d.distinctive <= msg:
                continue
            # Short names ("Oba", "Ada", "OH") collide with everyday words, so they count
            # only when capitalised mid-sentence or when the message is little more than the name.
            if (
                all(len(t) <= 3 for t in d.distinctive)
                and not whole_message_is_name
                and not d.distinctive <= capitalised
            ):
                continue
            matched = sum(t in msg for t in d.name_tokens)
            full = matched == len(d.name_tokens)
            scored.append(((matched, full, -len(d.name_tokens)), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return []
        best = scored[0][0]
        return [d for score, d in scored if score == best]

    def mentioned_places(self, message: str) -> list[Destination]:
        """Every distinct destination named in a message, in the order mentioned.

        Overlapping names keep the most specific ("Elmina Castle" over "Elmina").
        """
        msg = set(tokens(message))
        capitalised = {
            t for w in re.findall(r"\S+", message)[1:] if w[:1].isupper() for t in tokens(w)
        }
        found = []
        for d in self.destinations:
            if not d.distinctive or not d.distinctive <= msg:
                continue
            if all(len(t) <= 3 for t in d.distinctive) and not d.distinctive <= capitalised:
                continue
            found.append(d)
        matched = {d.passage_id: {t for t in d.name_tokens if t in msg} for d in found}
        specific = [
            d
            for d in found
            if not any(matched[d.passage_id] < matched[o.passage_id] for o in found if o is not d)
        ]
        best: dict[frozenset, Destination] = {}
        for d in specific:
            key = d.distinctive
            full = matched[d.passage_id] == set(d.name_tokens)
            current = best.get(key)
            if current is None or (
                full and matched[current.passage_id] != set(current.name_tokens)
            ):
                best[key] = d
        order = tokens(message)
        return sorted(best.values(), key=lambda d: min(order.index(t) for t in d.distinctive))

    def unknown_words(self, message: str) -> list[str]:
        """Content words that occur in no passage, e.g. places outside the corpus."""
        return [
            t
            for t in tokens(message)
            if len(t) >= 4 and t not in self._idf and t not in QUERY_WORDS and t not in GENERIC
        ]

    def search(self, message: str, k: int = 3) -> list[tuple[float, Destination]]:
        query = [t for t in tokens(message) if t in self._idf]
        if not query:
            return []
        scores = []
        for i, doc in enumerate(self._docs):
            norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / self._avg_len)
            s = sum(
                self._idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm) for t in query if t in doc
            )
            if s > 0:
                scores.append((s, self.destinations[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]


def proper_nouns(message: str) -> set[str]:
    """Capitalised words after the first, minus generic words: likely place names."""
    words = re.findall(r"[A-Za-zÀ-ÿ'][\w'-]*", message)
    found = set()
    for w in words[1:]:
        if w[0].isupper():
            found.update(t for t in tokens(w) if t not in GENERIC and t not in QUERY_WORDS)
    return found


def requested_themes(message: str) -> set[str]:
    msg = normalise(message)
    themes = {
        theme
        for theme, words in THEME_KEYWORDS.items()
        if any(re.search(rf"\b{re.escape(w)}\b", msg) for w in words)
    }
    for word, group in (("culture", "culture"), ("cultural", "culture")):
        if re.search(rf"\b{word}\b", msg):
            themes.update(THEME_GROUPS[group])
    if themes & {"Historic Site"}:
        themes.update(THEME_GROUPS["history"])
    return themes


ABBREVIATION = re.compile(r"\b(St|Ste|Mt|Ft|Dr|Mr|Mrs|Ms|No|Vol|ca|approx|e\.g|i\.e|c)\.$", re.I)


def first_sentence(text: str) -> str:
    """The first sentence, not fooled by abbreviations such as "Fort St. George"."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentence = ""
    for part in parts:
        sentence = f"{sentence} {part}".strip()
        if not ABBREVIATION.search(sentence):
            break
    return sentence


def preview(destination: Destination, max_words: int = 28) -> str:
    """A short, extractive description: the passage's first sentence without asides."""
    text = re.sub(r"\s*\[[^\]]*\]", "", destination.passage)  # […], which may sit inside (…)
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    words = first_sentence(text).split()
    return (
        " ".join(words)
        if len(words) <= max_words
        else " ".join(words[:max_words]).rstrip(",;:") + "…"
    )


def requested_subjects(message: str) -> set[str]:
    msg = set(tokens(message))
    return {subject for subject, words in SUBJECT_TRIGGERS.items() if msg & set(words)}


def mentions_subject(destination: Destination, subjects: set[str]) -> int:
    """How well a destination matches requested subjects ("waterfall", "ruins").

    2 if its name says so ("Kintampo Waterfalls"); 1 if its description does and its
    theme fits (a forest reserve whose text says "ancient" is not ruins); else 0.
    """
    name = set(destination.name_tokens)
    for s in subjects:
        # "park" or "reserve" in a name is not enough for wildlife: the theme must fit too.
        theme_ok = s not in NAME_NEEDS_THEME or destination.theme in SUBJECT_THEMES[s]
        if name & set(SUBJECTS[s]) and theme_ok:
            return 2
    passage = set(tokens(destination.passage[:1500]))
    for s in subjects:
        if passage & set(SUBJECTS[s]) and destination.theme in SUBJECT_THEMES.get(s, set()):
            return 1
    return 0


def is_person(destination: Destination) -> bool:
    return bool(PERSON.search(destination.passage[:300]))


ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
}  # fmt: skip
TELL_ABOUT = re.compile(
    r"\b(tell me (more )?about|more (on|about)|what about|go with|choose|pick)\b", re.I
)


def referenced_items(message: str, pending: list) -> list:
    """List items a message refers to, in the order mentioned.

    Recognises numbers ("1 and 2"), positions ("the first one", "the last"), part of
    a name ("Kano city walls") and a country when one item comes from it.
    """
    if not pending:
        return []
    text = normalise(message)
    words = tokens(message)
    found: list[tuple[int, object]] = []

    def add(position, item):
        if item not in [i for _, i in found]:
            found.append((position, item))

    for match in re.finditer(r"\b(\d{1,2})\b", text):
        n = int(match.group(1))
        if 1 <= n <= len(pending):
            add(match.start(), pending[n - 1])
    for match in re.finditer(r"\b(\w+)\b", text):
        word = match.group(1)
        if word in ORDINALS and ORDINALS[word] <= len(pending) and not word.isdigit():
            if word in ("one", "two", "three", "four", "five", "six") and not re.search(
                rf"\b(number|option|item|no\.?)\s+{word}\b", text
            ):
                continue  # "one" is usually not a list reference ("which one", "one day")
            add(match.start(), pending[ORDINALS[word] - 1])
        if word == "last":
            add(match.start(), pending[-1])
    msg = set(words)
    for item in pending:
        distinctive = {t for t in item.distinctive if len(t) > 3}
        if distinctive and len(distinctive & msg) >= max(1, len(distinctive) - 1):
            add(min(text.find(t) for t in distinctive & msg), item)
    for country in mentioned_countries(message):
        in_it = [i for i in pending if in_country(i, country)]
        if len(in_it) == 1:
            add(text.find(country.split()[0]), in_it[0])
    return [item for _, item in sorted(found, key=lambda x: x[0])]


def is_non_english(message: str) -> bool:
    words = tokens(message)
    hits = sum(w in NON_ENGLISH_WORDS for w in words)
    subdotted = bool(re.search(r"[ẹọṣ]", message.lower()))
    return bool(words) and (subdotted or (hits >= 2 and hits / len(words) >= 0.3))


def mentioned_countries(message: str) -> list[str]:
    found = []
    for match in COUNTRY_PATTERN.finditer(normalise(message)):
        name = COUNTRY_ALIASES.get(match.group(1), match.group(1))
        if name not in found:
            found.append(name)
    return found


def mentioned_country(message: str) -> str | None:
    msg = normalise(message)
    for name in (
        "burkina faso", "cape verde", "cabo verde", "cote d'ivoire", "ivory coast",
        "guinea-bissau", "guinea bissau", "sierra leone", "the gambia", "gambia", "benin",
        "ghana", "guinea", "liberia", "mali", "mauritania", "nigeria", "niger", "senegal", "togo",
    ):  # fmt: skip
        if re.search(rf"\b{re.escape(name)}\b", msg):
            return COUNTRY_ALIASES.get(name, name)
    return None


def in_country(destination: Destination, country: str) -> bool:
    """Exact country match; handles cross-border places listed under several countries."""
    parts = re.split(r"\s*[;/]\s*", normalise(destination.country))
    return normalise(country).removeprefix("the ") in {
        p.strip().removeprefix("the ") for p in parts
    }


COUNTRY_PATTERN = re.compile(
    r"\b(burkina faso|cape verde|cabo verde|cote d'ivoire|ivory coast|guinea-bissau|"
    r"guinea bissau|sierra leone|the gambia|gambia|benin|ghana|guinea|liberia|mali|"
    r"mauritania|nigeria|niger|senegal|togo)\b"
)
ABOUT_WORDS = {
    "tell", "more", "about", "what", "whats", "is", "like", "describe", "overview", "introduce",
    "introduction", "information", "info", "learn", "know", "the", "country", "me", "i", "want",
    "to", "a", "an", "of", "on", "please", "can", "you", "would",
}  # fmt: skip


def only_country_named(message: str) -> bool:
    """True when the message asks about a country and names nothing more specific."""
    rest = COUNTRY_PATTERN.sub(" ", normalise(message))
    return not [t for t in tokens(rest) if t not in ABOUT_WORDS]


COUNTRY_ALIASES = {
    "cabo verde": "cape verde",
    "ivory coast": "cote d'ivoire",
    "guinea bissau": "guinea-bissau",
    "the gambia": "gambia",
}


def load_country_guides(path: str | Path) -> dict[tuple[str, str], Destination]:
    """Country-level passages keyed by (normalised country, facet)."""
    guides = {}
    path = Path(path)
    if not path.exists():
        return guides
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            toks = tuple(tokens(r["name"]))
            guides[(normalise(r["country"]).removeprefix("the "), r["facet"])] = Destination(
                passage_id=r["passage_id"],
                name=r["name"],
                country=r["country"],
                source=r["source"],
                url=r["url"],
                passage=r["passage"],
                theme="Country Guide",
                name_tokens=toks,
                distinctive=frozenset(toks),
            )
    return guides


class Router:
    """Decides, per message, which passage (if any) the model should answer from."""

    def __init__(
        self,
        index: PassageIndex,
        guides: dict[tuple[str, str], Destination] | None = None,
        min_search_score: float = 6.0,
    ):
        self.index = index
        self.guides = guides or {}
        self.min_search_score = min_search_score
        self._word_counts_by_country: dict[str, Counter] = {}
        for d in index.destinations:
            key = normalise(d.country)
            self._word_counts_by_country.setdefault(key, Counter()).update(tokens(d.passage))
        self._places = [d for d in index.destinations if not is_person(d) and len(d.passage) >= 600]

    def recommend_variety(self, country: str | None, k: int = 6) -> tuple[list, list[str]]:
        """One notable place of each kind (ruins, wildlife, waterfall…), labelled."""
        picks, labels, countries = [], [], set()
        for label, themes, subjects in VARIETY:
            candidates = self.recommend(country, themes, subjects, k=12)
            fresh = [d for d in candidates if d not in picks]
            if not country:  # spread across countries when the whole region is in scope
                fresh = [d for d in fresh if d.country not in countries] or fresh
            if fresh:
                picks.append(fresh[0])
                labels.append(label)
                countries.add(fresh[0].country)
            if len(picks) == k:
                break
        return picks, labels

    def recommend(
        self, country: str | None, themes: set[str], subjects: set[str] = frozenset(), k: int = 5
    ) -> list[Destination]:
        """Notable destinations for a country, themes and/or specific subjects.

        Specific subjects ("waterfall", "ruins") must appear in the destination's
        name or description; World Heritage sites and richer descriptions rank first.
        """
        pool = self._places
        if country:
            pool = [d for d in pool if in_country(d, country)]
        if subjects:
            pool = [d for d in pool if mentions_subject(d, subjects)]
        elif themes:
            pool = [d for d in pool if d.theme in themes]
        elif not country:
            pool = [d for d in pool if d.theme != "Town or Area Guide"]

        def rank(d):
            return (
                mentions_subject(d, subjects) if subjects else 0,
                d.world_heritage,
                len(d.passage),
            )

        ranked = sorted(pool, key=rank, reverse=True)
        if country:
            return ranked[:k]
        picks, seen = [], set()  # one per country, for a spread across the region
        for d in ranked:
            if d.country not in seen:
                picks.append(d)
                seen.add(d.country)
            if len(picks) == k:
                break
        return picks

    def nearby(self, anchor: Destination, k: int = 6, radius_km: float = 80) -> list:
        """The closest places to a destination, with distances in km."""
        if anchor.latitude is None:
            return []
        found = []
        for d in self._places:
            if d is anchor or d.latitude is None:
                continue
            km = distance_km(anchor, d)
            if 0.5 <= km <= radius_km:
                found.append((km, d))
        found.sort(key=lambda x: x[0])
        return found[:k]

    def guide_search(self, facet: str, message: str) -> Destination | None:
        """A country guide on a facet that mentions the message's own words ("jollof", "Twi").

        Rarer words count for more; words that only frame the question ("say", "thank")
        are ignored; ties go to the country whose destinations mention the words most.
        """
        words = {
            t
            for t in tokens(message)
            if len(t) >= 3 and t not in QUERY_WORDS | GENERIC | PHRASE_WORDS
        }
        guides = [g for (_country, f), g in self.guides.items() if f == facet]
        counts = [Counter(tokens(g.passage)) for g in guides]
        doc_freq = {w: sum(w in c for c in counts) for w in words}
        scored = []
        for guide, count in zip(guides, counts, strict=True):
            score = sum(
                math.log(1 + len(guides) / doc_freq[w]) * min(count[w], 3)
                for w in words
                if count[w]
            )
            if score:
                # Tie-break: how often the country's destination passages use these words
                # (Twi is mentioned across Ghana, rarely elsewhere).
                in_corpus = sum(
                    self._word_counts_by_country.get(normalise(guide.country), Counter())[w]
                    for w in words
                )
                scored.append((score, in_corpus, guide))
        return max(scored, key=lambda x: (x[0], x[1]))[2] if scored else None

    def country_highlights(self, country: str, k: int = 3) -> list[Destination]:
        """A country's best-known places: World Heritage first, one per kind of place."""
        pool = [
            d
            for d in self._places
            if in_country(d, country) and d.theme not in ("Town or Area Guide", "")
        ]
        picks, themes = [], set()
        for d in sorted(pool, key=lambda d: (d.world_heritage, len(d.passage)), reverse=True):
            # Skip a second entry for the same place ("Royal Palaces of Abomey", "Abomey").
            if any(d.distinctive & p.distinctive for p in picks):
                continue
            if d.theme not in themes:
                picks.append(d)
                themes.add(d.theme)
            if len(picks) == k:
                break
        return picks

    def guide(self, country: str | None, facet: str) -> Destination | None:
        if not country:
            return None
        return self.guides.get((normalise(country).removeprefix("the "), facet))

    def best_for_facet(
        self, facet: str, country: str | None, place: Destination | None = None
    ) -> Destination | None:
        """The best passage for a practical question (language, climate).

        Uses the place itself if it covers the question; otherwise the country's
        Wikivoyage guide; otherwise the nearest passage that does (within 150 km);
        otherwise the country's regional travel guides.
        """
        is_guide = bool(place) and place.passage_id.startswith("country-")
        if place and not is_guide and facet_coverage(place, facet):
            return place
        if not country:
            return None
        if guide := self.guide(country, facet):
            return guide
        pool = [
            d
            for d in self.index.destinations
            if in_country(d, country) and facet_coverage(d, facet) and not is_person(d)
        ]
        if not pool:
            return None
        if place and place.latitude is not None:
            nearby = [
                (distance_km(place, d), d)
                for d in pool
                if d.latitude is not None and distance_km(place, d) <= 150
            ]
            if nearby:
                return min(nearby, key=lambda x: x[0])[1]
        return max(
            pool,
            key=lambda d: (is_region_guide(d), facet_coverage(d, facet), len(d.passage)),
        )

    def route(self, message: str, topic: Destination | None, pending: list[Destination]) -> Route:
        text = message.strip()

        if RESET.search(text):
            return Route("reset")

        if ASK_NAME.search(text):
            return Route("ask_name")
        intro = INTRODUCTION.match(text)
        if intro and (
            text.lower().startswith(("my name", "call me"))
            or not self.index.match_names(intro.group(1))
        ):
            name = " ".join(w.capitalize() for w in intro.group(1).split())
            return Route("introduce", place=name)

        describe = bool(DESCRIBE_EACH.search(text))
        refs = referenced_items(text, pending)
        if len(refs) >= 2 and COMPARE.search(text):
            return Route("compare", choices=refs[:2])
        if len(refs) >= 2 and DISTANCE.search(text):
            return Route("distance", choices=refs[:2])
        if refs and (describe or re.search(r"\bdescri\w*", text, re.I)):
            return Route("describe_list", choices=refs, describe=True)
        if len(refs) == 1 and TELL_ABOUT.search(text):
            return Route("answer", refs[0], new_topic=True, picked=True)
        if pending and describe and not (requested_themes(text) or requested_subjects(text)):
            return Route("describe_list", choices=pending, describe=True)

        # A reply to a "which one did you mean?" list: a number or a name.
        if pending:
            if text.isdigit() and 1 <= int(text) <= len(pending):
                return Route("answer", pending[int(text) - 1], new_topic=True, picked=True)
            # A name counts as a pick only when the message asks nothing else
            # ("What is near Elmina Castle?" is a nearby question, not a pick).
            asks_more = bool(
                re.search(
                    r"\b(what|where|how|which|who|when|why|is|are|does|do|can|could|should)\b",
                    text,
                    re.I,
                )
                or question_facet(text)
                or any(p.search(text) for p in (NEARBY, DISTANCE, COMPARE, PLAN, RECOMMEND))
                or OUT_OF_SCOPE.search(text)
                or requested_themes(text)
                or requested_subjects(text)
            )
            for d in pending:
                if not asks_more and d.distinctive <= set(tokens(text)):
                    return Route("answer", d, new_topic=True, picked=True)

        if len(tokens(text)) <= 4:
            if GREETING.match(text):
                return Route("greeting")
            if THANKS.search(text):
                return Route("thanks")

        if HELP.search(text):
            return Route("help")
        if COUNTRY_LIST.search(text) and not mentioned_countries(text):
            overviews = sorted(
                (g for (_c, f), g in self.guides.items() if f == "overview"), key=lambda g: g.name
            )
            if overviews:
                return Route(
                    "countries",
                    choices=overviews,
                    labels=[
                        ", ".join(
                            d.name + (" (UNESCO)" if d.world_heritage else "")
                            for d in self.country_highlights(g.country)
                        )
                        for g in overviews
                    ],
                )
        if is_non_english(text):
            return Route("non_english")
        if SMALL_TALK.search(text) and not self.index.mentioned_places(text):
            return Route("smalltalk")

        # Safety, health, visas, prices and bookings are declined before anything else,
        # even when the question names a place.
        if OUT_OF_SCOPE.search(text):
            return Route("out_of_scope")

        places = self.index.mentioned_places(text)
        countries = mentioned_countries(text)

        # "How far is Elmina from Cape Coast?" / "How do I get to Kakum?"
        if DISTANCE.search(text):
            # Pair with the current place only when the message refers to it ("from there").
            from_topic = bool(topic) and bool(REFERS_BACK.search(text)) and len(places) == 1
            pair = places[:2] if len(places) >= 2 else ([topic, *places] if from_topic else [])
            pair = [d for d in pair if d is not None]
            if len(pair) == 2 and pair[0] is not pair[1]:
                return Route("distance", choices=pair)
            target = places[0] if places else topic
            if target:
                return Route("transport", target)

        # "What is near Elmina Castle?" / "anything nearby?"
        if NEARBY.search(text):
            anchor = places[0] if places else topic
            if anchor:
                close = self.nearby(anchor)
                if close:
                    return Route(
                        "nearby",
                        anchor,
                        choices=[d for _, d in close],
                        labels=[f"{km:.0f} km away" for km, _ in close],
                        heading=f"Places near **{anchor.name}**:",
                        describe=describe,
                    )

        # "Which is better, Kakum or Mole?" / "Is Ghana better than Nigeria?"
        if COMPARE.search(text):
            if len(places) >= 2:
                return Route("compare", choices=places[:2])
            if len(countries) >= 2:
                guides = [self.guide(c, "overview") for c in countries[:2]]
                if all(guides):
                    return Route("compare", choices=guides)
            if topic and len(places) == 1 and places[0] is not topic:
                return Route("compare", choices=[topic, places[0]])

        facet = question_facet(text)
        if facet:
            if facet == "climate" and LIVE.search(text):
                return Route("live")
            names = self.index.match_names(text)
            named = names[0] if len(names) == 1 else None
            country = mentioned_country(text)
            place = named or (topic if not country else None)
            if not country and place and place.country:
                country = normalise(place.country)
            if not country and not place:
                if found := self.guide_search(facet, text):
                    return Route("fact", found, facet=facet)
                return Route("which_place", facet=facet)
            best = self.best_for_facet(facet, country, place)
            if best is None:
                where = (place.name if place else None) or (country or "").title()
                return Route("fact_missing", facet=facet, place=where)
            if topic and best == topic:
                return Route("answer", topic)
            return Route("fact", best, facet=facet)

        # "Where is Liberia?", "How big is Mali?": general questions about one country.
        if (
            len(countries) == 1
            and not places
            and (COUNTRY_FACT.search(text) or ABOUT_COUNTRY.search(text))
            and not (requested_themes(text) or requested_subjects(text))
            and not any(p.search(text) for p in (RECOMMEND, PLAN, BUDGET, ROMANTIC))
            and (overview := self.guide(countries[0], "overview"))
        ):
            return Route("answer", overview, new_topic=True)

        if topic and REFERS_BACK.search(text):
            return Route("answer", topic)  # "it", "there": a follow-up on the current place

        country = mentioned_country(text)
        where = f"in {country_title(country)}" if country else "across West Africa"
        if PLAN.search(text):
            picks, labels = self.recommend_variety(country)
            return Route(
                "plan",
                choices=picks,
                labels=labels,
                describe=describe,
                heading=(
                    "I can't build day-by-day itineraries or estimate travel times, but here are "
                    f"highlights {where} to plan around:"
                ),
            )
        if ROMANTIC.search(text):
            picks = self.recommend(country, set(), {"beach", "island"})
            if picks:
                return Route(
                    "recommend",
                    choices=picks,
                    describe=describe,
                    heading=(
                        "I can't judge what's romantic, but beaches and islands are popular "
                        f"for getaways {where}:"
                    ),
                )
        if BUDGET.search(text):
            picks, labels = self.recommend_variety(country)
            return Route(
                "recommend",
                choices=picks,
                labels=labels,
                describe=describe,
                heading=f"I don't have prices, but here's a mix of places worth visiting {where}:",
            )
        overview = self.guide(country, "overview")
        if overview and ABOUT_COUNTRY.search(text) and only_country_named(text):
            return Route("answer", overview, new_topic=True)

        names = self.index.match_names(text)
        if names:
            passage_words = set(tokens(topic.passage)) if topic else set()
            if topic and (topic in names or any(d.distinctive <= passage_words for d in names)):
                return Route("answer", topic)  # follow-up that mentions a nearby place
            if len(names) == 1:
                return Route("answer", names[0], new_topic=True)
            return Route("choose", choices=names[:5])

        country = mentioned_country(text)
        named = proper_nouns(text)
        topic_words = set(tokens(f"{topic.name} {topic.passage}")) if topic else set()
        new_country = bool(country and topic and not in_country(topic, country))
        new_place = bool(named - topic_words)

        themes = requested_themes(text)
        subjects = requested_subjects(text)
        overview = self.guide(country, "overview")
        if (
            overview
            and not (themes or subjects or new_place)
            and not RECOMMEND.search(text)
            and ABOUT_COUNTRY.search(text)
        ):
            return Route("answer", overview, new_topic=True)
        # A request for a kind of place replaces the current destination unless that
        # destination already matches it ("is there a waterfall?" about a falls).
        topic_matches = bool(topic) and (
            (subjects and mentions_subject(topic, subjects) == 2)
            or (not subjects and themes and topic.theme in themes)
        )
        wants_new_kind = bool(themes or subjects) and not topic_matches
        if RECOMMEND.search(text) or (
            wants_new_kind
            and not new_place
            and (not topic or new_country or CHANGE.search(text) or not REFERS_BACK.search(text))
        ):
            if not country and topic and not (themes or subjects):
                country = normalise(topic.country)  # "anything else interesting?" near here
            heading = list_heading(country, themes, subjects)
            if themes or subjects:
                picks = self.recommend(country, themes, subjects)
                labels = []  # the heading already says what kind of place these are
            else:
                picks, labels = self.recommend_variety(country)
            if picks:
                return Route(
                    "recommend", choices=picks, describe=describe, heading=heading, labels=labels
                )

        if topic and not new_country and not new_place:
            return Route("answer", topic)

        if self.index.unknown_words(text):
            return Route("not_found")
        if not [t for t in tokens(text) if t not in QUERY_WORDS and len(t) >= 3]:
            return Route("not_found")  # small talk with nothing to search for

        hits = [d for s, d in self.index.search(text, k=20) if s >= self.min_search_score]
        if named:
            # The user named a specific place: only accept destinations bearing that name.
            hits = [d for d in hits if named & set(d.name_tokens)]
        if country:
            hits = [d for d in hits if in_country(d, country)] or hits
        hits = hits[:3]
        if len(hits) == 1:
            return Route("answer", hits[0], new_topic=True)
        if hits:
            return Route("choose", choices=hits, describe=describe)
        return Route("not_found")
