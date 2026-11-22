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
    "nigerian", "ghanaian", "senegalese", "malian", "beninese", "togolese",
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
    r"border|exchange rate|currency|budget|flight\w*|airline\w*|book|booking|hotel\w*)\b",
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

# Specific things a traveller asks to see. A recommendation must mention one of
# these words in the destination's name or description, not merely share a theme.
SUBJECTS = {
    "waterfall": ("waterfall", "waterfalls", "falls", "cascade", "cascades", "chutes"),
    "ruins": ("ruins", "ruin", "archaeological", "excavation", "excavations", "ancient"),
    "beach": ("beach", "beaches", "praia", "plage"),
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
}  # fmt: skip

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
    r"businesswoman|businessperson|slaver|historian|businessman|ruler|queen|king)\b"
    # Lifespan right after the name, then "was": "X (c. 1615 – after 1670) was ..."
    r"|^[^.(]{0,80}\((c\.\s*)?\d{3,4}\s*[–-][^)]*\),?\s+(was|is)\b"
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
        re.compile(r"\b(language|languages|speak|spoken|dialect|dialects|lingua franca)\b", re.I),
        re.compile(r"\b(language|languages|spoken|speak|lingua franca|dialect)\b", re.I),
    ),
    "climate": (
        re.compile(
            r"\b(weather|climate|rainy|rains?|dry season|wet season|seasons?|harmattan|temperatures?|"
            r"humid|humidity|best time (to|for) (visit|go|travel)|when (should|to) (i )?(visit|go))\b",
            re.I,
        ),
        re.compile(
            r"\b(climate|rainy season|dry season|wet season|harmattan|rainfall|temperatures?|"
            r"humid|humidity|best time to visit|rains)\b",
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

    kind: str  # answer | fact | fact_missing | which_place | choose | recommend
    #             out_of_scope | live | greeting | thanks | reset | not_found
    destination: Destination | None = None
    choices: list[Destination] = field(default_factory=list)
    new_topic: bool = False
    facet: str | None = None  # for "fact" and "fact_missing": language or climate
    place: str = ""  # for "fact_missing": the country or place asked about


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


def requested_subjects(message: str) -> set[str]:
    msg = set(tokens(message))
    return {subject for subject, words in SUBJECT_TRIGGERS.items() if msg & set(words)}


def mentions_subject(destination: Destination, subjects: set[str]) -> int:
    """2 if the name mentions a requested subject, 1 if only the description does, else 0."""
    words = {w for s in subjects for w in SUBJECTS[s]}
    if set(destination.name_tokens) & words:
        return 2
    return 1 if set(tokens(destination.passage[:1500])) & words else 0


def is_person(destination: Destination) -> bool:
    return bool(PERSON.search(destination.passage[:300]))


def mentioned_country(message: str) -> str | None:
    msg = normalise(message)
    for name in (
        "burkina faso", "cape verde", "cabo verde", "cote d'ivoire", "ivory coast",
        "guinea-bissau", "guinea bissau", "sierra leone", "the gambia", "gambia", "benin",
        "ghana", "guinea", "liberia", "mali", "mauritania", "nigeria", "niger", "senegal", "togo",
    ):  # fmt: skip
        if re.search(rf"\b{re.escape(name)}\b", msg):
            return name
    return None


class Router:
    """Decides, per message, which passage (if any) the model should answer from."""

    def __init__(self, index: PassageIndex, min_search_score: float = 6.0):
        self.index = index
        self.min_search_score = min_search_score
        self._places = [d for d in index.destinations if not is_person(d) and len(d.passage) >= 600]

    def recommend(
        self, country: str | None, themes: set[str], subjects: set[str] = frozenset(), k: int = 5
    ) -> list[Destination]:
        """Notable destinations for a country, themes and/or specific subjects.

        Specific subjects ("waterfall", "ruins") must appear in the destination's
        name or description; World Heritage sites and richer descriptions rank first.
        """
        pool = self._places
        if country:
            pool = [d for d in pool if country.split()[0] in normalise(d.country)]
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

    def best_for_facet(
        self, facet: str, country: str | None, place: Destination | None = None
    ) -> Destination | None:
        """The best passage for a practical question (language, climate).

        Uses the place itself if it covers the question; otherwise the nearest
        passage that does (within 150 km); otherwise the country's regional travel
        guides, which describe language and climate for a whole region.
        """
        if place and facet_coverage(place, facet):
            return place
        if not country:
            return None
        pool = [
            d
            for d in self.index.destinations
            if country.split()[0] in normalise(d.country)
            and facet_coverage(d, facet)
            and not is_person(d)
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

        # A reply to a "which one did you mean?" list: a number or a name.
        if pending:
            if text.isdigit() and 1 <= int(text) <= len(pending):
                return Route("answer", pending[int(text) - 1], new_topic=True)
            for d in pending:
                if d.distinctive <= set(tokens(text)):
                    return Route("answer", d, new_topic=True)

        if len(tokens(text)) <= 4:
            if GREETING.match(text):
                return Route("greeting")
            if THANKS.search(text):
                return Route("thanks")

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
                return Route("which_place", facet=facet)
            best = self.best_for_facet(facet, country, place)
            if best is None:
                where = (place.name if place else None) or (country or "").title()
                return Route("fact_missing", facet=facet, place=where)
            if topic and best == topic:
                return Route("answer", topic)
            return Route("fact", best, facet=facet)

        if topic and REFERS_BACK.search(text):
            return Route("answer", topic)  # "it", "there": a follow-up on the current place

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
        new_country = bool(country and topic and country not in normalise(topic.country))
        new_place = bool(named - topic_words)
        if OUT_OF_SCOPE.search(text):
            return Route("out_of_scope")

        themes = requested_themes(text)
        subjects = requested_subjects(text)
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
            picks = self.recommend(country, themes, subjects)
            if picks:
                return Route("recommend", choices=picks)

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
            hits = [d for d in hits if country.split()[0] in normalise(d.country)] or hits
        hits = hits[:3]
        if len(hits) == 1:
            return Route("answer", hits[0], new_topic=True)
        if hits:
            return Route("choose", choices=hits)
        return Route("not_found")
