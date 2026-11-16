"""Portfolio Intelligence Assistant engine.

A deterministic, retrieval-based question answerer. It does NOT call an LLM and
has no network access: every sentence in an answer is assembled from values in
``data/portfolio.json``. If the portfolio does not contain something (for
example work experience), the assistant says so instead of guessing.

This module depends only on the Python standard library so it can be tested
without installing a web framework.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "portfolio.json"

MAX_LOOKUP_RESULTS = 6
MAX_TOPIC_SECTIONS = 3


def load_portfolio(path: Path | str | None = None) -> dict[str, Any]:
    """Load the portfolio JSON (the single source of truth)."""
    target = Path(path) if path else DATA_PATH
    with target.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


_ALIASES = {
    "statistical": "statistic", "algorithmic": "algorithm",
    "optimization": "optimizer", "optimisation": "optimizer", "optimiser": "optimizer",
    "prediction": "predictive", "predicting": "predictive", "predict": "predictive",
    "modelling": "modeling", "analytic": "analysi", "analyze": "analysi", "analyse": "analysi",
}


def _stem(word: str) -> str:
    """Very light plural stemming + a few aliases, applied identically to data and questions."""
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return _ALIASES.get(word, word)


def _analyze(text: str) -> list[tuple[str, str]]:
    """Return [(original_word, stem)] for every word in ``text``."""
    text = text.lower().translate(_SUBSCRIPTS).replace("&", " and ")
    text = re.sub(r"linkedin\s+learning", "certification", text)
    text = re.sub(r"[.'’]", "", text)  # b.tech -> btech, pm2.5 -> pm25
    words = re.findall(r"[a-z]+(?:\+\+|#)?|[0-9]+", text)  # html5 -> html, 5 ; c++ stays whole
    return [(w, _stem(w)) for w in words]


def _expand(stems: list[str]) -> list[str]:
    """Add short forms so 'machine learning' also matches 'ML' and vice versa."""
    out = list(stems)
    if _contains_sequence(stems, ["machine", "learning"]):
        out.append("ml")
    if _contains_sequence(stems, ["artificial", "intelligence"]):
        out.append("ai")
    return out


def _stems(text: str) -> list[str]:
    return [stem for _, stem in _analyze(text)]


def _first_sentence(text: str) -> str:
    match = re.match(r"(.+?[.!?])(\s|$)", text.strip())
    return match.group(1) if match else text.strip()


def _contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def _join_natural(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

_STOP_RAW = """
a an the is are was were be been am do does did can could would should will
what which who whom whose when where why how of in on at to for from with about
and or but not no yes any some all her his she he they them their it its this
that these those me my you your i we our tell show give list please also more
there here has have had having use used using uses know knows got get sri
nithya nitya portfolio site website section sections s t
like such etc eg ie just really very much many other another good great strong best
expert familiar able skilled capable decent completed finished done taken earned
currently now today recently exactly actually kind sort thing things detail details
info information way complete finish do long often
"""

TRIGGERS: dict[str, set[str]] = {
    "about": {
        "introduce", "introduction", "overview", "summary", "bio",
        "interest", "passion", "hobby", "background", "yourself",
        "motto", "tagline", "value", "philosophy",
    },
    "education": {
        "education", "university", "college", "degree", "btech", "study",
        "studying", "studied", "jain", "major", "specialization",
        "specialisation", "graduate", "graduation", "academic", "curriculum",
        "school", "qualification", "coursework",
    },
    "skills": {
        "skill", "technology", "stack", "tool", "language", "programming",
        "competency", "expertise", "proficient", "toolkit", "technical",
        "mastering",
    },
    "projects": {
        "project", "built", "build", "made", "created", "create", "developed",
        "app", "application", "featured", "prototype",
    },
    "certifications": {
        "certification", "certificate", "course", "credential", "certified",
    },
    "achievements": {
        "achievement", "award", "honor", "honour", "accomplishment",
        "recognition", "accolade", "prize", "scholarship", "publication",
    },
    "experience": {
        "experience", "employment", "job", "career", "professional",
        "employer", "worked", "history",
    },
    "availability": {
        "available", "availability", "hire", "hiring", "open", "seeking",
        "looking", "opportunity", "internship", "status", "collaborate",
        "collaboration", "mentorship",
    },
    "contact": {
        "contact", "email", "mail", "reach", "github", "linkedin", "phone",
        "connect", "message", "address", "touch", "social", "profile", "repo",
        "repository",
    },
    "location": {
        "where", "located", "location", "live", "based", "city", "bengaluru",
        "bangalore", "karnataka", "india",
    },
    "resume": {"resume", "cv", "curriculum"},
}

TOPIC_PRIORITY = [
    "projects", "certifications", "skills", "education", "experience",
    "achievements", "availability", "contact", "resume", "location", "about",
]

# Topics whose answers are better served by a targeted lookup when the
# question also contains a specific term (e.g. "projects that use Python").
LOOKUP_TOPICS = {"skills", "projects", "certifications", "education", "about"}

_STOP = {_stem(w) for w in _STOP_RAW.split()}
_TRIGGER_STEMS = {t: {_stem(w) for w in words} for t, words in TRIGGERS.items()}
_ALL_TRIGGER_STEMS = set().union(*_TRIGGER_STEMS.values())
_GREETINGS = {"hi", "hello", "hey", "hola", "namaste", "yo", "help", "start"}

SUGGESTIONS_DEFAULT = [
    "What projects has Sri Nithya built?",
    "What are the technical skills?",
    "Tell me about the education",
    "Which certifications are listed?",
    "How can I get in touch?",
]


class PortfolioAssistant:
    """Answers natural-language questions using only the supplied portfolio."""

    def __init__(self, portfolio: dict[str, Any]) -> None:
        self.p = portfolio
        self.name: str = portfolio["profile"]["name"]
        self._facts = self._build_facts()
        self._df: dict[str, int] = {}
        for fact in self._facts:
            for tok in fact["tokens"]:
                self._df[tok] = self._df.get(tok, 0) + 1

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ask(self, question: str) -> dict[str, Any]:
        analyzed = _analyze(question)
        stems = [s for _, s in analyzed]
        extra = [t for t in _expand(stems)[len(stems):]]  # e.g. "ml" for "machine learning"

        if not stems or self._is_greeting(stems):
            return self._help_response()

        project_hits = self._match_projects(stems)
        topic_scores = self._score_topics(stems)
        topics = [t for t in TOPIC_PRIORITY if topic_scores.get(t, 0) > 0]
        topics.sort(key=lambda t: -topic_scores[t])  # stable: keeps priority on ties

        content = [
            (orig, stem)
            for orig, stem in analyzed
            if stem not in _STOP
            and stem not in _ALL_TRIGGER_STEMS
            and (len(stem) > 1 or stem in {"c", "r"})
        ]
        content += [(s, s) for s in extra]
        known = [(o, s) for o, s in content if s in self._df]
        unknown = [o for o, s in content if s not in self._df]

        sections: list[str] = []
        sources: list[str] = []
        used_topics: list[str] = []
        notes: list[str] = []

        # 1) A specific project was named -> detailed project answer.
        if project_hits:
            for proj in project_hits[:2]:
                sections.append(self._project_detail(proj))
                sources.append(f"Projects › {proj['name']}")
            used_topics.append("projects")
            return self._finish(sections, sources, used_topics, notes)

        # 2) A specific term (Python, graph theory...) -> targeted lookup.
        # Only trust a lookup when most of the question's content words were
        # actually found; one incidental word match (e.g. "world" in
        # "real-world software") must not produce a confident-looking answer.
        coverage = len(known) / len(content) if content else 0.0
        strong_known = any(s in self._strong for _, s in known)
        trusted = bool(known) and (coverage > 0.5 or (coverage >= 0.5 and strong_known))
        hits = self._retrieve([s for _, s in known]) if trusted else []
        lookup_applies = bool(hits) and all(t in LOOKUP_TOPICS for t in topics)
        if lookup_applies:
            text, srcs = self._lookup_answer(hits, topics)
            sections.append(text)
            sources.extend(srcs)
            used_topics.append("search")
            if unknown:
                notes.append(self._missing_note(unknown, lead_in=False))
            return self._finish(sections, sources, used_topics, notes)

        # 3) Topic answers.
        if topics:
            if unknown:
                notes.append(self._missing_note(unknown))
            for topic in topics[:MAX_TOPIC_SECTIONS]:
                text, srcs = self._topic_answer(topic)
                sections.append(text)
                sources.extend(srcs)
                used_topics.append(topic)
            if hits:  # e.g. "does she have Python experience?"
                text, srcs = self._lookup_answer(hits, [])
                sections.append(text)
                sources.extend(srcs)
            return self._finish(sections, sources, used_topics, notes)

        # 4) Nothing relevant: be honest rather than guess.
        return self._fallback_response()

    # ------------------------------------------------------------------ #
    # Topic detection / retrieval
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_greeting(stems: list[str]) -> bool:
        if all(s in _GREETINGS or s in _STOP for s in stems) and any(s in _GREETINGS for s in stems):
            return True
        return _contains_sequence(stems, ["what", "can", "you", "do"])

    @staticmethod
    def _score_topics(stems: list[str]) -> dict[str, int]:
        present = set(stems)
        scores = {t: len(present & trig) for t, trig in _TRIGGER_STEMS.items() if present & trig}
        refers_to_person = bool(present & {"nithya", "nitya", "sri", "she", "her"})
        if refers_to_person and present & {"who", "name"}:
            scores["about"] = scores.get("about", 0) + 1
        return scores

    def _match_projects(self, stems: list[str]) -> list[dict[str, Any]]:
        matched = []
        for proj in self.p["projects"]:
            phrases = [proj["name"]] + proj.get("match_terms", [])
            if any(_contains_sequence(stems, _stems(ph)) for ph in phrases if _stems(ph)):
                matched.append(proj)
        return matched

    def _build_facts(self) -> list[dict[str, Any]]:
        p = self.p
        facts: list[dict[str, Any]] = []

        self._strong: set[str] = set()

        def add(kind: str, label: str, text: str, display: str, curated: list[str] | None = None) -> None:
            facts.append({"kind": kind, "label": label, "display": display, "tokens": set(_expand(_stems(text)))})
            for term in curated or []:
                self._strong.update(_expand(_stems(term)))

        for s in p["skills"]:
            add(
                "skills",
                f"Skills › {s['category']}",
                " ".join([s["category"], *s["items"], s["description"]]),
                f"{', '.join(s['items'])}",
                [s["category"], *s["items"]],
            )
        for proj in p["projects"]:
            add(
                "projects",
                f"Projects › {proj['name']}",
                " ".join([proj["name"], proj["category"], proj["kind"], proj["description"], *proj["tags"]]),
                f"{proj['kind']}. {_first_sentence(proj['description'])}",
                [proj["name"], proj["category"], *proj["tags"]],
            )
        for c in p["certifications"]:
            add(
                "certifications",
                f"Certifications › {c['name']}",
                " ".join([c["name"], c["provider"], c["date"], c["duration"], *c["skills"]]),
                f"{c['provider']}, completed {c['date']}, {c['duration']} (skills: {', '.join(c['skills'])})",
                [c["name"], c["provider"], c["date"], *c["skills"]],
            )
        edu = p["education"]
        add(
            "education",
            "Education",
            " ".join([edu["degree"], edu["major"], edu["institution"], *edu["focus_areas"]]),
            f"{edu['degree']} in {edu['major']}; focus areas include {', '.join(edu['focus_areas'])}",
            [edu["degree"], edu["major"], *edu["focus_areas"]],
        )
        prof = p["profile"]
        add(
            "about",
            "About",
            " ".join([prof["summary"], prof["current_focus"], *prof["core_interests"]]),
            f"Core interests: {', '.join(prof['core_interests'])}",
            list(prof["core_interests"]),
        )
        return facts

    def _retrieve(self, stems: list[str]) -> list[dict[str, Any]]:
        n = max(len(self._facts), 1)
        scored = []
        for fact in self._facts:
            score = sum(1 + math.log(n / self._df[s]) for s in set(stems) if s in fact["tokens"])
            if score > 0:
                scored.append((score, fact))
        if not scored:
            return []
        scored.sort(key=lambda x: -x[0])
        best = scored[0][0]
        return [f for sc, f in scored if sc >= 0.6 * best][:MAX_LOOKUP_RESULTS]

    # ------------------------------------------------------------------ #
    # Answer builders
    # ------------------------------------------------------------------ #
    def _finish(self, sections: list[str], sources: list[str], topics: list[str], notes: list[str]) -> dict[str, Any]:
        text = "\n\n".join([*notes, *sections])
        seen: set[str] = set()
        uniq_sources = [s for s in sources if not (s in seen or seen.add(s))]
        return {
            "answer": text,
            "matched": True,
            "topics": topics,
            "sources": uniq_sources,
            "suggestions": self._suggestions_after(topics),
        }

    def _help_response(self) -> dict[str, Any]:
        text = (
            f"Hi! I'm the Portfolio Assistant. I answer questions about {self.name}'s projects, "
            "education, technical skills, certifications and contact details — using only what is "
            "published on this portfolio.\n\nTry one of the questions below."
        )
        return {"answer": text, "matched": True, "topics": ["help"], "sources": [], "suggestions": SUGGESTIONS_DEFAULT}

    def _fallback_response(self) -> dict[str, Any]:
        text = (
            f"I couldn't find anything about that in {self.name}'s portfolio. I can only answer from "
            "what's published here: projects, education, technical skills, certifications, "
            "availability and contact details."
        )
        return {"answer": text, "matched": False, "topics": [], "sources": [], "suggestions": SUGGESTIONS_DEFAULT}

    @staticmethod
    def _missing_note(unknown: list[str], lead_in: bool = True) -> str:
        shown = ", ".join(f"“{w}”" for w in unknown[:3])
        if lead_in:
            return f"I couldn't find {shown} in the portfolio, so I won't guess. Here is what is listed:"
        return f"I couldn't find {shown} in the portfolio, so I won't guess about that part."

    def _lookup_answer(self, hits: list[dict[str, Any]], preferred: list[str]) -> tuple[str, list[str]]:
        pref = {t for t in preferred if t in {"skills", "projects", "certifications", "education"}}
        chosen = [h for h in hits if h["kind"] in pref] if pref else hits
        lines: list[str] = []
        if pref and not chosen:
            lines.append(f"Nothing in the {_join_natural(sorted(pref))} sections matches that, but it appears elsewhere:")
            chosen = hits
        elif pref:
            lines.append("Here's what the portfolio lists:")
        else:
            lines.append("Here's where that appears in the portfolio:")
        for h in chosen:
            lines.append(f"• {h['label']}: {h['display']}")
        return "\n".join(lines), [h["label"] for h in chosen]

    def _project_detail(self, proj: dict[str, Any]) -> str:
        lines = [
            f"{proj['name']} — {proj['category']} · {proj['kind']}",
            proj["description"],
            f"Tags: {', '.join(proj['tags'])}.",
        ]
        fm = proj.get("forecast_metric")
        if fm:
            lines.append(
                "The portfolio also shows a Crowd Forecast Metric for it: "
                f"peak congestion {fm['peak_congestion']}, estimated wait {fm['estimated_wait']}, "
                f"recommended window {fm['recommended_window']}."
            )
        return "\n".join(lines)

    def _topic_answer(self, topic: str) -> tuple[str, list[str]]:
        return getattr(self, f"_topic_{topic}")()

    def _topic_about(self) -> tuple[str, list[str]]:
        pr = self.p["profile"]
        vals = "; ".join(f"{v['title']} — {v['text']}" for v in pr["values"])
        text = (
            f"{pr['name']} is {pr['summary']}\n\n"
            f"Core interests: {', '.join(pr['core_interests'])}.\n"
            f"Current focus: {pr['current_focus']}\n"
            f"Guiding values: {vals}"
        )
        return text, ["About"]

    def _topic_education(self) -> tuple[str, list[str]]:
        e = self.p["education"]
        text = (
            f"{self.name} is pursuing a {e['degree']} in {e['major']} at {e['institution']} "
            f"(listed as “{e['status']}”).\n"
            f"Core focus areas: {', '.join(e['focus_areas'])}.\n"
            "The portfolio does not list a graduation year or grades."
        )
        return text, ["Education"]

    def _topic_skills(self) -> tuple[str, list[str]]:
        lines = ["Technical skills listed in the portfolio:"]
        for s in self.p["skills"]:
            lines.append(f"• {s['category']}: {', '.join(s['items'])}")
        return "\n".join(lines), [f"Skills › {s['category']}" for s in self.p["skills"]]

    def _topic_projects(self) -> tuple[str, list[str]]:
        projs = self.p["projects"]
        badge = self.p["profile"]["hero_badge"]
        lines = [f"The Projects section features {len(projs)} projects (the hero badge reads “{badge}”):"]
        for pr in projs:
            lines.append(f"• {pr['name']} ({pr['kind']}) — {_first_sentence(pr['description'])}")
        lines.append("Ask about any of them by name for details.")
        return "\n".join(lines), [f"Projects › {pr['name']}" for pr in projs]

    def _topic_certifications(self) -> tuple[str, list[str]]:
        certs = self.p["certifications"]
        lines = [f"The portfolio lists {len(certs)} completed courses/certificates:"]
        for c in certs:
            lines.append(
                f"• {c['name']} — {c['provider']}, completed {c['date']} ({c['duration']}). "
                f"Skills: {', '.join(c['skills'])}."
            )
        return "\n".join(lines), [f"Certifications › {c['name']}" for c in certs]

    def _topic_achievements(self) -> tuple[str, list[str]]:
        certs = self.p["certifications"]
        text = (
            "The portfolio doesn't have a dedicated achievements or awards section, and I don't want "
            "to invent any. The closest items it does list are certifications: "
            + _join_natural([c["name"] for c in certs])
            + "."
        )
        return text, ["Certifications"]

    def _topic_experience(self) -> tuple[str, list[str]]:
        pr = self.p["profile"]
        if self.p.get("experience"):
            lines = ["Experience listed in the portfolio:"]
            for item in self.p["experience"]:
                lines.append(f"• {item}")
            return "\n".join(lines), ["Experience"]
        text = (
            "The portfolio doesn't list any work or internship experience yet. "
            f"It describes the current status as “{pr['seeking']}” and carries the badge “{pr['status']}”. "
            f"What it does show: {len(self.p['projects'])} projects and "
            f"{len(self.p['certifications'])} completed certifications."
        )
        return text, ["About", "Projects"]

    def _topic_availability(self) -> tuple[str, list[str]]:
        pr, c = self.p["profile"], self.p["contact"]
        text = (
            f"Status: {pr['status']}. {pr['seeking']}.\n"
            f"The portfolio says {self.name} is open to {c['open_to']}.\n"
            f"No work experience is listed yet. Email: {c['email']}"
        )
        return text, ["About", "Contact"]

    def _topic_contact(self) -> tuple[str, list[str]]:
        c = self.p["contact"]
        lines = [
            f"Email: {c['email']}",
            f"GitHub: {c['github']}",
            f"LinkedIn: {c['linkedin']}",
        ]
        if c.get("has_contact_form"):
            lines.append("The site also has a “Send a Direct Note” contact form.")
        lines.append("No phone number is listed in the portfolio.")
        return "\n".join(lines), ["Contact"]

    def _topic_location(self) -> tuple[str, list[str]]:
        pr = self.p["profile"]
        text = f"{self.name} is based in {pr['location']} and studies at {pr['university']}, Bengaluru."
        return text, ["About", "Education"]

    def _topic_resume(self) -> tuple[str, list[str]]:
        c = self.p["contact"]
        text = (
            "I don't see a résumé or CV on the portfolio, so I can't point to one. "
            f"You can ask for it directly by email at {c['email']} or via LinkedIn ({c['linkedin']})."
        )
        return text, ["Contact"]

    # ------------------------------------------------------------------ #
    def _suggestions_after(self, topics: list[str]) -> list[str]:
        mapping = {
            "projects": ["Tell me about the AQI project", "Which projects use Python?", "How can I get in touch?"],
            "search": ["What projects has Sri Nithya built?", "What are the technical skills?"],
            "skills": ["Which projects use Python?", "Tell me about graph theory work"],
            "education": ["What are the technical skills?", "Which certifications are listed?"],
            "certifications": ["Tell me about the education", "What projects has Sri Nithya built?"],
            "contact": ["Is Sri Nithya open to internships?", "What projects has Sri Nithya built?"],
            "availability": ["What projects has Sri Nithya built?", "How can I get in touch?"],
            "experience": ["Is Sri Nithya open to internships?", "What projects has Sri Nithya built?"],
            "achievements": ["Which certifications are listed?", "What projects has Sri Nithya built?"],
        }
        for t in topics:
            if t in mapping:
                return mapping[t]
        return SUGGESTIONS_DEFAULT[:3]
