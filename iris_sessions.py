"""
iris_sessions.py — chat session persistence for the IRIS sidebar.

Each chat session is saved as a small JSON file so the sidebar can show past
conversations grouped by recency (Today / Yesterday / This Week / Earlier),
and clicking one reloads its messages. Pure Python, no Qt. Never raises out;
all disk errors degrade to in-memory behavior.
"""

from __future__ import annotations

import os
import re
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


def _default_dir() -> str:
    """A writable per-user location for session files."""
    base = (os.environ.get("APPDATA")
            or os.path.join(os.path.expanduser("~"), ".iris"))
    path = os.path.join(base, "iris", "sessions")
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        alt = os.path.join(os.getcwd(), "sessions")
        try:
            os.makedirs(alt, exist_ok=True)
        except Exception:
            pass
        return alt


@dataclass
class Session:
    id: str
    created: float
    updated: float
    title: str = "new session"
    messages: list = field(default_factory=list)   # [{role, content, ts}]
    # Optional context used for the rich sidebar label "location · person · HH:MM".
    location: str = ""                              # venue / city, if known
    people: list = field(default_factory=list)      # person names, if known
    summary: str = ""                               # cached LLM one-liner topic
    summary_msg_count: int = 0                       # #user-msgs when generated

    def to_dict(self) -> dict:
        return {"id": self.id, "created": self.created, "updated": self.updated,
                "title": self.title, "messages": self.messages,
                "location": self.location, "people": self.people,
                "summary": self.summary,
                "summary_msg_count": self.summary_msg_count}

    @staticmethod
    def from_dict(d: dict) -> "Session":
        return Session(
            id=d.get("id") or uuid.uuid4().hex,
            created=float(d.get("created") or time.time()),
            updated=float(d.get("updated") or time.time()),
            title=d.get("title") or "new session",
            messages=d.get("messages") or [],
            location=d.get("location") or "",
            people=list(d.get("people") or []),
            summary=d.get("summary") or "",
            summary_msg_count=int(d.get("summary_msg_count") or 0),
        )

    def when(self) -> datetime:
        return datetime.fromtimestamp(self.updated)


# ── context/location labels (decouples session names from user prompts) ──
def _location_label(location: str) -> str:
    """Map a raw venue/SSID string to a friendly, predefined context label
    (Home / Coffee Shop / Work / ...). Falls back to the venue name itself.
    An optional config_phase9.CONTEXT_LOCATION_LABELS dict overrides these."""
    if not location:
        return ""
    raw = str(location).strip()
    low = raw.lower()
    try:
        import config_phase9 as _cfg                     # type: ignore
        mapping = getattr(_cfg, "CONTEXT_LOCATION_LABELS", None)
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                if str(k).strip().lower() == low and v:
                    return str(v)
    except Exception:
        pass
    rules = (
        (("home",), "Home"),
        (("coffee", "cafe", "caf\u00e9", "starbucks", "espresso"), "Coffee Shop"),
        (("office", "work", "hq", "headquarters"), "Work"),
        (("gym", "fitness"), "Gym"),
        (("restaurant", "grill", "diner", "bistro", "kitchen"), "Restaurant"),
        (("library",), "Library"),
        (("airport", "gate", "terminal"), "Airport"),
        (("school", "university", "campus", "college"), "School"),
        (("hospital", "clinic"), "Clinic"),
        (("park",), "Park"),
        (("store", "market", "shop", "mall"), "Store"),
    )
    for keys, label in rules:
        if any(k in low for k in keys):
            return label
    # Don't surface a raw Wi-Fi SSID (e.g. "Att5Xby34A") as a session name.
    if _is_ssid_like(raw):
        return ""
    return raw.title()


def _is_ssid_like(name: str) -> bool:
    """Heuristic: a single space-less token containing digits looks like a
    Wi-Fi SSID / random identifier, not a human place name."""
    name = (name or "").strip()
    if not name or " " in name:
        return False
    return any(c.isdigit() for c in name)


def context_title(location: str, people) -> str:
    """A human context label for a session: prefers who was involved
    ("Talking to Mom", "Conversation with Ana and Sam", "Conversation between
    two people"), else the location ("Coffee Shop", "Home"). "" when there is
    no usable context (raw Wi-Fi SSIDs are ignored)."""
    raw = [str(p).strip() for p in (people or []) if str(p).strip()]
    named = [p for p in raw
             if not p.lower().startswith("unknown") and p.lower() != "n/a"]
    if len(named) == 1:
        return f"Talking to {named[0].title()}"
    if len(named) == 2:
        return f"Conversation with {named[0].title()} and {named[1].title()}"
    if len(named) >= 3:
        return f"Conversation with {len(named)} people"
    if len(raw) >= 2:
        n = len(raw)
        words = {2: "two", 3: "three", 4: "four"}.get(n, str(n))
        return f"Conversation between {words} people"
    return _location_label(location or "")


_STOP_PREFIX = re.compile(
    r"^(please\s+|can you\s+|could you\s+|would you\s+|hey iris,?\s+|"
    r"iris,?\s+|what'?s\s+|what is\s+|what are\s+|whats\s+|tell me about\s+|"
    r"tell me\s+|how do i\s+|how can i\s+|do you\s+|give me\s+|show me\s+|"
    r"i want to\s+|help me\s+)", re.I)


def _make_topic(text: str) -> str:
    """A short lowercase topic phrase from a chat message — the one-line
    'what was this about' summary when there's no location/person context."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return ""
    t = _STOP_PREFIX.sub("", t).strip().rstrip("?.!,")
    words = t.split()
    if len(words) > 7:
        t = " ".join(words[:7]) + "\u2026"
    return t.lower()


def session_label(s) -> str:
    """The sidebar label text (no timestamp): context if known, else a short
    topic summary of the conversation, else 'chat'."""
    ctx = context_title(getattr(s, "location", "") or "",
                        getattr(s, "people", None) or [])
    if ctx:
        return ctx
    summ = (getattr(s, "summary", "") or "").strip()
    if summ:
        return summ
    for m in (getattr(s, "messages", None) or []):
        role = m.get("role") if isinstance(m, dict) else None
        if role == "user":
            topic = _make_topic(m.get("content", "") if isinstance(m, dict) else "")
            if topic:
                return topic
    return "chat"


class SessionStore:
    """Loads/saves sessions and groups them for the sidebar."""

    def __init__(self, directory: Optional[str] = None):
        self.dir = directory or _default_dir()
        self._sessions: dict = {}
        self._load_all()

    # ── disk ──────────────────────────────────────────────────────────────
    def _path(self, sid: str) -> str:
        return os.path.join(self.dir, f"session_{sid}.json")

    def _load_all(self) -> None:
        self._sessions.clear()
        try:
            for fn in os.listdir(self.dir):
                if not (fn.startswith("session_") and fn.endswith(".json")):
                    continue
                try:
                    with open(os.path.join(self.dir, fn), "r",
                              encoding="utf-8") as f:
                        s = Session.from_dict(json.load(f))
                    self._sessions[s.id] = s
                except Exception:
                    continue
        except Exception:
            pass

    def _save(self, s: Session) -> None:
        try:
            with open(self._path(s.id), "w", encoding="utf-8") as f:
                json.dump(s.to_dict(), f, indent=2)
        except Exception:
            pass

    # ── lifecycle ───────────────────────────────────────────────────────────
    def new_session(self) -> Session:
        now = time.time()
        s = Session(id=uuid.uuid4().hex, created=now, updated=now)
        self._sessions[s.id] = s
        return s

    def get(self, sid: str) -> Optional[Session]:
        return self._sessions.get(sid)

    def add_message(self, sid: str, role: str, content: str) -> None:
        s = self._sessions.get(sid)
        if s is None:
            return
        s.messages.append({"role": role, "content": content, "ts": time.time()})
        s.updated = time.time()
        # Session titles come from context/location (set_context), NOT from
        # the user's prompt text (M8 decoupling).
        # An empty session (only the assistant greeting) isn't worth persisting
        # until there's at least one user message.
        if any(m.get("role") == "user" for m in s.messages):
            self._save(s)

    def set_context(self, sid: str, location: Optional[str] = None,
                    people: Optional[list] = None) -> bool:
        """Attach location / people to a session so the sidebar can render the
        "location · person · HH:MM" label. Returns True if anything changed."""
        s = self._sessions.get(sid)
        if s is None:
            return False
        changed = False
        if location:
            loc = str(location).strip()
            if loc and loc != s.location:
                s.location = loc
                changed = True
        if people:
            ppl = [str(p).strip() for p in people if str(p).strip()]
            if ppl and ppl != s.people:
                s.people = ppl
                changed = True
        if changed:
            ctx = context_title(s.location, s.people)
            if ctx and ctx != s.title:
                s.title = ctx
            self._save(s)
        return changed

    def set_summary(self, sid: str, summary: str,
                    msg_count: int = 0) -> bool:
        """Cache a background-generated one-line topic for a session."""
        s = self._sessions.get(sid)
        if s is None:
            return False
        summary = (summary or "").strip()
        if not summary:
            return False
        changed = summary != s.summary
        s.summary = summary
        s.summary_msg_count = int(msg_count or 0)
        self._save(s)
        return changed

    def delete(self, sid: str) -> None:
        self._sessions.pop(sid, None)
        try:
            p = self._path(sid)
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass

    @staticmethod
    def _make_title(text: str) -> str:
        t = re.sub(r"\s+", " ", (text or "").strip())
        return (t[:40] + "\u2026") if len(t) > 40 else (t or "new session")

    # ── sidebar grouping ────────────────────────────────────────────────────
    def grouped(self, exclude: Optional[str] = None) -> list:
        """Return [(group_label, [Session, ...]), ...] newest first, only
        sessions that actually have user messages."""
        now = datetime.now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        week_ago = today - timedelta(days=7)

        buckets = {"Today": [], "Yesterday": [], "This Week": [], "Earlier": []}
        items = [s for s in self._sessions.values()
                 if s.id != exclude
                 and any(m.get("role") == "user" for m in s.messages)]
        items.sort(key=lambda s: s.updated, reverse=True)
        for s in items:
            d = s.when().date()
            if d == today:
                buckets["Today"].append(s)
            elif d == yesterday:
                buckets["Yesterday"].append(s)
            elif d > week_ago:
                buckets["This Week"].append(s)
            else:
                buckets["Earlier"].append(s)
        return [(label, buckets[label]) for label in
                ("Today", "Yesterday", "This Week", "Earlier") if buckets[label]]