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

    def to_dict(self) -> dict:
        return {"id": self.id, "created": self.created, "updated": self.updated,
                "title": self.title, "messages": self.messages}

    @staticmethod
    def from_dict(d: dict) -> "Session":
        return Session(
            id=d.get("id") or uuid.uuid4().hex,
            created=float(d.get("created") or time.time()),
            updated=float(d.get("updated") or time.time()),
            title=d.get("title") or "new session",
            messages=d.get("messages") or [],
        )

    def when(self) -> datetime:
        return datetime.fromtimestamp(self.updated)


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
        if (s.title == "new session" and role == "user"):
            s.title = self._make_title(content)
        # An empty session (only the assistant greeting) isn't worth persisting
        # until there's at least one user message.
        if any(m.get("role") == "user" for m in s.messages):
            self._save(s)

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