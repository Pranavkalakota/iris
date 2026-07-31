"""
iris_intent_router.py — M1 intent router for IRIS voice commands.

Turns one transcribed utterance into a structured command:

    RouterIntent(intent, confidence, entities, source, ...)

Primary path : a single local llama3.2:1b call that returns strict JSON.
Fallback path: the existing keyword classifiers in iris_query.py plus a small
               app-open matcher — used whenever the LLM is unavailable, errors,
               returns junk, or is low-confidence.

Design notes
------------
* Additive & standalone. Importing this file changes nothing else in the app;
  it only *reads* config + iris_query. Safe to drop in next to iris_gui.py.
* Degrades gracefully. No Ollama, no llama3.2:1b, or no iris_query? It still
  routes using the built-in keyword rules, so you can test it anywhere.
* Dedup signature. Every intent carries a (intent + entities + 3s bucket)
  signature so repeated recognitions of the same command can be dropped
  (DedupGuard below) — this is the M1 hook the GUI will use.

Quick test (works with OR without Ollama):
    python iris_intent_router.py --selftest
    python iris_intent_router.py "open gmail"
    python iris_intent_router.py            # interactive: type commands, Ctrl-D to quit
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ── optional config (falls back to sane defaults if not importable) ──────────
try:
    import config_phase9 as _cfg                       # type: ignore
except Exception:
    _cfg = None


def _cfgval(name: str, default):
    return getattr(_cfg, name, default) if _cfg is not None else default


OLLAMA_URL    = _cfgval("OLLAMA_URL", "http://localhost:11434")
# The router uses the SMALL model on purpose — it's fast and cheap, and routing
# is a trivial classification, not a reasoning task. Override with the env var
# IRIS_ROUTER_MODEL if you haven't pulled llama3.2:1b yet (e.g. "llama3.2:3b").
ROUTER_MODEL  = os.environ.get("IRIS_ROUTER_MODEL",
                               _cfgval("OLLAMA_ROUTER_MODEL", "llama3.2:1b"))
ROUTER_TIMEOUT_S = float(_cfgval("OLLAMA_TIMEOUT_S", 120.0))

# Below this LLM confidence we defer to the keyword fallback instead.
LLM_MIN_CONFIDENCE = 0.60

# ── optional iris_query (existing classifiers reused as the fallback net) ────
try:
    import iris_query as _iq                            # type: ignore
except Exception:
    _iq = None


# ── canonical intent names (the M1 taxonomy, trimmed to the voice-command
#    world: open apps, ask about what the camera sees, ask general questions) ─
INTENTS = (
    "open_app",        # "open gmail", "open instagram", "open maps"
    "vision_query",    # "what am I looking at", "where did I leave my phone"
    "info",            # "what's the weather"
    "question",        # general Q&A → chat/LLM
    "memory_recall",   # "what did we talk about with Prani"
    "email",           # read/act on email
    "photo",           # photo query / capture
    "start_video",     # start recording video
    "start_audio",     # start recording audio
    "cancel",          # "cancel" / "stop" / "never mind"
    "none",            # nothing actionable
)

# ── app registry: alias → (canonical name, url-or-None) ──────────────────────
# URL apps open in the browser; None means a desktop app to launch by name.
APP_REGISTRY = {
    "gmail":     ("Gmail", "https://mail.google.com"),
    "email":     ("Gmail", "https://mail.google.com"),
    "inbox":     ("Gmail", "https://mail.google.com"),
    "instagram": ("Instagram", "https://www.instagram.com"),
    "insta":     ("Instagram", "https://www.instagram.com"),
    "youtube":   ("YouTube", "https://www.youtube.com"),
    "yt":        ("YouTube", "https://www.youtube.com"),
    "maps":      ("Maps", "https://www.google.com/maps"),
    "google maps": ("Maps", "https://www.google.com/maps"),
    "weather":   ("Weather", "https://www.google.com/search?q=weather"),
    "spotify":   ("Spotify", "https://open.spotify.com"),
    "calendar":  ("Google Calendar", "https://calendar.google.com"),
    "drive":     ("Google Drive", "https://drive.google.com"),
    "chatgpt":   ("ChatGPT", "https://chatgpt.com"),
    "chat gpt":  ("ChatGPT", "https://chatgpt.com"),
    "discord":   ("Discord", "https://discord.com/app"),
    "google docs": ("Google Docs", "https://docs.google.com/document/u/0/"),
    "google doc": ("Google Docs", "https://docs.google.com/document/u/0/"),
    "docs":      ("Google Docs", "https://docs.google.com/document/u/0/"),
    "google calendar": ("Google Calendar", "https://calendar.google.com"),
    "my calendar": ("Google Calendar", "https://calendar.google.com"),
    "twitter":   ("X", "https://x.com"),
    "x":         ("X", "https://x.com"),
    "reddit":    ("Reddit", "https://www.reddit.com"),
    "netflix":   ("Netflix", "https://www.netflix.com"),
    # Windows can't launch the native Android apps, so these map to their
    # web equivalents (both require the phone's account already linked in
    # that browser profile, same as Gmail/Discord/etc.).
    "whatsapp":  ("WhatsApp", "https://web.whatsapp.com"),
    "whats app": ("WhatsApp", "https://web.whatsapp.com"),
    "android messages": ("Google Messages", "https://messages.google.com/web"),
    "android messaging": ("Google Messages", "https://messages.google.com/web"),
    "google messages": ("Google Messages", "https://messages.google.com/web"),
    "chrome":    ("Chrome", None),
    "slack":     ("Slack", None),
    "spotify app": ("Spotify", None),
    "vs code":   ("VS Code", None),
    "vscode":    ("VS Code", None),
    "notion":    ("Notion", None),
}

# phrases that ask the camera "what is this?"
_VISION_IDENTIFY = (
    "what am i looking at", "what is this", "what's this", "what am i seeing",
    "identify this", "what do you see", "what dog is this", "what breed",
    "what kind of", "what plant is this", "what car is this", "read this sign",
)
# phrases that ask "where did I leave / put X?"
_VISION_LOCATE_RE = re.compile(
    r"\b(where (did i|'d i|do i)?\s*(leave|put|last see|last have)|"
    r"where (is|are|'s) my|where did i last)\b", re.I)

_CANCEL_RE = re.compile(r"\b(cancel|never mind|nevermind|stop|forget it|abort)\b", re.I)
_OPEN_RE   = re.compile(r"\b(open|launch|go to|pull up|bring up|show me|start)\b", re.I)
_WEATHER_RE = re.compile(r"\bweather|forecast|temperature (outside|today)|how (hot|cold)\b", re.I)
_NEW_RE    = re.compile(r"\b(new|another|fresh|second)\b", re.I)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RouterIntent:
    intent: str = "none"
    confidence: float = 0.0
    entities: dict = field(default_factory=dict)
    source: str = "keyword"          # "llm" | "keyword" | "llm+keyword"
    raw_text: str = ""
    corrected_text: str = ""

    @property
    def signature(self) -> str:
        """Stable (intent + canonical entities + 3s bucket) signature so the
        GUI can drop duplicate recognitions of the same command."""
        keys = ("app", "kind", "topic", "query", "question")
        parts = []
        for k in keys:
            v = self.entities.get(k)
            if v:
                parts.append(f"{k}={' '.join(str(v).lower().split())[:40]}")
        bucket = int(time.time() // 3)
        return f"{self.intent}|{'&'.join(parts)}|{bucket}"

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 2),
            "entities": self.entities,
            "source": self.source,
            "text": self.corrected_text or self.raw_text,
        }


# ── text normalization (reuse iris_query's cleanup when available) ───────────
def _normalize(text: str) -> str:
    t = (text or "").strip()
    if not t or _iq is None:
        return t
    for fn in ("correct_text", "normalize_casual"):
        try:
            t2 = getattr(_iq, fn)(t)
            if isinstance(t2, str) and t2.strip():
                t = t2
        except Exception:
            pass
    return t


# ── app matching ─────────────────────────────────────────────────────────────
def _match_app(low: str) -> Optional[tuple]:
    """Return (canonical_name, url_or_None, alias) for the longest alias found."""
    best = None
    for alias, (name, url) in APP_REGISTRY.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", low):
            if best is None or len(alias) > len(best[2]):
                best = (name, url, alias)
    return best


# ── keyword fallback router (the safety net; also grounds M2 later) ──────────
def keyword_route(text: str) -> RouterIntent:
    low = text.lower().strip()
    R = lambda intent, conf, **ent: RouterIntent(
        intent=intent, confidence=conf, entities=ent, source="keyword",
        raw_text=text, corrected_text=text)

    if not low:
        return R("none", 0.0)

    # 1) cancel — highest priority
    if _CANCEL_RE.search(low):
        return R("cancel", 0.96)

    # 2) explicit app open — "open/launch/go to <app>"
    app = _match_app(low)
    if app and _OPEN_RE.search(low):
        name, url, _alias = app
        return R("open_app", 0.92, app=name, url=url or "",
                 new=bool(_NEW_RE.search(low)))

    # 3) vision — "what am I looking at" / "where did I leave X"
    if any(p in low for p in _VISION_IDENTIFY):
        return R("vision_query", 0.88, kind="identify", question=text)
    if _VISION_LOCATE_RE.search(low):
        return R("vision_query", 0.86, kind="locate", question=text)

    # 4) quick info (weather) — before the weak app-name branch so
    #    "what's the weather" is info, not "open the weather site"
    if _WEATHER_RE.search(low):
        return R("info", 0.8, topic="weather")

    # 4b) a known app named without an open verb ("gmail", "youtube") — weak
    if app and app[1]:
        name, url, _alias = app
        return R("open_app", 0.72, app=name, url=url, new=bool(_NEW_RE.search(low)))

    # 5) existing iris_query classifiers (video/audio/email/memory/photo)
    if _iq is not None:
        try:
            ai = _iq.classify_action(text)
            k = getattr(ai, "kind", "none")
            if k == "action_start_video":
                return R("start_video", 0.85)
            if k == "action_start_audio":
                return R("start_audio", 0.85)
            if k == "action_open_email":
                return R("open_app", 0.8, app="Gmail",
                         url="https://mail.google.com", new=False)
        except Exception:
            pass
        try:
            ei = _iq.classify_email(text)
            if getattr(ei, "kind", "none") not in ("none", ""):
                return R("email", 0.8, kind=getattr(ei, "kind", ""))
        except Exception:
            pass
        try:
            mi = _iq.classify_memory(text)
            if getattr(mi, "kind", "none") not in ("none", ""):
                return R("memory_recall", 0.78, kind=getattr(mi, "kind", ""))
        except Exception:
            pass
        try:
            if _iq.is_photo_trigger(text) or _iq.is_photo_query(text):
                return R("photo", 0.8)
        except Exception:
            pass

    # 6) default: treat as a general question (low confidence = "not sure")
    return R("question", 0.4, query=text)


# ── LLM router (primary path) ────────────────────────────────────────────────
_LLM_SYSTEM = (
    "You are the intent router for a voice-controlled desktop assistant. "
    "Classify the user's single utterance into ONE intent and return ONLY a "
    "compact JSON object, no prose. Schema:\n"
    '{"intent": "<one of: open_app, vision_query, info, question, '
    'memory_recall, email, photo, start_video, start_audio, cancel, none>", '
    '"confidence": <0..1>, "entities": {<optional: app, url, kind, topic, '
    'question, query>}}\n'
    "Guidance: 'open/launch/go to <app>' -> open_app (entities.app). "
    "'what am I looking at' / 'what is this' -> vision_query kind=identify. "
    "'where did I leave/put my X' -> vision_query kind=locate. "
    "'what's the weather' -> info topic=weather. "
    "A general knowledge or chat question -> question. "
    "'cancel/stop/never mind' -> cancel. Unclear -> none with low confidence."
)

_LLM_FEWSHOT = [
    ("open gmail",
     '{"intent":"open_app","confidence":0.97,"entities":{"app":"Gmail"}}'),
    ("hey what am I looking at",
     '{"intent":"vision_query","confidence":0.94,"entities":{"kind":"identify"}}'),
    ("where did I leave my phone",
     '{"intent":"vision_query","confidence":0.93,"entities":{"kind":"locate"}}'),
    ("what's the capital of France",
     '{"intent":"question","confidence":0.9,"entities":{}}'),
    ("open instagram",
     '{"intent":"open_app","confidence":0.96,"entities":{"app":"Instagram"}}'),
]


def _extract_json(s: str) -> Optional[dict]:
    if not s:
        return None
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def llm_route(text: str, *, model: str = ROUTER_MODEL,
              url: str = OLLAMA_URL) -> Optional[RouterIntent]:
    """Single llama3.2:1b classification call. Returns None on any failure so
    the caller falls back to keyword_route()."""
    try:
        from ollama import Client                         # type: ignore
    except Exception:
        return None
    try:
        client = Client(host=url)
    except Exception:
        return None

    messages = [{"role": "system", "content": _LLM_SYSTEM}]
    for u, a in _LLM_FEWSHOT:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": text})

    try:
        resp = client.chat(model=model, messages=messages,
                           options={"temperature": 0.0},
                           format="json")
    except Exception as e:
        print(f"[router] llm call failed ({model}): {e}", file=sys.stderr)
        return None

    try:
        msg = resp["message"] if isinstance(resp, dict) else getattr(resp, "message", None)
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    except Exception:
        return None

    obj = _extract_json(str(content))
    if not obj:
        return None
    intent = str(obj.get("intent", "none")).strip()
    if intent not in INTENTS:
        return None
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    ent = obj.get("entities", {}) or {}
    if not isinstance(ent, dict):
        ent = {}
    # backfill a URL for open_app if the model named a known app
    if intent == "open_app" and ent.get("app") and not ent.get("url"):
        hit = _match_app(str(ent["app"]).lower())
        if hit and hit[1]:
            ent["url"] = hit[1]
    return RouterIntent(intent=intent, confidence=conf, entities=ent,
                        source="llm", raw_text=text, corrected_text=text)


# ── public entry point ───────────────────────────────────────────────────────
def route(text: str, *, use_llm: bool = True) -> RouterIntent:
    """Route one utterance. Tries the LLM first; falls back to keyword rules
    when the LLM is unavailable or low-confidence."""
    corrected = _normalize(text)
    kw = keyword_route(corrected)
    kw.raw_text = text

    llm = llm_route(corrected) if use_llm else None
    if llm is None:
        return kw
    llm.raw_text = text
    llm.corrected_text = corrected

    # Agreement → boost confidence and label the blend.
    if llm.intent == kw.intent:
        llm.confidence = max(llm.confidence, kw.confidence)
        llm.source = "llm+keyword"
        # prefer keyword's resolved entities (they carry url/app cleanly)
        for k, v in kw.entities.items():
            llm.entities.setdefault(k, v)
        return llm

    # Disagreement → trust the LLM only if it's confident; else keyword net.
    if llm.confidence >= LLM_MIN_CONFIDENCE:
        return llm
    return kw


# ── dedup guard (the M1 hook for "said it 5x → act once") ────────────────────
class DedupGuard:
    """Drops repeated recognitions of the same command inside a short window.
    Distinct from app-level idempotent open — this is about the *recognition*
    firing twice, not the app already being open."""

    def __init__(self, window_s: float = 3.0, maxlen: int = 64):
        self.window_s = window_s
        self._seen: deque = deque(maxlen=maxlen)

    def is_duplicate(self, intent: RouterIntent) -> bool:
        now = time.time()
        sig = intent.signature
        for s, ts in self._seen:
            if s == sig and (now - ts) <= self.window_s:
                return True
        self._seen.append((sig, now))
        return False


# ── CLI / self-test ──────────────────────────────────────────────────────────
_SELFTEST = [
    "open gmail", "open gmail", "open instagram", "open youtube",
    "go to maps", "what's the weather", "hey what am I looking at",
    "where did I leave my phone", "what's the capital of France",
    "record a video", "reply to Prani", "what did we talk about with Jack",
    "cancel", "open a new gmail",
]


def _print(intent: RouterIntent):
    d = intent.as_dict()
    print(f"  intent={d['intent']:<14} conf={d['confidence']:<4} "
          f"src={d['source']:<11} entities={d['entities']}")


def _selftest():
    have_ollama = llm_route("ping") is not None
    print(f"iris_intent_router self-test")
    print(f"  router model : {ROUTER_MODEL}")
    print(f"  iris_query   : {'loaded' if _iq else 'NOT found (keyword net limited)'}")
    print(f"  ollama/LLM   : {'available' if have_ollama else 'unavailable → keyword fallback'}")
    print("-" * 68)
    guard = DedupGuard()
    for t in _SELFTEST:
        r = route(t)
        dup = guard.is_duplicate(r)
        print(f'"{t}"')
        _print(r)
        if dup:
            print("   (duplicate within 3s window — GUI would drop this)")
    print("-" * 68)
    print("Done.")


def main(argv):
    if "--selftest" in argv:
        _selftest()
        return
    args = [a for a in argv if not a.startswith("--")]
    if args:
        r = route(" ".join(args))
        print(json.dumps(r.as_dict(), indent=2))
        return
    # interactive
    print("Type a command (Ctrl-D / Ctrl-C to quit):")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            _print(route(line))
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main(sys.argv[1:])