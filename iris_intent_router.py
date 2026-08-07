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
    "close_app",       # "close instagram", "close youtube tab"
    "vision_query",    # "what am I looking at", "where did I leave my phone"
    "info",            # "what's the weather"
    "question",        # general Q&A → chat/LLM
    "memory_recall",   # "what did we talk about with Prani"
    "email",           # read/act on email
    "photo",           # photo query / capture
    "start_video",     # start recording video
    "start_audio",     # start recording audio
    "cancel",          # "cancel" / "stop" / "never mind"
    "play_song",       # "pull up a kendrick lamar song", "play something by drake"
    "confirm_play",    # "you can play it now", "go ahead and play it"
    "add_to_playlist", # "add it to my workout playlist"
    # --- IRIS M2 youtube: ADD ---
    "yt_search",       # "search youtube for python tutorials"
    "yt_play",         # "play the latest veritasium video"
    "yt_pause",        # "pause the video" / "resume"
    "yt_seek",         # "skip ahead 30 seconds" / "go back 10 seconds"
    "yt_speed",        # "set speed to 1.5x"
    "yt_captions",     # "enable captions" / "turn off subtitles"
    "yt_subscribe",    # "subscribe to MKBHD"
    "yt_like",         # "like this video"
    "yt_channel",      # "open the fireship channel"
    "yt_watch_later",  # "show my watch later playlist"
    # --- IRIS M2 youtube: END ---
    # --- IRIS M2 gdocs: ADD ---
    "gdocs_create",    # "create a new document"
    "gdocs_search",    # "find documents about marketing"
    "gdocs_edit",      # "replace AI with Artificial Intelligence"
    "gdocs_heading",   # "add a heading 2"
    "gdocs_bullets",   # "turn this into bullets"
    "gdocs_comment",   # "comment: review this section"
    "gdocs_share",     # "share this with Alex"
    "gdocs_rename",    # "rename it to Project Proposal"
    "gdocs_export",    # "download this as a PDF"
    # --- IRIS M2 gdocs: END ---
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
_CLOSE_RE  = re.compile(r"\b(close|quit|exit|shut)\b", re.I)
_WEATHER_RE = re.compile(r"\bweather|forecast|temperature (outside|today)|how (hot|cold)\b", re.I)
_NEW_RE    = re.compile(r"\b(new|another|fresh|second)\b", re.I)

# --- IRIS M3 spotify: ADD ---
_PLAY_SONG_RE = re.compile(
    r"\b(play|pull up|put on|throw on|queue up)\b.*\b(song|track|music)\b|"
    r"\b(play|pull up|put on|throw on|queue up)\b.+\bby\s+\w+", re.I)
_CONFIRM_PLAY_RE = re.compile(
    r"\b(you can play it( now)?|play it now|go ahead and play( it)?|"
    r"play (it|that)( now)?)\b", re.I)
_ADD_PLAYLIST_RE = re.compile(
    r"\badd (it|that|this) to\b", re.I)
_BY_ARTIST_RE = re.compile(r"\bby\s+([a-z][a-z0-9 .'\-&]*)$", re.I)
_SONG_BEFORE_KEYWORD_RE = re.compile(
    r"(?:play|pull up|put on|throw on|queue up)\s+(?:a|an|some)?\s*"
    r"([a-z][a-z0-9 .'\-&]*?)\s+(?:song|track)\b", re.I)
_PLAYLIST_NAME_RE = re.compile(
    r"\badd (?:it|that|this) to (?:my )?(.+?)(?:\s+playlist)?$", re.I)
# --- IRIS M3 spotify: END ---

# --- IRIS M2 youtube: ADD ---
# Disambiguation: "play/watch/pause/resume/skip" → YouTube playback.
#                 "record/start recording/capture" → ESP32 video (start_video).
# Generic phrases like "play a video" or bare "pause" go to YouTube because
# the ESP32 recording path has its own verbs (record, capture, start recording).
_YT_SEARCH_RE = re.compile(
    r"\b(?:search|find|look up|look for)\b.*\b(?:youtube|yt|videos?)\b|"
    r"\b(?:youtube|yt)\b.*\b(?:search|find|look up|look for)\b", re.I)
_YT_PLAY_RE = re.compile(
    r"\b(?:play|watch|put on|throw on)\b.*\b(?:on youtube|youtube|videos?)\b|"
    r"\b(?:youtube|yt)\b.*\b(?:play|watch)\b|"
    # Generic: "play a video", "watch a video", "play something"
    r"\b(?:play|watch)\s+(?:a|the|some|this|that|next|another)\s+video\b|"
    r"\b(?:play|watch)\s+(?:something|anything)\b|"
    # "next video" / "previous video"
    r"\b(?:next|previous|prev)\s+video\b", re.I)
_YT_PAUSE_RE = re.compile(
    # Bare "pause", "resume", "unpause" → YouTube (ESP32 uses "stop recording")
    r"\b(?:pause|unpause|resume)\b(?:\s+(?:the\s+)?(?:video|playback|it|this))?\s*$|"
    r"\b(?:pause|unpause|resume)\b.*\b(?:video|youtube|yt|playback)\b", re.I)
_YT_SEEK_RE = re.compile(
    r"\b(?:skip|seek|jump|fast\s*forward|rewind|go\s*back|go\s*forward)\b.*?"
    r"\b(\d+)\s*(?:seconds?|secs?|minutes?|mins?)\b", re.I)
# Generic seek without explicit seconds (defaults to 10s in keyword_route)
_YT_SEEK_GENERIC_RE = re.compile(
    r"\b(?:skip\s*(?:ahead|forward)|fast\s*forward|jump\s*(?:ahead|forward))\b|"
    r"\b(?:go\s*back|rewind|skip\s*back(?:ward)?)\b", re.I)
_YT_SPEED_RE = re.compile(
    r"\b(?:speed|playback\s*(?:speed|rate))\b.*?(\d+\.?\d*)\s*x?\b|"
    r"\b(\d+\.?\d*)\s*x\s*(?:speed)?\b|"
    # Generic: "make it faster", "slow it down", "speed up", "slow down"
    r"\b(?:make\s+it\s+)?(?:faster|speed\s*up)\b|"
    r"\b(?:make\s+it\s+)?(?:slower|slow\s*(?:it\s+)?down)\b", re.I)
_YT_CAPTIONS_RE = re.compile(
    r"\b(?:captions?|subtitles?|cc|closed\s*captions?)\b|"
    r"\b(?:turn\s+(?:on|off)\s+(?:the\s+)?subs?)\b", re.I)
_YT_SUBSCRIBE_RE = re.compile(
    r"\bsubscribe\b(?!\s+to\s+(?:a\s+)?(?:plan|service|newsletter))", re.I)
_YT_LIKE_RE = re.compile(
    r"\blike\b.*\b(?:this|the|that)\s*(?:video)\b|"
    r"\b(?:thumbs?\s*up)\b|"
    r"\blike\s+(?:this|it)\b", re.I)
_YT_CHANNEL_RE = re.compile(
    r"\b(?:open|go to|visit|show|pull up)\b.*\bchannel\b", re.I)
_YT_WATCH_LATER_RE = re.compile(
    r"\bwatch\s*later\b|"
    r"\b(?:save|add)\s+(?:this\s+)?(?:to\s+)?watch\s*later\b", re.I)
# --- IRIS M2 youtube: END ---

# --- IRIS M2 gdocs: ADD ---
_GDOCS_CREATE_RE = re.compile(
    r"\b(?:create|make|new|start|open)\b.*\b(?:new\s+)?(?:doc(?:ument)?|google\s*doc)\b|"
    r"\bnew\s+doc(?:ument)?\b", re.I)
_GDOCS_SEARCH_RE = re.compile(
    r"\b(?:search|find|look for|look up)\b.*\b(?:doc(?:ument)?s?|google\s*doc|drive)\b", re.I)
_GDOCS_EDIT_RE = re.compile(
    r"\b(?:find\s*(?:and\s*)?replace|replace)\b.*\bwith\b", re.I)
_GDOCS_HEADING_RE = re.compile(
    r"\b(?:heading|header)\b.*?(\d)?\b|"
    r"\b(?:add|insert|make|set)\b.*\b(?:heading|header|title|h[1-6])\b|"
    r"\bmake\s+(?:this|it)\s+(?:a\s+)?heading\b", re.I)
_GDOCS_BULLETS_RE = re.compile(
    r"\bbullet(?:s|ed)?\s*(?:list|point|points)?\b|"
    r"\b(?:toggle|add|insert|make)\b.*\bbullet\b|"
    r"\b(?:add|make|turn\s+(?:this|it)\s+into)\b.*\b(?:bullet|list)\b", re.I)
_GDOCS_COMMENT_RE = re.compile(
    r"\b(?:add|insert|leave|make|put)\b.*\bcomment\b|"
    r"\bcomment\b.*\b(?:on|about|this|here)\b", re.I)
_GDOCS_SHARE_RE = re.compile(
    r"\bshare\b.*\b(?:doc(?:ument)?|this|it)\b|"
    r"\bshare\s+(?:with|to)\b", re.I)
_GDOCS_RENAME_RE = re.compile(
    r"\brename\b.*\b(?:doc(?:ument)?|this|it)\b|"
    r"\brename\s+(?:to|it)\b|"
    r"\b(?:change|update)\s+(?:the\s+)?(?:title|name)\b", re.I)
_GDOCS_EXPORT_RE = re.compile(
    r"\b(?:export|download)\b.*\b(?:pdf|doc(?:ument)?|this)\b|"
    r"\b(?:save|download)\s+(?:(?:it|this)\s+)?as\s+(?:a\s+)?pdf\b|"
    r"\b(?:convert|turn)\b.*\bpdf\b", re.I)
# --- IRIS M2 gdocs: END ---


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

    # 1b) close app — "close/quit/exit <app>" (e.g. "close instagram")
    if _CLOSE_RE.search(low):
        _capp = _match_app(low)
        if _capp:
            _n, _u, _ = _capp
            return R("close_app", 0.92, app=_n, url=_u or "")

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

    # --- IRIS M2 youtube: ADD (before weak app-name so "search youtube"
    #     doesn't fall into the open_app bucket) ---
    # Pause/resume — bare "pause"/"resume" goes here (ESP32 uses "stop recording")
    if _YT_PAUSE_RE.search(low):
        return R("yt_pause", 0.90)
    # Seek with explicit seconds
    if _YT_SEEK_RE.search(low):
        m = _YT_SEEK_RE.search(low)
        secs = int(m.group(1))
        if "min" in low:
            secs *= 60
        if any(w in low for w in ("back", "rewind", "behind")):
            secs = -secs
        return R("yt_seek", 0.88, seconds=secs)
    # Generic seek: "skip ahead", "go back", "fast forward" without seconds → default 10s
    if _YT_SEEK_GENERIC_RE.search(low):
        secs = 10
        if any(w in low for w in ("back", "rewind")):
            secs = -10
        return R("yt_seek", 0.85, seconds=secs)
    # Speed with explicit rate OR generic "faster"/"slower"
    if _YT_SPEED_RE.search(low):
        m = re.search(r"(\d+\.?\d*)\s*x", low, re.I)
        if not m:
            m = re.search(r"(?:speed|rate)\b.*?(\d+\.?\d*)", low, re.I)
        if m:
            rate = float(m.group(1))
        elif any(w in low for w in ("faster", "speed up")):
            rate = 1.5
        elif any(w in low for w in ("slower", "slow")):
            rate = 0.75
        else:
            rate = 1.0
        return R("yt_speed", 0.88, rate=rate)
    if _YT_CAPTIONS_RE.search(low):
        return R("yt_captions", 0.88)
    if _YT_SUBSCRIBE_RE.search(low):
        return R("yt_subscribe", 0.88)
    if _YT_LIKE_RE.search(low):
        return R("yt_like", 0.88)
    if _YT_WATCH_LATER_RE.search(low):
        return R("yt_watch_later", 0.88)
    if _YT_CHANNEL_RE.search(low):
        ch = re.sub(r".*\b(?:open|go to|visit|show|pull up)\b\s*(?:the\s*)?", "", low, flags=re.I)
        ch = re.sub(r"\s*\bchannel\b.*", "", ch, flags=re.I).strip()
        return R("yt_channel", 0.85, channel=ch)
    if _YT_SEARCH_RE.search(low):
        q = re.sub(r"\b(?:search|find|look up|look for|on|youtube|yt|for|videos?)\b", "", low, flags=re.I).strip()
        return R("yt_search", 0.88, query=q)
    if _YT_PLAY_RE.search(low):
        q = re.sub(r"\b(?:play|watch|put on|throw on|on youtube|youtube|yt|videos?|the|a|some|next|another|something|anything)\b", "", low, flags=re.I).strip()
        return R("yt_play", 0.85, query=q)
    # --- IRIS M2 youtube: END ---

    # --- IRIS M2 gdocs: ADD ---
    if _GDOCS_CREATE_RE.search(low):
        title = re.sub(r".*\b(?:create|make|new|start)\s+(?:a\s+)?(?:new\s+)?(?:google\s+)?doc(?:ument)?\s*(?:called|named|titled)?\s*", "", low, flags=re.I).strip()
        return R("gdocs_create", 0.88, title=title)
    if _GDOCS_EDIT_RE.search(low):
        m = re.search(r"(?:find\s*(?:and\s*)?replace|replace)\s+(.+?)\s+with\s+(.+)", low, re.I)
        find_t = m.group(1).strip() if m else ""
        repl_t = m.group(2).strip() if m else ""
        return R("gdocs_edit", 0.88, find=find_t, replace=repl_t)
    if _GDOCS_HEADING_RE.search(low):
        m = re.search(r"(\d)", low)
        level = int(m.group(1)) if m else 2
        return R("gdocs_heading", 0.88, level=level)
    if _GDOCS_BULLETS_RE.search(low):
        return R("gdocs_bullets", 0.88)
    if _GDOCS_COMMENT_RE.search(low):
        return R("gdocs_comment", 0.88)
    if _GDOCS_SHARE_RE.search(low):
        return R("gdocs_share", 0.88)
    if _GDOCS_RENAME_RE.search(low):
        name = re.sub(r".*\brename\s+(?:it\s+|this\s+|the\s+doc(?:ument)?\s+)?(?:to\s+)?", "", low, flags=re.I).strip()
        return R("gdocs_rename", 0.88, name=name)
    if _GDOCS_EXPORT_RE.search(low):
        return R("gdocs_export", 0.88)
    if _GDOCS_SEARCH_RE.search(low):
        q = re.sub(r"\b(?:search|find|look for|look up|in|on|my|google|drive|doc(?:ument)?s?|for)\b", "", low, flags=re.I).strip()
        return R("gdocs_search", 0.88, query=q)
    # --- IRIS M2 gdocs: END ---

    # 4b) a known app named without an open verb ("gmail", "youtube") — weak
    if app and app[1]:
        name, url, _alias = app
        return R("open_app", 0.72, app=name, url=url, new=bool(_NEW_RE.search(low)))

    # --- IRIS M3 spotify: ADD ---
    if _CONFIRM_PLAY_RE.search(low):
        return R("confirm_play", 0.75)
    if _ADD_PLAYLIST_RE.search(low):
        m = _PLAYLIST_NAME_RE.search(low)
        playlist = m.group(1).strip() if m else ""
        return R("add_to_playlist", 0.8, playlist=playlist)
    if _PLAY_SONG_RE.search(low):
        artist = ""
        m = _BY_ARTIST_RE.search(low)
        if m:
            artist = m.group(1).strip()
        else:
            m2 = _SONG_BEFORE_KEYWORD_RE.search(low)
            if m2:
                artist = m2.group(1).strip()
        return R("play_song", 0.7, artist=artist, track="")
    # --- IRIS M3 spotify: END ---

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
    'memory_recall, email, photo, start_video, start_audio, cancel, '
    'play_song, confirm_play, add_to_playlist, '
    'yt_search, yt_play, yt_pause, yt_seek, yt_speed, yt_captions, '
    'yt_subscribe, yt_like, yt_channel, yt_watch_later, '
    'gdocs_create, gdocs_search, gdocs_edit, gdocs_heading, gdocs_bullets, '
    'gdocs_comment, gdocs_share, gdocs_rename, gdocs_export, none>", '
    '"confidence": <0..1>, "entities": {<optional: app, url, kind, topic, '
    'question, query, artist, track, playlist, seconds, rate, level, '
    'channel, find, replace, name, title>}}\n'
    "Guidance: 'open/launch/go to <app>' -> open_app (entities.app). "
    "'what am I looking at' / 'what is this' -> vision_query kind=identify. "
    "'where did I leave/put my X' -> vision_query kind=locate. "
    "'what's the weather' -> info topic=weather. "
    "'play/pull up/put on <a song / a song by X / X's song>' -> play_song "
    "with entities.artist (required) and entities.track (only if a "
    "specific song title was named, else omit it). "
    "'you can play it now' / 'play it now' / 'go ahead and play it' -> "
    "confirm_play (no entities needed — this refers to whatever was just "
    "pulled up). "
    "'add it to my <playlist> playlist' -> add_to_playlist with "
    "entities.playlist (the playlist name only, no filler words). "
    "'search youtube for X' -> yt_search with entities.query. "
    "'play X on youtube' / 'watch X' -> yt_play with entities.query. "
    "'pause the video' / 'resume' -> yt_pause. "
    "'skip ahead 30 seconds' / 'go back 10 seconds' -> yt_seek with "
    "entities.seconds (negative = backward). "
    "'set speed to 1.5x' -> yt_speed with entities.rate. "
    "'turn on captions' / 'enable subtitles' -> yt_captions. "
    "'subscribe to this channel' -> yt_subscribe. "
    "'like this video' / 'thumbs up' -> yt_like. "
    "'open the fireship channel' -> yt_channel with entities.channel. "
    "'show my watch later' -> yt_watch_later. "
    "'create a new document' -> gdocs_create with optional entities.title. "
    "'find documents about marketing' -> gdocs_search with entities.query. "
    "'replace X with Y' -> gdocs_edit with entities.find and entities.replace. "
    "'add a heading 2' -> gdocs_heading with entities.level. "
    "'turn this into bullets' -> gdocs_bullets. "
    "'add a comment' -> gdocs_comment. "
    "'share this document' -> gdocs_share. "
    "'rename it to X' -> gdocs_rename with entities.name. "
    "'export as PDF' / 'download as PDF' -> gdocs_export. "
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
    ("pull up a kendrick lamar song",
     '{"intent":"play_song","confidence":0.93,"entities":{"artist":"Kendrick Lamar"}}'),
    ("play alright by kendrick lamar",
     '{"intent":"play_song","confidence":0.95,"entities":{"artist":"Kendrick Lamar","track":"Alright"}}'),
    ("you can play it now",
     '{"intent":"confirm_play","confidence":0.95,"entities":{}}'),
    ("add it to my workout playlist",
     '{"intent":"add_to_playlist","confidence":0.92,"entities":{"playlist":"workout"}}'),
    # --- IRIS M2 youtube few-shot ---
    ("search youtube for python tutorials",
     '{"intent":"yt_search","confidence":0.95,"entities":{"query":"python tutorials"}}'),
    ("play the latest veritasium video",
     '{"intent":"yt_play","confidence":0.93,"entities":{"query":"latest veritasium"}}'),
    ("pause the video",
     '{"intent":"yt_pause","confidence":0.95,"entities":{}}'),
    ("skip ahead 30 seconds",
     '{"intent":"yt_seek","confidence":0.94,"entities":{"seconds":30}}'),
    ("set speed to 1.5x",
     '{"intent":"yt_speed","confidence":0.94,"entities":{"rate":1.5}}'),
    ("turn on captions",
     '{"intent":"yt_captions","confidence":0.93,"entities":{}}'),
    ("subscribe to this channel",
     '{"intent":"yt_subscribe","confidence":0.93,"entities":{}}'),
    ("like this video",
     '{"intent":"yt_like","confidence":0.94,"entities":{}}'),
    ("open the fireship channel",
     '{"intent":"yt_channel","confidence":0.92,"entities":{"channel":"fireship"}}'),
    ("show my watch later",
     '{"intent":"yt_watch_later","confidence":0.93,"entities":{}}'),
    # --- IRIS M2 gdocs few-shot ---
    ("create a new document called Project Proposal",
     '{"intent":"gdocs_create","confidence":0.94,"entities":{"title":"Project Proposal"}}'),
    ("find documents about marketing",
     '{"intent":"gdocs_search","confidence":0.92,"entities":{"query":"marketing"}}'),
    ("replace AI with Artificial Intelligence",
     '{"intent":"gdocs_edit","confidence":0.93,"entities":{"find":"AI","replace":"Artificial Intelligence"}}'),
    ("add a heading 2",
     '{"intent":"gdocs_heading","confidence":0.93,"entities":{"level":2}}'),
    ("turn this into bullets",
     '{"intent":"gdocs_bullets","confidence":0.93,"entities":{}}'),
    ("add a comment",
     '{"intent":"gdocs_comment","confidence":0.92,"entities":{}}'),
    ("share this document",
     '{"intent":"gdocs_share","confidence":0.92,"entities":{}}'),
    ("rename it to Project Proposal",
     '{"intent":"gdocs_rename","confidence":0.93,"entities":{"name":"Project Proposal"}}'),
    ("export as PDF",
     '{"intent":"gdocs_export","confidence":0.94,"entities":{}}'),
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