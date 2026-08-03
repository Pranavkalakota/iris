"""
iris_app_launcher.py — M2 executor: actually open apps/sites, idempotently.

Two launchers live here:

1. `AppLauncher` (original) — a plain in-memory session registry. Simple,
   no browser cooperation needed, but can't tell if you closed the tab by
   hand, and can't focus a specific existing tab (only "don't reopen for
   `reopen_after_s` seconds").

2. `handle()` (new) — the same debug-Chrome pattern proven out on Gmail
   in iris_gui.py's ChatTab._do_action_open_email: real tab reuse/focus
   via Chrome's remote-debugging JSON API, self-launching IRIS's
   dedicated debug Chrome profile when nothing is running, and a
   graceful (cooldown-guarded) fallback when a *different*, non-debug
   Chrome window is already open, since Chrome can't have remote
   debugging turned on after the fact for an already-running window.
   iris_gui.py's M2 gate prefers `handle()` when it's present and falls
   back to `AppLauncher` otherwise.

Scope:
  * Handles URL apps: Instagram, YouTube, Maps, Weather, Spotify,
    Calendar, Google Docs, ChatGPT, Discord, WhatsApp Web, Google
    Messages for web, … (see APP_REGISTRY in iris_intent_router.py).
  * Deliberately SKIPS Gmail — iris_gui.py's ChatTab already has its own
    Gmail opener (same pattern, battle-tested first), so both `handle()`
    and `AppLauncher.open_app()` return None for it and let that
    existing path own it (no double-open).
  * Desktop-only apps (Chrome, Slack, VS Code — no URL) are not launched
    here yet; that's a later step.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.request
import webbrowser
from typing import Callable, Optional, Tuple

# open_app targets the app already handles elsewhere — skip to avoid double-open
_SKIP_URLS = ("mail.google.com",)          # Gmail: existing handler owns it

# ── debug-Chrome constants (IRIS's dedicated profile, same one Gmail uses) ───
DEBUG_PORT = 9222
CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CHROME_DEBUG_PROFILE = (
    r"C:\Users\delete me\AppData\Local\Google\ChromeDebugData")
LAUNCH_POLL_TIMEOUT_S = 10.0
LAUNCH_POLL_INTERVAL_S = 0.4
# Short on purpose: only long enough to swallow the speech-to-text echo that
# re-hears one spoken command a few times. A longer window used to make
# 'close a tab then open it again' wrongly say 'already open'.
REGULAR_CHROME_COOLDOWN_S = 8.0

_no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _debug_json(path: str, method: str = "GET", timeout: float = 1.5):
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{DEBUG_PORT}{path}", method=method)
        with _no_proxy_opener.open(req, timeout=timeout) as r:
            body = r.read()
        return json.loads(body) if body else True
    except Exception as e:
        print(f"[app-open] debug port call failed ({path!r}): {e}")
        return None


def _chrome_already_running() -> bool:
    """True if any chrome.exe process is running, debug-flagged or not.
    Chrome can't turn on remote debugging for a window that's already
    running, so if a plain window is already open, launching our own
    debug profile now would spawn a second, separate window instead of
    landing in the one you're looking at."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq chrome.exe"], text=True)
        return "chrome.exe" in out.lower()
    except Exception as e:
        print(f"[app-open] tasklist check failed: {e}")
        return False


def _normalize(url: str) -> str:
    """Strip the scheme and a leading 'www.' so two URLs can be compared
    as plain host+path+query strings -- e.g. 'https://www.google.com/maps'
    and 'google.com/maps/@40.7,-74.0' both become directly comparable."""
    u = url.split("://", 1)[-1]
    if u.startswith("www."):
        u = u[4:]
    return u


# Shared across every app -- there's only ONE debug Chrome process, so
# "is it currently launching" is a single flag/lock, not per-app.
_launch_lock = threading.Lock()
_launching = False
_regular_open_registry: dict[str, float] = {}   # url -> last-opened timestamp
                                                 # (only used for the non-debug
                                                 # "regular Chrome" fallback)


def _reuse_or_open(app: str, url: str, force_new: bool, tabs,
                    post: Optional[Callable[[str], None]]) -> None:
    """Runs once the debug port is confirmed up. Reports the result via
    `post` if given."""
    print(f"[app-open] {len(tabs)} tab(s) reported for {app}:")
    for t in tabs:
        if t.get("type") == "page":
            print(f"[app-open]   url={t.get('url')!r} title={t.get('title')!r}")

    # A bare domain-substring match (e.g. "google.com") would treat every
    # Google product as interchangeable -- Maps, Weather, Calendar, Drive,
    # and Gmail all contain "google.com" somewhere in their URL. Instead,
    # require the tab's normalized URL to actually START WITH the app's
    # full normalized registered URL (host + path + query), so
    # "google.com/maps" and "google.com/search?q=weather" -- which share
    # a host but nothing else -- can never match each other. The title
    # check stays as a fallback safety net for apps whose registered URL
    # gets redirected to a different domain entirely (e.g. ChatGPT's
    # chat.openai.com -> chatgpt.com).
    key = _normalize(url)
    app_lower = app.lower()
    if not force_new:
        target = next((t for t in tabs
                       if t.get("type") == "page"
                       and (_normalize(t.get("url", "")).startswith(key)
                            or app_lower in t.get("title", "").lower())),
                      None)
        if target is not None:
            print(f"[app-open] reusing tab id={target.get('id')} for {app}")
            _debug_json(f"/json/activate/{target['id']}")
            if post:
                post(f"Back to your {app} tab.")
            return
        print(f"[app-open] no existing {app} tab found, opening new")

    if _debug_json(f"/json/new?{url}", method="PUT") is not None:
        msg = f"Opened {app}."
    else:
        webbrowser.open(url)
        msg = f"Opened {app}."
    if post:
        post(msg)


# ── desktop (non-URL) apps: launch a real Windows .exe, idempotently ─────────
# proc = process image name (for the "already running?" check via tasklist);
# paths = candidate install locations (first that exists is launched);
# cmd  = optional shell fallback (e.g. VS Code's "code" on PATH).
DESKTOP_APPS = {
    "Chrome":  {"proc": "chrome.exe", "paths": [CHROME_EXE]},
    "Slack":   {"proc": "slack.exe",
                "paths": [r"%LOCALAPPDATA%\slack\slack.exe"]},
    "VS Code": {"proc": "Code.exe",
                "paths": [r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"],
                "cmd": "code"},
    "Notion":  {"proc": "Notion.exe",
                "paths": [r"%LOCALAPPDATA%\Programs\Notion\Notion.exe"]},
}


def _proc_running(procname: str) -> bool:
    """True if a process with this image name is running (Windows tasklist)."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"IMAGENAME eq {procname}"],
            text=True, stderr=subprocess.DEVNULL, timeout=4)
        return procname.lower() in out.lower()
    except Exception:
        return False


def _launch_desktop(app: str, post=None) -> str:
    """Open a desktop app by launching its .exe — but don't relaunch it if it's
    already running (idempotent, like the tab reuse but for native apps)."""
    spec = DESKTOP_APPS.get(app)
    if not spec:
        return f"I can't open the {app} app on this PC yet."
    proc = spec.get("proc")
    if proc and _proc_running(proc):
        return f"{app} is already open."
    for path in spec.get("paths", []):
        ep = os.path.expandvars(path)
        if os.path.exists(ep):
            try:
                subprocess.Popen([ep])
                return f"Opening {app}."
            except Exception as ex:
                print(f"[app-open] desktop launch failed for {app}: {ex}")
    cmd = spec.get("cmd")
    if cmd:
        try:
            subprocess.Popen(cmd, shell=True)
            return f"Opening {app}."
        except Exception as ex:
            print(f"[app-open] desktop cmd fallback failed for {app}: {ex}")
    return (f"I couldn't find {app} where I expected on this PC — "
            "you may need to open it manually (or tell me its install path).")


def _close_tab(app: str, url: str, post=None) -> str:
    """Close an app's Chrome tab via the debug port. Desktop apps aren't closed
    here (only browser tabs)."""
    if not url:
        return f"I can close browser tabs, but not the {app} app yet."
    tabs = _debug_json("/json")
    if tabs is None:
        return (f"I can't close {app} — that needs IRIS's Chrome "
                "(debug mode) running.")
    key = _normalize(url)
    app_lower = app.lower()
    target = next((t for t in tabs
                   if t.get("type") == "page"
                   and (_normalize(t.get("url", "")).startswith(key)
                        or app_lower in t.get("title", "").lower())),
                  None)
    if target is None:
        return f"{app} isn't open."
    # Chrome's /json/close replies with plain text ("Target is closing"), not
    # JSON, so _debug_json returns None even though the tab did close. Fire it
    # and report success rather than misreading that None as a failure.
    _debug_json(f"/json/close/{target['id']}", method="PUT")
    return f"Closed {app}."

def handle(intent, post: Optional[Callable[[str], None]] = None
           ) -> Optional[str]:
    """Generalized version of the Gmail opener for any URL-based app --
    Instagram, Calendar, YouTube, ChatGPT, Discord, Google Docs, Spotify,
    WhatsApp Web, Google Messages, etc. Idempotent open (v3 §7): reuses
    an existing tab via Chrome's debug API instead of piling up
    duplicates, self-launches IRIS's dedicated debug Chrome profile when
    nothing is running yet, and falls back (with its own cooldown) when
    a *different*, non-debug Chrome window is already open.

    Returns a message string immediately (e.g. "Opened Instagram." or
    "Starting Chrome for Discord…"). When a Chrome launch has to happen
    in the background, the immediate return is just the "starting"
    message -- the real result comes later via `post(message)`, so
    callers that want the final word should pass `post`.

    Returns None for Gmail (`_SKIP_URLS`) so iris_gui.py's existing,
    separately-tested Gmail opener keeps owning that flow."""
    kind = getattr(intent, "intent", "open_app")
    e = intent.entities or {}
    app = e.get("app", "App")
    url = (e.get("url") or "").strip()
    force_new = bool(e.get("new"))

    if kind == "close_app":
        return _close_tab(app, url, post)

    if any(s in url for s in _SKIP_URLS):
        return None  # let the dedicated handler (Gmail) own this
    if not url:
        return _launch_desktop(app, post)   # desktop exe (Chrome/Slack/VS Code/Notion)

    tabs = _debug_json("/json")
    if tabs is not None:
        # Fast path: debug Chrome already running -- synchronous, no
        # thread needed.
        result: list[str] = []
        _reuse_or_open(app, url, force_new, tabs, post=result.append)
        return result[0] if result else f"Opened {app}."

    # Debug port unreachable -- guard against overlapping launches with a
    # shared lock (there's only one debug Chrome profile for every app).
    global _launching
    with _launch_lock:
        if _launching:
            print("[app-open] launch already in progress, not re-launching")
            return "Still starting Chrome — one moment."
        _launching = True

    def _launch_and_wait():
        global _launching
        try:
            if not os.path.exists(CHROME_EXE):
                if post:
                    post("I couldn't find Chrome at the expected install "
                         "path — try opening it manually this time.")
                return

            if _chrome_already_running():
                # A regular (non-debug) Chrome window is already open.
                # We can't attach the debug port to it, and launching
                # our own profile now would pop up a second window --
                # so hand the URL to whatever Chrome is already running.
                # No tab-reuse detection is possible there, so guard
                # repeats with a per-url cooldown.
                now = time.time()
                last = _regular_open_registry.get(url)
                recently = (last is not None
                            and (now - last) < REGULAR_CHROME_COOLDOWN_S)
                if recently and not force_new:
                    print(f"[app-open] regular-Chrome cooldown active "
                          f"for {app}")
                    if post:
                        post(f"Already opened {app} in your Chrome "
                             "window a moment ago — check your tabs.")
                    return
                webbrowser.open(url)
                _regular_open_registry[url] = now
                if post:
                    post(f"Opening {app} in your current Chrome window.")
                return

            try:
                subprocess.Popen([
                    CHROME_EXE,
                    f"--remote-debugging-port={DEBUG_PORT}",
                    f"--user-data-dir={CHROME_DEBUG_PROFILE}",
                    url,
                ])
            except Exception as ex:
                print(f"[app-open] Chrome launch failed: {ex}")
                if post:
                    post("I couldn't start Chrome — try opening it "
                         "manually this time.")
                return

            deadline = time.time() + LAUNCH_POLL_TIMEOUT_S
            got_tabs = None
            while time.time() < deadline:
                got_tabs = _debug_json("/json")
                if got_tabs is not None:
                    break
                time.sleep(LAUNCH_POLL_INTERVAL_S)

            if got_tabs is not None:
                # Chrome was launched with the URL as a command-line arg,
                # so it's already open -- this just confirms/focuses it.
                _reuse_or_open(app, url, force_new, got_tabs, post=post)
            else:
                print(f"[app-open] debug port never came up within "
                      f"{LAUNCH_POLL_TIMEOUT_S}s for {app}")
                if post:
                    post(f"Opening {app}. (Chrome took longer than "
                         "expected to start, so tab reuse may not work "
                         "next time until it's fully up.)")
        finally:
            with _launch_lock:
                _launching = False

    threading.Thread(target=_launch_and_wait, daemon=True).start()
    return f"Starting Chrome for {app}…"


class AppLauncher:
    def __init__(self, opener=None, reopen_after_s: float = 600.0):
        # `opener` lets tests inject a fake; defaults to the stdlib browser.
        self._wb = opener if opener is not None else webbrowser
        self._opened: dict[str, float] = {}     # url -> last-opened timestamp
        self.reopen_after_s = reopen_after_s

    def open_app(self, intent) -> Optional[Tuple[str, str]]:
        e = intent.entities or {}
        app = e.get("app", "App")
        url = (e.get("url") or "").strip()
        new = bool(e.get("new"))

        # Gmail / anything owned by an existing handler → let that path do it.
        if any(s in url for s in _SKIP_URLS):
            return None

        if not url:
            # a known desktop app with no URL (Chrome, Slack, …) — not this slice
            return (app, "I can open websites for now — desktop apps come later")

        now = time.time()
        last = self._opened.get(url)
        recently_open = (last is not None) and (now - last < self.reopen_after_s)

        # already open this session and not explicitly asked for a new one → skip
        if recently_open and not new:
            self._opened[url] = now
            return (app, "Already open — not opening another tab")

        try:
            if new:
                self._wb.open_new_tab(url)
            else:
                self._wb.open(url)          # new=0: reuse the browser window
            self._opened[url] = now
            return (app, "Opened a new tab" if new else "Opened")
        except Exception as ex:
            return (app, f"Couldn't open it ({ex})")

    def forget(self, url: str = "") -> None:
        """Clear the registry (all, or one url) — e.g. if the user closed a tab
        and wants 'open X' to actually reopen before the reopen window elapses."""
        if url:
            self._opened.pop(url, None)
        else:
            self._opened.clear()


def make_executor(launcher: Optional[AppLauncher] = None):
    """Return an executor(intent) -> (title, detail) | None for CommandCenter.
    Handles open_app; returns None for every other intent so the command center
    falls back to its own default card text."""
    launcher = launcher or AppLauncher()

    def executor(intent):
        if getattr(intent, "intent", "") == "open_app":
            return launcher.open_app(intent)
        return None

    executor.launcher = launcher   # expose for callers/tests
    return executor


# ── self-test (uses a fake browser; no real tabs opened) ─────────────────────
if __name__ == "__main__":
    from iris_intent_router import route

    class _FakeBrowser:
        def __init__(self): self.calls = []
        def open(self, url): self.calls.append(("open", url))
        def open_new_tab(self, url): self.calls.append(("new_tab", url))

    fb = _FakeBrowser()
    ex = make_executor(AppLauncher(opener=fb, reopen_after_s=600))

    script = [
        "open instagram",       # opens
        "open instagram",       # already open → no new tab
        "open youtube",         # opens
        "go to maps",           # opens
        "open a new instagram", # forces a fresh tab
        "open gmail",           # skipped here (existing handler owns it)
    ]
    print("iris_app_launcher self-test (fake browser)\n" + "-" * 60)
    for u in script:
        intent = route(u, use_llm=False)
        res = ex(intent)
        print(f'"{u}"  -> intent={intent.intent:<9} '
              f'card={res}')
    print("-" * 60)
    print("Browser calls actually made:")
    for c in fb.calls:
        print("   ", c)
    print("Expect: instagram open once, youtube once, maps once, instagram "
          "new_tab once; gmail NOT opened here.")
