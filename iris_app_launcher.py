"""
iris_app_launcher.py — M2 executor: actually open apps/sites, idempotently.

Plugs into CommandCenter as the `executor`. For an `open_app` intent it opens
the site in the default browser and remembers it for the session, so repeating
"open Instagram" (or the wake listener re-hearing it) does NOT pile up duplicate
tabs. "open a NEW instagram" forces a fresh tab.

Why a session registry instead of the Chrome debug port: the debug-port lookup
( /json ) times out when Chrome isn't started with remote debugging, which is
exactly what spawned multiple tabs before. A local in-memory registry needs no
browser cooperation and is 100% reliable at preventing duplicates.

Scope of this M2 slice:
  * Handles URL apps: Instagram, YouTube, Maps, Weather, Spotify, Calendar, … .
  * Deliberately SKIPS Gmail — the app already has its own Gmail opener, so we
    return None for it and let that existing path handle it (no double-open).
  * Desktop-only apps (Chrome, Slack, VS Code — no URL) are not launched here
    yet; that's a later step.

Returns (title, detail) for the on-screen card, or None to let CommandCenter
show its own default text (used for Gmail and for non-open_app intents).
"""

from __future__ import annotations

import time
import webbrowser
from typing import Optional, Tuple

# open_app targets the app already handles elsewhere — skip to avoid double-open
_SKIP_URLS = ("mail.google.com",)          # Gmail: existing handler owns it


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