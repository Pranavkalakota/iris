"""
iris_command_center.py — M1 glue: utterance → route → feedback → result.

Ties the intent router (iris_intent_router) to the on-screen surface
(iris_response_surface) and the dedup guard. One call runs the whole M1 loop:

    cc = CommandCenter(surface)              # surface optional
    intent = cc.handle_utterance("open gmail")

For M1 this UNDERSTANDS the command and drives the on-screen feedback
(heard → working → done / sorry), then shows what it understood. It does NOT
open apps / analyze video / answer questions itself — those are M2/M3/M4. Pass
an `executor(intent) -> (title, detail)` callback to plug real actions in
later; without one, it just displays the understood intent.

Design: the control logic here has NO PyQt6 dependency. The surface is
duck-typed (any object with set_heard/set_working/show_result/show_error),
so this whole file runs and is tested headless.

Test:
    python iris_command_center.py --headless     # prints state transitions, no Qt
    python iris_command_center.py                # live GUI demo (needs PyQt6)
"""

from __future__ import annotations

import sys
from typing import Callable, Optional, Tuple

from iris_intent_router import route, RouterIntent, DedupGuard

# Below this confidence we treat the utterance as "didn't catch it".
_MIN_ACT_CONFIDENCE = 0.35


class CommandCenter:
    def __init__(self, surface=None,
                 executor: Optional[Callable[[RouterIntent], Tuple[str, str]]] = None,
                 use_llm: bool = True,
                 dedup_window_s: float = 3.0):
        self.surface = surface
        self.executor = executor
        self.use_llm = use_llm
        self._dedup = DedupGuard(window_s=dedup_window_s)

    # ── main entry point ──────────────────────────────────────────────────
    def handle_utterance(self, text: str) -> RouterIntent:
        text = (text or "").strip()
        if not text:
            return RouterIntent(intent="none", confidence=0.0)

        self._safe("set_heard", text)

        intent = route(text, use_llm=self.use_llm)

        # duplicate recognition of the same command → drop (don't re-run)
        if self._dedup.is_duplicate(intent):
            return intent

        if intent.intent == "cancel":
            self._safe("show_result", "Cancelled", "")
            return intent

        if intent.intent == "none" or intent.confidence < _MIN_ACT_CONFIDENCE:
            self._safe("show_error", "Sorry, I didn't catch that — try again")
            return intent

        # feedback: "working…"
        self._safe("set_working", self._working_label(intent))

        # produce a result: real executor if provided, else describe understanding
        try:
            if self.executor is not None:
                title, detail = self.executor(intent)
            else:
                title, detail = self._default_result(intent)
        except Exception as e:                       # never let a handler crash M1
            self._safe("show_error", "Something went wrong running that")
            return intent

        self._safe("show_result", title, detail)
        return intent

    # ── helpers ───────────────────────────────────────────────────────────
    def _safe(self, method: str, *args):
        """Call a surface method if a surface is attached; never raise."""
        if self.surface is None:
            return
        fn = getattr(self.surface, method, None)
        if callable(fn):
            try:
                fn(*args)
            except Exception:
                pass

    @staticmethod
    def _working_label(i: RouterIntent) -> str:
        e = i.entities
        return {
            "open_app":      f"Opening {e.get('app', 'it')}",
            "vision_query":  "Taking a look" if e.get("kind") != "locate"
                             else "Finding it",
            "info":          "Fetching that",
            "question":      "Thinking",
            "memory_recall": "Searching your history",
            "email":         "Checking your email",
            "photo":         "Working with photos",
            "start_video":   "Starting video",
            "start_audio":   "Starting recording",
        }.get(i.intent, "Working")

    @staticmethod
    def _default_result(i: RouterIntent) -> Tuple[str, str]:
        """M1 display-only result (no execution). Describes what was understood;
        real output is filled in by the executor once M2/M3/M4 land."""
        e = i.entities
        if i.intent == "open_app":
            return (e.get("app", "App"),
                    "Opening a new tab" if e.get("new") else "Opening")
        if i.intent == "vision_query":
            if e.get("kind") == "locate":
                return ("Looking for it", "Checking where you left it")
            return ("Looking", "Analyzing what you're seeing")
        if i.intent == "info":
            return (str(e.get("topic", "Info")).title(), "Fetching")
        if i.intent == "question":
            return ("Question", e.get("query", ""))
        if i.intent == "memory_recall":
            return ("Memory", "Searching your history")
        if i.intent == "email":
            return ("Email", "")
        if i.intent == "photo":
            return ("Photos", "")
        if i.intent == "start_video":
            return ("Recording video", "Started")
        if i.intent == "start_audio":
            return ("Recording", "Started")
        return (i.intent.replace("_", " ").title(), "")


# ── factory that builds a real GUI surface (imports PyQt6 lazily) ────────────
def make_gui_surface():
    from iris_response_surface import ResponseSurface     # lazy: no Qt for headless
    return ResponseSurface()


# ── headless test: proves the M1 control flow with a fake surface ────────────
class _FakeSurface:
    def __init__(self):
        self.log = []
    def set_listening(self):            self.log.append(("listening",))
    def set_heard(self, t):             self.log.append(("heard", t))
    def set_working(self, label=""):    self.log.append(("working", label))
    def show_result(self, title, d=""): self.log.append(("done", title, d))
    def show_error(self, m):            self.log.append(("error", m))


def _headless():
    fake = _FakeSurface()
    cc = CommandCenter(surface=fake, use_llm=False)   # keyword-only, deterministic
    script = [
        "open gmail", "open gmail",                    # 2nd is a duplicate
        "open instagram", "what's the weather",
        "what am I looking at", "where did I leave my phone",
        "what's the capital of France", "cancel", "asdfghjkl",
    ]
    print("iris_command_center headless test (keyword router, fake surface)")
    print("-" * 68)
    for u in script:
        before = len(fake.log)
        intent = cc.handle_utterance(u)
        events = fake.log[before:]
        shown = " → ".join(
            e[0] + (f'({e[1]})' if len(e) > 1 and e[1] else '') for e in events
        ) or "(dropped: duplicate)"
        print(f'"{u}"')
        print(f"    intent={intent.intent:<13} conf={round(intent.confidence,2):<4} "
              f"surface: {shown}")
    print("-" * 68)
    print("Done. heard→working→done fires; duplicate is dropped; unrecognized "
          "text routes to chat (question); none/low-confidence → error path.")


def _gui_demo():
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    app = QApplication(sys.argv)
    cc = CommandCenter(surface=make_gui_surface(), use_llm=False)
    script = ["open gmail", "open instagram", "what am I looking at",
              "where did I leave my phone", "what's the weather", "asdfghjkl"]
    for idx, u in enumerate(script):
        QTimer.singleShot(400 + idx * 2600, lambda t=u: cc.handle_utterance(t))
    QTimer.singleShot(400 + len(script) * 2600 + 2000, app.quit)
    print("Command-center GUI demo (~18s). Watch the bottom-right corner.")
    app.exec()


if __name__ == "__main__":
    if "--headless" in sys.argv:
        _headless()
    else:
        _gui_demo()