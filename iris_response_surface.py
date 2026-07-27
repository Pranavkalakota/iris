"""
iris_response_surface.py — M1 on-screen response surface + command feedback.

IRIS does NOT speak. This floating overlay is where every status and result
shows up instead: a small card pinned to the bottom-right of the screen that
moves through  heard you → working → done  (or "sorry" on error). It is the
replacement for the old TTS output.

Standalone & additive — importing it changes nothing else. To SEE it (needs
PyQt6, same as the app):

    python iris_response_surface.py         # cycles through the states as a demo

Public API (also used by iris_command_center.py):
    s = ResponseSurface()
    s.set_listening()
    s.set_heard("open gmail")
    s.set_working("Opening Gmail")
    s.show_result("Gmail", "Opened")
    s.show_error("Didn't catch that")

Note: this module imports PyQt6. The M1 control logic in iris_command_center.py
is deliberately decoupled from it, so that logic can run/tested without Qt.
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
)
from PyQt6.QtGui import QGuiApplication

# state -> (dot color, default status label)
_STATES = {
    "listening": ("#3B82F6", "Listening…"),
    "heard":     ("#14B8A6", "Heard you"),
    "working":   ("#F59E0B", "Working…"),
    "done":      ("#22C55E", "Done"),
    "error":     ("#EF4444", "Sorry"),
}

_MARGIN = 24          # px from the screen corner
_AUTO_HIDE_MS = 4200  # how long a result stays before fading out


class ResponseSurface(QWidget):
    """A frameless, always-on-top card that shows what IRIS heard and did."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self._build()

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._fade_out)
        self._fade = None

        self.resize(380, 104)

    # ── layout ────────────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._card = QWidget(self)
        self._card.setObjectName("card")
        # Solid rounded card with a visible border. No QGraphicsDropShadowEffect:
        # a drop shadow paints outside the widget rect, and on a translucent
        # layered window Windows rejects that repaint (UpdateLayeredWindowIndirect
        # failed). A border gives definition without painting out of bounds.
        self._card.setStyleSheet(
            "#card{background:#12151f; border:1px solid #38507e;"
            " border-radius:14px;}"
        )
        root.addWidget(self._card)

        card = QVBoxLayout(self._card)
        card.setContentsMargins(16, 12, 16, 12)
        card.setSpacing(4)

        # top row: status dot + status label + IRIS brand
        top = QHBoxLayout()
        top.setSpacing(8)
        self._dot = QLabel("●")
        self._dot.setStyleSheet("color:#3B82F6; font-size:14px;")
        self._status = QLabel("Listening…")
        self._status.setStyleSheet(
            "color:#9fb0d0; font-size:12px; font-weight:600;"
            " letter-spacing:0.3px;")
        brand = QLabel("IRIS")
        brand.setStyleSheet(
            "color:#5b6b88; font-size:11px; font-weight:700;"
            " letter-spacing:1px;")
        top.addWidget(self._dot)
        top.addWidget(self._status)
        top.addStretch(1)
        top.addWidget(brand)
        card.addLayout(top)

        # title (the command / result headline)
        self._title = QLabel("")
        self._title.setWordWrap(True)
        self._title.setStyleSheet(
            "color:#f2f5fb; font-size:16px; font-weight:700;")
        card.addWidget(self._title)

        # detail (optional secondary line)
        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color:#aeb9d2; font-size:12.5px;")
        card.addWidget(self._detail)

    # ── positioning ───────────────────────────────────────────────────────
    def _reposition(self):
        scr = QGuiApplication.primaryScreen()
        if scr is None:
            return
        g = scr.availableGeometry()
        self.move(g.right() - self.width() - _MARGIN,
                  g.bottom() - self.height() - _MARGIN)

    # ── state setters ─────────────────────────────────────────────────────
    def _apply_state(self, state: str, status_override: str = ""):
        color, label = _STATES.get(state, _STATES["heard"])
        self._dot.setStyleSheet(f"color:{color}; font-size:14px;")
        self._status.setText(status_override or label)

    def _pop(self):
        """Show (or keep showing) the card, cancel any pending auto-hide."""
        self._hide_timer.stop()
        self._reposition()
        if not self.isVisible():
            self.setWindowOpacity(0.0)
            self.show()
            self.raise_()
            self._fade = QPropertyAnimation(self, b"windowOpacity")
            self._fade.setDuration(160)
            self._fade.setStartValue(0.0)
            self._fade.setEndValue(1.0)
            self._fade.start()
        else:
            self.setWindowOpacity(1.0)
            self.raise_()

    def set_listening(self):
        self._apply_state("listening")
        self._title.setText("")
        self._detail.setText("")
        self._pop()

    def set_heard(self, transcript: str):
        self._apply_state("heard")
        self._title.setText(f"“{transcript}”" if transcript else "")
        self._detail.setText("")
        self._pop()

    def set_working(self, label: str = ""):
        self._apply_state("working")
        if label:
            self._title.setText(label)
        self._detail.setText("")
        self._pop()

    def show_result(self, title: str, detail: str = ""):
        self._apply_state("done")
        self._title.setText(title or "Done")
        self._detail.setText(detail or "")
        self._pop()
        self._hide_timer.start(_AUTO_HIDE_MS)

    def show_error(self, message: str):
        self._apply_state("error")
        self._title.setText(message or "Sorry, I didn't catch that")
        self._detail.setText("")
        self._pop()
        self._hide_timer.start(_AUTO_HIDE_MS)

    # ── fade out / hide ───────────────────────────────────────────────────
    def _fade_out(self):
        self._fade = QPropertyAnimation(self, b"windowOpacity")
        self._fade.setDuration(260)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.hide)
        self._fade.start()


# ── standalone demo ──────────────────────────────────────────────────────────
def _demo():
    app = QApplication(sys.argv)
    s = ResponseSurface()

    # scripted sequence so you can watch every state
    steps = [
        (200,  lambda: s.set_listening()),
        (1000, lambda: s.set_heard("open gmail")),
        (2000, lambda: s.set_working("Opening Gmail")),
        (3200, lambda: s.show_result("Gmail", "Focused the existing tab")),
        (5200, lambda: s.set_heard("what am I looking at")),
        (6200, lambda: s.set_working("Taking a look")),
        (7600, lambda: s.show_result("That's a dog", "Golden retriever, looks like")),
        (9800, lambda: s.set_heard("open the fridge")),
        (10600, lambda: s.show_error("I can't do that one")),
        (13500, app.quit),
    ]
    for ms, fn in steps:
        QTimer.singleShot(ms, fn)

    print("Response-surface demo running (~13s). Watch the bottom-right corner.")
    app.exec()


if __name__ == "__main__":
    _demo()