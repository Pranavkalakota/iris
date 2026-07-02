"""
IRIS — Tabbed parent GUI (M1) · PyQt6 liquid-glass version
==========================================================
Tab 1 — Chat with local Llama 3.2 3B (Ollama). Glass bubbles, avatar tiles,
        pill badges, snapshot cards, suggestion chips, session sidebar,
        glass input bar. Recording awareness is handled by iris_query.py;
        session history (sidebar) by iris_sessions.py.
Tab 2 — Audio (embedded glass Qt dashboard driving the Phase 9 backend).
Tab 3 — Location (Leaflet map)
Tab 4 — People (M5 placeholder)
Tab 5 — Stream (M2 placeholder)
Run (from inside the project folder):
    pip install PyQt6 ollama requests
    python iris_gui.py
"""
from __future__ import annotations
import os
import re
import sys
import json
import math
import wave
import queue
import glob
import random
import shutil
import socket
import time
import tempfile
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize, QRectF, QPoint
from PyQt6.QtGui import (
    QColor, QLinearGradient, QPainter, QBrush, QFont, QFontDatabase, QImage,
    QPainterPath, QPen, QShortcut, QKeySequence, QGuiApplication, QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QFrame, QLineEdit, QPushButton,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSlider, QAbstractItemView,
    QVBoxLayout, QHBoxLayout, QScrollArea, QGraphicsDropShadowEffect,
    QStackedWidget, QFileDialog, QSizePolicy, QSizeGrip,
    QGridLayout, QTextEdit, QComboBox, QDialog,
)
# Optional: real map needs PyQt6-WebEngine. Degrades to a glass list if absent.
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except Exception:
    QWebEngineView = None
# Optional: location sidecars (present in the Phase 9 backend).
try:
    from location_phase8 import load_location_sidecar   # type: ignore
except Exception:
    def load_location_sidecar(_path):                    # graceful fallback
        return None
# ── Recording-request engine + session history. Imported defensively so the
#    app still launches if a module is missing (chat just loses that feature).
try:
    import iris_query as iq                              # type: ignore
except Exception:
    iq = None
try:
    import iris_sessions as isess                        # type: ignore
except Exception:
    isess = None
try:
    import iris_photos as iphotos                        # type: ignore
except Exception:
    iphotos = None
try:
    import iris_videos as ivideos                        # type: ignore
except Exception:
    ivideos = None
try:
    import iris_fusion                                   # type: ignore
except Exception:
    iris_fusion = None
# ── Backend imports are defensive so the chat tab runs even without the full
#    Phase 9 backend present (recordings still work via disk scan). ──────────
try:
    import config_phase9 as config            # type: ignore
except Exception:
    config = None
try:
    from main_phase9 import Controller        # type: ignore
except Exception:
    Controller = None
try:
    from ollama import Client as OllamaClient
except ImportError:
    OllamaClient = None
def _cfg(attr: str, default):
    """Read an attribute from config_phase9 with a fallback."""
    if config is not None:
        v = getattr(config, attr, None)
        if v is not None:
            return v
    return default
OLLAMA_URL   = _cfg("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = _cfg("OLLAMA_MODEL", "llama3.2:3b")
# ── Photo capture — screenshot is the default and only active path. The
# ESP32 camera trigger below is wired in per the integration guide but OFF
# by default; flip ESP32_CAMERA_ENABLED in config_phase9.py once the camera
# board is actually available to test with. When on, it's still a fallback:
# if the camera doesn't respond in time, we still take a screenshot so the
# capture never just does nothing. No firmware changes are needed for this —
# the protocol below is exactly what terminal.py already implements. ────────
ESP32_CAMERA_ENABLED      = bool(_cfg("ESP32_CAMERA_ENABLED", False))
ESP32_CAMERA_IP           = _cfg("ESP32_CAMERA_IP", "192.168.1.210")
ESP32_CAMERA_PHOTO_PORT   = int(_cfg("ESP32_CAMERA_PHOTO_PORT", 5006))
# Where the EXISTING receiver app (terminal.py, run by Ali/Humza) already
# saves incoming photos. We watch this folder instead of opening our own
# listener on 5011 — that port is already owned by their receiver on the PC.
ESP32_CAMERA_PHOTOS_DIR   = _cfg(
    "ESP32_CAMERA_PHOTOS_DIR",
    os.path.join(os.path.expanduser("~"), "Desktop", "camera_photos"))
ESP32_CAMERA_WAIT_SECONDS = float(_cfg("ESP32_CAMERA_WAIT_SECONDS", 20.0))
# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
BG_TOP        = "#0b1120"
BG_MID        = "#121a2e"
BG_BOT        = "#1c1838"
TEXT_PRIMARY  = "#e6edf3"
TEXT_MUTED    = "#9ca3af"
TEXT_DIM      = "#6b7280"
TEXT_FAINT    = "#4b5563"
ACCENT        = "#5eead4"
ACCENT_HOVER  = "#2dd4bf"
USER_ACCENT   = "#a78bfa"
BADGE_FACE_FG  = "#34d399"
BADGE_VOICE_FG = "#60a5fa"
BADGE_LOC_FG   = "#fbbf24"
REC_FG         = "#34d399"
COLOR_STATUS_ON  = "#10b981"
COLOR_STATUS_OFF = "#6b7280"
COLOR_DANGER     = "#ef4444"
COLOR_RECORDING  = "#dc2626"
GLASS_FILL_TOP = "rgba(255,255,255,0.13)"
GLASS_FILL_MID = "rgba(255,255,255,0.055)"
GLASS_FILL_BOT = "rgba(255,255,255,0.03)"
GLASS_BORDER   = "rgba(255,255,255,0.14)"
GLASS_BORDER_SOFT = "rgba(255,255,255,0.08)"
BUBBLE_BORDER  = "rgba(255,255,255,0.24)"
WINDOW_RADIUS  = 22
WINDOW_OUTLINE = QColor(255, 255, 255, 42)
FONT_MONO = "Cascadia Code"
FONT_SANS = "Segoe UI"
def _glass_gradient_qss(radius: int = 16,
                        top: str = GLASS_FILL_TOP,
                        mid: str = GLASS_FILL_MID,
                        bot: str = GLASS_FILL_BOT,
                        border: str = GLASS_BORDER) -> str:
    return (
        f"background: qlineargradient(x1:0, y1:0, x2:0, y2:1, "
        f"stop:0 {top}, stop:0.45 {mid}, stop:1 {bot});"
        f"border: 1px solid {border};"
        f"border-radius: {radius}px;"
    )
def _add_glass_shadow(w: QWidget, blur: int = 26, dy: int = 6,
                      alpha: int = 150) -> None:
    eff = QGraphicsDropShadowEffect(w)
    eff.setBlurRadius(blur)
    eff.setXOffset(0)
    eff.setYOffset(dy)
    eff.setColor(QColor(0, 0, 0, alpha))
    w.setGraphicsEffect(eff)
# ─────────────────────────────────────────────────────────────────────────────
# Recording store — discovers recordings + transcripts/summaries from disk.
# Now also captures per-segment timestamps so the chat can answer "what was
# said at 5:30" and "when did we discuss X".
# ─────────────────────────────────────────────────────────────────────────────
RECORDINGS_DIR_OVERRIDE: Optional[str] = None
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma",
               ".webm", ".mp4"}
@dataclass
class Recording:
    name: str
    path: str
    mtime: float
    duration_sec: Optional[float] = None
    transcript: str = ""
    summary: str = ""
    segments: list = field(default_factory=list)   # [{start,end,speaker,text}]
    @property
    def has_transcript(self) -> bool:
        return bool(self.transcript.strip())
    def when(self) -> str:
        try:
            return datetime.fromtimestamp(self.mtime).strftime("%b %d %H:%M")
        except Exception:
            return "—"
    def length(self) -> str:
        if not self.duration_sec:
            return "--:--"
        m, s = divmod(int(self.duration_sec), 60)
        return f"{m:02d}:{s:02d}"
    def label(self) -> str:
        return f"{self.name} · {self.length()} · {self.when()}"
class RecordingStore:
    """Discovers recordings + their transcripts/summaries. Never raises."""
    def __init__(self, controller=None, audio_gui=None):
        self.controller = controller
        self.audio_gui = audio_gui
        self._cache = None
        self._cache_t = 0.0
    def list_recent(self, limit: int = 8) -> list[Recording]:
        import time as _t
        now = _t.time()
        if self._cache is not None and (now - self._cache_t) < 2.0:
            recs = self._cache
        else:
            recs = self._live_recordings()
            if not recs:
                recs = self._scan_disk()
            recs.sort(key=lambda r: r.mtime, reverse=True)
            self._cache = recs
            self._cache_t = now
        return recs[:limit]
    def build(self, audio_path: str) -> Optional[Recording]:
        return self._build_recording(audio_path)
    def _live_recordings(self) -> list[Recording]:
        return []
    def _scan_disk(self) -> list[Recording]:
        out: list[Recording] = []
        seen: set[str] = set()
        visited = 0
        for base in self._candidate_dirs():
            try:
                for root, dirs, files in os.walk(base):
                    dirs[:] = [d for d in dirs if d.lower() not in
                               {"transcripts", "summaries", "photos",
                                "__pycache__", ".git", "node_modules",
                                "chroma", "sqlite"}]
                    for fn in files:
                        if Path(fn).suffix.lower() not in _AUDIO_EXTS:
                            continue
                        full = os.path.abspath(os.path.join(root, fn))
                        key = os.path.normcase(os.path.realpath(full))
                        if key in seen:
                            continue
                        seen.add(key)
                        rec = self._build_recording(full)
                        if rec:
                            out.append(rec)
                        visited += 1
                        if visited > 4000:
                            return out
            except Exception:
                continue
        return out
    def _candidate_dirs(self) -> list[str]:
        raw: list[str] = []
        if RECORDINGS_DIR_OVERRIDE:
            raw.append(RECORDINGS_DIR_OVERRIDE)
        for attr in ("RECORDINGS_DIR", "RECORDING_DIR", "AUDIO_DIR",
                     "AUDIO_OUT_DIR", "AUDIO_SAVE_DIR", "DATA_DIR",
                     "OUTPUT_DIR", "SAVE_DIR", "CLIPS_DIR"):
            v = getattr(config, attr, None) if config is not None else None
            if isinstance(v, (str, os.PathLike)) and str(v).strip():
                raw.append(str(v))
        roots = [os.getcwd()]
        try:
            roots.append(os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            pass
        for r in roots:
            for sub in ("", "recordings", "Recordings", "audio", "Audio",
                        "data/recordings", "data/audio", "data", "clips",
                        "output", "outputs"):
                raw.append(os.path.join(r, sub))
        out, seen = [], set()
        for d in raw:
            try:
                rp = os.path.realpath(os.path.abspath(d))
            except Exception:
                continue
            key = os.path.normcase(rp)
            if key in seen:
                continue
            seen.add(key)
            if os.path.isdir(rp):
                out.append(rp)
        return out
    def _build_recording(self, audio_path: str) -> Optional[Recording]:
        try:
            stat = os.stat(audio_path)
        except Exception:
            return None
        name = os.path.basename(audio_path)
        transcript, summary, dur, segments = self._find_sidecars(audio_path)
        if dur is None:
            dur = self._wav_duration(audio_path)
        return Recording(
            name=name, path=audio_path, mtime=stat.st_mtime,
            duration_sec=dur, transcript=transcript, summary=summary,
            segments=segments,
        )
    def _find_sidecars(self, audio_path: str):
        p = Path(audio_path)
        stem = p.with_suffix("")
        d = p.parent
        transcript, summary, dur, segments = "", "", None, []
        for jpath in [str(stem) + ".json", str(stem) + ".transcript.json",
                      str(d / "transcripts" / (p.stem + ".json"))]:
            if os.path.isfile(jpath):
                t, s, du, segs = self._read_json(jpath)
                transcript = transcript or t
                summary = summary or s
                dur = dur if dur is not None else du
                segments = segments or segs
                break
        if not transcript:
            for tpath in [str(stem) + ".transcript.txt", str(stem) + ".txt",
                          str(stem) + ".transcript", str(stem) + ".srt",
                          str(stem) + ".vtt",
                          str(d / "transcripts" / (p.stem + ".txt")),
                          str(d / "transcripts" / (p.stem + ".srt"))]:
                if os.path.isfile(tpath):
                    transcript = self._clean_transcript(self._read_text(tpath))
                    break
        if not summary:
            for spath in [str(stem) + ".summary.txt", str(stem) + "_summary.txt",
                          str(stem) + ".summary",
                          str(d / "summaries" / (p.stem + ".txt")),
                          str(d / "summaries" / (p.stem + ".summary.txt"))]:
                if os.path.isfile(spath):
                    summary = self._read_text(spath).strip()
                    break
        return transcript, summary, dur, segments
    @staticmethod
    def _read_text(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""
    def _read_json(self, path: str):
        try:
            data = json.loads(self._read_text(path))
        except Exception:
            return "", "", None, []
        transcript, summary, dur, segments = "", "", None, []
        if isinstance(data, dict):
            summary = str(data.get("summary") or "").strip()
            dur = (data.get("duration_sec") or data.get("duration")
                   or data.get("duration_seconds"))
            try:
                dur = float(dur) if dur is not None else None
            except Exception:
                dur = None
            t = data.get("transcript")
            segs = data.get("segments") or data.get("words")
            if isinstance(segs, list):
                for seg in segs:
                    if isinstance(seg, dict):
                        txt = (seg.get("text") or seg.get("word") or "").strip()
                        if txt:
                            segments.append({
                                "start": seg.get("start"),
                                "end": seg.get("end"),
                                "speaker": seg.get("speaker"),
                                "text": txt,
                            })
            if isinstance(t, str) and t.strip():
                transcript = t
            elif segments:
                parts = []
                for seg in segments:
                    spk = seg.get("speaker")
                    txt = seg.get("text", "")
                    parts.append(f"{spk}: {txt}" if spk else txt)
                transcript = "\n".join(parts)
        return self._clean_transcript(transcript), summary, dur, segments
    @staticmethod
    def _clean_transcript(text: str) -> str:
        if not text:
            return ""
        lines = []
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.isdigit():
                continue
            if "-->" in s or "→" in s and "]" not in s:
                continue
            s = re.sub(r"^\[[0-9:.\s→\->]+\]\s*", "", s)
            if s:
                lines.append(s)
        return "\n".join(lines).strip()
    @staticmethod
    def _wav_duration(path: str) -> Optional[float]:
        if Path(path).suffix.lower() != ".wav":
            return None
        try:
            with wave.open(path, "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                if rate:
                    return frames / float(rate)
        except Exception:
            return None
        return None
# ─────────────────────────────────────────────────────────────────────────────
# Photo capture — screenshot (always available) + an ESP32 camera trigger that
# is wired in but inactive unless ESP32_CAMERA_ENABLED is set. Kept as plain
# functions (not a class) since each one is a single, independent operation
# the chat tab calls directly.
# ─────────────────────────────────────────────────────────────────────────────
def _trigger_esp32_photo(ip: str, port: int, timeout: float = 5.0):
    """Send the documented 'take_photo\\n' trigger to the camera ESP32.
    Exactly the protocol terminal.py already uses — connect, send, close, no
    response expected. Returns (ok, error_message)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        s.sendall(b"take_photo\n")
        s.close()
        return True, ""
    except Exception as exc:
        return False, str(exc)
def _grab_screenshot_to(path: str) -> bool:
    """Grab the primary screen and save it as a PNG. Must be called on the
    GUI thread — Qt screen capture isn't safe from a background thread."""
    try:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return False
        pixmap = screen.grabWindow(0)
        if pixmap.isNull():
            return False
        return bool(pixmap.save(path, "PNG"))
    except Exception:
        return False
def _grab_webcam_to(path: str, camera_index: int = 0):
    """Capture one frame from a webcam and save it as a PNG. This is the
    default for a bare 'take a photo' / 'take a picture' — an actual photo
    of the person, not the screen. Uses OpenCV (cv2), which is already in
    the project's pip list. Does blocking device I/O (opening a camera can
    take a noticeable moment), so call this off the GUI thread. Returns
    (ok, error_message).
    Tries a few backend/index combinations before giving up: plain
    cv2.VideoCapture(index) often fails even with a perfectly good, free
    camera because OpenCV doesn't always pick the right backend on its own.
    DSHOW/MSMF are the ones that reliably work on Windows; AVFoundation is
    the one for macOS. Laptops with both a Windows Hello IR camera and a
    regular webcam also sometimes expose the IR one at index 0, so a couple
    of indices are tried too when the caller didn't ask for a specific one.
    If nothing opens at all, the final error message is platform-specific —
    Windows and macOS block camera access for different reasons and fix it
    in different places, so a single generic message wouldn't actually help
    on either one.
    """
    try:
        import cv2
    except ImportError:
        return False, "opencv-python isn't installed (pip install opencv-python)"
    if sys.platform.startswith("win"):
        backend_attempts = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys.platform == "darwin":
        backend_attempts = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backend_attempts = [cv2.CAP_V4L2, cv2.CAP_ANY]
    indices = [camera_index] if camera_index else [0, 1, 2]
    last_err = "no webcam found"
    for idx in indices:
        for backend in backend_attempts:
            cap = None
            try:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    last_err = "no webcam found"
                    continue
                # Many webcams' first frames are dark/off-color before
                # auto-exposure/auto-white-balance settle — warm up first.
                for _ in range(8):
                    cap.read()
                ok, frame = cap.read()
                if not ok or frame is None:
                    last_err = "the webcam opened but didn't return a frame"
                    continue
                if not cv2.imwrite(path, frame):
                    last_err = "couldn't save the captured frame"
                    continue
                return True, ""
            except Exception as e:
                last_err = str(e)
            finally:
                if cap is not None:
                    cap.release()
    if last_err == "no webcam found":
        if sys.platform.startswith("win"):
            last_err = (
                "no webcam found \u2014 if you do have one, this is almost "
                "always Windows blocking it silently: open Settings \u2192 "
                "Privacy & security \u2192 Camera, and make sure both "
                "'Camera access' and 'Let desktop apps access your camera' "
                "are ON. Desktop apps like this one don't get a permission "
                "popup the way browser/Store apps do \u2014 if that toggle "
                "is off, access is just denied with no prompt at all.")
        elif sys.platform == "darwin":
            last_err = (
                "no webcam found \u2014 if you do have one, macOS has "
                "probably blocked it. The first time an app uses the "
                "camera, macOS asks for permission \u2014 but if that was "
                "denied (or missed) before, it won't ask again. Open System "
                "Settings \u2192 Privacy & Security \u2192 Camera, and make "
                "sure it's turned on for whatever's actually running this "
                "script \u2014 Terminal, iTerm, VS Code, PyCharm, etc. \u2014 "
                "not 'Python', since that's what macOS attributes the "
                "request to. After enabling it you may need to fully quit "
                "and reopen that app.")
        else:
            last_err = (
                "no webcam found \u2014 check that a camera is actually "
                "connected, that no other app (browser tab, video call, "
                "etc.) already has it open, and that your user has "
                "permission to access /dev/video* (on some distros that "
                "means being in the 'video' group).")
    return False, last_err
def _photo_source_label(source: str, verbose: bool = False) -> str:
    """Human-readable label for a photo's capture source. verbose=True gives
    the longer 'captured ...' phrasing used in full chat sentences; the
    short form is used in compact captions and list lines."""
    if verbose:
        return {"esp32": "via the ESP32 camera",
                "webcam": "with the webcam"}.get(source, "as a screenshot")
    return {"esp32": "esp32", "webcam": "webcam"}.get(source, "screenshot")
def _photos_dir() -> str:
    """<recordings root>/photos — mirrors how transcripts/summaries already
    sit next to recordings. Always ensured to exist."""
    override = _cfg("PHOTOS_DIR", None)
    if override:
        base = str(override)
    else:
        base = None
        for d in RecordingStore()._candidate_dirs():
            base = d
            break
        if base is None:
            base = os.getcwd()
        base = os.path.join(base, "photos")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return base
# ─────────────────────────────────────────────────────────────────────────────
# Glass widget primitives
# ─────────────────────────────────────────────────────────────────────────────
class GlassFrame(QFrame):
    def __init__(self, parent=None, radius: int = 16,
                 top=GLASS_FILL_TOP, mid=GLASS_FILL_MID, bot=GLASS_FILL_BOT,
                 border=GLASS_BORDER, shadow: bool = True,
                 blur: int = 26, dy: int = 6, shadow_alpha: int = 150):
        super().__init__(parent)
        self.setObjectName("glass")
        self.setStyleSheet(
            "QFrame#glass {" + _glass_gradient_qss(radius, top, mid, bot, border)
            + "}"
        )
        if shadow:
            _add_glass_shadow(self, blur=blur, dy=dy, alpha=shadow_alpha)
class Avatar(GlassFrame):
    def __init__(self, parent, initials: str, fg: str, tint: str):
        super().__init__(parent, radius=9,
                         top=f"rgba({_rgb(fg)},0.22)",
                         mid=f"rgba({_rgb(fg)},0.10)",
                         bot=f"rgba({_rgb(fg)},0.05)",
                         border=f"rgba({_rgb(fg)},0.35)",
                         blur=16, dy=3, shadow_alpha=120)
        self.setFixedSize(36, 36)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(initials)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color:{fg}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:11px; font-weight:700;"
        )
        lay.addWidget(lbl)
class Pill(QLabel):
    def __init__(self, parent, text: str, fg: str):
        super().__init__(text, parent)
        self.setStyleSheet(
            f"color:{fg};"
            f"background: rgba({_rgb(fg)},0.12);"
            f"border: 1px solid rgba({_rgb(fg)},0.30);"
            f"border-radius: 8px; padding: 2px 9px;"
            f"font-family:'{FONT_MONO}','Consolas',monospace; font-size:10px;"
        )
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
class SnapshotCard(GlassFrame):
    def __init__(self, parent, label: str):
        super().__init__(parent, radius=10, blur=18, dy=4, shadow_alpha=120)
        self.setFixedSize(96, 76)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 6)
        lay.setSpacing(2)
        cam = QLabel("\U0001F4F7")
        cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cam.setStyleSheet(f"color:{TEXT_DIM}; background:transparent;"
                          "border:none; font-size:22px;")
        cap = QLabel(label)
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setStyleSheet(f"color:{TEXT_MUTED}; background:transparent;"
                          f"border:none; font-family:'{FONT_MONO}','Consolas',"
                          "monospace; font-size:9px;")
        lay.addStretch(1)
        lay.addWidget(cam)
        lay.addWidget(cap)
        lay.addStretch(1)
class PhotoThumb(GlassFrame):
    """An actual image preview card — used for captured photos (chat inline
    confirmation + the Photos tab gallery). Separate from SnapshotCard, which
    stays a placeholder-style icon card used elsewhere. Optionally clickable
    (used by the Photos tab to make a photo the active chat reference) —
    existing call sites that don't pass on_click are unaffected.
    Fixed width AND height so QGridLayout can never stretch it to fill a row
    (that was the cause of the tall vertical-bar bug — a grid cell with only
    one row of content stretches to fill the whole scroll area unless the
    widget inside refuses to grow)."""
    def __init__(self, parent, image_path: str, caption: str,
                 size: int = 140, on_click=None):
        super().__init__(parent, radius=10, blur=18, dy=4, shadow_alpha=120)
        self._on_click = on_click
        if on_click is not None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        cap_h = 34                          # room for two short caption lines
        self.setFixedSize(size, size + cap_h)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        pic = QLabel()
        pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pic.setFixedSize(size - 12, size - 12)
        pic.setStyleSheet("background: rgba(0,0,0,0.25); border-radius:8px;"
                          "border:none;")
        pm = QPixmap()
        pm.load(image_path)
        if pm.isNull():
            pm.load(image_path, "JPEG")
        if not pm.isNull():
            pic.setPixmap(pm.scaled(
                size - 12, size - 12,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation))
        else:
            pic.setText("\U0001F4F7")
            pic.setStyleSheet(pic.styleSheet() + f"color:{TEXT_DIM}; font-size:24px;")
        lay.addWidget(pic)
        cap_lbl = QLabel(caption)
        cap_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap_lbl.setWordWrap(True)
        cap_lbl.setFixedHeight(cap_h - 4)
        cap_lbl.setStyleSheet(f"color:{TEXT_PRIMARY}; background:transparent;"
                              f"border:none; font-family:'{FONT_MONO}',"
                              "'Consolas',monospace; font-size:11px;")
        lay.addWidget(cap_lbl)
    def mousePressEvent(self, event) -> None:
        if self._on_click is not None and \
                event.button() == Qt.MouseButton.LeftButton:
            self._on_click()
        super().mousePressEvent(event)
class SuggestionChip(QPushButton):
    def __init__(self, parent, text: str, on_click):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QPushButton {"
            f"color:{TEXT_MUTED};"
            f"background: rgba(255,255,255,0.06);"
            f"border: 1px solid {GLASS_BORDER_SOFT};"
            "border-radius: 15px; padding: 6px 14px;"
            f"font-family:'{FONT_SANS}'; font-size:11px;"
            "}"
            "QPushButton:hover { background: rgba(255,255,255,0.11); }"
        )
        self.clicked.connect(lambda: on_click(text))
        _add_glass_shadow(self, blur=14, dy=3, alpha=110)
def _rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
class BubbleLabel(QLabel):
    MAXW = 500
    def __init__(self, text: str = ""):
        super().__init__("")
        f = QFont(FONT_MONO)
        f.setPixelSize(13)
        self.setFont(f)
        self.setWordWrap(True)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setText(text)
    def setText(self, text: str) -> None:
        super().setText(text)
        fm = self.fontMetrics()
        widest = max((fm.horizontalAdvance(ln)
                      for ln in str(text).split("\n")), default=0)
        self.setFixedWidth(min(widest + 2, self.MAXW))
        self.updateGeometry()
class GradientBackground(QWidget):
    def paintEvent(self, _evt):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = QLinearGradient(0, 0, self.width(), self.height())
        g.setColorAt(0.0, QColor(BG_TOP))
        g.setColorAt(0.55, QColor(BG_MID))
        g.setColorAt(1.0, QColor(BG_BOT))
        p.fillRect(self.rect(), QBrush(g))
# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Chat (glass). Recording understanding via iris_query; session history
# via iris_sessions. Rendering + threading are Qt.
# ─────────────────────────────────────────────────────────────────────────────
class ChatTab(QWidget):
    _main_invoke = pyqtSignal(object)
    def __init__(self, parent=None, controller=None, audio_gui=None,
                 switch_to_audio=None):
        super().__init__(parent)
        self._switch_to_audio = switch_to_audio
        self.history: list[dict] = []
        self.busy: bool = False
        self._client: Optional[object] = None
        self.store = RecordingStore(controller=controller, audio_gui=audio_gui)
        self._active: Optional[Recording] = None
        self._pending_pick: Optional[list[Recording]] = None
        self._polling: set[str] = set()
        self._system_prompt = (
            "You are IRIS, a local assistant. You can read the user's audio "
            "recordings, including their transcripts and summaries. When a "
            "recording's transcript is provided to you below, answer strictly "
            "from it and never invent details. If something isn't in the "
            "transcript, say so. Be concise, and when summarizing a recording, "
            "offer 2-3 specific follow-up questions the user could ask about it. "
            "You can ALSO access the ESP32 camera's saved video clips: when a "
            "list of saved video clips is provided below, it includes each "
            "clip's recording time, length, and how many people were detected "
            "in it (with recognised names when available). Use that data to "
            "answer questions about the videos, such as how many people were in "
            "a clip or who was seen. Answer strictly from the clip data given; "
            "if a detail isn't there, say so. "
            "If neither a transcript nor any video-clip data is included in the "
            "message you are answering, you do NOT have access to that "
            "recording's or clip's contents: do not guess or invent what it "
            "says, and say it isn't available."
        )
        # Session history (sidebar). Degrades gracefully if the module is gone.
        self._sessions = isess.SessionStore() if isess is not None else None
        self._session = (self._sessions.new_session()
                         if self._sessions is not None else None)
        # Photo capture store. Degrades gracefully if the module is gone.
        self._photos = (iphotos.PhotoStore(_photos_dir())
                        if iphotos is not None else None)
        # Video-clip store — lets the chat see the ESP32's saved .avi clips
        # and how many people were in them. Degrades gracefully if missing.
        self._videos = None
        if ivideos is not None:
            fusion_getter = (iris_fusion.get_fusion
                             if iris_fusion is not None else None)
            try:
                self._videos = ivideos.VideoStore(fusion_getter=fusion_getter)
            except Exception as e:
                print(f"[video] could not start VideoStore: {e}")
                self._videos = None
        else:
            print("[video] iris_videos.py not found — the chat will not be "
                  "able to answer questions about saved video clips. Make "
                  "sure iris_videos.py is in the same folder as iris_gui.py.")
        # The currently-selected photo (clicked in the Photos tab, or
        # resolved by a chat query) — lets follow-ups reference "this photo".
        self._active_photo: Optional[object] = None
        self._main_invoke.connect(lambda fn: fn())
        self._build_ui()
        self._init_ollama()
    # -- run something on the GUI thread from any thread --
    def _call_main(self, fn) -> None:
        self._main_invoke.emit(fn)
    # ── session logging ──────────────────────────────────────────────────
    def _log(self, role: str, content: str) -> None:
        if self._sessions is not None and self._session is not None:
            try:
                self._sessions.add_message(self._session.id, role, content)
                self._refresh_sidebar()
            except Exception:
                pass
    # ── UI scaffold ──────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_sidebar())
        root.addWidget(self._build_main_pane(), 1)
    # ── Sidebar (live session history) ───────────────────────────────────
    def _build_sidebar(self) -> QWidget:
        panel = GlassFrame(self, radius=16, shadow=True, blur=24, dy=6,
                           shadow_alpha=120,
                           top="rgba(255,255,255,0.06)",
                           mid="rgba(255,255,255,0.035)",
                           bot="rgba(255,255,255,0.02)",
                           border=GLASS_BORDER_SOFT)
        panel.setFixedWidth(236)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 16, 14, 16)
        lay.setSpacing(0)
        new_btn = QPushButton("+  new session")
        new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_btn.setFixedHeight(34)
        new_btn.setStyleSheet(
            "QPushButton {"
            f"color:{ACCENT}; background: rgba({_rgb(ACCENT)},0.12);"
            f"border:1px solid rgba({_rgb(ACCENT)},0.30); border-radius:11px;"
            f"font-family:'{FONT_SANS}'; font-size:12px; font-weight:700; }}"
            f"QPushButton:hover {{ background: rgba({_rgb(ACCENT)},0.20); }}")
        new_btn.clicked.connect(self._new_session)
        lay.addWidget(new_btn)
        lay.addSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:6px;background:transparent;}"
            "QScrollBar::handle:vertical{background:rgba(255,255,255,0.14);"
            "border-radius:3px;}")
        self._sidebar_holder = QWidget()
        self._sidebar_holder.setStyleSheet("background: transparent;")
        self._sidebar_lay = QVBoxLayout(self._sidebar_holder)
        self._sidebar_lay.setContentsMargins(0, 0, 4, 0)
        self._sidebar_lay.setSpacing(0)
        self._sidebar_lay.addStretch(1)
        scroll.setWidget(self._sidebar_holder)
        lay.addWidget(scroll, 1)
        self._refresh_sidebar()
        return panel
    def _refresh_sidebar(self) -> None:
        lay = getattr(self, "_sidebar_lay", None)
        if lay is None:
            return
        while lay.count() > 1:                       # keep the trailing stretch
            item = lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        groups = (self._sessions.grouped(exclude=None)
                  if self._sessions is not None else [])
        active_id = self._session.id if self._session is not None else None
        if not groups:
            lay.insertWidget(0, self._section("TODAY"))
            lay.insertWidget(1, self._session_label("new session", active=True))
            return
        idx = 0
        for label, sessions in groups:
            lay.insertWidget(idx, self._section(label)); idx += 1
            for s in sessions:
                row = self._session_label(s.title, active=(s.id == active_id),
                                          sid=s.id)
                lay.insertWidget(idx, row); idx += 1
    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            f"color:{TEXT_DIM}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:9px; font-weight:700;"
            "padding: 14px 4px 4px 4px; letter-spacing:1px;")
        return lbl
    def _session_label(self, text: str, active: bool = False,
                       sid: Optional[str] = None) -> QWidget:
        dot = "\u25CF" if active else "\u25CB"
        color = ACCENT if active else TEXT_MUTED
        weight = "700" if active else "400"
        btn = QPushButton(f"{dot}  {text}")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton {"
            f"color:{color}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:12px; font-weight:{weight};"
            "text-align:left; padding: 4px 4px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.06);"
            "border-radius:8px; }")
        if sid is not None:
            btn.clicked.connect(lambda _=False, i=sid: self._load_session(i))
        return btn
    def _new_session(self) -> None:
        if self._sessions is not None:
            self._session = self._sessions.new_session()
        self.history.clear()
        self._active = None
        self._active_photo = None
        self._pending_pick = None
        self._clear_log()
        self._init_ollama()
        self._refresh_sidebar()
    def _load_session(self, sid: str) -> None:
        if self._sessions is None:
            return
        s = self._sessions.get(sid)
        if s is None:
            return
        self._session = s
        self.history = [{"role": m["role"], "content": m["content"]}
                        for m in s.messages]
        self._active = None
        self._active_photo = None
        self._pending_pick = None
        self._clear_log()
        for m in s.messages:
            if m["role"] == "user":
                self._append_user(m["content"], log=False)
            else:
                self._append_iris(m["content"], log=False)
        self._refresh_sidebar()
    def _clear_log(self) -> None:
        lay = self.chat_log
        while lay.count() > 1:
            item = lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
    # ── Main pane ────────────────────────────────────────────────────────
    def _build_main_pane(self) -> QWidget:
        pane = QWidget(self)
        lay = QVBoxLayout(pane)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(0)
        header = QHBoxLayout()
        title = QLabel("new session")
        title.setStyleSheet(
            f"color:{TEXT_PRIMARY}; background:transparent;"
            f"font-family:'{FONT_SANS}'; font-size:16px; font-weight:700;")
        header.addWidget(title)
        header.addStretch(1)
        rec_pill = Pill(pane, "\u25CF  ready", REC_FG)
        face_pill = Pill(pane, "face: \u2014", TEXT_DIM)
        header.addWidget(rec_pill)
        header.addSpacing(6)
        header.addWidget(face_pill)
        lay.addLayout(header)
        lay.addSpacing(8)
        self.scroll = QScrollArea(pane)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: transparent; width: 8px; }"
            "QScrollBar::handle:vertical {"
            "  background: rgba(255,255,255,0.14); border-radius: 4px; }"
            "QScrollBar::add-line, QScrollBar::sub-line { height: 0; }")
        self._log_holder = QWidget()
        self._log_holder.setStyleSheet("background: transparent;")
        self.chat_log = QVBoxLayout(self._log_holder)
        self.chat_log.setContentsMargins(2, 4, 12, 4)
        self.chat_log.setSpacing(0)
        self.chat_log.addStretch(1)
        self.scroll.setWidget(self._log_holder)
        lay.addWidget(self.scroll, 1)
        chips = QHBoxLayout()
        chips.setContentsMargins(0, 6, 0, 6)
        chips.addWidget(SuggestionChip(pane, "what's in my last recording?",
                                       self._on_chip))
        chips.addSpacing(8)
        chips.addWidget(SuggestionChip(pane, "summarize today", self._on_chip))
        chips.addStretch(1)
        lay.addLayout(chips)
        input_bar = GlassFrame(pane, radius=22, blur=22, dy=5, shadow_alpha=150)
        input_bar.setFixedHeight(54)
        ib = QHBoxLayout(input_bar)
        ib.setContentsMargins(18, 0, 8, 0)
        ib.setSpacing(8)
        prefix = QLabel(">")
        prefix.setStyleSheet(
            f"color:{TEXT_DIM}; background:transparent; border:none;"
            f"font-family:'{FONT_MONO}','Consolas',monospace;"
            "font-size:16px; font-weight:700;")
        ib.addWidget(prefix)
        self.input = QLineEdit()
        self.input.setPlaceholderText("ask iris anything\u2026")
        self.input.setStyleSheet(
            f"QLineEdit {{ color:{TEXT_PRIMARY}; background:transparent;"
            f"border:none; font-family:'{FONT_SANS}'; font-size:13px; }}")
        self.input.returnPressed.connect(self._on_submit)
        ib.addWidget(self.input, 1)
        self.status_dot = QLabel("\u25A0")
        self.status_dot.setStyleSheet(
            f"color:{ACCENT}; background:transparent; border:none; font-size:13px;")
        ib.addWidget(self.status_dot)
        mic = QPushButton("\U0001F399")
        mic.setCursor(Qt.CursorShape.PointingHandCursor)
        mic.setFixedSize(38, 38)
        mic.setStyleSheet(
            "QPushButton {"
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 rgba({_rgb(ACCENT)},0.95), stop:1 rgba({_rgb(ACCENT_HOVER)},0.95));"
            f"color:{BG_TOP}; border:none; border-radius:19px; font-size:16px; }}"
            f"QPushButton:hover {{ background: {ACCENT_HOVER}; }}")
        _add_glass_shadow(mic, blur=16, dy=3, alpha=130)
        ib.addWidget(mic)
        camera_btn = QPushButton("\U0001F4F7")
        camera_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        camera_btn.setFixedSize(38, 38)
        camera_btn.setToolTip("Take a photo now")
        camera_btn.setStyleSheet(
            "QPushButton {"
            f"background: rgba(255,255,255,0.08);"
            f"border: 1px solid {GLASS_BORDER_SOFT};"
            "border-radius:19px; font-size:15px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.14); }")
        _add_glass_shadow(camera_btn, blur=14, dy=3, alpha=110)
        camera_btn.clicked.connect(self._on_manual_photo_button)
        ib.addWidget(camera_btn)
        lay.addSpacing(6)
        lay.addWidget(input_bar)
        return pane
    # ── Ollama ───────────────────────────────────────────────────────────
    def _init_ollama(self) -> None:
        if OllamaClient is None:
            self._append_iris("(ollama python package missing — pip install ollama)",
                              log=False)
            return
        try:
            self._client = OllamaClient(host=OLLAMA_URL)
            self._append_iris(
                f"Session started. Connected to {OLLAMA_MODEL}. "
                f"Ask me anything — including about your audio recordings, "
                f"e.g. \u201cwhat's my last recording?\u201d",
                pills=[("voice match", BADGE_VOICE_FG)], log=False)
        except Exception as exc:
            self._append_iris(f"(could not connect to Ollama: {exc})", log=False)
    # ── Message rendering ────────────────────────────────────────────────
    def _append_iris(self, body: str,
                     pills: list[tuple[str, str]] | None = None,
                     snapshots: list[str] | None = None,
                     photo_paths: list[str] | None = None,
                     log: bool = True) -> QLabel:
        if log:
            self._log("assistant", body)
        return self._render_message(
            "iris", body, is_user=False, avatar_initials="AI",
            avatar_fg=ACCENT, pills=pills, snapshots=snapshots,
            photo_paths=photo_paths)
    def _append_user(self, body: str, log: bool = True) -> QLabel:
        if log:
            self._log("user", body)
        return self._render_message(
            "you", body, is_user=True, avatar_initials="MA",
            avatar_fg=USER_ACCENT)
    def _render_message(self, author: str, body: str, is_user: bool,
                        avatar_initials: str, avatar_fg: str,
                        pills: list[tuple[str, str]] | None = None,
                        snapshots: list[str] | None = None,
                        photo_paths: list[str] | None = None) -> QLabel:
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        rlay = QHBoxLayout(row)
        rlay.setContentsMargins(4, 10, 4, 0)
        rlay.setSpacing(12)
        avatar = Avatar(row, avatar_initials, avatar_fg, avatar_fg)
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        rlay.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
        rlay.addLayout(col, 1)
        head = QHBoxLayout()
        head.setSpacing(8)
        name = QLabel(author)
        name.setStyleSheet(
            f"color:{avatar_fg}; background:transparent; border:none;"
            f"font-family:'{FONT_MONO}','Consolas',monospace;"
            "font-size:11px; font-weight:700;")
        head.addWidget(name)
        tm = QLabel(f"\u00b7  {datetime.now().strftime('%H:%M')}")
        tm.setStyleSheet(
            f"color:{TEXT_DIM}; background:transparent; border:none;"
            f"font-family:'{FONT_MONO}','Consolas',monospace; font-size:10px;")
        head.addWidget(tm)
        if pills:
            for text, fg in pills:
                head.addWidget(Pill(row, text, fg))
        head.addStretch(1)
        col.addLayout(head)
        bubble = GlassFrame(row, radius=14, border=BUBBLE_BORDER,
                            blur=22, dy=5, shadow_alpha=140)
        blay = QVBoxLayout(bubble)
        blay.setContentsMargins(16, 11, 16, 11)
        body_lbl = BubbleLabel(body)
        body_lbl.setStyleSheet(
            f"color:{TEXT_PRIMARY}; background:transparent; border:none;")
        blay.addWidget(body_lbl)
        brow = QHBoxLayout()
        brow.setContentsMargins(0, 0, 0, 0)
        brow.addWidget(bubble)
        brow.addStretch(1)
        col.addLayout(brow)
        if snapshots:
            snaps = QHBoxLayout()
            snaps.setContentsMargins(0, 6, 0, 2)
            snaps.setSpacing(8)
            for label in snapshots:
                snaps.addWidget(SnapshotCard(row, label))
            snaps.addStretch(1)
            col.addLayout(snaps)
        if photo_paths:
            pics = QHBoxLayout()
            pics.setContentsMargins(0, 6, 0, 2)
            pics.setSpacing(8)
            for p in photo_paths:
                cap = os.path.basename(p)
                pics.addWidget(PhotoThumb(row, p, cap))
            pics.addStretch(1)
            col.addLayout(pics)
        self.chat_log.insertWidget(self.chat_log.count() - 1, row)
        QTimer.singleShot(0, self._scroll_to_bottom)
        return body_lbl
    def _scroll_to_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
    def _on_chip(self, text: str) -> None:
        self.input.setText(text)
        self.input.setFocus()
    # ══════════════════════════════════════════════════════════════════════
    # Routing — pending picks first, then the iris_query classifier.
    # ══════════════════════════════════════════════════════════════════════
    def _on_submit(self) -> None:
        if self.busy:
            return
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self._append_user(text)
        self.history.append({"role": "user", "content": text})
        low = text.lower().strip()
        # (1) A reply that picks from the most recently shown list. The list is
        # kept after a pick so several recordings can be chosen from one list.
        if self._pending_pick and self._is_pick_reply(low):
            rec = self._resolve_pending(low, self._pending_pick)
            if rec is not None:
                self._start_bg(lambda: self._handle_recording(rec))
                return
            n = len(self._pending_pick)
            self._append_iris(
                f"I didn't catch which one. Reply with a number (1-{n}), "
                "a time like 09:40, or a duration like '6 seconds'.")
            return
        # (2) Classify the request with the language engine.
        if iq is None:
            self._start_bg(lambda: self._ask_ollama(text))
            return
        # UI-action intents (start camera / start audio recording) take
        # priority over both memory and recording classifiers — they're
        # asking IRIS to *do* something, not answer a question.
        try:
            act_intent = iq.classify_action(text)
        except Exception:
            act_intent = None
        if act_intent is not None and act_intent.kind != "none":
            self._handle_action_intent(act_intent)
            return
        # A question ABOUT a saved video clip ("how many people were in the
        # video?", "who was in that clip?"). Intercept it here so it always
        # gets the real clip data — otherwise phrasings like "who was in the
        # video" get swallowed by the memory recall classifier below and the
        # video data is never consulted.
        if self._videos is not None and self._is_video_question(low):
            self._start_bg(lambda: self._answer_video_question(text))
            return
        # M7: memory recall takes priority. "Conversations with Pranav"
        # should route to ChromaDB, not to a fuzzy WAV-name match.
        try:
            known = self._known_people_names()
            mem_intent = iq.classify_memory(
                text, known_names=known, today=datetime.now())
        except Exception:
            mem_intent = None
        if mem_intent is not None and mem_intent.kind != "none":
            self._start_bg(lambda: self._handle_memory_query(mem_intent, text))
            return
        intent = iq.classify(text, self._all_recordings(),
                             datetime.now(), has_active=bool(self._active))
        self._dispatch_intent(intent, text)
    def _dispatch_intent(self, intent, text: str) -> None:
        k = intent.kind
        if k == "photo":
            self._trigger_photo_capture(intent.corrected_text or text,
                                        mode=intent.capture_mode)
            return
        if k == "photo_query":
            self._do_photo_query(intent)
            return
        if k == "list":
            if intent.summarize_all:
                recs = [r for r in self._all_recordings()
                        if iq.is_meaningful(r)]
                self._summarize_many(recs, "all recordings")
            else:
                self._append_iris(self._list_recordings_text())
            return
        if k == "latest":
            rec = iq.latest(self._all_recordings())
            if rec is None:
                self._append_iris("I don't see any recordings yet.")
            else:
                self._start_bg(lambda: self._handle_recording(rec))
            return
        if k == "random":
            pool = [r for r in self._all_recordings() if iq.is_meaningful(r)]
            if not pool:
                self._append_iris("I don't see any recordings to pick from.")
            else:
                rec = random.choice(pool)
                self._start_bg(lambda: self._handle_recording(rec))
            return
        if k == "name":
            m = intent.name_matches
            if len(m) == 1:
                self._start_bg(lambda: self._handle_recording(m[0]))
            else:
                self._pending_pick = m[:30]
                self._append_iris(self._format_generic_pick(
                    m[:30], f"I found {len(m)} recordings that could match. "
                    "Which one?"))
            return
        if k == "date":
            self._do_date(intent)
            return
        if k == "date_range":
            self._do_range(intent)
            return
        if k == "index_range":
            self._do_index_range(intent)
            return
        if k == "month":
            self._do_month(intent)
            return
        if k == "time":
            self._do_time(intent)
            return
        if k == "content_search":
            self._do_content(intent)
            return
        # k == "none"
        if self._active is not None or self._active_photo is not None:
            self._start_bg(lambda: self._answer_followup(text))
            return
        self._start_bg(lambda: self._ask_ollama(text))
    # ── Photo capture ──────────────────────────────────────────────────────
    def _on_manual_photo_button(self) -> None:
        """The 📷 button — same action as typing 'take a photo'. A bare
        photo request takes a screenshot of the current screen; if you
        want a photo "of me" via the ESP32 camera, ask in chat or use the
        wake word."""
        self._append_user("\U0001F4F7 take a photo")
        self.history.append({"role": "user", "content": "take a photo"})
        self._trigger_photo_capture("manual capture", mode="camera")
    def handle_voice_trigger(self, phrase: str) -> None:
        """Entry point for a wake-word trigger heard via live audio
        (AudioTab's live transcription listener), as opposed to typed chat
        text or the manual button. Posts a bubble showing what was heard,
        then reuses the exact same capture path as every other trigger
        source — nothing about the capture itself differs by source."""
        heard = (phrase or "").strip()
        self._append_user(f"\U0001F3A4 (heard) {heard}")
        self.history.append({"role": "user", "content": heard})
        mode = iq.photo_capture_mode(heard) if iq is not None else "camera"
        self._trigger_photo_capture(heard or "voice trigger", mode=mode)
    @staticmethod
    def _wants_esp32_selfie(text: str) -> bool:
        """A photo request that targets the user (e.g. "take a picture of
        me", "selfie", "photo of myself") goes through the ESP32 camera
        in the stream tab. A bare "take a photo" stays on the webcam."""
        low = (text or "").lower()
        if not low:
            return False
        if "selfie" in low:
            return True
        if re.search(r"\b(of|with)\s+(me|myself|us)\b", low):
            return True
        return False
    def _trigger_photo_capture(self, trigger_text: str,
                               mode: str = "camera") -> None:
        if self._photos is None:
            self._append_iris(
                "Photo capture isn't available — iris_photos.py is missing.")
            return
        if mode == "screen":
            self._capture_screenshot_now(trigger_text)
            return
        # Selfies ("take a picture of me", "selfie", etc.) → ESP32 camera
        # in the stream tab. Post a "Taking a picture..." bubble in chat
        # first; when the JPEG lands, _on_esp32_photo_arrived() posts the
        # actual image bubble + records it in the Photos tab.
        if self._wants_esp32_selfie(trigger_text):
            stream_cb = getattr(self, "_stream_photo_callback", None)
            if stream_cb is not None:
                try:
                    # Remember the trigger text so the arrival handler can
                    # tag the PhotoStore entry with it.
                    self._pending_esp32_trigger = trigger_text
                    msg = "\U0001F4F8 Taking a picture\u2026"
                    self._append_iris(msg)
                    self.history.append({"role": "assistant", "content": msg})
                    stream_cb()
                    return
                except Exception as e:
                    self._append_iris(
                        f"Couldn't reach the ESP32 camera ({e}); falling "
                        "back to a screenshot.")
        # Everything else with mode="camera" — bare "take a photo" / "take
        # a picture" — is treated as a screenshot of the current screen.
        self._capture_screenshot_now(trigger_text)
    def _capture_webcam_now(self, trigger_text: str) -> None:
        """Webcam capture needs to open a device (can take a noticeable
        moment) so it runs on a background thread; only the final Qt
        posting happens back on the GUI thread via _call_main."""
        def work():
            path = self._photos.new_path("png")
            ok, err = _grab_webcam_to(path)
            self._call_main(lambda: self._finish_webcam_capture(
                trigger_text, path if ok else None, err))
        threading.Thread(target=work, daemon=True).start()
    def _finish_webcam_capture(self, trigger_text: str,
                               path: Optional[str], err: str) -> None:
        if not path:
            msg = f"I couldn't take a photo \u2014 {err}."
            self._append_iris(msg)
            self.history.append({"role": "assistant", "content": msg})
            return
        self._photos.record(path, source="webcam", trigger_text=trigger_text)
        msg = "\U0001F4F8 Got it \u2014 snapped a photo."
        self._append_iris(msg, photo_paths=[path])
        self.history.append({"role": "assistant", "content": msg})
    def _capture_screenshot_now(self, trigger_text: str,
                                note: str = "") -> None:
        """Grab + save a screenshot. Must run on the GUI thread."""
        path = self._photos.new_path("png")
        if not _grab_screenshot_to(path):
            fail_msg = "I couldn't capture a screenshot just now."
            self._append_iris(fail_msg)
            self.history.append({"role": "assistant", "content": fail_msg})
            return
        self._photos.record(path, source="screenshot",
                            trigger_text=trigger_text, note=note)
        msg = "\U0001F4F8 Got it — saved a screenshot."
        if note:
            msg += f" {note}"
        self._append_iris(msg, photo_paths=[path])
        self.history.append({"role": "assistant", "content": msg})
    def _capture_via_esp32(self, trigger_text: str) -> str:
        """Background-thread work: trigger the real camera, wait for the
        existing receiver app to drop the JPEG, fall back to a screenshot if
        it doesn't arrive in time. The photo is saved to the store either
        way; this returns the single status text for the chat bubble (no
        separate thumbnail bubble here, to avoid posting twice for one
        action — the result is always visible in the Photos tab)."""
        since = time.time()        # baseline BEFORE triggering, so even a
        ok, err = _trigger_esp32_photo(ESP32_CAMERA_IP, ESP32_CAMERA_PHOTO_PORT)
        found = None
        if ok:
            deadline = since + ESP32_CAMERA_WAIT_SECONDS
            while time.time() < deadline:
                found = self._photos.newest_new_file(
                    ESP32_CAMERA_PHOTOS_DIR, since)
                if found:
                    break
                time.sleep(1.0)
        if found:
            ext = os.path.splitext(found)[1].lstrip(".") or "jpg"
            dest = self._photos.new_path(ext)
            try:
                shutil.copy2(found, dest)
            except Exception:
                dest = found
            self._photos.record(dest, source="esp32", trigger_text=trigger_text)
            time.sleep(0.3)   # let the file fully flush before Qt loads the thumbnail
            msg = "\U0001F4F8 Got it \u2014 photo received from the ESP32 camera."
            self._call_main(lambda d=dest, m=msg: self._append_iris(
                m, photo_paths=[d]))
            return ""  # empty: _finish_response will remove the thinking bubble cleanly
        # Fallback: hop to the GUI thread for the screenshot grab and wait
        # for it to finish before returning (keeps _start_bg's contract of
        # "background work returns the final text" intact).
        done = threading.Event()
        captured = {}
        def grab():
            path = self._photos.new_path("png")
            captured["ok"] = _grab_screenshot_to(path)
            captured["path"] = path
            done.set()
        self._call_main(grab)
        done.wait(timeout=5.0)
        path = captured.get("path") if captured.get("ok") else None
        reason = ("the camera didn't respond in time" if ok
                  else f"couldn't reach the camera ({err})")
        if not path:
            return f"I couldn't reach the camera ({reason}), and the " \
                   "screenshot fallback failed too."
        self._photos.record(path, source="screenshot", trigger_text=trigger_text,
                            note=f"esp32 fallback: {reason}")
        return f"\U0001F4F8 Took a screenshot instead \u2014 {reason}. See it " \
               "in the Photos tab."
    def _on_esp32_photo_arrived(self, jpeg_path: str) -> None:
        """Called by IrisApp once the ESP32 receiver has saved a new JPEG.
        Copies it into the PhotoStore (so it shows up in the Photos tab,
        dated/timestamped), switches back to the chat tab, and posts an
        inline photo bubble of the result."""
        if self._photos is None or not jpeg_path or not os.path.exists(jpeg_path):
            return
        trigger_text = getattr(self, "_pending_esp32_trigger",
                               "esp32 selfie") or "esp32 selfie"
        self._pending_esp32_trigger = None
        try:
            dest = self._photos.new_path("jpg")
            shutil.copy2(jpeg_path, dest)
        except Exception as exc:
            self._append_iris(
                f"Photo arrived but couldn't be copied into the Photos "
                f"tab ({exc}). It's still in the stream tab's folder.")
            return
        try:
            self._photos.record(dest, source="esp32",
                                trigger_text=trigger_text)
        except Exception:
            pass
        msg = "\U0001F4F8 Here's the photo from the ESP32 camera."
        self._append_iris(msg, photo_paths=[dest])
        self.history.append({"role": "assistant", "content": msg})
    # ── Photo selection + lookup ──────────────────────────────────────────
    def select_photo(self, photo) -> None:
        """Make `photo` the active reference for chat follow-ups. Called both
        when a photo resolves a chat query and when one is clicked in the
        Photos tab gallery."""
        self._active_photo = photo
        tag = _photo_source_label(photo.source, verbose=True)
        msg = f"\U0001F4F7 That photo was taken {photo.when()}, captured {tag}"
        if photo.trigger_text:
            msg += f" (triggered by \u201c{photo.trigger_text}\u201d)"
        msg += (".\n\nI can tell you when or how it was captured, or you can "
                "reference it by date/time \u2014 I can't describe what's "
                "actually in the image, since there's no vision model "
                "wired into chat yet.")
        self._append_iris(msg, photo_paths=[photo.path])
        self.history.append({"role": "assistant", "content": msg})
    def _do_photo_query(self, intent) -> None:
        if self._photos is None:
            self._append_iris("Photo storage isn't available right now.")
            return
        photos = self._photos.list_all()           # newest first
        if not photos:
            self._append_iris(
                "I don't see any photos yet. Say \u201chey iris, take a "
                "photo\u201d or use the \U0001F4F7 button.")
            return
        action = intent.photo_action
        if action == "latest":
            self.select_photo(photos[0])
            return
        if action == "range" and intent.date_range:
            start, end = intent.date_range
            matches = self._photos_in_range(photos, start, end)
            self._show_photo_set(
                matches, f"{self._date_label(start)} \u2192 "
                f"{self._date_label(end)}")
            return
        if action == "date" and intent.dates:
            d = intent.dates[0]
            matches = self._photos_on_date(photos, d)
            if intent.time is not None and matches:
                narrowed = [p for p in matches
                           if self._photo_time_matches(p, intent.time)]
                matches = narrowed or matches
            self._show_photo_set(matches, self._date_label(d))
            return
        if action == "time" and intent.time:
            matches = [p for p in photos
                      if self._photo_time_matches(p, intent.time)]
            h, mi, s = intent.time
            clock = f"{h:02d}:{mi:02d}" + (f":{s:02d}" if s is not None else "")
            self._show_photo_set(matches, clock)
            return
        # action == "all" (or anything unrecognized) -> the most recent batch
        self._show_photo_set(photos[:8],
                             "your photos" if len(photos) > 1 else "your photo")
    def _show_photo_set(self, photos, label: str) -> None:
        if not photos:
            self._append_iris(f"I don't see any photos for {label}.")
            return
        if len(photos) == 1:
            self.select_photo(photos[0])
            return
        shown = photos[:8]
        lines = [f"\U0001F4F8 {len(photos)} photo{'s' if len(photos) != 1 else ''} "
                f"for {label}:"]
        for p in shown:
            tag = _photo_source_label(p.source)
            lines.append(f"  \u2022 {p.when()} \u00b7 {tag}")
        if len(photos) > len(shown):
            lines.append(f"  \u2026and {len(photos) - len(shown)} more \u2014 "
                         "see the Photos tab.")
        text = "\n".join(lines)
        self._append_iris(text, photo_paths=[p.path for p in shown])
        self.history.append({"role": "assistant", "content": text})
    @staticmethod
    def _photos_on_date(photos, d) -> list:
        y, mo, day = d
        out = []
        for p in photos:
            dt = datetime.fromtimestamp(p.taken_at)
            if dt.month == mo and dt.day == day and (y is None or dt.year == y):
                out.append(p)
        return out
    @staticmethod
    def _photos_in_range(photos, start, end) -> list:
        def to_dt(dd):
            yy = (dd[0] if dd[0] is not None
                 else (start[0] or end[0] or datetime.now().year))
            return datetime(yy, dd[1], dd[2])
        lo, hi = to_dt(start), to_dt(end)
        if lo > hi:
            lo, hi = hi, lo
        hi = hi + timedelta(days=1)
        return [p for p in photos
                if lo <= datetime.fromtimestamp(p.taken_at) < hi]
    @staticmethod
    def _photo_time_matches(p, tm) -> bool:
        h, mi, s = tm
        dt = datetime.fromtimestamp(p.taken_at)
        if dt.hour != h or dt.minute != mi:
            return False
        if s is not None and dt.second != s:
            return False
        return True
    # ── date / range / month / time handlers ─────────────────────────────
    def _do_date(self, intent) -> None:
        recs = self._all_recordings()
        d = intent.dates[0]
        cands = iq.candidates_for_date(recs, d)
        if intent.time is not None:
            h, mi, s = intent.time
            nd = [r for r in cands if iq.rec_dt(r).hour == h
                  and iq.rec_dt(r).minute == mi
                  and (s is None or iq.rec_dt(r).second == s)]
            cands = nd or cands
        if not cands:
            self._append_iris(
                f"I don't see a recording on {self._date_label(d)}. "
                "Pick one from the file explorer instead.")
            self._open_picker_and_handle()
            return
        if intent.summarize_all and len(cands) > 1:
            self._summarize_many(cands, self._date_label(d))
            return
        if len(cands) == 1:
            self._start_bg(lambda: self._handle_recording(cands[0]))
            return
        self._pending_pick = cands
        self._append_iris(self._format_pick(
            cands, f"You have {len(cands)} recordings on "
            f"{self._date_label(d)}. Which one?", show="time"))
    def _do_range(self, intent) -> None:
        start, end = intent.date_range
        cands = iq.candidates_for_range(self._all_recordings(), start, end)
        if not cands:
            self._append_iris(
                f"I don't see any recordings between {self._date_label(start)} "
                f"and {self._date_label(end)}.")
            return
        self._summarize_many(
            cands, f"{self._date_label(start)} \u2192 {self._date_label(end)}")
    def _do_index_range(self, intent) -> None:
        a, b = intent.index_range
        base = self._pending_pick if self._pending_pick else \
            sorted(self._all_recordings(), key=iq.rec_dt, reverse=True)
        base = [r for r in base if not iq.is_empty(r)] if not self._pending_pick \
            else base
        sel = base[a - 1:b]
        if not sel:
            self._append_iris(
                f"I only have {len(base)} recordings in that list, so I can't "
                f"reach {a}\u2013{b}. Try a smaller range.")
            return
        self._summarize_many(sel, f"items {a}\u2013{b}")
    def _do_month(self, intent) -> None:
        y, mo, _ = intent.dates[0]
        cands = iq.candidates_for_month(self._all_recordings(), y, mo)
        if not cands:
            self._append_iris(
                f"I don't see any recordings in {self._month_label((y, mo))}. "
                "Pick one from the file explorer instead.")
            self._open_picker_and_handle()
            return
        if intent.summarize_all and len(cands) > 1:
            self._summarize_many(cands, self._month_label((y, mo)))
            return
        if len(cands) == 1:
            self._start_bg(lambda: self._handle_recording(cands[0]))
            return
        self._pending_pick = cands
        self._append_iris(self._format_pick(
            cands, f"You have {len(cands)} recordings in "
            f"{self._month_label((y, mo))}. Which one?", show="date"))
    def _do_time(self, intent) -> None:
        cands = iq.candidates_for_time(self._all_recordings(), intent.time)
        if len(cands) == 1:
            self._start_bg(lambda: self._handle_recording(cands[0]))
            return
        if len(cands) > 1:
            h, mi, s = intent.time
            clock = f"{h:02d}:{mi:02d}" + (f":{s:02d}" if s is not None else "")
            self._pending_pick = cands
            self._append_iris(self._format_pick(
                cands, f"I found {len(cands)} recordings at {clock}. Which one?",
                show="datetime"))
            return
        self._append_iris("I don't see a recording at that time.")
    def _do_content(self, intent) -> None:
        topic = intent.content_query
        hits = iq.content_search(topic, self._all_recordings())
        if not hits:
            self._append_iris(
                f"I couldn't find a recording where you talked about "
                f"\u201c{topic}\u201d. It may not be transcribed yet, or the "
                "topic was phrased differently.")
            return
        if len(hits) == 1:
            rec = hits[0]
            self._append_iris(
                f"That sounds like \u201c{rec.name}\u201d ({rec.when()}). "
                "Pulling it up\u2026")
            self._start_bg(lambda: self._handle_recording(rec))
            return
        self._pending_pick = hits[:30]
        self._append_iris(self._format_pick(
            hits[:30], f"I found {len(hits)} recordings that mention "
            f"\u201c{topic}\u201d. Which one?", show="datetime"))
    # ── picker ───────────────────────────────────────────────────────────
    def _open_picker_and_handle(self) -> None:
        path = self._pick_via_dialog()
        if not path:
            self._append_iris(
                "No file selected. Ask me again and choose a recording from "
                "the picker, or type part of its name or date.")
            return
        rec = self.store.build(path)
        if rec is None:
            self._append_iris("I couldn't read that file.")
            return
        self._start_bg(lambda: self._handle_recording(rec))
    # ── Background runner ────────────────────────────────────────────────
    def _start_bg(self, work) -> None:
        self.busy = True
        self.status_dot.setStyleSheet(
            f"color:{USER_ACCENT}; background:transparent; border:none; font-size:13px;")
        thinking = self._append_iris("\u2026", log=False)
        def run():
            try:
                reply = work()
            except Exception as exc:
                reply = f"(error handling that: {exc})"
            self._call_main(lambda: self._finish_response(thinking, reply))
        threading.Thread(target=run, daemon=True).start()
    def _finish_response(self, thinking_label: QLabel, reply: str) -> None:
        try:
            if reply:
                thinking_label.setText(reply)
            else:
                # Empty reply means the handler (e.g. _capture_via_esp32) already
                # posted its own bubble via _call_main — remove the thinking "…" widget.
                thinking_label.setParent(None)
                thinking_label.deleteLater()
        except Exception:
            pass
        if reply:
            self.history.append({"role": "assistant", "content": reply})
            self._log("assistant", reply)
        self.busy = False
        self.status_dot.setStyleSheet(
            f"color:{ACCENT}; background:transparent; border:none; font-size:13px;")
        QTimer.singleShot(0, self._scroll_to_bottom)
    # ── Recordings access (mirror audio tab, merge duplicate rows) ───────
    def _all_recordings(self) -> list[Recording]:
        gui = self.store.audio_gui
        rows = getattr(gui, "_rows", None) if gui is not None else None
        recs: list[Recording] = []
        if rows:
            recs = [self.store.build(p) for _, p in rows]
            recs = [r for r in recs if r is not None]
        if not recs:
            recs = self.store.list_recent(limit=500)
        return self._merge_dupes(recs)
    @staticmethod
    def _merge_dupes(recs: list[Recording]) -> list[Recording]:
        """Collapse rows that are the same clip (same name, start time, and
        length) into one, preferring the transcribed copy."""
        best: dict = {}
        for r in recs:
            dt = iq.rec_dt(r) if iq is not None else \
                datetime.fromtimestamp(r.mtime)
            key = (r.name.lower(), dt.replace(microsecond=0),
                   round(r.duration_sec) if r.duration_sec else None)
            cur = best.get(key)
            if cur is None or (r.has_transcript and not cur.has_transcript):
                best[key] = r
        return list(best.values())
    def _list_recordings_text(self) -> str:
        recs = [r for r in self._all_recordings() if not iq.is_empty(r)]
        if not recs:
            return ("I don't see any recordings yet \u2014 record one in the "
                    "Audio tab or import a file, and it'll show up here.")
        recs.sort(key=iq.rec_dt, reverse=True)
        self._pending_pick = recs[:30]
        n = len(recs)
        head = (f"I can see {n} recording{'s' if n != 1 else ''}"
                + (" (showing the 30 most recent)" if n > 30 else "") + ":\n")
        lines = [head]
        for i, r in enumerate(self._pending_pick, 1):
            when = iq.rec_dt(r).strftime("%b %d %H:%M")
            mark = "" if r.has_transcript else "  (not transcribed)"
            lines.append(f"  {i}. {r.name} \u00b7 {when} \u00b7 {r.length()}{mark}")
        lines.append("\nReference any by name or date, or reply with its "
                     "number, and I'll pull up its transcript.")
        return "\n".join(lines)
    # ── Pick-list formatting ─────────────────────────────────────────────
    def _format_pick(self, cands, prompt: str, show: str = "time") -> str:
        lines = [prompt + "\n"]
        for i, r in enumerate(cands, 1):
            dt = iq.rec_dt(r)
            if show == "time":
                stamp = dt.strftime("%H:%M:%S")
            elif show == "date":
                stamp = dt.strftime("%b %d %H:%M")
            elif show == "datetime":
                stamp = dt.strftime("%b %d %H:%M:%S")
            else:
                stamp = dt.strftime("%b %d %H:%M")
            mark = "" if r.has_transcript else "  (not transcribed yet)"
            lines.append(f"  {i}. {stamp} \u00b7 {r.name} \u00b7 {r.length()}{mark}")
        lines.append("\nReply with a number, a time like 09:40, or a duration "
                     "like '6 seconds'.")
        return "\n".join(lines)
    def _format_generic_pick(self, cands, prompt: str) -> str:
        return self._format_pick(cands, prompt, show="datetime")
    def _date_label(self, d) -> str:
        y, mo, day = d
        name = [k for k, v in iq.MONTHS.items() if v == mo][0].capitalize()
        return f"{name} {day}" + (f", {y}" if y else "")
    def _month_label(self, mo) -> str:
        year, month = mo
        name = [k for k, v in iq.MONTHS.items() if v == month][0].capitalize()
        return f"{name}" + (f" {year}" if year else "")
    # ── Pick-reply detection + resolution (multi-pick from one list) ─────
    @staticmethod
    def _is_pick_reply(low: str) -> bool:
        if re.search(r"\b(?:option|number|item|no\.?|#)\s*\d{1,3}\b", low):
            return True
        if re.fullmatch(r"\s*#?\d{1,3}\s*", low):
            return True
        if re.search(r"\b\d{1,2}(?:st|nd|rd|th)\b", low):
            return True
        if re.search(r"\b\d{1,3}\s*-?\s*(?:seconds?|secs?|minutes?|mins?)\b", low):
            return True
        if re.search(r"\b\d{1,2}:[0-5]\d(?::[0-5]\d)?\b", low):
            return True
        qwords = ("who", "what", "when", "where", "why", "how", "did", "was",
                  "were", "is", "are", "does", "do", "can", "could", "should")
        if not any(re.search(rf"\b{w}\b", low) for w in qwords):
            if any(re.search(rf"\b{w}\b", low) for w in (
                    "first", "second", "third", "fourth", "fifth", "sixth",
                    "seventh", "eighth", "ninth", "tenth", "earliest",
                    "latest", "newest")):
                return True
            if "most recent" in low or re.search(
                    r"\bthe last (one|recording)\b", low):
                return True
        return False
    def _resolve_pending(self, low: str, cands) -> Optional[Recording]:
        n = len(cands)
        dur = iq.parse_duration(low)
        if dur is not None:
            matches = [r for r in cands if r.duration_sec is not None
                       and round(r.duration_sec) == dur]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                return matches[-1]
        idx = self._parse_ordinal(low)
        if idx is not None and 1 <= idx <= n:
            return cands[idx - 1]
        if "earliest" in low:
            return cands[0]
        if ("latest" in low or "most recent" in low
                or re.search(r"\blast\b", low)):
            return cands[-1]
        digits = re.sub(r"[^0-9]", "", low)
        if digits and len(digits) >= 3:
            for r in cands:
                rdt = iq.rec_dt(r)
                hhmmss = f"{rdt.hour:02d}{rdt.minute:02d}{rdt.second:02d}"
                hhmm = f"{rdt.hour:02d}{rdt.minute:02d}"
                if digits in (hhmmss, hhmm) or (len(digits) >= 4
                                                and digits in hhmmss):
                    return r
        idx = self._parse_index(low)
        if idx is not None and 1 <= idx <= n:
            return cands[idx - 1]
        for r in cands:
            stem = os.path.splitext(r.name)[0].lower()
            if stem and (stem in low or low in stem):
                return r
        return None
    @staticmethod
    def _parse_ordinal(low: str) -> Optional[int]:
        words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                 "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
        for w, i in words.items():
            if re.search(rf"\b{w}\b", low):
                return i
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b", low)
        if m:
            return int(m.group(1))
        return None
    @staticmethod
    def _parse_index(low: str) -> Optional[int]:
        m = re.search(r"\b(?:number|option|item|no\.?|#)\s*(\d{1,3})\b", low)
        if m:
            return int(m.group(1))
        if re.fullmatch(r"\s*#?(\d{1,3})\s*", low):
            return int(re.search(r"\d{1,3}", low).group())
        m = re.search(r"\b(\d{1,3})\b", low)
        if m:
            return int(m.group(1))
        return None
    # ── File picker ──────────────────────────────────────────────────────
    def _recordings_dir(self) -> str:
        named = None
        for d in self.store._candidate_dirs():
            try:
                for fn in os.listdir(d):
                    if Path(fn).suffix.lower() in _AUDIO_EXTS:
                        return d
            except Exception:
                pass
            if named is None and os.path.basename(d).lower() in (
                    "recordings", "recording"):
                named = d
        return named or os.getcwd()
    def _pick_via_dialog(self) -> str:
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select a recording", self._recordings_dir(),
                "Audio files (*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma "
                "*.webm *.mp4);;All files (*.*)")
            return path or ""
        except Exception:
            return ""
    # ── Handle a chosen recording ────────────────────────────────────────
    def _handle_recording(self, rec: Recording) -> str:
        self._active = rec
        header = f"\U0001F4FC {rec.name} \u00b7 {rec.length()} \u00b7 {rec.when()}\n\n"
        if rec.has_transcript:
            return self._summarize_recording(rec)
        if rec.duration_sec is not None and rec.duration_sec <= 0:
            return (header + "The audio you selected is zero seconds long, so "
                    "there's nothing I can transcribe. Pick a different "
                    "recording.")
        self._call_main(lambda: self._do_transcribe_ui(rec))
        return (header + "This recording isn't transcribed yet. I've opened the "
                "Audio tab and started transcribing it for you. Once it "
                "finishes, ask me about it again and I'll summarize it.")
    # ── Auto-transcription (GUI thread) ──────────────────────────────────
    _TRANSCRIBE_POLL_MS = 2000
    _TRANSCRIBE_POLL_MAX = 150
    def _do_transcribe_ui(self, rec: Recording) -> None:
        try:
            if self._switch_to_audio is not None:
                self._switch_to_audio()
        except Exception:
            pass
        if self._invoke_audio_transcription(rec):
            if rec.path not in self._polling:
                self._polling.add(rec.path)
                QTimer.singleShot(self._TRANSCRIBE_POLL_MS,
                                  lambda: self._poll_transcription(rec.path, 0))
        else:
            self._append_iris(
                "I couldn't auto-start transcription, but I've taken you to the "
                f"Audio tab \u2014 select \"{rec.name}\" and click the "
                "transcribe button (the third blue button in Recordings).")
    def _poll_transcription(self, path: str, attempts: int) -> None:
        try:
            rec = self.store.build(path)
        except Exception:
            rec = None
        if rec is not None and rec.has_transcript:
            self._polling.discard(path)
            self._active = rec
            self._post_auto_summary(rec)
            return
        if attempts >= self._TRANSCRIBE_POLL_MAX:
            self._polling.discard(path)
            self._append_iris(
                f"Transcription of {os.path.basename(path)} is still running. "
                "Ask me about it once it finishes and I'll summarize it.")
            return
        QTimer.singleShot(self._TRANSCRIBE_POLL_MS,
                          lambda: self._poll_transcription(path, attempts + 1))
    def _post_auto_summary(self, rec: Recording) -> None:
        label = self._append_iris(
            f"\u2705 {rec.name} finished transcribing. Summarizing\u2026")
        def run():
            reply = self._summarize_recording(rec)
            self._call_main(lambda: self._safe_set(label, reply))
        threading.Thread(target=run, daemon=True).start()
    def _safe_set(self, label: QLabel, text: str) -> None:
        try:
            label.setText(text)
        except Exception:
            pass
        self.history.append({"role": "assistant", "content": text})
        self._log("assistant", text)
        QTimer.singleShot(0, self._scroll_to_bottom)
    def _invoke_audio_transcription(self, rec: Recording) -> bool:
        gui = self.store.audio_gui
        if gui is not None:
            try:
                if hasattr(gui, "_select"):
                    gui._select(rec.path)
                else:
                    gui._selected_path = rec.path
                if hasattr(gui, "_on_transcribe_clicked"):
                    gui._on_transcribe_clicked()
                    return True
                ctrl = getattr(gui, "controller", None)
                if ctrl is not None and hasattr(ctrl, "transcribe_file"):
                    ctrl.transcribe_file(rec.path)
                    return True
            except Exception:
                pass
        ctrl = self.store.controller
        if ctrl is not None and hasattr(ctrl, "transcribe_file"):
            try:
                ctrl.transcribe_file(rec.path)
                return True
            except Exception:
                pass
        return False
    # ── Summaries (single + many) ────────────────────────────────────────
    def _summarize_recording(self, rec: Recording) -> str:
        header = f"\U0001F4FC {rec.name} \u00b7 {rec.length()} \u00b7 {rec.when()}\n\n"
        if not rec.has_transcript:
            return (header + "This recording hasn't been transcribed yet. Open "
                    "the Audio tab, select it, and run transcription first.")
        transcript = self._truncate(rec.transcript, 7000)
        if self._client is not None:
            prompt = (
                "Summarize this recording transcript in 3-4 sentences, then "
                "list 2-3 specific follow-up questions the user could ask "
                "about it. Use only what's in the transcript.\n\n"
                f"TRANSCRIPT:\n{transcript}")
            try:
                resp = self._client.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "system", "content": self._system_prompt},
                              {"role": "user", "content": prompt}])
                return header + resp["message"]["content"].strip()
            except Exception as exc:
                if rec.summary:
                    return header + rec.summary
                return (header + f"(couldn't reach the model: {exc})\n\n"
                        "Transcript excerpt:\n" + self._truncate(rec.transcript, 800))
        if rec.summary:
            return header + rec.summary
        return header + "Transcript excerpt:\n" + self._truncate(rec.transcript, 800)
    def _summarize_many(self, recs, label: str) -> None:
        recs = [r for r in recs if not iq.is_empty(r)]
        if not recs:
            self._append_iris(f"I don't see any recordings for {label}.")
            return
        capped = recs[:8]
        note = "" if len(recs) <= 8 else f" (first 8 of {len(recs)})"
        self._start_bg(
            lambda: self._do_summarize_many(capped, label, note))
    def _do_summarize_many(self, recs, label: str, note: str) -> str:
        header = f"\U0001F4CA {label}{note} \u2014 {len(recs)} recording(s)\n\n"
        transcribed = [r for r in recs if r.has_transcript]
        missing = [r for r in recs if not r.has_transcript]
        if not transcribed:
            lines = [header + "None of these are transcribed yet:"]
            for r in recs:
                lines.append(f"  \u2022 {r.name} \u00b7 {r.when()} \u00b7 {r.length()}")
            lines.append("\nOpen one and I'll transcribe it, then summarize.")
            return "\n".join(lines)
        if self._client is None:
            lines = [header]
            for r in transcribed:
                s = r.summary or self._truncate(r.transcript, 200)
                lines.append(f"\u2022 {r.name} ({r.when()}): {s}")
            return "\n".join(lines)
        blocks = []
        for r in transcribed:
            blocks.append(f"=== {r.name} ({r.when()}, {r.length()}) ===\n"
                          + self._truncate(r.transcript, 2500))
        prompt = (
            "Summarize each of the following recordings in 1-2 sentences, "
            "labeled by file name, then finish with a short overall takeaway "
            "across all of them. Use only what's in each transcript.\n\n"
            + "\n\n".join(blocks))
        try:
            resp = self._client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "system", "content": self._system_prompt},
                          {"role": "user", "content": prompt}])
            out = header + resp["message"]["content"].strip()
        except Exception as exc:
            out = header + f"(couldn't reach the model: {exc})"
        if missing:
            out += ("\n\n(Not transcribed yet: "
                    + ", ".join(r.name for r in missing) + ")")
        return out
    # ── Follow-up about the active recording (incl. timestamp / topic) ───
    _RECORDING_Q_WORDS = (
        "summar", "transcript", "recording", "what did", "what was",
        "who said", "who is", "who was", "what happened", "talk about",
        "talked about", "discuss", "mention", "meeting", "the call",
        "this call", "what's in", "whats in", "recap", "time frame",
        "timeframe", "what time", "when did", "they said", "conversation",
    )
    def _is_about_recording(self, low: str) -> bool:
        if any(k in low for k in self._RECORDING_Q_WORDS):
            return True
        return len(low.split()) <= 4
    def _topic_from_question(self, text: str) -> str:
        topic = iq.extract_topic(text) if iq is not None else ""
        if topic:
            return topic
        low = text.lower()
        low = re.sub(r"[?.!,]", " ", low)
        drop = {"when", "what", "time", "where", "did", "we", "i", "you", "the",
                "a", "an", "at", "point", "in", "this", "recording", "talk",
                "talked", "about", "discuss", "discussed", "mention",
                "mentioned", "was", "is", "of", "do", "does", "happen", "say",
                "said", "happened"}
        toks = [t for t in re.split(r"\s+", low) if t and t not in drop
                and len(t) >= 3]
        return " ".join(toks)
    def _answer_followup(self, text: str) -> str:
        low = text.lower().strip()
        rec = self._active
        if rec is not None:
            try:
                fresh = self.store.build(rec.path)
                if fresh is not None:
                    self._active = rec = fresh
            except Exception:
                pass
        # "this/that photo", "when was it taken" — answered from metadata
        # only; there's no vision model wired into chat to describe content.
        if self._active_photo is not None and re.search(
                r"\b(this|that|the)\s+(photo|picture|screenshot|pic|image)\b"
                r"|\bwhen\s+(was\s+)?(it|this|that)\s+(taken|captured)\b"
                r"|\bhow\s+(was\s+)?(it|this|that)\s+(taken|captured)\b",
                low):
            p = self._active_photo
            tag = _photo_source_label(p.source, verbose=True)
            msg = f"That photo was taken {p.when()}, captured {tag}"
            if p.trigger_text:
                msg += f" (triggered by \u201c{p.trigger_text}\u201d)"
            msg += (". I can't see what's actually in the image \u2014 no "
                    "vision model is connected to chat yet \u2014 but I can "
                    "tell you when or how anything was captured.")
            return msg
        # "what was said at 5:30" / "around 1:20"
        m = re.search(r"\b(?:at|around|near|by|@)\s*(\d{1,2}):([0-5]\d)\b", low)
        if rec is not None and m:
            secs = int(m.group(1)) * 60 + int(m.group(2))
            head = f"\U0001F4FC {rec.name}\n\n"
            if rec.segments:
                seg = iq.lookup_offset(rec, secs)
                if seg is not None:
                    spk = seg.get("speaker")
                    who = f"{spk}: " if spk else ""
                    return (head + f"Around {iq.fmt_offset(secs)} \u2014 "
                            + who + seg.get("text", "").strip())
                return (head + f"This recording is only {rec.length()} long, so "
                        f"there's nothing at {iq.fmt_offset(secs)}.")
            return (head + "This recording doesn't have timestamped segments, "
                    f"so I can't pin down exactly what was said at "
                    f"{iq.fmt_offset(secs)}.")
        # "when did we talk about X" / "where is X mentioned"
        if rec is not None and re.search(
                r"\b(when|what time|where|at what point)\b", low):
            topic = self._topic_from_question(text)
            if topic:
                hits = iq.find_topic_in_recording(topic, rec)
                if hits:
                    lines = [f"\U0001F4FC {rec.name} \u2014 \u201c{topic}\u201d "
                             "comes up here:"]
                    for start, spk, txt in hits[:4]:
                        when = (iq.fmt_offset(start) if start is not None
                                else "?")
                        who = f"{spk}: " if spk else ""
                        snippet = txt if len(txt) <= 160 else txt[:157] + "\u2026"
                        lines.append(f"  \u2022 {when} \u2014 {who}{snippet}")
                    if not rec.segments:
                        lines.append("\n(This recording has no per-line "
                                     "timestamps, so I can only show the lines.)")
                    return "\n".join(lines)
                return (f"I don't see \u201c{topic}\u201d mentioned in "
                        f"{rec.name}.")
        if (rec is not None and not rec.has_transcript
                and self._is_about_recording(low)):
            return (f"\U0001F4FC {rec.name} isn't transcribed yet, so I can't "
                    "answer from it. It's transcribing now \u2014 I'll post the "
                    "summary automatically when it's ready, or ask again in a "
                    "moment.")
        return self._ask_ollama(text)
    def _active_context_block(self) -> Optional[str]:
        if not self._active or not self._active.has_transcript:
            return None
        return (f"The user is asking about this recording:\n"
                f"name: {self._active.name}\n"
                f"recorded: {self._active.when()}  length: {self._active.length()}\n"
                f"TRANSCRIPT:\n{self._truncate(self._active.transcript, 7000)}")
    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + " \u2026[truncated]"
    # ── M7: memory recall ────────────────────────────────────────────────
    # ── UI-action intents (start camera / start audio recording) ─────────
    def _handle_action_intent(self, intent) -> None:
        """Switch tabs and trigger the appropriate button for action intents.
        Runs on the UI thread (no _start_bg) since it's pure Qt navigation —
        no Llama call, no I/O. The chat reply is appended synchronously."""
        kind = getattr(intent, "kind", "none")
        app = self._find_iris_app()
        if app is None:
            self._append_iris(
                "I couldn't navigate the tabs — try clicking the "
                "Stream or Audio tab manually.")
            return

        if kind == "action_start_video":
            self._do_action_start_video(app)
            return
        if kind == "action_start_audio":
            self._do_action_start_audio(app)
            return
        # Unknown action — fall back to a generic note.
        self._append_iris("I'm not sure what action you wanted me to take.")

    def _find_iris_app(self):
        """Walk up the parent chain to the IrisApp that owns the tab bar
        and the stack. Returns None if we can't find it (shouldn't happen
        at runtime, but the chat is defensive)."""
        w = self.parent()
        while w is not None:
            if hasattr(w, "tabbar") and hasattr(w, "stack"):
                return w
            w = w.parent()
        return None

    def _do_action_start_video(self, app) -> None:
        stream = getattr(app, "stream", None)
        if stream is None:
            self._append_iris(
                "I can't find the Stream tab. Try clicking it manually.")
            return
        # Tab indexes are fixed in IrisApp: chat=0, audio=1, location=2,
        # people=3, stream=4, photos=5.
        try:
            app.tabbar._select(4)
        except Exception as e:
            print(f"[chat-action] tab switch failed: {e}")

        already_listening = bool(getattr(stream, "listening", False))
        if not already_listening:
            try:
                stream._start_listening()
            except Exception as e:
                print(f"[chat-action] _start_listening failed: {e}")
                self._append_iris(
                    "Switched to the Stream tab, but I couldn't start the "
                    "receiver automatically. Click 'Start Listening' there.")
                return
            self._append_iris(
                "Switched to the Stream tab and started listening. The "
                "ESP32 camera records in fixed ~35-second clips and "
                "streams them automatically — you can't pick a custom "
                "duration, but whatever it records will show up here.")
        else:
            self._append_iris(
                "Switched to the Stream tab. The receiver was already "
                "listening — incoming clips will appear in the table.")

    def _do_action_start_audio(self, app) -> None:
        audio = getattr(app, "audio", None)
        if audio is None:
            self._append_iris(
                "I can't find the Audio tab. Try clicking it manually.")
            return
        try:
            app.tabbar._select(1)
        except Exception as e:
            print(f"[chat-action] tab switch failed: {e}")

        # AudioTab's record state lives on the controller, not on the tab.
        controller = getattr(audio, "controller", None)
        is_recording = False
        try:
            if controller is not None:
                # Different controller versions expose this differently.
                for attr in ("is_recording", "recording"):
                    v = getattr(controller, attr, None)
                    if callable(v):
                        is_recording = bool(v())
                        break
                    if isinstance(v, bool):
                        is_recording = v
                        break
        except Exception:
            is_recording = False

        if not is_recording:
            try:
                # Fire the exact same handler the Start Recording button
                # uses — guarantees identical behavior, no parallel paths.
                audio._on_record_clicked()
            except Exception as e:
                print(f"[chat-action] _on_record_clicked failed: {e}")
                self._append_iris(
                    "Switched to the Audio tab, but I couldn't start the "
                    "recorder automatically. Click 'Start Recording' there.")
                return
            self._append_iris(
                "Switched to the Audio tab and started recording. Talk "
                "naturally — when you click Stop, the diarizer will run "
                "and the conversation will be added to memory.")
        else:
            self._append_iris(
                "Switched to the Audio tab. A recording is already in "
                "progress — click Stop Recording when you're done.")

    def _known_people_names(self) -> list:
        """Pull the current list of known person names from the People
        registry. Used as the resolver vocabulary for classify_memory."""
        try:
            import iris_fusion
            fusion = iris_fusion.get_fusion()
            if fusion is None:
                return []
            return [p.name for p in fusion.list_people()
                    if p.name and not p.name.startswith("Unknown")]
        except Exception:
            return []

    def _handle_memory_query(self, intent, original_text: str) -> str:
        """Query ChromaDB based on the memory intent, build a context
        block of relevant records, and have Llama synthesize a natural
        paragraph answer. Runs on a background thread."""
        print(f"[chat-memory] handling kind={intent.kind!r} "
              f"person={intent.person_name!r} query={intent.query!r}")
        try:
            import iris_fusion
            import iris_memory
            fusion = iris_fusion.get_fusion()
            memory = iris_memory.get_memory() if iris_memory else None
        except Exception as e:
            print(f"[chat-memory] import failed: {e}")
            return f"(memory unavailable: {e})"
        if memory is None:
            return ("I couldn't load the memory module. Make sure "
                    "iris_memory.py is in the project folder.")
        # Force a re-initialisation if the store isn't ready. This
        # bypasses the 'init was attempted once' lockout in case the
        # first attempt failed silently.
        if not memory.is_ready():
            print(f"[chat-memory] memory not ready, forcing reinit...")
            ok = memory.force_reinit()
            print(f"[chat-memory] force_reinit returned {ok}, "
                  f"is_ready={memory.is_ready()}")
        if not memory.is_ready():
            return ("I couldn't open the memory store (ChromaDB). Check "
                    "the terminal for a [memory] error line — most "
                    "common cause is that the persist directory at "
                    "data/chroma is missing or not writable.")

        stats = memory.stats()
        n_total = stats.get("count", 0)
        print(f"[chat-memory] memory has {n_total} stored records")
        if n_total == 0:
            return ("Memory is initialised but contains no records yet. "
                    "Run backfill_memory.py --apply to import past "
                    "recordings, or record a new conversation.")

        kind = intent.kind
        records: list = []
        if kind == "memory_person":
            records = memory.search_combined(
                query=intent.query or "",
                person_name=intent.person_name,
                date_start=intent.date_start,
                date_end=intent.date_end,
                limit=10,
            )
        elif kind == "memory_semantic":
            records = memory.search_combined(
                query=intent.query,
                person_name=intent.person_name or "",
                date_start=intent.date_start,
                date_end=intent.date_end,
                limit=8,
            )
        elif kind == "memory_who":
            # Pull every record in the date window; aggregate speakers.
            start = intent.date_start if intent.date_start is not None else 0.0
            end   = intent.date_end   if intent.date_end   is not None \
                    else (time.time() + 1)
            records = memory.search_by_date(start, end, limit=200)
            print(f"[chat-memory] memory_who: searched "
                  f"[{start:.0f}, {end:.0f}], found {len(records)}")
            if records:
                return self._format_who_summary(records, intent)
            return self._memory_empty_reply(intent)

        print(f"[chat-memory] retrieved {len(records)} record(s)")
        if not records:
            return self._memory_empty_reply(intent)

        # Build a context block, then ask Llama to synthesize a paragraph.
        context = self._build_memory_context(records, intent)
        prompt = self._memory_llama_prompt(original_text, context, intent)
        return self._ask_llama_with_prompt(prompt)

    def _memory_empty_reply(self, intent) -> str:
        if intent.kind == "memory_person":
            who = intent.person_name or "that person"
            return (f"I don't have any recorded conversations with {who} yet. "
                    f"Memory builds up as IRIS captures audio — once a "
                    f"conversation with {who} is recorded, it will show up here.")
        if intent.kind == "memory_semantic":
            topic = intent.query or "that topic"
            return (f"I couldn't find a recorded conversation about "
                    f"{topic!r}. Either it hasn't happened yet, or it "
                    f"wasn't captured by IRIS.")
        return ("I don't have any memory records matching that. Recordings "
                "are stored as you have conversations with IRIS active.")

    @staticmethod
    def _format_who_summary(records, intent) -> str:
        """Aggregate speakers across a list of records and format a
        plain-prose answer for 'who did I talk to' queries."""
        from collections import Counter
        counts: Counter = Counter()
        for r in records:
            for n in (r.people_names or []):
                if n and not str(n).startswith("Unknown"):
                    counts[n] += 1
        if not counts:
            return ("I have some recordings in that window but couldn't "
                    "pull a list of named people from them.")
        top = counts.most_common()
        if len(top) == 1:
            name, n = top[0]
            return (f"You spoke with {name} in {n} recorded "
                    f"conversation{'s' if n != 1 else ''}.")
        parts = [f"{name} ({n})" for name, n in top]
        return ("In that window you spoke with: " + ", ".join(parts) + ".")

    @staticmethod
    def _build_memory_context(records, intent) -> str:
        """Stringify the top memory records as a context block for Llama."""
        lines: list = []
        for i, r in enumerate(records, 1):
            when = r.when_str()
            dur = r.duration_str()
            people = ", ".join(r.people_names) if r.people_names else "(unknown speakers)"
            summary = r.summary.strip() if r.summary else ""
            transcript = (r.transcript or "").strip()
            # Cap each transcript so the prompt doesn't blow past Llama's
            # context window. The summary, if present, is more useful.
            if summary:
                body = f"Summary: {summary}"
                if transcript:
                    snip = transcript[:600]
                    if len(transcript) > 600:
                        snip += " …"
                    body += f"\nTranscript excerpt: {snip}"
            else:
                snip = transcript[:1200]
                if len(transcript) > 1200:
                    snip += " …"
                body = f"Transcript: {snip}" if snip else "(no transcript)"
            location_bit = f" · {r.location}" if r.location else ""
            confirmed = " · confirmed" if r.confirmed else " · provisional"
            lines.append(
                f"[{i}] {when} · {dur} · {people}{location_bit}{confirmed}\n"
                f"{body}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _memory_llama_prompt(user_text: str, context: str, intent) -> str:
        """The augmented system+user prompt that asks Llama to answer
        from the retrieved memory records."""
        guidance = (
            "You are IRIS, a personal AI assistant with access to the "
            "user's conversation history. Answer the user's question "
            "using ONLY the memory records below. Write a natural, "
            "conversational paragraph — no bullet points, no headers. "
            "Refer to people by name. If the records don't fully answer "
            "the question, say so honestly."
        )
        return (
            f"{guidance}\n\n"
            f"The user asked: {user_text!r}\n\n"
            f"Relevant memory records:\n"
            f"---\n{context}\n---\n\n"
            f"Now answer the user's question in 2-4 sentences."
        )

    def _ask_llama_with_prompt(self, prompt: str) -> str:
        """Synchronous Llama call from a background thread. Falls back
        to a graceful error string if Ollama isn't available."""
        if self._client is None:
            return "(ollama not connected — can't synthesize an answer)"
        try:
            resp = self._client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}])
            return resp["message"]["content"].strip()
        except Exception as exc:
            return f"(ollama error while answering memory query: {exc})"

    @staticmethod
    def _is_video_question(low: str) -> bool:
        """True for a question about a saved ESP32 video clip — a video noun
        plus either a question word or a 'how many / who' cue. Deliberately
        does NOT fire on 'record a video' (that's an action, handled earlier)
        because those never contain 'was in / were in / how many / who'."""
        # "recording" is deliberately excluded — in this app it means an audio
        # recording, so it must keep routing to the transcript handlers.
        if not any(w in low for w in ("video", "clip", "footage")):
            return False
        cues = ("how many", "how much", "who ", "who's", "whos", "was in",
                "were in", "people in", "person in", "in the video",
                "in that video", "in the clip", "in that clip", "in the "
                "footage", "count", "what happened", "what's in", "whats in",
                "show me the video", "last video", "the video", "latest",
                "most recent", "newest")
        # A question mark alone also qualifies when a video noun is present.
        return low.strip().endswith("?") or any(c in low for c in cues)

    @staticmethod
    def _is_latest_video_question(low: str) -> bool:
        """True when the user wants specifically the single most recent clip
        ('latest video', 'give me the newest clip', etc). These must NEVER
        be answered by handing Llama a multi-clip list and hoping it reads
        the order correctly — smaller local models get this wrong even when
        the list is already sorted newest-first. Resolve it in Python."""
        if not any(w in low for w in ("video", "clip", "footage")):
            return False
        return any(c in low for c in (
            "latest video", "latest clip", "latest footage",
            "last video", "last clip", "most recent video",
            "most recent clip", "newest video", "newest clip"))

    _SCENE_FAST_CUES = (
        "describe", "wearing", "happened", "happening", "doing",
        "activity", "clothing", "objects", "what do you see",
        "what did you see", "what's in", "whats in", "what was in",
        "what were in", "tell me about", "look like",
    )
    _SCENE_CLASSIFY_MODEL = "llama3.2:1b"
    _SCENE_CLASSIFY_PROMPT = (
        "You classify a single chat message about saved video clips from a "
        "wearable camera. Reply with EXACTLY one word, nothing else:\n"
        "YES - the message asks for a VISUAL description of a clip's "
        "content: who is visible, what they look like, what they're "
        "wearing, what they're doing, objects, or the setting.\n"
        "NO - the message only asks for simple facts: how many clips "
        "exist, when something was recorded, how long a clip is, how many "
        "people were detected (a number), or a filename.\n"
        "Reply with exactly YES or NO, nothing else."
    )

    def _is_scene_description_question(self, low: str, text: str) -> bool:
        """True when the user wants a visual description of what's IN a
        clip — who's visible, clothing, objects, setting — not just facts
        (name/time/people-count) that describe_recent() already covers.
        This is the expensive path (a real LLaVA call on several frames),
        so it only fires on genuinely descriptive phrasing.

        Fast path: obvious keyword hit → True immediately, no extra call.
        Fallback: ambiguous phrasing ('what's in the LATEST video', 'what
        was the person DOING') that the keyword list doesn't catch gets
        classified by a cheap llama3.2:1b yes/no call instead of trying to
        keyword-match every possible English phrasing — that approach
        doesn't scale, this generalizes to wording we never hardcoded."""
        if not any(w in low for w in ("video", "clip", "footage")):
            return False
        if any(c in low for c in self._SCENE_FAST_CUES):
            return True
        if self._client is None:
            return False
        try:
            resp = self._client.chat(
                model=self._SCENE_CLASSIFY_MODEL,
                messages=[
                    {"role": "system", "content": self._SCENE_CLASSIFY_PROMPT},
                    {"role": "user", "content": text},
                ],
                options={"num_predict": 3},
            )
            answer = resp["message"]["content"].strip().upper()
            if "YES" in answer:
                result = True
            elif "NO" in answer:
                result = False
            else:
                # Model didn't follow the YES/NO instruction (small models
                # occasionally don't) — fail toward YES. Worst case is one
                # extra (cached) LLaVA call; worst case of failing NO is
                # silently giving the person a useless answer.
                result = True
            print(f"[video] scene classify: {text!r} -> {answer!r} "
                  f"({'YES' if result else 'NO'})")
            return result
        except Exception as e:
            print(f"[video] scene classify failed: {e}")
            return False

    _ORDINAL_WORDS = {
        "first": 0, "1st": 0,
        "second": 1, "2nd": 1,
        "third": 2, "3rd": 2,
        "fourth": 3, "4th": 3,
        "fifth": 4, "5th": 4,
    }

    def _resolve_target_clip(self, low: str):
        """Figure out which saved clip a video question is about.
        Priority: an explicit filename mentioned in the message → an
        ordinal ('the second video', '3rd clip' — position counted from
        MOST RECENT, matching how describe_recent() lists them) → 'oldest'
        / 'earliest' → default to the latest clip."""
        if self._videos is None:
            return None
        try:
            clips = self._videos.list_all(limit=10)
        except Exception:
            clips = []
        if not clips:
            return None
        for c in clips:
            stem = os.path.splitext(c.name)[0].lower()
            if stem and stem in low:
                return c
        for word, idx in self._ORDINAL_WORDS.items():
            if word in low:
                return clips[idx] if idx < len(clips) else clips[-1]
        if "oldest" in low or "earliest" in low:
            return clips[-1]
        return clips[0]

    def _answer_video_question(self, text: str) -> str:
        """Answer a question about saved video clips using their real analysis
        (people counts, recognised names, times). Guarantees the clip data is
        in context regardless of how the classifier routed things."""
        if self._client is None:
            return "(ollama not connected)"
        low = text.lower()

        # A visual "what happened / what were they wearing / what's in the
        # video" question — resolve which clip (explicit filename, ordinal
        # position like 'second video', or default to latest), run the
        # on-demand LLaVA scene description on it (slow — several
        # vision-model calls), and hand ONLY that description to Llama to
        # phrase as an answer.
        if self._videos is not None and self._is_scene_description_question(low, text):
            try:
                clip = self._resolve_target_clip(low)
            except Exception as e:
                print(f"[video] clip resolution failed: {e}")
                clip = None
            if clip is None:
                return "There are no saved video clips on disk yet."
            try:
                description = self._videos.describe(clip.path)
            except Exception as e:
                print(f"[video] describe() failed: {e}")
                description = ""
            if not description:
                return (f"I found {clip.name} ({clip.when()}, "
                        f"{clip.length()} long), but couldn't generate a "
                        "visual description right now — the vision model "
                        "may not be running (check Ollama has llava:7b "
                        "available).")
            vctx = (
                f"Visual description of the video clip {clip.name} "
                f"(recorded {clip.when()}, length {clip.length()}), "
                f"generated by looking at several frames spread across "
                f"the clip:\n{description}\n"
                "Answer the user's question using only this description. "
                "Do not invent details not mentioned in it.")
            messages = [{"role": "system", "content": self._system_prompt},
                        {"role": "system", "content": vctx}]
            messages.extend(self.history)
            try:
                resp = self._client.chat(model=OLLAMA_MODEL, messages=messages)
                return resp["message"]["content"].strip()
            except Exception as exc:
                return f"(ollama error: {exc})"

        # "Latest video" is resolved deterministically — never left for the
        # LLM to pick out of a list.
        if self._videos is not None and self._is_latest_video_question(low):
            try:
                clip = self._videos.latest()
            except Exception as e:
                print(f"[video] latest() failed: {e}")
                clip = None
            if clip is None:
                return "There are no saved video clips on disk yet."
            try:
                clip = self._videos.analyze(clip.path)
            except Exception:
                pass
            vctx = (
                "The single most recent saved ESP32 video clip is:\n"
                f"filename: {clip.name}\n"
                f"recorded: {clip.when()}\n"
                f"length: {clip.length()}\n"
                f"people detected: {clip.people_summary()}\n"
                "This IS the latest clip on disk right now — do not "
                "suggest or name any other clip.")
            messages = [{"role": "system", "content": self._system_prompt},
                        {"role": "system", "content": vctx}]
            messages.extend(self.history)
            try:
                resp = self._client.chat(model=OLLAMA_MODEL, messages=messages)
                return resp["message"]["content"].strip()
            except Exception as exc:
                return f"(ollama error: {exc})"

        vctx = ""
        if self._videos is not None:
            try:
                vctx = self._videos.describe_recent(limit=8)
            except Exception as e:
                print(f"[video] describe_recent failed: {e}")
        messages = [{"role": "system", "content": self._system_prompt}]
        if vctx:
            messages.append({"role": "system", "content": vctx})
        else:
            messages.append({"role": "system", "content":
                "No saved ESP32 video clips were found on disk yet. Tell the "
                "user there are no analysed video clips available to answer "
                "from."})
        messages.extend(self.history)
        try:
            resp = self._client.chat(model=OLLAMA_MODEL, messages=messages)
            return resp["message"]["content"].strip()
        except Exception as exc:
            return f"(ollama error: {exc})"

    def _ask_ollama(self, text: str) -> str:
            """General-purpose Ollama call. Injects summaries from the most
            recent transcribed recordings into context so questions like
            'what did we just discuss' work naturally."""
            if self._client is None:
                return "(ollama not connected)"

            messages = [{"role": "system", "content": self._system_prompt}]

            # Build context from summaries only (much shorter than full
            # transcripts — lets us include several recent recordings at once).
            ctx = self._active_context_block()   # uses full transcript if active
            if not ctx:
                try:
                    recs = sorted(self._all_recordings(),
                                key=lambda r: r.mtime, reverse=True)
                    # Grab the 5 most recent that have either a summary or transcript
                    recent = [r for r in recs
                            if r.summary.strip() or r.has_transcript][:5]
                    if recent:
                        lines = ["Recent recording summaries the user may be asking about:\n"]
                        for r in recent:
                            # Prefer the summary; fall back to a short transcript snippet
                            body = r.summary.strip() if r.summary.strip() \
                                else self._truncate(r.transcript, 300)
                            lines.append(
                                f"• {r.name} | {r.when()} | {r.length()}\n"
                                f"  {body}\n"
                            )
                        ctx = "\n".join(lines)
                except Exception:
                    pass

            if ctx:
                messages.append({"role": "system", "content": ctx})

            # Video-clip awareness: list the ESP32's recent saved clips and
            # how many people were in each, so questions like "how many people
            # were in the video" are answered from real data instead of the
            # old "I can't access video recordings" fallback.
            if self._videos is not None:
                try:
                    vctx = self._videos.describe_recent(limit=5)
                    if vctx:
                        messages.append({"role": "system", "content": vctx})
                except Exception:
                    pass

            messages.extend(self.history)
            try:
                resp = self._client.chat(model=OLLAMA_MODEL, messages=messages)
                return resp["message"]["content"].strip()
            except Exception as exc:
                return f"(ollama error: {exc})"

# Placeholder tabs (glass)
# ─────────────────────────────────────────────────────────────────────────────
class PlaceholderTab(QWidget):
    def __init__(self, parent, title: str, items: list[str], milestone: str):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        card = GlassFrame(self, radius=18, blur=30, dy=8)
        card.setMaximumWidth(460)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 24, 28, 26)
        cl.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet(
            f"color:{TEXT_PRIMARY}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:18px; font-weight:700;")
        cl.addWidget(t)
        ms = QLabel(f"arrives in {milestone}")
        ms.setStyleSheet(
            f"color:{ACCENT}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:11px; padding-bottom:10px;")
        cl.addWidget(ms)
        for item in items:
            it = QLabel(f"\u00b7  {item}")
            it.setStyleSheet(
                f"color:{TEXT_MUTED}; background:transparent; border:none;"
                f"font-family:'{FONT_SANS}'; font-size:11px; padding:1px 0;")
            cl.addWidget(it)
        wrap = QHBoxLayout()
        wrap.addStretch(1)
        wrap.addWidget(card)
        wrap.addStretch(1)
        outer.addLayout(wrap)
        outer.addStretch(2)
# ─────────────────────────────────────────────────────────────────────────────
# People tab — M5 registry UI.
# ─────────────────────────────────────────────────────────────────────────────
class _ProfileDialog(QDialog):
    """Shared base for Add/Edit profile dialogs — holds the form widgets
    so both can reuse the same layout and validation."""

    def __init__(self, parent, *, title: str, fusion,
                 initial: Optional[dict] = None,
                 allow_face_pick: bool = True,
                 allow_is_self: bool = True):
        super().__init__(parent)
        self.fusion = fusion
        self._face_path: Optional[str] = None
        self.setWindowTitle(title)
        self.setMinimumWidth(420)
        self.setStyleSheet(
            f"QDialog {{ background: {BG_TOP}; color: {TEXT_PRIMARY};"
            f" font-family: '{FONT_SANS}'; }}"
            f"QLabel {{ color: {TEXT_MUTED}; background: transparent;"
            f" font-size: 11px; }}"
            f"QLineEdit, QTextEdit {{ background: rgba(255,255,255,0.05);"
            f" color: {TEXT_PRIMARY}; border: 1px solid rgba(255,255,255,0.08);"
            f" border-radius: 6px; padding: 6px 8px;"
            f" font-family: '{FONT_SANS}'; font-size: 12px; }}"
            f"QCheckBox {{ color: {TEXT_PRIMARY}; background: transparent;"
            f" font-family: '{FONT_SANS}'; font-size: 11px; }}")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 14)
        outer.setSpacing(10)

        from PyQt6.QtWidgets import (QFormLayout, QLineEdit, QCheckBox,
                                     QDialogButtonBox)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.ed_name = QLineEdit()
        self.ed_title = QLineEdit()
        self.ed_company = QLineEdit()
        self.ed_relationship = QLineEdit()
        self.ed_role_note = QLineEdit()
        for ed, ph in (
            (self.ed_name, "e.g. Humza Malik"),
            (self.ed_title, "e.g. ESP32 firmware engineer"),
            (self.ed_company, "e.g. IRIS team"),
            (self.ed_relationship, "e.g. teammate, friend, family"),
            (self.ed_role_note, "free-form note"),
        ):
            ed.setPlaceholderText(ph)
        form.addRow("Name", self.ed_name)
        form.addRow("Title", self.ed_title)
        form.addRow("Company", self.ed_company)
        form.addRow("Relationship", self.ed_relationship)
        form.addRow("Role note", self.ed_role_note)

        if allow_is_self:
            self.chk_is_self = QCheckBox("This is me (mark as self profile)")
            form.addRow("", self.chk_is_self)
        else:
            self.chk_is_self = None

        if allow_face_pick:
            face_row = QHBoxLayout()
            self.lbl_face = QLabel("No face image selected")
            self.lbl_face.setStyleSheet(
                f"color:{TEXT_DIM}; background:transparent;"
                f"font-family:'{FONT_MONO}','Consolas',monospace; font-size:10px;")
            btn_pick = _audio_btn("Choose face image\u2026",
                                  self._pick_face_image, height=28)
            face_row.addWidget(self.lbl_face, 1)
            face_row.addWidget(btn_pick)
            form.addRow("Face photo", _wrap_in_widget(face_row))
        else:
            self.lbl_face = None

        outer.addLayout(form)

        # Seed initial values if editing.
        if initial:
            self.ed_name.setText(str(initial.get("name", "")))
            self.ed_title.setText(str(initial.get("title", "")))
            self.ed_company.setText(str(initial.get("company", "")))
            self.ed_relationship.setText(str(initial.get("relationship", "")))
            self.ed_role_note.setText(str(initial.get("role_note", "")))
            if self.chk_is_self is not None:
                self.chk_is_self.setChecked(bool(initial.get("is_self", False)))

        # Standard OK/Cancel.
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.setStyleSheet(
            f"QPushButton {{ background: rgba(255,255,255,0.06);"
            f" color: {TEXT_PRIMARY}; border: 1px solid rgba(255,255,255,0.10);"
            f" border-radius: 6px; padding: 6px 14px;"
            f" font-family: '{FONT_SANS}'; font-size: 11px; }}"
            f"QPushButton:default {{"
            f" background: rgba({_rgb(ACCENT)},0.22);"
            f" color: {ACCENT};"
            f" border: 1px solid rgba({_rgb(ACCENT)},0.45); }}")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _pick_face_image(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick face image", "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)")
        if path:
            self._face_path = path
            if self.lbl_face is not None:
                self.lbl_face.setText(os.path.basename(path))

    def values(self) -> dict:
        return {
            "name": self.ed_name.text().strip(),
            "title": self.ed_title.text().strip(),
            "company": self.ed_company.text().strip(),
            "relationship": self.ed_relationship.text().strip(),
            "role_note": self.ed_role_note.text().strip(),
            "is_self": (self.chk_is_self.isChecked()
                        if self.chk_is_self is not None else False),
            "face_image_path": self._face_path,
        }


def _wrap_in_widget(layout) -> QWidget:
    """Wrap a QLayout in a QWidget so QFormLayout.addRow() accepts it."""
    w = QWidget()
    w.setStyleSheet("background: transparent;")
    w.setLayout(layout)
    return w


class _ConversationsDialog(QDialog):
    """Shows every conversation row for a given person — date, duration,
    speaker count, confirmed / provisional, and the WAV / clip paths."""

    def __init__(self, parent, *, fusion, person):
        super().__init__(parent)
        self.setWindowTitle(f"Conversations with {person.name}")
        self.setMinimumSize(640, 420)
        self.setStyleSheet(
            f"QDialog {{ background: {BG_TOP}; color: {TEXT_PRIMARY}; }}"
            f"QLabel {{ color: {TEXT_MUTED}; background: transparent;"
            f" font-family: '{FONT_SANS}'; font-size: 11px; }}")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(8)
        head = QLabel(
            f"All conversations with <b>{person.name}</b>"
            f"  \u00b7  {person.times_seen} total encounters")
        head.setStyleSheet(
            f"color:{TEXT_PRIMARY}; background:transparent;"
            f"font-family:'{FONT_SANS}'; font-size:13px;")
        head.setTextFormat(Qt.TextFormat.RichText)
        outer.addWidget(head)

        table = QTableWidget(0, 5, self)
        table.setHorizontalHeaderLabels(
            ["date", "duration", "speakers", "status", "files"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setShowGrid(False)
        table.setStyleSheet(
            f"QTableWidget {{ background: transparent;"
            f" color: {TEXT_PRIMARY}; border: none;"
            f" font-family: '{FONT_MONO}','Consolas',monospace; font-size: 11px; }}"
            f"QHeaderView::section {{ background: rgba(255,255,255,0.03);"
            f" color: {TEXT_MUTED}; border: none;"
            f" border-bottom: 1px solid rgba(255,255,255,0.06);"
            f" padding: 6px 8px;"
            f" font-family: '{FONT_SANS}'; font-size: 11px; }}")
        hdr = table.horizontalHeader()
        for i, mode in enumerate((QHeaderView.ResizeMode.Fixed,
                                  QHeaderView.ResizeMode.Fixed,
                                  QHeaderView.ResizeMode.Fixed,
                                  QHeaderView.ResizeMode.Fixed,
                                  QHeaderView.ResizeMode.Stretch)):
            hdr.setSectionResizeMode(i, mode)
        for i, w in enumerate((150, 90, 80, 100)):
            table.setColumnWidth(i, w)

        try:
            convs = fusion.list_conversations_for(person.id)
        except Exception as e:
            print(f"[people-tab] list_conversations_for failed: {e}")
            convs = []
        for c in convs:
            row = table.rowCount()
            table.insertRow(row)
            dt = "\u2014"
            try:
                from datetime import datetime as _dt
                ts = float(c.get("session_start", 0) or 0)
                if ts > 0:
                    dt = _dt.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
            dur = float(c.get("duration_seconds", 0) or 0)
            dur_s = f"{int(dur // 60)}m {int(dur % 60)}s" if dur > 0 else "\u2014"
            sp = int(c.get("speaker_count", 0) or 0)
            sp_s = str(sp) if sp > 0 else "\u2014"
            confirmed = bool(int(c.get("confirmed", 0) or 0))
            status = "confirmed" if confirmed else "provisional"
            wav = os.path.basename(c.get("wav_path", "") or "")
            clip = os.path.basename(c.get("clip_path", "") or "")
            files = "  ".join(s for s in (wav, clip) if s) or "\u2014"
            for col, text in enumerate((dt, dur_s, sp_s, status, files)):
                item = QTableWidgetItem(text)
                if col == 3:
                    fg = BADGE_FACE_FG if confirmed else TEXT_DIM
                    item.setForeground(QColor(fg))
                table.setItem(row, col, item)
        if table.rowCount() == 0:
            empty = QLabel("No conversations recorded yet.")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(
                f"color:{TEXT_DIM}; background:transparent;"
                f"font-family:'{FONT_SANS}'; font-size:12px; padding: 30px;")
            outer.addWidget(empty, 1)
        else:
            outer.addWidget(table, 1)

        from PyQt6.QtWidgets import QDialogButtonBox
        close_btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_btn.setStyleSheet(
            f"QPushButton {{ background: rgba(255,255,255,0.06);"
            f" color: {TEXT_PRIMARY}; border: 1px solid rgba(255,255,255,0.10);"
            f" border-radius: 6px; padding: 6px 14px;"
            f" font-family: '{FONT_SANS}'; font-size: 11px; }}")
        close_btn.rejected.connect(self.reject)
        outer.addWidget(close_btn)


class PeopleTab(QWidget):
    _voice_ingested_signal  = pyqtSignal(object)
    _faces_processed_signal = pyqtSignal(object)
    _MAX_ACTIVITY_ROWS = 25
    _COL_PHOTO, _COL_NAME, _COL_ROLE, _COL_FACES, _COL_VOICES, \
        _COL_SEEN, _COL_LAST = range(7)
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.fusion = None
        if iris_fusion is not None:
            try:
                self.fusion = iris_fusion.get_fusion()
            except Exception as e:
                print(f"[people-tab] could not create fusion: {e}")
        self._build()
        self._voice_ingested_signal.connect(self._on_voice_ingested_ui)
        self._faces_processed_signal.connect(self._on_faces_processed_ui)
        if self.fusion is not None:
            try:
                self.fusion.on_voice_ingested  = self._voice_ingested_signal.emit
                self.fusion.on_faces_processed = self._faces_processed_signal.emit
                self.fusion.start(controller=controller)
            except Exception as e:
                print(f"[people-tab] fusion.start() failed: {e}")
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_header_only)
        self._refresh_timer.start(2000)
        self.refresh()
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = GlassFrame(self, radius=16, blur=24, dy=6, shadow_alpha=120,
                           top="rgba(255,255,255,0.06)",
                           mid="rgba(255,255,255,0.035)",
                           bot="rgba(255,255,255,0.02)",
                           border=GLASS_BORDER_SOFT)
        outer.addWidget(frame)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)
        head = QHBoxLayout()
        title = QLabel("people")
        title.setStyleSheet(f"color:{TEXT_PRIMARY}; background:transparent;"
                            f"border:none; font-family:'{FONT_SANS}';"
                            "font-size:15px; font-weight:700;")
        head.addWidget(title)
        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet(
            f"color:{TEXT_DIM}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:11px; padding-left:10px;")
        head.addWidget(self.lbl_stats)
        head.addStretch(1)
        self.pill_deepface = Pill(frame, "DeepFace: \u2014", TEXT_DIM)
        head.addWidget(self.pill_deepface)
        head.addWidget(_audio_btn("\u002b Add Person",
                                  self._add_person_clicked, height=30,
                                  accent=_rgb(ACCENT), fg=ACCENT))
        head.addWidget(_audio_btn("\u21bb Refresh", self.refresh, height=30,
                                  accent=_rgb(ACCENT), fg=ACCENT))
        head.addWidget(_audio_btn("\U0001F4C2 DB folder",
                                  self._open_db_folder, height=30))
        head.addWidget(_audio_btn("\U0001F5D1 Reset all",
                                  self._reset_all_people, height=30,
                                  accent=_rgb(COLOR_DANGER), fg=COLOR_DANGER))
        lay.addLayout(head)

        # Self profile card — only visible when an is_self row exists.
        self.self_card = QFrame(frame)
        self.self_card.setStyleSheet(
            f"QFrame {{ background: rgba({_rgb(ACCENT)},0.06);"
            f" border: 1px solid rgba({_rgb(ACCENT)},0.20);"
            f" border-radius: 10px; }}")
        sc_lay = QHBoxLayout(self.self_card)
        sc_lay.setContentsMargins(12, 8, 12, 8)
        sc_lay.setSpacing(10)
        self.self_label = QLabel("\u2014")
        self.self_label.setStyleSheet(
            f"color:{TEXT_PRIMARY}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:12px;")
        self.self_label.setTextFormat(Qt.TextFormat.RichText)
        sc_lay.addWidget(self.self_label, 1)
        sc_lay.addWidget(_audio_btn("Edit self profile",
                                    self._edit_self_clicked, height=26))
        self.self_card.setVisible(False)
        lay.addWidget(self.self_card)

        self.table = QTableWidget(0, 7, frame)
        self.table.setHorizontalHeaderLabels(
            ["", "name", "role", "faces", "voices", "seen", "last seen"])
        self.table.verticalHeader().setVisible(False)
        # Tall enough to show 40×40 face thumbnails comfortably.
        self.table.verticalHeader().setDefaultSectionSize(50)
        self.table.setIconSize(QSize(40, 40))
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(False)
        self.table.setStyleSheet(
            "QTableWidget {"
            f"background: transparent; border: none; color: {TEXT_PRIMARY};"
            f"font-family: '{FONT_MONO}','Consolas',monospace; font-size: 11px;"
            "gridline-color: rgba(255,255,255,0.05);}"
            "QTableWidget::item {padding: 6px 6px; border: none; background: transparent;}"
            "QTableWidget::item:selected {"
            f"background: rgba({_rgb(ACCENT)},0.15); color: {TEXT_PRIMARY};}}"
            "QHeaderView {background: transparent; border: none;}"
            "QHeaderView::section {"
            "background: rgba(255,255,255,0.03);"
            f"color: {TEXT_MUTED}; border: none;"
            "border-bottom: 1px solid rgba(255,255,255,0.06);"
            f"font-family: '{FONT_SANS}'; font-size: 11px; padding: 8px 8px;}}"
            "QScrollBar:vertical {width: 8px; background: transparent;}"
            "QScrollBar::handle:vertical {background: rgba(255,255,255,0.14);"
            "border-radius: 4px;}")
        hdr = self.table.horizontalHeader()
        # Photo column — fixed narrow width just for the thumbnail.
        hdr.setSectionResizeMode(self._COL_PHOTO, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(self._COL_PHOTO, 56)
        hdr.setSectionResizeMode(self._COL_NAME, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(self._COL_ROLE, QHeaderView.ResizeMode.Stretch)
        for c in (self._COL_FACES, self._COL_VOICES, self._COL_SEEN):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(c, 70)
        hdr.setSectionResizeMode(self._COL_LAST, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(self._COL_LAST, 140)
        self.table.itemSelectionChanged.connect(self._update_action_buttons)
        lay.addWidget(self.table, 1)
        actions = QHBoxLayout()
        self.btn_profile = _audio_btn("\U0001F464  Profile",
                                      self._edit_profile_selected, height=30,
                                      accent=_rgb(ACCENT), fg=ACCENT)
        self.btn_convos = _audio_btn("\U0001F4AC  Conversations",
                                     self._show_conversations_selected,
                                     height=30)
        self.btn_rename = _audio_btn("\u270e  Rename", self._rename_selected, height=30)
        self.btn_role   = _audio_btn("\U0001F4DD  Edit role", self._edit_role_selected, height=30)
        self.btn_merge  = _audio_btn("\u2702  Merge into\u2026", self._merge_selected,
                                     height=30, accent=_rgb(BADGE_VOICE_FG), fg=BADGE_VOICE_FG)
        self.btn_delete = _audio_btn("\U0001F5D1  Delete", self._delete_selected,
                                     height=30, accent=_rgb(COLOR_DANGER), fg=COLOR_DANGER)
        for b in (self.btn_profile, self.btn_convos, self.btn_rename,
                  self.btn_role, self.btn_merge, self.btn_delete):
            actions.addWidget(b)
        actions.addStretch(1)
        lay.addLayout(actions)
        self._update_action_buttons()

        # Pending prompts panel — surfaces things the system wants the
        # user to look at: new face needs naming, mentioned-name not in
        # registry, low-confidence match needs confirming, etc.
        self._build_prompts_panel(frame, lay)

        feed_label = QLabel("activity")
        feed_label.setStyleSheet(
            f"color:{TEXT_DIM}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:11px; padding-top:6px;")
        lay.addWidget(feed_label)
        feed_scroll = QScrollArea(frame)
        feed_scroll.setWidgetResizable(True)
        feed_scroll.setFrameShape(QFrame.Shape.NoFrame)
        feed_scroll.setFixedHeight(110)
        feed_scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:8px;background:transparent;}"
            "QScrollBar::handle:vertical{background:rgba(255,255,255,0.14);border-radius:4px;}")
        self._feed_holder = QWidget()
        self._feed_holder.setStyleSheet("background: transparent;")
        self._feed_layout = QVBoxLayout(self._feed_holder)
        self._feed_layout.setContentsMargins(2, 2, 2, 2)
        self._feed_layout.setSpacing(2)
        self._feed_layout.addStretch(1)
        feed_scroll.setWidget(self._feed_holder)
        lay.addWidget(feed_scroll)
        if self.fusion is None:
            self._show_unavailable_note(lay)
    def _show_unavailable_note(self, lay) -> None:
        for w in (self.table, self.btn_profile, self.btn_convos,
                  self.btn_rename, self.btn_role,
                  self.btn_merge, self.btn_delete):
            w.setVisible(False)
        note = QLabel(
            "people registry unavailable \u2014 iris_fusion, iris_people, "
            "iris_faces, and iris_voices must all be present.")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet(
            f"color:{TEXT_MUTED}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:12px; padding: 30px;")
        lay.insertWidget(1, note)
    def refresh(self) -> None:
        if self.fusion is None:
            return
        try:
            people = self.fusion.list_people()
        except Exception as e:
            print(f"[people-tab] list_people failed: {e}")
            people = []
        prev_id = self._selected_person_id()
        self.table.setRowCount(0)
        for p in people:
            row = self.table.rowCount()
            self.table.insertRow(row)
            # Photo cell — face_ref.jpg from the person's folder if present.
            photo_item = QTableWidgetItem("")
            photo_item.setFlags(photo_item.flags()
                                & ~Qt.ItemFlag.ItemIsEditable)
            photo_path = ""
            if p.folder_path:
                candidate = os.path.join(p.folder_path, "face_ref.jpg")
                if os.path.exists(candidate):
                    photo_path = candidate
            if photo_path:
                try:
                    pix = QPixmap(photo_path)
                    if not pix.isNull():
                        pix = pix.scaled(
                            40, 40,
                            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                            Qt.TransformationMode.SmoothTransformation)
                        photo_item.setData(
                            Qt.ItemDataRole.DecorationRole, pix)
                except Exception:
                    pass
            photo_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, self._COL_PHOTO, photo_item)

            cells = [p.name, p.role_note or "\u2014", str(p.face_count),
                     str(p.voice_count), str(p.times_seen), p.when_last()]
            for offset, text in enumerate(cells):
                col = offset + 1  # cells start at _COL_NAME (which is 1)
                item = QTableWidgetItem(text)
                if col == self._COL_NAME:
                    item.setData(Qt.ItemDataRole.UserRole, int(p.id))
                if col in (self._COL_FACES, self._COL_VOICES, self._COL_SEEN):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter
                                         | Qt.AlignmentFlag.AlignVCenter)
                if col == self._COL_LAST:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                         | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col, item)
            if prev_id is not None and p.id == prev_id:
                self.table.selectRow(row)
        self._refresh_header_only()
        self._update_action_buttons()
    def _refresh_header_only(self) -> None:
        if self.fusion is None:
            self.lbl_stats.setText("")
            return
        try:
            s = self.fusion.stats()
        except Exception:
            return
        n = s.get("people", 0)
        f = s.get("face_embeddings", 0)
        v = s.get("voice_embeddings", 0)
        self.lbl_stats.setText(
            f"{n} known \u00b7 {f} face emb \u00b7 {v} voice emb")
        ready = bool(s.get("face_pipeline_ready", False))
        if ready:
            self.pill_deepface.setText("DeepFace: ready")
            self._restyle_pill(self.pill_deepface, BADGE_FACE_FG)
        else:
            self.pill_deepface.setText("DeepFace: loading\u2026")
            self._restyle_pill(self.pill_deepface, TEXT_DIM)
        # Refresh self card + prompts panel on the same cadence.
        self._refresh_self_card()
        self._refresh_prompts_panel()

    # ── self profile card ────────────────────────────────────────────────
    def _refresh_self_card(self) -> None:
        if self.fusion is None:
            return
        try:
            me = self.fusion.get_self()
        except Exception:
            me = None
        if me is None:
            self.self_card.setVisible(False)
            return
        bits: list[str] = [f"<b>{me.name}</b>"]
        for v in (me.title, me.company, me.relationship):
            if v:
                bits.append(v)
        bits.append(f"{me.face_count} face \u00b7 {me.voice_count} voice")
        self.self_label.setText(
            "<span style='color:{0};'>This is you</span> &nbsp;&nbsp; ".format(ACCENT)
            + " &nbsp;\u00b7&nbsp; ".join(bits))
        self.self_card.setVisible(True)

    # ── pending prompts panel ────────────────────────────────────────────
    def _build_prompts_panel(self, frame, lay) -> None:
        """A collapsible card under the action buttons that lists open
        pending_prompts rows with action buttons (Add / Confirm / Skip /
        Don't ask again). Re-rendered on every tab refresh."""
        self.prompts_card = QFrame(frame)
        self.prompts_card.setStyleSheet(
            f"QFrame {{ background: rgba(255,255,255,0.025);"
            f" border: 1px solid rgba(255,255,255,0.06);"
            f" border-radius: 10px; }}")
        pc_lay = QVBoxLayout(self.prompts_card)
        pc_lay.setContentsMargins(12, 8, 12, 8)
        pc_lay.setSpacing(6)
        head = QHBoxLayout()
        self.prompts_title = QLabel("attention \u00b7 0 open")
        self.prompts_title.setStyleSheet(
            f"color:{TEXT_PRIMARY}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:12px; font-weight:600;")
        head.addWidget(self.prompts_title)
        head.addStretch(1)
        pc_lay.addLayout(head)
        self._prompts_list = QVBoxLayout()
        self._prompts_list.setContentsMargins(0, 0, 0, 0)
        self._prompts_list.setSpacing(4)
        pc_lay.addLayout(self._prompts_list)
        self.prompts_card.setVisible(False)
        lay.addWidget(self.prompts_card)

    def _refresh_prompts_panel(self) -> None:
        if self.fusion is None:
            return
        try:
            prompts = self.fusion.list_pending_prompts()
        except Exception:
            prompts = []
        # Wipe existing rows.
        while self._prompts_list.count():
            item = self._prompts_list.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.deleteLater()
        if not prompts:
            self.prompts_title.setText("attention \u00b7 0 open")
            self.prompts_card.setVisible(False)
            return
        self.prompts_title.setText(f"attention \u00b7 {len(prompts)} open")
        self.prompts_card.setVisible(True)
        # Render at most ~5 visible at a time to stay scannable.
        for pr in prompts[:5]:
            self._prompts_list.addWidget(self._build_prompt_row(pr))
        if len(prompts) > 5:
            more = QLabel(f"+ {len(prompts) - 5} more queued\u2026")
            more.setStyleSheet(
                f"color:{TEXT_DIM}; background:transparent; border:none;"
                f"font-family:'{FONT_SANS}'; font-size:10px; padding: 2px 4px;")
            self._prompts_list.addWidget(more)

    def _build_prompt_row(self, pr) -> QWidget:
        row = QFrame()
        row.setStyleSheet(
            f"QFrame {{ background: rgba(0,0,0,0.18);"
            f" border: 1px solid rgba(255,255,255,0.04);"
            f" border-radius: 8px; }}")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(10, 6, 8, 6)
        rl.setSpacing(8)
        msg, primary_label, primary_handler = self._prompt_description(pr)
        text = QLabel(msg)
        text.setStyleSheet(
            f"color:{TEXT_PRIMARY}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:11px;")
        text.setTextFormat(Qt.TextFormat.RichText)
        text.setWordWrap(True)
        rl.addWidget(text, 1)
        if primary_label and primary_handler is not None:
            btn = _audio_btn(primary_label, primary_handler, height=26,
                             accent=_rgb(ACCENT), fg=ACCENT)
            rl.addWidget(btn)
        btn_skip = _audio_btn("Skip",
                              lambda _checked=False, pid=pr.id:
                              self._dismiss_prompt(pid, acted_on=False),
                              height=26)
        rl.addWidget(btn_skip)
        return row

    def _prompt_description(self, pr):
        """Return (rich_text_message, primary_button_label, primary_handler)
        for a given PendingPrompt. Handlers close over the prompt id so we
        can dismiss after acting."""
        payload = pr.payload or {}
        pid = int(pr.id)
        if pr.type == "name_mentioned":
            name = payload.get("mentioned_name", "someone")
            msg = (f"<b>{name}</b> was mentioned but isn't in your registry. "
                   f"Add them as a person?")
            handler = lambda _checked=False, n=name, p=pid: \
                self._handle_name_mention(p, n)
            return msg, "Add\u2026", handler
        if pr.type == "long_unknown":
            name = payload.get("name", "Unknown")
            dur = float(payload.get("duration_seconds", 0))
            mins = f"{int(dur // 60)}m" if dur >= 60 else f"{int(dur)}s"
            reason = payload.get("reason", "")
            if reason == "dominant_unknown_speaker":
                msg = (f"<b>{name}</b> dominated a {mins} conversation "
                       f"({int(float(payload.get('dominant_share',0))*100)}%"
                       f" of the talk). Identify them?")
            else:
                msg = (f"<b>{name}</b> spoke in a {mins} conversation with "
                       f"{int(payload.get('speaker_count',0))} speakers. "
                       f"Identify them?")
            handler = lambda _checked=False, p=pid, person_id=pr.person_id: \
                self._handle_identify_unknown(p, person_id)
            return msg, "Identify\u2026", handler
        if pr.type == "confirm_identity":
            name = payload.get("name", "this person")
            sim = payload.get("similarity",
                              payload.get("face_similarity", 0))
            reason = payload.get("reason", "")
            pct = f"{int(float(sim) * 100)}%" if sim else ""
            if reason == "voice_video_mismatch":
                msg = (f"Voice said <b>{name}</b>, but the video doesn't "
                       f"confirm it (face match {pct}). Is this them?")
            else:
                msg = (f"Borderline match for <b>{name}</b> "
                       f"({pct}). Is this them?")
            handler = lambda _checked=False, p=pid, person_id=pr.person_id: \
                self._handle_confirm_identity(p, person_id, True)
            return msg, "Yes, it's them", handler
        # Unknown prompt type — show raw type so we can debug.
        return f"{pr.type}: {payload}", "", None

    def _handle_name_mention(self, prompt_id: int, name: str) -> None:
        """Open the Add Person dialog pre-filled with the mentioned name."""
        if self.fusion is None:
            return
        dlg = _ProfileDialog(self, title=f"Add {name}",
                             fusion=self.fusion,
                             initial={"name": name},
                             allow_face_pick=True, allow_is_self=False)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        v = dlg.values()
        if not v["name"]:
            return
        try:
            self.fusion.add_person_manual(**{k: v[k] for k in (
                "name", "title", "company", "relationship",
                "role_note", "face_image_path")})
        except Exception as e:
            QMessageBox.warning(self, "Add failed", str(e))
            return
        self.fusion.dismiss_prompt(prompt_id, acted_on=True)
        self._append_activity(f"added {v['name']}")
        self.refresh()

    def _handle_identify_unknown(self, prompt_id: int,
                                 person_id: int) -> None:
        """For long_unknown prompts — rename the Unknown N row in place
        and optionally fill in the rest of the profile."""
        if self.fusion is None:
            return
        person = self.fusion.get_person(person_id)
        if person is None:
            return
        dlg = _ProfileDialog(self,
                             title=f"Identify {person.name}",
                             fusion=self.fusion,
                             initial={
                                 "name": "",
                                 "title": person.title,
                                 "company": person.company,
                                 "relationship": person.relationship,
                                 "role_note": person.role_note,
                                 "is_self": person.is_self,
                             },
                             allow_face_pick=False, allow_is_self=True)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        v = dlg.values()
        if not v["name"]:
            return
        try:
            self.fusion.update_profile(
                person_id,
                name=v["name"], title=v["title"],
                company=v["company"], relationship=v["relationship"],
                role_note=v["role_note"], is_self=v["is_self"])
        except Exception as e:
            QMessageBox.warning(self, "Update failed", str(e))
            return
        self.fusion.dismiss_prompt(prompt_id, acted_on=True)
        self._append_activity(f"identified \u2192 {v['name']}")
        self.refresh()

    def _handle_confirm_identity(self, prompt_id: int,
                                 person_id: int, yes: bool) -> None:
        """For confirm_identity prompts — yes = leave row as-is, dismiss
        as acted_on. The matching/reinforcement already happened; the
        prompt is just a flag for the user to look at."""
        if self.fusion is None:
            return
        self.fusion.dismiss_prompt(prompt_id, acted_on=yes)
        self._append_activity("confirmed identity" if yes else
                              "skipped confirmation")
        self.refresh()

    def _dismiss_prompt(self, prompt_id: int,
                        *, acted_on: bool) -> None:
        if self.fusion is None:
            return
        self.fusion.dismiss_prompt(prompt_id, acted_on=acted_on)
        self._refresh_prompts_panel()

    # ── + Add Person ─────────────────────────────────────────────────────
    def _add_person_clicked(self) -> None:
        if self.fusion is None:
            return
        dlg = _ProfileDialog(self, title="Add Person",
                             fusion=self.fusion,
                             initial=None,
                             allow_face_pick=True, allow_is_self=True)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        v = dlg.values()
        if not v["name"]:
            QMessageBox.information(self, "Add", "Name is required.")
            return
        try:
            p = self.fusion.add_person_manual(**{k: v[k] for k in (
                "name", "title", "company", "relationship", "role_note",
                "is_self", "face_image_path")})
        except Exception as e:
            QMessageBox.warning(self, "Add failed", str(e))
            return
        if p is None:
            QMessageBox.warning(self, "Add failed", "Could not save person.")
            return
        self._append_activity(f"added {p.name}"
                              + (" (self)" if p.is_self else ""))
        self.refresh()

    def _edit_self_clicked(self) -> None:
        if self.fusion is None:
            return
        me = self.fusion.get_self()
        if me is None:
            return
        self._open_profile_editor(me)

    def _edit_profile_selected(self) -> None:
        pid = self._selected_person_id()
        if pid is None or self.fusion is None:
            return
        person = self.fusion.get_person(pid)
        if person is None:
            return
        self._open_profile_editor(person)

    def _open_profile_editor(self, person) -> None:
        dlg = _ProfileDialog(
            self, title=f"Edit profile \u2014 {person.name}",
            fusion=self.fusion,
            initial={
                "name": person.name,
                "title": person.title,
                "company": person.company,
                "relationship": person.relationship,
                "role_note": person.role_note,
                "is_self": person.is_self,
            },
            allow_face_pick=False, allow_is_self=True)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        v = dlg.values()
        if not v["name"]:
            return
        try:
            self.fusion.update_profile(
                person.id,
                name=v["name"], title=v["title"],
                company=v["company"], relationship=v["relationship"],
                role_note=v["role_note"], is_self=v["is_self"])
        except Exception as e:
            QMessageBox.warning(self, "Update failed", str(e))
            return
        self._append_activity(f"updated {v['name']}")
        self.refresh()

    def _show_conversations_selected(self) -> None:
        pid = self._selected_person_id()
        if pid is None or self.fusion is None:
            return
        person = self.fusion.get_person(pid)
        if person is None:
            return
        dlg = _ConversationsDialog(self, fusion=self.fusion, person=person)
        dlg.exec()

    @staticmethod
    def _restyle_pill(pill: Pill, fg: str) -> None:
        pill.setStyleSheet(
            f"color:{fg}; background: rgba({_rgb(fg)},0.12);"
            f"border: 1px solid rgba({_rgb(fg)},0.30);"
            f"border-radius: 8px; padding: 2px 9px;"
            f"font-family:'{FONT_MONO}','Consolas',monospace; font-size:10px;")
    def _selected_person_id(self) -> Optional[int]:
        rows = self.table.selectionModel().selectedRows() \
            if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), self._COL_NAME)
        if item is None:
            return None
        try:
            return int(item.data(Qt.ItemDataRole.UserRole))
        except Exception:
            return None
    def _update_action_buttons(self) -> None:
        if self.fusion is None:
            return
        has_sel = self._selected_person_id() is not None
        for b in (self.btn_profile, self.btn_convos, self.btn_rename,
                  self.btn_role, self.btn_merge, self.btn_delete):
            b.setEnabled(has_sel)
    def _on_voice_ingested_ui(self, result) -> None:
        try:
            name = os.path.basename(getattr(result, "wav_path", "") or "")
            tot  = int(getattr(result, "clusters_total", 0))
            mat  = int(getattr(result, "clusters_matched", 0))
            enr  = int(getattr(result, "clusters_enrolled", 0))
            if getattr(result, "skipped", False):
                msg = f"{name}  \u2014  skipped"
            elif getattr(result, "error", ""):
                msg = f"{name}  \u26a0  {result.error}"
            else:
                bits = []
                if mat: bits.append(f"{mat} matched")
                if enr: bits.append(f"{enr} new")
                msg = f"{name}  voice \u2192  {tot} cluster" + \
                      ("s" if tot != 1 else "") + \
                      (("  (" + ", ".join(bits) + ")") if bits else "")
            self._append_activity(msg)
        except Exception:
            pass
        self.refresh()
    def _on_faces_processed_ui(self, results) -> None:
        try:
            for pf in results or []:
                tag = "new" if getattr(pf, "was_new_enrollment", False) \
                    else (f"sim {pf.similarity:.2f}")
                self._append_activity(f"face  \u2192  {pf.name}  ({tag})")
        except Exception:
            pass
        self.refresh()
    def _append_activity(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        row = QLabel(f"{ts}  \u00b7  {text}")
        row.setStyleSheet(
            f"color:{TEXT_MUTED}; background:transparent; border:none;"
            f"font-family:'{FONT_MONO}','Consolas',monospace; font-size:10px;"
            "padding: 1px 2px;")
        self._feed_layout.insertWidget(0, row)
        while self._feed_layout.count() - 1 > self._MAX_ACTIVITY_ROWS:
            item = self._feed_layout.takeAt(self._feed_layout.count() - 2)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.deleteLater()
    def _rename_selected(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        pid = self._selected_person_id()
        if pid is None or self.fusion is None:
            return
        person = self.fusion.get_person(pid)
        if person is None:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename person",
            f"New name for \u201c{person.name}\u201d:", text=person.name)
        if not ok:
            return
        new_name = (new_name or "").strip()
        if not new_name or new_name == person.name:
            return
        if self.fusion.rename(pid, new_name):
            self._append_activity(f"renamed \u2192 {new_name}")
            self.refresh()
    def _edit_role_selected(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        pid = self._selected_person_id()
        if pid is None or self.fusion is None:
            return
        person = self.fusion.get_person(pid)
        if person is None:
            return
        new_note, ok = QInputDialog.getText(
            self, "Edit role note",
            f"Role note for \u201c{person.name}\u201d:",
            text=person.role_note or "")
        if not ok:
            return
        if self.fusion.update_role_note(pid, (new_note or "").strip()):
            self.refresh()
    def _delete_selected(self) -> None:
        pid = self._selected_person_id()
        if pid is None or self.fusion is None:
            return
        person = self.fusion.get_person(pid)
        if person is None:
            return
        resp = QMessageBox.question(
            self, "Delete person",
            f"Delete \u201c{person.name}\u201d and all their embeddings? "
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if resp != QMessageBox.StandardButton.Yes:
            return
        if self.fusion.delete_person(pid):
            self._append_activity(f"deleted {person.name}")
            self.refresh()
    def _merge_selected(self) -> None:
        from PyQt6.QtWidgets import QInputDialog
        drop_id = self._selected_person_id()
        if drop_id is None or self.fusion is None:
            return
        drop = self.fusion.get_person(drop_id)
        if drop is None:
            return
        others = [p for p in self.fusion.list_people() if p.id != drop_id]
        if not others:
            QMessageBox.information(self, "Merge", "No other people to merge with.")
            return
        labels = [f"{p.name}  (id {p.id})" for p in others]
        choice, ok = QInputDialog.getItem(
            self, "Merge into\u2026",
            f"Merge \u201c{drop.name}\u201d into which person?",
            labels, 0, False)
        if not ok or not choice:
            return
        keep = others[labels.index(choice)]
        report = self.fusion.merge_people(keep_id=keep.id, drop_id=drop_id)
        if report.success:
            self._append_activity(
                f"merged {drop.name} \u2192 {report.kept_name}")
            self.refresh()
        else:
            QMessageBox.warning(self, "Merge failed", report.error or "Unknown error")
    def _open_db_folder(self) -> None:
        if self.fusion is None:
            return
        folder = os.path.dirname(self.fusion.db_path) or os.getcwd()
        try:
            os.startfile(folder)                          # type: ignore
        except Exception:
            try:
                subprocess.Popen(["xdg-open", folder])
            except Exception:
                pass

    def _reset_all_people(self) -> None:
        if self.fusion is None:
            return
        resp = QMessageBox.question(
            self, "Reset all people",
            "This will delete ALL people, embeddings, conversations, and "
            "re-ingestion markers so recordings are re-processed from scratch.\n\n"
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if resp != QMessageBox.StandardButton.Yes:
            return
        try:
            for person in self.fusion.list_people():
                self.fusion.delete_person(person.id)
        except Exception as e:
            QMessageBox.warning(self, "Reset failed", str(e))
            return
        # Delete all .voice_ingested.json marker files so WAVs get re-ingested.
        try:
            recordings_dir = self.fusion.recordings_dir or ""
            if recordings_dir and os.path.isdir(recordings_dir):
                import glob
                markers = glob.glob(
                    os.path.join(recordings_dir, "*.voice_ingested.json"))
                for m in markers:
                    try:
                        os.remove(m)
                    except Exception:
                        pass
                self._append_activity(
                    f"deleted {len(markers)} ingestion marker(s)")
        except Exception as e:
            print(f"[people] marker cleanup failed: {e}")
        self._append_activity("reset all people and embeddings")
        self.refresh()

    def shutdown(self) -> None:
        try:
            self._refresh_timer.stop()
        except Exception:
            pass
        if self.fusion is not None:
            try:
                self.fusion.shutdown()
            except Exception:
                pass
    def showEvent(self, event) -> None:
        self.refresh()
        super().showEvent(event)
# ─────────────────────────────────────────────────────────────────────────────
# Stream tab — ESP32 Video + Photo Receiver, rebuilt natively in PyQt6.
# Networking / file-transfer logic is ported verbatim from terminal.py.
# Only the configuration constants (ports, folder paths) are imported from
# terminal.py at runtime so the user only needs to edit one file.
# ─────────────────────────────────────────────────────────────────────────────
class StreamTab(QWidget):
    # ── terminal.py dark-IDE color palette ────────────────────────────────
    _BG     = "#1e1e1e"
    _PANEL  = "#252526"
    _CARD   = "#2d2d30"
    _FG     = "#d4d4d4"
    _MUTED  = "#9a9a9a"
    _ACCENT = "#3b82f6"
    _GREEN  = "#22c55e"
    _RED    = "#ef4444"
    _ORANGE = "#f59e0b"
    _CYAN   = "#06b6d4"
    _YELLOW = "#eab308"
    def __init__(self, parent=None):
        super().__init__(parent)
        self._import_constants()
        self._init_state()
        self._build_ui()
        # QTimer replaces terminal.py's root.after(300, _poll_queue)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._drain_queue)
        self._poll_timer.start(300)
        self._load_existing_recordings()
        self._log("Ready. Click Start Listening to wait for the ESP32.")
        try:
            QApplication.instance().aboutToQuit.connect(self._cleanup)
        except Exception:
            pass
    # ── pull constants from terminal.py (only non-UI module-level vars) ───
    def _import_constants(self):
        import re as _re
        try:
            import sys as _sys, os as _os
            _here = _os.path.dirname(_os.path.abspath(__file__))
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            import terminal as _t
            self.SAVE_FOLDER        = _t.SAVE_FOLDER
            self.PHOTO_FOLDER       = _t.PHOTO_FOLDER
            self.TRANSFER_PORT      = _t.TRANSFER_PORT
            self.CMD_PORT           = _t.CMD_PORT
            self.PHOTO_CMD_PORT     = _t.PHOTO_CMD_PORT
            self.PHOTO_RECEIVE_PORT = _t.PHOTO_RECEIVE_PORT
            self.PAUSE_CMD_PORT     = _t.PAUSE_CMD_PORT
            self.COMPUTER_IP        = _t.COMPUTER_IP
            self.ESP32_IP_DEFAULT   = _t.ESP32_IP
            self.VID_W              = _t.VID_W
            self.VID_H              = _t.VID_H
            self.TIMESTAMP_RE       = _t.TIMESTAMP_RE
        except Exception:
            self.SAVE_FOLDER        = r"C:\Users\delete me\Desktop\ESP32_Recording"
            self.PHOTO_FOLDER       = r"C:\Users\delete me\Desktop\camera_photos"
            self.TRANSFER_PORT      = 5010
            self.CMD_PORT           = 5005
            self.PHOTO_CMD_PORT     = 5006
            self.PHOTO_RECEIVE_PORT = 5011
            self.PAUSE_CMD_PORT     = 5007
            self.COMPUTER_IP        = "0.0.0.0"
            self.ESP32_IP_DEFAULT   = "192.168.1.210"
            self.VID_W              = 480
            self.VID_H              = 320
            self.TIMESTAMP_RE       = _re.compile(r"_(\d{8}_\d{6})")
    def _init_state(self):
        self.clip_queue          = queue.Queue()
        self.clips               = {}        # row_index → clip dict
        self.server_socket       = None
        self.photo_server_socket = None
        self.listening           = False
        self.stop_event          = threading.Event()
        self.pending_row         = None
        self.esp32_ip            = self.ESP32_IP_DEFAULT
        self.paused              = False
        # video player state
        self.cap             = None
        self.playing         = False
        self.current_frame   = 0
        self.frame_count     = 0
        self.fps             = 15
        self.delay_ms        = 66
        self.current_path    = None
        try:
            import cv2
            self._cv2     = cv2
            self.HAVE_CV2 = True
        except ImportError:
            self._cv2     = None
            self.HAVE_CV2 = False
    # ── stylesheet helpers ─────────────────────────────────────────────────
    @staticmethod
    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    def _big_btn_ss(self, color):
        r,g,b = self._hex_to_rgb(color)
        return (
            f"QPushButton{{background:{color};color:white;border:none;border-radius:4px;"
            f"font-family:'Segoe UI';font-size:11pt;font-weight:bold;"
            f"text-align:left;padding-left:12px;height:38px;}}"
            f"QPushButton:disabled{{background:rgba({r},{g},{b},0.45);color:rgba(255,255,255,0.38);}}"
            f"QPushButton:hover:enabled{{background:rgba({r},{g},{b},0.80);}}"
        )
    def _small_btn_ss(self, color):
        r,g,b = self._hex_to_rgb(color)
        return (
            f"QPushButton{{background:{color};color:white;border:none;border-radius:3px;"
            f"padding:4px 10px;font-family:'Segoe UI';font-size:9pt;}}"
            f"QPushButton:disabled{{background:rgba({r},{g},{b},0.45);color:rgba(255,255,255,0.38);}}"
            f"QPushButton:hover:enabled{{background:rgba({r},{g},{b},0.80);}}"
        )
    def _toolbar_btn_ss(self):
        return (
            f"QPushButton{{background:{self._CARD};color:{self._FG};border:none;"
            f"border-radius:3px;padding:4px 10px;font-family:'Segoe UI';font-size:9pt;}}"
            f"QPushButton:hover{{background:#3c3c3c;}}"
        )
    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        self.setStyleSheet(f"background:{self._BG};")
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet(f"background:{self._PANEL};")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        self._build_sidebar(sl)
        root.addWidget(sidebar)
        main = QWidget()
        main.setStyleSheet(f"background:{self._BG};")
        ml = QVBoxLayout(main)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        self._build_main(ml)
        root.addWidget(main, 1)
    def _build_sidebar(self, sl):
        hdr = QLabel("Status")
        hdr.setStyleSheet(
            f"color:{self._FG};background:transparent;"
            f"font-family:'Segoe UI';font-size:12pt;font-weight:bold;"
            f"padding:16px 16px 8px 16px;")
        sl.addWidget(hdr)
        self._conn_dot,  self._conn_lbl  = self._mk_dot_row(sl, "ESP32: waiting...")
        self._srv_dot,   self._srv_lbl   = self._mk_dot_row(sl, "Receiver: stopped")
        self._photo_dot, self._photo_lbl = self._mk_dot_row(sl, "Photo: idle")
        self._pause_dot, self._pause_lbl = self._mk_dot_row(sl, "Recording: running")
        # log box
        log_wrap = QWidget()
        log_wrap.setStyleSheet("background:transparent;")
        lwl = QVBoxLayout(log_wrap)
        lwl.setContentsMargins(16, 8, 16, 8)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(90)
        self.log_box.setStyleSheet(
            f"QTextEdit{{background:{self._CARD};border:none;color:{self._FG};"
            f"font-family:Consolas,'Courier New',monospace;font-size:9pt;}}")
        lwl.addWidget(self.log_box)
        sl.addWidget(log_wrap, 1)
        # pending clip decision
        pend = QWidget()
        pend.setStyleSheet("background:transparent;")
        pl = QVBoxLayout(pend)
        pl.setContentsMargins(16, 8, 16, 4)
        pl.setSpacing(4)
        self._pending_lbl = QLabel("No clip waiting on a decision.")
        self._pending_lbl.setWordWrap(True)
        self._pending_lbl.setStyleSheet(
            f"color:{self._MUTED};background:transparent;"
            f"font-family:'Segoe UI';font-size:9pt;")
        pl.addWidget(self._pending_lbl)
        brow = QHBoxLayout()
        brow.setSpacing(6)
        brow.setContentsMargins(0, 0, 0, 0)
        self._keep_btn   = QPushButton("Keep")
        self._delete_btn = QPushButton("Delete")
        self._format_btn = QPushButton("Format SD")
        for btn, color in [(self._keep_btn,   self._GREEN),
                           (self._delete_btn, self._RED),
                           (self._format_btn, self._ORANGE)]:
            btn.setEnabled(False)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self._small_btn_ss(color))
        self._keep_btn.clicked.connect(self._keep_pending)
        self._delete_btn.clicked.connect(self._delete_pending)
        self._format_btn.clicked.connect(self._format_sd_pending)
        brow.addWidget(self._keep_btn)
        brow.addWidget(self._delete_btn)
        brow.addWidget(self._format_btn)
        brow.addStretch(1)
        pl.addLayout(brow)
        sl.addWidget(pend)
        # action buttons
        bw = QWidget()
        bw.setStyleSheet("background:transparent;")
        bwl = QVBoxLayout(bw)
        bwl.setContentsMargins(16, 8, 16, 16)
        bwl.setSpacing(4)
        self._photo_btn     = QPushButton("\U0001f4f7  Take Photo")
        self._pause_btn     = QPushButton("\u23f8  Pause Recording")
        self._format_sd_btn = QPushButton("\U0001f5d1  Format SD Card")
        self._toggle_btn    = QPushButton("\u25b6  Start Listening")
        self._photo_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self._format_sd_btn.setEnabled(False)
        self._toggle_btn.setEnabled(True)
        for btn, color in [(self._photo_btn,     self._CYAN),
                           (self._pause_btn,     self._YELLOW),
                           (self._format_sd_btn, self._ORANGE),
                           (self._toggle_btn,    self._ACCENT)]:
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(38)
            btn.setStyleSheet(self._big_btn_ss(color))
        self._photo_btn.clicked.connect(self._request_photo)
        self._pause_btn.clicked.connect(self._toggle_pause)
        self._format_sd_btn.clicked.connect(self._format_sd_standalone)
        self._toggle_btn.clicked.connect(self._toggle_listening)
        bwl.addWidget(self._photo_btn)
        bwl.addWidget(self._pause_btn)
        bwl.addWidget(self._format_sd_btn)
        bwl.addWidget(self._toggle_btn)
        sl.addWidget(bw)
    def _mk_dot_row(self, layout, text):
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(16, 2, 16, 2)
        rl.setSpacing(0)
        dot = QLabel("\u25cf")
        dot.setFixedWidth(14)
        dot.setStyleSheet(f"color:{self._MUTED};background:transparent;font-size:9pt;")
        lbl = QLabel(" " + text)
        lbl.setStyleSheet(
            f"color:{self._FG};background:transparent;"
            f"font-family:'Segoe UI';font-size:9pt;")
        rl.addWidget(dot)
        rl.addWidget(lbl)
        rl.addStretch(1)
        layout.addWidget(row)
        return dot, lbl
    def _build_main(self, ml):
        # header
        hdr = QWidget()
        hdr.setStyleSheet("background:transparent;")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(16, 16, 16, 8)
        hl.setSpacing(6)
        title = QLabel("Recordings")
        title.setStyleSheet(
            f"color:{self._FG};background:transparent;"
            f"font-family:'Segoe UI';font-size:12pt;font-weight:bold;")
        hl.addWidget(title)
        hl.addStretch(1)
        for txt, cmd in [("\u25b6  Play",    self._play_selected),
                          ("\u23f9  Stop",    self._stop_playback),
                          ("Open Folder",     self._open_selected_folder)]:
            b = QPushButton(txt)
            b.clicked.connect(cmd)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(self._toolbar_btn_ss())
            hl.addWidget(b)
        ml.addWidget(hdr)
        # video player (left-aligned, fixed width)
        player = QWidget()
        player.setStyleSheet("background:transparent;")
        pvl = QVBoxLayout(player)
        pvl.setContentsMargins(16, 0, 16, 8)
        pvl.setSpacing(4)
        pvl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._video_title = QLabel("No clip loaded")
        self._video_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_title.setFixedWidth(self.VID_W)
        self._video_title.setStyleSheet(
            f"color:{self._MUTED};background:transparent;"
            f"font-family:'Segoe UI';font-size:9pt;")
        pvl.addWidget(self._video_title)
        self._video_lbl = QLabel()
        self._video_lbl.setFixedSize(self.VID_W, self.VID_H)
        self._video_lbl.setStyleSheet("background:black;border:none;")
        self._blank_pixmap = QPixmap(self.VID_W, self.VID_H)
        self._blank_pixmap.fill(QColor("black"))
        self._video_lbl.setPixmap(self._blank_pixmap)
        pvl.addWidget(self._video_lbl)
        ctrl = QWidget()
        ctrl.setStyleSheet("background:transparent;")
        ctrl.setFixedWidth(self.VID_W)
        crl = QHBoxLayout(ctrl)
        crl.setContentsMargins(0, 4, 0, 0)
        crl.setSpacing(8)
        self._play_btn = QPushButton("\u25b6")
        self._play_btn.setFixedSize(32, 24)
        self._play_btn.clicked.connect(self._toggle_play)
        self._play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._play_btn.setStyleSheet(
            f"QPushButton{{background:{self._CARD};color:{self._FG};"
            f"border:none;border-radius:3px;}}"
            f"QPushButton:hover{{background:#3c3c3c;}}")
        crl.addWidget(self._play_btn)
        crl.addStretch(1)
        self._time_lbl = QLabel("0:00 / 0:00")
        self._time_lbl.setStyleSheet(
            f"color:{self._MUTED};background:transparent;"
            f"font-family:Consolas,'Courier New',monospace;font-size:9pt;")
        crl.addWidget(self._time_lbl)
        pvl.addWidget(ctrl)
        self._seek = QSlider(Qt.Orientation.Horizontal)
        self._seek.setFixedWidth(self.VID_W)
        self._seek.setRange(0, 100)
        self._seek.setValue(0)
        self._seek.setStyleSheet(
            "QSlider::groove:horizontal{background:#3c3c3c;height:4px;border-radius:2px;}"
            f"QSlider::sub-page:horizontal{{background:{self._ACCENT};height:4px;border-radius:2px;}}"
            "QSlider::handle:horizontal{background:#9a9a9a;width:10px;height:10px;"
            "border-radius:5px;margin:-3px 0;}")
        self._seek.valueChanged.connect(self._on_seek)
        pvl.addWidget(self._seek)
        player_row = QHBoxLayout()
        player_row.addWidget(player)
        player_row.addStretch(1)
        ml.addLayout(player_row)
        # recordings table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Time","Filename","Size","Transfer","Status"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(True)
        self.table.setStyleSheet(
            f"QTableWidget{{background:{self._CARD};border:none;"
            f"gridline-color:#3c3c3c;color:{self._FG};"
            f"font-family:Consolas,'Courier New',monospace;font-size:9pt;}}"
            f"QTableWidget::item:selected{{background:{self._ACCENT};color:white;}}"
            f"QHeaderView::section{{background:{self._PANEL};color:{self._MUTED};"
            f"border:none;padding:4px;font-family:'Segoe UI';font-size:9pt;}}")
        for col, w in enumerate([130, 220, 75, 140, 100]):
            self.table.setColumnWidth(col, w)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(lambda r, c: self._play_selected())
        ml.addWidget(self.table, 1)
    # ── logging ────────────────────────────────────────────────────────────
    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())
    # ── helpers ────────────────────────────────────────────────────────────
    def _fmt_time(self, seconds):
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        return f"{m}:{s:02d}"
    def _guess_received_at(self, filepath):
        match = self.TIMESTAMP_RE.search(os.path.basename(filepath))
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            except ValueError:
                pass
        return datetime.fromtimestamp(os.path.getmtime(filepath))
    def _load_existing_recordings(self):
        if not os.path.isdir(self.SAVE_FOLDER):
            return
        entries = []
        for fname in os.listdir(self.SAVE_FOLDER):
            if not fname.lower().endswith(".avi"):
                continue
            path = os.path.join(self.SAVE_FOLDER, fname)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            entries.append((self._guess_received_at(path), fname, path, size))
        entries.sort(key=lambda e: e[0])
        for received_at, filename, filepath, size in entries:
            size_mb  = size / (1024 * 1024)
            time_str = received_at.strftime("%Y-%m-%d %H:%M:%S")
            self._add_table_row(
                time_str, filename, f"{size_mb:.2f} MB", "\u2014", "saved",
                {"filename": filename, "filepath": filepath,
                 "size_bytes": size, "received_at": received_at, "ip": None})
        n = self.table.rowCount()
        if n:
            self._log(f"Loaded {n} previous recording(s) from disk.")
    def _add_table_row(self, time_str, filename, size, transfer, status, clip_dict):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, val in enumerate([time_str, filename, size, transfer, status]):
            item = QTableWidgetItem(val)
            item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, col, item)
        self.clips[row] = clip_dict
        return row
    def _set_row_status(self, row, status):
        if row is None:
            return
        item = self.table.item(row, 4)
        if item:
            item.setText(status)
    def _selected_row(self):
        items = self.table.selectedItems()
        return self.table.currentRow() if items else None
    # ── listening ──────────────────────────────────────────────────────────
    def _toggle_listening(self):
        if self.listening:
            self._stop_listening()
        else:
            self._start_listening()
    def _start_listening(self):
        os.makedirs(self.SAVE_FOLDER, exist_ok=True)
        os.makedirs(self.PHOTO_FOLDER, exist_ok=True)
        self.stop_event.clear()
        self.listening = True
        self._toggle_btn.setText("\u25a0  Stop Listening")
        self._toggle_btn.setStyleSheet(self._big_btn_ss(self._RED))
        self._set_dot(self._srv_dot, self._srv_lbl,
                      self._GREEN, f" Receiver: listening on port {self.TRANSFER_PORT}")
        self._photo_btn.setEnabled(True)
        self._pause_btn.setEnabled(True)
        self._format_sd_btn.setEnabled(True)
        self._log(f"Listening for clips on port {self.TRANSFER_PORT}...")
        self._log(f"Listening for photos on port {self.PHOTO_RECEIVE_PORT}...")
        threading.Thread(target=self._server_loop,       daemon=True).start()
        threading.Thread(target=self._photo_server_loop, daemon=True).start()
    def _stop_listening(self):
        self.stop_event.set()
        self.listening = False
        self._toggle_btn.setText("\u25b6  Start Listening")
        self._toggle_btn.setStyleSheet(self._big_btn_ss(self._ACCENT))
        self._set_dot(self._srv_dot, self._srv_lbl, self._MUTED, " Receiver: stopped")
        self._photo_btn.setEnabled(False)
        self._pause_btn.setEnabled(False)
        self._format_sd_btn.setEnabled(False)
        self._pause_btn.setText("\u23f8  Pause Recording")
        self._pause_btn.setStyleSheet(self._big_btn_ss(self._YELLOW))
        self.paused = False
        self._set_dot(self._pause_dot, self._pause_lbl, self._MUTED, " Recording: stopped")
        self._log("Stopped listening.")
        for sock in (self.server_socket, self.photo_server_socket):
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass
    def _set_dot(self, dot, lbl, color, text):
        dot.setStyleSheet(
            f"color:{color};background:transparent;font-size:9pt;")
        lbl.setText(text)
    # ── server loops (verbatim from terminal.py) ───────────────────────────
    def _server_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        server.settimeout(1.0)
        try:
            server.bind((self.COMPUTER_IP, self.TRANSFER_PORT))
            server.listen(1)
        except OSError as e:
            self.clip_queue.put({"type": "error",
                                 "message": f"Couldn't bind port {self.TRANSFER_PORT}: {e}"})
            return
        self.server_socket = server
        while not self.stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.esp32_ip = addr[0]
            self.clip_queue.put({"type": "connect", "ip": addr[0]})
            threading.Thread(target=self._receive_file,
                             args=(conn, addr), daemon=True).start()
        try:
            server.close()
        except OSError:
            pass
    def _photo_server_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        server.settimeout(1.0)
        try:
            server.bind((self.COMPUTER_IP, self.PHOTO_RECEIVE_PORT))
            server.listen(1)
        except OSError as e:
            self.clip_queue.put({"type": "error",
                                 "message": f"Couldn't bind photo port {self.PHOTO_RECEIVE_PORT}: {e}"})
            return
        self.photo_server_socket = server
        while not self.stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.esp32_ip = addr[0]
            threading.Thread(target=self._receive_photo,
                             args=(conn, addr), daemon=True).start()
        try:
            server.close()
        except OSError:
            pass
    def _receive_file(self, conn, addr):
        filename = None
        filepath = None
        received = 0
        start_time = time.time()
        try:
            header = b""
            while b"\n" not in header:
                chunk = conn.recv(1)
                if not chunk:
                    return
                header += chunk
            raw_filename, filesize = header.decode().strip().split(":")
            filesize = int(filesize)
            stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(raw_filename)
            filename = f"{name}_{stamp}{ext}"
            filepath = os.path.join(self.SAVE_FOLDER, filename)
            self.clip_queue.put({"type": "receiving",
                                 "filename": filename, "size_bytes": filesize})
            last_progress = -1
            with open(filepath, "wb") as f:
                while received < filesize:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)
                    progress = int((received * 100) / filesize) if filesize else 100
                    if progress >= last_progress + 10:
                        last_progress = progress
                        self.clip_queue.put({
                            "type": "progress", "filename": filename,
                            "progress": progress, "elapsed": time.time() - start_time})
        finally:
            conn.close()
        if filename is None:
            return
        elapsed   = time.time() - start_time
        speed_kbs = (received / 1024) / elapsed if elapsed > 0 else 0
        self.clip_queue.put({
            "type": "clip", "filename": filename, "filepath": filepath,
            "size_bytes": received, "ip": addr[0], "received_at": datetime.now(),
            "transfer_seconds": elapsed, "transfer_speed_kbs": speed_kbs,
        })
    def _receive_photo(self, conn, addr):
        self.clip_queue.put({"type": "photo_receiving"})
        received = 0
        try:
            header = b""
            while b"\n" not in header:
                byte = conn.recv(1)
                if not byte:
                    return
                header += byte
            _filename, filesize = header.decode().strip().split(":")
            filesize  = int(filesize)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath  = os.path.join(self.PHOTO_FOLDER, f"photo_{timestamp}.jpg")
            start_time = time.time()
            with open(filepath, "wb") as f:
                while received < filesize:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)
            elapsed = time.time() - start_time
            self.clip_queue.put({
                "type": "photo_done", "filepath": filepath,
                "size": received, "elapsed": elapsed,
            })
        finally:
            conn.close()
    # ── pause / resume ─────────────────────────────────────────────────────
    def _toggle_pause(self):
        ip = self.esp32_ip
        if not ip:
            self._log("[PAUSE] No ESP32 IP known yet.")
            return
        if self.paused:
            def send_resume():
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(5)
                    s.connect((ip, self.PAUSE_CMD_PORT))
                    s.sendall(b"resume\n")
                    s.close()
                    self.clip_queue.put({"type": "resumed"})
                except Exception as e:
                    self.clip_queue.put({"type": "pause_failed", "message": str(e)})
            threading.Thread(target=send_resume, daemon=True).start()
        else:
            def send_pause():
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(5)
                    s.connect((ip, self.CMD_PORT))
                    s.sendall(b"pause\n")
                    s.close()
                    self.clip_queue.put({"type": "paused"})
                except Exception as e:
                    self.clip_queue.put({"type": "pause_failed", "message": str(e)})
            threading.Thread(target=send_pause, daemon=True).start()
    # ── photo ──────────────────────────────────────────────────────────────
    def _request_photo(self):
        ip = self.esp32_ip
        if not ip:
            self._log("[PHOTO] No ESP32 IP known yet — connect first.")
            return
        self._log("[PHOTO] Sending take_photo command to ESP32...")
        self._set_dot(self._photo_dot, self._photo_lbl,
                      self._ORANGE, " Photo: requesting...")
        self._photo_btn.setEnabled(False)
        def send_cmd():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect((ip, self.PHOTO_CMD_PORT))
                s.sendall(b"take_photo\n")
                s.close()
                self.clip_queue.put({"type": "photo_cmd_sent"})
            except Exception as e:
                self.clip_queue.put({"type": "photo_cmd_failed", "message": str(e)})
        threading.Thread(target=send_cmd, daemon=True).start()
    # ── queue drain (replaces root.after poll) ─────────────────────────────
    def _drain_queue(self):
        try:
            while True:
                self._handle_queue_item(self.clip_queue.get_nowait())
        except queue.Empty:
            pass
    def _handle_queue_item(self, item):
        t = item["type"]
        if t == "connect":
            self._set_dot(self._conn_dot, self._conn_lbl,
                          self._GREEN, f" ESP32: connected ({item['ip']})")
            self._log(f"Connection from {item['ip']}")
        elif t == "error":
            self._log(f"ERROR: {item['message']}")
        elif t == "receiving":
            self._log(f"Receiving {item['filename']} "
                      f"({item['size_bytes']/1048576:.2f} MB)...")
        elif t == "progress":
            self._log(f"  {item['progress']}% \u2014 {item['elapsed']:.1f}s elapsed")
        elif t == "clip":
            self._add_clip(item)
        elif t == "paused":
            self.paused = True
            self._pause_btn.setText("\u25b6  Resume Recording")
            self._pause_btn.setStyleSheet(self._big_btn_ss(self._GREEN))
            self._set_dot(self._pause_dot, self._pause_lbl,
                          self._YELLOW, " Recording: paused \u23f8")
            self._log("[PAUSE] Recording paused.")
        elif t == "resumed":
            self.paused = False
            self._pause_btn.setText("\u23f8  Pause Recording")
            self._pause_btn.setStyleSheet(self._big_btn_ss(self._YELLOW))
            self._set_dot(self._pause_dot, self._pause_lbl,
                          self._GREEN, " Recording: running")
            self._log("[PAUSE] Recording resumed.")
        elif t == "pause_failed":
            self._log(f"[PAUSE] Failed: {item['message']}")
        elif t == "photo_cmd_sent":
            self._log("[PHOTO] Command sent \u2014 waiting for ESP32...")
            self._photo_lbl.setText(" Photo: waiting for capture...")
        elif t == "photo_cmd_failed":
            self._log(f"[PHOTO] Command failed: {item['message']}")
            self._set_dot(self._photo_dot, self._photo_lbl,
                          self._RED, " Photo: command failed")
            self._photo_btn.setEnabled(True)
        elif t == "photo_receiving":
            self._log("[PHOTO] Receiving photo...")
            self._set_dot(self._photo_dot, self._photo_lbl,
                          self._ORANGE, " Photo: receiving...")
        elif t == "photo_done":
            size_kb = item["size"] / 1024
            self._log(f"[PHOTO] Saved: {os.path.basename(item['filepath'])} "
                      f"({size_kb:.1f} KB, {item['elapsed']:.1f}s)")
            self._set_dot(self._photo_dot, self._photo_lbl,
                          self._GREEN, " Photo: saved \u2713")
            self._photo_btn.setEnabled(True)
            # Notify any subscriber (IrisApp wires the chat tab in here)
            # so the photo can flow back to chat + the Photos tab. If no
            # subscriber is hooked up, fall back to opening the file so
            # the user still sees it (terminal.py's original behavior).
            cb = getattr(self, "_on_photo_arrived_cb", None)
            if cb is not None:
                try:
                    cb(item["filepath"])
                except Exception as e:
                    print(f"[stream] photo arrival callback failed: {e}")
            else:
                try:
                    os.startfile(item["filepath"])
                except Exception:
                    pass
    def _add_clip(self, item):
        size_mb      = item["size_bytes"] / (1024 * 1024)
        time_str     = item["received_at"].strftime("%Y-%m-%d %H:%M:%S")
        elapsed      = item.get("transfer_seconds", 0)
        speed        = item.get("transfer_speed_kbs", 0)
        transfer_str = f"{elapsed:.1f}s @ {speed:.0f} KB/s"
        row = self._add_table_row(
            time_str, item["filename"],
            f"{size_mb:.2f} MB", transfer_str, "received", item)
        self._log(f"Received {item['filename']} \u2014 {elapsed:.1f}s, {speed:.0f} KB/s")
        #self.pending_row = row
        # self._pending_lbl.setText(
        #    f"New clip: {item['filename']}\nKeep it, or delete it?")
        # self._pending_lbl.setStyleSheet(
         #   f"color:{self._ORANGE};background:transparent;"
        #  f"font-family:'Segoe UI';font-size:9pt;")
        # self._keep_btn.setEnabled(True)
        # self._delete_btn.setEnabled(True)
        # self._format_btn.setEnabled(True)

        self._send_command("keep", item.get("ip"))
        self._set_row_status(row, "kept")
       
        # ── face recognition: process keyframes in a worker thread ────
        # Only fires when iris_fusion is loaded and DeepFace is ready.
        # Safe to skip if not — the clip is still kept/deleted normally.
       # ── face recognition + Gap 2 reconcile ───────────────────────────
        filepath = item.get("filepath")
        if filepath and iris_fusion is not None:
            threading.Thread(
                target=self._process_clip_for_faces,
                args=(filepath,),
                daemon=True,
                name="ClipFaceProc",
            ).start()
            threading.Thread(
                target=self._reconcile_clip_identities,
                args=(filepath,),
                daemon=True,
                name="ClipReconcile",
            ).start()

    def _process_clip_for_faces(self, filepath: str) -> None:
        """Worker thread: extract 1 keyframe/sec from a received AVI clip
        and run each through iris_fusion.process_frame(). Results flow
        automatically into the People registry and fire the
        on_faces_processed callback which updates the People tab.

        Design decisions:
        - 1 frame per second: matches the blueprint spec and the face
          pipeline's per-clip processing plan. Enough to catch every
          person who appears for more than a second; not so many that
          we saturate the CPU. At 30 fps a 35-second clip is ~1050
          frames; we reduce that to ~35 embeddings.
        - OpenCV VideoCapture: already available (StreamTab imports cv2
          in _init_state). No new dependency.
        - Drops frames if a previous process_frame() is still in flight
          (iris_fusion.process_frame() uses a non-blocking lock). This
          keeps latency bounded on slower hardware (HP Envy / integrated
          GPU) — we never queue up more work than the CPU can handle.
        - All errors are caught and logged; nothing here can break the
          existing clip-receive or keep/delete flow.
        """
        if not self.HAVE_CV2:
            return
        fusion = None
        if iris_fusion is not None:
            try:
                fusion = iris_fusion.get_fusion()
            except Exception:
                return
        if fusion is None:
            return
        if not fusion.faces.is_ready():
            # DeepFace still loading — skip rather than block.
            # Once it's ready, subsequent clips will be processed.
            return
        cv2 = self._cv2
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            # How many frames to skip to get ~1 keyframe per second.
            frame_step = max(1, int(fps))
            frame_idx  = 0
            faces_found = 0
            person_ids = set()        # distinct people → "how many people"
            names      = {}           # person_id → recognised name
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_step == 0:
                    # process_frame() is non-blocking internally — drops
                    # the frame if the previous call is still running.
                    results = fusion.process_frame(frame)
                    faces_found += len(results)
                    for pf in results:
                        person_ids.add(pf.person_id)
                        if getattr(pf, "name", ""):
                            names[pf.person_id] = pf.name
                frame_idx += 1
            cap.release()
            if faces_found:
                print(f"[faces] {os.path.basename(filepath)}: "
                      f"{faces_found} face detection(s) across "
                      f"{frame_idx} frames")
            # Persist a per-clip analysis sidecar so the chat can answer
            # "how many people were in that video?" — the distinct-person
            # count and recognised names, cached next to the .avi.
            if ivideos is not None:
                try:
                    clip_names = [names[i] for i in person_ids if i in names]
                    ivideos.record_analysis(
                        filepath,
                        people_count=len(person_ids),
                        people_names=clip_names,
                        duration_sec=(frame_idx / fps) if fps else None,
                        frames_sampled=(frame_idx // frame_step) + 1,
                        method="fusion",
                        note=f"{len(person_ids)} distinct "
                             f"person(s) via face recognition")
                    who = f" ({', '.join(clip_names)})" if clip_names else ""
                    print(f"[video] analysed {os.path.basename(filepath)}: "
                          f"{len(person_ids)} person(s){who} — saved so the "
                          f"chat can answer questions about this clip.")
                except Exception as e:
                    print(f"[faces] could not write video sidecar "
                          f"({os.path.basename(filepath)}): {e}")
        except Exception as e:
                    print(f"[faces] clip processing failed "
                        f"({os.path.basename(filepath)}): {e}")

    def _reconcile_clip_identities(self, filepath: str) -> None:
        """Confirms provisional conversation rows when a video clip arrives."""
        if iris_fusion is None:
            return
        try:
            fusion = iris_fusion.get_fusion()
        except Exception:
            return
        # Do NOT bail if DeepFace isn't ready — voice-only people (enrolled
        # by name from the diarizer, no face embeddings) can still be
        # confirmed without face matching. reconcile_clip() handles the
        # no-face-embeddings case by confirming by default.
        try:
            n = fusion.reconcile_clip(filepath)
            if n:
                self._log(f"[reconcile] confirmed {n} conversation"
                          f"{'s' if n != 1 else ''} from "
                          f"{os.path.basename(filepath)}")
        except Exception as e:
            print(f"[reconcile] failed for "
                  f"{os.path.basename(filepath)}: {e}")


    # ── keep / delete / format ─────────────────────────────────────────────
    def _send_command(self, cmd, ip):
        if not ip:
            self._log(f"Cannot send '{cmd}' \u2014 no ESP32 IP.")
            return False
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((ip, self.CMD_PORT))
            s.sendall((cmd + "\n").encode())
            s.close()
            self._log(f"Sent '{cmd}' to ESP32 at {ip}")
            return True
        except Exception as e:
            self._log(f"Couldn't send '{cmd}': {e}")
            return False
    def _keep_pending(self):
        self._decide(self.pending_row, "keep")
    def _delete_pending(self):
        self._decide(self.pending_row, "delete")
    def _format_sd_pending(self):
        if self.pending_row is None or self.pending_row not in self.clips:
            return
        clip = self.clips[self.pending_row]
        reply = QMessageBox.question(
            self, "Format SD Card",
            "This will delete ALL files on the ESP32 SD card.\nAre you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._send_command("format_sd", clip.get("ip"))
        self._set_row_status(self.pending_row, "SD formatted")
        self._reset_pending()
    def _format_sd_standalone(self):
        """Large Format SD Card button — works any time we have an ESP32 IP."""
        reply = QMessageBox.question(
            self, "Format SD Card",
            "This will delete ALL files on the ESP32 SD card.\nAre you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._send_command("format_sd", self.esp32_ip)
    def _decide(self, row, decision):
        if row is None or row not in self.clips:
            return
        clip   = self.clips[row]
        ok     = self._send_command(decision, clip.get("ip"))
        status = "kept" if decision == "keep" else "deleted"
        if not ok:
            status += " (send failed)"
        self._set_row_status(row, status)
        if row == self.pending_row:
            self._reset_pending()
    def _reset_pending(self):
        self._pending_lbl.setText("No clip waiting on a decision.")
        self._pending_lbl.setStyleSheet(
            f"color:{self._MUTED};background:transparent;"
            f"font-family:'Segoe UI';font-size:9pt;")
        self._keep_btn.setEnabled(False)
        self._delete_btn.setEnabled(False)
        self._format_btn.setEnabled(False)
        self.pending_row = None
    # ── video playback ─────────────────────────────────────────────────────
    def _play_selected(self):
        row = self._selected_row()
        if row is None:
            QMessageBox.information(self, "Play",
                                    "Select a clip in the list first.")
            return
        clip = self.clips.get(row)
        if not clip:
            return
        if not self.HAVE_CV2:
            QMessageBox.information(self, "opencv-python needed",
                "Playback needs opencv-python.\n\nRun:  pip install opencv-python")
            return
        self._start_playback(clip["filepath"], clip["filename"])
    def _start_playback(self, path, filename):
        self._stop_playback()
        cv2 = self._cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._log(f"Couldn't open {path}")
            return
        self.cap           = cap
        self.current_path  = path
        self.fps           = cap.get(cv2.CAP_PROP_FPS) or 15
        self.delay_ms      = max(1, int(1000 / self.fps))
        self.frame_count   = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 1)
        self.current_frame = 0
        self._seek.blockSignals(True)
        self._seek.setRange(0, max(self.frame_count - 1, 1))
        self._seek.setValue(0)
        self._seek.blockSignals(False)
        self.playing = True
        self._play_btn.setText("\u23f8")
        self._video_title.setText(filename)
        self._video_title.setStyleSheet(
            f"color:{self._GREEN};background:transparent;"
            f"font-family:'Segoe UI';font-size:9pt;")
        self._player_loop()
    def _player_loop(self):
        if not self.playing or self.cap is None:
            return
        cv2 = self._cv2
        ok, frame = self.cap.read()
        if not ok:
            self._stop_playback()
            return
        self.current_frame += 1
        self._render_frame(frame)
        self._update_progress()
        QTimer.singleShot(self.delay_ms, self._player_loop)
    def _render_frame(self, frame):
        cv2 = self._cv2
        if frame.shape[1] != self.VID_W or frame.shape[0] != self.VID_H:
            frame = cv2.resize(frame, (self.VID_W, self.VID_H))
        rgb = frame[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._video_lbl.setPixmap(QPixmap.fromImage(qimg))
    def _update_progress(self):
        self._seek.blockSignals(True)
        self._seek.setValue(self.current_frame)
        self._seek.blockSignals(False)
        fps     = self.fps or 1
        cur_s   = self.current_frame / fps
        total_s = self.frame_count / fps
        self._time_lbl.setText(
            f"{self._fmt_time(cur_s)} / {self._fmt_time(total_s)}")
    def _on_seek(self, value):
        if self.cap is None:
            return
        cv2 = self._cv2
        frame_idx = int(value)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if ok:
            self.current_frame = frame_idx
            self._render_frame(frame)
            self._update_progress()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    def _toggle_play(self):
        if self.cap is None:
            return
        if self.playing:
            self.playing = False
            self._play_btn.setText("\u25b6")
        else:
            self.playing = True
            self._play_btn.setText("\u23f8")
            self._player_loop()
    def _stop_playback(self):
        self.playing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._video_lbl.setPixmap(self._blank_pixmap)
        self._play_btn.setText("\u25b6")
        self.current_frame = 0
        self._seek.blockSignals(True)
        self._seek.setValue(0)
        self._seek.blockSignals(False)
        self._time_lbl.setText("0:00 / 0:00")
        if self.current_path:
            name = os.path.basename(self.current_path)
            self._video_title.setText(f"{name} (stopped)")
        else:
            self._video_title.setText("No clip loaded")
        self._video_title.setStyleSheet(
            f"color:{self._MUTED};background:transparent;"
            f"font-family:'Segoe UI';font-size:9pt;")
    def _open_selected_folder(self):
        row    = self._selected_row()
        clip   = self.clips.get(row) if row is not None else None
        folder = (os.path.dirname(clip["filepath"])
                  if clip and clip.get("filepath") else self.SAVE_FOLDER)
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)
        except Exception as e:
            self._log(f"Couldn't open folder: {e}")
    # ── cleanup ────────────────────────────────────────────────────────────
    def _cleanup(self):
        self._stop_playback()
        self._stop_listening()
        if self._poll_timer.isActive():
            self._poll_timer.stop()
# ─────────────────────────────────────────────────────────────────────────────
# Audio dashboard — glass Qt port of gui_phase9.AudioStreamGUI, embedded in the
# same window (no popup). Drives the same Controller + speaker_db + event_queue.
# ─────────────────────────────────────────────────────────────────────────────
def _audio_btn(text: str, on_click=None, *, fg: str = TEXT_PRIMARY,
               accent: str = "255,255,255", height: int = 36,
               bold: bool = False, width: Optional[int] = None) -> QPushButton:
    b = QPushButton(text)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if height:
        b.setFixedHeight(height)
    if width:
        b.setFixedWidth(width)
    weight = "700" if bold else "500"
    b.setStyleSheet(
        "QPushButton {"
        f"color:{fg}; background: rgba({accent},0.12);"
        f"border: 1px solid rgba({accent},0.30); border-radius: 10px;"
        "padding: 0 12px;"
        f"font-family:'{FONT_SANS}'; font-size:12px; font-weight:{weight};"
        "}"
        f"QPushButton:hover {{ background: rgba({accent},0.20); }}")
    if on_click:
        b.clicked.connect(on_click)
    _add_glass_shadow(b, blur=12, dy=2, alpha=90)
    return b
class VUMeter(QWidget):
    """Segmented input-level meter with a falling peak hold."""
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(22)
        self._level = 0.0
        self._peak = 0.0
    def setLevel(self, lvl: float) -> None:
        lvl = max(0.0, min(1.0, lvl))
        self._level = lvl
        self._peak = lvl if lvl > self._peak else max(lvl, self._peak * 0.92)
        self.update()
    def paintEvent(self, _evt):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 16))
        p.drawRoundedRect(0, 0, w, h, 6, 6)
        seg = 30
        sw = w / seg
        for i in range(seg):
            frac = i / seg
            if frac > self._level:
                col = QColor(255, 255, 255, 28)
            elif frac > 0.85:
                col = QColor("#ef4444")
            elif frac > 0.7:
                col = QColor("#f59e0b")
            else:
                col = QColor("#10b981")
            x0 = int(i * sw) + 2
            x1 = int((i + 1) * sw) - 1
            p.setBrush(col)
            p.drawRect(x0, 3, max(1, x1 - x0), h - 6)
        if self._peak > 0.02:
            px = int(self._peak * w)
            p.setBrush(QColor("#ffffff"))
            p.drawRect(max(0, px - 2), 2, 2, h - 4)
class StatusDot(QWidget):
    """Coloured dot + label, e.g. '\u25cf Audio stream: receiving'."""
    def __init__(self, text: str):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._dot = QLabel("\u25CF")
        self._dot.setStyleSheet(
            f"color:{COLOR_STATUS_OFF}; background:transparent; border:none;"
            "font-size:13px; font-weight:700;")
        self._label = QLabel(text)
        self._label.setStyleSheet(
            f"color:{TEXT_MUTED}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:12px;")
        lay.addWidget(self._dot)
        lay.addWidget(self._label)
        lay.addStretch(1)
    def set(self, *, on: bool = False, text: Optional[str] = None,
            color: Optional[str] = None) -> None:
        c = color if color else (COLOR_STATUS_ON if on else COLOR_STATUS_OFF)
        self._dot.setStyleSheet(
            f"color:{c}; background:transparent; border:none;"
            "font-size:13px; font-weight:700;")
        if text is not None:
            self._label.setText(text)
class ManageSpeakersDialog(QDialog):
    """Port of gui_phase9.ManageSpeakersDialog \u2014 list / rename / delete."""
    def __init__(self, parent, speaker_db, recordings_dir, on_changed):
        super().__init__(parent)
        self.setWindowTitle("Manage Speaker Profiles")
        self.resize(560, 480)
        self.setStyleSheet(
            f"QDialog {{ background:{BG_MID}; }}"
            f"QLabel {{ color:{TEXT_PRIMARY}; font-family:'{FONT_SANS}'; }}")
        self._db = speaker_db
        self._dir = recordings_dir
        self._on_changed = on_changed
        self._root = QVBoxLayout(self)
        self._build()
    def _build(self):
        while self._root.count():
            item = self._root.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        title = QLabel("\U0001F464  Saved Speaker Profiles")
        title.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:16px;"
                            "font-weight:700;")
        self._root.addWidget(title)
        try:
            profiles = self._db.all_info() if self._db else []
        except Exception:
            profiles = []
        if not profiles:
            note = QLabel("No speakers enrolled yet. Tag a speaker in a "
                          "transcript to enroll them.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color:{TEXT_MUTED}; font-size:12px;")
            self._root.addWidget(note)
            self._root.addStretch(1)
            self._root.addWidget(_audio_btn("Close", self.accept,
                                            accent=_rgb(ACCENT), fg=ACCENT,
                                            width=100),
                                 0, Qt.AlignmentFlag.AlignRight)
            return
        counts = self._count_appearances()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        holder = QWidget()
        holder.setStyleSheet("background: transparent;")
        vl = QVBoxLayout(holder)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(6)
        for info in profiles:
            card = GlassFrame(holder, radius=10, blur=14, dy=3, shadow_alpha=90)
            cl = QHBoxLayout(card)
            cl.setContentsMargins(12, 8, 10, 8)
            txt = QVBoxLayout()
            nm = QLabel(info.get("name", "?"))
            nm.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:13px;"
                             "font-weight:700;")
            appears = counts.get(info.get("name"), 0)
            sc = info.get("sample_count", 0)
            sub = QLabel(f"{sc} voice sample{'s' if sc != 1 else ''}  \u2022  "
                         f"appears in {appears} recording"
                         f"{'s' if appears != 1 else ''}")
            sub.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
            txt.addWidget(nm)
            txt.addWidget(sub)
            cl.addLayout(txt, 1)
            cl.addWidget(_audio_btn("Rename",
                                    lambda _=False, n=info["name"]: self._rename(n),
                                    accent=_rgb(BADGE_VOICE_FG),
                                    fg=BADGE_VOICE_FG, width=80, height=30))
            cl.addWidget(_audio_btn("Delete",
                                    lambda _=False, n=info["name"]: self._delete(n),
                                    accent=_rgb(COLOR_DANGER),
                                    fg="#fca5a5", width=80, height=30))
            vl.addWidget(card)
        vl.addStretch(1)
        scroll.setWidget(holder)
        self._root.addWidget(scroll, 1)
        self._root.addWidget(_audio_btn("Close", self.accept,
                                        accent=_rgb(ACCENT), fg=ACCENT,
                                        width=100),
                             0, Qt.AlignmentFlag.AlignRight)
    def _count_appearances(self) -> dict:
        counts: dict = {}
        try:
            for jp in glob.glob(os.path.join(self._dir, "recording_*.json")):
                try:
                    with open(jp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    names = {seg.get("speaker") for seg in data.get("segments", [])
                             if seg.get("speaker")}
                    for n in names:
                        counts[n] = counts.get(n, 0) + 1
                except Exception:
                    pass
        except Exception:
            pass
        return counts
    def _rename(self, old: str):
        from PyQt6.QtWidgets import QInputDialog
        new, ok = QInputDialog.getText(self, "Rename Speaker",
                                       f"New name for \"{old}\":", text=old)
        new = new.strip() if ok else ""
        if not new or new == old:
            return
        try:
            self._db.rename(old, new)
        except Exception:
            pass
        self._rename_in_transcripts(old, new)
        self._on_changed()
        self._build()
    def _rename_in_transcripts(self, old: str, new: str):
        for jp in glob.glob(os.path.join(self._dir, "recording_*.json")):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                changed = False
                for seg in data.get("segments", []):
                    if seg.get("speaker") == old:
                        seg["speaker"] = new
                        changed = True
                if changed:
                    with open(jp, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
            except Exception:
                pass
    def _delete(self, name: str):
        from PyQt6.QtWidgets import QMessageBox
        r = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete \"{name}\" and all their voice samples?\n"
            "Transcript labels using this name will remain.")
        if r != QMessageBox.StandardButton.Yes:
            return
        try:
            self._db.delete(name)
        except Exception:
            pass
        self._on_changed()
        self._build()
# ─────────────────────────────────────────────────────────────────────────────
# Live Transcription window — a separate floating window that shows the
# rolling whisper chunks (with [hh:mm:ss → hh:mm:ss] timestamps and speaker
# tags) extended to fill the entire panel height, plus a "together" panel
# at the bottom that accumulates everything. Opened by AudioTab when the
# user starts live transcription, closed (and final summary shown) on stop.
# ─────────────────────────────────────────────────────────────────────────────
class LiveTranscriptionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Transcription")
        self.resize(900, 720)
        self.setStyleSheet(
            f"QDialog {{ background:{BG_MID}; }}"
            f"QLabel {{ color:{TEXT_PRIMARY}; "
            f"font-family:'{FONT_SANS}'; }}")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(8)
        # Header
        head = QHBoxLayout()
        title = QLabel("Live Transcription")
        title.setStyleSheet(
            f"color:{TEXT_PRIMARY}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:16px; font-weight:700;")
        head.addWidget(title)
        head.addStretch(1)
        self.lbl_status = QLabel("listening\u2026")
        self.lbl_status.setStyleSheet(
            f"color:{ACCENT}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:11px;")
        head.addWidget(self.lbl_status)
        outer.addLayout(head)
        # Segmented transcript area — fills the entire window height.
        # Uses the same look as the audio tab's static transcript view
        # ([hh:mm:ss → hh:mm:ss] [Speaker] text) but is appended chunk
        # by chunk as whisper finishes them.
        self.txt_segments = QTextEdit()
        self.txt_segments.setReadOnly(True)
        self.txt_segments.setStyleSheet(
            "QTextEdit {"
            f"color:{TEXT_PRIMARY}; background: rgba(255,255,255,0.04);"
            f"border: 1px solid {GLASS_BORDER_SOFT}; border-radius: 10px;"
            f"padding: 10px; "
            f"font-family:'{FONT_MONO}','Consolas',monospace; font-size:12px;"
            "}"
            "QScrollBar:vertical{width:8px;background:transparent;}"
            "QScrollBar::handle:vertical{background:rgba(255,255,255,0.14);"
            "border-radius:4px;}")
        outer.addWidget(self.txt_segments, 1)   # stretch=1 → fills window
        # Bottom: a small "live (together)" panel that keeps the rolling
        # concatenated transcript so words flow naturally across chunks.
        sub = QLabel("Live (together)")
        sub.setStyleSheet(
            f"color:{TEXT_MUTED}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:12px; font-weight:700;")
        outer.addWidget(sub)
        self.txt_together = QTextEdit()
        self.txt_together.setReadOnly(True)
        self.txt_together.setFixedHeight(120)
        self.txt_together.setStyleSheet(self.txt_segments.styleSheet())
        outer.addWidget(self.txt_together)
        # Internal rolling state used to format new segments.
        self._elapsed = 0.0          # tracks total seconds across chunks
        self._segment_index = 0      # ordinal speaker label
    @staticmethod
    def _fmt_clock(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(int(m), 60)
        if h:
            return f"{h:02d}:{int(m):02d}:{s:05.2f}"
        return f"{int(m):02d}:{s:05.2f}"
    def append_segment(self, text: str, duration_sec: float,
                       speaker_label: str = "Unknown 1") -> None:
        """Append one rolling chunk produced by the whisper pipeline.
        `duration_sec` is the chunk window length (used to advance the
        running [start → end] timestamps). Text is appended verbatim — no
        truncation — so nothing in the transcript is ever lost."""
        text = (text or "").strip()
        if not text:
            return
        start = self._elapsed
        end = start + max(0.1, float(duration_sec))
        self._elapsed = end
        self._segment_index += 1
        line = (f"[{self._fmt_clock(start)} \u2192 {self._fmt_clock(end)}]  "
                f"[{speaker_label}]  {text}")
        cur = self.txt_segments.toPlainText().rstrip()
        new = (cur + "\n\n" + line) if cur else line
        self.txt_segments.setPlainText(new)
        sb = self.txt_segments.verticalScrollBar()
        sb.setValue(sb.maximum())
        # "Together" pane — natural concatenation across chunks.
        tcur = self.txt_together.toPlainText().strip()
        if not tcur:
            self.txt_together.setPlainText(text)
        else:
            if not tcur.endswith((" ", "\n")):
                tcur += " "
            self.txt_together.setPlainText(tcur + text)
        sb2 = self.txt_together.verticalScrollBar()
        sb2.setValue(sb2.maximum())
    def set_status(self, text: str, color: str = None) -> None:
        col = color if color else ACCENT
        self.lbl_status.setStyleSheet(
            f"color:{col}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:11px;")
        self.lbl_status.setText(text)
    def get_full_segments_text(self) -> str:
        return self.txt_segments.toPlainText()
    def get_full_together_text(self) -> str:
        return self.txt_together.toPlainText()
class AudioTab(QWidget):
    """Embedded glass audio dashboard. Names like _select / _on_transcribe_clicked
    match the chat tab's auto-transcribe hook so it drives this tab directly."""
    poll_signal = pyqtSignal()
    def __init__(self, parent, controller, app_config, location_tab=None,
                 switch=None):
        super().__init__(parent)
        self.controller = controller
        self.cfg = app_config
        self.location_tab = location_tab
        self._selected_path: Optional[str] = None
        self._rows: list[tuple[QPushButton, str]] = []
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Live wake-word listener state. See _on_live_transcribe_clicked.
        self._wake_active = False
        self._wake_dir: Optional[str] = None
        self._wake_parts_dir: Optional[str] = None
        self._wake_counter = 0
        self._wake_cooldown_until = 0.0
        self._wake_callback = None
        self._wake_owns_mic = False          # did we start the PC mic capture?
        self._wake_last_text = None          # last chunk shown (light dedup)
        self._wake_last_peek_ts = 0.0        # for adaptive window sizing
        self._live_dialog: Optional["LiveTranscriptionDialog"] = None
        # Live transcription rendering state. _live_elapsed advances by the
        # peek-window length per chunk so the [start -> end] timestamps stay
        # consistent across the whole session. Rendered into txt_transcript
        # directly now that the popup window has been removed.
        self._live_elapsed: float = 0.0
        self._live_segment_index: int = 0
        # When True, the transcript / summary panels show a live session
        # (in progress or just finished) and must not be overwritten by
        # incoming "transcribe_done" / "summary_done" events or by the
        # automatic re-show in _refresh_recordings(). Cleared as soon as
        # the user clicks any row in the Recordings list (via _select).
        self._live_panel_locked: bool = False
        # Without a backend, show a glass notice instead of crashing.
        if controller is None or app_config is None:
            self._build_notice()
            return
        self._build()
        self._bind_hotkeys()
        self._start_timers()
        self._refresh_recordings()
        if self.location_tab is not None:
            self.location_tab.refresh()
    # ---- config access with safe defaults ----
    def _c(self, attr, default):
        return getattr(self.cfg, attr, default) if self.cfg else default
    # ---- fallback notice (no backend) ----
    def _build_notice(self):
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        card = GlassFrame(self, radius=18, blur=30, dy=8)
        card.setMaximumWidth(520)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 24, 28, 26)
        t = QLabel("audio dashboard")
        t.setStyleSheet(f"color:{TEXT_PRIMARY}; background:transparent;"
                        f"border:none; font-family:'{FONT_SANS}';"
                        "font-size:18px; font-weight:700;")
        note = QLabel("The audio backend isn't loaded. Run iris_gui.py from the "
                      "project folder so config_phase9 and main_phase9 are "
                      "importable, and the full dashboard appears here.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{TEXT_MUTED}; background:transparent;"
                           f"border:none; font-family:'{FONT_SANS}';"
                           "font-size:12px;")
        cl.addWidget(t)
        cl.addWidget(note)
        wrap = QHBoxLayout()
        wrap.addStretch(1); wrap.addWidget(card); wrap.addStretch(1)
        outer.addLayout(wrap)
        outer.addStretch(2)
    # ---- layout: 2x2 glass grid ----
    def _build(self):
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 3)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.addWidget(self._panel(self._build_status_panel()), 0, 0)
        grid.addWidget(self._panel(self._build_recordings_panel()), 1, 0)
        # Transcript spans both rows -> fills the entire right half.
        grid.addWidget(self._panel(self._build_transcript_panel()), 0, 1, 2, 1)
    def _panel(self, inner: QWidget) -> QWidget:
        frame = GlassFrame(self, radius=16, blur=24, dy=6, shadow_alpha=120,
                           top="rgba(255,255,255,0.06)",
                           mid="rgba(255,255,255,0.035)",
                           bot="rgba(255,255,255,0.02)",
                           border=GLASS_BORDER_SOFT)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.addWidget(inner)
        return frame
    def _h(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{TEXT_PRIMARY}; background:transparent;"
                          f"border:none; font-family:'{FONT_SANS}';"
                          "font-size:15px; font-weight:700;")
        return lbl
    # ---- status panel ----
    def _build_status_panel(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background: transparent;")
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:8px;background:transparent;}"
            "QScrollBar::handle:vertical{background:rgba(255,255,255,0.14);"
            "border-radius:4px;}")
        scroll.setWidget(w)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(6)
        lay.addWidget(self._h("Status"))
        self.dot_wifi = StatusDot("Wi-Fi: waiting for ESP32")
        self.dot_stream = StatusDot("Audio stream: idle")
        self.dot_monitor = StatusDot("Monitoring: off")
        self.dot_location = StatusDot("Location: fetching\u2026")
        self.dot_wake = StatusDot("Live transcription: off")
        for d in (self.dot_wifi, self.dot_stream, self.dot_monitor,
                  self.dot_location, self.dot_wake):
            lay.addWidget(d)
        cap = QLabel("Input level")
        cap.setStyleSheet(f"color:{TEXT_DIM}; background:transparent;"
                          f"border:none; font-family:'{FONT_SANS}'; font-size:11px;")
        lay.addSpacing(6)
        lay.addWidget(cap)
        self.vu = VUMeter()
        lay.addWidget(self.vu)
        self.btn_record = _audio_btn("\u25CF  Start Recording",
                                     self._on_record_clicked,
                                     accent=_rgb(COLOR_DANGER), fg="#fca5a5",
                                     height=46, bold=True)
        self.btn_monitor = _audio_btn("\U0001F50A  Start Monitoring",
                                      self._on_monitor_clicked, height=40)
        self.btn_wake = _audio_btn("\U0001F399  Start Live Transcription",
                                   self._on_live_transcribe_clicked, height=40)
        self.btn_manage = _audio_btn("\U0001F464  Manage Speakers",
                                     self._open_manage_speakers, height=36)
        lay.addSpacing(4)
        lay.addWidget(self.btn_record)
        lay.addWidget(self.btn_monitor)
        lay.addWidget(self.btn_wake)
        lay.addWidget(self.btn_manage)
        # Queue / stats grid
        grid = QGridLayout()
        grid.setContentsMargins(0, 8, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        rows = [
            ("Recording:", "lbl_rec_duration", "--:--"),
            ("Chunk:", "lbl_rec_chunk", "--"),
            ("Transcribe queue:", "lbl_queue", "0"),
            ("Diarize queue:", "lbl_diarize_queue", "0"),
            ("Summarize queue:", "lbl_sum_queue", "0"),
            ("Packet loss:", "lbl_loss", "--"),
        ]
        for i, (label, attr, default) in enumerate(rows):
            k = QLabel(label)
            k.setStyleSheet(f"color:{TEXT_DIM}; background:transparent;"
                            f"border:none; font-family:'{FONT_SANS}'; font-size:11px;")
            v = QLabel(default)
            v.setStyleSheet(f"color:{TEXT_PRIMARY}; background:transparent;"
                            f"border:none; font-family:'{FONT_MONO}','Consolas',"
                            "monospace; font-size:11px; font-weight:700;")
            grid.addWidget(k, i, 0, Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(v, i, 1, Qt.AlignmentFlag.AlignLeft)
            setattr(self, attr, v)
        holder = QWidget(); holder.setStyleSheet("background:transparent;")
        holder.setLayout(grid)
        lay.addWidget(holder)
        lay.addStretch(1)
        return scroll
    # ---- transcript panel ----
    def _build_transcript_panel(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(self._h("Transcript"))
        head.addStretch(1)
        self.lbl_transcript_target = QLabel("(no recording selected)")
        self.lbl_transcript_target.setStyleSheet(
            f"color:{TEXT_DIM}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:11px;")
        head.addWidget(self.lbl_transcript_target)
        lay.addLayout(head)
        sh = QLabel("Summary")
        sh.setStyleSheet(f"color:{TEXT_MUTED}; background:transparent;"
                         f"border:none; font-family:'{FONT_SANS}'; font-size:13px;"
                         "font-weight:700;")
        lay.addWidget(sh)
        self.txt_summary = self._textbox(read_only=True, mono=False)
        self.txt_summary.setFixedHeight(150)
        lay.addWidget(self.txt_summary)
        # Transcript (rolling chunks)
        self.txt_transcript = self._textbox(read_only=True, mono=True)
        lay.addWidget(self.txt_transcript, 1)
        # No "Live (together)" panel below — txt_transcript now extends
        # to the bottom of the panel and is also used for live segments.
        self.txt_live_together = None
        btns = QHBoxLayout()
        btns.addWidget(_audio_btn("\U0001F464 Tag Speaker",
                                  self._on_tag_speaker_manual, height=30))
        btns.addStretch(1)
        btns.addWidget(_audio_btn("\u21bb Re-summarize",
                                  self._on_resummarize, height=30,
                                  accent=_rgb(ACCENT), fg=ACCENT))
        lay.addLayout(btns)
        return w
    def _textbox(self, read_only: bool, mono: bool) -> QTextEdit:
        t = QTextEdit()
        t.setReadOnly(read_only)
        fam = (f"'{FONT_MONO}','Consolas',monospace" if mono
               else f"'{FONT_SANS}'")
        t.setStyleSheet(
            "QTextEdit {"
            f"color:{TEXT_PRIMARY}; background: rgba(255,255,255,0.04);"
            f"border: 1px solid {GLASS_BORDER_SOFT}; border-radius: 10px;"
            f"padding: 8px; font-family:{fam}; font-size:12px;"
            "}"
            "QScrollBar:vertical{width:8px;background:transparent;}"
            "QScrollBar::handle:vertical{background:rgba(255,255,255,0.14);"
            "border-radius:4px;}")
        return t
    # ---- recordings panel ----
    def _build_recordings_panel(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(self._h("Recordings"))
        head.addStretch(1)
        for text, cmd in [("\u21bb", self._refresh_all),
                          ("\u25B6", self._on_play_clicked),
                          ("\U0001F4DD", self._on_transcribe_clicked),
                          ("\U0001F4C2", self._on_open_folder),
                          ("\u2B06", self._on_import_file)]:
            head.addWidget(_audio_btn(text, cmd, width=36, height=32,
                                      accent=_rgb(ACCENT), fg=ACCENT))
        lay.addLayout(head)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:8px;background:transparent;}"
            "QScrollBar::handle:vertical{background:rgba(255,255,255,0.14);"
            "border-radius:4px;}")
        self._list_holder = QWidget()
        self._list_holder.setStyleSheet("background: transparent;")
        self._list_lay = QVBoxLayout(self._list_holder)
        self._list_lay.setContentsMargins(0, 0, 6, 0)
        self._list_lay.setSpacing(2)
        self._list_lay.addStretch(1)
        scroll.setWidget(self._list_holder)
        lay.addWidget(scroll, 1)
        return w
    # ---- hotkeys (scoped to this tab so they don't hijack chat input) ----
    def _bind_hotkeys(self):
        binds = {"R": self._on_record_clicked, "M": self._on_monitor_clicked,
                 "P": self._on_play_clicked, "T": self._on_transcribe_clicked}
        for key, fn in binds.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
    # ---- timers (replace tkinter after-loops) ----
    def _start_timers(self):
        self._evt_timer = QTimer(self)
        self._evt_timer.timeout.connect(self._poll_events)
        self._evt_timer.start(int(self._c("GUI_POLL_MS", 100)))
        self._vu_timer = QTimer(self)
        self._vu_timer.timeout.connect(self._poll_vu)
        self._vu_timer.start(int(self._c("GUI_VU_DECAY_MS", 50)))
    def _poll_events(self):
        if self.controller is None:
            return
        try:
            while True:
                evt = self.controller.event_queue.get_nowait()
                self._handle_event(evt)
        except queue.Empty:
            pass
        except Exception:
            pass
    def _poll_vu(self):
        try:
            self.vu.setLevel(self.controller.peek_level())
        except Exception:
            pass
    def _handle_event(self, evt: dict):
        et = evt.get("type")
        if et == "esp32_connected":
            self.dot_stream.set(on=True, text="Audio stream: receiving")
            self.dot_wifi.set(on=True, text="Wi-Fi: ESP32 connected")
        elif et == "recording_started":
            self.btn_record.setText("\u25A0  Stop Recording")
            self.dot_stream.set(color=COLOR_RECORDING,
                                text=f"RECORDING ({evt.get('session', '')})")
        elif et == "recording_stopped":
            self.btn_record.setText("\u25CF  Start Recording")
            self.dot_stream.set(on=True, text="Audio stream: receiving")
            self.lbl_rec_duration.setText("--:--")
            self.lbl_rec_chunk.setText("--")
            self._refresh_all()
        elif et == "recording_tick":
            m, s = divmod(int(evt.get("duration", 0.0)), 60)
            self.lbl_rec_duration.setText(f"{m:02d}:{s:02d}")
            self.lbl_rec_chunk.setText(str(evt.get("chunk", "--")))
        elif et == "monitor_started":
            self.btn_monitor.setText("\U0001F507  Stop Monitoring")
            self.dot_monitor.set(on=True, text="Monitoring: on")
        elif et == "monitor_stopped":
            self.btn_monitor.setText("\U0001F50A  Start Monitoring")
            self.dot_monitor.set(on=False, text="Monitoring: off")
        elif et == "chunk_finalized":
            self._refresh_all()
        elif et in ("transcribe_done", "diarize_done", "summary_done"):
            self._refresh_recordings()
            if (not self._live_panel_locked
                    and self._selected_path == evt.get("wav")):
                self._show_content(self._selected_path)
        elif et == "transcribe_queue":
            self.lbl_queue.setText(str(evt.get("depth", 0)))
        elif et == "diarize_queue":
            self.lbl_diarize_queue.setText(str(evt.get("depth", 0)))
        elif et == "summarize_queue":
            self.lbl_sum_queue.setText(str(evt.get("depth", 0)))
        elif et == "net_stats":
            self.lbl_loss.setText(f"{evt.get('loss_pct', 0.0):.2f}%")
        elif et == "location_ready":
            loc = evt.get("location")
            if loc:
                place = f"{loc['city']}, {loc['region']}"
                self.dot_location.set(on=True, text=f"Location: {place}")
                if self.location_tab is not None:
                    self.location_tab.set_location(loc)
            else:
                self.dot_location.set(on=False, text="Location: unavailable")
    # ---- button handlers (same Controller calls as gui_phase9) ----
    def _on_record_clicked(self):
        try: self.controller.toggle_recording()
        except Exception: pass
    def _on_monitor_clicked(self):
        try: self.controller.toggle_monitoring()
        except Exception: pass
    def _on_play_clicked(self):
        if self._selected_path:
            try: self.controller.play_file(self._selected_path)
            except Exception: pass
    def _on_transcribe_clicked(self):
        # Tell the user what's happening - Whisper can take a while, and
        # without feedback the button looks like it did nothing.
        if not self._selected_path:
            try:
                self.txt_summary.setPlainText(
                    "Pick a recording from the list first, then click the "
                    "transcribe button (\U0001F4DD).")
            except Exception:
                pass
            return
        if self.controller is None or not hasattr(
                self.controller, "transcribe_file"):
            try:
                self.txt_summary.setPlainText(
                    "Transcription backend isn't available right now.")
            except Exception:
                pass
            return
        try:
            self.controller.transcribe_file(self._selected_path)
        except Exception as exc:
            try:
                self.txt_summary.setPlainText(
                    f"Couldn't start transcription: {exc}")
            except Exception:
                pass
            return
        # Show progress in the transcript panel so it's obvious the click
        # registered. The transcribe_done event will replace this with the
        # real transcript once Whisper finishes (usually 30-60s).
        try:
            name = os.path.basename(self._selected_path)
            self.lbl_transcript_target.setText(name)
            self.txt_transcript.setPlainText(
                f"Transcribing {name}\u2026 this can take 30-60 seconds "
                "depending on the recording length. The transcript will "
                "appear here automatically when it's ready.")
            self.txt_summary.setPlainText(
                "Transcription in progress\u2026 summary will appear after "
                "transcription finishes.")
        except Exception:
            pass
    def _on_resummarize(self):
        if self._selected_path:
            try: self.controller.summarize_file(self._selected_path)
            except Exception: pass
    def _on_open_folder(self):
        try:
            os.startfile(self._c("RECORDINGS_DIR", os.getcwd()))  # type: ignore
        except Exception:
            pass
    def _on_import_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import audio files", self._c("RECORDINGS_DIR", os.getcwd()),
            "WAV files (*.wav);;All files (*.*)")
        if not paths:
            return
        imported = 0
        dest = self._c("RECORDINGS_DIR", os.getcwd())
        for src in paths:
            dst = os.path.join(dest, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                try:
                    shutil.copy2(src, dst); imported += 1
                except Exception:
                    pass
        if imported:
            self._refresh_recordings()
    def _open_manage_speakers(self):
        try:
            ManageSpeakersDialog(self, self.controller.speaker_db,
                                 self._c("RECORDINGS_DIR", os.getcwd()),
                                 self._refresh_all).exec()
        except Exception as exc:
            print(f"[iris] manage speakers failed: {exc}")
    # ---- live wake-word listener ------------------------------------------
    # "Live transcription" here means: periodically peek a short rolling
    # window of the live ring buffer, run it through the EXISTING file-based
    # transcription queue (Controller.transcribe_file), and check the result
    # against iris_query.is_photo_trigger — the same check already gating the
    # typed-chat trigger. This is not literal word-by-word streaming ASR
    # (the underlying Transcriber is file-based, not a streaming model); it's
    # a practical approximation built on what's actually available. Windows
    # overlap (6s window, peeked every 3s) so a phrase near a window boundary
    # isn't split across two snippets and missed.
    # Whisper-small runs ~1.5-2x realtime on this CPU, so a ~6s window takes
    # ~9-12s to transcribe. The listener is sequential (one snippet at a time),
    # paced by transcription latency: peek -> transcribe -> show -> peek next.
    _WAKE_WINDOW_SECONDS = 6.0      # nominal window; actual size is adaptive
    _WAKE_WINDOW_MIN = 5.0
    _WAKE_WINDOW_MAX = 12.0
    _WAKE_CYCLE_MS = 300           # near-immediate next peek (latency paces us)
    _WAKE_COOLDOWN_SECONDS = 8.0
    _WAKE_POLL_MS = 700
    _WAKE_POLL_MAX = 40            # ~28s max wait — comfortably covers ~6-12s
    def set_wake_callback(self, fn) -> None:
        """Called with the heard phrase text when a wake trigger fires.
        Wired by IrisApp once the chat tab exists. No-op (listener still
        works, just doesn't act) if never set."""
        self._wake_callback = fn
    def _on_live_transcribe_clicked(self) -> None:
        if self._wake_active:
            self._stop_live_transcription()
        else:
            self._start_live_transcription()
    def _start_live_transcription(self) -> None:
        if not hasattr(self.controller, "peek_audio_wav"):
            self.dot_wake.set(
                on=False,
                text="Live transcription: unavailable (backend needs the "
                     "peek_audio_wav update)")
            return
        try:
            self._wake_dir = tempfile.mkdtemp(prefix="iris_wake_")
        except Exception:
            self.dot_wake.set(on=False,
                              text="Live transcription: couldn't start "
                                   "(no scratch directory)")
            return
        # Start mic capture FIRST so VU/input level can respond immediately
        # when live transcription begins (avoids the "first click doesn't
        # select a usable input device" behavior).
        self._wake_owns_mic = False
        source = "ESP32 stream"
        # helper: attempt mic once and log outcome
        def _try_mic_once() -> bool:
            if not hasattr(self.controller, "start_mic_capture"):
                return False
            try:
                ok = bool(self.controller.start_mic_capture())
            except Exception as e:
                ok = False
                print(f"[wake] start_mic_capture() exception: {e}")
            print(f"[wake] start_mic_capture() -> {ok}")
            return ok
        if _try_mic_once():
            self._wake_owns_mic = True
            source = "mic"
        else:
            print("[wake] mic unavailable on first attempt; retrying in 1.0s...")
            time.sleep(1.0)
            if _try_mic_once():
                self._wake_owns_mic = True
                source = "mic"
            else:
                print("[wake] mic still unavailable; falling back to ESP32 stream")
                self._set_live_panel(
                    "No microphone input device was available, so live "
                    "transcription is falling back to the ESP32 audio stream. "
                    "If no ESP32 is streaming, nothing will appear.\n")
        # Lock the transcript/summary panels for the live session so the
        # static "currently selected recording" view can't overwrite them.
        # Also drop the current selection so the refresh after stop doesn't
        # snap back to the old wav.
        self._live_panel_locked = True
        self._selected_path = None
        try:
            self._highlight()
        except Exception:
            pass
        # Live session aggregation state
        self._wake_active = True
        self._wake_counter = 0
        self._wake_cooldown_until = 0.0
        self._wake_last_text = None
        self._wake_last_peek_ts = time.time()
        # Save/catenate live audio into a single WAV on stop. The final
        # combined wav lands in RECORDINGS_DIR; the per-chunk snippets are
        # buffered in a temp scratch dir so they never appear in the
        # Recordings list and never clutter the user's folder.
        self._wake_session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._wake_live_stem = os.path.join(
            self._c("RECORDINGS_DIR", os.getcwd()),
            f"live_{self._wake_session_id}")
        try:
            self._wake_parts_dir = tempfile.mkdtemp(
                prefix=f"iris_live_parts_{self._wake_session_id}_")
        except Exception as e:
            # Fall back to RECORDINGS_DIR if mkdtemp fails (shouldn't happen,
            # but we still need _wake_parts_dir to be a string).
            print(f"[wake] could not create scratch dir, falling back: {e}")
            self._wake_parts_dir = self._c("RECORDINGS_DIR", os.getcwd())
        self._wake_copied_snippets: list[str] = []  # wav paths in scratch dir
        self._wake_output_wav = self._wake_live_stem + ".wav"
        self._wake_output_txt = self._wake_live_stem + ".txt"
        # No popup window: render the live transcript directly into the
        # existing txt_transcript box (which now extends to the bottom of
        # the panel since the "Live (together)" widget was removed).
        self.txt_transcript.setPlainText("")
        self.txt_summary.setPlainText(
            "Live transcription is on. Speak into the mic - rolling "
            "chunks will appear below. When you click Stop, the audio is "
            "saved as a new recording (named with the current date/time) "
            "and summarized so the chat can answer questions about it.")
        self._live_elapsed = 0.0
        self._live_segment_index = 0
        self.btn_wake.setText("\U0001F507  Stop Live Transcription")
        self.dot_wake.set(on=True,
                          text=f"Live transcription: listening ({source})\u2026")
        if self._wake_owns_mic:
            self._set_live_panel("")
        self.lbl_transcript_target.setText("(live transcription)")
        QTimer.singleShot(500, self._wake_cycle_peek)
    def _set_live_panel(self, text: str) -> None:
        try:
            self.txt_transcript.setPlainText(text)
        except Exception:
            pass
    def _stop_live_transcription(self) -> None:
        self._wake_active = False
        self.btn_wake.setText("\U0001F399  Start Live Transcription")
        self.dot_wake.set(on=False, text="Live transcription: off")
        if self._wake_owns_mic and hasattr(self.controller, "stop_mic_capture"):
            try:
                self.controller.stop_mic_capture()
            except Exception:
                pass
        self._wake_owns_mic = False
        # Concatenate copied snippets into one WAV and auto-summarize -
        # but only if actual speech was captured. _live_segment_index
        # increments once per non-empty Whisper chunk, so if it's 0 the
        # whole session was silence and we should NOT save a wav, fire
        # transcription, or run the summarizer (nothing to summarize).
        combined_ok = False
        spoke = getattr(self, "_live_segment_index", 0) > 0
        try:
            snippets = getattr(self, "_wake_copied_snippets", [])
            if snippets and spoke:
                self._concat_live_snippets_to_wav(snippets,
                                                   self._wake_output_wav)
                combined_ok = True
                # Queue a real transcription of the saved wav. The .json
                # sidecar Whisper produces is what the chat tab keys off,
                # so we don't need to also write a redundant .txt copy of
                # the on-screen segments here.
                try:
                    if hasattr(self.controller, "transcribe_file"):
                        self.controller.transcribe_file(self._wake_output_wav)
                except Exception:
                    pass
                # Auto-summarize the saved combined file
                try:
                    self.txt_summary.setPlainText(
                        "Live transcription stopped. Summarizing saved live audio…")
                    if hasattr(self.controller, "summarize_file"):
                        self.controller.summarize_file(self._wake_output_wav)
                except Exception:
                    pass
        except Exception as e:
            print(f"[wake] concat/summarize failed: {e}")
        # If the session was silent, tell the user clearly instead of
        # leaving an empty / stale panel.
        if not spoke:
            try:
                self.txt_summary.setPlainText(
                    "Live transcription stopped. Nothing was heard during "
                    "this session, so no recording was saved and there's "
                    "nothing to summarize. Click Start Live Transcription "
                    "again whenever you're ready.")
                self.txt_transcript.setPlainText(
                    "(no speech was captured in this session)")
                self.lbl_transcript_target.setText("(no audio captured)")
            except Exception:
                pass
        # Remove the per-chunk scratch dir now that we've concatenated
        # (or failed to). Either way the parts are no longer needed.
        try:
            parts_dir = getattr(self, "_wake_parts_dir", None)
            if (parts_dir and parts_dir != self._c("RECORDINGS_DIR", os.getcwd())
                    and os.path.isdir(parts_dir)):
                shutil.rmtree(parts_dir, ignore_errors=True)
        except Exception as e:
            print(f"[wake] scratch parts cleanup failed: {e}")
        self._wake_parts_dir = None
        # Cleanup scratch snippets + json from peek_audio_wav temp dir.
        # Important: the backend transcriber writes txt/json sidecars next to
        # the wav inside this same temp directory. If we delete the directory
        # immediately on stop, the transcriber thread can crash with
        # FileNotFoundError. So we delete with a delay.
        if self._wake_dir:
            wake_dir = self._wake_dir
            self._wake_dir = None
            def _delayed_cleanup():
                # Give the transcriber thread enough time to finish
                # writing snippet_XXXX.txt/json before removing the wake_dir.
                # Windows PortAudio/mic issues can delay the transcription
                # pipeline, so 10s was not sufficient.
                try:
                    time.sleep(60.0)
                    shutil.rmtree(wake_dir, ignore_errors=True)
                except Exception:
                    pass
            threading.Thread(target=_delayed_cleanup, daemon=True).start()
        # Refresh the Recordings list so the freshly saved wav appears
        # immediately (the transcribe/summary results stream in after).
        try:
            if combined_ok:
                self._refresh_recordings()
        except Exception:
            pass
        # Keep the live transcript visible after stop so the user can read
        # what was captured. If nothing was actually captured, fall back to
        # the previously selected recording (if any). When the session was
        # silent we already wrote a clear message into the panels above,
        # so leave them alone in that case.
        try:
            if not combined_ok and spoke:
                if self._selected_path:
                    self._show_content(self._selected_path)
                else:
                    self.lbl_transcript_target.setText("(no recording selected)")
            elif combined_ok:
                # Point the label at the new file so it's obvious which
                # recording this transcript belongs to.
                self.lbl_transcript_target.setText(
                    os.path.basename(self._wake_output_wav))
        except Exception:
            pass
    def _wake_cycle_peek(self) -> None:
        if not self._wake_active:
            return
        if time.time() < self._wake_cooldown_until:
            QTimer.singleShot(self._WAKE_CYCLE_MS, self._wake_cycle_peek)
            return
        self._wake_counter += 1
        snippet = os.path.join(self._wake_dir,
                                f"snippet_{self._wake_counter:04d}.wav")
        # Adaptive window: grab roughly the audio that accumulated since the
        # last peek (transcription latency means that's >6s), clamped, so we
        # don't drop speech that arrived while the previous chunk transcribed.
        now = time.time()
        window = now - self._wake_last_peek_ts
        window = max(self._WAKE_WINDOW_MIN,
                     min(self._WAKE_WINDOW_MAX, window))
        self._wake_last_peek_ts = now
        try:
            ok = self.controller.peek_audio_wav(window, snippet)
        except Exception:
            ok = False
        if not ok:
            QTimer.singleShot(self._WAKE_CYCLE_MS, self._wake_cycle_peek)
            return
        try:
            if hasattr(self.controller, "transcribe_file_only"):
                self.controller.transcribe_file_only(snippet)
            else:
                self.controller.transcribe_file(snippet)
        except Exception:
            self._cleanup_wake_snippet(snippet)
            QTimer.singleShot(self._WAKE_CYCLE_MS, self._wake_cycle_peek)
            return
        QTimer.singleShot(self._WAKE_POLL_MS,
                          lambda: self._wake_cycle_poll(snippet, 0))
    def _wake_cycle_poll(self, snippet: str, attempts: int) -> None:
        if not self._wake_active:
            self._cleanup_wake_snippet(snippet)
            return
        json_path = os.path.splitext(snippet)[0] + ".json"
        text = self._read_wake_transcript(json_path)
        if text is None and attempts < self._WAKE_POLL_MAX:
            QTimer.singleShot(
                self._WAKE_POLL_MS,
                lambda: self._wake_cycle_poll(snippet, attempts + 1))
            return
        # Copy each snippet into the temp scratch dir (NOT RECORDINGS_DIR)
        # so per-chunk WAVs never show up in the recordings list. They get
        # concatenated into the single live_<timestamp>.wav on stop, and
        # the scratch dir is removed afterwards.
        try:
            if os.path.exists(snippet) and self._wake_parts_dir:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                dst = os.path.join(
                    self._wake_parts_dir, f"part_{ts}.wav")
                shutil.copy2(snippet, dst)
                self._wake_copied_snippets.append(dst)
        except Exception as e:
            print(f"[wake] snippet copy failed: {e}")
        # Cleanup only json in temp dir; wav was copied and can be removed.
        self._cleanup_wake_snippet(snippet)
        # Show chunk in rolling transcript
        if text:
            self._append_live_text(text)
        if text and iq is not None and iq.is_photo_trigger(text):
            heard = text.strip()
            short = heard if len(heard) <= 50 else heard[:47] + "\u2026"
            self.dot_wake.set(
                on=True, text=f"Live transcription: heard \u201c{short}\u201d "
                             "\u2014 capturing\u2026")
            self._wake_cooldown_until = (
                time.time() + self._WAKE_COOLDOWN_SECONDS)
            if self._wake_callback is not None:
                try:
                    self._wake_callback(heard)
                except Exception:
                    pass
            QTimer.singleShot(1800, self._reset_wake_status)
        QTimer.singleShot(self._WAKE_CYCLE_MS, self._wake_cycle_peek)
    def _append_live_text(self, text: str) -> None:
        """Append one rolling transcript chunk to txt_transcript in the same
        segmented format the static transcript view uses:
           [hh:mm:ss -> hh:mm:ss]  [Speaker N]  text
        _live_elapsed tracks the cumulative seconds across chunks so the
        start/end timestamps stay continuous across the whole session."""
        text = (text or "").strip()
        if not text or text == self._wake_last_text:
            return
        self._wake_last_text = text
        # Advance the running clock by this chunk's window length
        # (clamped to the same min/max used when peeking the audio).
        try:
            window_sec = max(self._WAKE_WINDOW_MIN,
                             min(self._WAKE_WINDOW_MAX,
                                 float(self._WAKE_WINDOW_SECONDS)))
        except Exception:
            window_sec = float(self._WAKE_WINDOW_SECONDS)
        start = self._live_elapsed
        end = start + window_sec
        self._live_elapsed = end
        self._live_segment_index += 1
        speaker_label = f"Speaker {self._live_segment_index}"
        try:
            line = (f"[{self._fmt_ts(start)} \u2192 {self._fmt_ts(end)}]  "
                    f"[{speaker_label}]  {text}")
            cur = self.txt_transcript.toPlainText().rstrip()
            new = (cur + "\n\n" + line) if cur else line
            self.txt_transcript.setPlainText(new)
            sb = self.txt_transcript.verticalScrollBar()
            sb.setValue(sb.maximum())
        except Exception:
            pass
    def _reset_wake_status(self) -> None:
        if self._wake_active:
            self.dot_wake.set(on=True, text="Live transcription: listening\u2026")
    def _close_live_dialog_safely(self) -> None:
        try:
            if self._live_dialog is not None:
                self._live_dialog.close()
        except Exception:
            pass
    @staticmethod
    def _concat_live_snippets_to_wav(snippets: list, dest_path: str) -> None:
        """Concatenate the rolling whisper-window snippet WAVs (copied into
        RECORDINGS_DIR during live transcription) into one combined WAV at
        `dest_path`. Uses the built-in `wave` module so it has no extra
        dependency. Skips any snippet that can't be read instead of
        failing the whole save."""
        if not snippets:
            return
        params = None
        frames = []
        for path in snippets:
            try:
                with wave.open(path, "rb") as w:
                    p = w.getparams()
                    if params is None:
                        params = p
                    if (p.nchannels, p.sampwidth, p.framerate) != \
                            (params.nchannels, params.sampwidth,
                             params.framerate):
                        # Skip snippets with a different format rather than
                        # mangling them — keeps the combined file playable.
                        continue
                    frames.append(w.readframes(p.nframes))
            except Exception as e:
                print(f"[wake] skip {path}: {e}")
                continue
        if not frames or params is None:
            return
        try:
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            with wave.open(dest_path, "wb") as out:
                out.setnchannels(params.nchannels)
                out.setsampwidth(params.sampwidth)
                out.setframerate(params.framerate)
                for buf in frames:
                    out.writeframes(buf)
        except Exception as e:
            print(f"[wake] failed to write combined wav: {e}")
    @staticmethod
    def _read_wake_transcript(json_path: str) -> Optional[str]:
        """None = not ready yet (keep polling). '' = ready, but empty/no
        speech (stop polling, no trigger). Non-empty = ready, has text."""
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        t = data.get("transcript")
        if isinstance(t, str) and t.strip():
            return t
        segs = data.get("segments")
        if isinstance(segs, list):
            parts = [seg.get("text", "") for seg in segs
                     if isinstance(seg, dict)]
            joined = " ".join(p for p in parts if p).strip()
            if joined:
                return joined
        return ""
    @staticmethod
    def _cleanup_wake_snippet(snippet: str) -> None:
        # snippet wav is in temp dir; safe to remove after we've copied it.
        for p in (snippet, os.path.splitext(snippet)[0] + ".json"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    # ---- recordings list ----
    def _refresh_all(self):
        self._refresh_recordings()
        if self.location_tab is not None:
            self.location_tab.refresh()
    def _refresh_recordings(self):
        # One-time hygiene: stale per-chunk parts from older versions of
        # the live-transcription code were written into RECORDINGS_DIR.
        # Drop them so the list isn't polluted by hundreds of small
        # snippet wavs. We only delete files matching live_*_part_*.wav so
        # nothing else (real recordings, imports, etc.) is touched.
        try:
            rec_dir = self._c("RECORDINGS_DIR", os.getcwd())
            for stray in glob.glob(os.path.join(
                    rec_dir, "live_*_part_*.wav")):
                try:
                    os.remove(stray)
                except Exception:
                    pass
        except Exception:
            pass
        for btn, _ in self._rows:
            btn.deleteLater()
        self._rows.clear()
        # Sort by modification time (newest first) so a freshly saved
        # recording always lands at the top of the list, regardless of
        # filename prefix (live_/recording_/testing_upload, etc.).
        raw = glob.glob(os.path.join(
            self._c("RECORDINGS_DIR", os.getcwd()), "*.wav"))
        def _mtime_safe(p):
            try:
                return os.path.getmtime(p)
            except Exception:
                return 0.0
        files = sorted(raw, key=_mtime_safe, reverse=True)
        for path in files:
            btn = self._make_row(path)
            self._list_lay.insertWidget(self._list_lay.count() - 1, btn)
            self._rows.append((btn, path))
        if self._live_panel_locked:
            # A live session owns the transcript view; just rebuild the
            # rows and leave the panels alone.
            self._highlight()
        elif self._selected_path and self._selected_path in files:
            self._show_content(self._selected_path)
            self._highlight()
        elif files:
            self._select(files[0])
        else:
            self._show_content(None)
    def _make_row(self, path: str) -> QPushButton:
        base = os.path.splitext(path)[0]
        flags = ""
        if os.path.exists(base + ".txt"):            flags += "\u2713"
        if os.path.exists(base + ".embeddings.npz"): flags += "\U0001F464"
        if os.path.exists(base + ".summary.txt"):    flags += "\U0001F4CB"
        if os.path.exists(base + ".location.json"):  flags += "\U0001F4CD"
        if not flags:                                flags = "\u22EF"
        dur = self._wav_duration(path)
        m, s = divmod(int(dur), 60)
        name = os.path.basename(path)
        parts = name.replace("recording_", "").replace(".wav", "").split("_chunk")
        ts_part = parts[0] if len(parts) == 2 else name
        chunk = f"ch{parts[1]}" if len(parts) == 2 else ""
        label = f"  {ts_part.replace('_', ' ')}  {chunk}  {m:02d}:{s:02d}  {flags}"
        btn = QPushButton(label)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(32)
        self._style_row(btn, selected=False)
        btn.clicked.connect(lambda _=False, p=path: self._select(p))
        return btn
    def _style_row(self, btn: QPushButton, selected: bool):
        if selected:
            bg = f"rgba({_rgb(ACCENT)},0.14)"
            border = f"rgba({_rgb(ACCENT)},0.30)"
            fg = ACCENT
        else:
            bg = "transparent"
            border = "transparent"
            fg = TEXT_PRIMARY
        btn.setStyleSheet(
            "QPushButton {"
            f"color:{fg}; background:{bg}; border:1px solid {border};"
            "border-radius:8px; text-align:left; padding:0 8px;"
            f"font-family:'{FONT_MONO}','Consolas',monospace; font-size:11px;"
            "}"
            "QPushButton:hover { background: rgba(255,255,255,0.07); }")
    def _highlight(self):
        for btn, path in self._rows:
            self._style_row(btn, selected=(path == self._selected_path))
    def _select(self, path: str):
        # Clicking any recording releases the live-session lock so the
        # transcript / summary panels swap to the chosen recording.
        self._live_panel_locked = False
        self._selected_path = path
        self._show_content(path)
        self._highlight()
        loc = load_location_sidecar(path)
        if loc and self.location_tab is not None:
            self.location_tab.center_on((loc["lat"], loc["lon"]))
    # ---- content display ----
    def _show_content(self, path: Optional[str]):
        self._show_summary(path)
        self._show_transcript(path)
    def _show_summary(self, path: Optional[str]):
        if path is None:
            self.txt_summary.setPlainText("")
            return
        sp = os.path.splitext(path)[0] + ".summary.txt"
        if os.path.exists(sp):
            try:
                with open(sp, "r", encoding="utf-8") as f:
                    self.txt_summary.setPlainText(f.read().strip())
            except Exception:
                self.txt_summary.setPlainText("(error reading summary)")
        else:
            self.txt_summary.setPlainText(
                "No summary yet. Auto-summarize runs after transcription, "
                "or click \u21bb Re-summarize.")
    def _show_transcript(self, path: Optional[str]):
        if path is None:
            self.lbl_transcript_target.setText("(no recording selected)")
            self.txt_transcript.setPlainText("")
            return
        self.lbl_transcript_target.setText(os.path.basename(path))
        jp = os.path.splitext(path)[0] + ".json"
        if not os.path.exists(jp):
            self.txt_transcript.setPlainText(
                "No transcript yet. Click \U0001F4DD to generate one.")
            return
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.txt_transcript.setPlainText(f"(error reading JSON: {e})")
            return
        out = []
        show_ts = self._c("GUI_SHOW_TIMESTAMPS", False)
        for seg in data.get("segments", []):
            speaker = seg.get("speaker")
            conf = seg.get("speaker_confidence", 0.0)
            kind = seg.get("speaker_kind", "unknown")
            text = seg.get("text", "").strip()
            line = ""
            if show_ts:
                line += (f"[{self._fmt_ts(seg['start'])} \u2192 "
                         f"{self._fmt_ts(seg['end'])}]  ")
            if speaker:
                line += (f"[{speaker} \u2014 {conf:.0%}]  " if kind == "weak"
                         else f"[{speaker}]  ")
            line += text
            out.append(line)
        self.txt_transcript.setPlainText("\n\n".join(out))
    # ---- tag speaker (port of _on_tag_speaker_manual) ----
    def _on_tag_speaker_manual(self):
        if not self._selected_path:
            return
        jp = os.path.splitext(self._selected_path)[0] + ".json"
        if not os.path.exists(jp):
            return
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        segments = data.get("segments", [])
        if not segments:
            return
        labels = list(dict.fromkeys(
            seg.get("speaker", "Unknown") for seg in segments
            if seg.get("speaker")))
        if not labels:
            for seg in segments:
                seg["speaker"] = "Speaker 1"
                seg["speaker_kind"] = "unknown"
                seg["speaker_confidence"] = 0.0
            labels = ["Speaker 1"]
        dlg = QDialog(self)
        dlg.setWindowTitle("Tag Speaker")
        dlg.resize(420, 240)
        dlg.setStyleSheet(f"QDialog {{ background:{BG_MID}; }}"
                          f"QLabel {{ color:{TEXT_PRIMARY};"
                          f"font-family:'{FONT_SANS}'; }}")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Who is speaking in this recording?"))
        cap = QLabel("Pick the current label, then enter the real name.")
        cap.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px;")
        v.addWidget(cap)
        v.addWidget(QLabel("Current label in transcript:"))
        combo = QComboBox()
        combo.addItems(labels)
        combo.setStyleSheet(
            f"QComboBox {{ color:{TEXT_PRIMARY}; background:rgba(255,255,255,0.06);"
            f"border:1px solid {GLASS_BORDER_SOFT}; border-radius:8px;"
            "padding:4px 8px; }")
        v.addWidget(combo)
        v.addWidget(QLabel("Real name (who this actually is):"))
        entry = QLineEdit()
        entry.setPlaceholderText("e.g. Humza, Mom, \u2026")
        entry.setStyleSheet(
            f"QLineEdit {{ color:{TEXT_PRIMARY}; background:rgba(255,255,255,0.06);"
            f"border:1px solid {GLASS_BORDER_SOFT}; border-radius:8px;"
            "padding:6px 8px; }")
        v.addWidget(entry)
        def _save():
            old = combo.currentText()
            new = entry.text().strip()
            if not new:
                return
            for seg in segments:
                if seg.get("speaker") == old:
                    seg["speaker"] = new
                    seg["speaker_kind"] = "strict"
                    seg["speaker_confidence"] = 1.0
            data["diarized"] = True
            try:
                with open(jp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"[gui] could not save speaker tag: {e}")
                dlg.reject(); return
            db = getattr(self.controller, "speaker_db", None)
            if db is not None:
                try:
                    if new not in db.list_names():
                        import numpy as _np
                        db.create(new, _np.zeros(192, dtype=_np.float32))
                except Exception as e:
                    print(f"[gui] could not create placeholder profile: {e}")
                emb_path = (os.path.splitext(self._selected_path)[0]
                            + ".embeddings.npz")
                if os.path.exists(emb_path):
                    try:
                        import numpy as np
                        npz = np.load(emb_path)
                        cids = list({seg.get("_cluster", -1) for seg in segments
                                     if seg.get("speaker") == new
                                     and seg.get("_cluster", -1) >= 0})
                        for cid in cids:
                            key = f"cluster_{cid}"
                            if key in npz:
                                db.add_to(new, npz[key])
                    except Exception as e:
                        print(f"[gui] could not save voiceprint: {e}")
            dlg.accept()
            self._show_content(self._selected_path)
            self._refresh_recordings()
        entry.returnPressed.connect(_save)
        row = QHBoxLayout()
        row.addWidget(_audio_btn("Save", _save, accent=_rgb(COLOR_STATUS_ON),
                                 fg="#86efac", width=90))
        row.addWidget(_audio_btn("Cancel", dlg.reject, width=90))
        row.addStretch(1)
        v.addLayout(row)
        dlg.exec()
    @staticmethod
    def _fmt_ts(s: float) -> str:
        m = int(s // 60); sec = s - m * 60
        return f"{m:02d}:{sec:05.2f}"
    @staticmethod
    def _wav_duration(path: str) -> float:
        try:
            with wave.open(path, "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0
# ─────────────────────────────────────────────────────────────────────────────
# Location tab — the map (Leaflet in QWebEngineView, else a located-recordings
# list). Driven by the audio tab: location events + recording selection.
# ─────────────────────────────────────────────────────────────────────────────
class LocationTab(QWidget):
    def __init__(self, parent, app_config):
        super().__init__(parent)
        self.cfg = app_config
        self._map_view = None
        self._map_note = None
        if app_config is None:
            self._build_notice()
            return
        self._build()
        self.refresh()
    def _c(self, attr, default):
        return getattr(self.cfg, attr, default) if self.cfg else default
    def _build_notice(self):
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        card = GlassFrame(self, radius=18, blur=30, dy=8)
        card.setMaximumWidth(520)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 24, 28, 26)
        t = QLabel("location & gps")
        t.setStyleSheet(f"color:{TEXT_PRIMARY}; background:transparent;"
                        f"border:none; font-family:'{FONT_SANS}';"
                        "font-size:18px; font-weight:700;")
        note = QLabel("Location backend isn't loaded. Run iris_gui.py from the "
                      "project folder so recordings and their location "
                      "sidecars are available, and the map appears here.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{TEXT_MUTED}; background:transparent;"
                           f"border:none; font-family:'{FONT_SANS}';"
                           "font-size:12px;")
        cl.addWidget(t)
        cl.addWidget(note)
        wrap = QHBoxLayout()
        wrap.addStretch(1); wrap.addWidget(card); wrap.addStretch(1)
        outer.addLayout(wrap)
        outer.addStretch(2)
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = GlassFrame(self, radius=16, blur=24, dy=6, shadow_alpha=120,
                           top="rgba(255,255,255,0.06)",
                           mid="rgba(255,255,255,0.035)",
                           bot="rgba(255,255,255,0.02)",
                           border=GLASS_BORDER_SOFT)
        outer.addWidget(frame)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)
        head = QHBoxLayout()
        title = QLabel("location & gps")
        title.setStyleSheet(f"color:{TEXT_PRIMARY}; background:transparent;"
                            f"border:none; font-family:'{FONT_SANS}';"
                            "font-size:15px; font-weight:700;")
        head.addWidget(title)
        head.addStretch(1)
        self.lbl_location = QLabel("")
        self.lbl_location.setStyleSheet(
            f"color:{TEXT_DIM}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:11px;")
        head.addWidget(self.lbl_location)
        lay.addLayout(head)
        if QWebEngineView is not None:
            try:
                self._map_view = QWebEngineView()
                self._map_view.setStyleSheet("border-radius:10px;")
                lay.addWidget(self._map_view, 1)
            except Exception:
                self._map_view = None
        if self._map_view is None:
            self._map_note = QTextEdit()
            self._map_note.setReadOnly(True)
            self._map_note.setStyleSheet(
                "QTextEdit {"
                f"color:{TEXT_PRIMARY}; background: rgba(255,255,255,0.04);"
                f"border: 1px solid {GLASS_BORDER_SOFT}; border-radius: 10px;"
                f"padding: 10px; font-family:'{FONT_MONO}','Consolas',monospace;"
                "font-size:12px; }")
            self._map_note.setPlainText(
                "Map needs PyQt6-WebEngine.\n"
                "  pip install PyQt6-WebEngine\n\n"
                "Located recordings will be listed here until it's installed.")
            lay.addWidget(self._map_note, 1)
    # ---- public API used by the audio tab ----
    def set_location(self, loc: dict):
        if self.cfg is None:
            return
        try:
            self.lbl_location.setText(f"{loc['city']}, {loc['region']}")
            self.center_on((loc["lat"], loc["lon"]))
        except Exception:
            pass
    def center_on(self, latlon):
        if self.cfg is not None:
            self._render(center=latlon)
    def refresh(self):
        if self.cfg is not None:
            self._render()
    # ---- render ----
    def _render(self, center=None):
        files = sorted(glob.glob(os.path.join(
            self._c("RECORDINGS_DIR", os.getcwd()), "*.wav")))
        located = []
        for path in files:
            loc = load_location_sidecar(path)
            if loc:
                located.append((loc["lat"], loc["lon"], path))
        if self._map_view is None:
            if self._map_note is not None:
                if located:
                    lines = ["Located recordings:\n"]
                    for lat, lon, p in located:
                        lines.append(f"  \u2022 {os.path.basename(p)}  "
                                     f"({lat:.4f}, {lon:.4f})")
                    self._map_note.setPlainText("\n".join(lines))
                else:
                    self._map_note.setPlainText("No located recordings yet.")
            return
        if center is None:
            center = ((located[0][0], located[0][1]) if located
                      else (self._c("MAP_FALLBACK_LAT", 0.0),
                            self._c("MAP_FALLBACK_LON", 0.0)))
        self._map_view.setHtml(self._map_html(located, center))
    def _map_html(self, located, center) -> str:
        tile = self._c("MAP_TILE_URL",
                       "https://tile.openstreetmap.org/{z}/{x}/{y}.png")
        zoom = int(self._c("MAP_DEFAULT_ZOOM", 13))
        clusters = self._cluster_pins(
            located, self._c("MAP_CLUSTER_RADIUS_M", 60))
        markers = []
        for cl in clusters:
            lat = sum(c[0] for c in cl) / len(cl)
            lon = sum(c[1] for c in cl) / len(cl)
            if len(cl) > 1:
                text = f"{len(cl)} recordings"
            else:
                text = (os.path.basename(cl[0][2]).split("_chunk")[0]
                        .replace("recording_", ""))
            markers.append(f"L.marker([{lat},{lon}]).addTo(map)"
                           f".bindPopup({json.dumps(text)});")
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>"
            "<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>"
            "<style>html,body,#m{height:100%;margin:0;background:#0b1120;}</style>"
            "</head><body><div id='m'></div><script>"
            f"var map=L.map('m').setView([{center[0]},{center[1]}],{zoom});"
            f"L.tileLayer({json.dumps(tile)},{{maxZoom:19}}).addTo(map);"
            + "".join(markers) +
            "</script></body></html>")
    @staticmethod
    def _cluster_pins(points, radius_m):
        unassigned = list(points)
        clusters = []
        while unassigned:
            seed = unassigned.pop(0)
            cluster = [seed]
            remaining = []
            for p in unassigned:
                if any(LocationTab._hav_m(p[0], p[1], q[0], q[1]) <= radius_m
                       for q in cluster):
                    cluster.append(p)
                else:
                    remaining.append(p)
            unassigned = remaining
            clusters.append(cluster)
        return clusters
    @staticmethod
    def _hav_m(lat1, lon1, lat2, lon2):
        R = 6_371_000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = (math.sin(dp / 2) ** 2
             + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(a))
# ─────────────────────────────────────────────────────────────────────────────
# Photos tab — gallery of everything captured via "hey iris, take a photo" or
# the manual camera button. Backed by iris_photos.PhotoStore, the same module
# ChatTab uses, both pointed at the same <recordings root>/photos folder.
# ─────────────────────────────────────────────────────────────────────────────
class PhotosTab(QWidget):
    THUMB = 150
    COLS = 5
    def __init__(self, parent, app_config, on_select=None):
        super().__init__(parent)
        self.cfg = app_config
        self._on_select = on_select
        self._store = iphotos.PhotoStore(_photos_dir()) if iphotos is not None else None
        self._build()
        self.refresh()
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = GlassFrame(self, radius=16, blur=24, dy=6, shadow_alpha=120,
                           top="rgba(255,255,255,0.06)",
                           mid="rgba(255,255,255,0.035)",
                           bot="rgba(255,255,255,0.02)",
                           border=GLASS_BORDER_SOFT)
        outer.addWidget(frame)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        head = QHBoxLayout()
        title = QLabel("photos")
        title.setStyleSheet(f"color:{TEXT_PRIMARY}; background:transparent;"
                            f"border:none; font-family:'{FONT_SANS}';"
                            "font-size:15px; font-weight:700;")
        head.addWidget(title)
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet(
            f"color:{TEXT_DIM}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:11px; padding-left:8px;")
        head.addWidget(self.lbl_count)
        head.addStretch(1)
        head.addWidget(_audio_btn("\u21bb Refresh", self.refresh, height=30,
                                  accent=_rgb(ACCENT), fg=ACCENT))
        head.addWidget(_audio_btn("\U0001F4C2 Open folder", self._open_folder,
                                  height=30))
        lay.addLayout(head)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:8px;background:transparent;}"
            "QScrollBar::handle:vertical{background:rgba(255,255,255,0.14);"
            "border-radius:4px;}")
        self._grid_holder = QWidget()
        self._grid_holder.setStyleSheet("background: transparent;")
        self._grid = QGridLayout(self._grid_holder)
        self._grid.setContentsMargins(2, 2, 2, 2)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(10)
        scroll.setWidget(self._grid_holder)
        lay.addWidget(scroll, 1)
        self._empty_note = QLabel(
            "No photos yet. Say \u201chey iris, take a photo\u201d in the "
            "Chat tab, or use the \U0001F4F7 button there.")
        self._empty_note.setWordWrap(True)
        self._empty_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_note.setStyleSheet(
            f"color:{TEXT_MUTED}; background:transparent; border:none;"
            f"font-family:'{FONT_SANS}'; font-size:12px; padding: 30px;")
        lay.addWidget(self._empty_note)
    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
    def refresh(self) -> None:
        if self._store is None:
            self.lbl_count.setText("(iris_photos.py missing)")
            return
        self._clear_grid()
        photos = self._store.list_all()         # already newest-first
        self.lbl_count.setText(
            f"{len(photos)} photo{'s' if len(photos) != 1 else ''}")
        self._empty_note.setVisible(not photos)
        cols = self.COLS
        for i, p in enumerate(photos):
            tag = _photo_source_label(p.source)
            caption = f"{p.when()}\n{tag}"
            on_click = ((lambda ph=p: self._on_select(ph))
                       if self._on_select is not None else None)
            thumb = PhotoThumb(self._grid_holder, p.path, caption,
                               size=self.THUMB, on_click=on_click)
            self._grid.addWidget(thumb, i // cols, i % cols,
                                 Qt.AlignmentFlag.AlignLeft
                                 | Qt.AlignmentFlag.AlignTop)
    def _open_folder(self) -> None:
        if self._store is None:
            return
        try:
            os.startfile(self._store.dir)              # type: ignore
        except Exception:
            try:
                subprocess.Popen(["xdg-open", self._store.dir])
            except Exception:
                pass
    def showEvent(self, event) -> None:
        self.refresh()
        super().showEvent(event)
# ─────────────────────────────────────────────────────────────────────────────
# Top tab bar — glass segmented buttons (chat / audio / location / people / stream)
# ─────────────────────────────────────────────────────────────────────────────
class TabBar(QWidget):
    changed = pyqtSignal(int)
    def __init__(self, parent, labels: list[str]):
        super().__init__(parent)
        self._buttons: list[QPushButton] = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 14, 0, 6)
        lay.setSpacing(6)
        lay.addStretch(1)
        for i, name in enumerate(labels):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, idx=i: self._select(idx))
            self._buttons.append(b)
            lay.addWidget(b)
        lay.addStretch(1)
        self._select(0)
    def _select(self, idx: int) -> None:
        for i, b in enumerate(self._buttons):
            on = (i == idx)
            b.setChecked(on)
            if on:
                b.setStyleSheet(
                    "QPushButton {"
                    f"color:{ACCENT};"
                    f"background: rgba({_rgb(ACCENT)},0.14);"
                    f"border: 1px solid rgba({_rgb(ACCENT)},0.30);"
                    "border-radius: 13px; padding: 6px 18px;"
                    f"font-family:'{FONT_MONO}','Consolas',monospace; font-size:13px;"
                    "}")
            else:
                b.setStyleSheet(
                    "QPushButton {"
                    f"color:{TEXT_MUTED};"
                    "background: transparent; border: 1px solid transparent;"
                    "border-radius: 13px; padding: 6px 18px;"
                    f"font-family:'{FONT_MONO}','Consolas',monospace; font-size:13px;"
                    "}"
                    "QPushButton:hover { background: rgba(255,255,255,0.06); }")
        self.changed.emit(idx)
# ─────────────────────────────────────────────────────────────────────────────
# Title strip — macOS-style traffic-light controls + live session timer.
# The whole strip is the window's drag handle (frameless windows have none).
# ─────────────────────────────────────────────────────────────────────────────
class TitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedHeight(44)
        self._drag: Optional[QPoint] = None
        self._secs = 0
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 0, 20, 0)
        lay.setSpacing(8)
        # Traffic lights — functional: close / minimise / maximise
        lay.addWidget(self._dot("#ff5f57", self._close))   # red
        lay.addWidget(self._dot("#febc2e", self._minimise)) # yellow
        lay.addWidget(self._dot("#28c840", self._maximise)) # green
        lay.addStretch(1)
        self.session = QLabel("iris \u00b7 session 00:00:00")
        self.session.setStyleSheet(
            f"color:{TEXT_DIM}; background:transparent; border:none;"
            f"font-family:'{FONT_MONO}','Consolas',monospace; font-size:12px;")
        lay.addWidget(self.session)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
    def _dot(self, color: str, on_click) -> QPushButton:
        b = QPushButton()
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFixedSize(13, 13)
        b.setStyleSheet(
            "QPushButton {"
            f"background:{color}; border:none; border-radius:6px;"
            "}"
            "QPushButton:hover { border: 1px solid rgba(0,0,0,0.25); }")
        b.clicked.connect(on_click)
        return b
    def _tick(self) -> None:
        self._secs += 1
        h, rem = divmod(self._secs, 3600)
        m, s = divmod(rem, 60)
        self.session.setText(f"iris \u00b7 session {h:02d}:{m:02d}:{s:02d}")
    def _close(self):     self.window().close()
    def _minimise(self):  self.window().showMinimized()
    def _maximise(self):
        w = self.window()
        w.showNormal() if w.isMaximized() else w.showMaximized()
    # Drag the frameless window by its title strip
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = (e.globalPosition().toPoint()
                          - self.window().frameGeometry().topLeft())
            e.accept()
    def mouseMoveEvent(self, e):
        if self._drag is not None and (e.buttons() & Qt.MouseButton.LeftButton):
            self.window().move(e.globalPosition().toPoint() - self._drag)
            e.accept()
    def mouseReleaseEvent(self, e):
        self._drag = None
    def mouseDoubleClickEvent(self, e):
        self._maximise()
# ─────────────────────────────────────────────────────────────────────────────
# Main IRIS window — a single rounded, frameless "bubble" floating on the desktop
# ─────────────────────────────────────────────────────────────────────────────
class IrisApp(QWidget):
    TAB_NAMES = ["chat", "audio", "location", "people", "stream", "photos"]
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("iris")
        self.resize(1400, 850)
        self.setMinimumSize(1100, 700)
        # Frameless + translucent so the rounded corners show the desktop behind
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.titlebar = TitleBar(self)
        root.addWidget(self.titlebar)
        self.tabbar = TabBar(self, self.TAB_NAMES)
        root.addWidget(self.tabbar)
        # Body holds the tab stack inside a small gutter so panels float and the
        # rounded window corners stay clean (no square widget pokes through).
        body = QWidget(self)
        body.setStyleSheet("background: transparent;")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(14, 0, 14, 14)
        bl.setSpacing(0)
        self.stack = QStackedWidget(body)
        bl.addWidget(self.stack)
        root.addWidget(body, 1)
        # Location tab built first so the audio tab can drive its map.
        self.location = LocationTab(self, config)
        # Audio tab built next so the chat can drive it for auto-transcription.
        self.audio = AudioTab(self, controller, config,
                              location_tab=self.location,
                              switch=lambda: self.tabbar._select(1))
        # Chat first in the stack; it can switch to the audio tab + drive it.
        self.chat = ChatTab(
            self, controller=controller, audio_gui=self.audio,
            switch_to_audio=lambda: self.tabbar._select(1))
        # Set up the stream tab first so the chat can route ESP32 selfies
        # to it. The stream tab handles its own ESP32 networking exactly
        # like terminal.py and is the right home for "of me" photos.
        self.stream = StreamTab(self)
        # Point the chat's video store at the Stream tab's real save folder
        # (from terminal.py's SAVE_FOLDER) so "how many people were in the
        # video?" reads the exact clips the receiver just wrote to disk,
        # instead of only the auto-guessed default locations.
        try:
            if getattr(self.chat, "_videos", None) is not None:
                for attr in ("SAVE_FOLDER", "PHOTO_FOLDER"):
                    folder = getattr(self.stream, attr, None)
                    if folder:
                        self.chat._videos.add_folder(folder)
                try:
                    n = len(self.chat._videos.list_all())
                    print(f"[video] chat can see {n} saved clip(s) across: "
                          f"{self.chat._videos.folders()}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[video] could not link stream folder to chat: {e}")
        # Give the chat tab a way to fire the ESP32 photo command + show
        # the stream tab while it captures. The chat decides whether a
        # request is "of me" (selfie) vs a bare "take a photo"
        # (screenshot) inside _trigger_photo_capture.
        def _stream_photo_callback():
            try:
                self.tabbar._select(4)            # switch to stream tab
                self.stream._request_photo()
            except Exception as e:
                print(f"[iris] stream photo callback failed: {e}")
        self.chat._stream_photo_callback = _stream_photo_callback
        # When the ESP32 photo arrives, hand it back to the chat tab
        # (which adds it to the Photos tab and posts an inline bubble)
        # and switch the view back to chat so the user sees the result.
        def _on_esp32_photo_arrived(jpeg_path):
            try:
                self.chat._on_esp32_photo_arrived(jpeg_path)
                self.tabbar._select(0)            # back to chat tab
                if hasattr(self, "photos"):
                    try:
                        self.photos.refresh()
                    except Exception:
                        pass
            except Exception as e:
                print(f"[iris] photo arrival handler failed: {e}")
        self.stream._on_photo_arrived_cb = _on_esp32_photo_arrived
        def _on_wake_trigger(phrase):
            # "Hey Jarvis, take a picture of me" → ESP32 camera in the
            # stream tab. Other wake-word photo requests fall through to
            # the chat (which captures with the webcam).
            if ChatTab._wants_esp32_selfie(phrase or ""):
                try:
                    _stream_photo_callback()
                    return
                except Exception:
                    pass
            self.chat.handle_voice_trigger(phrase)
            self.tabbar._select(0)
        self.audio.set_wake_callback(_on_wake_trigger)
        self.stack.addWidget(self.chat)
        self.stack.addWidget(self.audio)
        self.stack.addWidget(self.location)
        self.people = PeopleTab(self, controller)
        self.stack.addWidget(self.people)
        self.stack.addWidget(self.stream)
        def _select_photo_from_gallery(photo):
            self.chat.select_photo(photo)
            self.tabbar._select(0)
        self.photos = PhotosTab(self, config, on_select=_select_photo_from_gallery)
        self.stack.addWidget(self.photos)
        self.tabbar.changed.connect(self.stack.setCurrentIndex)
        self.stack.setCurrentIndex(0)
        # Bottom-right resize grip (frameless windows lose native resizing)
        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(18, 18)
        self._grip.setStyleSheet("background: transparent;")
    # Paint the rounded gradient shell + a thin outline = the "bubble"
    def paintEvent(self, _evt):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, WINDOW_RADIUS, WINDOW_RADIUS)
        g = QLinearGradient(0, 0, self.width(), self.height())
        g.setColorAt(0.0, QColor(BG_TOP))
        g.setColorAt(0.55, QColor(BG_MID))
        g.setColorAt(1.0, QColor(BG_BOT))
        p.fillPath(path, QBrush(g))
        pen = QPen(WINDOW_OUTLINE)
        pen.setWidth(1)
        p.setPen(pen)
        p.drawPath(path)
    def resizeEvent(self, evt):
        self._grip.move(self.width() - self._grip.width() - 8,
                        self.height() - self._grip.height() - 8)
        super().resizeEvent(evt)
    def closeEvent(self, evt):
        try:
            if hasattr(self, "people") and self.people is not None:
                self.people.shutdown()
        except Exception:
            pass
        try:
            if self.controller is not None:
                self.controller.shutdown()
        except Exception:
            pass
        super().closeEvent(evt)
def main() -> int:
    app = QApplication(sys.argv)
    # Prefer Cascadia Code; fall back to a mono the OS has.
    families = set(QFontDatabase.families())
    mono = ("Cascadia Code" if "Cascadia Code" in families else
            "Consolas" if "Consolas" in families else "Monospace")
    globals()["FONT_MONO"] = mono
    app.setFont(QFont(FONT_SANS if FONT_SANS in families else "Sans", 10))
    controller = None
    if Controller is not None:
        try:
            controller = Controller()
            if hasattr(controller, "start"):
                controller.start()
        except Exception as exc:
            print(f"[iris] backend controller unavailable: {exc}")
            controller = None
    win = IrisApp(controller)
    win.show()
    try:
        return app.exec()
    finally:
        try:
            if controller is not None:
                controller.shutdown()
        except Exception:
            pass
if __name__ == "__main__":
    sys.exit(main())