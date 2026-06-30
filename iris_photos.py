"""
iris_photos.py — photo capture persistence for the IRIS Photos tab.

Pure Python, no Qt. Manages the folder + JSON-sidecar metadata for every
captured photo (timestamp, trigger phrase, source) and lists them
newest-first for the gallery. The actual image bytes are written by the
caller — taking a screenshot needs Qt (QGuiApplication), so that part lives
in iris_gui.py. This module only owns "where photos live and how to find
them," the same split used by iris_sessions.py for chat history.
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


@dataclass
class Photo:
    path: str
    taken_at: float
    source: str = "screenshot"      # "screenshot" | "esp32"
    trigger_text: str = ""
    note: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def when(self) -> str:
        try:
            return datetime.fromtimestamp(self.taken_at).strftime("%b %d %H:%M:%S")
        except Exception:
            return "—"


class PhotoStore:
    """Owns one photos directory. Never raises out to the caller."""

    def __init__(self, directory: str):
        self.dir = directory
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception:
            pass

    # ── writing ──────────────────────────────────────────────────────────
    def new_path(self, ext: str = "png") -> str:
        """A fresh, collision-free path inside this store for a new photo."""
        ext = (ext or "png").lstrip(".").lower()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return os.path.join(self.dir, f"photo_{ts}.{ext}")

    def record(self, image_path: str, source: str = "screenshot",
               trigger_text: str = "", note: str = "") -> Photo:
        """Write the metadata sidecar for an already-saved image file."""
        taken_at = time.time()
        meta = {"taken_at": taken_at, "source": source,
                "trigger_text": trigger_text, "note": note}
        sidecar = os.path.splitext(image_path)[0] + ".json"
        try:
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass
        return Photo(path=image_path, taken_at=taken_at, source=source,
                     trigger_text=trigger_text, note=note)

    # ── reading ──────────────────────────────────────────────────────────
    def list_all(self) -> list[Photo]:
        out: list[Photo] = []
        try:
            files = os.listdir(self.dir)
        except Exception:
            return out
        for fn in files:
            if not fn.lower().endswith(_IMAGE_EXTS):
                continue
            full = os.path.join(self.dir, fn)
            meta = self._read_sidecar(full)
            taken_at = meta.get("taken_at")
            if taken_at is None:
                try:
                    taken_at = os.path.getmtime(full)
                except Exception:
                    taken_at = 0.0
            out.append(Photo(
                path=full, taken_at=float(taken_at),
                source=meta.get("source", "screenshot"),
                trigger_text=meta.get("trigger_text", ""),
                note=meta.get("note", ""),
            ))
        out.sort(key=lambda p: p.taken_at, reverse=True)
        return out

    @staticmethod
    def _read_sidecar(image_path: str) -> dict:
        sidecar = os.path.splitext(image_path)[0] + ".json"
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    # ── watching an external folder (the existing ESP32 receiver's output) ─
    @staticmethod
    def newest_new_file(folder: str, since: float,
                        exclude: Optional[set] = None) -> Optional[str]:
        """The newest image in `folder` modified after `since`, skipping any
        path in `exclude`. Used to detect a freshly-arrived ESP32 photo that
        the existing receiver app (terminal.py) already saved to disk — we
        watch its output folder rather than opening our own listening socket,
        since that receiver already owns ports 5010/5011 on the PC."""
        exclude = exclude or set()
        try:
            entries = os.listdir(folder)
        except Exception:
            return None
        best, best_t = None, since
        for fn in entries:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            full = os.path.join(folder, fn)
            if full in exclude:
                continue
            try:
                mt = os.path.getmtime(full)
            except Exception:
                continue
            if mt > best_t:
                best, best_t = full, mt
        return best