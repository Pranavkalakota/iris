"""
iris_videos.py — ESP32 video-clip awareness for the IRIS chat.

Why it exists:
  The Stream tab already receives .avi clips from the ESP32 and saves them to
  disk, and iris_fusion already runs face recognition over their keyframes.
  But nothing ever *recorded* what was found, and the chat tab only ever
  indexed audio recordings (see RecordingStore in iris_gui.py, which filters
  on _AUDIO_EXTS). So when the user asked "how many people were in that
  video?", the assistant answered that it had no access to video recordings —
  because, as far as the chat context was concerned, they didn't exist.

  This module fixes that by owning the "video" half of what iris_photos.py /
  iris_sessions.py already do for photos and chat history:

    • it finds the saved clips on disk (the Stream-tab save folder, the
      per-person people_store/<id>/sessions folders, and the usual project
      sub-dirs),
    • it reads / writes a small JSON sidecar next to each clip
      (<clip>.video.json) holding the analysis — how many distinct people
      were seen, their recognised names, duration, frames sampled, and how
      the count was obtained, and
    • it can analyse a clip on demand: preferring the real iris_fusion /
      DeepFace pipeline (which also yields names and feeds the People
      registry, exactly like the live path), and falling back to a
      dependency-light OpenCV face/upper-body detector when fusion isn't
      loaded — so existing clips captured before this feature can still be
      answered about.

Pure-ish Python: OpenCV (cv2) is imported lazily and every path degrades
gracefully, so importing this module never breaks the app even on a machine
without cv2 or without the fusion backend.
"""

from __future__ import annotations

import os
import re
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

# Capture timestamps embedded in clip filenames, e.g.
# "2026-06-29_13-54-49.avi" or "clip_20260629_135449.avi".
_TS_PATTERNS = (
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[_-](\d{2})-(\d{2})-(\d{2})"), None),
    (re.compile(r"(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})"), None),
)


def _timestamp_from_name(path: str) -> Optional[float]:
    """Epoch seconds parsed from the clip's filename, or None."""
    base = os.path.basename(path)
    for pat, _ in _TS_PATTERNS:
        m = pat.search(base)
        if m:
            try:
                y, mo, d, h, mi, s = (int(x) for x in m.groups())
                return datetime(y, mo, d, h, mi, s).timestamp()
            except Exception:
                continue
    return None

# Extensions we treat as a saved video clip.
_VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".m4v", ".webm")

# Sidecar suffix. Kept distinct from the plain ".json" that audio recordings
# use so the two never collide for a clip that also has an audio sibling.
_SIDECAR_SUFFIX = ".video.json"


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VideoClip:
    """One saved ESP32 video clip plus whatever analysis we have for it."""
    path: str
    received_at: float
    source: str = "esp32"
    duration_sec: Optional[float] = None
    frames_sampled: int = 0
    people_count: Optional[int] = None          # None → never analysed yet
    people_names: list = field(default_factory=list)
    method: str = ""                            # "fusion" | "opencv" | ""
    note: str = ""
    scene_description: str = ""                 # "" → never described yet
    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def analyzed(self) -> bool:
        return self.people_count is not None

    @property
    def described(self) -> bool:
        return bool(self.scene_description)

    def when(self) -> str:
        try:
            return datetime.fromtimestamp(self.received_at).strftime(
                "%b %d %H:%M:%S")
        except Exception:
            return "—"

    def length(self) -> str:
        if not self.duration_sec:
            return "—"
        s = int(self.duration_sec)
        return f"{s // 60}:{s % 60:02d}"

    def people_summary(self) -> str:
        """A short human phrase describing who/how many were seen."""
        if self.people_count is None:
            return "not analysed yet"
        if self.people_count == 0:
            return "no people detected"
        named = [n for n in self.people_names if n and "unknown" not in n.lower()]
        who = f" ({', '.join(named)})" if named else ""
        noun = "person" if self.people_count == 1 else "people"
        return f"{self.people_count} {noun}{who}"


# ─────────────────────────────────────────────────────────────────────────────
class VideoStore:
    """Owns discovery + sidecars for saved video clips. Never raises out."""

    def __init__(self, folders: Optional[Sequence[str]] = None,
                 fusion_getter=None):
        # fusion_getter: a zero-arg callable returning an iris_fusion Fusion
        # (usually iris_fusion.get_fusion). Optional — analysis falls back to
        # OpenCV when it's absent or the pipeline isn't ready.
        self._folders = list(folders) if folders else None
        self._fusion_getter = fusion_getter

    # ── folder discovery ────────────────────────────────────────────────
    def add_folder(self, path) -> None:
        """Register an authoritative clip folder (e.g. the Stream tab's real
        SAVE_FOLDER). Merged with — not instead of — the auto-discovered
        defaults, so nothing that already worked stops working."""
        if not path:
            return
        if self._folders is None:
            self._folders = []
        p = str(path)
        if p not in self._folders:
            self._folders.append(p)

    def folders(self) -> list[str]:
        # Explicitly-registered folders take priority, but we always also
        # scan the auto-discovered defaults so a clip is found no matter
        # which folder the receiver saved it to.
        out, seen = [], set()
        for d in list(self._folders or []) + default_video_dirs():
            if not d:
                continue
            try:
                rp = os.path.realpath(os.path.abspath(d))
            except Exception:
                continue
            key = os.path.normcase(rp)
            if key in seen or not os.path.isdir(rp):
                continue
            seen.add(key)
            out.append(rp)
        return out

    # ── listing ─────────────────────────────────────────────────────────
    def list_all(self, limit: Optional[int] = None) -> list[VideoClip]:
        """Every known clip, newest first, with sidecar analysis if present."""
        out: list[VideoClip] = []
        seen: set[str] = set()
        for folder in self.folders():
            for root, dirs, files in os.walk(folder):
                dirs[:] = [d for d in dirs
                           if d.lower() not in {"__pycache__", ".git",
                                                "node_modules", "chroma"}]
                for fn in files:
                    if not fn.lower().endswith(_VIDEO_EXTS):
                        continue
                    full = os.path.abspath(os.path.join(root, fn))
                    key = os.path.normcase(os.path.realpath(full))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(self._load_clip(full))
        out.sort(key=lambda c: c.received_at, reverse=True)
        return out[:limit] if limit else out

    def latest(self) -> Optional[VideoClip]:
        clips = self.list_all(limit=1)
        return clips[0] if clips else None

    def _load_clip(self, path: str) -> VideoClip:
        meta = read_sidecar(path)
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0.0
        # Prefer sidecar → filename timestamp → file mtime.
        received_at = meta.get("received_at")
        if received_at is None:
            received_at = _timestamp_from_name(path) or mtime
        received_at = float(received_at)
        return VideoClip(
            path=path,
            received_at=received_at,
            source=meta.get("source", "esp32"),
            duration_sec=meta.get("duration_sec"),
            frames_sampled=int(meta.get("frames_sampled", 0) or 0),
            people_count=meta.get("people_count"),
            people_names=list(meta.get("people_names", []) or []),
            method=meta.get("method", ""),
            note=meta.get("note", ""),
            scene_description=meta.get("scene_description", ""),
        )

    # ── analysis ────────────────────────────────────────────────────────
    def analyze(self, path: str, force: bool = False) -> VideoClip:
        """Return analysis for one clip, computing + caching it if needed.

        Uses the real fusion/DeepFace pipeline when available (accurate,
        yields names, updates the People registry just like the live path);
        otherwise a light OpenCV face/upper-body pass. Result is written to
        the <clip>.video.json sidecar so it's instant next time.
        """
        if not force:
            existing = self._load_clip(path)
            if existing.analyzed:
                return existing

        fusion = None
        if self._fusion_getter is not None:
            try:
                fusion = self._fusion_getter()
            except Exception:
                fusion = None

        result = analyze_clip(path, fusion=fusion)
        record_analysis(
            path,
            people_count=result.people_count,
            people_names=result.people_names,
            duration_sec=result.duration_sec,
            frames_sampled=result.frames_sampled,
            method=result.method,
            source=result.source,
            note=result.note,
        )
        return result

    def describe(self, path: str, force: bool = False,
                 frame_count: int = 4) -> str:
        """On-demand scene description (who's visible, clothing, objects,
        setting) for one clip. Cached in the sidecar — only ever costs an
        LLaVA call once per clip unless force=True. Deliberately separate
        from analyze(): this is heavier (a real vision-language model call,
        several frames) and only runs when someone actually asks about the
        clip's content, never automatically on arrival.

        Returns '' if fusion/LLaVA isn't available or the clip can't be
        opened — callers should treat that as 'description unavailable',
        not 'nothing was in the clip'.
        """
        if not force:
            existing = self._load_clip(path)
            if existing.described:
                return existing.scene_description

        fusion = None
        if self._fusion_getter is not None:
            try:
                fusion = self._fusion_getter()
            except Exception:
                fusion = None
        if fusion is None or not hasattr(fusion, "describe_scene"):
            print(f"[video] describe(): fusion/LLaVA not available for "
                  f"{os.path.basename(path)}")
            return ""

        frames = _sample_frames_spread(path, frame_count)
        if not frames:
            print(f"[video] describe(): no readable frames sampled from "
                  f"{os.path.basename(path)}")
            return ""

        try:
            description = fusion.describe_scene(frames)
        except Exception as e:
            print(f"[video] describe_scene call failed: {e}")
            return ""
        if not description:
            print(f"[video] describe(): LLaVA returned an empty "
                  f"description for {os.path.basename(path)}")
        else:
            record_scene_description(path, description)
        return description

    # ── context for the chat LLM ────────────────────────────────────────
    def describe_recent(self, limit: int = 5, analyze_missing: bool = True,
                        analyze_budget: int = 2) -> str:
        """A context block listing recent clips + their people counts, ready
        to drop into the chat's system context so questions like 'how many
        people were in the video' can be answered from real data.

        `analyze_budget` caps how many not-yet-analysed clips we compute on
        the spot — face recognition on CPU is slow, so we only analyse the
        newest few here (the ones users actually ask about) and leave the
        rest to be analysed when a fresh clip arrives. Results are cached, so
        the cost is paid once per clip."""
        clips = self.list_all(limit=limit)
        if not clips:
            return ""
        lines = ["Saved ESP32 video clips the user may be asking about "
                 "(newest first):\n"]
        spent = 0
        for c in clips:
            if not c.analyzed and analyze_missing and spent < analyze_budget:
                try:
                    c = self.analyze(c.path)
                    spent += 1
                except Exception:
                    pass
            lines.append(
                f"• {c.name} | recorded {c.when()} | length {c.length()} | "
                f"{c.people_summary()}\n")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Sidecar read/write — module-level so the Stream tab can persist analysis
# without holding a VideoStore instance.
# ─────────────────────────────────────────────────────────────────────────────
def sidecar_path(clip_path: str) -> str:
    return os.path.splitext(clip_path)[0] + _SIDECAR_SUFFIX


def read_sidecar(clip_path: str) -> dict:
    try:
        with open(sidecar_path(clip_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def record_analysis(clip_path: str, *, people_count: Optional[int],
                    people_names: Optional[Sequence[str]] = None,
                    duration_sec: Optional[float] = None,
                    frames_sampled: int = 0, method: str = "",
                    source: str = "esp32", note: str = "") -> dict:
    """Write (or refresh) the analysis sidecar next to a clip. Returns the
    dict written. Never raises."""
    meta = read_sidecar(clip_path)
    if "received_at" not in meta:
        ts = _timestamp_from_name(clip_path)
        if ts is None:
            try:
                ts = os.path.getmtime(clip_path)
            except Exception:
                ts = time.time()
        meta["received_at"] = ts
    meta.update({
        "source": source,
        "people_count": people_count,
        "people_names": list(people_names or []),
        "duration_sec": duration_sec,
        "frames_sampled": frames_sampled,
        "method": method,
        "note": note,
        "analyzed_at": time.time(),
    })
    try:
        with open(sidecar_path(clip_path), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    return meta


def record_scene_description(clip_path: str, description: str) -> dict:
    """Merge a scene description into the clip's sidecar without touching
    whatever people/analysis fields are already there (or aren't yet —
    this can run before or independently of record_analysis()). Never
    raises."""
    meta = read_sidecar(clip_path)
    if "received_at" not in meta:
        ts = _timestamp_from_name(clip_path)
        if ts is None:
            try:
                ts = os.path.getmtime(clip_path)
            except Exception:
                ts = time.time()
        meta["received_at"] = ts
    meta["scene_description"] = description
    meta["scene_described_at"] = time.time()
    try:
        with open(sidecar_path(clip_path), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    return meta


def analyze_clip(path: str, fusion=None) -> VideoClip:
    """Count the distinct people in a clip. Prefers the fusion/DeepFace
    pipeline (names + registry side-effects); falls back to OpenCV. Always
    returns a VideoClip (people_count may be 0, never None on success)."""
    received_at = _timestamp_from_name(path)
    if received_at is None:
        try:
            received_at = os.path.getmtime(path)
        except Exception:
            received_at = time.time()

    # Backend 1: real face-recognition pipeline (accurate, gives names).
    if fusion is not None:
        try:
            if fusion.faces.is_ready():
                return _analyze_with_fusion(path, fusion, received_at)
        except Exception:
            pass

    # Backend 2: OpenCV fallback — no identity, just a head/person count.
    return _analyze_with_opencv(path, received_at)


def _open_capture(path):
    import cv2  # lazy — importing this module must not require cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_step = max(1, int(fps))              # ~1 keyframe / second
    return cap, fps, frame_step

def _sample_frames_spread(path: str, count: int = 4) -> list:
    """Grab `count` frames evenly spread across the clip (e.g. at ~12%,
    37%, 62%, 87% of total length) rather than bunched at the start.

    Reads every frame sequentially and picks evenly from what's ACTUALLY
    decodable, rather than trusting CAP_PROP_FRAME_COUNT — on corrupted or
    truncated ESP32 clips that metadata can badly overstate the real frame
    count (seen in practice: reported 1062 frames, only 44 actually
    decodable), which silently produced zero sampled frames before. Caps
    the read at a safety limit so a very long/corrupt file can't hang.
    """
    try:
        import cv2
    except Exception:
        return []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return []

    all_frames = []
    SAFETY_CAP = 2000  # ~65s at 30fps — comfortably above the ~35s clips
    while len(all_frames) < SAFETY_CAP:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        return []

    n = len(all_frames)
    count = max(1, min(count, n))
    targets = [int(n * (i + 1) / (count + 1)) for i in range(count)]
    return [all_frames[min(t, n - 1)] for t in targets]
def _analyze_with_fusion(path: str, fusion, received_at: float) -> VideoClip:
    import cv2
    cap, fps, step = _open_capture(path)
    if cap is None:
        return VideoClip(path=path, received_at=received_at,
                         people_count=0, method="fusion",
                         note="could not open clip")
    person_ids: set = set()
    names: dict = {}
    idx = sampled = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            sampled += 1
            try:
                for pf in fusion.process_frame(frame):   # list[ProcessedFace]
                    person_ids.add(pf.person_id)
                    if pf.name:
                        names[pf.person_id] = pf.name
            except Exception:
                pass
        idx += 1
    total = idx
    cap.release()
    duration = (total / fps) if fps else None
    return VideoClip(
        path=path, received_at=received_at, duration_sec=duration,
        frames_sampled=sampled, people_count=len(person_ids),
        people_names=[names[i] for i in person_ids if i in names],
        method="fusion",
        note=f"{sampled} keyframes over {total} frames via face recognition",
    )


def _analyze_with_opencv(path: str, received_at: float) -> VideoClip:
    """Dependency-light head count: the most faces/upper-bodies seen in any
    single sampled frame is a decent estimate of how many people were present
    at once. No identity — used when the fusion backend isn't available."""
    try:
        import cv2
    except Exception:
        return VideoClip(path=path, received_at=received_at, people_count=None,
                         method="", note="cv2 unavailable — cannot analyse")
    cap, fps, step = _open_capture(path)
    if cap is None:
        return VideoClip(path=path, received_at=received_at, people_count=0,
                         method="opencv", note="could not open clip")

    def _cascade(name):
        try:
            c = cv2.CascadeClassifier(
                os.path.join(cv2.data.haarcascades, name))
            return c if not c.empty() else None
        except Exception:
            return None

    face = _cascade("haarcascade_frontalface_default.xml")
    profile = _cascade("haarcascade_profileface.xml")
    upper = _cascade("haarcascade_upperbody.xml")

    max_people = 0
    idx = sampled = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            sampled += 1
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                counts = [0]
                for casc, mn in ((face, 5), (profile, 5), (upper, 3)):
                    if casc is not None:
                        found = casc.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=mn,
                            minSize=(30, 30))
                        counts.append(len(found))
                max_people = max(max_people, max(counts))
            except Exception:
                pass
        idx += 1
    total = idx
    cap.release()
    duration = (total / fps) if fps else None
    note = (f"{sampled} keyframes over {total} frames via OpenCV "
            f"face/body detection (no identity)")
    if max_people == 0:
        note += "; no clearly-visible faces — count may undercount "\
                "steep-angle footage"
    return VideoClip(
        path=path, received_at=received_at, duration_sec=duration,
        frames_sampled=sampled, people_count=max_people,
        people_names=[], method="opencv", note=note)


# ─────────────────────────────────────────────────────────────────────────────
def default_video_dirs() -> list[str]:
    """Best-effort list of folders where ESP32 clips get saved. Mirrors the
    Stream tab's SAVE_FOLDER, the per-person people_store sessions, and the
    usual project sub-dirs. Only existing directories are returned."""
    raw: list[str] = []

    # 1) Whatever the Stream-tab receiver (terminal.py) saves to. terminal.py
    #    lives next to iris_gui.py on the target machine (not in the repo), so
    #    make sure that directory is importable before trying.
    try:
        import sys as _sys
        try:
            _here = os.path.dirname(os.path.abspath(__file__))
            if _here and _here not in _sys.path:
                _sys.path.insert(0, _here)
        except Exception:
            pass
        import terminal as _t              # type: ignore
        for attr in ("SAVE_FOLDER", "VIDEO_FOLDER", "CLIP_FOLDER"):
            v = getattr(_t, attr, None)
            if isinstance(v, str) and v.strip():
                raw.append(v)
    except Exception:
        pass

    # 2) Config overrides, if the Phase 9 config exposes any.
    try:
        import config_phase9 as _cfg       # type: ignore
        for attr in ("VIDEO_DIR", "CLIPS_DIR", "SAVE_DIR", "ESP32_VIDEO_DIR"):
            v = getattr(_cfg, attr, None)
            if isinstance(v, str) and v.strip():
                raw.append(v)
    except Exception:
        pass

    # 3) Common on-disk locations relative to the app + home.
    roots = [os.getcwd()]
    try:
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass
    home = os.path.expanduser("~")
    raw += [
        os.path.join(home, "Desktop", "ESP32_Recording"),
        os.path.join(home, "Desktop", "esp32_recording"),
    ]
    for r in roots:
        raw += [
            r,
            os.path.join(r, "recordings"),
            os.path.join(r, "clips"),
            os.path.join(r, "videos"),
            os.path.join(r, "data", "sqlite", "people_store"),
        ]

    out, seen = [], set()
    for d in raw:
        try:
            rp = os.path.realpath(os.path.abspath(d))
        except Exception:
            continue
        key = os.path.normcase(rp)
        if key in seen or not os.path.isdir(rp):
            continue
        seen.add(key)
        out.append(rp)
    return out