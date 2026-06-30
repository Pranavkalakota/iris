"""
iris_faces.py — face detection + ArcFace embedding pipeline (M5 Stage 2).

Pure compute layer. Sits between the camera (which gives us a numpy frame)
and iris_people.PeopleStore (which holds the registry). No Qt, no camera
opening, no UI — Stages 4 and 5 will wire those in. This module just
exposes:

    pipeline = get_pipeline()
    pipeline.warm_up(blocking=False)                # lazy-loads DeepFace
    pipeline.extract_faces(image_bgr)               # detect, no embedding
    pipeline.embed(face_crop_bgr)                   # 512-dim ArcFace vec
    pipeline.process_frame(image_bgr, store)        # the full pipeline

Off-limits siblings — DO NOT modify from here:
  - any audio path (transcriber, diarizer, summarizer, speaker_db)
  - iris_gui.py, terminal.py, ESP32 firmware code
  - iris_photos.py, iris_sessions.py

The first call to warm_up() triggers TensorFlow import (slow, ~10-20s) and
downloads ArcFace weights (~130 MB, one-time, to ~/.deepface/weights/).
Same pattern as diarizer_phase9._load_encoder() — done once in a worker
thread so the GUI never blocks. Subsequent inferences are fast.

Behavioural decision (chosen by Pranav, locked in for M5):
  Enroll on first sight. Any face that doesn't match an existing row in
  the registry above MATCH_THRESHOLD gets a new "Unknown N" row created
  immediately, with its embedding stored. Stage 5 UI will let the user
  rename them. No two-clip cooldown, no manual confirmation.
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Defer the heavy imports (cv2, deepface, tensorflow) into _load() so this
# module is cheap to import. iris_gui.py can `import iris_faces` at startup
# without paying the TF init cost.

import iris_people  # cheap; no heavy deps


# ── tuneables ────────────────────────────────────────────────────────────
# Detection backend.
#
# We use 'opencv' (Haar cascade): fast, CPU-friendly, already installed,
# no extra weights to download. Good in normal conditions — face roughly
# forward-facing, decent lighting, face larger than ~60x60 pixels in the
# frame. Plenty for a wearable camera at conversational distance.
#
# For harder cases (extreme side angles, dim rooms, very small faces at
# the edge of frame, partial occlusion) DeepFace also bundles
# 'retinaface' which we already installed but are NOT using. Retinaface
# is more accurate but slower and uses more RAM. Switching to it is a
# single-line change — flip DETECTOR_BACKEND below — with zero schema
# changes, zero data migration, and zero API changes in any other file.
# The ArcFace recognition model is the same either way; only the
# face-finding step changes. Revisit this if we start missing faces in
# real use.
DETECTOR_BACKEND = "opencv"

# ArcFace produces 512-dim embeddings. Documented here for readers;
# the actual dimension comes from whatever DeepFace returns and is stored
# alongside the blob in the embeddings table.
RECOGNITION_MODEL = "ArcFace"
EXPECTED_EMBEDDING_DIM = 512

# Cosine-similarity threshold for "this is the same person we already
# know". Below this we treat the face as new and enroll on first sight.
# Pulled from the M5 build plan (the IRIS blueprint suggested 0.6 as a
# conservative DeepFace/ArcFace threshold).
MATCH_THRESHOLD = 0.60

# Reinforcement sweet-spot: when a face matches an existing person, we
# only write the new embedding back to the registry if its similarity
# falls in this range.
#   - Below MATCH_THRESHOLD: not a match at all, doesn't reach this code
#   - Between MATCH_THRESHOLD and REINFORCE_MAX: useful new sample,
#     genuinely different angle/lighting → store it
#   - Above REINFORCE_MAX: near-duplicate, no learning value, would just
#     fill the per-person cap with redundant samples → skip
REINFORCE_MIN = MATCH_THRESHOLD
REINFORCE_MAX = 0.95

# Ignore tiny face boxes — too noisy for ArcFace to embed reliably.
MIN_FACE_SIZE_PX = 60

# DeepFace's face-detection confidence. The opencv backend doesn't always
# populate this; leave at 0.0 so we don't accidentally drop valid faces.
# Retinaface populates it reliably — bump this to ~0.9 if we switch.
MIN_FACE_CONFIDENCE = 0.0


# ── dataclasses ──────────────────────────────────────────────────────────
@dataclass
class FaceCrop:
    """A detected face: a numpy BGR crop + where it came from in the
    original frame. The crop is what we feed to ArcFace; the bbox is
    what the UI will draw a rectangle around in Stage 5."""
    image: np.ndarray
    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0


@dataclass
class ProcessedFace:
    """Result of running detect→embed→match→write on one face from one
    frame. Stage 4 fusion uses these to tag transcript segments; Stage 5
    UI uses them to draw 'Jacob · 0.84' overlays on the live feed."""
    person_id: int
    name: str
    similarity: float                  # best cosine sim during match
    bbox: tuple[int, int, int, int]    # (x, y, w, h) in original frame
    was_new_enrollment: bool = False
    was_reinforced: bool = False
    detect_confidence: float = 0.0


# ── pipeline ─────────────────────────────────────────────────────────────
class FacePipeline:
    """Detect faces and compute ArcFace embeddings. Module-level
    singleton; the DeepFace + TensorFlow load is expensive so we keep
    one instance around for the life of the process.

    Thread-safe: a lock serialises calls into DeepFace. TF on CPU is not
    meaningfully re-entrant for our workload, and Stage 4 will be running
    detection on a worker thread anyway.
    """

    def __init__(self):
        self._lock                  = threading.RLock()
        self._loaded                = False
        self._loading               = False
        self._ready_event           = threading.Event()
        self._unknown_counter_lock  = threading.Lock()
        # Hold references so the GC can't dump the loaded module objects.
        self._df  = None    # the deepface module
        self._cv2 = None    # cv2 module

    # ── loading ──────────────────────────────────────────────────────────
    def warm_up(self, blocking: bool = False, timeout: float = 60.0) -> bool:
        """Kick off the heavy import + model load. If blocking, returns
        True once ready or False on timeout. If non-blocking, starts a
        worker thread and returns True immediately."""
        with self._lock:
            if self._loaded:
                return True
            if not self._loading:
                self._loading = True
                t = threading.Thread(target=self._load,
                                     name="FacePipelineLoad",
                                     daemon=True)
                t.start()
        if blocking:
            return self._ready_event.wait(timeout)
        return True

    def is_ready(self) -> bool:
        return self._loaded

    def wait_ready(self, timeout: float = 60.0) -> bool:
        return self._ready_event.wait(timeout)

    def _load(self) -> None:
        try:
            print("[faces] loading DeepFace + ArcFace "
                  "(first run downloads ~130 MB)...")
            import cv2

            # ── tf-keras compatibility patch ─────────────────────────────
            # Newer tf-keras removed `floatx` from the backend module.
            # DeepFace still calls it, so we shim it back in before import.
            try:
                import tf_keras.src.backend as _tfk_backend
                if not hasattr(_tfk_backend, "floatx"):
                    _tfk_backend.floatx = lambda: "float32"
                    print("[faces] patched tf_keras.src.backend.floatx")
            except Exception as _patch_err:
                print(f"[faces] tf_keras patch skipped: {_patch_err}")

            from deepface import DeepFace

            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            try:
                DeepFace.represent(
                    img_path=dummy,
                    model_name=RECOGNITION_MODEL,
                    detector_backend="skip",
                    enforce_detection=False,
                    align=False,
                )
            except Exception as e:
                print(f"[faces] ArcFace warm-up failed "
                      f"(will retry on first frame): {e}")
            self._cv2    = cv2
            self._df     = DeepFace
            self._loaded = True
            print("[faces] ready")
        except ImportError as e:
            print(f"[faces] DeepFace not installed: {e}")
            print("[faces] run: pip install deepface tf-keras")
        except Exception as e:
            print(f"[faces] load failed: {e}")
        finally:
            self._ready_event.set()
            self._loading = False

    # ── detection ────────────────────────────────────────────────────────
    def extract_faces(self, image_bgr: np.ndarray) -> list[FaceCrop]:
        """Find all faces in a BGR frame. Returns FaceCrops with crops
        ready to embed. Never raises — returns [] on any failure."""
        if not self._loaded or self._df is None:
            return []
        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            return []
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            return []
        try:
            with self._lock:
                faces = self._df.extract_faces(
                    img_path=image_bgr,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True,
                )
        except Exception as e:
            print(f"[faces] extract_faces failed: {e}")
            return []

        out: list[FaceCrop] = []
        h_img, w_img = image_bgr.shape[:2]
        for f in faces or []:
            try:
                area = f.get("facial_area") or {}
                x = int(area.get("x", 0))
                y = int(area.get("y", 0))
                w = int(area.get("w", 0))
                h = int(area.get("h", 0))
                conf = float(f.get("confidence",
                                   f.get("face_confidence", 0.0)) or 0.0)
            except Exception:
                continue
            if w < MIN_FACE_SIZE_PX or h < MIN_FACE_SIZE_PX:
                continue
            if conf < MIN_FACE_CONFIDENCE:
                continue
            # Clip bbox to the image and re-crop from the original BGR so
            # downstream code (thumbnails, debugging) has a real picture
            # to work with. DeepFace also returns an aligned float crop
            # in 'face' but we prefer the raw pixels.
            x0 = max(0, x); y0 = max(0, y)
            x1 = min(w_img, x + w); y1 = min(h_img, y + h)
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            try:
                crop = image_bgr[y0:y1, x0:x1].copy()
            except Exception:
                continue
            if crop.size == 0:
                continue
            out.append(FaceCrop(image=crop, x=x0, y=y0,
                                w=x1 - x0, h=y1 - y0,
                                confidence=conf))
        return out

    # ── embedding ────────────────────────────────────────────────────────
    def embed(self, face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Compute a 512-dim ArcFace embedding from a BGR face crop.
        Returns a unit-normalised float32 numpy array, or None on
        failure."""
        if not self._loaded or self._df is None:
            return None
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None
        try:
            with self._lock:
                results = self._df.represent(
                    img_path=face_crop_bgr,
                    model_name=RECOGNITION_MODEL,
                    detector_backend="skip",   # already cropped + aligned
                    enforce_detection=False,
                    align=False,
                )
        except Exception as e:
            print(f"[faces] embed failed: {e}")
            return None
        if not results:
            return None
        try:
            vec = np.asarray(results[0]["embedding"], dtype=np.float32)
        except Exception:
            return None
        if vec.size == 0 or not np.all(np.isfinite(vec)):
            return None
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return None
        return (vec / norm).astype(np.float32, copy=False)

    # ── headline: full pipeline for one frame ────────────────────────────
    def process_frame(self, image_bgr: np.ndarray,
                      store: iris_people.PeopleStore
                      ) -> list[ProcessedFace]:
        """Detect → embed → match → reinforce-or-enroll for every face
        in a frame. Stage 4 will call this once per keyframe."""
        if not self._loaded or store is None:
            return []
        crops = self.extract_faces(image_bgr)
        if not crops:
            return []
        out: list[ProcessedFace] = []
        for crop in crops:
            emb = self.embed(crop.image)
            if emb is None:
                continue
            result = store.match_face(emb, threshold=MATCH_THRESHOLD)
            if result is not None:
                # Known face → mark seen, optionally reinforce
                store.mark_seen(result.person.id)
                reinforced = False
                if REINFORCE_MIN <= result.similarity <= REINFORCE_MAX:
                    if store.add_embedding(result.person.id,
                                           iris_people.KIND_FACE, emb):
                        reinforced = True
                out.append(ProcessedFace(
                    person_id=result.person.id,
                    name=result.person.name,
                    similarity=result.similarity,
                    bbox=(crop.x, crop.y, crop.w, crop.h),
                    was_new_enrollment=False,
                    was_reinforced=reinforced,
                    detect_confidence=crop.confidence,
                ))
            else:
                # Unknown face → enroll on first sight (Pranav's call).
                name = self._next_unknown_name(store)
                person = store.add(name, face_embedding=emb)
                if person is None:
                    continue
                store.mark_seen(person.id)
                out.append(ProcessedFace(
                    person_id=person.id,
                    name=person.name,
                    similarity=0.0,
                    bbox=(crop.x, crop.y, crop.w, crop.h),
                    was_new_enrollment=True,
                    was_reinforced=False,
                    detect_confidence=crop.confidence,
                ))
        return out

    # ── helpers ──────────────────────────────────────────────────────────
    def _next_unknown_name(self,
                           store: iris_people.PeopleStore) -> str:
        """Generate the next 'Unknown N' name not already in the
        registry. Walks existing names — cheap because the registry is
        small. Lock prevents two concurrent enrollments from picking the
        same N."""
        with self._unknown_counter_lock:
            existing = {p.name for p in store.list_all()}
            n = 1
            while f"Unknown {n}" in existing:
                n += 1
            return f"Unknown {n}"


# ── module-level singleton ───────────────────────────────────────────────
_pipeline_singleton: Optional[FacePipeline] = None
_singleton_lock = threading.Lock()


def get_pipeline() -> FacePipeline:
    """Return the shared FacePipeline, creating it on first call. Cheap
    — construction does not trigger any heavy load. Call warm_up() on
    the returned object to start loading DeepFace in a worker thread."""
    global _pipeline_singleton
    with _singleton_lock:
        if _pipeline_singleton is None:
            _pipeline_singleton = FacePipeline()
        return _pipeline_singleton