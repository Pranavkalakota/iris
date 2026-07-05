"""
event_boundary_phase9.py — M6 (§6.3) Event Boundary Detector for IRIS.

Blueprint role (this is the blueprint's `boundary_detector.py`):
  The EventBoundaryDetector class determines whether each new clip is part of
  the current event or a new one. It requires 2 consecutive confirming clips
  before firing — prevents false positives from a single odd clip.

§6.3 signal model (weighted similarity vs. the previous clip — higher = more
"same event"):

    Signal                    Weight   How computed                         New-event threshold
    ─────────────────────────────────────────────────────────────────────────────────────────
    Visual scene embedding    0.40     LLaVA/MiniLM embedding cosine sim     < 0.50
    Ambient audio fingerprint 0.30     librosa MFCC cosine similarity        < 0.45
    Face presence overlap     0.20     Jaccard of face-ID sets               = 0 (no overlap)
    Motion pattern            0.10     optical-flow mean-magnitude delta     > 3× baseline
    Time gap (override)       —        seconds between consecutive clips     > 300s = force boundary

Combined score → decision band:
    > 0.75          → SAME event, append to current
    0.50 – 0.75     → AMBIGUOUS, hold as "in transit", wait for next clip
    < 0.50          → boundary candidate; needs a 2nd consecutive confirming
                      clip → then close the current event and open a new one

Any signal may be missing (a clip with no readable scene text, no audio, etc.).
Missing signals are simply dropped and the remaining weights are renormalised,
so the detector degrades gracefully rather than fabricating boundaries.

Design mirrors iris_fusion / iris_videos:
  • module-level singleton via get_detector() (like iris_fusion.get_fusion),
  • per-clip JSON sidecars next to each clip (<clip>.event.json),
  • a persisted state file + append-only events log so a restart keeps the
    current event instead of splitting it, and
  • never raises out — every disk/parse error degrades to in-memory behaviour
    and prints a "[event] ..." diagnostic, matching the rest of the codebase.
"""

from __future__ import annotations

import os
import json
import math
import time
import uuid
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Sequence, Callable, Any


# ── config (read from config_phase9 with safe fallbacks) ──────────────────
def _cfg(name: str, default):
    try:
        import config_phase9 as _c            # type: ignore
        v = getattr(_c, name, None)
        return v if v is not None else default
    except Exception:
        return default


def _events_dir() -> str:
    """Where the state file and events log live. Co-located with the other
    IRIS data so it's backed up / synced with everything else."""
    override = _cfg("EVENTS_DIR", None)
    if isinstance(override, str) and override.strip():
        base = override
    else:
        base = os.path.join(os.getcwd(), "data", "events")
    try:
        os.makedirs(base, exist_ok=True)
        return base
    except Exception:
        alt = os.path.join(os.getcwd(), "events")
        try:
            os.makedirs(alt, exist_ok=True)
        except Exception:
            pass
        return alt


_SIDECAR_SUFFIX = ".event.json"


def sidecar_path(clip_path: str) -> str:
    return os.path.splitext(clip_path)[0] + _SIDECAR_SUFFIX


# ── §6.3 signal weights ────────────────────────────────────────────────────
_W_VISUAL = float(_cfg("EVENT_W_VISUAL", 0.40))
_W_AUDIO  = float(_cfg("EVENT_W_AUDIO",  0.30))
_W_FACE   = float(_cfg("EVENT_W_FACE",   0.20))
_W_MOTION = float(_cfg("EVENT_W_MOTION", 0.10))


# ── inputs / result / state types ─────────────────────────────────────────
@dataclass
class ClipSignals:
    """Per-clip evidence handed to observe(). Every field is optional — the
    detector renormalises over whatever is present. iris_gui fills in what it
    cheaply has (faces, location, timestamp) and, when available, the LLaVA
    scene embedding / MFCC / motion magnitude."""
    received_at: Optional[float] = None
    visual_embedding: Optional[Sequence[float]] = None   # LLaVA/MiniLM vector
    scene_description: str = ""                           # embedded if no vec
    audio_mfcc: Optional[Sequence[float]] = None          # librosa MFCC mean
    face_ids: Optional[Sequence[Any]] = None              # ids OR names
    people_names: Optional[Sequence[str]] = None          # for logs/sidecar
    motion_magnitude: Optional[float] = None              # optical-flow mean
    location: Optional[dict] = None


@dataclass
class EventDecision:
    """Returned by EventBoundaryDetector.observe() for one clip."""
    event_id: str
    status: str = "same"              # same | in_transit | boundary_pending | new_event
    is_new_event: bool = False        # True only on the clip that fires
    combined_score: Optional[float] = None   # None when this is the 1st clip
    signal_scores: dict = field(default_factory=dict)     # per-signal sims
    boundary_pending: int = 0         # consecutive confirming clips so far
    confirm_needed: int = 2           # EVENT_MIN_CONFIRM_CLIPS
    reason: str = ""                  # human-readable why
    clip_index_in_event: int = 0      # 0-based position within its event

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _EventState:
    """The event currently in progress plus the debounce bookkeeping."""
    event_id: str = ""
    started_at: float = 0.0
    last_clip_at: float = 0.0
    n_clips: int = 0
    people: list = field(default_factory=list)     # accumulated names, deduped
    location: Optional[dict] = None                 # last known location dict
    # previous-clip signal signature (for cosine/jaccard/motion comparison)
    prev_visual: Optional[list] = None
    prev_mfcc: Optional[list] = None
    prev_faces: Optional[list] = None
    motion_baseline: Optional[float] = None
    # debounce bookkeeping (§6.3): consecutive <0.50 confirming clips
    pending_count: int = 0
    pending_since: float = 0.0
    pending_reason: str = ""
    pending_people: list = field(default_factory=list)
    pending_location: Optional[dict] = None
    pending_visual: Optional[list] = None
    pending_mfcc: Optional[list] = None
    pending_faces: Optional[list] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "_EventState":
        s = _EventState()
        if not isinstance(d, dict):
            return s
        s.event_id        = str(d.get("event_id") or "")
        s.started_at      = float(d.get("started_at") or 0.0)
        s.last_clip_at    = float(d.get("last_clip_at") or 0.0)
        s.n_clips         = int(d.get("n_clips") or 0)
        s.people          = list(d.get("people") or [])
        s.location        = d.get("location")
        s.prev_visual     = d.get("prev_visual")
        s.prev_mfcc       = d.get("prev_mfcc")
        s.prev_faces      = d.get("prev_faces")
        s.motion_baseline = d.get("motion_baseline")
        s.pending_count   = int(d.get("pending_count") or 0)
        s.pending_since   = float(d.get("pending_since") or 0.0)
        s.pending_reason  = str(d.get("pending_reason") or "")
        s.pending_people  = list(d.get("pending_people") or [])
        s.pending_location = d.get("pending_location")
        s.pending_visual  = d.get("pending_visual")
        s.pending_mfcc    = d.get("pending_mfcc")
        s.pending_faces   = d.get("pending_faces")
        return s


# ── similarity helpers ─────────────────────────────────────────────────────
def _cosine01(a: Optional[Sequence[float]],
              b: Optional[Sequence[float]]) -> Optional[float]:
    """Cosine similarity clamped to [0, 1]. None if either vector missing."""
    if not a or not b:
        return None
    try:
        n = min(len(a), len(b))
        if n == 0:
            return None
        dot = na = nb = 0.0
        for i in range(n):
            x = float(a[i]); y = float(b[i])
            dot += x * y; na += x * x; nb += y * y
        if na <= 0.0 or nb <= 0.0:
            return None
        c = dot / math.sqrt(na * nb)
        return max(0.0, min(1.0, (c + 1.0) / 2.0)) if c < 0 else max(0.0, min(1.0, c))
    except Exception:
        return None


def _jaccard(a: Optional[Sequence[Any]], b: Optional[Sequence[Any]]) -> Optional[float]:
    """Jaccard similarity of two ID/name sets. None if BOTH are missing;
    if exactly one side is empty the overlap is genuinely 0."""
    if a is None and b is None:
        return None
    sa = {str(x).strip().lower() for x in (a or []) if str(x).strip()}
    sb = {str(x).strip().lower() for x in (b or []) if str(x).strip()}
    if not sa and not sb:
        return None                      # no face evidence either side
    union = len(sa | sb)
    return (len(sa & sb) / union) if union else None


def _motion_sim(cur: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """Similarity from optical-flow mean magnitude vs a running baseline.
    Blueprint: delta > 3× baseline → new-event signal. We map the delta onto
    a [0,1] similarity: 0 when delta ≥ 3×baseline, 1 when identical."""
    if cur is None or baseline is None or baseline <= 0.0:
        return None
    try:
        delta = abs(float(cur) - float(baseline))
        return max(0.0, min(1.0, 1.0 - (delta / (3.0 * baseline))))
    except Exception:
        return None


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    try:
        r = 6371000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        h = (math.sin(dphi / 2) ** 2
             + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
        return 2 * r * math.asin(min(1.0, math.sqrt(h)))
    except Exception:
        return 0.0


def _norm_names(names: Optional[Sequence[str]]) -> list:
    out, seen = [], set()
    for n in (names or []):
        s = str(n or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _as_list(v):
    if v is None:
        return None
    try:
        return [float(x) for x in v]
    except Exception:
        try:
            return list(v)
        except Exception:
            return None


# Lazy MiniLM embedder (reuses iris_memory's model) so a scene *description*
# with no precomputed vector can still contribute a visual-similarity signal.
_EMBED = None
_EMBED_TRIED = False


def _embed_text(text: str) -> Optional[list]:
    global _EMBED, _EMBED_TRIED
    if not text or not text.strip():
        return None
    if _EMBED is None and not _EMBED_TRIED:
        _EMBED_TRIED = True
        try:
            import iris_memory                     # type: ignore
            mem = iris_memory.get_memory()
            # Prefer a public embed(), fall back to the raw model if present.
            if hasattr(mem, "embed"):
                _EMBED = mem.embed
            elif getattr(mem, "_model", None) is not None:
                _EMBED = lambda t: mem._model.encode(t)   # noqa: E731
        except Exception as e:
            print(f"[event] text embedder unavailable: {e}")
            _EMBED = None
    if _EMBED is None:
        return None
    try:
        vec = _EMBED(text)
        return _as_list(vec)
    except Exception:
        return None


# ── the detector ───────────────────────────────────────────────────────────
class EventBoundaryDetector:
    """Decides, per clip, whether a new event has started. Thread-safe; a
    single instance is shared across the clip-worker threads iris_gui spawns
    (see get_detector())."""

    def __init__(self, state_dir: Optional[str] = None):
        self.dir = state_dir or _events_dir()
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception as e:
            print(f"[event] could not create state dir {self.dir}: {e}")
        self._state_path = os.path.join(self.dir, "events_state.json")
        self._log_path = os.path.join(self.dir, "events.json")
        self._lock = threading.RLock()

        # tunables (§6.3)
        self.confirm_needed = int(_cfg("EVENT_MIN_CONFIRM_CLIPS", 2))
        self.max_gap_s      = float(_cfg("EVENT_MAX_GAP_SECONDS", 300.0))
        self.same_thresh    = float(_cfg("EVENT_SAME_THRESHOLD", 0.75))
        self.new_thresh     = float(_cfg("EVENT_NEW_THRESHOLD", 0.50))

        # Optional callback fired when an event closes, so iris_gui can ask an
        # LLM for the 3–5 sentence event summary (§6.4). Signature:
        #   on_event_closed(event_record: dict) -> None
        self.on_event_closed: Optional[Callable[[dict], None]] = None

        self._state = self._load_state()

    # ── public API ─────────────────────────────────────────────────────
    def observe(self, clip_path: str,
                signals: Optional[ClipSignals] = None,
                **kwargs) -> EventDecision:
        """Feed one clip to the detector and get back an EventDecision.

        Either pass a ClipSignals, or the common fields as keywords
        (received_at, people_names, face_ids, location, scene_description,
        visual_embedding, audio_mfcc, motion_magnitude). Writes a
        <clip>.event.json sidecar and, on a confirmed boundary, appends the
        closed event to the events log. Never raises."""
        if signals is None:
            signals = ClipSignals(
                received_at=kwargs.get("received_at"),
                visual_embedding=kwargs.get("visual_embedding"),
                scene_description=kwargs.get("scene_description", "") or "",
                audio_mfcc=kwargs.get("audio_mfcc"),
                face_ids=kwargs.get("face_ids", kwargs.get("people_names")),
                people_names=kwargs.get("people_names"),
                motion_magnitude=kwargs.get("motion_magnitude"),
                location=kwargs.get("location"),
            )
        try:
            return self._observe_locked(clip_path, signals)
        except Exception as e:
            print(f"[event] observe failed for "
                  f"{os.path.basename(clip_path or '?')}: {e}")
            eid = self._state.event_id or _new_event_id()
            return EventDecision(event_id=eid,
                                 confirm_needed=self.confirm_needed,
                                 reason="error — treated as same event")

    def current_event_id(self) -> str:
        with self._lock:
            return self._state.event_id

    # ── core logic ─────────────────────────────────────────────────────
    def _observe_locked(self, clip_path: str,
                        sig: ClipSignals) -> EventDecision:
        now = float(sig.received_at) if sig.received_at else time.time()
        people = _norm_names(sig.people_names)
        faces = list(sig.face_ids) if sig.face_ids is not None else \
            (people or None)

        # Resolve the visual vector: explicit embedding wins, else embed the
        # scene description text through MiniLM.
        visual = _as_list(sig.visual_embedding)
        if visual is None and sig.scene_description:
            visual = _embed_text(sig.scene_description)
        mfcc = _as_list(sig.audio_mfcc)

        with self._lock:
            st = self._state

            # First clip ever (or after a wiped state) → seed event 1.
            if not st.event_id:
                self._start_event(st, now, people, sig.location,
                                  visual, mfcc, faces, sig.motion_magnitude)
                dec = EventDecision(
                    event_id=st.event_id, status="new_event",
                    is_new_event=True, combined_score=None,
                    confirm_needed=self.confirm_needed,
                    reason="first event", clip_index_in_event=0)
                self._save_state()
                self._write_sidecar(clip_path, dec, now, people, sig)
                print(f"[event] first event {st.event_id} started")
                return dec

            # ── compute per-signal similarities vs the previous clip ──────
            scores: dict = {}
            v = _cosine01(st.prev_visual, visual)
            if v is not None:
                scores["visual"] = round(v, 3)
            a = _cosine01(st.prev_mfcc, mfcc)
            if a is not None:
                scores["audio"] = round(a, 3)
            f = _jaccard(st.prev_faces, faces)
            if f is not None:
                scores["face"] = round(f, 3)
            m = _motion_sim(sig.motion_magnitude, st.motion_baseline)
            if m is not None:
                scores["motion"] = round(m, 3)

            # weighted, renormalised over whatever signals are present
            combined = self._combine(scores)

            # time-gap override (§6.3): >300s forces a boundary
            gap = (now - st.last_clip_at) if st.last_clip_at else 0.0
            forced = gap > self.max_gap_s
            if forced:
                combined = 0.0

            # ── decision bands ────────────────────────────────────────────
            if forced or (combined is not None and combined < self.new_thresh):
                reason = (f"{gap/60:.0f} min gap (override)" if forced
                          else f"combined {combined:.2f} < {self.new_thresh:.2f}")
                dec = self._register_boundary_candidate(
                    st, now, people, faces, visual, mfcc, sig, combined,
                    scores, reason)
            elif combined is not None and combined <= self.same_thresh:
                # ambiguous band → hold as "in transit", wait for next clip.
                # Do NOT count as a confirming boundary clip, do NOT reset.
                st.last_clip_at = now
                dec = EventDecision(
                    event_id=st.event_id, status="in_transit",
                    is_new_event=False, combined_score=combined,
                    signal_scores=scores,
                    boundary_pending=st.pending_count,
                    confirm_needed=self.confirm_needed,
                    reason=(f"in transit (combined {combined:.2f} in "
                            f"{self.new_thresh:.2f}–{self.same_thresh:.2f})"),
                    clip_index_in_event=st.n_clips - 1)
                # NB: do NOT update the event's reference signature here — an
                # ambiguous clip must keep being compared against the current
                # event so the "wait for the next clip" hold works.
            else:
                # combined > 0.75 (or no comparable signals) → same event.
                if st.pending_count:
                    print(f"[event] boundary streak reset after "
                          f"{st.pending_count} clip(s) — continuation of "
                          f"event {st.event_id}")
                self._clear_pending(st)
                self._extend_event(st, now, people, sig.location,
                                   visual, mfcc, faces, sig.motion_magnitude)
                cs = combined if combined is not None else None
                dec = EventDecision(
                    event_id=st.event_id, status="same",
                    is_new_event=False, combined_score=cs,
                    signal_scores=scores, boundary_pending=0,
                    confirm_needed=self.confirm_needed,
                    reason=("same event" if cs is None
                            else f"same event (combined {cs:.2f})"),
                    clip_index_in_event=st.n_clips - 1)

            self._save_state()
            self._write_sidecar(clip_path, dec, now, people, sig)
            return dec

    def _register_boundary_candidate(self, st, now, people, faces, visual,
                                     mfcc, sig, combined, scores, reason
                                     ) -> EventDecision:
        """Handle a clip whose score says 'boundary'. Advances the debounce
        streak; fires a new event only once confirm_needed consecutive
        confirming clips have been seen (§6.3: '2 consecutive clips')."""
        if st.pending_count <= 0:
            st.pending_since = now
            st.pending_reason = reason
            st.pending_people = list(people)
            st.pending_location = sig.location or st.pending_location
            st.pending_visual = visual
            st.pending_mfcc = mfcc
            st.pending_faces = faces
        else:
            st.pending_people = _norm_names(list(st.pending_people) + people)
            if sig.location:
                st.pending_location = sig.location
            if visual is not None:
                st.pending_visual = visual
            if mfcc is not None:
                st.pending_mfcc = mfcc
            if faces is not None:
                st.pending_faces = faces
        st.pending_count += 1

        if st.pending_count >= self.confirm_needed:
            # Confirmed boundary — close the current event, open a new one.
            closed_reason = st.pending_reason or reason
            seed_people = list(st.pending_people)
            seed_loc = st.pending_location or sig.location
            seed_visual = st.pending_visual if st.pending_visual is not None else visual
            seed_mfcc = st.pending_mfcc if st.pending_mfcc is not None else mfcc
            seed_faces = st.pending_faces if st.pending_faces is not None else faces
            first_at = st.pending_since or now
            self._close_and_log_event(st)
            self._start_event(st, now, seed_people, seed_loc, seed_visual,
                              seed_mfcc, seed_faces, sig.motion_magnitude,
                              first_clip_at=first_at)
            print(f"[event] new event {st.event_id} started ({closed_reason})")
            return EventDecision(
                event_id=st.event_id, status="new_event", is_new_event=True,
                combined_score=combined, signal_scores=scores,
                boundary_pending=0, confirm_needed=self.confirm_needed,
                reason=f"new event: {closed_reason}",
                clip_index_in_event=st.n_clips - 1)

        # Boundary suspected but unconfirmed — this clip still belongs to the
        # current event until a 2nd confirming clip arrives. Deliberately do
        # NOT update the event's reference signature: the next clip must be
        # compared against the *current event*, so that two consecutive
        # divergent clips both register as confirming (§6.3).
        st.last_clip_at = now
        return EventDecision(
            event_id=st.event_id, status="boundary_pending", is_new_event=False,
            combined_score=combined, signal_scores=scores,
            boundary_pending=st.pending_count, confirm_needed=self.confirm_needed,
            reason=(f"possible boundary ({reason}); {st.pending_count}/"
                    f"{self.confirm_needed} confirming clips"),
            clip_index_in_event=st.n_clips - 1)

    def _combine(self, scores: dict) -> Optional[float]:
        """Weighted mean of available per-signal similarities, weights
        renormalised over whichever signals are present. None if no
        comparable signals exist (e.g. the very first comparison)."""
        weights = {"visual": _W_VISUAL, "audio": _W_AUDIO,
                   "face": _W_FACE, "motion": _W_MOTION}
        num = den = 0.0
        for k, s in scores.items():
            w = weights.get(k, 0.0)
            num += w * float(s)
            den += w
        if den <= 0.0:
            return None
        return round(num / den, 3)

    # ── state mutation ─────────────────────────────────────────────────
    def _set_prev_signature(self, st, visual, mfcc, faces, motion) -> None:
        if visual is not None:
            st.prev_visual = visual
        if mfcc is not None:
            st.prev_mfcc = mfcc
        if faces is not None:
            st.prev_faces = list(faces)
        self._update_motion_baseline(st, motion)

    def _update_motion_baseline(self, st, motion) -> None:
        if motion is None:
            return
        try:
            mv = float(motion)
        except Exception:
            return
        if st.motion_baseline is None:
            st.motion_baseline = mv
        else:
            # slow EWMA so a single fast/slow clip doesn't skew the baseline
            st.motion_baseline = 0.8 * st.motion_baseline + 0.2 * mv

    def _clear_pending(self, st) -> None:
        st.pending_count = 0
        st.pending_since = 0.0
        st.pending_reason = ""
        st.pending_people = []
        st.pending_location = None
        st.pending_visual = None
        st.pending_mfcc = None
        st.pending_faces = None

    def _start_event(self, st, now, people, location, visual, mfcc, faces,
                     motion, first_clip_at: Optional[float] = None) -> None:
        st.event_id = _new_event_id()
        st.started_at = float(first_clip_at or now)
        st.last_clip_at = now
        st.n_clips = 1
        st.people = _norm_names(people)
        st.location = location
        st.prev_visual = visual
        st.prev_mfcc = mfcc
        st.prev_faces = list(faces) if faces else None
        st.motion_baseline = float(motion) if motion is not None else None
        self._clear_pending(st)

    def _extend_event(self, st, now, people, location, visual, mfcc, faces,
                      motion) -> None:
        st.last_clip_at = now
        st.n_clips += 1
        st.people = _norm_names(list(st.people) + people)
        if location:
            st.location = location
        self._set_prev_signature(st, visual, mfcc, faces, motion)

    def _close_and_log_event(self, st) -> None:
        """Append the event that is ending to the append-only events log and
        fire the optional on_event_closed hook (event-summary generation)."""
        if not st.event_id:
            return
        record = {
            "event_id": st.event_id,
            "started_at": st.started_at,
            "ended_at": st.last_clip_at,
            "started_iso": _iso(st.started_at),
            "ended_iso": _iso(st.last_clip_at),
            "duration_seconds": max(0.0, st.last_clip_at - st.started_at),
            "n_clips": st.n_clips,
            "people": list(st.people),
            "location": st.location,
        }
        try:
            log = []
            if os.path.exists(self._log_path):
                with open(self._log_path, "r", encoding="utf-8") as f:
                    log = json.load(f) or []
            log.append(record)
            with open(self._log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            print(f"[event] could not append to events log: {e}")
        cb = self.on_event_closed
        if cb is not None:
            try:
                cb(record)
            except Exception as e:
                print(f"[event] on_event_closed callback failed: {e}")

    # ── persistence ────────────────────────────────────────────────────
    def _load_state(self) -> "_EventState":
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path, "r", encoding="utf-8") as f:
                    return _EventState.from_dict(json.load(f))
        except Exception as e:
            print(f"[event] could not load state: {e}")
        return _EventState()

    def _save_state(self) -> None:
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(self._state.to_dict(), f, indent=2)
        except Exception as e:
            print(f"[event] could not save state: {e}")

    def _write_sidecar(self, clip_path: str, dec: EventDecision, now: float,
                       people: list, sig: ClipSignals) -> None:
        if not clip_path:
            return
        meta = {
            "event_id": dec.event_id,
            "status": dec.status,
            "is_new_event": dec.is_new_event,
            "combined_score": dec.combined_score,
            "signal_scores": dec.signal_scores,
            "boundary_pending": dec.boundary_pending,
            "confirm_needed": dec.confirm_needed,
            "reason": dec.reason,
            "clip_index_in_event": dec.clip_index_in_event,
            "received_at": now,
            "received_iso": _iso(now),
            "people_names": list(people),
            "location": sig.location,
            "scene_description": sig.scene_description or "",
            "decided_at": time.time(),
        }
        try:
            with open(sidecar_path(clip_path), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[event] could not write sidecar for "
                  f"{os.path.basename(clip_path)}: {e}")


def read_sidecar(clip_path: str) -> dict:
    """Read the <clip>.event.json sidecar, or {} if absent/unreadable."""
    try:
        with open(sidecar_path(clip_path), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _new_event_id() -> str:
    return "evt_" + uuid.uuid4().hex[:12]


def _iso(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:
        return ""


# ── module-level singleton (mirrors iris_fusion.get_fusion) ───────────────
_detector: Optional[EventBoundaryDetector] = None
_detector_lock = threading.Lock()


def get_detector() -> EventBoundaryDetector:
    """Shared EventBoundaryDetector for the whole app."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = EventBoundaryDetector()
    return _detector