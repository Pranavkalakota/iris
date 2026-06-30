"""
iris_fusion.py — the wiring layer for M5 People (Stage 4).

Gap 1 changes vs original:
  - people_store/ folder-per-person system added.
  - _slugify_name(), _default_people_store_dir() module-level helpers.
  - PeopleFusion.__init__ accepts people_store_dir, creates the root dir.
  - ensure_person_folder() — idempotent, creates folder + profile.json.
  - _unique_folder_for() — collision-safe slug → path.
  - _write_profile_json() — human-readable sidecar, kept in sync with DB.
  - save_reference_face() — writes face_ref.jpg once at enrollment.
  - save_reference_voice() — copies voice_ref.wav once at enrollment.
  - archive_session_media() — copies attributed clips into sessions/.
  - _archive_face_enrollments() — called from process_frame() for new faces.
  - _ingest_voice() updated to call folder hooks after every ingest.
  - _merge_folders() — moves sessions/ on merge_people(), removes drop folder.
  - merge_people() calls _merge_folders() and refreshes profile.json.

Gap 2 changes vs original:
  - _ingest_voice() writes a provisional conversations row (confirmed=False)
    for each newly enrolled or matched person.
  - reconcile_clip() — called by StreamTab after a video clip lands; matches
    unconfirmed rows in the same time window, flips confirmed=True, archives
    the clip into each attributed person's sessions/ folder.
  - stats() includes conversation counts from the DB.
"""

from __future__ import annotations

import os
import re
import json
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

import iris_people
import iris_faces
import iris_voices
import iris_memory


# ── people_store/ layout ─────────────────────────────────────────────────
# Per Pranav's design: every enrolled person gets their own folder on disk.
# SQLite is the index (fast lookups, embedding search, conversations log).
# The folder is the archive (raw media, human-readable profile, future
# cloud-sync target). people.folder_path links the two.
#
# Layout:
#   people_store/
#     humza_malik/
#       profile.json          ← name, role, timestamps (human-readable)
#       face_ref.jpg          ← reference face crop at first enrollment
#       voice_ref.wav         ← reference voice clip at first enrollment
#       sessions/
#         2026-06-27_14-32-15.wav   ← attributed audio copies
#         2026-06-27_14-32-15.avi   ← attributed video copies
#     jacob_chen/
#       ...
#     unknown_1/
#       ...
PEOPLE_STORE_DIRNAME = "people_store"

# How long before the current time to look back when reconciling a newly
# arrived video clip against provisional (unconfirmed) conversation rows.
# 120s = well beyond the 55s worst-case video lag + 35s USB transfer,
# with comfortable headroom.
RECONCILE_WINDOW_SECONDS = 120.0

# Minimum face cosine similarity required for a clip to confirm a
# provisional audio-only identification. Lower than MATCH_THRESHOLD so
# partial views / motion blur still count.
RECONCILE_FACE_THRESHOLD = 0.55


def _slugify_name(name: str, max_len: int = 60) -> str:
    """Turn a person's name into a safe folder name.
    'Humza Malik' → 'humza_malik', 'Unknown 1' → 'unknown_1'.
    ASCII-only, lowercase, underscore-separated, truncated to max_len."""
    s = (name or "").strip().lower()
    # Light accent stripping — covers common names without pulling in
    # the full unicodedata module.
    for src, dst in (("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),
                     ("ñ","n"),("ü","u"),("ä","a"),("ö","o")):
        s = s.replace(src, dst)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if not s:
        s = "person"
    return s[:max_len]


def _default_people_store_dir(db_path: str) -> str:
    """Co-locate people_store/ with the SQLite DB so the whole People
    registry lives in one directory tree (easy to back up / sync)."""
    base = os.path.dirname(os.path.abspath(db_path)) or os.getcwd()
    return os.path.join(base, PEOPLE_STORE_DIRNAME)


# ── LLaVA role inference ─────────────────────────────────────────────────
def _read_ollama_cfg() -> tuple[str, str]:
    url, model = "http://localhost:11434", "llava:7b"
    try:
        import config_phase9 as _cfg                          # type: ignore
        v = getattr(_cfg, "OLLAMA_URL", None)
        if isinstance(v, str) and v:
            url = v
        v = getattr(_cfg, "OLLAMA_LLAVA_MODEL", None)
        if isinstance(v, str) and v:
            model = v
    except Exception:
        pass
    return url, model


def _read_llama_cfg() -> tuple[str, str]:
    """Chat-class Llama for synthesizing conversation summaries. Falls
    back to llama3.2:3b — matches the model the rest of IRIS uses."""
    url, model = "http://localhost:11434", "llama3.2:3b"
    try:
        import config_phase9 as _cfg                          # type: ignore
        v = getattr(_cfg, "OLLAMA_URL", None)
        if isinstance(v, str) and v:
            url = v
        v = getattr(_cfg, "OLLAMA_MODEL", None)
        if isinstance(v, str) and v:
            model = v
    except Exception:
        pass
    return url, model


_SUMMARY_SYSTEM = (
    "You are an assistant that writes 2-3 sentence summaries of "
    "conversations. Be neutral and factual. Focus on what was discussed, "
    "decisions made, and who was involved by name. Do not use bullet "
    "points, headers, or markdown. Plain prose only."
)


class _LlamaSummaryGenerator:
    """Calls Ollama's Llama model to produce a short conversation summary.
    Serialized — only one call at a time so we never compete with the
    chat tab for the same model. Failures swallowed silently."""

    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model
        self._lock = threading.RLock()
        self._client = None
        self._unavailable = False

    def _ensure_client(self):
        if self._unavailable:
            return None
        if self._client is not None:
            return self._client
        try:
            from ollama import Client                          # type: ignore
        except Exception as e:
            print(f"[summary] ollama package not installed: {e}")
            self._unavailable = True
            return None
        try:
            self._client = Client(host=self.url)
        except Exception as e:
            print(f"[summary] client open failed: {e}")
            self._unavailable = True
            return None
        return self._client

    def summarize(self, transcript: str, *,
                  people_names: Optional[list] = None) -> str:
        """Return a 2-3 sentence summary, or '' on any failure. Caller is
        expected to be on a background thread — Llama 3B on CPU takes
        roughly 5-15s for this prompt."""
        if not transcript or not transcript.strip():
            return ""
        client = self._ensure_client()
        if client is None:
            return ""
        names_hint = ""
        if people_names:
            names_hint = "Speakers in this conversation: " + \
                         ", ".join(str(n) for n in people_names if n) + ".\n"
        user_msg = (
            f"{names_hint}Transcript:\n{transcript.strip()}\n\n"
            f"Write a 2-3 sentence summary."
        )
        with self._lock:
            try:
                resp = client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SUMMARY_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                )
            except Exception as e:
                print(f"[summary] llama call failed: {e}")
                return ""
        try:
            msg = resp["message"] if isinstance(resp, dict) \
                else getattr(resp, "message", None)
            if isinstance(msg, dict):
                text = msg.get("content", "") or ""
            else:
                text = getattr(msg, "content", "") or ""
            text = str(text).strip()
        except Exception:
            return ""
        return text


_LLAVA_PROMPT = (
    "You are helping identify someone seen on a wearable camera. Based on "
    "this image, give a SHORT phrase (5-10 words max) describing the "
    "person's likely role or context. Examples: 'barista at coffee shop', "
    "'colleague in office meeting', 'cashier at retail store'. Just the "
    "role phrase, no preamble, no quotes, no explanation."
)


class _LlavaInference:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model
        self._lock = threading.RLock()
        self._client = None
        self._unavailable = False

    def _ensure_client(self):
        if self._unavailable:
            return None
        if self._client is not None:
            return self._client
        try:
            from ollama import Client                          # type: ignore
        except Exception as e:
            print(f"[llava] ollama package not installed: {e}")
            self._unavailable = True
            return None
        try:
            self._client = Client(host=self.url)
        except Exception as e:
            print(f"[llava] could not open client: {e}")
            self._unavailable = True
            return None
        return self._client

    def infer_role(self, image_bgr) -> str:
        client = self._ensure_client()
        if client is None:
            return ""
        try:
            import cv2, base64
            ok, buf = cv2.imencode(".jpg", image_bgr,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return ""
            img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception as e:
            print(f"[llava] image encode failed: {e}")
            return ""
        with self._lock:
            try:
                resp = client.chat(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": _LLAVA_PROMPT,
                        "images": [img_b64],
                    }],
                )
            except Exception as e:
                print(f"[llava] inference failed: {e}")
                return ""
        try:
            msg = resp["message"] if isinstance(resp, dict) \
                else getattr(resp, "message", None)
            if isinstance(msg, dict):
                text = msg.get("content", "") or ""
            else:
                text = getattr(msg, "content", "") or ""
            text = str(text).strip()
        except Exception:
            return ""
        text = text.replace("\n", " ").replace("\r", " ").strip()
        text = text.strip(" \"'.\u201c\u201d")
        if len(text) > 80:
            text = text[:77] + "\u2026"
        return text


# ── result types ────────────────────────────────────────────────────────
@dataclass
class MergeReport:
    keep_id: int
    drop_id: int
    drop_name: str = ""
    kept_name: str = ""
    embeddings_moved_face: int = 0
    embeddings_moved_voice: int = 0
    times_seen_added: int = 0
    success: bool = False
    error: str = ""


# ── the fusion object ───────────────────────────────────────────────────
class PeopleFusion:
    """Single object the rest of the app talks to about people."""

    def __init__(self,
                 db_path: Optional[str] = None,
                 recordings_dir: Optional[str] = None,
                 people_store_dir: Optional[str] = None):
        self.db_path = db_path or iris_people.default_db_path()
        self.recordings_dir = recordings_dir

        # people_store/ root — every enrolled person gets a subfolder here.
        self.people_store_dir = (people_store_dir
                                 or _default_people_store_dir(self.db_path))
        try:
            os.makedirs(self.people_store_dir, exist_ok=True)
        except Exception as e:
            print(f"[fusion] could not create people_store dir "
                  f"{self.people_store_dir}: {e}")

        self.store: iris_people.PeopleStore = iris_people.PeopleStore(
            self.db_path)
        self.faces = iris_faces.get_pipeline()
        self.voices = iris_voices.get_pipeline()

        self._started = False
        self._lock = threading.RLock()
        self._backfill_thread: Optional[threading.Thread] = None

        self._diarizer: Optional[object] = None
        self._original_on_done: Optional[Callable] = None

        self._frame_inflight = threading.Lock()

        self.on_voice_ingested: Optional[Callable[
            [iris_voices.IngestionResult], None]] = None
        self.on_faces_processed: Optional[Callable[
            [list[iris_faces.ProcessedFace]], None]] = None

        _u, _m = _read_ollama_cfg()
        self.llava = _LlavaInference(_u, _m)
        self._llava_inflight: set[int] = set()
        self._llava_inflight_lock = threading.Lock()

        # ChromaDB memory store — lazy-loaded so IRIS startup isn't
        # blocked by sentence-transformers initialization (~5-10s on CPU).
        self.memory = iris_memory.get_memory()
        # Llama client for the background conversation-summary worker.
        # Same lazy pattern as LLaVA: client opened on first use.
        _llama_url, _llama_model = _read_llama_cfg()
        self.llama_summary = _LlamaSummaryGenerator(_llama_url, _llama_model)
        self._summary_inflight: set[str] = set()
        self._summary_inflight_lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self, controller=None, *,
              warm_faces: bool = True,
              backfill: bool = True) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True

        if controller is not None:
            self._attach_to_controller(controller)

        if self.recordings_dir is None and controller is not None:
            self.recordings_dir = self._resolve_recordings_dir(controller)

        if backfill and self.recordings_dir:
            self._backfill_thread = threading.Thread(
                target=self._backfill_then_warm_faces,
                args=(warm_faces,),
                name="PeopleFusionBackfill",
                daemon=True,
            )
            self._backfill_thread.start()
        elif warm_faces:
            self.faces.warm_up(blocking=False)

    def _backfill_then_warm_faces(self, warm_faces: bool) -> None:
        try:
            results = self.voices.scan_directory(
                self.recordings_dir, self.store)
        except Exception as e:
            print(f"[fusion] backfill scan failed: {e}")
            results = []
        fresh   = sum(1 for r in results if not r.skipped and not r.error)
        skipped = sum(1 for r in results if r.skipped)
        if fresh or skipped:
            print(f"[fusion] backfill: {fresh} new WAVs ingested, "
                  f"{skipped} already done")
        # Ensure folders exist for everyone already in the DB.
        for person in self.store.list_all():
            try:
                self.ensure_person_folder(person.id)
            except Exception:
                pass
        if warm_faces:
            time.sleep(2.0)
            self.faces.warm_up(blocking=False)

    def shutdown(self) -> None:
        with self._lock:
            if not self._started:
                self._safe_close_store()
                return
            self._started = False
        if self._diarizer is not None:
            try:
                self._diarizer._on_done = self._original_on_done
            except Exception:
                pass
            self._diarizer = None
            self._original_on_done = None
        if self._backfill_thread is not None:
            try:
                self._backfill_thread.join(timeout=3.0)
            except Exception:
                pass
            self._backfill_thread = None
        self._safe_close_store()

    def _safe_close_store(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass

    # ── audio side: attach to the diarizer ───────────────────────────────
    def _attach_to_controller(self, controller) -> None:
        diarizer = getattr(controller, "diarizer", None)
        if diarizer is None:
            print("[fusion] controller has no diarizer; skipping audio hook")
            return
        self._diarizer = diarizer
        self._original_on_done = getattr(diarizer, "_on_done", None)

        def chained(wav_path: str) -> None:
            if self._original_on_done is not None:
                try:
                    self._original_on_done(wav_path)
                except Exception as e:
                    print(f"[fusion] original on_done raised: {e}")
            try:
                self._ingest_voice(wav_path)
            except Exception as e:
                print(f"[fusion] voice ingest failed: {e}")

        diarizer._on_done = chained
        print("[fusion] attached to diarizer; voice ingestion is live")

    def _ingest_voice(self, wav_path: str) -> None:
        """Ingest a completed WAV, update the People registry, write
        provisional conversation rows, and archive media into person folders."""
        result = self.voices.process_recording(wav_path, self.store)

        session_start = time.time()
        # Track the conv_ids we just wrote so we can stamp duration/speaker
        # stats and run significance checks afterwards.
        conv_ids: list[int] = []
        # And keep around the per-person ProcessedVoice so we can issue
        # confidence-confirmation prompts for borderline matches.
        for pv in result.processed_voices:
            try:
                # Ensure folder exists for every attributed person.
                self.ensure_person_folder(pv.person_id)

                # New enrollments get the WAV as their voice reference.
                if pv.was_new_enrollment:
                    self.save_reference_voice(pv.person_id, wav_path)

                # Archive a copy of this session's audio into their folder.
                self.archive_session_media(pv.person_id, wav_path)

                # Write a provisional conversation row (Gap 2: confirmed=False).
                # The StreamTab reconcile_clip() call will flip this to True
                # once the matching video clip arrives (≤55s later).
                cid = self.store.add_conversation(
                    person_id=pv.person_id,
                    session_start=session_start,
                    wav_path=wav_path,
                    audio_received_at=time.time(),
                )
                if cid is not None:
                    conv_ids.append(cid)

                # Revamp: queue a confidence-confirmation prompt for
                # borderline voice matches (0.60-0.70 band). Dedupe in
                # the people DB handles the "don't prompt repeatedly" rule.
                if pv.needs_confirmation and not pv.was_new_enrollment:
                    self.store.add_pending_prompt(
                        "confirm_identity",
                        person_id=pv.person_id,
                        payload={
                            "name": pv.name,
                            "similarity": round(pv.similarity, 3),
                            "wav_path": wav_path,
                            "session_start": session_start,
                            "reason": "voice_match_low_confidence",
                        })

            except Exception as e:
                print(f"[fusion] voice folder/convo hook failed for "
                      f"{pv.name!r}: {e}")

        # Revamp: stamp conversation-level stats and evaluate significance.
        try:
            self._stamp_conversation_stats(conv_ids, result)
        except Exception as e:
            print(f"[fusion] conversation stats stamp failed: {e}")

        try:
            self._evaluate_significance(result, conv_ids)
        except Exception as e:
            print(f"[fusion] significance check failed: {e}")

        # Revamp: spoken-name prompts ("you mentioned Sarah — add her?").
        try:
            self._queue_name_mention_prompts(result)
        except Exception as e:
            print(f"[fusion] mentioned-name prompt failed: {e}")

        # ── M7: write a searchable memory record ─────────────────────────
        # Stored immediately with summary="" so the record is searchable
        # by transcript text and person right away. A background worker
        # generates the Llama summary and patches it in seconds later.
        try:
            self._store_memory_record(wav_path, result)
        except Exception as e:
            print(f"[fusion] memory store failed: {e}")

        cb = self.on_voice_ingested
        if cb is not None:
            try:
                cb(result)
            except Exception as e:
                print(f"[fusion] on_voice_ingested callback failed: {e}")

    # ── revamp helpers: stats + significance + name-mention prompts ──────
    def _stamp_conversation_stats(self, conv_ids: list[int],
                                  result: iris_voices.IngestionResult
                                  ) -> None:
        """Write duration / speaker_count / dominant_share onto every
        conversation row we just created for this recording."""
        if not conv_ids:
            return
        for cid in conv_ids:
            self.store.update_conversation_stats(
                cid,
                duration_seconds=result.duration_seconds,
                speaker_count=result.speaker_count,
                dominant_share=result.dominant_share,
            )

    def _evaluate_significance(self,
                               result: iris_voices.IngestionResult,
                               conv_ids: list[int]) -> None:
        """Decide if this conversation warrants surfacing a prompt:
          • 5+ min AND 2+ speakers → if any speaker is Unknown N, prompt
            "who is this?" so the user can profile them.
          • Even short convos (<5 min) → if dominant speaker is Unknown
            and spoke >60% of the time, prompt "identify dominant speaker".
        Dedupe in pending_prompts prevents the same prompt re-firing."""
        dur = result.duration_seconds
        n_speakers = result.speaker_count
        dom_share = result.dominant_share
        dom_pid = result.dominant_person_id

        # Find Unknown attendees in this conversation.
        unknown_attendees: list[iris_voices.ProcessedVoice] = []
        for pv in result.processed_voices:
            person = self.store.get(pv.person_id)
            if person is None:
                continue
            if self._is_placeholder_name(person.name):
                unknown_attendees.append(pv)

        # Rule A: long multi-speaker conversation with Unknown attendees.
        if dur >= 300.0 and n_speakers >= 2 and unknown_attendees:
            for pv in unknown_attendees:
                self.store.add_pending_prompt(
                    "long_unknown",
                    person_id=pv.person_id,
                    payload={
                        "name": pv.name,
                        "duration_seconds": round(dur, 1),
                        "speaker_count": n_speakers,
                        "wav_path": pv.wav_path,
                        "reason": "long_multi_speaker_conversation",
                    })

        # Rule B: dominant speaker is an Unknown — even on short clips.
        if dom_pid > 0 and dom_share >= 0.60:
            dom_person = self.store.get(dom_pid)
            if dom_person is not None \
                    and self._is_placeholder_name(dom_person.name):
                self.store.add_pending_prompt(
                    "long_unknown",
                    person_id=dom_pid,
                    payload={
                        "name": dom_person.name,
                        "duration_seconds": round(dur, 1),
                        "dominant_share": round(dom_share, 2),
                        "speaker_count": n_speakers,
                        "reason": "dominant_unknown_speaker",
                    })

    def _queue_name_mention_prompts(self,
                                    result: iris_voices.IngestionResult
                                    ) -> None:
        """For each proper-noun-looking name spoken in the transcript that
        doesn't match an existing person, queue an 'add_person' prompt.
        Uses dedupe so a name mentioned across multiple recordings only
        prompts once per 24h."""
        for name in (result.mentioned_names or []):
            # Skip if a person with this name already exists (case-insens).
            existing = self.store.get_by_name(name)
            if existing is not None:
                continue
            # Try case-insensitive lookup too.
            already = False
            for p in self.store.list_all():
                if p.name.lower() == name.lower():
                    already = True
                    break
            if already:
                continue
            self.store.add_pending_prompt(
                "name_mentioned",
                person_id=0,
                payload={
                    "mentioned_name": name,
                    "wav_path": result.wav_path,
                    "reason": "transcript_name_not_in_registry",
                })

    # ── M7: memory + summary ─────────────────────────────────────────────
    @staticmethod
    def _read_transcript_text(wav_path: str) -> str:
        """Read the .json sidecar that lives next to the wav and pull
        out the full transcript text. Returns '' if not available."""
        try:
            stem, _ = os.path.splitext(wav_path)
            jpath = stem + ".json"
            if not os.path.exists(jpath):
                return ""
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return ""
        # Prefer joining segment texts so we get speaker labels if present.
        segs = data.get("segments") or []
        if segs:
            chunks: list[str] = []
            for s in segs:
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                speaker = (s.get("speaker") or "").strip()
                if speaker:
                    chunks.append(f"{speaker}: {text}")
                else:
                    chunks.append(text)
            return "\n".join(chunks)
        # Fall back to a top-level 'text' field if the sidecar has one.
        t = data.get("text") or ""
        return str(t)

    def _store_memory_record(self, wav_path: str,
                             result: iris_voices.IngestionResult) -> None:
        """Write a ChromaDB record for this recording immediately, then
        kick off a background worker to fill in the Llama summary."""
        if self.memory is None:
            return
        # Build a stable seg_id from the wav stem so re-ingestion of the
        # same file upserts cleanly instead of duplicating.
        try:
            stem = os.path.splitext(os.path.basename(wav_path))[0]
        except Exception:
            stem = f"session_{int(time.time())}"
        seg_id = f"seg_{stem}"

        transcript = self._read_transcript_text(wav_path)

        people_ids: list[int] = []
        people_names: list[str] = []
        seen_ids: set[int] = set()
        for pv in result.processed_voices:
            if pv.person_id in seen_ids:
                continue
            seen_ids.add(pv.person_id)
            people_ids.append(int(pv.person_id))
            # Pull the *current* name from the store so renames are
            # reflected — the ProcessedVoice has the name at ingest time
            # which can be 'Unknown N' even after the user later renames.
            person = self.store.get(pv.person_id)
            people_names.append(person.name if person else pv.name)

       # Derive session_start from file mtime; fall back to now.
        # Guard against empty processed_voices to avoid IndexError.
        try:
            session_start = float(os.path.getmtime(wav_path)) \
                if os.path.exists(wav_path) else time.time()
        except Exception:
            session_start = time.time()

        ok = self.memory.store_segment(
            seg_id=seg_id,
            session_start=session_start,
            duration_seconds=float(result.duration_seconds),
            people_names=people_names,
            people_ids=people_ids,
            dominant_person_id=int(result.dominant_person_id or 0),
            dominant_share=float(result.dominant_share or 0.0),
            transcript=transcript,
            summary="",                # filled in by background worker
            location="",               # filled in by M6 location pipeline
            wav_path=wav_path,
            clip_path="",
            confirmed=False,
        )
        if not ok:
            return  # ChromaDB unavailable; nothing more to do.

        # Kick off the summary worker. Dedupe so re-ingest of the same
        # wav doesn't fire two summary calls in parallel.
        with self._summary_inflight_lock:
            if seg_id in self._summary_inflight:
                return
            self._summary_inflight.add(seg_id)
        if not transcript.strip():
            # Nothing to summarize. Mark done so we don't loop.
            with self._summary_inflight_lock:
                self._summary_inflight.discard(seg_id)
            return
        threading.Thread(
            target=self._summary_worker,
            args=(seg_id, transcript, list(people_names)),
            name=f"LlamaSummary-{seg_id}",
            daemon=True,
        ).start()

    def _summary_worker(self, seg_id: str, transcript: str,
                        people_names: list) -> None:
        """Background: ask Llama for a 2-3 sentence summary and patch it
        onto the memory record. Never raises."""
        try:
            summary = self.llama_summary.summarize(
                transcript, people_names=people_names)
            if summary:
                if self.memory.update_summary(seg_id, summary):
                    print(f"[summary] {seg_id}: {summary[:80]}"
                          f"{'…' if len(summary) > 80 else ''}")
        except Exception as e:
            print(f"[summary] worker failed for {seg_id}: {e}")
        finally:
            with self._summary_inflight_lock:
                self._summary_inflight.discard(seg_id)

    @staticmethod
    def _resolve_recordings_dir(controller) -> Optional[str]:
        for attr in ("recordings_dir", "RECORDINGS_DIR"):
            v = getattr(controller, attr, None)
            if isinstance(v, str) and v:
                return v
        try:
            import config_phase9 as cfg     # type: ignore
            for attr in ("RECORDINGS_DIR", "RECORDING_DIR",
                         "AUDIO_DIR", "AUDIO_SAVE_DIR"):
                v = getattr(cfg, attr, None)
                if isinstance(v, str) and v:
                    return v
        except Exception:
            pass
        return None

    # ── face side: process a frame ───────────────────────────────────────
    def process_frame(self, image_bgr: np.ndarray
                      ) -> list[iris_faces.ProcessedFace]:
        if not self._frame_inflight.acquire(blocking=False):
            return []
        try:
            if not self.faces.is_ready():
                return []
            results = self.faces.process_frame(image_bgr, self.store)
        except Exception as e:
            print(f"[fusion] process_frame failed: {e}")
            results = []
        finally:
            self._frame_inflight.release()

        if results:
            self._maybe_infer_roles(image_bgr, results)
            self._archive_face_enrollments(image_bgr, results)

        cb = self.on_faces_processed
        if cb is not None and results:
            try:
                cb(results)
            except Exception as e:
                print(f"[fusion] on_faces_processed callback failed: {e}")
        return results

    # ── Gap 2: reconcile a video clip against provisional conversations ──
    def reconcile_clip(self, clip_path: str) -> int:
        """Called by StreamTab after a video clip has been received and
        processed. Looks up any unconfirmed conversation rows from the past
        RECONCILE_WINDOW_SECONDS, checks whether the faces found in the
        clip match those persons, and flips confirmed=True for every match.

        Returns the number of conversation rows confirmed.

        This is the two-pass ID bridge: audio runs first (~2s), writes
        provisional rows; video arrives up to 55s later, this method
        reconciles them. A row only stays unconfirmed if no face evidence
        arrives in the window (e.g. person was off-camera the whole clip).
        """
        if not clip_path or not os.path.exists(clip_path):
            return 0
        if not self.faces.is_ready():
            return 0

        # Find unconfirmed rows in the recent window.
        since = time.time() - RECONCILE_WINDOW_SECONDS
        pending = self.store.list_unconfirmed(since=since)
        if not pending:
            return 0

        # Extract face embeddings from the clip (1 keyframe/sec).
        clip_faces = self._extract_clip_face_embeddings(clip_path)
        if not clip_faces:
            # No detectable faces in the clip — can't confirm anything,
            # but don't discard the rows either; they stay pending until
            # the next clip or until expiry.
            return 0

        confirmed_count = 0
        for row in pending:
            person_id = int(row["person_id"])
            conv_id   = int(row["id"])
            stored_embs = self.store.list_embeddings(
                person_id, iris_people.KIND_FACE)
            if not stored_embs:
                # Voice-only person — no face embeddings to compare.
                # Confirm by default (audio was enough).
                if self.store.confirm_conversation(
                        conv_id, clip_path=clip_path,
                        video_received_at=time.time()):
                    self.archive_session_media(person_id, clip_path)
                    confirmed_count += 1
                continue
            # Check whether any detected face in the clip matches this person.
            best_sim = 0.0
            for clip_emb in clip_faces:
                from iris_people import _best_sim
                sim = _best_sim(clip_emb, stored_embs)
                if sim > best_sim:
                    best_sim = sim
            if best_sim >= RECONCILE_FACE_THRESHOLD:
                if self.store.confirm_conversation(
                        conv_id, clip_path=clip_path,
                        video_received_at=time.time()):
                    self.archive_session_media(person_id, clip_path)
                    confirmed_count += 1
                    print(f"[fusion] reconcile: confirmed {row['name']} "
                          f"(sim={best_sim:.2f}) from "
                          f"{os.path.basename(clip_path)}")
                    # M7: link the clip to the matching memory record.
                    try:
                        wav_path = row.get("wav_path", "") or ""
                        if wav_path and self.memory is not None:
                            stem = os.path.splitext(
                                os.path.basename(wav_path))[0]
                            self.memory.update_clip(
                                f"seg_{stem}", clip_path, confirmed=True)
                    except Exception as e:
                        print(f"[fusion] memory update_clip failed: {e}")
            else:
                # Revamp: face evidence exists in the clip but doesn't
                # confirm this person above threshold. The voice ID was
                # below 70% or the face disagrees — queue a confirmation
                # prompt so the user can adjudicate. Dedupe takes care of
                # repeat suppression across multiple clips.
                self.store.add_pending_prompt(
                    "confirm_identity",
                    person_id=person_id,
                    payload={
                        "name": row["name"],
                        "face_similarity": round(best_sim, 3),
                        "clip_path": clip_path,
                        "wav_path": row.get("wav_path", ""),
                        "reason": "voice_video_mismatch",
                    })

        return confirmed_count

    def _extract_clip_face_embeddings(self,
                                      clip_path: str
                                      ) -> list[np.ndarray]:
        """Extract one keyframe per second from the clip, detect faces,
        and return all ArcFace embeddings found. Returns [] on any failure.
        Deliberately light-weight: no DB writes, no reinforce, just the
        raw embedding vectors for comparison in reconcile_clip()."""
        try:
            import cv2
        except ImportError:
            return []
        embeddings: list[np.ndarray] = []
        try:
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                return []
            fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            step = max(1, int(fps))
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    crops = self.faces.extract_faces(frame)
                    for crop in crops:
                        emb = self.faces.embed(crop.image)
                        if emb is not None:
                            embeddings.append(emb)
                idx += 1
            cap.release()
        except Exception as e:
            print(f"[fusion] _extract_clip_face_embeddings failed: {e}")
        return embeddings

    # ── LLaVA role inference ─────────────────────────────────────────────
    def _maybe_infer_roles(self, image_bgr: np.ndarray,
                           results: list[iris_faces.ProcessedFace]
                           ) -> None:
        for pf in results:
            if not getattr(pf, "was_new_enrollment", False):
                continue
            pid = int(pf.person_id)
            with self._llava_inflight_lock:
                if pid in self._llava_inflight:
                    continue
                self._llava_inflight.add(pid)
            try:
                frame_copy = image_bgr.copy() if image_bgr is not None else None
            except Exception:
                frame_copy = None
            threading.Thread(
                target=self._infer_role_worker,
                args=(pid, frame_copy),
                name=f"LlavaRole-{pid}",
                daemon=True,
            ).start()

    def _infer_role_worker(self, person_id: int,
                           image_bgr: Optional[np.ndarray]) -> None:
        try:
            person = self.store.get(person_id)
            if person is None or person.role_note:
                return
            if image_bgr is None:
                return
            role = self.llava.infer_role(image_bgr)
            if not role:
                return
            fresh = self.store.get(person_id)
            if fresh is None or fresh.role_note:
                return
            if self.store.update_role_note(person_id, role):
                print(f"[llava] {fresh.name} \u2192 '{role}'")
                # Refresh profile.json with the new role.
                self.ensure_person_folder(person_id)
        except Exception as e:
            print(f"[llava] role worker failed for id={person_id}: {e}")
        finally:
            with self._llava_inflight_lock:
                self._llava_inflight.discard(person_id)

    # ── face-side folder hook ────────────────────────────────────────────
    def _archive_face_enrollments(self, image_bgr: np.ndarray,
                                  results: list[iris_faces.ProcessedFace]
                                  ) -> None:
        """For each newly-enrolled face, save its detection crop as
        face_ref.jpg in the person's folder."""
        if image_bgr is None:
            return
        for pf in results:
            if not getattr(pf, "was_new_enrollment", False):
                continue
            try:
                x, y, w, h = pf.bbox
                h_img, w_img = image_bgr.shape[:2]
                x0 = max(0, int(x)); y0 = max(0, int(y))
                x1 = min(w_img, int(x + w)); y1 = min(h_img, int(y + h))
                if x1 - x0 <= 0 or y1 - y0 <= 0:
                    continue
                crop = image_bgr[y0:y1, x0:x1]
                self.save_reference_face(pf.person_id, crop)
            except Exception as e:
                print(f"[fusion] face folder hook failed for "
                      f"{pf.name!r}: {e}")

    # ── folder-per-person ────────────────────────────────────────────────
    def ensure_person_folder(self, person_id: int) -> Optional[str]:
        """Create (if missing) and return the absolute path to a person's
        folder. Idempotent — safe to call on every interaction. Writes
        profile.json on creation and refreshes it on subsequent calls so
        renames and role changes stay in sync with the DB."""
        person = self.store.get(person_id)
        if person is None:
            return None
        folder = (person.folder_path or "").strip()
        if not folder:
            folder = self._unique_folder_for(person.name)
            self.store.set_folder_path(person_id, folder)
        try:
            os.makedirs(folder, exist_ok=True)
            os.makedirs(os.path.join(folder, "sessions"), exist_ok=True)
        except Exception as e:
            print(f"[fusion] could not create folder for "
                  f"{person.name!r}: {e}")
            return None
        self._write_profile_json(folder, person)
        return folder

    def _unique_folder_for(self, name: str) -> str:
        """Pick a fresh folder path for `name`. Falls back to a numeric
        suffix if the slug collides with an existing folder — never
        overwrites, to avoid accidentally merging two different people."""
        base_slug = _slugify_name(name)
        candidate = os.path.join(self.people_store_dir, base_slug)
        if not os.path.exists(candidate):
            return candidate
        n = 2
        while True:
            candidate = os.path.join(self.people_store_dir,
                                     f"{base_slug}_{n}")
            if not os.path.exists(candidate):
                return candidate
            n += 1

    def _write_profile_json(self, folder: str,
                            person: iris_people.Person) -> None:
        """Human-readable profile sidecar. Mirrors the SQLite row but
        excludes embeddings (those belong in the DB for matching).
        Refreshed on every ensure_person_folder() call."""
        path = os.path.join(folder, "profile.json")
        payload = {
            "id": person.id,
            "name": person.name,
            "role_note": person.role_note,
            "first_seen": person.first_seen,
            "last_seen": person.last_seen,
            "times_seen": person.times_seen,
            "created_at": person.created_at,
            "updated_at": person.updated_at,
            "face_count": person.face_count,
            "voice_count": person.voice_count,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[fusion] could not write profile.json for "
                  f"{person.name!r}: {e}")

    def save_reference_face(self, person_id: int,
                            image_bgr: np.ndarray) -> Optional[str]:
        """Drop a face_ref.jpg into the person's folder. Called once at
        enrollment with the detection crop. Skipped if a reference already
        exists (first sight is the canonical reference)."""
        folder = self.ensure_person_folder(person_id)
        if folder is None:
            return None
        path = os.path.join(folder, "face_ref.jpg")
        if os.path.exists(path):
            return path
        try:
            import cv2
            ok = cv2.imwrite(path, image_bgr,
                             [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return None
            return path
        except Exception as e:
            print(f"[fusion] could not save face_ref.jpg: {e}")
            return None

    def save_reference_voice(self, person_id: int,
                             source_wav: str) -> Optional[str]:
        """Copy the WAV that produced this person's first voiceprint into
        their folder as voice_ref.wav. Skipped if a reference exists."""
        folder = self.ensure_person_folder(person_id)
        if folder is None:
            return None
        if not source_wav or not os.path.exists(source_wav):
            return None
        path = os.path.join(folder, "voice_ref.wav")
        if os.path.exists(path):
            return path
        try:
            shutil.copy2(source_wav, path)
            return path
        except Exception as e:
            print(f"[fusion] could not save voice_ref.wav: {e}")
            return None

    def archive_session_media(self, person_id: int,
                              source_path: str) -> Optional[str]:
        """Copy an attributed clip or recording into the person's
        sessions/ subfolder, named by current timestamp. Best-effort:
        failure here never affects identification or DB state."""
        if not source_path or not os.path.exists(source_path):
            return None
        folder = self.ensure_person_folder(person_id)
        if folder is None:
            return None
        sessions_dir = os.path.join(folder, "sessions")
        ext = os.path.splitext(source_path)[1] or ".bin"
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        dest = os.path.join(sessions_dir, f"{stamp}{ext}")
        # Don't double-copy if already archived in the same second.
        if os.path.exists(dest):
            return dest
        try:
            shutil.copy2(source_path, dest)
            return dest
        except Exception as e:
            print(f"[fusion] could not archive {source_path}: {e}")
            return None

    # ── fusion primitive: merge two rows ─────────────────────────────────
    def merge_people(self, keep_id: int, drop_id: int) -> MergeReport:
        report = MergeReport(keep_id=keep_id, drop_id=drop_id)
        if keep_id == drop_id:
            report.error = "keep_id and drop_id are the same"
            return report

        keep = self.store.get(keep_id)
        drop = self.store.get(drop_id)
        if keep is None:
            report.error = f"keep_id {keep_id} not found"
            return report
        if drop is None:
            report.error = f"drop_id {drop_id} not found"
            return report
        report.kept_name = keep.name
        report.drop_name = drop.name

        for kind in (iris_people.KIND_FACE, iris_people.KIND_VOICE):
            for emb in self.store.list_embeddings(drop_id, kind):
                if self.store.add_embedding(keep_id, kind, emb):
                    if kind == iris_people.KIND_FACE:
                        report.embeddings_moved_face += 1
                    else:
                        report.embeddings_moved_voice += 1

        for _ in range(int(drop.times_seen or 0)):
            self.store.mark_seen(keep_id)
            report.times_seen_added += 1

        if self._is_placeholder_name(keep.name) \
                and not self._is_placeholder_name(drop.name):
            self.store.rename(keep_id, drop.name)
            report.kept_name = drop.name

        if drop.role_note and not keep.role_note:
            self.store.update_role_note(keep_id, drop.role_note)

        # Folder side: move sessions/ from drop into keep before deleting.
        try:
            self._merge_folders(keep_id=keep_id, drop_id=drop_id)
        except Exception as e:
            print(f"[fusion] folder merge failed (non-fatal): {e}")

        ok = self.store.delete(drop_id)
        report.success = ok
        if not ok:
            report.error = "delete failed"
        else:
            # Refresh profile.json with post-merge state.
            self.ensure_person_folder(keep_id)
        return report

    def _merge_folders(self, *, keep_id: int, drop_id: int) -> None:
        """Move sessions/ contents from the dropped person's folder into
        the kept person's folder, then remove the drop folder entirely.
        face_ref / voice_ref from the drop are NOT moved — the kept person
        already has their canonical references."""
        drop = self.store.get(drop_id)
        keep = self.store.get(keep_id)
        if drop is None or keep is None:
            return
        drop_folder = (drop.folder_path or "").strip()
        if not drop_folder or not os.path.isdir(drop_folder):
            return
        keep_folder = self.ensure_person_folder(keep_id)
        if keep_folder is None:
            return
        drop_sessions = os.path.join(drop_folder, "sessions")
        keep_sessions = os.path.join(keep_folder, "sessions")
        os.makedirs(keep_sessions, exist_ok=True)
        if os.path.isdir(drop_sessions):
            for fn in os.listdir(drop_sessions):
                src = os.path.join(drop_sessions, fn)
                dst = os.path.join(keep_sessions, fn)
                if os.path.exists(dst):
                    stem, ext = os.path.splitext(fn)
                    dst = os.path.join(keep_sessions,
                                       f"{stem}_merged{ext}")
                try:
                    shutil.move(src, dst)
                except Exception as e:
                    print(f"[fusion] could not move {src} → {dst}: {e}")
        try:
            shutil.rmtree(drop_folder, ignore_errors=True)
        except Exception as e:
            print(f"[fusion] could not remove drop folder "
                  f"{drop_folder}: {e}")

    @staticmethod
    def _is_placeholder_name(name: str) -> bool:
        n = (name or "").strip()
        if not n:
            return True
        return n.lower().startswith("unknown")

    # ── convenience pass-throughs for Stage 5 UI ────────────────────────
    def list_people(self) -> list[iris_people.Person]:
        return self.store.list_all()

    def get_person(self, person_id: int) -> Optional[iris_people.Person]:
        return self.store.get(person_id)

    def rename(self, person_id: int, new_name: str) -> bool:
        ok = self.store.rename(person_id, new_name)
        if ok:
            # Refresh profile.json so the human-readable sidecar stays in
            # sync, but do NOT rename the folder — that would break the
            # stored folder_path. The slug is created at enrollment time
            # and is stable from then on.
            self.ensure_person_folder(person_id)
        return ok

    def update_role_note(self, person_id: int, role_note: str) -> bool:
        ok = self.store.update_role_note(person_id, role_note)
        if ok:
            self.ensure_person_folder(person_id)
        return ok

    def delete_person(self, person_id: int) -> bool:
        return self.store.delete(person_id)

    # ── revamp: profile editing + manual add + prompts + conversations ──
    def add_person_manual(self, name: str,
                          *, title: str = "", company: str = "",
                          relationship: str = "", role_note: str = "",
                          is_self: bool = False,
                          face_image_path: Optional[str] = None,
                          voice_wav_path: Optional[str] = None
                          ) -> Optional[iris_people.Person]:
        """Manual enrollment from the People tab '+ Add Person' form.
        Optionally seeds a face from an image file path and/or a voice
        from a WAV path. Face embedding goes through iris_faces; voice
        embedding requires the diarizer-emitted NPZ sidecar so we just
        archive the WAV as voice_ref.wav without a fresh embedding."""
        face_emb = None
        if face_image_path and os.path.exists(face_image_path):
            try:
                import cv2
                img = cv2.imread(face_image_path)
                if img is not None and self.faces.is_ready():
                    crops = self.faces.extract_faces(img)
                    if crops:
                        # Pick the largest face in the image.
                        crops.sort(key=lambda c: c.image.shape[0]
                                                  * c.image.shape[1],
                                   reverse=True)
                        face_emb = self.faces.embed(crops[0].image)
            except Exception as e:
                print(f"[fusion] manual face embed failed: {e}")
        person = self.store.add(
            name=name,
            role_note=role_note,
            face_embedding=face_emb,
            title=title,
            company=company,
            relationship=relationship,
            is_self=is_self,
        )
        if person is None:
            return None
        try:
            self.ensure_person_folder(person.id)
            # Stamp face_ref.jpg if we got one.
            if face_image_path and os.path.exists(face_image_path):
                try:
                    import cv2
                    img = cv2.imread(face_image_path)
                    if img is not None:
                        self.save_reference_face(person.id, img)
                except Exception:
                    pass
            if voice_wav_path and os.path.exists(voice_wav_path):
                self.save_reference_voice(person.id, voice_wav_path)
        except Exception as e:
            print(f"[fusion] manual add post-hooks failed: {e}")
        return person

    def update_profile(self, person_id: int, **fields) -> bool:
        """Patch any profile field — passes through to PeopleStore and
        refreshes the on-disk profile.json afterwards."""
        ok = self.store.update_person_details(person_id, **fields)
        if ok:
            self.ensure_person_folder(person_id)
        return ok

    def get_self(self) -> Optional[iris_people.Person]:
        return self.store.get_self()

    def list_pending_prompts(self) -> list[iris_people.PendingPrompt]:
        return self.store.list_pending_prompts()

    def dismiss_prompt(self, prompt_id: int,
                       *, acted_on: bool = False) -> bool:
        return self.store.dismiss_prompt(prompt_id, acted_on=acted_on)

    def list_conversations_for(self, person_id: int,
                               *, confirmed_only: bool = False
                               ) -> list[dict]:
        return self.store.list_conversations(person_id,
                                             confirmed_only=confirmed_only)

    def stats(self) -> dict:
        s = self.store.stats()
        s["face_pipeline_ready"]  = self.faces.is_ready()
        s["recordings_dir"]       = self.recordings_dir or ""
        s["db_path"]              = self.db_path
        s["people_store_dir"]     = self.people_store_dir
        return s


# ── module-level singleton ───────────────────────────────────────────────
_fusion_singleton: Optional[PeopleFusion] = None
_singleton_lock = threading.Lock()


def get_fusion(db_path: Optional[str] = None,
               recordings_dir: Optional[str] = None,
               people_store_dir: Optional[str] = None) -> PeopleFusion:
    """Return the shared PeopleFusion singleton."""
    global _fusion_singleton
    with _singleton_lock:
        if _fusion_singleton is None:
            _fusion_singleton = PeopleFusion(
                db_path=db_path,
                recordings_dir=recordings_dir,
                people_store_dir=people_store_dir,
            )
        return _fusion_singleton