"""
iris_voices.py — voice-print → people-registry bridge (M5 Stage 3).

Gap 1 / Gap 3 fixes vs original:
  - _process_one_cluster was at module level (indentation bug) — now
    correctly indented as a method of VoicePipeline.
  - Added _is_placeholder_name() staticmethod that _process_one_cluster
    references but that didn't exist in the class.
  - No other logic changes — all matching, reinforce band, name bridging,
    and idempotent marker behaviour is identical to the original.

Reads the diarizer's output and feeds it into iris_people.PeopleStore.
No heavy model load here — the expensive SpeechBrain inference already
happens inside diarizer_phase9.Diarizer. We just consume the byproducts:

    <stem>.wav                 (audio)
    <stem>.json                (whisper transcript + diarizer labels)
    <stem>.embeddings.npz      (one 192-dim ECAPA voiceprint per cluster)

Off-limits siblings — DO NOT modify from here:
  - diarizer_phase9.Diarizer      (we read its output, never edit it)
  - speakers_phase9.SpeakerDB     (diarizer's own audio-only registry)
  - transcriber, summarizer, ring buffer, wifi reader
  - iris_gui.py, terminal.py, ESP32 firmware
  - iris_photos.py, iris_sessions.py
  - iris_people.py and iris_faces.py from earlier M5 stages
"""

from __future__ import annotations

import os
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import iris_people  # cheap; no heavy deps


# ── tuneables ────────────────────────────────────────────────────────────
MATCH_THRESHOLD = 0.60
REINFORCE_MIN   = MATCH_THRESHOLD
REINFORCE_MAX   = 0.95
TRUSTED_NAME_KINDS = ("strict",)

# Revamp: confidence band that triggers a user-confirmation prompt instead
# of silently accepting a match. Below 0.60 = enroll new; 0.60-0.70 =
# match but flag as uncertain so the UI can ask "is this Pranav?";
# >= 0.70 = silent confident match.
CONFIRM_BELOW       = 0.70
LOW_CONFIDENCE_MIN  = MATCH_THRESHOLD   # 0.60
LOW_CONFIDENCE_MAX  = CONFIRM_BELOW     # 0.70


# ── dataclasses ──────────────────────────────────────────────────────────
@dataclass
class ProcessedVoice:
    person_id: int
    name: str
    similarity: float
    cluster_id: int
    cluster_size: int = 0
    diarizer_name: str = ""
    diarizer_kind: str = "unknown"
    was_new_enrollment: bool = False
    was_reinforced: bool = False
    needs_confirmation: bool = False   # revamp: 0.60-0.70 match band
    wav_path: str = ""


@dataclass
class IngestionResult:
    wav_path: str
    clusters_total: int = 0
    clusters_matched: int = 0
    clusters_enrolled: int = 0
    skipped: bool = False
    error: str = ""
    processed_voices: list[ProcessedVoice] = field(default_factory=list)
    # Revamp: per-recording stats for significance evaluation in fusion.
    duration_seconds: float = 0.0
    speaker_count: int = 0
    dominant_share: float = 0.0       # 0..1, how dominant the top speaker was
    dominant_person_id: int = 0       # the dominant speaker's people row
    mentioned_names: list[str] = field(default_factory=list)  # transcript proper nouns not in registry


# ── pipeline ─────────────────────────────────────────────────────────────
class VoicePipeline:
    """Reads diarizer sidecars and routes voiceprints into PeopleStore."""

    def __init__(self):
        self._lock = threading.RLock()
        self._unknown_counter_lock = threading.Lock()

    # ── headline: process one recording ──────────────────────────────────
    def process_recording(self, wav_path: str,
                          store: iris_people.PeopleStore,
                          force: bool = False) -> IngestionResult:
        result = IngestionResult(wav_path=wav_path)

        if not wav_path or not os.path.exists(wav_path):
            result.error = "wav not found"
            return result

        if not force and self.is_ingested(wav_path):
            result.skipped = True
            return result

        npz_path  = os.path.splitext(wav_path)[0] + ".embeddings.npz"
        json_path = os.path.splitext(wav_path)[0] + ".json"

        if not os.path.exists(npz_path):
            result.error = "embeddings npz missing (diarizer not done?)"
            return result
        if not os.path.exists(json_path):
            result.error = "transcript json missing"
            return result

        try:
            cluster_embs = self._load_npz(npz_path)
        except Exception as e:
            result.error = f"npz load failed: {e}"
            return result
        if not cluster_embs:
            result.error = "npz contained no clusters"
            return result

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        except Exception as e:
            result.error = f"json load failed: {e}"
            return result

        cluster_meta = self._cluster_meta_from_transcript(transcript)

        with self._lock:
            for cluster_id, emb in cluster_embs.items():
                meta = cluster_meta.get(cluster_id,
                                        {"name": f"cluster_{cluster_id}",
                                         "kind": "unknown",
                                         "n_segments": 0})
                pv = self._process_one_cluster(
                    cluster_id=cluster_id,
                    embedding=emb,
                    diarizer_name=meta["name"],
                    diarizer_kind=meta["kind"],
                    n_segments=meta["n_segments"],
                    wav_path=wav_path,
                    store=store,
                )
                if pv is None:
                    continue
                result.processed_voices.append(pv)
                result.clusters_total += 1
                if pv.was_new_enrollment:
                    result.clusters_enrolled += 1
                else:
                    result.clusters_matched += 1

            # Revamp: enrich result with conversation stats and mentioned
            # names. All defensive — failures fall through with sensible
            # defaults so the core ingestion path is unaffected.
            try:
                self._enrich_result_with_stats(result, transcript)
            except Exception as e:
                print(f"[voices] stats enrichment failed: {e}")
            try:
                result.mentioned_names = self._scan_mentioned_names(
                    transcript, store)
            except Exception as e:
                print(f"[voices] mentioned-name scan failed: {e}")
                result.mentioned_names = []

            self._write_marker(wav_path, result)

        print(f"[voices] {os.path.basename(wav_path)}: "
              f"{result.clusters_total} cluster"
              f"{'s' if result.clusters_total != 1 else ''}, "
              f"{result.clusters_matched} matched, "
              f"{result.clusters_enrolled} enrolled")
        return result

    # ── batch helper ─────────────────────────────────────────────────────
    def scan_directory(self, directory: str,
                       store: iris_people.PeopleStore,
                       force: bool = False) -> list[IngestionResult]:
        results: list[IngestionResult] = []
        if not directory or not os.path.isdir(directory):
            return results
        for root, _dirs, files in os.walk(directory):
            for fn in files:
                if not fn.lower().endswith(".wav"):
                    continue
                wav = os.path.join(root, fn)
                stem = os.path.splitext(wav)[0]
                if not (os.path.exists(stem + ".embeddings.npz")
                        and os.path.exists(stem + ".json")):
                    continue
                results.append(self.process_recording(wav, store, force=force))
        return results

    # ── ingestion marker ─────────────────────────────────────────────────
    @staticmethod
    def _marker_path(wav_path: str) -> str:
        return os.path.splitext(wav_path)[0] + ".voice_ingested.json"

    def is_ingested(self, wav_path: str) -> bool:
        return os.path.exists(self._marker_path(wav_path))

    def _write_marker(self, wav_path: str, result: IngestionResult) -> None:
        try:
            with open(self._marker_path(wav_path), "w",
                      encoding="utf-8") as f:
                json.dump({
                    "ingested_at": time.time(),
                    "wav_path": wav_path,
                    "clusters_total": result.clusters_total,
                    "clusters_matched": result.clusters_matched,
                    "clusters_enrolled": result.clusters_enrolled,
                    "people": [
                        {"person_id": pv.person_id,
                         "name": pv.name,
                         "similarity": round(pv.similarity, 3),
                         "cluster_id": pv.cluster_id,
                         "diarizer_kind": pv.diarizer_kind,
                         "new": pv.was_new_enrollment,
                         "reinforced": pv.was_reinforced}
                        for pv in result.processed_voices
                    ],
                }, f, indent=2)
        except Exception as e:
            print(f"[voices] could not write ingestion marker: {e}")

    # ── per-cluster routing ───────────────────────────────────────────────
    # Revamp: fixes the duplicate-Humza bug. When the diarizer gives us a
    # strict name like "Humza" and there's no embedding match above 0.60,
    # we now (a) strip any " 2"/" 3" suffix to recover the base name and
    # (b) look for an existing row with that base name to merge into,
    # before falling back to disambiguation. Also adds the 0.60-0.70
    # confidence band as a "needs_confirmation" flag instead of silently
    # accepting borderline matches.
    def _process_one_cluster(self, *, cluster_id: int,
                             embedding: np.ndarray,
                             diarizer_name: str,
                             diarizer_kind: str,
                             n_segments: int,
                             wav_path: str,
                             store: iris_people.PeopleStore
                             ) -> Optional[ProcessedVoice]:
        try:
            v = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if v.size == 0 or not np.all(np.isfinite(v)):
            return None
        if float(np.linalg.norm(v)) == 0.0:
            return None

        match = store.match_voice(v, threshold=MATCH_THRESHOLD)

        if match is not None:
            store.mark_seen(match.person.id)
            reinforced = False
            # Only reinforce above LOW_CONFIDENCE_MAX so we don't pollute
            # high-confidence persons with low-confidence embeddings that
            # may belong to someone else entirely.
            if REINFORCE_MIN <= match.similarity <= REINFORCE_MAX \
                    and match.similarity >= CONFIRM_BELOW:
                if store.add_embedding(match.person.id,
                                       iris_people.KIND_VOICE, v):
                    reinforced = True
            needs_confirm = (LOW_CONFIDENCE_MIN <= match.similarity
                             < LOW_CONFIDENCE_MAX)
            return ProcessedVoice(
                person_id=match.person.id,
                name=match.person.name,
                similarity=match.similarity,
                cluster_id=cluster_id,
                cluster_size=n_segments,
                diarizer_name=diarizer_name,
                diarizer_kind=diarizer_kind,
                was_new_enrollment=False,
                was_reinforced=reinforced,
                needs_confirmation=needs_confirm,
                wav_path=wav_path,
            )

        # No strict match. If the diarizer is confident about a name,
        # try harder before creating a new row.
        if diarizer_kind in TRUSTED_NAME_KINDS and diarizer_name \
                and not diarizer_name.startswith("Unknown") \
                and not diarizer_name.endswith("?"):

            # (1) Relaxed embedding match — sometimes the registry has the
            # person but cosine is in the 0.45-0.60 band.
            relaxed = store.match_voice(v, threshold=0.45)
            if relaxed is not None \
                    and not self._is_placeholder_name(relaxed.person.name):
                store.mark_seen(relaxed.person.id)
                store.add_embedding(relaxed.person.id,
                                    iris_people.KIND_VOICE, v)
                return ProcessedVoice(
                    person_id=relaxed.person.id,
                    name=relaxed.person.name,
                    similarity=relaxed.similarity,
                    cluster_id=cluster_id,
                    cluster_size=n_segments,
                    diarizer_name=diarizer_name,
                    diarizer_kind=diarizer_kind,
                    was_new_enrollment=False,
                    was_reinforced=True,
                    needs_confirmation=True,   # below 0.60 — surface for confirm
                    wav_path=wav_path,
                )

            # (2) Base-name consolidation — this is the duplicate-Humza fix.
            # "Humza 2" / "Humza 3" / etc. coming from the diarizer should
            # collapse into the existing "Humza" row instead of creating
            # yet another suffixed duplicate. We strip the suffix, look up
            # all rows whose name starts with the base, and pick the
            # earliest-created one to keep the canonical row stable.
            base_name = self._strip_numeric_suffix(diarizer_name)
            canonical = self._find_canonical_by_base_name(base_name, store)
            if canonical is not None:
                store.mark_seen(canonical.id)
                store.add_embedding(canonical.id,
                                    iris_people.KIND_VOICE, v)
                return ProcessedVoice(
                    person_id=canonical.id,
                    name=canonical.name,
                    similarity=0.5,
                    cluster_id=cluster_id,
                    cluster_size=n_segments,
                    diarizer_name=diarizer_name,
                    diarizer_kind=diarizer_kind,
                    was_new_enrollment=False,
                    was_reinforced=True,
                    needs_confirmation=True,
                    wav_path=wav_path,
                )

           # (3) Name-based cross-reference — the person may have been
            # enrolled by face (photo trigger) and therefore has face
            # embeddings but NO voice embeddings. match_voice() scores 0
            # for them. If the diarizer is confident about the name, look
            # the person up by name in the registry and merge into them
            # rather than creating a duplicate Unknown row.
            existing_by_name = store.get_by_name(base_name)
            if existing_by_name is None:
                # Also try case-insensitive scan.
                for p in store.list_all():
                    if p.name.lower() == base_name.lower():
                        existing_by_name = p
                        break

            if existing_by_name is not None:
                store.mark_seen(existing_by_name.id)
                store.add_embedding(existing_by_name.id,
                                    iris_people.KIND_VOICE, v)
                return ProcessedVoice(
                    person_id=existing_by_name.id,
                    name=existing_by_name.name,
                    similarity=0.65,   # name-confirmed, treat as mid-confidence
                    cluster_id=cluster_id,
                    cluster_size=n_segments,
                    diarizer_name=diarizer_name,
                    diarizer_kind=diarizer_kind,
                    was_new_enrollment=False,
                    was_reinforced=True,
                    needs_confirmation=False,
                    wav_path=wav_path,
                )

            # (4) No canonical row exists yet — enroll with the base name.
            new_name = base_name
            if store.get_by_name(new_name) is not None:
                new_name = self._disambiguate_name(new_name, store)
        else:
            new_name = self._next_unknown_name(store)

        person = store.add(new_name, voice_embedding=v)
        if person is None:
            return None
        store.mark_seen(person.id)
        return ProcessedVoice(
            person_id=person.id,
            name=person.name,
            similarity=0.0,
            cluster_id=cluster_id,
            cluster_size=n_segments,
            diarizer_name=diarizer_name,
            diarizer_kind=diarizer_kind,
            was_new_enrollment=True,
            was_reinforced=False,
            wav_path=wav_path,
        )

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _load_npz(npz_path: str) -> dict[int, np.ndarray]:
        data = np.load(npz_path, allow_pickle=False)
        out: dict[int, np.ndarray] = {}
        for key in data.files:
            if not key.startswith("cluster_"):
                continue
            try:
                cid = int(key.split("_", 1)[1])
            except Exception:
                continue
            arr = np.asarray(data[key], dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            out[cid] = arr
        return out

    @staticmethod
    def _cluster_meta_from_transcript(transcript: dict
                                      ) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for seg in transcript.get("segments", []) or []:
            try:
                cid = int(seg.get("_cluster", -1))
            except Exception:
                cid = -1
            if cid < 0:
                continue
            entry = out.setdefault(cid, {
                "name": str(seg.get("speaker", "") or ""),
                "kind": str(seg.get("speaker_kind", "unknown") or "unknown"),
                "n_segments": 0,
            })
            entry["n_segments"] += 1
            if seg.get("speaker_kind") == "strict":
                entry["kind"] = "strict"
                entry["name"] = str(seg.get("speaker", entry["name"]))
        return out

    def _next_unknown_name(self,
                           store: iris_people.PeopleStore) -> str:
        with self._unknown_counter_lock:
            existing = {p.name for p in store.list_all()}
            n = 1
            while f"Unknown {n}" in existing:
                n += 1
            return f"Unknown {n}"

    @staticmethod
    def _strip_numeric_suffix(name: str) -> str:
        """'Humza 2' -> 'Humza', 'Humza' -> 'Humza', 'Mr 5 X' -> 'Mr 5 X'.
        Only strips a trailing single integer with a single space."""
        n = (name or "").strip()
        if not n:
            return n
        import re as _re
        m = _re.match(r"^(.+?)\s+(\d+)$", n)
        if m:
            return m.group(1).strip()
        return n

    @staticmethod
    def _find_canonical_by_base_name(
            base_name: str,
            store: iris_people.PeopleStore
    ) -> Optional[iris_people.Person]:
        """Find the earliest-created row whose name equals base_name or
        starts with 'base_name <digit>'. Returns None if no such row.
        Used to consolidate the diarizer's suffixed labels.

        Conservative: only matches when names share the exact base. Won't
        merge 'Humza' with 'Humza Malik' (full name) because that could
        be a deliberate distinct identity."""
        if not base_name:
            return None
        candidates: list[iris_people.Person] = []
        for p in store.list_all():
            if p.name == base_name:
                candidates.append(p)
                continue
            # Match 'base_name N' but not 'base_name something_else'.
            if p.name.startswith(base_name + " "):
                tail = p.name[len(base_name) + 1:].strip()
                if tail.isdigit():
                    candidates.append(p)
        if not candidates:
            return None
        # Prefer the row with the bare base name; otherwise the oldest one.
        for p in candidates:
            if p.name == base_name:
                return p
        candidates.sort(key=lambda x: x.created_at or 0.0)
        return candidates[0]

    @staticmethod
    def _enrich_result_with_stats(result: "IngestionResult",
                                  transcript: dict) -> None:
        """Populate duration_seconds / speaker_count / dominant_share /
        dominant_person_id on the result, derived from the transcript +
        the processed voices we just wrote. Best-effort — any failure
        leaves the defaults in place."""
        segments = transcript.get("segments", []) or []
        max_end = 0.0
        per_cluster_seconds: dict[int, float] = {}
        for seg in segments:
            try:
                start = float(seg.get("start", 0.0) or 0.0)
                end   = float(seg.get("end", 0.0) or 0.0)
            except Exception:
                continue
            if end > max_end:
                max_end = end
            try:
                cid = int(seg.get("_cluster", -1))
            except Exception:
                cid = -1
            if cid < 0:
                continue
            per_cluster_seconds[cid] = (
                per_cluster_seconds.get(cid, 0.0) + max(0.0, end - start))
        # If the transcript carries an explicit total duration prefer it.
        try:
            total = float(transcript.get("duration", max_end) or max_end)
        except Exception:
            total = max_end
        result.duration_seconds = float(total)
        result.speaker_count = len({pv.person_id
                                    for pv in result.processed_voices})

        # Dominant share = top cluster's spoken seconds / sum-of-all.
        if per_cluster_seconds:
            total_speech = sum(per_cluster_seconds.values()) or 0.0
            top_cid, top_secs = max(per_cluster_seconds.items(),
                                    key=lambda kv: kv[1])
            if total_speech > 0:
                result.dominant_share = float(top_secs / total_speech)
            # Map top cluster back to its person_id.
            for pv in result.processed_voices:
                if pv.cluster_id == top_cid:
                    result.dominant_person_id = pv.person_id
                    break

    @staticmethod
    def _scan_mentioned_names(transcript: dict,
                              store: iris_people.PeopleStore
                              ) -> list[str]:
        """Look at the transcript text for proper-noun-looking tokens that
        don't match any existing person name. Returns up to ~5 unique
        candidate names so the UI can offer 'add this person?' prompts.

        Heuristic: capitalised single tokens that aren't sentence-initial,
        aren't common English words, and aren't already in the registry.
        Conservative on purpose — false positives mean noisy prompts."""
        text_chunks: list[str] = []
        for seg in transcript.get("segments", []) or []:
            t = seg.get("text") or ""
            if t:
                text_chunks.append(str(t))
        if not text_chunks:
            return []
        text = " ".join(text_chunks)
        # Quick known-name set (lowercased) — both real and "Unknown N".
        known = set()
        for p in store.list_all():
            n = (p.name or "").strip().lower()
            if n:
                known.add(n)
                for piece in n.split():
                    if len(piece) >= 3:
                        known.add(piece)
        # Common false-positive words to ignore.
        STOPWORDS = {
            "the","a","an","i","you","we","they","he","she","it","this",
            "that","there","here","what","why","how","when","where","who",
            "and","or","but","so","if","then","yes","no","ok","okay","yeah",
            "oh","wow","hey","hi","hello","thanks","thank","please","sure",
            "monday","tuesday","wednesday","thursday","friday","saturday",
            "sunday","january","february","march","april","may","june",
            "july","august","september","october","november","december",
            "google","apple","microsoft","amazon","meta","openai","claude",
            "iris","gpt","python","windows","mac","linux",
        }
        import re as _re
        # Find Capitalised tokens (2+ chars, alpha) that are NOT at the
        # start of a sentence — meaning the previous token isn't a period.
        # Simplified pattern: token after a space, before a non-letter.
        candidates: dict[str, int] = {}
        for m in _re.finditer(r"(?<=\s)([A-Z][a-z]{2,})", text):
            tok = m.group(1)
            lo = tok.lower()
            if lo in STOPWORDS or lo in known:
                continue
            candidates[tok] = candidates.get(tok, 0) + 1
        # Any single mention is a candidate — dedupe in pending_prompts
        # ensures no spam if the same name shows up across recordings.
        ranked = list(candidates.keys())
        # Two-word names ("John Smith") — pick capitalised pairs.
        for m in _re.finditer(
                r"(?<=\s)([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})", text):
            full = f"{m.group(1)} {m.group(2)}"
            if full.lower() in known:
                continue
            if full not in ranked:
                ranked.append(full)
        return ranked[:5]

    @staticmethod
    def _is_placeholder_name(name: str) -> bool:
        """Treat 'Unknown N' and empty names as placeholders. Mirrors
        PeopleFusion._is_placeholder_name — kept separate so each module
        stays self-contained without cross-importing iris_fusion."""
        n = (name or "").strip()
        if not n:
            return True
        return n.lower().startswith("unknown")

    @staticmethod
    def _disambiguate_name(name: str,
                           store: iris_people.PeopleStore) -> str:
        existing = {p.name for p in store.list_all()}
        if name not in existing:
            return name
        n = 2
        while f"{name} {n}" in existing:
            n += 1
        return f"{name} {n}"


# ── module-level singleton ───────────────────────────────────────────────
_pipeline_singleton: Optional[VoicePipeline] = None
_singleton_lock = threading.Lock()


def get_pipeline() -> VoicePipeline:
    global _pipeline_singleton
    with _singleton_lock:
        if _pipeline_singleton is None:
            _pipeline_singleton = VoicePipeline()
        return _pipeline_singleton