"""
backfill_memory.py — one-shot import of existing recordings into ChromaDB.

Scans the recordings directory for WAVs that have transcript sidecars
(.json), reads speaker info from the .embeddings.npz markers and the
voice_ingested markers, then writes a ChromaDB record for each one.

Usage (from C:\\audio_stream_glass_version):
    python backfill_memory.py              # dry-run — shows what would happen
    python backfill_memory.py --apply      # backfill all, with Llama summaries
    python backfill_memory.py --apply --no-summary   # skip summaries (fast)
    python backfill_memory.py --apply --limit 5      # test on first 5

Idempotent: re-running just upserts existing records.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from typing import Optional


def _find_recordings_dir() -> Optional[str]:
    """Look for the recordings directory using the same priority order
    as iris_fusion does."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sys.path.insert(0, here)
        import config_phase9 as cfg  # type: ignore
        d = getattr(cfg, "RECORDINGS_DIR", None)
        if d and os.path.isdir(d):
            return d
    except Exception:
        pass
    for candidate in (
        os.path.join(here, "recordings"),
        r"C:\audio_stream_glass_version\recordings",
    ):
        if os.path.isdir(candidate):
            return candidate
    return None


def _read_transcript(wav_path: str) -> str:
    """Read transcript text from .json sidecar (same logic as fusion)."""
    try:
        stem, _ = os.path.splitext(wav_path)
        jpath = stem + ".json"
        if not os.path.exists(jpath):
            return ""
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""
    segs = data.get("segments") or []
    if segs:
        chunks = []
        for s in segs:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            speaker = (s.get("speaker") or "").strip()
            chunks.append(f"{speaker}: {text}" if speaker else text)
        return "\n".join(chunks)
    return str(data.get("text") or "")


def _read_duration(wav_path: str) -> float:
    """Read duration from .json sidecar. Tries duration_seconds first
    (our format), then duration, then derives from last segment end."""
    try:
        stem, _ = os.path.splitext(wav_path)
        jpath = stem + ".json"
        if not os.path.exists(jpath):
            return 0.0
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Our sidecar uses 'duration_seconds'.
        v = data.get("duration_seconds")
        if v:
            return float(v)
        # Fallback: generic 'duration' key.
        v = data.get("duration")
        if v:
            return float(v)
        # Fallback: last segment end timestamp.
        segs = data.get("segments") or []
        if segs:
            ends = [float(s.get("end", 0) or 0) for s in segs]
            if ends:
                return max(ends)
        return 0.0
    except Exception:
        return 0.0


def _read_speakers_from_json(wav_path: str, store) -> list[str]:
    """Extract unique speaker names from the segment-level 'speaker'
    field in the .json sidecar. Looks each name up in people.db so
    renames are reflected. Falls back to the raw diarizer label if
    the person isn't in the DB."""
    try:
        stem, _ = os.path.splitext(wav_path)
        jpath = stem + ".json"
        if not os.path.exists(jpath):
            return []
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        segs = data.get("segments") or []
        seen: dict[str, str] = {}   # diarizer_label -> resolved_name
        for s in segs:
            label = (s.get("speaker") or "").strip()
            if not label or label in seen:
                continue
            # Try to find a matching person in the DB by name.
            try:
                p = store.get_by_name(label)
                seen[label] = p.name if p else label
            except Exception:
                seen[label] = label
        return list(seen.values())
    except Exception:
        return []


def _read_voice_ingested_names(wav_path: str,
                               store) -> list[str]:
    """Read the .voice_ingested.json marker and pull out person names by
    looking up their current name in people.db (so renames are reflected)."""
    try:
        stem, _ = os.path.splitext(wav_path)
        marker = stem + ".voice_ingested.json"
        if not os.path.exists(marker):
            return []
        with open(marker, "r", encoding="utf-8") as f:
            data = json.load(f)
        person_ids = []
        # Marker stores either 'person_ids' list or 'processed_voices' list.
        if "person_ids" in data:
            person_ids = [int(p) for p in (data["person_ids"] or [])]
        elif "processed_voices" in data:
            for pv in (data["processed_voices"] or []):
                pid = pv.get("person_id")
                if pid is not None:
                    person_ids.append(int(pid))
        if not person_ids:
            return []
        names = []
        seen = set()
        for pid in person_ids:
            if pid in seen:
                continue
            seen.add(pid)
            try:
                p = store.get(pid)
                if p is not None:
                    names.append(p.name)
            except Exception:
                pass
        return names
    except Exception:
        return []


def _read_diarizer_speaker_count(wav_path: str) -> int:
    """Count clusters in the .embeddings.npz sidecar."""
    try:
        import numpy as np
        stem, _ = os.path.splitext(wav_path)
        npz = stem + ".embeddings.npz"
        if not os.path.exists(npz):
            return 0
        with np.load(npz) as data:
            return len(data.files)
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Actually write to ChromaDB (default: dry-run).")
    ap.add_argument("--no-summary", action="store_true",
                    help="Skip Llama summary generation (much faster).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N WAVs (0 = all).")
    ap.add_argument("--recordings-dir", default=None,
                    help="Override the recordings directory path.")
    ap.add_argument("--db", default=None,
                    help="Override path to people.db.")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    try:
        import iris_people
        import iris_fusion
        import iris_memory
    except ImportError as e:
        print(f"ERROR: could not import iris modules: {e}")
        return 1

    # ── recordings directory ─────────────────────────────────────────────
    rec_dir = args.recordings_dir or _find_recordings_dir()
    if not rec_dir or not os.path.isdir(rec_dir):
        print("ERROR: recordings directory not found. "
              "Pass --recordings-dir PATH to specify it.")
        return 1
    print(f"Recordings dir: {rec_dir}")

    # ── people.db ───────────────────────────────────────────────────────
    db_path = args.db or iris_people.default_db_path()
    if not os.path.exists(db_path):
        print(f"ERROR: people.db not found at {db_path}")
        return 1
    print(f"People DB:      {db_path}")
    store = iris_people.PeopleStore(db_path)

    # ── find all WAVs that have a .json transcript sidecar ───────────────
    all_wavs = []
    for fn in sorted(os.listdir(rec_dir)):
        if not fn.lower().endswith(".wav"):
            continue
        wav = os.path.join(rec_dir, fn)
        stem = os.path.splitext(wav)[0]
        if os.path.exists(stem + ".json"):
            all_wavs.append(wav)

    if not all_wavs:
        print(f"\nNo WAVs with transcript sidecars found in {rec_dir}.")
        print("Make sure you have run at least one recording session so "
              "Whisper has transcribed the audio.")
        return 0

    print(f"\nFound {len(all_wavs)} WAV(s) with transcripts\n")
    if args.limit > 0:
        all_wavs = all_wavs[:args.limit]
        print(f"Limit: processing first {len(all_wavs)}\n")

    # ── open memory + Llama (only if applying) ───────────────────────────
    memory = None
    llama  = None
    if args.apply:
        memory = iris_memory.get_memory()
        if not memory._ensure_init():
            print("ERROR: ChromaDB failed to initialise. Make sure "
                  "'chromadb' and 'sentence-transformers' are installed.")
            return 1
        print(f"ChromaDB ready ({memory.stats().get('count', 0)} existing records)\n")
        if not args.no_summary:
            try:
                url, model = iris_fusion._read_llama_cfg()
                llama = iris_fusion._LlamaSummaryGenerator(url, model)
                print(f"Llama summary ready (model={model})\n")
            except Exception as e:
                print(f"WARN: Llama summary unavailable: {e}\n")
                llama = None

    # ── process ──────────────────────────────────────────────────────────
    stored = skipped = failed = 0

    for i, wav_path in enumerate(all_wavs, 1):
        fn = os.path.basename(wav_path)
        print(f"[{i}/{len(all_wavs)}] {fn}")

        transcript = _read_transcript(wav_path)
        if not transcript.strip():
            print("  SKIP: empty transcript")
            skipped += 1
            continue

        # Derive session_start from file mtime (stable, readable).
        try:
            session_start = float(os.path.getmtime(wav_path))
        except Exception:
            session_start = float(time.time())

        duration = _read_duration(wav_path)
        people_names = (_read_voice_ingested_names(wav_path, store)
                        or _read_speakers_from_json(wav_path, store))
        speaker_count = (_read_diarizer_speaker_count(wav_path)
                         or len(people_names))

        when = datetime.fromtimestamp(session_start).strftime(
            "%Y-%m-%d %H:%M")
        dur_str = (f"{int(duration // 60)}m {int(duration % 60)}s"
                   if duration >= 60 else f"{int(duration)}s")
        print(f"  {when}  ·  {dur_str}  ·  "
              f"{len(people_names)} speaker(s)"
              f"{': ' + ', '.join(people_names) if people_names else ''}")
        print(f"  transcript: {len(transcript)} chars")

        if not args.apply:
            stored += 1
            continue

        # Llama summary (synchronous in backfill mode).
        summary = ""
        if llama is not None:
            t0 = time.time()
            try:
                summary = llama.summarize(transcript,
                                          people_names=people_names)
            except Exception as e:
                print(f"  summary failed: {e}")
            elapsed = time.time() - t0
            if summary:
                preview = summary[:100] + ("…" if len(summary) > 100 else "")
                print(f"  summary ({elapsed:.1f}s): {preview}")

        seg_id = f"seg_{os.path.splitext(fn)[0]}"
        ok = memory.store_segment(
            seg_id=seg_id,
            session_start=session_start,
            duration_seconds=float(duration),
            people_names=people_names,
            people_ids=[],          # ids not critical — names are the key
            dominant_person_id=0,
            dominant_share=0.0,
            transcript=transcript,
            summary=summary,
            location="",
            wav_path=wav_path,
            clip_path="",
            confirmed=False,
        )
        if ok:
            stored += 1
            print(f"  STORED  {seg_id}")
        else:
            failed += 1
            print(f"  FAILED  {seg_id}")

    # ── summary ──────────────────────────────────────────────────────────
    print()
    if args.apply:
        print(f"Done: {stored} stored, {skipped} skipped, {failed} failed.")
        try:
            n = memory.stats().get("count", 0)
            print(f"ChromaDB now contains {n} total memory record(s).")
        except Exception:
            pass
        print()
        print("Launch IRIS and try in the Chat tab:")
        print("  give me all my recent conversations")
        print("  what did we discuss about <topic>")
        print("  who did I talk to this week")
    else:
        print(f"DRY-RUN: would store {stored}, skip {skipped}.")
        print("Re-run with --apply to actually write to ChromaDB.")
        if not args.no_summary:
            print("Tip: add --no-summary to skip Llama and run faster.")

    store.close()
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())