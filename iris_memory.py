"""
iris_memory.py — ChromaDB-backed conversational memory for IRIS.

Stores one searchable record per audio recording so the Chat tab can answer
recall questions like 'give me all conversations with Pranav', 'what did
we discuss about the project', 'who did I talk to yesterday'. The full
transcript is embedded with all-MiniLM-L6-v2 (384-dim) and indexed in
ChromaDB; metadata holds people, dates, summary, location, etc.

Design notes:
  * Lazy-load — ChromaDB + sentence-transformers eat ~500MB and take
    several seconds to initialize. We only pay for that on first access,
    not at import time.
  * Defensive — if ChromaDB isn't installed or fails to open, every
    method silently no-ops and returns empty results. The rest of IRIS
    keeps working.
  * Background-friendly — summary fill is async. The record is searchable
    by transcript and person immediately; the Llama summary patches in
    seconds later.
  * Singleton — get_memory() returns the same instance across the app.

Pure Python, no Qt. Stores to data/chroma/ by default.
"""

from __future__ import annotations

import os
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# ── tuneables ────────────────────────────────────────────────────────────
DEFAULT_COLLECTION = "iris_sessions"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, ~90MB on first download
MAX_TRANSCRIPT_BYTES = 200_000              # ChromaDB upper bound is generous
# Where embedded model weights cache to (HuggingFace default unless we
# override). Letting it default keeps things simple.


def default_chroma_dir() -> str:
    """data/chroma/ alongside data/sqlite/ — mirrors how iris_people
    chose its default db path. Survives across restarts."""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data", "chroma")


# ── public dataclass ─────────────────────────────────────────────────────
@dataclass
class MemoryRecord:
    """A single recall result. Mirrors what ChromaDB stores plus the
    transcript pulled back out as the 'document' field."""
    id: str
    session_start: float = 0.0          # unix timestamp
    duration_seconds: float = 0.0
    people_names: list = field(default_factory=list)
    people_ids: list = field(default_factory=list)
    dominant_person_id: int = 0
    dominant_share: float = 0.0
    transcript: str = ""
    summary: str = ""
    location: str = ""
    wav_path: str = ""
    clip_path: str = ""
    confirmed: bool = False
    distance: float = 0.0               # ChromaDB cosine distance (0 = exact)

    def when_str(self) -> str:
        from datetime import datetime
        if self.session_start <= 0:
            return "(unknown date)"
        return datetime.fromtimestamp(self.session_start)\
                       .strftime("%Y-%m-%d %H:%M")

    def duration_str(self) -> str:
        d = int(round(self.duration_seconds))
        if d < 60:
            return f"{d}s"
        return f"{d // 60}m {d % 60}s"


# ── core store ───────────────────────────────────────────────────────────
class MemoryStore:
    """ChromaDB wrapper. Methods never raise — failures are logged and
    return empty results."""

    def __init__(self, persist_dir: Optional[str] = None,
                 collection_name: str = DEFAULT_COLLECTION,
                 embed_model: str = DEFAULT_EMBED_MODEL):
        self.persist_dir = persist_dir or default_chroma_dir()
        self.collection_name = collection_name
        self.embed_model = embed_model
        self._client = None
        self._collection = None
        self._embed_fn = None
        self._lock = threading.RLock()
        self._init_attempted = False
        self._available = False
        # Ensure the directory exists so persistence works.
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
        except Exception as e:
            print(f"[memory] could not create persist dir: {e}")

    # ── lazy init ────────────────────────────────────────────────────────
    def _ensure_init(self, force: bool = False) -> bool:
        """Lazy bootstrap. Returns True if the store is usable, False if
        ChromaDB / sentence-transformers couldn't be loaded.

        Without `force`, a previously-failed init is NOT retried. Pass
        `force=True` to retry — useful from the Chat tab when the user
        explicitly asks for memory queries."""
        if self._available:
            return True
        if self._init_attempted and not force:
            return False
        with self._lock:
            if self._available:
                return True
            if self._init_attempted and not force:
                return False
            # Reset attempt flag so we can try again on force.
            self._init_attempted = True
            print(f"[memory] initialising (persist={self.persist_dir})...")
            try:
                import chromadb
            except ImportError as e:
                print(f"[memory] ERROR: chromadb not available: {e}")
                return False
            print(f"[memory] loading embedding model (ONNX, no PyTorch)...")
            try:
                from chromadb.utils.embedding_functions import \
                    DefaultEmbeddingFunction
                self._embed_fn = DefaultEmbeddingFunction()
            except Exception as e:
                print(f"[memory] ERROR: embedding model load failed: {e}")
                return False
            try:
                print(f"[memory] opening ChromaDB persistent client...")
                self._client = chromadb.PersistentClient(
                    path=self.persist_dir)
                # get_or_create matches our backwards-compat needs — old
                # DBs from earlier runs are picked up automatically.
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self._embed_fn,
                    metadata={"hnsw:space": "cosine"})
            except Exception as e:
                print(f"[memory] ERROR: chromadb open failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            self._available = True
            try:
                n = self._collection.count()
            except Exception:
                n = "?"
            print(f"[memory] ready ({n} stored records, "
                  f"persist={self.persist_dir})")
            return True

    def force_reinit(self) -> bool:
        """Explicitly retry initialisation after a previous failure.
        Resets internal state and runs _ensure_init again. Returns True
        if init succeeded, False otherwise."""
        with self._lock:
            self._init_attempted = False
            self._available = False
            self._client = None
            self._collection = None
        return self._ensure_init(force=True)

    def is_ready(self) -> bool:
        return self._available

    # ── write ────────────────────────────────────────────────────────────
    def store_segment(self, *,
                      seg_id: str,
                      session_start: float,
                      duration_seconds: float,
                      people_names: list,
                      people_ids: list,
                      dominant_person_id: int,
                      dominant_share: float,
                      transcript: str,
                      summary: str = "",
                      location: str = "",
                      wav_path: str = "",
                      clip_path: str = "",
                      confirmed: bool = False) -> bool:
        """Insert a new memory record. Idempotent — re-storing the same
        seg_id updates the existing record."""
        if not self._ensure_init():
            return False
        # ChromaDB only supports str/int/float/bool in metadata. List
        # fields go through json.dumps so we can rehydrate on read.
        try:
            transcript = transcript or ""
            if len(transcript.encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
                # Truncate at byte level to stay under any backend limit.
                transcript = transcript.encode("utf-8")[
                    :MAX_TRANSCRIPT_BYTES].decode("utf-8", errors="ignore")
            # ChromaDB requires the document text to be non-empty for the
            # embedder to produce a vector. Fall back to summary or a
            # one-line placeholder if transcript is empty.
            doc_text = transcript or summary or "(no transcript available)"
            meta = {
                "session_start": float(session_start),
                "duration_seconds": float(duration_seconds),
                "people_names": json.dumps(people_names or []),
                "people_ids": json.dumps([int(p) for p in (people_ids or [])]),
                "dominant_person_id": int(dominant_person_id or 0),
                "dominant_share": float(dominant_share or 0.0),
                "summary": str(summary or ""),
                "location": str(location or ""),
                "wav_path": str(wav_path or ""),
                "clip_path": str(clip_path or ""),
                "confirmed": bool(confirmed),
            }
            with self._lock:
                # upsert semantics: replace if same id exists.
                self._collection.upsert(
                    ids=[seg_id],
                    documents=[doc_text],
                    metadatas=[meta],
                )
            return True
        except Exception as e:
            print(f"[memory] store_segment failed for {seg_id!r}: {e}")
            return False

    def update_summary(self, seg_id: str, summary: str) -> bool:
        """Patch the Llama-generated summary onto an existing record."""
        if not self._ensure_init():
            return False
        try:
            with self._lock:
                self._collection.update(
                    ids=[seg_id],
                    metadatas=[{"summary": str(summary or "")}])
            return True
        except Exception as e:
            print(f"[memory] update_summary failed for {seg_id!r}: {e}")
            return False

    def update_clip(self, seg_id: str, clip_path: str,
                    confirmed: bool = True) -> bool:
        """Link a video clip to an existing record. Called from
        reconcile_clip() when audio+video are matched."""
        if not self._ensure_init():
            return False
        try:
            with self._lock:
                self._collection.update(
                    ids=[seg_id],
                    metadatas=[{"clip_path": str(clip_path or ""),
                                "confirmed": bool(confirmed)}])
            return True
        except Exception as e:
            print(f"[memory] update_clip failed for {seg_id!r}: {e}")
            return False

    def update_location(self, seg_id: str, location: str) -> bool:
        """Patch in location once M6 location detection lands."""
        if not self._ensure_init():
            return False
        try:
            with self._lock:
                self._collection.update(
                    ids=[seg_id],
                    metadatas=[{"location": str(location or "")}])
            return True
        except Exception as e:
            print(f"[memory] update_location failed: {e}")
            return False

    def delete_segment(self, seg_id: str) -> bool:
        if not self._ensure_init():
            return False
        try:
            with self._lock:
                self._collection.delete(ids=[seg_id])
            return True
        except Exception as e:
            print(f"[memory] delete_segment failed: {e}")
            return False

    # ── read ─────────────────────────────────────────────────────────────
    def get_by_id(self, seg_id: str) -> Optional[MemoryRecord]:
        if not self._ensure_init():
            return None
        try:
            with self._lock:
                got = self._collection.get(ids=[seg_id],
                                           include=["documents", "metadatas"])
            ids = got.get("ids") or []
            if not ids:
                return None
            return self._row_to_record(
                ids[0],
                (got.get("documents") or [""])[0],
                (got.get("metadatas") or [{}])[0],
                distance=0.0)
        except Exception as e:
            print(f"[memory] get_by_id failed: {e}")
            return None

    def search_by_person(self, person_name: str,
                         *, limit: int = 20) -> list[MemoryRecord]:
        """Return all records that include the named person, newest first.
        Case-insensitive substring match against the JSON-serialized
        people_names metadata field."""
        if not self._ensure_init() or not person_name:
            return []
        target = person_name.strip().lower()
        try:
            with self._lock:
                got = self._collection.get(
                    include=["documents", "metadatas"])
        except Exception as e:
            print(f"[memory] search_by_person failed: {e}")
            return []
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        out: list[MemoryRecord] = []
        for i, sid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            doc = docs[i] if i < len(docs) else ""
            try:
                names = json.loads(meta.get("people_names") or "[]")
            except Exception:
                names = []
            if not any(target in (n or "").lower() for n in names):
                continue
            out.append(self._row_to_record(sid, doc, meta, distance=0.0))
        # Newest first.
        out.sort(key=lambda r: r.session_start, reverse=True)
        return out[:limit]

    def search_semantic(self, query: str,
                        *, limit: int = 10,
                        where: Optional[dict] = None) -> list[MemoryRecord]:
        """Vector similarity search over transcript embeddings. Optional
        `where` is a ChromaDB metadata filter dict (e.g.
        {'session_start': {'$gte': start_ts}})."""
        if not self._ensure_init() or not query.strip():
            return []
        try:
            with self._lock:
                got = self._collection.query(
                    query_texts=[query],
                    n_results=max(1, min(limit, 50)),
                    where=where if where else None,
                    include=["documents", "metadatas", "distances"])
        except Exception as e:
            print(f"[memory] search_semantic failed: {e}")
            return []
        # query() returns lists-of-lists keyed by query (we sent only one).
        ids   = (got.get("ids") or [[]])[0]
        docs  = (got.get("documents") or [[]])[0]
        metas = (got.get("metadatas") or [[]])[0]
        dists = (got.get("distances") or [[]])[0]
        out: list[MemoryRecord] = []
        for i, sid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            doc = docs[i] if i < len(docs) else ""
            dist = float(dists[i]) if i < len(dists) else 0.0
            out.append(self._row_to_record(sid, doc, meta, distance=dist))
        return out

    def search_by_date(self, start_ts: float, end_ts: float,
                       *, limit: int = 50) -> list[MemoryRecord]:
        """All records whose session_start falls in [start_ts, end_ts]."""
        if not self._ensure_init():
            return []
        try:
            with self._lock:
                # ChromaDB's where syntax — combined comparison needs $and.
                got = self._collection.get(
                    where={
                        "$and": [
                            {"session_start": {"$gte": float(start_ts)}},
                            {"session_start": {"$lte": float(end_ts)}},
                        ]
                    },
                    include=["documents", "metadatas"])
        except Exception as e:
            print(f"[memory] search_by_date failed: {e}")
            return []
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        out: list[MemoryRecord] = []
        for i, sid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            doc = docs[i] if i < len(docs) else ""
            out.append(self._row_to_record(sid, doc, meta, distance=0.0))
        out.sort(key=lambda r: r.session_start, reverse=True)
        return out[:limit]

    def search_combined(self, *,
                        query: str = "",
                        person_name: str = "",
                        date_start: Optional[float] = None,
                        date_end: Optional[float] = None,
                        limit: int = 10) -> list[MemoryRecord]:
        """Smart router: applies whichever filters are non-empty. If a
        query is given, prefers semantic ranking. If only person/date are
        given, falls back to metadata filtering with date sort."""
        if not self._ensure_init():
            return []

        # Build a ChromaDB where-clause for date if provided.
        where_clauses: list[dict] = []
        if date_start is not None:
            where_clauses.append({"session_start": {"$gte": float(date_start)}})
        if date_end is not None:
            where_clauses.append({"session_start": {"$lte": float(date_end)}})
        where: Optional[dict] = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) >= 2:
            where = {"$and": where_clauses}

        # Path 1: semantic query (with optional date filter).
        if query.strip():
            candidates = self.search_semantic(query, limit=limit * 3,
                                              where=where)
            if person_name:
                target = person_name.strip().lower()
                candidates = [
                    r for r in candidates
                    if any(target in (n or "").lower() for n in r.people_names)
                ]
            return candidates[:limit]

        # Path 2: person filter (with optional date narrowing).
        if person_name:
            recs = self.search_by_person(person_name, limit=limit * 3)
            if date_start is not None:
                recs = [r for r in recs if r.session_start >= date_start]
            if date_end is not None:
                recs = [r for r in recs if r.session_start <= date_end]
            return recs[:limit]

        # Path 3: date only.
        if date_start is not None or date_end is not None:
            lo = date_start if date_start is not None else 0.0
            hi = date_end   if date_end   is not None else time.time() + 1
            return self.search_by_date(lo, hi, limit=limit)

        return []

    def stats(self) -> dict:
        info = {"available": self._available, "count": 0,
                "persist_dir": self.persist_dir}
        if not self._ensure_init():
            return info
        try:
            with self._lock:
                info["count"] = int(self._collection.count())
        except Exception:
            pass
        return info

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _row_to_record(seg_id: str, doc: str, meta: dict,
                       *, distance: float) -> MemoryRecord:
        try:
            people_names = json.loads(meta.get("people_names") or "[]")
        except Exception:
            people_names = []
        try:
            people_ids = json.loads(meta.get("people_ids") or "[]")
        except Exception:
            people_ids = []
        return MemoryRecord(
            id=str(seg_id),
            session_start=float(meta.get("session_start", 0) or 0),
            duration_seconds=float(meta.get("duration_seconds", 0) or 0),
            people_names=people_names,
            people_ids=[int(p) for p in people_ids if isinstance(p, (int, float))],
            dominant_person_id=int(meta.get("dominant_person_id", 0) or 0),
            dominant_share=float(meta.get("dominant_share", 0) or 0),
            transcript=doc or "",
            summary=str(meta.get("summary", "") or ""),
            location=str(meta.get("location", "") or ""),
            wav_path=str(meta.get("wav_path", "") or ""),
            clip_path=str(meta.get("clip_path", "") or ""),
            confirmed=bool(meta.get("confirmed", False)),
            distance=float(distance or 0.0),
        )


# ── module-level singleton ───────────────────────────────────────────────
_INSTANCE: Optional[MemoryStore] = None
_INSTANCE_LOCK = threading.Lock()


def get_memory(persist_dir: Optional[str] = None) -> MemoryStore:
    """Lazy-create-or-return the process-wide MemoryStore. iris_fusion
    calls this once at startup; the Chat tab calls it on every recall."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = MemoryStore(persist_dir=persist_dir)
    return _INSTANCE