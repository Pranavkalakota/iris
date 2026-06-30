"""
iris_people.py — people registry persistence for the IRIS People tab (M5).

Pure Python, no Qt. Owns the SQLite database that ties together a person's
identity (name, role note, first/last seen, times seen) with their face and
voice embeddings. This is the data layer ONLY — capturing embeddings from
the camera (DeepFace ArcFace, 512-dim) and from the audio pipeline
(SpeechBrain ECAPA-TDNN, 192-dim) is the caller's job.

Gap 1 changes vs original:
  - Person dataclass gains folder_path str field.
  - people table gains folder_path TEXT NOT NULL DEFAULT ''.
  - _migrate() patches existing DBs forward-compatibly (no destructive ALTER).
  - set_folder_path() lets iris_fusion stamp the on-disk folder at enroll time.
  - _row_to_person() safely reads folder_path with IndexError guard for
    pre-migration rows.

Off-limits siblings — DO NOT modify from here:
  - speakers_phase9.SpeakerDB  : audio-only registry used by the diarizer
  - diarizer_phase9            : already produces 192-dim ECAPA voiceprints
  - any audio / chat / stream / photos code path
"""

from __future__ import annotations

import os
import time
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


# ── constants ────────────────────────────────────────────────────────────
KIND_FACE  = "face"
KIND_VOICE = "voice"
_VALID_KINDS = (KIND_FACE, KIND_VOICE)

DEFAULT_MAX_EMBEDDINGS_PER_KIND = 10
DEFAULT_MATCH_THRESHOLD         = 0.60   # cosine similarity


# ── dataclasses ──────────────────────────────────────────────────────────
@dataclass
class Person:
    id: int
    name: str
    role_note: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    times_seen: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    folder_path: str = ""          # absolute path to this person's folder on disk
    # Revamp fields — full profile detail per Pranav's spec.
    title: str = ""                # job title / role (e.g. "CTO")
    company: str = ""              # company / org (e.g. "Acme Inc.")
    relationship: str = ""         # relationship to user (e.g. "colleague", "friend")
    is_self: bool = False          # marks the user's own profile (one row max)
    # Counts of stored embeddings, populated when fetched via the Store.
    face_count: int = 0
    voice_count: int = 0

    def when_first(self) -> str:
        return _fmt_ts(self.first_seen)

    def when_last(self) -> str:
        return _fmt_ts(self.last_seen)


@dataclass
class PendingPrompt:
    """A queued UI prompt that the People tab should surface. The
    pending_prompts table is the durable store: once a prompt is queued
    here, it survives restarts and the system never queues a duplicate
    of the same (type, person_id) within the dedupe window."""
    id: int
    type: str                      # 'add_person' | 'confirm_identity' | 'name_mentioned' | 'long_unknown'
    person_id: int                 # 0 if not tied to an existing row (e.g. spoken-name prompts)
    payload: dict = field(default_factory=dict)
    created_at: float = 0.0
    shown_at: float = 0.0
    dismissed: int = 0             # 0=open, 1=dismissed, 2=acted-on


@dataclass
class MatchResult:
    person: Person
    similarity: float
    kind: str            # 'face' | 'voice' | 'combined'


# ── store ────────────────────────────────────────────────────────────────
class PeopleStore:
    """Owns one SQLite database for the People registry. Never raises out
    to the caller — any failure prints to stderr and returns a safe value
    (None / empty list / False)."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS people (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        role_note   TEXT    NOT NULL DEFAULT '',
        first_seen  REAL    NOT NULL,
        last_seen   REAL    NOT NULL,
        times_seen  INTEGER NOT NULL DEFAULT 0,
        created_at  REAL    NOT NULL,
        updated_at  REAL    NOT NULL,
        folder_path TEXT    NOT NULL DEFAULT '',
        title       TEXT    NOT NULL DEFAULT '',
        company     TEXT    NOT NULL DEFAULT '',
        relationship TEXT   NOT NULL DEFAULT '',
        is_self     INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id   INTEGER NOT NULL,
        kind        TEXT    NOT NULL,
        vector      BLOB    NOT NULL,
        dim         INTEGER NOT NULL,
        captured_at REAL    NOT NULL,
        FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS conversations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id       INTEGER NOT NULL,
        session_start   REAL    NOT NULL,
        audio_received_at REAL,
        video_received_at REAL,
        wav_path        TEXT    NOT NULL DEFAULT '',
        clip_path       TEXT    NOT NULL DEFAULT '',
        confirmed       INTEGER NOT NULL DEFAULT 0,
        duration_seconds REAL   NOT NULL DEFAULT 0,
        speaker_count   INTEGER NOT NULL DEFAULT 0,
        dominant_share  REAL    NOT NULL DEFAULT 0,
        created_at      REAL    NOT NULL,
        FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS pending_prompts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        type        TEXT    NOT NULL,
        person_id   INTEGER NOT NULL DEFAULT 0,
        payload     TEXT    NOT NULL DEFAULT '{}',
        created_at  REAL    NOT NULL,
        shown_at    REAL    NOT NULL DEFAULT 0,
        dismissed   INTEGER NOT NULL DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_people_name          ON people(name);
    CREATE INDEX IF NOT EXISTS idx_people_last_seen     ON people(last_seen);
    CREATE INDEX IF NOT EXISTS idx_emb_person           ON embeddings(person_id);
    CREATE INDEX IF NOT EXISTS idx_emb_kind             ON embeddings(kind);
    CREATE INDEX IF NOT EXISTS idx_emb_person_kind      ON embeddings(person_id, kind);
    CREATE INDEX IF NOT EXISTS idx_conv_person          ON conversations(person_id);
    CREATE INDEX IF NOT EXISTS idx_conv_confirmed       ON conversations(confirmed);
    CREATE INDEX IF NOT EXISTS idx_conv_session_start   ON conversations(session_start);
    CREATE INDEX IF NOT EXISTS idx_prompts_type         ON pending_prompts(type);
    CREATE INDEX IF NOT EXISTS idx_prompts_dismissed    ON pending_prompts(dismissed);
    CREATE INDEX IF NOT EXISTS idx_prompts_person       ON pending_prompts(person_id);
    """

    def __init__(self, db_path: str,
                 max_embeddings_per_kind: int = DEFAULT_MAX_EMBEDDINGS_PER_KIND):
        self.db_path = db_path
        self.max_embeddings_per_kind = int(max_embeddings_per_kind)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._open()

    # ── connection lifecycle ────────────────────────────────────────────
    def _open(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        except Exception:
            pass
        try:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,           # autocommit
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.executescript(self._SCHEMA)
            self._migrate()
        except Exception as e:
            print(f"[people] could not open db {self.db_path}: {e}")
            self._conn = None

    def _migrate(self) -> None:
        """Forward-compatible schema migrations. Existing databases created
        before folder_path, conversations, or the revamp columns were added
        need columns/tables patched in. Idempotent — running this on a fresh
        DB is a no-op."""
        if self._conn is None:
            return
        try:
            # ── people table: add new columns if missing ────────────────
            cols = {r["name"] for r in self._conn.execute(
                "PRAGMA table_info(people)").fetchall()}
            for col_name, col_ddl in (
                ("folder_path",  "TEXT NOT NULL DEFAULT ''"),
                ("title",        "TEXT NOT NULL DEFAULT ''"),
                ("company",      "TEXT NOT NULL DEFAULT ''"),
                ("relationship", "TEXT NOT NULL DEFAULT ''"),
                ("is_self",      "INTEGER NOT NULL DEFAULT 0"),
            ):
                if col_name not in cols:
                    self._conn.execute(
                        f"ALTER TABLE people ADD COLUMN {col_name} {col_ddl}")
                    print(f"[people] migrated: added {col_name} column")

            # is_self index — must be created AFTER the column exists.
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_people_is_self "
                "ON people(is_self)")

            # ── conversations table (idempotent CREATE for old DBs) ────
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id         INTEGER NOT NULL,
                    session_start     REAL    NOT NULL,
                    audio_received_at REAL,
                    video_received_at REAL,
                    wav_path          TEXT    NOT NULL DEFAULT '',
                    clip_path         TEXT    NOT NULL DEFAULT '',
                    confirmed         INTEGER NOT NULL DEFAULT 0,
                    duration_seconds  REAL    NOT NULL DEFAULT 0,
                    speaker_count     INTEGER NOT NULL DEFAULT 0,
                    dominant_share    REAL    NOT NULL DEFAULT 0,
                    created_at        REAL    NOT NULL,
                    FOREIGN KEY (person_id)
                        REFERENCES people(id) ON DELETE CASCADE
                )
            """)
            # Patch new conversation columns into pre-revamp DBs.
            conv_cols = {r["name"] for r in self._conn.execute(
                "PRAGMA table_info(conversations)").fetchall()}
            for col_name, col_ddl in (
                ("duration_seconds", "REAL NOT NULL DEFAULT 0"),
                ("speaker_count",    "INTEGER NOT NULL DEFAULT 0"),
                ("dominant_share",   "REAL NOT NULL DEFAULT 0"),
            ):
                if col_name not in conv_cols:
                    self._conn.execute(
                        f"ALTER TABLE conversations ADD COLUMN "
                        f"{col_name} {col_ddl}")
                    print(f"[people] migrated: added conversations.{col_name}")

            # ── pending_prompts table (revamp) ──────────────────────────
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_prompts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    type        TEXT    NOT NULL,
                    person_id   INTEGER NOT NULL DEFAULT 0,
                    payload     TEXT    NOT NULL DEFAULT '{}',
                    created_at  REAL    NOT NULL,
                    shown_at    REAL    NOT NULL DEFAULT 0,
                    dismissed   INTEGER NOT NULL DEFAULT 0
                )
            """)

            for ddl in (
                "CREATE INDEX IF NOT EXISTS idx_conv_person "
                "    ON conversations(person_id)",
                "CREATE INDEX IF NOT EXISTS idx_conv_confirmed "
                "    ON conversations(confirmed)",
                "CREATE INDEX IF NOT EXISTS idx_conv_session_start "
                "    ON conversations(session_start)",
                "CREATE INDEX IF NOT EXISTS idx_people_is_self "
                "    ON people(is_self)",
                "CREATE INDEX IF NOT EXISTS idx_prompts_type "
                "    ON pending_prompts(type)",
                "CREATE INDEX IF NOT EXISTS idx_prompts_dismissed "
                "    ON pending_prompts(dismissed)",
                "CREATE INDEX IF NOT EXISTS idx_prompts_person "
                "    ON pending_prompts(person_id)",
            ):
                self._conn.execute(ddl)

        except Exception as e:
            print(f"[people] migration failed (non-fatal): {e}")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    # ── people: create / read / update / delete ─────────────────────────
    def add(self, name: str, role_note: str = "",
            face_embedding: Optional[np.ndarray] = None,
            voice_embedding: Optional[np.ndarray] = None,
            *,
            title: str = "",
            company: str = "",
            relationship: str = "",
            is_self: bool = False) -> Optional[Person]:
        """Create a new person row. Optional first embeddings are stored
        too. Returns the new Person, or None on failure."""
        if self._conn is None:
            return None
        name = (name or "").strip() or "Unknown"
        role_note = (role_note or "").strip()
        title = (title or "").strip()
        company = (company or "").strip()
        relationship = (relationship or "").strip()
        now = time.time()
        with self._lock:
            try:
                # If is_self requested, demote any existing self row first.
                if is_self:
                    self._conn.execute(
                        "UPDATE people SET is_self = 0 WHERE is_self = 1")
                cur = self._conn.execute(
                    "INSERT INTO people "
                    "(name, role_note, first_seen, last_seen, times_seen, "
                    " created_at, updated_at, folder_path, "
                    " title, company, relationship, is_self) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (name, role_note, now, now, 0, now, now, "",
                     title, company, relationship, 1 if is_self else 0),
                )
                person_id = int(cur.lastrowid)
            except Exception as e:
                print(f"[people] add failed: {e}")
                return None
        if face_embedding is not None:
            self.add_embedding(person_id, KIND_FACE, face_embedding)
        if voice_embedding is not None:
            self.add_embedding(person_id, KIND_VOICE, voice_embedding)
        return self.get(person_id)

    def get(self, person_id: int) -> Optional[Person]:
        if self._conn is None:
            return None
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT * FROM people WHERE id = ?", (person_id,)
                ).fetchone()
            except Exception as e:
                print(f"[people] get failed: {e}")
                return None
        return self._row_to_person(row)

    def get_by_name(self, name: str) -> Optional[Person]:
        if self._conn is None or not name:
            return None
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT * FROM people WHERE name = ? "
                    "ORDER BY last_seen DESC LIMIT 1",
                    (name,),
                ).fetchone()
            except Exception as e:
                print(f"[people] get_by_name failed: {e}")
                return None
        return self._row_to_person(row)

    def list_all(self) -> list[Person]:
        """All people, newest-seen first."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT * FROM people ORDER BY last_seen DESC"
                ).fetchall()
            except Exception as e:
                print(f"[people] list_all failed: {e}")
                return []
        out: list[Person] = []
        for r in rows:
            p = self._row_to_person(r)
            if p is not None:
                out.append(p)
        return out

    def rename(self, person_id: int, new_name: str) -> bool:
        if self._conn is None:
            return False
        new_name = (new_name or "").strip()
        if not new_name:
            return False
        now = time.time()
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE people SET name = ?, updated_at = ? WHERE id = ?",
                    (new_name, now, person_id),
                )
                return True
            except Exception as e:
                print(f"[people] rename failed: {e}")
                return False

    def update_role_note(self, person_id: int, role_note: str) -> bool:
        if self._conn is None:
            return False
        role_note = (role_note or "").strip()
        now = time.time()
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE people SET role_note = ?, updated_at = ? "
                    "WHERE id = ?",
                    (role_note, now, person_id),
                )
                return True
            except Exception as e:
                print(f"[people] update_role_note failed: {e}")
                return False

    def set_folder_path(self, person_id: int, folder_path: str) -> bool:
        """Record the person's on-disk folder. Called once at enrollment
        time by iris_fusion._create_person_folder(). Idempotent — setting
        the same path twice is a no-op other than the updated_at bump."""
        if self._conn is None:
            return False
        folder_path = (folder_path or "").strip()
        now = time.time()
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE people SET folder_path = ?, updated_at = ? "
                    "WHERE id = ?",
                    (folder_path, now, person_id),
                )
                return True
            except Exception as e:
                print(f"[people] set_folder_path failed: {e}")
                return False

    def mark_seen(self, person_id: int) -> bool:
        """Bump times_seen and last_seen. Called after a successful match."""
        if self._conn is None:
            return False
        now = time.time()
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE people "
                    "SET times_seen = times_seen + 1, "
                    "    last_seen  = ?, "
                    "    updated_at = ? "
                    "WHERE id = ?",
                    (now, now, person_id),
                )
                return True
            except Exception as e:
                print(f"[people] mark_seen failed: {e}")
                return False

    def delete(self, person_id: int) -> bool:
        """Delete a person and (via ON DELETE CASCADE) all their
        embeddings and conversations."""
        if self._conn is None:
            return False
        with self._lock:
            try:
                self._conn.execute(
                    "DELETE FROM people WHERE id = ?", (person_id,))
                # Defensive manual sweep for older SQLite builds.
                self._conn.execute(
                    "DELETE FROM embeddings WHERE person_id = ?",
                    (person_id,))
                self._conn.execute(
                    "DELETE FROM conversations WHERE person_id = ?",
                    (person_id,))
                return True
            except Exception as e:
                print(f"[people] delete failed: {e}")
                return False

    # ── embeddings: add / list / evict ──────────────────────────────────
    def add_embedding(self, person_id: int, kind: str,
                      vector: np.ndarray) -> bool:
        if self._conn is None:
            return False
        if kind not in _VALID_KINDS:
            print(f"[people] add_embedding: bad kind {kind!r}")
            return False
        vec = _to_unit_float32(vector)
        if vec is None:
            return False
        dim = int(vec.shape[0])
        blob = vec.tobytes()
        now = time.time()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO embeddings "
                    "(person_id, kind, vector, dim, captured_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (person_id, kind, blob, dim, now),
                )
                self._enforce_cap(person_id, kind)
                self._conn.execute(
                    "UPDATE people SET updated_at = ? WHERE id = ?",
                    (now, person_id),
                )
                return True
            except Exception as e:
                print(f"[people] add_embedding failed: {e}")
                return False

    def list_embeddings(self, person_id: int,
                        kind: Optional[str] = None) -> list[np.ndarray]:
        if self._conn is None:
            return []
        with self._lock:
            try:
                if kind is None:
                    rows = self._conn.execute(
                        "SELECT vector, dim FROM embeddings "
                        "WHERE person_id = ? "
                        "ORDER BY captured_at DESC",
                        (person_id,),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT vector, dim FROM embeddings "
                        "WHERE person_id = ? AND kind = ? "
                        "ORDER BY captured_at DESC",
                        (person_id, kind),
                    ).fetchall()
            except Exception as e:
                print(f"[people] list_embeddings failed: {e}")
                return []
        out: list[np.ndarray] = []
        for r in rows:
            v = _blob_to_unit_float32(r["vector"], int(r["dim"]))
            if v is not None:
                out.append(v)
        return out

    def _enforce_cap(self, person_id: int, kind: str) -> None:
        cap = self.max_embeddings_per_kind
        if cap <= 0:
            return
        try:
            n = self._conn.execute(
                "SELECT COUNT(*) AS n FROM embeddings "
                "WHERE person_id = ? AND kind = ?",
                (person_id, kind),
            ).fetchone()["n"]
            if n <= cap:
                return
            self._conn.execute(
                "DELETE FROM embeddings WHERE id IN ("
                "  SELECT id FROM embeddings "
                "  WHERE person_id = ? AND kind = ? "
                "  ORDER BY captured_at ASC "
                "  LIMIT ?"
                ")",
                (person_id, kind, int(n - cap)),
            )
        except Exception as e:
            print(f"[people] _enforce_cap failed: {e}")

    # ── conversations ────────────────────────────────────────────────────
    def add_conversation(self, person_id: int, session_start: float,
                         wav_path: str = "",
                         audio_received_at: Optional[float] = None
                         ) -> Optional[int]:
        """Write a provisional conversation row (confirmed=False). Returns
        the new row id, or None on failure. Called by iris_fusion at audio-
        ingestion time — video confirmation comes later via confirm_conversation."""
        if self._conn is None:
            return None
        now = time.time()
        with self._lock:
            try:
                cur = self._conn.execute(
                    "INSERT INTO conversations "
                    "(person_id, session_start, audio_received_at, "
                    " wav_path, confirmed, created_at) "
                    "VALUES (?, ?, ?, ?, 0, ?)",
                    (person_id, session_start,
                     audio_received_at or now,
                     (wav_path or ""), now),
                )
                return int(cur.lastrowid)
            except Exception as e:
                print(f"[people] add_conversation failed: {e}")
                return None

    def confirm_conversation(self, conv_id: int,
                             clip_path: str = "",
                             video_received_at: Optional[float] = None
                             ) -> bool:
        """Flip a provisional conversation row to confirmed=True once the
        video clip has arrived and face-matching agrees."""
        if self._conn is None:
            return False
        now = time.time()
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE conversations "
                    "SET confirmed = 1, "
                    "    clip_path = ?, "
                    "    video_received_at = ? "
                    "WHERE id = ?",
                    (clip_path or "", video_received_at or now, conv_id),
                )
                return True
            except Exception as e:
                print(f"[people] confirm_conversation failed: {e}")
                return False

    def list_unconfirmed(self,
                         since: Optional[float] = None,
                         person_id: Optional[int] = None
                         ) -> list[dict]:
        """Return unconfirmed conversation rows. `since` filters by
        session_start (unix timestamp). Used by StreamTab reconcile logic
        to find provisional tags waiting on a video clip."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                q = ("SELECT c.id, c.person_id, c.session_start, "
                     "       c.wav_path, c.audio_received_at, "
                     "       p.name "
                     "FROM conversations c "
                     "JOIN people p ON p.id = c.person_id "
                     "WHERE c.confirmed = 0")
                params: list = []
                if since is not None:
                    q += " AND c.session_start >= ?"
                    params.append(since)
                if person_id is not None:
                    q += " AND c.person_id = ?"
                    params.append(person_id)
                q += " ORDER BY c.session_start DESC"
                rows = self._conn.execute(q, params).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                print(f"[people] list_unconfirmed failed: {e}")
                return []

    def list_conversations(self, person_id: int,
                           confirmed_only: bool = False) -> list[dict]:
        """All conversations for a person. Used by the People tab 'list
        all convos with Humza' query path."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                q = ("SELECT * FROM conversations WHERE person_id = ?")
                if confirmed_only:
                    q += " AND confirmed = 1"
                q += " ORDER BY session_start DESC"
                rows = self._conn.execute(q, (person_id,)).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                print(f"[people] list_conversations failed: {e}")
                return []

    # ── full profile updates (revamp) ────────────────────────────────────
    def update_person_details(self, person_id: int, *,
                              title: Optional[str] = None,
                              company: Optional[str] = None,
                              relationship: Optional[str] = None,
                              role_note: Optional[str] = None,
                              name: Optional[str] = None,
                              is_self: Optional[bool] = None) -> bool:
        """Patch any subset of profile fields. None means 'leave alone'."""
        if self._conn is None:
            return False
        sets: list[str] = []
        vals: list = []
        if name is not None:
            n = (name or "").strip()
            if n:
                sets.append("name = ?")
                vals.append(n)
        if role_note is not None:
            sets.append("role_note = ?")
            vals.append((role_note or "").strip())
        if title is not None:
            sets.append("title = ?")
            vals.append((title or "").strip())
        if company is not None:
            sets.append("company = ?")
            vals.append((company or "").strip())
        if relationship is not None:
            sets.append("relationship = ?")
            vals.append((relationship or "").strip())
        if is_self is not None:
            sets.append("is_self = ?")
            vals.append(1 if is_self else 0)
        if not sets:
            return True
        now = time.time()
        sets.append("updated_at = ?")
        vals.append(now)
        vals.append(person_id)
        with self._lock:
            try:
                # Ensure at most one self row.
                if is_self is True:
                    self._conn.execute(
                        "UPDATE people SET is_self = 0 "
                        "WHERE is_self = 1 AND id != ?", (person_id,))
                self._conn.execute(
                    f"UPDATE people SET {', '.join(sets)} WHERE id = ?",
                    vals)
                return True
            except Exception as e:
                print(f"[people] update_person_details failed: {e}")
                return False

    def get_self(self) -> Optional[Person]:
        """Return the row marked is_self=1, or None if no self profile."""
        if self._conn is None:
            return None
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT * FROM people WHERE is_self = 1 LIMIT 1"
                ).fetchone()
            except Exception as e:
                print(f"[people] get_self failed: {e}")
                return None
        return self._row_to_person(row)

    # ── conversation enrichment (revamp) ────────────────────────────────
    def update_conversation_stats(self, conv_id: int, *,
                                  duration_seconds: Optional[float] = None,
                                  speaker_count: Optional[int] = None,
                                  dominant_share: Optional[float] = None
                                  ) -> bool:
        """Stamp duration / speaker count / dominant-speaker share onto a
        conversation row. Called once per recording after the diarizer
        finishes so the significance rules have something to evaluate."""
        if self._conn is None:
            return False
        sets: list[str] = []
        vals: list = []
        if duration_seconds is not None:
            sets.append("duration_seconds = ?")
            vals.append(float(duration_seconds))
        if speaker_count is not None:
            sets.append("speaker_count = ?")
            vals.append(int(speaker_count))
        if dominant_share is not None:
            sets.append("dominant_share = ?")
            vals.append(float(dominant_share))
        if not sets:
            return True
        vals.append(conv_id)
        with self._lock:
            try:
                self._conn.execute(
                    f"UPDATE conversations SET {', '.join(sets)} "
                    f"WHERE id = ?", vals)
                return True
            except Exception as e:
                print(f"[people] update_conversation_stats failed: {e}")
                return False

    # ── pending prompts (revamp) ────────────────────────────────────────
    def add_pending_prompt(self, type_: str, person_id: int = 0,
                           payload: Optional[dict] = None,
                           *, dedupe_window_s: float = 86400.0
                           ) -> Optional[int]:
        """Queue a UI prompt. Returns the new row id, or None if a recent
        equivalent prompt exists (dedupe by type+person_id within window).
        The People tab polls list_pending_prompts() to surface these."""
        if self._conn is None:
            return None
        import json as _json
        cutoff = time.time() - max(0.0, dedupe_window_s)
        now = time.time()
        with self._lock:
            try:
                # Dedupe: skip if same (type, person_id) was created or
                # acted-on within the dedupe window, AND is not dismissed
                # as a "don't ask again" (dismissed=1 doesn't suppress
                # future prompts here — dedupe_window does).
                dup = self._conn.execute(
                    "SELECT id FROM pending_prompts "
                    "WHERE type = ? AND person_id = ? AND created_at >= ? "
                    "LIMIT 1",
                    (type_, int(person_id), cutoff),
                ).fetchone()
                if dup is not None:
                    return None
                cur = self._conn.execute(
                    "INSERT INTO pending_prompts "
                    "(type, person_id, payload, created_at, shown_at, dismissed) "
                    "VALUES (?, ?, ?, ?, 0, 0)",
                    (type_, int(person_id),
                     _json.dumps(payload or {}, default=str), now),
                )
                return int(cur.lastrowid)
            except Exception as e:
                print(f"[people] add_pending_prompt failed: {e}")
                return None

    def list_pending_prompts(self, *,
                             include_dismissed: bool = False
                             ) -> list[PendingPrompt]:
        if self._conn is None:
            return []
        import json as _json
        with self._lock:
            try:
                if include_dismissed:
                    rows = self._conn.execute(
                        "SELECT * FROM pending_prompts "
                        "ORDER BY created_at DESC").fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM pending_prompts "
                        "WHERE dismissed = 0 "
                        "ORDER BY created_at DESC").fetchall()
            except Exception as e:
                print(f"[people] list_pending_prompts failed: {e}")
                return []
        out: list[PendingPrompt] = []
        for r in rows:
            try:
                payload = _json.loads(r["payload"] or "{}")
            except Exception:
                payload = {}
            out.append(PendingPrompt(
                id=int(r["id"]),
                type=str(r["type"]),
                person_id=int(r["person_id"] or 0),
                payload=payload,
                created_at=float(r["created_at"] or 0.0),
                shown_at=float(r["shown_at"] or 0.0),
                dismissed=int(r["dismissed"] or 0),
            ))
        return out

    def dismiss_prompt(self, prompt_id: int,
                       *, acted_on: bool = False) -> bool:
        """Mark a prompt as handled. dismissed=1 means user skipped/closed;
        dismissed=2 means user acted on it (added person, confirmed, etc.).
        Either way it disappears from the open queue."""
        if self._conn is None:
            return False
        state = 2 if acted_on else 1
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE pending_prompts SET dismissed = ?, "
                    "shown_at = COALESCE(NULLIF(shown_at, 0), ?) "
                    "WHERE id = ?",
                    (state, time.time(), prompt_id))
                return True
            except Exception as e:
                print(f"[people] dismiss_prompt failed: {e}")
                return False

    # ── matching ────────────────────────────────────────────────────────
    def match_face(self, vector: np.ndarray,
                   threshold: float = DEFAULT_MATCH_THRESHOLD
                   ) -> Optional[MatchResult]:
        return self._match_single(vector, KIND_FACE, threshold)

    def match_voice(self, vector: np.ndarray,
                    threshold: float = DEFAULT_MATCH_THRESHOLD
                    ) -> Optional[MatchResult]:
        return self._match_single(vector, KIND_VOICE, threshold)

    def match_combined(self,
                       face_vector: Optional[np.ndarray] = None,
                       voice_vector: Optional[np.ndarray] = None,
                       threshold: float = DEFAULT_MATCH_THRESHOLD,
                       face_weight: float = 0.6,
                       voice_weight: float = 0.4
                       ) -> Optional[MatchResult]:
        if self._conn is None:
            return None
        if face_vector is None and voice_vector is None:
            return None
        fv = _to_unit_float32(face_vector)  if face_vector  is not None else None
        vv = _to_unit_float32(voice_vector) if voice_vector is not None else None
        if fv is None and vv is None:
            return None
        if fv is not None and vv is not None:
            wsum = float(face_weight + voice_weight) or 1.0
            wf, wv = face_weight / wsum, voice_weight / wsum
        elif fv is not None:
            wf, wv = 1.0, 0.0
        else:
            wf, wv = 0.0, 1.0

        best: Optional[MatchResult] = None
        for person in self.list_all():
            face_best  = _best_sim(fv, self.list_embeddings(person.id, KIND_FACE))  if fv is not None else 0.0
            voice_best = _best_sim(vv, self.list_embeddings(person.id, KIND_VOICE)) if vv is not None else 0.0
            score = wf * face_best + wv * voice_best
            if best is None or score > best.similarity:
                best = MatchResult(person=person, similarity=score,
                                   kind="combined")
        if best is None or best.similarity < threshold:
            return None
        return best

    def _match_single(self, vector: np.ndarray, kind: str,
                      threshold: float) -> Optional[MatchResult]:
        if self._conn is None:
            return None
        if kind not in _VALID_KINDS:
            return None
        v = _to_unit_float32(vector)
        if v is None:
            return None
        best: Optional[MatchResult] = None
        for person in self.list_all():
            stored = self.list_embeddings(person.id, kind)
            if not stored:
                continue
            sim = _best_sim(v, stored)
            if best is None or sim > best.similarity:
                best = MatchResult(person=person, similarity=sim, kind=kind)
        if best is None or best.similarity < threshold:
            return None
        return best

    # ── stats ────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        if self._conn is None:
            return {"people": 0, "face_embeddings": 0,
                    "voice_embeddings": 0, "conversations": 0}
        with self._lock:
            try:
                np_ = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM people").fetchone()["n"]
                nf = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM embeddings WHERE kind = ?",
                    (KIND_FACE,)).fetchone()["n"]
                nv = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM embeddings WHERE kind = ?",
                    (KIND_VOICE,)).fetchone()["n"]
                nc = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
                nu = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM conversations "
                    "WHERE confirmed = 0").fetchone()["n"]
            except Exception as e:
                print(f"[people] stats failed: {e}")
                return {"people": 0, "face_embeddings": 0,
                        "voice_embeddings": 0, "conversations": 0}
        return {"people": int(np_),
                "face_embeddings": int(nf),
                "voice_embeddings": int(nv),
                "conversations": int(nc),
                "unconfirmed_conversations": int(nu)}

    # ── helpers ─────────────────────────────────────────────────────────
    def _row_to_person(self, row) -> Optional[Person]:
        if row is None:
            return None
        try:
            pid = int(row["id"])
        except Exception:
            return None
        face_count = voice_count = 0
        try:
            face_count = int(self._conn.execute(
                "SELECT COUNT(*) AS n FROM embeddings "
                "WHERE person_id = ? AND kind = ?",
                (pid, KIND_FACE)).fetchone()["n"])
            voice_count = int(self._conn.execute(
                "SELECT COUNT(*) AS n FROM embeddings "
                "WHERE person_id = ? AND kind = ?",
                (pid, KIND_VOICE)).fetchone()["n"])
        except Exception:
            pass
        # folder_path guard: old rows written before the migration have no
        # column yet in the row_factory dict — catch gracefully.
        folder_path = ""
        try:
            folder_path = str(row["folder_path"] or "")
        except (IndexError, KeyError):
            folder_path = ""
        # Same guard pattern for the revamp columns.
        def _safe(col: str, default=""):
            try:
                v = row[col]
                return v if v is not None else default
            except (IndexError, KeyError):
                return default
        return Person(
            id=pid,
            name=str(row["name"]),
            role_note=str(row["role_note"] or ""),
            first_seen=float(row["first_seen"] or 0.0),
            last_seen=float(row["last_seen"]  or 0.0),
            times_seen=int(row["times_seen"]  or 0),
            created_at=float(row["created_at"] or 0.0),
            updated_at=float(row["updated_at"] or 0.0),
            folder_path=folder_path,
            title=str(_safe("title", "")),
            company=str(_safe("company", "")),
            relationship=str(_safe("relationship", "")),
            is_self=bool(int(_safe("is_self", 0) or 0)),
            face_count=face_count,
            voice_count=voice_count,
        )


# ── module-level helpers ────────────────────────────────────────────────
def _to_unit_float32(vector) -> Optional[np.ndarray]:
    if vector is None:
        return None
    try:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if v.size == 0 or not np.all(np.isfinite(v)):
        return None
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return None
    return (v / norm).astype(np.float32, copy=False)


def _blob_to_unit_float32(blob: bytes,
                          expected_dim: int) -> Optional[np.ndarray]:
    try:
        v = np.frombuffer(blob, dtype=np.float32)
    except Exception:
        return None
    if expected_dim > 0 and v.size != expected_dim:
        return None
    return _to_unit_float32(v)


def _best_sim(query: np.ndarray, stored: list[np.ndarray]) -> float:
    if not stored:
        return 0.0
    best = -1.0
    for s in stored:
        if s.shape != query.shape:
            continue
        sim = float(np.dot(query, s))
        if sim > best:
            best = sim
    return max(0.0, best)


def _fmt_ts(ts: float) -> str:
    try:
        if ts <= 0:
            return "—"
        return datetime.fromtimestamp(ts).strftime("%b %d %H:%M:%S")
    except Exception:
        return "—"


def default_db_path() -> str:
    candidates: list[str] = []
    env = os.environ.get("IRIS_PEOPLE_DB")
    if env:
        candidates.append(env)
    try:
        candidates.append(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "sqlite",
            "people.db"))
    except Exception:
        pass
    candidates.append(os.path.join(os.getcwd(), "data", "sqlite",
                                   "people.db"))
    return candidates[0]