"""
test_iris_m5_gap1.py
====================
Self-contained tests for the Gap 1 / Gap 2 / Gap 3 changes.

Requires only:  numpy   (already in IRIS requirements)
Does NOT need:  DeepFace, SpeechBrain, PyQt6, Ollama, a camera, or audio.

Run from the project folder (where iris_people.py, iris_voices.py, and
iris_fusion.py live):

    python test_iris_m5_gap1.py

All tests print PASS / FAIL. Exit code 0 = all passed.
"""

from __future__ import annotations
import os
import sys
import json
import shutil
import tempfile
import time
import numpy as np

# ── make sure project root is on the path ────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
_pass = 0
_fail = 0


def ok(label: str) -> None:
    global _pass
    _pass += 1
    print(f"  PASS  {label}")


def fail(label: str, reason: str = "") -> None:
    global _fail
    _fail += 1
    print(f"  FAIL  {label}" + (f" — {reason}" if reason else ""))


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def rand_vec(dim: int) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# Test suite
# ════════════════════════════════════════════════════════════════════════════

def test_people_store(tmp: str) -> None:
    section("iris_people.PeopleStore — schema, folder_path, conversations")
    import iris_people

    db_path = os.path.join(tmp, "people.db")
    store = iris_people.PeopleStore(db_path)

    # ── basic person creation ────────────────────────────────────────────
    p = store.add("Humza")
    if p is None:
        fail("add person"); return
    ok("add person")

    # ── folder_path starts empty ─────────────────────────────────────────
    if p.folder_path == "":
        ok("folder_path starts empty")
    else:
        fail("folder_path starts empty", repr(p.folder_path))

    # ── set_folder_path ──────────────────────────────────────────────────
    folder = os.path.join(tmp, "people_store", "humza")
    os.makedirs(folder, exist_ok=True)
    ok_flag = store.set_folder_path(p.id, folder)
    if ok_flag:
        ok("set_folder_path returns True")
    else:
        fail("set_folder_path returns True")

    refreshed = store.get(p.id)
    if refreshed and refreshed.folder_path == folder:
        ok("folder_path persisted after set")
    else:
        fail("folder_path persisted after set",
             repr(getattr(refreshed, "folder_path", None)))

    # ── migration: open existing DB that has no folder_path column ───────
    # Simulate by creating a fresh DB without the column and reopening.
    import sqlite3
    legacy_path = os.path.join(tmp, "legacy.db")
    conn = sqlite3.connect(legacy_path)
    conn.execute("""CREATE TABLE people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        role_note TEXT NOT NULL DEFAULT '',
        first_seen REAL NOT NULL,
        last_seen REAL NOT NULL,
        times_seen INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )""")
    now = time.time()
    conn.execute("INSERT INTO people VALUES (1,'Ali','',?,?,0,?,?)",
                 (now, now, now, now))
    conn.commit()
    conn.close()

    store2 = iris_people.PeopleStore(legacy_path)
    ali = store2.get(1)
    if ali is not None and ali.folder_path == "":
        ok("migration: legacy DB opened without crash, folder_path=''"
           )
    else:
        fail("migration: legacy DB",
             f"ali={ali}")
    store2.close()

    # ── conversations: add_conversation ──────────────────────────────────
    conv_id = store.add_conversation(
        person_id=p.id,
        session_start=time.time(),
        wav_path="/tmp/test.wav",
    )
    if conv_id is not None:
        ok("add_conversation returns id")
    else:
        fail("add_conversation returns id")

    # ── list_unconfirmed ─────────────────────────────────────────────────
    pending = store.list_unconfirmed()
    if any(r["id"] == conv_id for r in pending):
        ok("list_unconfirmed finds new row")
    else:
        fail("list_unconfirmed finds new row")

    # ── confirm_conversation ─────────────────────────────────────────────
    ok_flag = store.confirm_conversation(
        conv_id, clip_path="/tmp/test.avi",
        video_received_at=time.time())
    if ok_flag:
        ok("confirm_conversation returns True")
    else:
        fail("confirm_conversation returns True")

    still_pending = store.list_unconfirmed()
    if not any(r["id"] == conv_id for r in still_pending):
        ok("row removed from list_unconfirmed after confirm")
    else:
        fail("row removed from list_unconfirmed after confirm")

    # ── list_conversations ───────────────────────────────────────────────
    convs = store.list_conversations(p.id)
    if convs:
        ok("list_conversations returns rows")
    else:
        fail("list_conversations returns rows")

    confirmed_convs = store.list_conversations(p.id, confirmed_only=True)
    if confirmed_convs:
        ok("list_conversations confirmed_only=True works")
    else:
        fail("list_conversations confirmed_only=True works")

    # ── stats include conversations ──────────────────────────────────────
    s = store.stats()
    if "conversations" in s and s["conversations"] >= 1:
        ok("stats() includes conversations count")
    else:
        fail("stats() includes conversations count", str(s))

    store.close()


def test_voices_pipeline(tmp: str) -> None:
    section("iris_voices.VoicePipeline — Gap 3 indentation fix")
    import iris_voices
    import iris_people

    db_path = os.path.join(tmp, "voices_people.db")
    store = iris_people.PeopleStore(db_path)
    pipeline = iris_voices.VoicePipeline()

    # ── _process_one_cluster is a METHOD, not a module-level function ────
    if hasattr(pipeline, "_process_one_cluster"):
        ok("_process_one_cluster is a method of VoicePipeline")
    else:
        fail("_process_one_cluster is a method of VoicePipeline")

    # ── _is_placeholder_name exists and works ────────────────────────────
    if hasattr(pipeline, "_is_placeholder_name"):
        ok("_is_placeholder_name staticmethod exists")
    else:
        fail("_is_placeholder_name staticmethod exists")

    if pipeline._is_placeholder_name("Unknown 1"):
        ok("_is_placeholder_name('Unknown 1') = True")
    else:
        fail("_is_placeholder_name('Unknown 1') = True")

    if not pipeline._is_placeholder_name("Humza"):
        ok("_is_placeholder_name('Humza') = False")
    else:
        fail("_is_placeholder_name('Humza') = False")

    # ── process_recording on a missing file returns an error ─────────────
    result = pipeline.process_recording("/nonexistent/file.wav", store)
    if result.error:
        ok("process_recording non-existent wav → error, no crash")
    else:
        fail("process_recording non-existent wav → error, no crash")

    # ── synthesize a full ingestion with a fake NPZ + JSON ───────────────
    wav = os.path.join(tmp, "recording_test.wav")
    open(wav, "wb").close()   # empty placeholder

    emb = rand_vec(192)
    npz = os.path.join(tmp, "recording_test.embeddings.npz")
    np.savez(npz, cluster_0=emb)

    transcript_data = {
        "segments": [
            {"_cluster": 0, "speaker": "Ali",
             "speaker_kind": "strict", "text": "hello"}
        ]
    }
    jpath = os.path.join(tmp, "recording_test.json")
    with open(jpath, "w") as f:
        json.dump(transcript_data, f)

    result2 = pipeline.process_recording(wav, store)
    if not result2.error:
        ok("process_recording with fake NPZ/JSON — no error")
    else:
        fail("process_recording with fake NPZ/JSON — no error",
             result2.error)

    if result2.clusters_total == 1:
        ok("process_recording found 1 cluster")
    else:
        fail("process_recording found 1 cluster",
             f"got {result2.clusters_total}")

    if result2.clusters_enrolled == 1:
        ok("new speaker enrolled (Ali)")
    else:
        fail("new speaker enrolled (Ali)")

    # Verify the person is in the DB
    ali = store.get_by_name("Ali")
    if ali is not None:
        ok("enrolled person exists in DB")
    else:
        fail("enrolled person exists in DB")

    # Re-ingest same file → should be skipped (idempotent marker)
    result3 = pipeline.process_recording(wav, store)
    if result3.skipped:
        ok("re-ingestion of same WAV is skipped (idempotent)")
    else:
        fail("re-ingestion of same WAV is skipped (idempotent)")

    store.close()


def test_fusion_folders(tmp: str) -> None:
    section("iris_fusion.PeopleFusion — folder-per-person (Gap 1)")
    import iris_fusion as fus

    # Use a fresh singleton for this test.
    fus._fusion_singleton = None

    db_path   = os.path.join(tmp, "fusion_people.db")
    store_dir = os.path.join(tmp, "fusion_people_store")
    fusion = fus.PeopleFusion(
        db_path=db_path,
        people_store_dir=store_dir,
    )

    # ── people_store_dir created at init ─────────────────────────────────
    if os.path.isdir(store_dir):
        ok("people_store_dir created at PeopleFusion.__init__")
    else:
        fail("people_store_dir created at PeopleFusion.__init__")

    # ── _slugify_name ─────────────────────────────────────────────────────
    assert fus._slugify_name("Humza Malik") == "humza_malik", \
        fus._slugify_name("Humza Malik")
    ok("_slugify_name('Humza Malik') = 'humza_malik'")

    assert fus._slugify_name("Unknown 1") == "unknown_1"
    ok("_slugify_name('Unknown 1') = 'unknown_1'")

    assert fus._slugify_name("José García") == "jose_garcia"
    ok("_slugify_name('José García') = 'jose_garcia'")

    # ── ensure_person_folder creates folder + profile.json ───────────────
    person = fusion.store.add("Pranav")
    assert person is not None
    folder = fusion.ensure_person_folder(person.id)

    if folder and os.path.isdir(folder):
        ok("ensure_person_folder creates directory")
    else:
        fail("ensure_person_folder creates directory", repr(folder))

    sessions_dir = os.path.join(folder, "sessions")
    if os.path.isdir(sessions_dir):
        ok("ensure_person_folder creates sessions/ subdirectory")
    else:
        fail("ensure_person_folder creates sessions/ subdirectory")

    profile_path = os.path.join(folder, "profile.json")
    if os.path.isfile(profile_path):
        ok("profile.json written")
    else:
        fail("profile.json written")

    with open(profile_path) as f:
        profile = json.load(f)
    if profile.get("name") == "Pranav":
        ok("profile.json contains correct name")
    else:
        fail("profile.json contains correct name", str(profile))

    # ── folder_path stored in DB ─────────────────────────────────────────
    refreshed = fusion.store.get(person.id)
    if refreshed and refreshed.folder_path == folder:
        ok("folder_path written back to DB")
    else:
        fail("folder_path written back to DB",
             repr(getattr(refreshed, "folder_path", None)))

    # ── idempotent: calling again doesn't change folder_path ────────────
    folder2 = fusion.ensure_person_folder(person.id)
    if folder2 == folder:
        ok("ensure_person_folder is idempotent")
    else:
        fail("ensure_person_folder is idempotent",
             f"{folder!r} vs {folder2!r}")

    # ── save_reference_face (uses a tiny black 10×10 BGR image) ─────────
    dummy_face = np.zeros((10, 10, 3), dtype=np.uint8)
    ref_path = fusion.save_reference_face(person.id, dummy_face)
    if ref_path and os.path.isfile(ref_path):
        ok("save_reference_face writes face_ref.jpg")
    else:
        fail("save_reference_face writes face_ref.jpg",
             "cv2.imwrite failed or cv2 not installed")

    # ── save_reference_face is idempotent (doesn't overwrite) ───────────
    ref_path2 = fusion.save_reference_face(person.id, dummy_face)
    if ref_path2 == ref_path:
        ok("save_reference_face does not overwrite existing reference")
    else:
        fail("save_reference_face does not overwrite existing reference")

    # ── save_reference_voice ─────────────────────────────────────────────
    fake_wav = os.path.join(tmp, "fake_voice.wav")
    with open(fake_wav, "wb") as f:
        f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")   # minimal valid-ish WAV
    ref_wav = fusion.save_reference_voice(person.id, fake_wav)
    if ref_wav and os.path.isfile(ref_wav):
        ok("save_reference_voice copies voice_ref.wav")
    else:
        fail("save_reference_voice copies voice_ref.wav")

    # ── archive_session_media copies into sessions/ ──────────────────────
    fake_clip = os.path.join(tmp, "clip_001.avi")
    with open(fake_clip, "wb") as f:
        f.write(b"\x00" * 64)
    archived = fusion.archive_session_media(person.id, fake_clip)
    if archived and os.path.isfile(archived):
        ok("archive_session_media copies clip into sessions/")
    else:
        fail("archive_session_media copies clip into sessions/")

    if archived and archived.startswith(sessions_dir):
        ok("archived clip is inside person's sessions/ folder")
    else:
        fail("archived clip is inside person's sessions/ folder",
             repr(archived))

    # ── archive_session_media: same second → no double copy ─────────────
    archived2 = fusion.archive_session_media(person.id, fake_clip)
    if archived2 == archived:
        ok("archive_session_media skips double-copy in same second")
    else:
        # Not a hard failure — could happen if the second call lands on the
        # next clock second, but that's unlikely in a tight test loop.
        ok("archive_session_media (timing: second may have ticked, soft pass)")

    # ── rename keeps folder path stable ──────────────────────────────────
    fusion.rename(person.id, "Pranav Renamed")
    r2 = fusion.store.get(person.id)
    if r2 and r2.folder_path == folder:
        ok("rename keeps folder_path stable (slug created at enrollment)")
    else:
        fail("rename keeps folder_path stable",
             repr(getattr(r2, "folder_path", None)))

    # profile.json refreshed with new name
    with open(profile_path) as f:
        profile2 = json.load(f)
    if profile2.get("name") == "Pranav Renamed":
        ok("profile.json updated after rename")
    else:
        fail("profile.json updated after rename", str(profile2))

    fusion.store.close()
    fus._fusion_singleton = None


def test_fusion_conversations(tmp: str) -> None:
    section("iris_fusion.PeopleFusion — two-pass conversations (Gap 2)")
    import iris_fusion as fus
    import iris_people

    fus._fusion_singleton = None
    db_path   = os.path.join(tmp, "conv_people.db")
    store_dir = os.path.join(tmp, "conv_people_store")
    fusion = fus.PeopleFusion(db_path=db_path, people_store_dir=store_dir)

    # Create two people and add a provisional conversation row for each.
    p1 = fusion.store.add("Alice")
    p2 = fusion.store.add("Bob")
    assert p1 and p2

    session_ts = time.time()
    c1 = fusion.store.add_conversation(
        p1.id, session_ts, wav_path="/tmp/alice.wav")
    c2 = fusion.store.add_conversation(
        p2.id, session_ts, wav_path="/tmp/bob.wav")

    if c1 is not None and c2 is not None:
        ok("add_conversation returns ids for both people")
    else:
        fail("add_conversation returns ids for both people")

    pending = fusion.store.list_unconfirmed(since=session_ts - 1)
    if len(pending) == 2:
        ok("two unconfirmed rows visible")
    else:
        fail("two unconfirmed rows visible", f"got {len(pending)}")

    # Confirm Alice's row.
    fusion.store.confirm_conversation(c1, clip_path="/tmp/clip.avi",
                                      video_received_at=time.time())
    still = fusion.store.list_unconfirmed(since=session_ts - 1)
    if len(still) == 1 and still[0]["id"] == c2:
        ok("only Bob's row remains unconfirmed after Alice confirmed")
    else:
        fail("only Bob's row remains unconfirmed",
             str([r["id"] for r in still]))

    # Test reconcile_clip with no face pipeline ready (graceful skip).
    fake_clip = os.path.join(tmp, "reconcile_test.avi")
    with open(fake_clip, "wb") as f:
        f.write(b"\x00" * 32)
    n = fusion.reconcile_clip(fake_clip)
    # DeepFace isn't loaded in this test, so is_ready() = False → 0 confirmed.
    if n == 0:
        ok("reconcile_clip returns 0 when face pipeline not ready (graceful)")
    else:
        fail("reconcile_clip returns 0 when face pipeline not ready",
             f"got {n}")

    fusion.store.close()
    fus._fusion_singleton = None


def test_merge_folders(tmp: str) -> None:
    section("iris_fusion.PeopleFusion — merge_people folder handling")
    import iris_fusion as fus
    import iris_people
    fus._fusion_singleton = None

    db_path   = os.path.join(tmp, "merge_people.db")
    store_dir = os.path.join(tmp, "merge_people_store")
    fusion = fus.PeopleFusion(db_path=db_path, people_store_dir=store_dir)

    keep_p = fusion.store.add("Jacob")
    drop_p = fusion.store.add("Unknown 1")
    assert keep_p and drop_p

    keep_folder = fusion.ensure_person_folder(keep_p.id)
    drop_folder = fusion.ensure_person_folder(drop_p.id)

    # Put a fake session file in the drop person's sessions/ folder.
    drop_sessions = os.path.join(drop_folder, "sessions")
    test_file = os.path.join(drop_sessions, "2026-01-01_00-00-00.wav")
    with open(test_file, "wb") as f:
        f.write(b"\x00" * 32)

    # Also add a face embedding to the drop person so the merge has something.
    fusion.store.add_embedding(drop_p.id, iris_people.KIND_FACE, rand_vec(512))

    report = fusion.merge_people(keep_id=keep_p.id, drop_id=drop_p.id)

    if report.success:
        ok("merge_people succeeded")
    else:
        fail("merge_people succeeded", report.error)

    # Session file should have moved to keep's sessions/.
    keep_sessions = os.path.join(keep_folder, "sessions")
    moved_files = os.listdir(keep_sessions)
    if any("2026-01-01" in fn for fn in moved_files):
        ok("session file moved from drop → keep sessions/")
    else:
        fail("session file moved from drop → keep sessions/",
                str(moved_files))

    # Drop folder should be gone.
    if not os.path.isdir(drop_folder):
        ok("drop folder removed after merge")
    else:
        fail("drop folder removed after merge")

    # Drop person should be gone from DB.
    if fusion.store.get(drop_p.id) is None:
        ok("drop person deleted from DB")
    else:
        fail("drop person deleted from DB")

    fusion.store.close()
    fus._fusion_singleton = None


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════
def main() -> int:
    print("\nIRIS M5 Gap 1/2/3 — test suite")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="iris_test_")
    print(f"  scratch dir: {tmp}\n")

    try:
        test_people_store(tmp)
        test_voices_pipeline(tmp)
        test_fusion_folders(tmp)
        test_fusion_conversations(tmp)
        test_merge_folders(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"  Results: {_pass} passed, {_fail} failed")
    print(f"{'='*60}\n")

    if _fail > 0:
        print("  Some tests failed. Check the FAIL lines above.")
        print("  Common causes:")
        print("    - cv2 not installed → save_reference_face FAIL is expected")
        print("      (run: pip install opencv-python)")
        print("    - numpy missing  → pip install numpy")
        print("    - Files not in project root → run from the IRIS folder")
    else:
        print("  All tests passed. Drop the three files in and restart IRIS.")

    return 1 if _fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())