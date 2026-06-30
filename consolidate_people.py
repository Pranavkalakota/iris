"""
consolidate_people.py — one-shot cleanup for duplicate suffixed-name rows.

Finds rows like 'Humza', 'Humza 2', 'Humza 3', 'Humza 4', 'Humza 5' in
people.db and merges them all into the canonical (suffix-free) row. The
merge moves all voice/face embeddings, sums times_seen, moves the on-disk
sessions/ folders, and deletes the duplicate rows.

How to run (from C:\\audio_stream_glass_version):
    python consolidate_people.py            # dry-run: shows what would happen
    python consolidate_people.py --apply    # actually does the merges

Safe to re-run. Idempotent. Only merges rows whose names match the
pattern '<base>' or '<base> <integer>'. Will never merge 'Humza' with
'Humza Malik' (full name is a different identity).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict


def _strip_numeric_suffix(name: str) -> str:
    """'Humza 3' -> 'Humza', 'Humza' -> 'Humza', 'Unknown 7' -> 'Unknown'."""
    n = (name or "").strip()
    m = re.match(r"^(.+?)\s+(\d+)$", n)
    return m.group(1).strip() if m else n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Actually perform the merges (default: dry-run only).")
    ap.add_argument("--include-unknowns", action="store_true",
                    help="Also consolidate 'Unknown N' rows (default: skip "
                         "them — they're meant to be distinct unenrolled "
                         "people, you rename them yourself).")
    ap.add_argument("--db", default=None,
                    help="Path to people.db (defaults to the iris_people "
                         "default path).")
    args = ap.parse_args()

    # Import lazily so the script still prints help even outside the project.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import iris_people
        import iris_fusion
    except ImportError as e:
        print(f"ERROR: could not import iris modules: {e}")
        print("Run this script from your IRIS project root "
              "(where iris_people.py and iris_fusion.py live).")
        return 1

    db_path = args.db or iris_people.default_db_path()
    if not os.path.exists(db_path):
        print(f"ERROR: people.db not found at {db_path}")
        return 1

    print(f"Opening {db_path}\n")

    # Use fusion so merge_people() also moves on-disk folder contents.
    fusion = iris_fusion.PeopleFusion(db_path=db_path)
    store = fusion.store

    # Group every row by its base name.
    people = store.list_all()
    groups: dict[str, list] = defaultdict(list)
    for p in people:
        base = _strip_numeric_suffix(p.name)
        groups[base].append(p)

    # Filter to groups with 2+ members.
    plans: list[tuple[str, "iris_people.Person", list]] = []
    for base, members in groups.items():
        if len(members) < 2:
            continue
        if not args.include_unknowns and base.lower() == "unknown":
            continue

        # Pick the canonical: prefer the exact-base-name row, otherwise
        # the earliest-created. Stable choice so dry-run and apply agree.
        exact = [m for m in members if m.name == base]
        if exact:
            keep = exact[0]
        else:
            keep = min(members, key=lambda m: m.created_at or 0.0)
        drop = [m for m in members if m.id != keep.id]
        plans.append((base, keep, drop))

    if not plans:
        print("No suffixed-name duplicates found. Nothing to do.")
        fusion.shutdown()
        return 0

    # Show the plan.
    print("=" * 72)
    print("MERGE PLAN")
    print("=" * 72)
    total_to_delete = 0
    for base, keep, drop in plans:
        print(f"\nBase name: {base!r}")
        print(f"  KEEP    #{keep.id:3d}  {keep.name!r}  "
              f"({keep.voice_count} voices, {keep.face_count} faces, "
              f"{keep.times_seen} seen)")
        for d in drop:
            print(f"    \u2192 merge #{d.id:3d}  {d.name!r}  "
                  f"({d.voice_count} voices, {d.face_count} faces, "
                  f"{d.times_seen} seen)")
            total_to_delete += 1
    print()
    print(f"Total rows to delete after merge: {total_to_delete}")
    print(f"Rows kept (canonical):            {len(plans)}")

    if not args.apply:
        print()
        print("This was a DRY-RUN. Re-run with --apply to actually perform "
              "the merges.")
        fusion.shutdown()
        return 0

    # Confirm before destructive action.
    print()
    resp = input("Proceed with merge? Type 'yes' to confirm: ").strip().lower()
    if resp != "yes":
        print("Aborted.")
        fusion.shutdown()
        return 0

    # Apply.
    print()
    succeeded = 0
    failed = 0
    for base, keep, drop in plans:
        for d in drop:
            try:
                report = fusion.merge_people(keep_id=keep.id, drop_id=d.id)
            except Exception as e:
                print(f"  EXCEPTION merging #{d.id} \u2192 #{keep.id}: {e}")
                failed += 1
                continue
            if report.success:
                print(f"  \u2713 merged #{d.id} {d.name!r} \u2192 "
                      f"#{keep.id} {report.kept_name!r}  "
                      f"(moved {report.embeddings_moved_voice} voice + "
                      f"{report.embeddings_moved_face} face emb)")
                succeeded += 1
            else:
                print(f"  \u2717 FAILED to merge #{d.id} {d.name!r}: "
                      f"{report.error}")
                failed += 1

    print()
    print(f"Done: {succeeded} merged, {failed} failed.")
    print("Restart iris_gui.py to see the cleaned-up People tab.")
    fusion.shutdown()
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())