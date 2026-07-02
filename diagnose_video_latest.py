"""
diagnose_video_latest.py — run this from the same folder as iris_gui.py on
your Windows machine (python diagnose_video_latest.py). It prints exactly
what VideoStore sees for your ESP32_Recording folder: every clip found, its
computed 'received_at' timestamp, where that timestamp came from (sidecar /
filename / mtime), and the real on-disk mtime for comparison. Paste the
output back — that will show exactly why 191759 beat 203006.
"""
import os
import iris_videos as ivideos
 
vs = ivideos.VideoStore()
print("Folders being scanned:")
for f in vs.folders():
    print("  ", f)
print()
 
clips = vs.list_all()
print(f"Total clips found: {len(clips)}\n")
 
print(f"{'filename':45} {'received_at (parsed)':22} {'real mtime':22} {'sidecar?'}")
for c in clips[:15]:
    real_mtime = "?"
    try:
        real_mtime = ivideos.datetime.fromtimestamp(
            os.path.getmtime(c.path)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    sidecar = ivideos.read_sidecar(c.path)
    has_sidecar = "yes" if sidecar else "no"
    print(f"{c.name:45} {c.when():22} {real_mtime:22} {has_sidecar}")
 
print("\n--- Specifically checking the two files in question ---")
for target in ("191759", "203006"):
    match = [c for c in clips if target in c.name]
    for c in match:
        print(f"\n{c.name}")
        print(f"  parsed received_at : {c.when()}  ({c.received_at})")
        print(f"  filename-regex ts  : {ivideos._timestamp_from_name(c.path)}")
        try:
            print(f"  real file mtime    : {os.path.getmtime(c.path)} "
                  f"({ivideos.datetime.fromtimestamp(os.path.getmtime(c.path))})")
        except Exception as e:
            print(f"  real file mtime    : error ({e})")
        print(f"  sidecar contents   : {ivideos.read_sidecar(c.path)}")
 