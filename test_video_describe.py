"""
Standalone smoke test for video scene description — bypasses the full
GUI/fusion stack (no ChromaDB, no DeepFace, no PyQt6 needed) and talks
directly to Ollama's llava:7b, same code path iris_gui.py uses.
Run from the same folder as iris_videos.py / iris_fusion.py.
"""
import glob, os, sys

from iris_fusion import _LlavaInference, _read_ollama_cfg
from iris_videos import _sample_frames_spread

RECORDING_DIR = r"C:\Users\delete me\Desktop\ESP32_Recording"

def latest_clip(folder):
    clips = glob.glob(os.path.join(folder, "*.avi"))
    if not clips:
        return None
    return max(clips, key=os.path.getmtime)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest_clip(RECORDING_DIR)
    if not path or not os.path.exists(path):
        print(f"No clip found. Looked in: {RECORDING_DIR}")
        return
    print(f"Testing clip: {path}")

    url, model = _read_ollama_cfg()
    print(f"Ollama: {url} | model: {model}")

    frames = _sample_frames_spread(path, count=4)
    print(f"Sampled {len(frames)} frames")
    if not frames:
        print("No frames decoded — file may be corrupt or cv2 can't open it.")
        return

    llava = _LlavaInference(url, model)
    print("Calling llava.describe_frames() ... (can take 30s-2min on CPU)")
    description = llava.describe_frames(frames)
    print("\n--- DESCRIPTION ---")
    print(description or "(empty — see [llava] error prints above)")

if __name__ == "__main__":
    main()
