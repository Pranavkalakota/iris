"""
Phase 9 configuration: adds speaker ID and LLM summarization on top of Phase 8.
"""
import os
# --- Network ---
STREAM_PORT          = 5005
DISCOVERY_PORT       = 5006
DISCOVERY_INTERVAL_S = 2.0
DISCOVERY_MESSAGE    = b"AUDIO_HOST"
# --- Packet layout ---
SEQ_HEADER_BYTES     = 4
SAMPLES_PER_PACKET   = 512
PACKET_PAYLOAD_BYTES = SAMPLES_PER_PACKET * 2
# --- Audio format ---
SAMPLE_RATE          = 16000
CHANNELS             = 1
SAMPLE_DTYPE         = "int16"
# --- Playback / buffering ---
BLOCK_SAMPLES        = 512
PREROLL_SAMPLES      = 1600
RING_CAPACITY        = 192000
# --- Recording ---
RECORDINGS_DIR       = r"C:\audio_stream_glass_version\recordings"
CHUNK_SECONDS        = 60
# --- Transcription ---
WHISPER_MODEL        = "medium.en"
WHISPER_DEVICE       = "cpu"
WHISPER_COMPUTE      = "int8"
WHISPER_BEAM_SIZE    = 8
AUTO_TRANSCRIBE      = True
# --- GUI ---
GUI_APPEARANCE       = "dark"
GUI_COLOR_THEME      = "blue"
GUI_WINDOW_W         = 1400
GUI_WINDOW_H         = 850
GUI_POLL_MS          = 50
GUI_VU_DECAY_MS      = 80
GUI_SHOW_TIMESTAMPS  = True
# --- Location ---
LOCATION_PROVIDER    = "ip-api"
LOCATION_TIMEOUT_S   = 5.0
# --- Map ---
MAP_TILE_URL         = "https://cartodb-basemaps-c.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png"
MAP_DEFAULT_ZOOM     = 11
MAP_CLUSTER_RADIUS_M = 30
MAP_FALLBACK_LAT     = 39.7684
MAP_FALLBACK_LON     = -86.1581
# --- Speaker identification (new in Phase 9) ---
SPEAKERS_DB_PATH     = r"C:\audio_stream_glass_version\speakers.json"
AUTO_DIARIZE         = True
MATCH_STRICT_THRESH  = 0.85    # >= this -> auto-tag as known speaker
MATCH_WEAK_THRESH    = 0.60    # >= this -> tag with "Name?" + confidence
MAX_EMBEDDINGS_PER_PROFILE = 30   # keep at most N samples per known speaker
# --- Summarization (new in Phase 9) ---
AUTO_SUMMARIZE       = True
OLLAMA_URL           = "http://localhost:11434"
OLLAMA_MODEL         = "llama3.2:3b"
OLLAMA_TIMEOUT_S     = 120.0
# --- Diagnostics ---
STATS_INTERVAL_S     = 1.0
SHOW_REC_DURATION    = True
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(r"C:\audio_stream_glass_version\photos", exist_ok=True)
# --- ESP32 Camera (photo capture) ---
# ESP32_CAMERA_ENABLED reads from the IRIS_CAMERA env var at launch:
#   unset (default) or "1"  -> camera enabled  (same as the previous True)
#   "0"                     -> camera disabled (skips photo trigger, no timeouts)
# To launch with the camera disabled in one PowerShell session:
#   $env:IRIS_CAMERA="0"; python iris_gui.py
# Closing the PowerShell window resets it — no persistent state to clean up.
ESP32_CAMERA_ENABLED      = os.environ.get("IRIS_CAMERA", "1") == "1"
ESP32_CAMERA_IP           = "192.168.1.210"
ESP32_CAMERA_PHOTO_PORT   = 5006
ESP32_CAMERA_PHOTOS_DIR   = r"C:\Users\delete me\Desktop\camera_photos"
ESP32_CAMERA_WAIT_SECONDS = 20.0
# --- Photos storage ---
PHOTOS_DIR = r"C:\audio_stream_glass_version\photos"
# ── LLaVA vision model (new for M6: role/scene/OCR share one client) ──────
# iris_fusion reads this; kept explicit here so scene description + OCR use a
# known model. Any Ollama LLaVA tag works (e.g. "llava:7b", "llava:13b").
OLLAMA_LLAVA_MODEL   = "llava:7b"
# ── M6 §6.3 Event Boundary Detection ──────────────────────────────────────
EVENT_BOUNDARY_ENABLED     = True
EVENTS_DIR                 = r"C:\audio_stream_glass_version\data\events"
EVENT_MIN_CONFIRM_CLIPS    = 2       # consecutive confirming clips before firing
EVENT_MAX_GAP_SECONDS      = 300.0   # >300s between clips → force boundary
EVENT_SAME_THRESHOLD       = 0.75    # combined score > this  → same event
EVENT_NEW_THRESHOLD        = 0.50    # combined score < this  → boundary candidate
# Signal weights (must reflect §6.3; renormalised over whichever are present).
EVENT_W_VISUAL             = 0.40    # LLaVA/MiniLM scene-embedding cosine
EVENT_W_AUDIO              = 0.30    # librosa MFCC cosine
EVENT_W_FACE               = 0.20    # Jaccard of face-ID sets
EVENT_W_MOTION             = 0.10    # optical-flow mean-magnitude delta
# ── M6 §6.5 Location inference (OCR → Wi-Fi SSID → ip-api) ─────────────────
LOCATION_OCR_ENABLED       = True    # Source 1: LLaVA OCR of signage (primary)
LOCATION_WIFI_ENABLED      = True    # Source 2: connected Wi-Fi SSID
LOCATION_IP_ENABLED        = True    # Source 3: ip-api.com fallback
# Optional: map known SSIDs to friendly venue/home names.
#   e.g. {"Yaffa_Guest": "Yaffa Coffee House", "HomeNet": "Home"}
SSID_LOCATION_MAP          = {}
# ── IRIS Option B wake-word (openwakeword + Google backup) ────────────────
# Run the Google-speech-based backup wake listener in parallel with
# openwakeword. Both fire the same downstream callback; a 2-second
# debouncer in main_phase9.Controller collapses double-fires so
# handle_voice_trigger only runs once per utterance. Set to False to
# disable the backup and rely on openwakeword alone.
USE_GOOGLE_WAKE_BACKUP = True
# ── IRIS input source ─────────────────────────────────────────────────────
# Option B (default): audio comes from the ESP32 ring buffer, no Windows
# audio device is opened. Set to True to switch to a local Windows mic
# (Option A) — useful for demoing without the ESP32 attached.
USE_PC_MIC = False