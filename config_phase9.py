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