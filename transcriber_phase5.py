"""
Background transcription worker.

Loads a faster-whisper model once at startup (loading is ~10 seconds).
Then waits on a queue. When a WAV path is pushed, transcribes it and
writes two files alongside it: a .txt (human-readable timestamps) and
a .json (machine-readable, for downstream LLM use).

Designed to be tolerant of being slower than recording. If chunks pile
up faster than they can be transcribed, they just queue. No data loss.
"""

import json
import os
import queue
import threading
import time
import wave
from typing import Optional

# faster_whisper import is deferred to load_model() so this file can
# be imported even if the package isn't installed yet — useful for
# config inspection without forcing the heavy import.


def _format_timestamp(seconds: float) -> str:
    """Convert 73.456 -> '01:13.46'"""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


class Transcriber(threading.Thread):
    def __init__(self, model_name: str, device: str, compute_type: str,
                 beam_size: int = 5):
        super().__init__(daemon=True, name="Transcriber")
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size

        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._model = None
        self._ready_event = threading.Event()

        # Stats
        self.jobs_completed = 0
        self.jobs_failed = 0

    # ---------- public API ----------
    def submit(self, wav_path: str) -> None:
        """Queue a WAV file for transcription. Returns immediately."""
        self._queue.put(wav_path)

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)  # unblock the queue.get()

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """Block until the model has finished loading."""
        return self._ready_event.wait(timeout)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ---------- worker ----------
    def run(self) -> None:
        # Load the model. This is slow (first run downloads ~460 MB).
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("[transcriber] faster-whisper not installed. "
                  "Run: pip install faster-whisper")
            return

        print(f"[transcriber] loading model '{self._model_name}' "
              f"({self._device}, {self._compute_type})...")
        t0 = time.time()
        try:
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        except Exception as e:
            print(f"[transcriber] failed to load model: {e}")
            return
        print(f"[transcriber] ready ({time.time() - t0:.1f}s)")
        self._ready_event.set()

        # Worker loop
        while not self._stop_event.is_set():
            try:
                path = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if path is None:
                break
            self._process_one(path)

    def _process_one(self, wav_path: str) -> None:
        if not os.path.exists(wav_path):
            print(f"[transcriber] file gone: {wav_path}")
            self.jobs_failed += 1
            return

        name = os.path.basename(wav_path)
        print(f"[transcriber] {name}...")
        t0 = time.time()

        try:
            segments_iter, info = self._model.transcribe(
                wav_path,
                beam_size=self._beam_size,
                language="en",
                vad_filter=True,
            )
            # segments_iter is a generator; pulling it triggers actual work.
            segments = [
                {"start": float(s.start), "end": float(s.end),
                 "text": s.text.strip()}
                for s in segments_iter
            ]
        except Exception as e:
            print(f"[transcriber] error on {name}: {e}")
            self.jobs_failed += 1
            return

        elapsed = time.time() - t0
        duration = self._wav_duration(wav_path)
        rtf = elapsed / duration if duration > 0 else 0.0

        # --- write .txt ---
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"[{_format_timestamp(seg['start'])} -> "
                        f"{_format_timestamp(seg['end'])}]  "
                        f"{seg['text']}\n")

        # --- write .json ---
        json_path = os.path.splitext(wav_path)[0] + ".json"
        payload = {
            "audio_file": name,
            "model": self._model_name,
            "duration_seconds": duration,
            "language": info.language,
            "language_probability": info.language_probability,
            "transcription_seconds": elapsed,
            "realtime_factor": rtf,
            "segments": segments,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        text_preview = " ".join(s["text"] for s in segments)[:80]
        print(f"[transcriber] {name} done  "
              f"({elapsed:.1f}s, {rtf:.2f}x rt)  "
              f"\"{text_preview}{'...' if len(text_preview) >= 80 else ''}\"")
        self.jobs_completed += 1

    @staticmethod
    def _wav_duration(path: str) -> float:
        try:
            with wave.open(path, "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0
