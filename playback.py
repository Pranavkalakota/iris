"""
Plays back a list of WAV files in sequence through the default
audio output. Used by the P key to review recorded chunks.

Runs in its own thread so the main loop can still react to keys
(e.g. R to cancel playback and start a new recording).
"""

import threading
import wave
import time
import numpy as np
import sounddevice as sd


class FilePlayer(threading.Thread):
    def __init__(self, file_paths: list[str], block_samples: int = 512):
        super().__init__(daemon=True, name="FilePlayer")
        self._files = list(file_paths)
        self._block = block_samples
        self._stop_event = threading.Event()

    def run(self) -> None:
        for path in self._files:
            if self._stop_event.is_set():
                break
            try:
                self._play_one(path)
            except Exception as e:
                print(f"[player] error playing {path}: {e}")
        print("[player] done")

    def _play_one(self, path: str) -> None:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()

            if sampwidth != 2:
                print(f"[player] skipping {path}: not 16-bit")
                return

            print(f"[player] playing {path}  "
                  f"({n_frames / sr:.1f}s)")
            stream = sd.OutputStream(
                samplerate=sr,
                channels=channels,
                dtype="int16",
                blocksize=self._block,
            )
            stream.start()
            try:
                while not self._stop_event.is_set():
                    raw = wf.readframes(self._block)
                    if not raw:
                        break
                    samples = np.frombuffer(raw, dtype=np.int16)
                    # Reshape for multi-channel; for mono this is a no-op.
                    if channels > 1:
                        samples = samples.reshape(-1, channels)
                    stream.write(samples)
            finally:
                stream.stop()
                stream.close()

    def stop(self) -> None:
        self._stop_event.set()
