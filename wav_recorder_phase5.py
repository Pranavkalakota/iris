"""
Chunked WAV recorder (Phase 5 version).

Difference from Phase 3: calls on_chunk_finalized(path) every time
a chunk WAV file is closed. The main controller uses this hook to
queue chunks for transcription.
"""

import os
import time
import threading
import wave
from datetime import datetime
from typing import Callable, Optional

from ring_buffer import RingBuffer


class WavRecorder(threading.Thread):
    def __init__(self, ring: RingBuffer, sample_rate: int, channels: int,
                 block_samples: int, output_dir: str, chunk_seconds: int,
                 on_chunk_finalized: Optional[Callable[[str], None]] = None):
        super().__init__(daemon=True, name="WavRecorder")
        self._ring = ring
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_samples = block_samples
        self._output_dir = output_dir
        self._chunk_samples = sample_rate * chunk_seconds
        self._on_chunk_finalized = on_chunk_finalized

        self._stop_event = threading.Event()

        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.chunk_index = 0
        self.total_samples_written = 0
        self.files_written: list[str] = []
        self._current_path: Optional[str] = None

    def _open_chunk(self) -> wave.Wave_write:
        self.chunk_index += 1
        filename = (
            f"recording_{self.session_id}_chunk{self.chunk_index:02d}.wav"
        )
        path = os.path.join(self._output_dir, filename)
        wf = wave.open(path, "wb")
        wf.setnchannels(self._channels)
        wf.setsampwidth(2)
        wf.setframerate(self._sample_rate)
        self.files_written.append(path)
        self._current_path = path
        print(f"[recorder] -> {filename}")
        return wf

    def _close_chunk(self, wf: wave.Wave_write) -> None:
        try:
            wf.close()
        except Exception:
            pass
        path = self._current_path
        self._current_path = None
        if path and self._on_chunk_finalized:
            try:
                self._on_chunk_finalized(path)
            except Exception as e:
                print(f"[recorder] callback error: {e}")

    def run(self) -> None:
        wf = self._open_chunk()
        samples_in_chunk = 0
        try:
            while not self._stop_event.is_set():
                available = self._ring.available()
                if available < self._block_samples:
                    time.sleep(self._block_samples / self._sample_rate / 2)
                    continue

                samples = self._ring.read(self._block_samples)
                wf.writeframes(samples.tobytes())
                samples_in_chunk += len(samples)
                self.total_samples_written += len(samples)

                if samples_in_chunk >= self._chunk_samples:
                    self._close_chunk(wf)
                    wf = self._open_chunk()
                    samples_in_chunk = 0
        finally:
            self._close_chunk(wf)

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def duration_seconds(self) -> float:
        return self.total_samples_written / self._sample_rate
