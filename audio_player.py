"""
Wraps a sounddevice.OutputStream. Pulls samples from a RingBuffer
in the audio callback. Handles pre-roll before starting playback.

The callback is a hot path inside PortAudio's native thread:
  - never allocate
  - never block on I/O
  - never raise
"""

import sys
import time
import numpy as np
import sounddevice as sd

from ring_buffer import RingBuffer


class AudioPlayer:
    def __init__(self, ring: RingBuffer, sample_rate: int,
                 block_samples: int, preroll_samples: int,
                 channels: int = 1):
        self._ring = ring
        self._sample_rate = sample_rate
        self._block_samples = block_samples
        self._preroll = preroll_samples
        self._channels = channels
        self._stream: sd.OutputStream | None = None

    def _callback(self, outdata, frames, time_info, status):
        # `status` flags are informational; print but don't abort.
        if status:
            print(f"[AudioPlayer] {status}", file=sys.stderr)
        try:
            samples = self._ring.read(frames)
            # outdata shape is (frames, channels). Assign with slicing
            # to write into the existing buffer (don't rebind outdata).
            outdata[:, 0] = samples
        except Exception as e:
            # If anything goes wrong, write silence and keep the stream alive.
            outdata.fill(0)
            print(f"[AudioPlayer] callback error: {e}", file=sys.stderr)

    def start(self) -> None:
        # Spin until enough audio has accumulated to start cleanly.
        print(f"[AudioPlayer] pre-rolling ({self._preroll} samples)...")
        while self._ring.available() < self._preroll:
            time.sleep(0.01)

        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._block_samples,
            callback=self._callback,
        )
        self._stream.start()
        print("[AudioPlayer] playing")

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
