"""
Thread-safe bounded ring buffer for int16 audio samples.

Producer (SerialReader thread) calls write().
Consumer (PortAudio callback) calls read().

Policies:
  - Overflow: drop oldest data (latency matters more than completeness).
  - Underflow: pad with silence (callback must never return short).
"""

import threading
import numpy as np


class RingBuffer:
    def __init__(self, capacity_samples: int):
        self._capacity = int(capacity_samples)
        self._buf = np.zeros(self._capacity, dtype=np.int16)
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0
        self._lock = threading.Lock()

        # Diagnostics
        self.underrun_count = 0
        self.overrun_count = 0

    def write(self, samples: np.ndarray) -> None:
        """Append samples. Drops oldest if buffer would overflow."""
        n = len(samples)
        if n == 0:
            return
        if n > self._capacity:
            # More data than the whole buffer; keep only the newest tail.
            samples = samples[-self._capacity:]
            n = self._capacity

        with self._lock:
            # If this write would overflow, advance read_idx (drop oldest).
            free = self._capacity - self._count
            if n > free:
                drop = n - free
                self._read_idx = (self._read_idx + drop) % self._capacity
                self._count -= drop
                self.overrun_count += drop

            # Write in up to two segments (wrap-around).
            end_space = self._capacity - self._write_idx
            if n <= end_space:
                self._buf[self._write_idx:self._write_idx + n] = samples
            else:
                self._buf[self._write_idx:] = samples[:end_space]
                self._buf[:n - end_space] = samples[end_space:]
            self._write_idx = (self._write_idx + n) % self._capacity
            self._count += n

    def read(self, n: int) -> np.ndarray:
        """Return exactly n samples. Pads with zeros on underrun."""
        out = np.zeros(n, dtype=np.int16)
        with self._lock:
            available = self._count
            take = min(n, available)
            if take < n:
                self.underrun_count += (n - take)

            if take > 0:
                end_space = self._capacity - self._read_idx
                if take <= end_space:
                    out[:take] = self._buf[self._read_idx:self._read_idx + take]
                else:
                    out[:end_space] = self._buf[self._read_idx:]
                    out[end_space:take] = self._buf[:take - end_space]
                self._read_idx = (self._read_idx + take) % self._capacity
                self._count -= take
        return out

    def available(self) -> int:
        with self._lock:
            return self._count

    def stats(self) -> dict:
        with self._lock:
            return {
                "fill": self._count,
                "capacity": self._capacity,
                "underruns": self.underrun_count,
                "overruns": self.overrun_count,
            }
