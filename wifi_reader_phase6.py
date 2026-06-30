"""
Wi-Fi UDP audio reader.

Replaces SerialReader for Phase 6. Two cooperating threads:

  1. DiscoveryAnnouncer - broadcasts our presence so the ESP32 can find us.
     Runs every DISCOVERY_INTERVAL_S forever (continues during streaming
     so the ESP32 can re-acquire if our IP changes).

  2. PacketReceiver - blocks on the UDP stream socket, validates sequence
     numbers, pads with silence on packet loss, pushes samples into the
     ring buffer.

From the rest of the system's point of view, this drops in as a
replacement for SerialReader: same start()/stop() interface, same
RingBuffer output.
"""

import socket
import struct
import threading
import time
import numpy as np

from ring_buffer import RingBuffer


class WifiReader:
    """Public facade. Owns the two underlying threads."""

    def __init__(self, ring: RingBuffer,
                 stream_port: int, discovery_port: int,
                 discovery_message: bytes, discovery_interval_s: float,
                 seq_header_bytes: int, samples_per_packet: int):
        self._ring = ring
        self._stream_port = stream_port
        self._discovery_port = discovery_port
        self._discovery_message = discovery_message
        self._discovery_interval = discovery_interval_s
        self._seq_header_bytes = seq_header_bytes
        self._samples_per_packet = samples_per_packet

        self._stop_event = threading.Event()
        self._announcer = threading.Thread(
            target=self._announcer_loop, daemon=True, name="DiscoveryAnnouncer")
        self._receiver = threading.Thread(
            target=self._receiver_loop, daemon=True, name="PacketReceiver")

        # Diagnostic counters
        self.packets_received = 0
        self.packets_lost = 0          # gaps detected in sequence
        self.packets_out_of_order = 0  # arrived late, ignored
        self.bytes_received = 0
        self.client_seen = False       # have we ever received a packet?
        self._expected_seq: int | None = None

    # ---------- public API ----------
    def start(self) -> None:
        self._announcer.start()
        self._receiver.start()

    def stop(self) -> None:
        self._stop_event.set()

    def is_alive(self) -> bool:
        return self._receiver.is_alive()

    # ---------- discovery announcer ----------
    def _announcer_loop(self) -> None:
        # Send UDP broadcast on every interface every DISCOVERY_INTERVAL_S.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            while not self._stop_event.is_set():
                try:
                    sock.sendto(self._discovery_message,
                                ("255.255.255.255", self._discovery_port))
                except OSError:
                    pass
                # Sleep in small chunks so stop() is responsive
                t_end = time.time() + self._discovery_interval
                while time.time() < t_end and not self._stop_event.is_set():
                    time.sleep(0.1)
        finally:
            sock.close()

    # ---------- packet receiver ----------
    def _receiver_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Generous OS-side receive buffer (default is tiny on Windows)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        sock.bind(("", self._stream_port))
        sock.settimeout(0.5)

        try:
            while not self._stop_event.is_set():
                try:
                    data, _addr = sock.recvfrom(2048)
                except socket.timeout:
                    continue
                except OSError:
                    break

                self._handle_packet(data)
        finally:
            sock.close()

    def _handle_packet(self, data: bytes) -> None:
        if len(data) < self._seq_header_bytes + 2:
            return  # malformed, too small to contain even one sample

        # Parse header
        (seq,) = struct.unpack("<I",
                               data[:self._seq_header_bytes])
        payload = data[self._seq_header_bytes:]

        # Trim odd byte (shouldn't happen but cheap insurance)
        if len(payload) & 1:
            payload = payload[:-1]
        if not payload:
            return

        samples = np.frombuffer(payload, dtype="<i2")
        self.bytes_received += len(payload)
        self.client_seen = True

        # Sequence handling
        if self._expected_seq is None:
            # First packet: lock on to whatever sequence number arrived.
            self._expected_seq = seq
        elif seq == self._expected_seq:
            pass  # ideal case
        elif self._is_in_future(seq, self._expected_seq):
            # Packet(s) lost. Pad with silence to maintain timing.
            gap = self._distance(self._expected_seq, seq)
            self.packets_lost += gap
            silence = np.zeros(gap * self._samples_per_packet, dtype=np.int16)
            self._ring.write(silence)
        else:
            # Out-of-order / late packet. Drop it; don't go back in time.
            self.packets_out_of_order += 1
            return

        self._ring.write(samples)
        self._expected_seq = (seq + 1) & 0xFFFFFFFF
        self.packets_received += 1

    @staticmethod
    def _is_in_future(seq: int, expected: int) -> bool:
        # Treat half the 32-bit space as "future", rest as "past".
        # Handles wrap-around at 2^32 cleanly.
        return ((seq - expected) & 0xFFFFFFFF) < (1 << 31)

    @staticmethod
    def _distance(from_seq: int, to_seq: int) -> int:
        return (to_seq - from_seq) & 0xFFFFFFFF
