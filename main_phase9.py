"""
Phase 9 main controller.

Pipeline after a chunk closes:
  1. Whisper transcribes  -> chunk.json
  2. Diarizer tags speakers -> chunk.json (updated), chunk.embeddings.npz
  3. Summarizer summarizes -> chunk.summary.txt

All three run in background threads with independent queues. The GUI
receives events for each stage completing.
"""

import sys
import os
import time
import glob
import queue
import threading
import numpy as np

import config_phase9 as config
from ring_buffer import RingBuffer
from wifi_reader_phase6 import WifiReader
from audio_player import AudioPlayer
from wav_recorder_phase5 import WavRecorder
from playback import FilePlayer
from transcriber_phase5 import Transcriber
from diarizer_phase9 import Diarizer
from summarizer_phase9 import Summarizer
from speakers_phase9 import SpeakerDB
from location_phase8 import LocationService, save_location_sidecar
from gui_phase9 import AudioStreamWindow


class Controller:
    def __init__(self):
        self.ring = RingBuffer(config.RING_CAPACITY)
        self.reader = WifiReader(
            ring=self.ring,
            stream_port=config.STREAM_PORT,
            discovery_port=config.DISCOVERY_PORT,
            discovery_message=config.DISCOVERY_MESSAGE,
            discovery_interval_s=config.DISCOVERY_INTERVAL_S,
            seq_header_bytes=config.SEQ_HEADER_BYTES,
            samples_per_packet=config.SAMPLES_PER_PACKET,
        )

        self.event_queue: "queue.Queue[dict]" = queue.Queue()

        self.speaker_db = SpeakerDB(
            path=config.SPEAKERS_DB_PATH,
            max_embeddings_per_profile=config.MAX_EMBEDDINGS_PER_PROFILE,
        )

        self.transcriber = _NotifyingTranscriber(
            model_name=config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
            beam_size=config.WHISPER_BEAM_SIZE,
        )
        self.transcriber._event_queue = self.event_queue
        self.transcriber._on_done_callback = self._on_transcribe_done

        self.diarizer = Diarizer(
            speaker_db=self.speaker_db,
            strict_thresh=config.MATCH_STRICT_THRESH,
            weak_thresh=config.MATCH_WEAK_THRESH,
            event_queue=self.event_queue,
        )

        self.summarizer = Summarizer(
            ollama_url=config.OLLAMA_URL,
            model=config.OLLAMA_MODEL,
            timeout_s=config.OLLAMA_TIMEOUT_S,
            event_queue=self.event_queue,
        )

        self.location_service = LocationService(
            timeout_s=config.LOCATION_TIMEOUT_S)

        self.monitor = None
        self.recorder = None
        self.file_player = None
        self.startup_location = None

        self.mic = None
        self._transcribe_only: set = set()
        self._wake_listener = None
        # --- IRIS wake-word: ADD ---
        # Option B: a secondary SpeechRecognition-based listener runs in
        # parallel with openwakeword. Both fire the same on_wake callback,
        # coalesced by _on_wake_debounced so a double-fire within 2s only
        # produces one downstream trigger.
        self._wake_fallback = None
        self._wake_debounce_ts = 0.0
        self._wake_debounce_lock = threading.Lock()
        # --- IRIS wake-word: END ---

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._client_was_seen = False

    def start(self):
        self.reader.start()
        self.transcriber.start()
        self.diarizer.start()
        self.summarizer.start()
        threading.Thread(target=self._stats_loop,
                         daemon=True, name="StatsThread").start()
        threading.Thread(target=self._fetch_location_async,
                         daemon=True, name="LocationFetch").start()

    def shutdown(self):
        self._stop.set()
        self.stop_mic_capture()
        self.stop_wake_listener()
        self._stop_recording()
        self._stop_monitoring()
        self._stop_playback()
        self.reader.stop()
        self.transcriber.stop()
        self.diarizer.stop()
        self.summarizer.stop()
        for t in [self.transcriber, self.diarizer, self.summarizer]:
            try: t.join(timeout=2.0)
            except Exception: pass

    def _fetch_location_async(self):
        loc = self.location_service.get()
        self.startup_location = loc
        self._post({"type": "location_ready", "location": loc})

    def _on_transcribe_done(self, wav_path: str):
        if wav_path in self._transcribe_only:
            self._transcribe_only.discard(wav_path)
            return
        if config.AUTO_DIARIZE:
            self.diarizer.submit(wav_path)
            self._post({"type": "diarize_queue",
                        "depth": self.diarizer.queue_depth})

    def _on_diarize_done(self, wav_path: str):
        if config.AUTO_SUMMARIZE:
            self.summarizer.submit(wav_path)
            self._post({"type": "summarize_queue",
                        "depth": self.summarizer.queue_depth})

    def toggle_recording(self):
        with self._lock:
            if self.recorder is None:
                self._stop_playback()
                self._start_recording()
            else:
                self._stop_recording()

    def _start_recording(self):
        self.recorder = WavRecorder(
            ring=self.ring,
            sample_rate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            block_samples=config.BLOCK_SAMPLES,
            output_dir=config.RECORDINGS_DIR,
            chunk_seconds=config.CHUNK_SECONDS,
            on_chunk_finalized=self._on_chunk_done,
        )
        self.ring.read(self.ring.available())
        self.recorder.start()
        self._post({"type": "recording_started",
                    "session": self.recorder.session_id})

    def _stop_recording(self):
        if self.recorder is None:
            return
        self.recorder.stop()
        self.recorder.join(timeout=2.0)
        dur = self.recorder.duration_seconds
        n   = self.recorder.chunk_index
        self._post({"type": "recording_stopped", "duration": dur, "chunks": n})
        self.recorder = None

    def _on_chunk_done(self, wav_path: str):
        if self.startup_location:
            save_location_sidecar(wav_path, self.startup_location)
        self.transcriber.submit(wav_path)
        self._post({"type": "chunk_finalized", "path": wav_path})

    def toggle_monitoring(self):
        with self._lock:
            if self.monitor is None:
                self._start_monitoring()
            else:
                self._stop_monitoring()

    def _start_monitoring(self):
        self.monitor = AudioPlayer(
            ring=self.ring,
            sample_rate=config.SAMPLE_RATE,
            block_samples=config.BLOCK_SAMPLES,
            preroll_samples=config.PREROLL_SAMPLES,
            channels=config.CHANNELS,
        )
        def _bg():
            try:
                self.monitor.start()
                self._post({"type": "monitor_started"})
            except Exception as e:
                self.monitor = None
                self._post({"type": "monitor_failed", "error": str(e)})
        threading.Thread(target=_bg, daemon=True).start()

    def _stop_monitoring(self):
        if self.monitor is None: return
        self.monitor.stop()
        self.monitor = None
        self._post({"type": "monitor_stopped"})

    def play_file(self, path: str):
        with self._lock:
            self._stop_playback()
            self.file_player = FilePlayer([path],
                                          block_samples=config.BLOCK_SAMPLES)
            self.file_player.start()

    def transcribe_file(self, path: str):
        self.transcriber.submit(path)

    def transcribe_file_only(self, path: str):
        self._transcribe_only.add(path)
        self.transcriber.submit(path)

    def diarize_file(self, path: str):
        self.diarizer.submit(path)

    def summarize_file(self, path: str):
        self.summarizer.submit(path)

    def _stop_playback(self):
        if self.file_player is None: return
        self.file_player.stop()
        self.file_player.join(timeout=1.0)
        self.file_player = None

    def start_mic_capture(self) -> bool:
        use_pc_mic = getattr(config, "USE_PC_MIC", False)
        if not use_pc_mic:
            print("[iris] mic capture: using ESP32 ring buffer (USE_PC_MIC=False)")
            return True

        with self._lock:
            if self.mic is not None:
                return True
            try:
                mic = _MicCapture(config.SAMPLE_RATE, seconds=30.0)
                mic.start()
                self.mic = mic
                print(f"[iris] mic capture started @ {mic.sample_rate} Hz")
                return True
            except Exception as e:
                print(f"[iris] mic capture failed to start: {e}")
                self.mic = None
                return False

    def stop_mic_capture(self) -> None:
        with self._lock:
            if self.mic is not None:
                try:
                    self.mic.stop()
                except Exception:
                    pass
                self.mic = None

    # --- IRIS wake-word: CHANGE ---
    # Option B: openwakeword remains the primary detector; a Google
    # SpeechRecognition listener runs in parallel as a backup. Both fire
    # _on_wake_debounced, which coalesces double-fires within 2s so
    # downstream sees exactly one trigger per utterance.
    def _on_wake_debounced(self, on_wake, phrase: str) -> None:
        with self._wake_debounce_lock:
            now = time.time()
            if now - self._wake_debounce_ts < 2.0:
                return
            self._wake_debounce_ts = now
        try:
            on_wake(phrase)
        except Exception:
            pass

    def start_wake_listener(self, on_wake) -> bool:
        with self._lock:
            if self._wake_listener is not None:
                return True
            debounced = lambda phrase: self._on_wake_debounced(on_wake, phrase)
            try:
                use_pc_mic = getattr(config, "USE_PC_MIC", False)
                if use_pc_mic:
                    listener = _WakeWordListener(on_wake=debounced)
                else:
                    listener = _WakeWordListener(
                        on_wake=debounced,
                        ring=self.ring,
                        ring_sample_rate=config.SAMPLE_RATE,
                    )
                listener.start()
                self._wake_listener = listener
                print("[iris] wake word listener started (openwakeword primary)")
            except Exception as e:
                print(f"[iris] openwakeword listener failed to start: {e}")
                self._wake_listener = None
                return False
            # Option B backup: SpeechRecognition + Google. Off by default
            # if USE_GOOGLE_WAKE_BACKUP is False in config_phase9.py; on
            # otherwise. Failures here don't affect the primary path.
            try:
                if getattr(config, "USE_GOOGLE_WAKE_BACKUP", True):
                    fallback = _SpeechRecognitionWakeListener(
                        on_wake=debounced,
                        ring=self.ring,
                        ring_sample_rate=config.SAMPLE_RATE,
                    )
                    fallback.start()
                    self._wake_fallback = fallback
                    print("[iris] google speech backup wake listener started")
            except Exception as e:
                print(f"[iris] backup wake listener disabled: {e}")
                self._wake_fallback = None
            return True

    def stop_wake_listener(self) -> None:
        with self._lock:
            if self._wake_listener is not None:
                try:
                    self._wake_listener.stop()
                except Exception:
                    pass
                self._wake_listener = None
            if self._wake_fallback is not None:
                try:
                    self._wake_fallback.stop()
                except Exception:
                    pass
                self._wake_fallback = None
    # --- IRIS wake-word: END ---

    def peek_level(self) -> float:
        mic = self.mic
        if mic is not None:
            try:
                return mic.level()
            except Exception:
                pass
        n = 256
        with self.ring._lock:
            if self.ring._count < n:
                return 0.0
            start = (self.ring._write_idx - n) % self.ring._capacity
            if start + n <= self.ring._capacity:
                samples = self.ring._buf[start:start + n].copy()
            else:
                wrap = self.ring._capacity - start
                samples = np.concatenate([
                    self.ring._buf[start:],
                    self.ring._buf[:n - wrap]
                ])
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        return min(1.0, rms / 3000.0)

    def peek_audio_wav(self, seconds: float, dest_path: str) -> bool:
        mic = self.mic
        if mic is not None:
            try:
                return mic.peek_wav(seconds, dest_path)
            except Exception as e:
                print(f"[iris] mic peek_wav failed: {e}")
                return False
        try:
            n = int(seconds * config.SAMPLE_RATE)
            if n <= 0:
                return False
            with self.ring._lock:
                avail = self.ring._count
                cap = self.ring._capacity
                requested = n
                n = min(n, avail)
                print(f"[iris] peek: requested={requested} avail={avail} "
                      f"cap={cap} sr={config.SAMPLE_RATE} -> using={n}")
                if n <= 0:
                    return False
                start = (self.ring._write_idx - n) % self.ring._capacity
                if start + n <= self.ring._capacity:
                    samples = self.ring._buf[start:start + n].copy()
                else:
                    wrap = self.ring._capacity - start
                    samples = np.concatenate([
                        self.ring._buf[start:],
                        self.ring._buf[:n - wrap],
                    ])
            samples = samples.astype(np.float32)

            _raw_rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
            _raw_peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            print(f"[iris] ESP32 audio window: rms={_raw_rms:.0f} peak={_raw_peak:.0f} "
                  f"n={samples.size}")

            gain = float(getattr(config, "LIVE_AUDIO_GAIN", 0.0))
            if gain > 0.0:
                samples = samples * gain
            else:
                peak = float(np.max(np.abs(samples))) if samples.size else 0.0
                if peak > 1.0:
                    target_peak = 0.6 * 32767.0
                    auto_gain = min(8.0, target_peak / peak)
                    samples = samples * auto_gain

            samples = np.clip(samples, -32768, 32767).astype(np.int16)

            import wave as _wave
            with _wave.open(dest_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(config.SAMPLE_RATE)
                wf.writeframes(samples.tobytes())
            return True
        except Exception as e:
            print(f"[iris] peek_audio_wav failed: {e}")
            return False

    def _stats_loop(self):
        last_chunk = -1; last_dur = -1.0
        while not self._stop.is_set():
            time.sleep(0.25)
            if self.reader.client_seen and not self._client_was_seen:
                self._client_was_seen = True
                self._post({"type": "esp32_connected"})
            if self.recorder is not None:
                dur = self.recorder.duration_seconds
                chunk = self.recorder.chunk_index
                if int(dur) != int(last_dur) or chunk != last_chunk:
                    self._post({"type": "recording_tick",
                                "duration": dur, "chunk": chunk})
                    last_dur, last_chunk = dur, chunk
            self._post({"type": "transcribe_queue",
                        "depth": self.transcriber.queue_depth})
            self._post({"type": "diarize_queue",
                        "depth": self.diarizer.queue_depth})
            self._post({"type": "summarize_queue",
                        "depth": self.summarizer.queue_depth})
            total = self.reader.packets_received + self.reader.packets_lost
            if total > 0:
                self._post({"type": "net_stats",
                            "loss_pct": 100.0 * self.reader.packets_lost / total})

    def _post(self, evt: dict):
        try:
            self.event_queue.put_nowait(evt)
        except queue.Full:
            pass


class _NotifyingTranscriber(Transcriber):
    _event_queue = None
    _on_done_callback = None

    def _process_one(self, wav_path: str):
        super()._process_one(wav_path)
        if self._event_queue is not None:
            try:
                self._event_queue.put_nowait(
                    {"type": "transcribe_done", "wav": wav_path})
            except queue.Full:
                pass
        if self._on_done_callback is not None:
            try:
                self._on_done_callback(wav_path)
            except Exception:
                pass


class _MicCapture:
    def __init__(self, target_sr: int, seconds: float = 30.0):
        self._req_sr = int(target_sr) if target_sr else 16000
        self._seconds = float(seconds)
        self._sr = self._req_sr
        self._device = None
        self._buf = None
        self._cap = 0
        self._widx = 0
        self._count = 0
        self._lock = threading.Lock()
        self._stream = None

    @property
    def sample_rate(self) -> int:
        return self._sr

    def _list_input_devices(self, sd) -> list:
        devices = []
        try:
            all_devs = sd.query_devices()
        except Exception as e:
            print(f"[iris] sd.query_devices failed: {e}")
            return devices
        for i, d in enumerate(all_devs):
            try:
                if int(d.get("max_input_channels", 0)) > 0:
                    devices.append((
                        i,
                        d.get("name", f"device {i}"),
                        int(round(float(d.get("default_samplerate", 0) or 0))),
                    ))
            except Exception:
                continue
        print(f"[iris] available input devices ({len(devices)} found):")
        for idx, name, sr in devices:
            print(f"       [{idx}] {name}  (default {sr} Hz)")
        return devices

    def _pick_device_and_rate(self, sd):
        rates_to_try = [44100, 48000, self._req_sr, 32000, 16000]
        seen = set()
        rates_to_try = [r for r in rates_to_try if not (r in seen or seen.add(r))]

        api_priority = ["MME", "DirectSound", "WASAPI"]
        try:
            host_apis = sd.query_hostapis()
            all_devs = list(enumerate(sd.query_devices()))
        except Exception as e:
            print(f"[iris] device enumeration failed: {e}")
            return None, None, 1

        def api_rank(hostapi_idx):
            name = host_apis[hostapi_idx]["name"] if hostapi_idx < len(host_apis) else ""
            for rank, api in enumerate(api_priority):
                if api in name:
                    return rank
            return len(api_priority)

        input_devs = [
            (i, d) for i, d in all_devs if d["max_input_channels"] > 0
        ]
        input_devs.sort(key=lambda x: (
            api_rank(x[1]["hostapi"]),
            0 if "Microphone" in x[1]["name"] else 1,
        ))

        print(f"[iris] input device search order:")
        for i, d in input_devs:
            api_name = host_apis[d["hostapi"]]["name"] if d["hostapi"] < len(host_apis) else "?"
            print(f"       [{i}] {d['name']} ({api_name})")

        for i, d in input_devs:
            for ch in (2, 1):
                for r in rates_to_try:
                    try:
                        sd.check_input_settings(device=i, channels=ch,
                                                samplerate=r, dtype="int16")
                        api_name = host_apis[d["hostapi"]]["name"]
                        print(f"[iris] selected [{i}] {d['name']} ({api_name}) @ {r} Hz ch={ch}")
                        return i, r, ch
                    except Exception:
                        continue

        return None, None, 1

    def start(self) -> None:
        import sounddevice as sd
        device, sr, channels = self._pick_device_and_rate(sd)
        if device is None or sr is None:
            raise RuntimeError(
                "no input device accepted any sample rate - check Windows Sound "
                "Settings > Privacy > Microphone (must be ON) and that the device "
                "is not in exclusive mode. Devices printed above.")
        self._sr = sr
        self._channels = channels
        self._device = device
        self._cap = max(1, int(sr * self._seconds))
        self._buf = np.zeros(self._cap, dtype=np.int16)
        self._widx = 0
        self._count = 0
        self._stream = sd.InputStream(
            device=device,
            samplerate=sr,
            channels=channels,
            dtype="int16",
            blocksize=2048,
            callback=self._callback,
        )
        self._stream.start()
        print(f"[iris] stream opened: device={device} sr={sr} ch={channels} blocksize=2048")

    def _callback(self, indata, frames, time_info, status):
        try:
            if status:
                print(f"[mic] cb status={status}")
            if indata.ndim > 1 and indata.shape[1] > 1:
                mono = indata.mean(axis=1).astype(np.int16)
            elif indata.ndim > 1:
                mono = indata[:, 0]
            else:
                mono = indata.reshape(-1)
            n = len(mono)
            if n == 0:
                return
            with self._lock:
                if self._buf is None:
                    return
                if n >= self._cap:
                    self._buf[:] = mono[-self._cap:]
                    self._widx = 0
                    self._count = self._cap
                    return
                end = self._widx + n
                if end <= self._cap:
                    self._buf[self._widx:end] = mono
                else:
                    first = self._cap - self._widx
                    self._buf[self._widx:] = mono[:first]
                    self._buf[:n - first] = mono[first:]
                self._widx = (self._widx + n) % self._cap
                self._count = min(self._cap, self._count + n)
        except Exception:
            pass

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _recent(self, n: int):
        with self._lock:
            if self._buf is None or self._count < n or n <= 0:
                return None
            start = (self._widx - n) % self._cap
            if start + n <= self._cap:
                return self._buf[start:start + n].copy()
            wrap = self._cap - start
            return np.concatenate([self._buf[start:], self._buf[:n - wrap]])

    def level(self) -> float:
        s = self._recent(256)
        if s is None:
            return 0.0
        rms = float(np.sqrt(np.mean(s.astype(np.float32) ** 2)))
        return min(1.0, rms / 3000.0)

    def peek_wav(self, seconds: float, dest_path: str) -> bool:
        n = int(seconds * self._sr)
        with self._lock:
            avail = self._count
        n = min(n, avail)
        if n <= 0:
            return False
        samples = self._recent(n)
        if samples is None:
            return False
        samples = samples.astype(np.float32)
        if self._sr != 16000:
            target_len = max(1, int(round(len(samples) * 16000 / self._sr)))
            idx = np.linspace(0, len(samples) - 1, target_len)
            samples = np.interp(idx, np.arange(len(samples)), samples)
        import wave as _wave
        with _wave.open(dest_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(
                np.clip(samples, -32768, 32767).astype(np.int16).tobytes())
        return True


class _WakeWordListener(threading.Thread):
    SAMPLE_RATE   = 16000
    CHUNK_SAMPLES = 1280
    THRESHOLD     = 0.5
    INPUT_DEVICE_INDEX = 10

    def __init__(self, on_wake, ring=None, ring_sample_rate=16000):
        super().__init__(daemon=True, name="WakeWordListener")
        self._on_wake = on_wake
        self._stop_event = threading.Event()
        self._ring = ring
        self._ring_sr = int(ring_sample_rate) if ring_sample_rate else 16000

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            from openwakeword.model import Model
        except ImportError as e:
            print(f"[wake] openwakeword not installed: {e}\n"
                  f"       Run: pip install openwakeword onnxruntime")
            return

        try:
            oww = Model(wakeword_models=["hey_jarvis"],
                        inference_framework="onnx")
        except Exception as e:
            print(f"[wake] failed to load hey_jarvis model: {e}")
            return

        if self._ring is not None:
            self._run_from_ring(oww)
            return

        try:
            import pyaudio
        except ImportError as e:
            print(f"[wake] pyaudio not installed: {e}\n"
                  f"       Run: pip install pyaudio")
            return

        pa = pyaudio.PyAudio()
        stream = None
        actual_rate = None
        chosen_device = None

        devices = []
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    devices.append((
                        i,
                        info.get("name", f"device {i}"),
                        int(round(float(info.get("defaultSampleRate", 0)))),
                    ))
        except Exception as e:
            print(f"[wake] could not enumerate input devices: {e}")

        if devices:
            print(f"[wake] available input devices ({len(devices)} found):")
            for idx, name, sr in devices:
                print(f"       [{idx}] {name}  (default {sr} Hz)")

        if self.INPUT_DEVICE_INDEX is not None:
            attempts = [(self.INPUT_DEVICE_INDEX, 44100),
                        (self.INPUT_DEVICE_INDEX, 48000),
                        (self.INPUT_DEVICE_INDEX, 32000),
                        (self.INPUT_DEVICE_INDEX, 16000),
                        (self.INPUT_DEVICE_INDEX, 22050)]
        else:
            attempts = [(None, self.SAMPLE_RATE), (None, 44100), (None, 48000)]
            for idx, _name, dsr in devices:
                rates = [self.SAMPLE_RATE, 48000, 44100, 32000, 16000]
                if dsr and dsr not in rates:
                    rates.insert(0, dsr)
                for r in rates:
                    attempts.append((idx, r))

        for dev_idx, rate in attempts:
            try:
                kwargs = dict(rate=rate, channels=1,
                              format=pyaudio.paInt16, input=True,
                              frames_per_buffer=self.CHUNK_SAMPLES)
                if dev_idx is not None:
                    kwargs["input_device_index"] = dev_idx
                stream = pa.open(**kwargs)
                actual_rate = rate
                chosen_device = dev_idx
                break
            except Exception:
                stream = None
                continue

        if stream is None or actual_rate is None:
            print("[wake] no usable input device - wake listener inactive")
            pa.terminate()
            return

        dev_label = f"device {chosen_device}" if chosen_device is not None else "default"
        print(f"[wake] listening for 'hey jarvis' on {dev_label} @ {actual_rate} Hz")
        try:
            while not self._stop_event.is_set():
                try:
                    raw = stream.read(self.CHUNK_SAMPLES,
                                      exception_on_overflow=False)
                except Exception:
                    continue
                audio = np.frombuffer(raw, dtype=np.int16)
                if actual_rate != self.SAMPLE_RATE:
                    target_len = int(round(
                        len(audio) * self.SAMPLE_RATE / actual_rate))
                    idx = np.linspace(0, len(audio) - 1, target_len)
                    audio = np.interp(
                        idx, np.arange(len(audio)),
                        audio.astype(np.float32)).astype(np.int16)
                try:
                    oww.predict(audio)
                except Exception:
                    continue
                scores = oww.prediction_buffer.get("hey_jarvis", [])
                if scores and float(scores[-1]) >= self.THRESHOLD:
                    print(f"[wake] 'hey jarvis' detected "
                          f"(score={float(scores[-1]):.2f})")
                    try:
                        self._on_wake("hey jarvis")
                    except Exception:
                        pass
                    self._stop_event.wait(timeout=2.0)
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            pa.terminate()
            print("[wake] listener stopped")

    def _run_from_ring(self, oww) -> None:
        ring = self._ring
        src_sr = self._ring_sr
        src_chunk = max(1, int(round(self.CHUNK_SAMPLES * src_sr / self.SAMPLE_RATE)))

        print(f"[wake] listening for 'hey jarvis' on ESP32 ring buffer "
              f"@ {src_sr} Hz (chunk={src_chunk})")

        def _grab(n):
            with ring._lock:
                if ring._count < n:
                    return None
                start = (ring._write_idx - n) % ring._capacity
                if start + n <= ring._capacity:
                    return ring._buf[start:start + n].copy()
                wrap = ring._capacity - start
                return np.concatenate([ring._buf[start:], ring._buf[:n - wrap]])

        while not self._stop_event.is_set():
            chunk = _grab(src_chunk)
            if chunk is None:
                self._stop_event.wait(timeout=0.02)
                continue

            audio = chunk.astype(np.float32)
            if src_sr != self.SAMPLE_RATE:
                target_len = max(1, int(round(
                    len(audio) * self.SAMPLE_RATE / src_sr)))
                idx = np.linspace(0, len(audio) - 1, target_len)
                audio = np.interp(idx, np.arange(len(audio)), audio)
            audio = audio.astype(np.int16)

            try:
                oww.predict(audio)
            except Exception:
                self._stop_event.wait(timeout=0.02)
                continue

            scores = oww.prediction_buffer.get("hey_jarvis", [])
            if scores and float(scores[-1]) >= self.THRESHOLD:
                print(f"[wake] 'hey jarvis' detected "
                      f"(score={float(scores[-1]):.2f})")
                try:
                    self._on_wake("hey jarvis")
                except Exception:
                    pass
                self._stop_event.wait(timeout=2.0)
            else:
                self._stop_event.wait(
                    timeout=self.CHUNK_SAMPLES / self.SAMPLE_RATE * 0.5)

        print("[wake] ring listener stopped")


# --- IRIS wake-word: ADD ---
class _SpeechRecognitionWakeListener(threading.Thread):
    """Google-speech-based wake listener that runs in PARALLEL with the
    openwakeword listener (Option B). Reads 3-second WAV chunks from the
    ESP32 ring buffer, sends them to Google's free speech API, and looks
    for 'hey jarvis' / 'hi jarvis' / 'okay jarvis' / 'jarvis' in the
    returned text.

    This is the safety net: if openwakeword misclassifies (a whisper,
    thick accent, unusually noisy room), Google's ASR often still
    catches the phrase. Controller._on_wake_debounced collapses
    double-fires from both listeners into one downstream trigger.

    Requires: pip install SpeechRecognition
    Fails gracefully (prints a note, exits the thread) if the library or
    network is unavailable.
    """
    SAMPLE_RATE = 16000
    CHUNK_SECONDS = 3.0
    WAKE_PHRASES = ("hey jarvis", "hi jarvis", "okay jarvis",
                     "hey jervis", "jarvis")

    def __init__(self, on_wake, ring=None, ring_sample_rate=16000):
        super().__init__(daemon=True, name="SRWakeListener")
        self._on_wake = on_wake
        self._stop_event = threading.Event()
        self._ring = ring
        self._ring_sr = int(ring_sample_rate) if ring_sample_rate else 16000

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError as e:
            print(f"[wake-backup] SpeechRecognition not installed: {e}\n"
                  f"              Run: pip install SpeechRecognition")
            return

        if self._ring is None:
            print("[wake-backup] no ring buffer supplied; backup exits")
            return

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True

        print(f"[wake-backup] listening for 'hey jarvis' via Google speech "
              f"(3s windows, ring @ {self._ring_sr} Hz)")

        import io as _io
        import wave as _wave

        while not self._stop_event.is_set():
            samples = self._grab_seconds(self.CHUNK_SECONDS)
            if samples is None:
                self._stop_event.wait(timeout=0.5)
                continue
            audio = samples.astype(np.float32)
            if self._ring_sr != self.SAMPLE_RATE:
                target_len = max(1, int(round(
                    len(audio) * self.SAMPLE_RATE / self._ring_sr)))
                idx = np.linspace(0, len(audio) - 1, target_len)
                audio = np.interp(idx, np.arange(len(audio)), audio)
            pcm = np.clip(audio, -32768, 32767).astype(np.int16)

            buf = _io.BytesIO()
            with _wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())
            buf.seek(0)

            try:
                with sr.AudioFile(buf) as source:
                    audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data).lower()
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"[wake-backup] google API error: {e}; backing off 5s")
                self._stop_event.wait(timeout=5.0)
                continue
            except Exception:
                continue

            if any(p in text for p in self.WAKE_PHRASES):
                print(f"[wake-backup] 'hey jarvis' heard (via google): {text!r}")
                try:
                    self._on_wake("hey jarvis")
                except Exception:
                    pass
                self._stop_event.wait(timeout=2.0)
            else:
                self._stop_event.wait(timeout=0.3)

        print("[wake-backup] listener stopped")

    def _grab_seconds(self, seconds: float):
        n = int(seconds * self._ring_sr)
        if n <= 0:
            return None
        ring = self._ring
        with ring._lock:
            if ring._count < n:
                return None
            start = (ring._write_idx - n) % ring._capacity
            if start + n <= ring._capacity:
                return ring._buf[start:start + n].copy()
            wrap = ring._capacity - start
            return np.concatenate([ring._buf[start:], ring._buf[:n - wrap]])
# --- IRIS wake-word: END ---


def main() -> int:
    ctrl = Controller()
    ctrl.start()
    gui = AudioStreamWindow(ctrl, config)
    try:
        gui.mainloop()
    finally:
        ctrl.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())