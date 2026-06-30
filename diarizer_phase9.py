"""
Diarizer for Phase 9 — SpeechBrain edition.

Replaces resemblyzer with SpeechBrain's ECAPA-TDNN speaker embedding model.
No C compiler required. Installs cleanly on Python 3.14 with:

    pip install speechbrain

The model (~100 MB) downloads automatically from HuggingFace on first use
and is cached forever. All processing is local — no API calls, no cost.

Behavior is identical to the original resemblyzer version from the outside:
  - Reads a WAV + its .json transcript
  - Computes a voice embedding for each segment
  - Clusters segments by speaker using agglomerative clustering
  - Matches clusters against the known-speaker database
  - Updates the .json with speaker tags + confidence scores
  - Saves a .embeddings.npz sidecar for later GUI relabeling
  - Fires event_queue event when done so the GUI can refresh
"""

import json
import os
import queue
import threading
import wave
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Windows symlink fix.
# SpeechBrain tries to symlink its model files from the HuggingFace hub cache
# into its own cache directory. Windows blocks symlink creation for normal
# users (WinError 1314). Setting these two env vars before importing
# SpeechBrain tells HuggingFace to copy files instead of symlinking, and
# points SpeechBrain's cache to a plain local folder that needs no privileges.
# These must be set before any speechbrain import, so we do it at module load.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE",
                      os.path.join("C:\\", "audio_stream", "hf_cache"))
_SB_CACHE = os.path.join("C:\\", "audio_stream", "sb_cache")
os.makedirs(_SB_CACHE, exist_ok=True)
# ---------------------------------------------------------------------------


def _load_encoder():
    """
    Load SpeechBrain ECAPA-TDNN encoder. Called once in the worker thread.

    Windows symlink workaround: SpeechBrain tries to create symlinks from
    the HuggingFace hub cache into its own savedir. Windows blocks this
    without Developer Mode. We work around it by:
      1. Wiping the broken incomplete savedir so SpeechBrain starts fresh.
      2. Monkey-patching os.symlink to silently do a file copy instead,
         so the rest of SpeechBrain's init code works unmodified.
    """
    import shutil
    import builtins

    # Step 1: patch os.symlink to copy instead of symlink
    _real_symlink = os.symlink

    def _safe_symlink(src, dst, target_is_directory=False, **kwargs):
        try:
            _real_symlink(src, dst, target_is_directory=target_is_directory,
                          **kwargs)
        except (OSError, NotImplementedError):
            # Symlink failed (no privilege) — fall back to copying
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    os.symlink = _safe_symlink

    try:
        from speechbrain.inference.speaker import EncoderClassifier
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
            savedir=_SB_CACHE,
        )
    finally:
        # Always restore the real os.symlink when done
        os.symlink = _real_symlink
    return encoder


class Diarizer(threading.Thread):
    def __init__(self, speaker_db,
                 strict_thresh: float,
                 weak_thresh: float,
                 on_done=None,
                 event_queue: Optional[queue.Queue] = None):
        super().__init__(daemon=True, name="Diarizer")
        self._db            = speaker_db
        self._strict        = strict_thresh
        self._weak          = weak_thresh
        self._on_done       = on_done
        self._event_queue   = event_queue

        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stop_event    = threading.Event()
        self._encoder       = None
        self._ready_event   = threading.Event()

        self.jobs_completed = 0
        self.jobs_failed    = 0

    # ---------- public API ----------
    def submit(self, wav_path: str) -> None:
        self._queue.put(wav_path)

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def wait_ready(self, timeout: float = 120.0) -> bool:
        return self._ready_event.wait(timeout)

    # ---------- worker ----------
    def run(self) -> None:
        try:
            print("[diarizer] loading SpeechBrain ECAPA-TDNN encoder "
                  "(first run downloads ~100 MB)...")
            self._encoder = _load_encoder()
            print("[diarizer] ready")
            self._ready_event.set()
        except ImportError:
            print("[diarizer] SpeechBrain not installed. "
                  "Run: pip install speechbrain")
            return
        except Exception as e:
            print(f"[diarizer] failed to load encoder: {e}")
            return

        while not self._stop_event.is_set():
            try:
                path = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if path is None:
                break
            self._process(path)

    # ---------- processing ----------
    def _process(self, wav_path: str) -> None:
        json_path = os.path.splitext(wav_path)[0] + ".json"
        if not os.path.exists(json_path) or not os.path.exists(wav_path):
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        except Exception as e:
            print(f"[diarizer] could not read {json_path}: {e}")
            self.jobs_failed += 1
            return

        segments = transcript.get("segments", [])
        if not segments:
            return

        try:
            audio, sr = self._load_wav_float(wav_path)
        except Exception as e:
            print(f"[diarizer] could not load audio: {e}")
            self.jobs_failed += 1
            return

        # Compute one embedding per usable segment
        embeddings_map: dict[int, np.ndarray] = {}
        for i, seg in enumerate(segments):
            start_s  = float(seg["start"])
            end_s    = float(seg["end"])
            duration = end_s - start_s
            if duration < 0.6:
                continue
            i0 = max(0, int(start_s * sr))
            i1 = min(len(audio), int(end_s * sr))
            if i1 - i0 < int(0.6 * sr):
                continue
            try:
                emb = self._embed(audio[i0:i1])
                if emb is not None:
                    embeddings_map[i] = emb
            except Exception:
                continue

        if not embeddings_map:
            print(f"[diarizer] no usable segments in "
                  f"{os.path.basename(wav_path)}")
            return

        valid_indices = list(embeddings_map.keys())
        valid_embs    = [embeddings_map[i] for i in valid_indices]

        # Cluster segments by speaker
        labels = self._cluster(valid_embs)

        # Average embedding per cluster
        cluster_embs: dict[int, np.ndarray] = {}
        for local_i, label in enumerate(labels):
            e = valid_embs[local_i]
            if label not in cluster_embs:
                cluster_embs[label] = e.copy()
            else:
                cluster_embs[label] = (cluster_embs[label] + e) / 2.0

        # Match each cluster against the known-speaker database
        unknown_counter = 1
        cluster_names: dict[int, tuple[str, float, str]] = {}
        for label, emb in cluster_embs.items():
            result = self._db.match(emb)
            if result is not None:
                name, conf = result
                if conf >= self._strict:
                    cluster_names[label] = (name, conf, "strict")
                    continue
                elif conf >= self._weak:
                    cluster_names[label] = (f"{name}?", conf, "weak")
                    continue
            cluster_names[label] = (
                f"Unknown {unknown_counter}", 0.0, "unknown")
            unknown_counter += 1

        # Apply labels to transcript segments
        local_i = 0
        last_name, last_conf, last_kind = "Unknown ?", 0.0, "unknown"
        for i, seg in enumerate(segments):
            if i in embeddings_map:
                label = labels[local_i]
                name, conf, kind = cluster_names[label]
                seg["speaker"]            = name
                seg["speaker_confidence"] = round(float(conf), 3)
                seg["speaker_kind"]       = kind
                seg["_cluster"]           = int(label)
                last_name, last_conf, last_kind = name, conf, kind
                local_i += 1
            else:
                # Short segment — inherit nearest cluster label
                seg["speaker"]            = last_name
                seg["speaker_confidence"] = round(float(last_conf), 3)
                seg["speaker_kind"]       = last_kind
                seg["_cluster"]           = -1

        transcript["diarized"] = True
        transcript["diarizer"] = "speechbrain-ecapa"

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, indent=2)
        except Exception as e:
            print(f"[diarizer] could not save transcript: {e}")
            self.jobs_failed += 1
            return

        # Save embeddings sidecar so GUI can let user confirm labels later
        emb_path = os.path.splitext(wav_path)[0] + ".embeddings.npz"
        try:
            np.savez(emb_path,
                     **{f"cluster_{lbl}": e
                        for lbl, e in cluster_embs.items()})
        except Exception as e:
            print(f"[diarizer] could not save embeddings sidecar: {e}")

        n_clusters = len(cluster_embs)
        print(f"[diarizer] {os.path.basename(wav_path)} done  "
              f"({n_clusters} speaker cluster"
              f"{'s' if n_clusters != 1 else ''})")
        self.jobs_completed += 1

        # Notify summarizer and GUI
        if self._event_queue is not None:
            try:
                self._event_queue.put_nowait(
                    {"type": "diarize_done", "wav": wav_path})
            except queue.Full:
                pass
        if self._on_done is not None:
            try:
                self._on_done(wav_path)
            except Exception:
                pass

    # ---------- embedding ----------
    def _embed(self, audio_float: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute a 192-dim ECAPA speaker embedding from float32 audio at 16 kHz.
        Returns a unit-normalised numpy array, or None on failure.
        """
        import torch
        tensor = torch.from_numpy(audio_float).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            emb = self._encoder.encode_batch(tensor)          # (1, 1, 192)
        emb = emb.squeeze().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm == 0.0:
            return None
        return emb / norm

    # ---------- helpers ----------
    @staticmethod
    def _load_wav_float(path: str) -> tuple[np.ndarray, int]:  
        with wave.open(path, "rb") as wf:
            sr  = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

        # ECAPA-TDNN requires 16kHz. Resample if the WAV is a different rate
        # (e.g. YouTube audio is typically 44100Hz).
        if sr != 16000:
            import math
            target_sr = 16000
            ratio = target_sr / sr
            target_len = int(math.ceil(len(samples) * ratio))
            indices = np.linspace(0, len(samples) - 1, target_len)
            samples = np.interp(indices, np.arange(len(samples)), samples)
            sr = target_sr

        return samples, sr

    @staticmethod
    def _cluster(embeddings: list[np.ndarray]) -> list[int]:
        """
        Agglomerative clustering on cosine distances.
        distance_threshold=0.4 means cosine similarity > 0.6 → same speaker.
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_distances

        if len(embeddings) == 1:
            return [0]

        X        = np.stack(embeddings)
        dist_mat = cosine_distances(X)
        model    = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=0.43,
        )
        return [int(l) for l in model.fit_predict(dist_mat)]
