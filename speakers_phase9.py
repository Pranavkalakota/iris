"""
Speaker profile database for Phase 9.

Stores voice embeddings for known speakers and provides:
  - match(embedding) -> (name, confidence) or None
  - add_to(name, embedding) -> grows that profile
  - create(name, embedding) -> new profile from scratch
  - delete(name) -> removes profile entirely
  - rename(old, new) -> renames a profile

Embeddings are stored in a single JSON file. Each profile keeps multiple
embeddings (averaged when matching) so accuracy improves over time as
you confirm more samples.

Thread-safe: all mutations go through a single lock. Reads use a lock
too because we hold numpy arrays in the data.
"""

import json
import os
import threading
import time
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Returns similarity in [-1, 1]; 1 = identical direction."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SpeakerDB:
    def __init__(self, path: str, max_embeddings_per_profile: int = 30):
        self._path = path
        self._max_emb = max_embeddings_per_profile
        self._lock = threading.Lock()
        self._profiles: dict[str, dict] = {}
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for name, data in raw.get("speakers", {}).items():
                embeddings = [np.array(e, dtype=np.float32)
                              for e in data.get("embeddings", [])]
                self._profiles[name] = {
                    "embeddings": embeddings,
                    "created": data.get("created", ""),
                    "sample_count": data.get("sample_count", len(embeddings)),
                }
        except Exception as e:
            print(f"[speakers] error loading db: {e}")

    def _save(self) -> None:
        out = {"speakers": {}}
        for name, data in self._profiles.items():
            out["speakers"][name] = {
                "embeddings": [e.tolist() for e in data["embeddings"]],
                "created": data["created"],
                "sample_count": data["sample_count"],
            }
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            print(f"[speakers] error saving db: {e}")

    # ---------- API ----------
    def list_names(self) -> list[str]:
        with self._lock:
            return list(self._profiles.keys())

    def get_info(self, name: str) -> dict | None:
        with self._lock:
            p = self._profiles.get(name)
            if p is None:
                return None
            return {
                "name": name,
                "sample_count": p["sample_count"],
                "embeddings_stored": len(p["embeddings"]),
                "created": p["created"],
            }

    def all_info(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name": name,
                    "sample_count": p["sample_count"],
                    "embeddings_stored": len(p["embeddings"]),
                    "created": p["created"],
                }
                for name, p in self._profiles.items()
            ]

    def match(self, embedding: np.ndarray) -> tuple[str, float] | None:
        """
        Returns (best_name, confidence) over all profiles, or None if no
        profiles exist. Caller decides how to interpret the confidence.
        """
        with self._lock:
            best: tuple[str, float] | None = None
            for name, p in self._profiles.items():
                if not p["embeddings"]:
                    continue
                # Average embedding for the profile, then cosine similarity.
                avg = np.mean(np.stack(p["embeddings"]), axis=0)
                sim = cosine_similarity(embedding, avg)
                if best is None or sim > best[1]:
                    best = (name, sim)
            return best

    def create(self, name: str, embedding: np.ndarray) -> None:
        with self._lock:
            if name in self._profiles:
                # Treat as add_to
                self._add_to_locked(name, embedding)
                return
            self._profiles[name] = {
                "embeddings": [embedding.astype(np.float32)],
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "sample_count": 1,
            }
            self._save()

    def add_to(self, name: str, embedding: np.ndarray) -> None:
        with self._lock:
            if name not in self._profiles:
                # Create new profile if it doesn't exist
                self._profiles[name] = {
                    "embeddings": [embedding.astype(np.float32)],
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "sample_count": 1,
                }
            else:
                self._add_to_locked(name, embedding)
            self._save()

    def _add_to_locked(self, name: str, embedding: np.ndarray) -> None:
        p = self._profiles[name]
        p["embeddings"].append(embedding.astype(np.float32))
        p["sample_count"] += 1
        # Keep only the most recent N embeddings to bound storage
        if len(p["embeddings"]) > self._max_emb:
            p["embeddings"] = p["embeddings"][-self._max_emb:]

    def delete(self, name: str) -> bool:
        with self._lock:
            if name not in self._profiles:
                return False
            del self._profiles[name]
            self._save()
            return True

    def rename(self, old: str, new: str) -> bool:
        with self._lock:
            if old not in self._profiles:
                return False
            if new in self._profiles:
                # Merge into existing
                target = self._profiles[new]
                source = self._profiles[old]
                target["embeddings"].extend(source["embeddings"])
                target["sample_count"] += source["sample_count"]
                if len(target["embeddings"]) > self._max_emb:
                    target["embeddings"] = target["embeddings"][-self._max_emb:]
                del self._profiles[old]
            else:
                self._profiles[new] = self._profiles.pop(old)
            self._save()
            return True
