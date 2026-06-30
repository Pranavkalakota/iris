"""
Location service for Phase 8.

Currently supports IP geolocation via ip-api.com (free, no API key,
no signup, soft limit of 45 requests/minute for personal use).

Returns a Location dict or None on failure:
    {
        "lat": 39.7684,
        "lon": -86.1581,
        "accuracy_m": 50000,        # approximate, IP-level
        "city": "Indianapolis",
        "region": "Indiana",
        "country": "US",
        "source": "ip-api",
        "timestamp_iso": "2026-05-12T21:17:23"
    }

Stores location data as a sidecar JSON next to each WAV:
    recording_2026-05-12_21-17-23_chunk01.wav
    recording_2026-05-12_21-17-23_chunk01.location.json

The sidecar pattern keeps the WAV format clean and lets us add or
remove location data without rewriting recordings.
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Optional

import requests


class LocationService:
    """Fetches and caches the host's geolocation."""

    def __init__(self, timeout_s: float = 5.0):
        self._timeout = timeout_s
        self._cached: Optional[dict] = None
        self._cached_at: float = 0.0
        self._cache_ttl: float = 3600.0    # 1 hour
        self._lock = threading.Lock()

    def get(self, force_refresh: bool = False) -> Optional[dict]:
        """Returns a location dict, or None if lookup failed."""
        with self._lock:
            now = time.time()
            if (not force_refresh
                    and self._cached is not None
                    and now - self._cached_at < self._cache_ttl):
                return self._cached

            loc = self._fetch_ip_api()
            if loc is not None:
                self._cached = loc
                self._cached_at = now
            return loc

    def _fetch_ip_api(self) -> Optional[dict]:
        try:
            resp = requests.get(
                "http://ip-api.com/json/",
                timeout=self._timeout,
                params={"fields": "status,country,regionName,city,lat,lon,query"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[location] ip-api fetch failed: {e}")
            return None

        if data.get("status") != "success":
            return None

        return {
            "lat": float(data["lat"]),
            "lon": float(data["lon"]),
            "accuracy_m": 50000,         # ip-api is city-level; ~50 km guess
            "city": data.get("city", ""),
            "region": data.get("regionName", ""),
            "country": data.get("country", ""),
            "source": "ip-api",
            "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
        }


def save_location_sidecar(wav_path: str, location: dict) -> None:
    """Write a .location.json next to the given WAV."""
    if location is None:
        return
    sidecar = os.path.splitext(wav_path)[0] + ".location.json"
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(location, f, indent=2)
    except Exception as e:
        print(f"[location] could not save sidecar: {e}")


def load_location_sidecar(wav_path: str) -> Optional[dict]:
    """Read the .location.json next to the given WAV, if present."""
    sidecar = os.path.splitext(wav_path)[0] + ".location.json"
    if not os.path.exists(sidecar):
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
