"""
iris_spotify.py — M3: Spotify search / playback / playlist control.

Talks to the real Spotify Web API (not DOM scraping) via spotipy. Two-step
play by design: `resolve_song()` finds and stages a track WITHOUT playing
it; `play_track()` is a separate, explicit step, matching the "pull it
up, then say 'you can play it now'" flow.

Degrades gracefully: if spotipy isn't installed or OAuth hasn't been done
yet, every function returns (False, <human-readable reason>) instead of
raising, so iris_gui.py never crashes because of this module.
"""

from __future__ import annotations

import random
from typing import Optional

try:
    import config_phase9 as config  # type: ignore
except Exception:
    config = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except Exception:
    spotipy = None
    SpotifyOAuth = None

SCOPES = (
    "user-read-playback-state user-modify-playback-state "
    "playlist-read-private playlist-modify-private playlist-modify-public"
)

_client: Optional["spotipy.Spotify"] = None


def _cfg(attr: str, default=""):
    return getattr(config, attr, default) if config is not None else default


def get_client():
    """Lazy singleton — OAuth browser popup only happens once per token
    lifetime (spotipy caches to .spotify_cache and auto-refreshes)."""
    global _client
    if _client is not None:
        return _client
    if spotipy is None:
        return None
    client_id = _cfg("SPOTIFY_CLIENT_ID")
    client_secret = _cfg("SPOTIFY_CLIENT_SECRET")
    redirect_uri = _cfg("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
    if not client_id or not client_secret:
        return None
    try:
        auth = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=SCOPES,
            cache_path=".spotify_cache",
            open_browser=True,
        )
        _client = spotipy.Spotify(auth_manager=auth)
    except Exception as e:
        print(f"[spotify] client init failed: {e}")
        return None
    return _client


def resolve_song(artist: str, track: Optional[str] = None):
    """Search for a track. If `track` is given, matches it exactly against
    that artist. If not, picks RANDOMLY from that artist's tracks rather
    than always defaulting to the most popular song.

    Returns (True, track_dict) on success, (False, reason) on failure.
    track_dict: {"uri", "name", "artist", "album", "image_url", "url"}
    """
    sp = get_client()
    if sp is None:
        return False, "Spotify isn't connected yet — check the API credentials."
    if not artist:
        return False, "I need an artist name."

    try:
        if track:
            q = f"track:{track} artist:{artist}"
            res = sp.search(q=q, type="track", limit=1)
            items = res.get("tracks", {}).get("items", [])
            if not items:
                # fall back to a looser search if the exact match misses
                res = sp.search(q=f"{track} {artist}", type="track", limit=1)
                items = res.get("tracks", {}).get("items", [])
        else:
            res = sp.search(q=f"artist:{artist}", type="track", limit=10)
            items = res.get("tracks", {}).get("items", [])

        if not items:
            return False, f"I couldn't find anything by {artist}."

        item = items[0] if track else random.choice(items)
        images = item.get("album", {}).get("images", [])
        return True, {
            "uri": item["uri"],
            "name": item["name"],
            "artist": ", ".join(a["name"] for a in item.get("artists", [])),
            "album": item.get("album", {}).get("name", ""),
            "image_url": images[0]["url"] if images else None,
            "url": item.get("external_urls", {}).get("spotify", ""),
        }
    except Exception as e:
        return False, f"Spotify search failed: {e}"


def play_track(track_uri: str):
    """Start playback via Spotify Connect on whatever device is currently
    active. Requires Premium + an active device — Spotify's API itself
    enforces that, this just surfaces a clear message when it fails."""
    sp = get_client()
    if sp is None:
        return False, "Spotify isn't connected yet."
    try:
        sp.start_playback(uris=[track_uri])
        return True, "Playing now."
    except spotipy.exceptions.SpotifyException as e:  # type: ignore
        if e.http_status == 404:
            return False, ("No active Spotify device found — open Spotify "
                            "on your phone, desktop app, or a browser tab "
                            "first, then ask again.")
        if e.http_status == 403:
            return False, "Playback control needs Spotify Premium."
        return False, f"Couldn't start playback: {e}"
    except Exception as e:
        return False, f"Couldn't start playback: {e}"


def list_playlists():
    sp = get_client()
    if sp is None:
        return []
    try:
        out, results = [], sp.current_user_playlists(limit=50)
        while results:
            out.extend(results["items"])
            results = sp.next(results) if results.get("next") else None
        return [{"id": p["id"], "name": p["name"]} for p in out]
    except Exception as e:
        print(f"[spotify] list_playlists failed: {e}")
        return []


def match_playlist(name: str):
    """Best-effort fuzzy match of a spoken playlist name against the
    user's actual playlists. Falls back to substring matching if rapidfuzz
    isn't installed."""
    playlists = list_playlists()
    if not playlists:
        return None
    name_l = name.lower().strip()
    try:
        from rapidfuzz import process, fuzz
        choices = {p["id"]: p["name"] for p in playlists}
        match = process.extractOne(name_l, choices, scorer=fuzz.WRatio)
        if match and match[1] >= 60:
            pid = match[2]
            return next(p for p in playlists if p["id"] == pid)
    except Exception:
        for p in playlists:
            if name_l in p["name"].lower() or p["name"].lower() in name_l:
                return p
    return None


def add_to_playlist(playlist_id: str, track_uri: str):
    sp = get_client()
    if sp is None:
        return False, "Spotify isn't connected yet."
    try:
        sp.playlist_add_items(playlist_id, [track_uri])
        return True, "Added."
    except Exception as e:
        return False, f"Couldn't add it to that playlist: {e}"

def list_devices():
    sp = get_client()
    if sp is None:
        return []
    try:
        return sp.devices().get("devices", [])
    except Exception as e:
        print(f"[spotify] list_devices failed: {e}")
        return []