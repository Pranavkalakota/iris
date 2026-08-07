"""
iris_youtube.py — M2: YouTube voice-control via Data API + Chrome DevTools.

Hybrid approach:
  • Search / channel lookup / play-by-query → YouTube Data API v3 (fast,
    structured results, no DOM scraping). Falls back to Chrome navigation
    if the API key is missing or quota is exhausted.
  • Playback controls (pause, seek, speed, captions, like, subscribe) →
    Chrome DevTools Protocol (JS injection into the YouTube tab). These
    MUST run in-browser; the API can't control the player.

Every public function returns (True, message) | (False, reason).

Requirements:
  pip install google-api-python-client websocket-client
  Environment variable: YOUTUBE_API_KEY  (set via setx on Windows)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Optional, Tuple

# ── config ──────────────────────────────────────────────────────────────────
DEBUG_PORT = 9222
_no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def _get_api_key() -> str:
    """Get YouTube API key — check config first, then env var directly."""
    try:
        import config_phase9 as _cfg
        key = getattr(_cfg, "YOUTUBE_API_KEY", "") or ""
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("YOUTUBE_API_KEY", "")


# ── YouTube Data API v3 helpers ─────────────────────────────────────────────

_yt_service = None

def _get_yt_service():
    """Lazy-init the YouTube Data API client. Returns None if unavailable."""
    global _yt_service
    if _yt_service is not None:
        return _yt_service
    key = _get_api_key()
    if not key:
        print("[youtube] No YOUTUBE_API_KEY found in config or environment")
        return None
    try:
        from googleapiclient.discovery import build  # type: ignore
        _yt_service = build("youtube", "v3", developerKey=key)
        print(f"[youtube] API initialized successfully")
        return _yt_service
    except Exception as e:
        print(f"[youtube] API init failed: {e}")
        return None


def _api_search(query: str, max_results: int = 5, search_type: str = "video"):
    """Search YouTube via Data API. Returns list of {title, videoId/channelId, url}."""
    svc = _get_yt_service()
    if not svc:
        return None
    try:
        resp = svc.search().list(
            q=query, part="snippet", type=search_type,
            maxResults=max_results
        ).execute()
        results = []
        for item in resp.get("items", []):
            snippet = item["snippet"]
            vid = item["id"].get("videoId")
            cid = item["id"].get("channelId")
            entry = {"title": snippet["title"]}
            if vid:
                entry["videoId"] = vid
                entry["url"] = f"https://www.youtube.com/watch?v={vid}"
            elif cid:
                entry["channelId"] = cid
                entry["url"] = f"https://www.youtube.com/channel/{cid}"
            results.append(entry)
        return results
    except Exception as e:
        print(f"[youtube] API search failed: {e}")
        return None


# ── Chrome debug helpers ────────────────────────────────────────────────────

def _debug_json(path: str, method: str = "GET", timeout: float = 2.0):
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{DEBUG_PORT}{path}", method=method)
        with _no_proxy.open(req, timeout=timeout) as r:
            body = r.read()
        return json.loads(body) if body else True
    except Exception:
        return None


def _find_yt_tab() -> Optional[dict]:
    """Find the first YouTube tab in Chrome's debug tabs."""
    tabs = _debug_json("/json")
    if not tabs:
        return None
    for t in tabs:
        url = t.get("url", "")
        if t.get("type") == "page" and "youtube.com" in url:
            return t
    return None


def _ws_eval(tab: dict, js: str, timeout: float = 3.0) -> Optional[str]:
    """Evaluate JS in a Chrome tab via DevTools WebSocket."""
    ws_url = tab.get("webSocketDebuggerUrl")
    if not ws_url:
        return None
    try:
        import websocket  # type: ignore
        ws = websocket.create_connection(ws_url, timeout=timeout)
        msg = json.dumps({
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {"expression": js, "returnByValue": True}
        })
        ws.send(msg)
        resp = json.loads(ws.recv())
        ws.close()
        result = resp.get("result", {}).get("result", {})
        return result.get("value")
    except Exception as e:
        print(f"[youtube] ws_eval failed: {e}")
        return None


def _navigate_yt(url: str) -> bool:
    """Navigate an existing YT tab or open a new one."""
    tab = _find_yt_tab()
    if tab:
        _debug_json(f"/json/activate/{tab['id']}")
        _ws_eval(tab, f"window.location.href = '{url}'")
    else:
        _debug_json(f"/json/new?{url}", method="PUT")
    return True


# ── Public YouTube actions ──────────────────────────────────────────────────

def search(query: str) -> Tuple[bool, str]:
    """Search YouTube. Uses API if available, falls back to browser nav."""
    if not query:
        return False, "What should I search for on YouTube?"

    # Try API first — faster and structured
    results = _api_search(query, max_results=5)
    if results:
        # Navigate browser to the first result's search page so user sees it
        url = f"https://www.youtube.com/results?search_query={urllib.request.quote(query)}"
        _navigate_yt(url)
        titles = [r["title"] for r in results[:3]]
        preview = ", ".join(titles)
        return True, f"Found: {preview}"

    # Fallback: direct browser navigation
    url = f"https://www.youtube.com/results?search_query={urllib.request.quote(query)}"
    _navigate_yt(url)
    return True, f"Searching YouTube for \"{query}\"."


def play_video_by_query(query: str) -> Tuple[bool, str]:
    """Search and auto-play the first result."""
    if not query:
        return False, "What video should I play?"

    # Try API first — get the direct video URL (skips search page entirely)
    results = _api_search(query, max_results=1)
    if results and results[0].get("videoId"):
        video_url = results[0]["url"]
        title = results[0]["title"]
        _navigate_yt(video_url)
        return True, f"Playing \"{title}\"."

    # Fallback: navigate to search, then click first REAL video result
    # (skip ads and playlist shelves)
    url = f"https://www.youtube.com/results?search_query={urllib.request.quote(query)}"
    _navigate_yt(url)
    # Retry click with increasing wait — page may take time to render
    tab = None
    for wait in (2.0, 2.0, 2.0):
        time.sleep(wait)
        tab = _find_yt_tab()
        if not tab:
            continue
        js = """
        (function() {
            // Skip ads (ytd-ad-slot-renderer) and playlists — only match
            // real video renderers (ytd-video-renderer).
            var renderers = document.querySelectorAll('ytd-video-renderer');
            for (var r of renderers) {
                // Make sure it's not inside an ad shelf
                if (r.closest('ytd-ad-slot-renderer')) continue;
                var link = r.querySelector('a#video-title');
                if (link && link.href) { link.click(); return 'clicked'; }
                var thumb = r.querySelector('a#thumbnail');
                if (thumb && thumb.href) { thumb.click(); return 'clicked'; }
            }
            return 'no_result';
        })()
        """
        result = _ws_eval(tab, js)
        if result == "clicked":
            return True, f"Playing \"{query}\" on YouTube."
    return True, f"Searched for \"{query}\" — click on a video to play it."


def play_first_result() -> Tuple[bool, str]:
    """Click the first real video in YouTube search results (skips ads)."""
    tab = _find_yt_tab()
    if not tab:
        return False, "YouTube isn't open."
    js = """
    (function() {
        var renderers = document.querySelectorAll('ytd-video-renderer');
        for (var r of renderers) {
            if (r.closest('ytd-ad-slot-renderer')) continue;
            var link = r.querySelector('a#video-title');
            if (link && link.href) { link.click(); return 'clicked'; }
            var thumb = r.querySelector('a#thumbnail');
            if (thumb && thumb.href) { thumb.click(); return 'clicked'; }
        }
        return 'no_result';
    })()
    """
    result = _ws_eval(tab, js)
    if result == "clicked":
        return True, "Playing the first result."
    return False, "I couldn't find a video to play on this page."


def pause_resume() -> Tuple[bool, str]:
    """Toggle pause/play on the current YouTube video."""
    tab = _find_yt_tab()
    if not tab:
        return False, "YouTube isn't open."
    js = """
    (function() {
        var v = document.querySelector('video');
        if (!v) return 'no_video';
        if (v.paused) { v.play(); return 'playing'; }
        else { v.pause(); return 'paused'; }
    })()
    """
    result = _ws_eval(tab, js)
    if result == "playing":
        return True, "Resumed playback."
    if result == "paused":
        return True, "Paused."
    return False, "No video is playing on YouTube right now."


def seek(seconds: float) -> Tuple[bool, str]:
    """Skip forward or backward by N seconds."""
    tab = _find_yt_tab()
    if not tab:
        return False, "YouTube isn't open."
    js = f"""
    (function() {{
        var v = document.querySelector('video');
        if (!v) return 'no_video';
        v.currentTime += {seconds};
        return String(Math.round(v.currentTime));
    }})()
    """
    result = _ws_eval(tab, js)
    if result and result != "no_video":
        direction = "forward" if seconds > 0 else "back"
        return True, f"Skipped {direction} {abs(int(seconds))} seconds."
    return False, "No video is playing."


def set_speed(rate: float) -> Tuple[bool, str]:
    """Set playback speed (0.25 to 2.0)."""
    rate = max(0.25, min(2.0, rate))
    tab = _find_yt_tab()
    if not tab:
        return False, "YouTube isn't open."
    js = f"""
    (function() {{
        var v = document.querySelector('video');
        if (!v) return 'no_video';
        v.playbackRate = {rate};
        return String(v.playbackRate);
    }})()
    """
    result = _ws_eval(tab, js)
    if result and result != "no_video":
        return True, f"Playback speed set to {rate}x."
    return False, "No video is playing."


def toggle_captions() -> Tuple[bool, str]:
    """Toggle captions on/off."""
    tab = _find_yt_tab()
    if not tab:
        return False, "YouTube isn't open."
    js = """
    (function() {
        var btn = document.querySelector('.ytp-subtitles-button');
        if (!btn) return 'no_btn';
        btn.click();
        var on = btn.getAttribute('aria-pressed') === 'true';
        return on ? 'on' : 'off';
    })()
    """
    result = _ws_eval(tab, js)
    if result == "on":
        return True, "Captions turned on."
    if result == "off":
        return True, "Captions turned off."
    return False, "I couldn't find the captions button — make sure a video is playing."


def subscribe() -> Tuple[bool, str]:
    """Click the subscribe button on the current channel."""
    tab = _find_yt_tab()
    if not tab:
        return False, "YouTube isn't open."
    js = """
    (function() {
        var btn = document.querySelector('ytd-subscribe-button-renderer button');
        if (!btn) return 'no_btn';
        if (btn.getAttribute('subscribed') !== null ||
            btn.getAttribute('aria-label')?.toLowerCase().includes('unsubscribe')) {
            return 'already';
        }
        btn.click();
        return 'subscribed';
    })()
    """
    result = _ws_eval(tab, js)
    if result == "subscribed":
        return True, "Subscribed!"
    if result == "already":
        return True, "You're already subscribed to this channel."
    return False, "I couldn't find the subscribe button — navigate to a channel or video first."


def like_video() -> Tuple[bool, str]:
    """Click the like (thumbs up) button."""
    tab = _find_yt_tab()
    if not tab:
        return False, "YouTube isn't open."
    js = """
    (function() {
        var btns = document.querySelectorAll(
            'ytd-menu-renderer like-button-view-model button, ' +
            'ytd-menu-renderer ytd-toggle-button-renderer button');
        for (var b of btns) {
            var label = (b.getAttribute('aria-label') || '').toLowerCase();
            if (label.includes('like') && !label.includes('dislike')) {
                b.click();
                return 'liked';
            }
        }
        var seg = document.querySelector('like-button-view-model button');
        if (seg) { seg.click(); return 'liked'; }
        return 'no_btn';
    })()
    """
    result = _ws_eval(tab, js)
    if result == "liked":
        return True, "Liked this video!"
    return False, "I couldn't find the like button."


def open_channel(channel_name: str) -> Tuple[bool, str]:
    """Navigate to a YouTube channel. Uses API if available."""
    if not channel_name:
        return False, "Which channel?"

    # Try API — search for channels specifically
    results = _api_search(channel_name, max_results=1, search_type="channel")
    if results and results[0].get("channelId"):
        url = results[0]["url"]
        _navigate_yt(url)
        return True, f"Opening {results[0]['title']}."

    # Fallback: search with channel filter
    query = urllib.request.quote(channel_name)
    url = f"https://www.youtube.com/results?search_query={query}&sp=EgIQAg%253D%253D"
    _navigate_yt(url)
    return True, f"Searching for the {channel_name} channel."


def open_watch_later() -> Tuple[bool, str]:
    """Open the Watch Later playlist."""
    _navigate_yt("https://www.youtube.com/playlist?list=WL")
    return True, "Opening your Watch Later playlist."