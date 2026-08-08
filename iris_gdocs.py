"""
iris_gdocs.py — M2: Google Docs voice-control via Docs/Drive API + Chrome DevTools.

Hybrid approach:
  • Create, search, find/replace, rename, export, insert text →
    Google Docs & Drive APIs (reliable, structured).
  • Formatting (bold, italic, headings, lists, alignment, etc.) →
    Chrome DevTools Protocol Input.dispatchKeyEvent (sends REAL
    keystrokes through the browser input pipeline — Google Docs
    treats these as genuine user input, unlike JavaScript
    KeyboardEvent which Docs ignores).

Every public function returns (True, message) | (False, reason).

Requirements:
  pip install google-api-python-client google-auth-oauthlib websocket-client
  File: gdocs_credentials.json (OAuth 2.0 Desktop client, next to iris_gui.py)
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

# Path to OAuth credentials — look next to this file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CREDS_FILE = os.path.join(_THIS_DIR, "gdocs_credentials.json")
_TOKEN_FILE = os.path.join(_THIS_DIR, "gdocs_token.json")

# OAuth scopes needed
_SCOPES = [
    "https://www.googleapis.com/auth/documents",       # read/write docs
    "https://www.googleapis.com/auth/drive",            # search/create/rename/export
]


# ── Google API auth ─────────────────────────────────────────────────────────

_docs_service = None
_drive_service = None

def _get_creds():
    """Load or refresh OAuth credentials. Opens browser for first-time auth."""
    if not os.path.exists(_CREDS_FILE):
        print(f"[gdocs] No credentials file at {_CREDS_FILE}")
        return None
    try:
        from google.oauth2.credentials import Credentials          # type: ignore
        from google_auth_oauthlib.flow import InstalledAppFlow     # type: ignore
        from google.auth.transport.requests import Request         # type: ignore
    except ImportError as e:
        print(f"[gdocs] Missing library: {e}. "
              "Run: pip install google-api-python-client google-auth-oauthlib")
        return None

    creds = None
    if os.path.exists(_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(_TOKEN_FILE, _SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(_CREDS_FILE, _SCOPES)
            creds = flow.run_local_server(port=8889)
        with open(_TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return creds


def _get_docs():
    """Lazy-init Google Docs API client."""
    global _docs_service
    if _docs_service is not None:
        return _docs_service
    creds = _get_creds()
    if not creds:
        return None
    try:
        from googleapiclient.discovery import build  # type: ignore
        _docs_service = build("docs", "v1", credentials=creds)
        return _docs_service
    except Exception as e:
        print(f"[gdocs] Docs API init failed: {e}")
        return None


def _get_drive():
    """Lazy-init Google Drive API client."""
    global _drive_service
    if _drive_service is not None:
        return _drive_service
    creds = _get_creds()
    if not creds:
        return None
    try:
        from googleapiclient.discovery import build  # type: ignore
        _drive_service = build("drive", "v3", credentials=creds)
        return _drive_service
    except Exception as e:
        print(f"[gdocs] Drive API init failed: {e}")
        return None


# ── Chrome debug helpers ──────────────────────────────────────────────────

def _debug_json(path: str, method: str = "GET", timeout: float = 2.0):
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{DEBUG_PORT}{path}", method=method)
        with _no_proxy.open(req, timeout=timeout) as r:
            body = r.read()
        return json.loads(body) if body else True
    except Exception:
        return None


def _find_docs_tab() -> Optional[dict]:
    """Find the first Google Docs tab."""
    tabs = _debug_json("/json")
    if not tabs:
        return None
    for t in tabs:
        url = t.get("url", "")
        if t.get("type") == "page" and "docs.google.com/document" in url:
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
        print(f"[gdocs] ws_eval failed: {e}")
        return None


def _navigate_docs(url: str) -> bool:
    """Navigate an existing Docs tab or open a new one."""
    tab = _find_docs_tab()
    if tab:
        _debug_json(f"/json/activate/{tab['id']}")
        _ws_eval(tab, f"window.location.href = '{url}'")
    else:
        _debug_json(f"/json/new?{url}", method="PUT")
    return True


# ── CDP keyboard input (the key fix) ─────────────────────────────────────
#
# Google Docs ignores JavaScript KeyboardEvent and DOM .click() on toolbar
# buttons. The ONLY reliable way to trigger shortcuts is via CDP's
# Input.dispatchKeyEvent, which feeds real keystrokes through the browser's
# input pipeline — Google Docs treats them as genuine user input.

def _cdp_key_combo(tab: dict, key_char: str,
                   ctrl: bool = False, shift: bool = False,
                   alt: bool = False, timeout: float = 3.0) -> bool:
    """Send a real keyboard shortcut via CDP Input.dispatchKeyEvent."""
    ws_url = tab.get("webSocketDebuggerUrl")
    if not ws_url:
        return False

    # Modifier bitmask: Alt=1, Ctrl=2, Meta=4, Shift=8
    modifiers = 0
    if alt:   modifiers |= 1
    if ctrl:  modifiers |= 2
    if shift: modifiers |= 8

    # Map key to virtual key code and code string
    _SPECIAL_KEYS = {
        '\\': (220, "Backslash", "\\"),
        '.':  (190, "Period",    "."),
        ',':  (188, "Comma",     ","),
        '/':  (191, "Slash",     "/"),
        ';':  (186, "Semicolon", ";"),
        "'":  (222, "Quote",     "'"),
        '[':  (219, "BracketLeft", "["),
        ']':  (221, "BracketRight", "]"),
        '-':  (189, "Minus",     "-"),
        '=':  (187, "Equal",     "="),
    }

    if key_char in _SPECIAL_KEYS:
        vk, code, key_val = _SPECIAL_KEYS[key_char]
    elif len(key_char) == 1 and key_char.isalpha():
        vk = ord(key_char.upper())
        code = f"Key{key_char.upper()}"
        key_val = key_char.lower()
    elif len(key_char) == 1 and key_char.isdigit():
        vk = ord(key_char)
        code = f"Digit{key_char}"
        key_val = key_char
    else:
        print(f"[gdocs] Unknown key: {key_char!r}")
        return False

    try:
        import websocket  # type: ignore
        ws = websocket.create_connection(ws_url, timeout=timeout)

        # keyDown
        ws.send(json.dumps({
            "id": 1,
            "method": "Input.dispatchKeyEvent",
            "params": {
                "type": "rawKeyDown",
                "modifiers": modifiers,
                "windowsVirtualKeyCode": vk,
                "key": key_val,
                "code": code,
            }
        }))
        ws.recv()

        # keyUp
        ws.send(json.dumps({
            "id": 2,
            "method": "Input.dispatchKeyEvent",
            "params": {
                "type": "keyUp",
                "modifiers": modifiers,
                "windowsVirtualKeyCode": vk,
                "key": key_val,
                "code": code,
            }
        }))
        ws.recv()

        ws.close()
        return True
    except Exception as e:
        print(f"[gdocs] CDP key combo failed: {e}")
        return False


def _cdp_type_text(tab: dict, text: str, timeout: float = 3.0) -> bool:
    """Type text into a focused element via CDP Input.insertText."""
    ws_url = tab.get("webSocketDebuggerUrl")
    if not ws_url:
        return False
    try:
        import websocket  # type: ignore
        ws = websocket.create_connection(ws_url, timeout=timeout)
        ws.send(json.dumps({
            "id": 1,
            "method": "Input.insertText",
            "params": {"text": text}
        }))
        ws.recv()
        ws.close()
        return True
    except Exception as e:
        print(f"[gdocs] CDP type failed: {e}")
        return False


def _cdp_click_at(tab: dict, x: float, y: float, timeout: float = 3.0) -> bool:
    """Send a real mouse click at (x, y) via CDP Input.dispatchMouseEvent.
    Google Docs ignores JS .click() on toolbar buttons but responds to
    real mouse events dispatched through CDP."""
    ws_url = tab.get("webSocketDebuggerUrl")
    if not ws_url:
        return False
    try:
        import websocket  # type: ignore
        ws = websocket.create_connection(ws_url, timeout=timeout)
        # mousePressed
        ws.send(json.dumps({
            "id": 1,
            "method": "Input.dispatchMouseEvent",
            "params": {
                "type": "mousePressed",
                "x": int(x), "y": int(y),
                "button": "left",
                "clickCount": 1,
            }
        }))
        ws.recv()
        # mouseReleased
        ws.send(json.dumps({
            "id": 2,
            "method": "Input.dispatchMouseEvent",
            "params": {
                "type": "mouseReleased",
                "x": int(x), "y": int(y),
                "button": "left",
                "clickCount": 1,
            }
        }))
        ws.recv()
        ws.close()
        return True
    except Exception as e:
        print(f"[gdocs] CDP click failed: {e}")
        return False


def _get_element_center(tab: dict, js_selector: str) -> Optional[Tuple[float, float]]:
    """Run JS to find an element and return its center (x, y) coordinates."""
    js = f"""
    (function() {{
        var el = {js_selector};
        if (!el) return null;
        var rect = el.getBoundingClientRect();
        return JSON.stringify({{x: rect.x + rect.width / 2,
                               y: rect.y + rect.height / 2}});
    }})()
    """
    result = _ws_eval(tab, js)
    if result:
        try:
            pos = json.loads(result)
            return (pos["x"], pos["y"])
        except Exception:
            pass
    return None


def _cdp_press_enter(tab: dict, timeout: float = 3.0) -> bool:
    """Press Enter via CDP."""
    ws_url = tab.get("webSocketDebuggerUrl")
    if not ws_url:
        return False
    try:
        import websocket  # type: ignore
        ws = websocket.create_connection(ws_url, timeout=timeout)
        for i, evt_type in enumerate(["rawKeyDown", "keyUp"], 1):
            ws.send(json.dumps({
                "id": i,
                "method": "Input.dispatchKeyEvent",
                "params": {
                    "type": evt_type,
                    "modifiers": 0,
                    "windowsVirtualKeyCode": 13,
                    "key": "Enter",
                    "code": "Enter",
                }
            }))
            ws.recv()
        ws.close()
        return True
    except Exception as e:
        print(f"[gdocs] CDP enter failed: {e}")
        return False


def _ensure_docs_tab() -> Optional[dict]:
    """Find and activate the Docs tab, returning it. None if not found."""
    tab = _find_docs_tab()
    if not tab:
        return None
    _debug_json(f"/json/activate/{tab['id']}")
    time.sleep(0.15)  # give tab a moment to come to front
    return tab


# ── Public Google Docs actions ──────────────────────────────────────────────

# ---------- Document management (API-based) ----------

def create_document(title: str = "", content: str = "") -> Tuple[bool, str]:
    """Create a new Google Doc. Uses API if available.
    If content is provided, inserts it into the document body."""
    docs = _get_docs()
    if docs:
        try:
            body = {"title": title or "Untitled document"}
            doc = docs.documents().create(body=body).execute()
            doc_id = doc["documentId"]

            # Insert content if provided
            if content:
                requests = [{
                    "insertText": {
                        "location": {"index": 1},
                        "text": content
                    }
                }]
                docs.documents().batchUpdate(
                    documentId=doc_id, body={"requests": requests}
                ).execute()

            url = f"https://docs.google.com/document/d/{doc_id}/edit"
            _navigate_docs(url)
            name = title or "Untitled document"
            detail = " with your content" if content else ""
            return True, f"Created \"{name}\"{detail}."
        except Exception as e:
            print(f"[gdocs] API create failed: {e}")

    # Fallback: browser
    _navigate_docs("https://docs.google.com/document/create")
    if title:
        time.sleep(3)
        tab = _find_docs_tab()
        if tab:
            js = f"""
            (function() {{
                var t = document.querySelector('.docs-title-input');
                if (t) {{
                    t.focus();
                    document.execCommand('selectAll');
                    document.execCommand('insertText', false, '{title}');
                    t.blur();
                    return 'renamed';
                }}
                return 'no_title';
            }})()
            """
            _ws_eval(tab, js)
    return True, f"Created \"{title or 'new document'}\"."


def search_documents(query: str) -> Tuple[bool, str]:
    """Search Google Drive for documents. Uses API if available."""
    if not query:
        return False, "What should I search for?"

    drive = _get_drive()
    if drive:
        try:
            q = f"mimeType='application/vnd.google-apps.document' and name contains '{query}'"
            resp = drive.files().list(
                q=q, pageSize=5, fields="files(id, name, modifiedTime)"
            ).execute()
            files = resp.get("files", [])
            if files:
                first = files[0]
                url = f"https://docs.google.com/document/d/{first['id']}/edit"
                _navigate_docs(url)
                names = [f["name"] for f in files[:3]]
                return True, f"Found: {', '.join(names)}. Opened \"{first['name']}\"."
            return True, f"No documents found for \"{query}\"."
        except Exception as e:
            print(f"[gdocs] API search failed: {e}")

    # Fallback: Drive search in browser
    encoded = urllib.request.quote(query)
    url = f"https://drive.google.com/drive/search?q={encoded}"
    tabs = _debug_json("/json")
    drive_tab = None
    if tabs:
        for t in tabs:
            if t.get("type") == "page" and "drive.google.com" in t.get("url", ""):
                drive_tab = t
                break
    if drive_tab:
        _debug_json(f"/json/activate/{drive_tab['id']}")
        _ws_eval(drive_tab, f"window.location.href = '{url}'")
    else:
        _debug_json(f"/json/new?{url}", method="PUT")
    return True, f"Searching for \"{query}\" in Google Drive."


def open_docs_home() -> Tuple[bool, str]:
    """Open Google Docs home page."""
    _navigate_docs("https://docs.google.com/document/u/0/")
    return True, "Opened Google Docs."


def find_replace(find_text: str, replace_text: str) -> Tuple[bool, str]:
    """Find and replace text in the current document. Uses API if possible."""
    if not find_text:
        return False, "What should I find?"

    tab = _find_docs_tab()
    if tab:
        import re
        url = tab.get("url", "")
        m = re.search(r"/document/d/([a-zA-Z0-9_-]+)", url)
        if m and _get_docs():
            doc_id = m.group(1)
            try:
                requests = [{
                    "replaceAllText": {
                        "containsText": {
                            "text": find_text,
                            "matchCase": False
                        },
                        "replaceText": replace_text
                    }
                }]
                result = _get_docs().documents().batchUpdate(
                    documentId=doc_id, body={"requests": requests}
                ).execute()
                count = 0
                for reply in result.get("replies", []):
                    count += reply.get("replaceAllText", {}).get(
                        "occurrencesChanged", 0)
                _ws_eval(tab, "window.location.reload()")
                return True, (f"Replaced {count} occurrence(s) of "
                              f"\"{find_text}\" with \"{replace_text}\".")
            except Exception as e:
                print(f"[gdocs] API find/replace failed: {e}")

    # Fallback: open Find & Replace with real Ctrl+H
    if not tab:
        return False, "No Google Doc is open."
    _debug_json(f"/json/activate/{tab['id']}")
    _cdp_key_combo(tab, 'h', ctrl=True)
    return True, (f"Opened Find & Replace — look for "
                  f"\"{find_text}\" → \"{replace_text}\".")


def rename_document(new_name: str) -> Tuple[bool, str]:
    """Rename the current document. Uses API if possible."""
    if not new_name:
        return False, "What should I rename it to?"

    tab = _find_docs_tab()
    if tab:
        import re
        url = tab.get("url", "")
        m = re.search(r"/document/d/([a-zA-Z0-9_-]+)", url)
        if m and _get_drive():
            doc_id = m.group(1)
            try:
                _get_drive().files().update(
                    fileId=doc_id,
                    body={"name": new_name}
                ).execute()
                _ws_eval(tab, "window.location.reload()")
                return True, f"Renamed to \"{new_name}\"."
            except Exception as e:
                print(f"[gdocs] API rename failed: {e}")

    if not tab:
        return False, "No Google Doc is open."
    _debug_json(f"/json/activate/{tab['id']}")
    js = f"""
    (function() {{
        var title = document.querySelector('.docs-title-input');
        if (!title) return 'no_title';
        title.focus();
        title.select();
        document.execCommand('selectAll');
        document.execCommand('insertText', false, '{new_name}');
        return 'renamed';
    }})()
    """
    result = _ws_eval(tab, js)
    if result == "renamed":
        return True, f"Renamed to \"{new_name}\"."
    return False, "I couldn't find the document title to rename."


def export_pdf() -> Tuple[bool, str]:
    """Download the document as PDF. Uses API if possible."""
    tab = _find_docs_tab()
    if not tab:
        return False, "No Google Doc is open."

    import re
    url = tab.get("url", "")
    m = re.search(r"/document/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        return False, "I couldn't identify the document ID."
    doc_id = m.group(1)

    drive = _get_drive()
    if drive:
        try:
            meta = drive.files().get(fileId=doc_id, fields="name").execute()
            filename = meta.get("name", "document") + ".pdf"
            from googleapiclient.http import MediaIoBaseDownload  # type: ignore
            import io
            request = drive.files().export_media(
                fileId=doc_id, mimeType="application/pdf"
            )
            downloads = os.path.join(os.path.expanduser("~"), "Downloads")
            filepath = os.path.join(downloads, filename)
            fh = io.FileIO(filepath, "wb")
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.close()
            return True, f"Exported \"{filename}\" to Downloads."
        except Exception as e:
            print(f"[gdocs] API export failed: {e}")

    # Fallback: URL trick
    _debug_json(f"/json/activate/{tab['id']}")
    js = f"""
    (function() {{
        window.open(
            'https://docs.google.com/document/d/{doc_id}/export?format=pdf',
            '_blank');
        return 'exporting';
    }})()
    """
    result = _ws_eval(tab, js)
    if result == "exporting":
        return True, "Downloading as PDF."
    return False, "I couldn't export this document."


def insert_text(text: str) -> Tuple[bool, str]:
    """Insert text into the current document via the Docs API (appends at end)."""
    if not text:
        return False, "What should I type?"

    tab = _find_docs_tab()
    if not tab:
        return False, "No Google Doc is open."

    import re as _re
    url = tab.get("url", "")
    m = _re.search(r"/document/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        return False, "I couldn't identify the document."

    docs = _get_docs()
    if docs:
        doc_id = m.group(1)
        try:
            doc = docs.documents().get(documentId=doc_id).execute()
            body_content = doc.get("body", {}).get("content", [{}])
            end_index = body_content[-1].get("endIndex", 1) - 1
            if end_index < 1:
                end_index = 1
            requests = [{
                "insertText": {
                    "location": {"index": end_index},
                    "text": text + "\n"
                }
            }]
            docs.documents().batchUpdate(
                documentId=doc_id, body={"requests": requests}
            ).execute()
            _ws_eval(tab, "window.location.reload()")
            preview = text[:50] + ("..." if len(text) > 50 else "")
            return True, f"Inserted \"{preview}\"."
        except Exception as e:
            print(f"[gdocs] API insert_text failed: {e}")

    return False, "I couldn't insert text — make sure you're signed in."


def share_document() -> Tuple[bool, str]:
    """Open the Share dialog via JS click (Share button is a standard DOM button)."""
    tab = _find_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    _debug_json(f"/json/activate/{tab['id']}")
    js = """
    (function() {
        var btn = document.querySelector('[data-tooltip="Share"]') ||
                  document.querySelector('.docs-titlebar-share-client-button') ||
                  document.querySelector('[aria-label="Share"]') ||
                  document.querySelector('[aria-label="Share. Private to only me."]');
        if (btn) { btn.click(); return 'opened'; }
        // Try broader search
        var all = document.querySelectorAll('[role="button"]');
        for (var b of all) {
            var lbl = (b.getAttribute('aria-label') || '').toLowerCase();
            if (lbl.startsWith('share')) { b.click(); return 'opened'; }
        }
        return 'no_btn';
    })()
    """
    result = _ws_eval(tab, js)
    if result == "opened":
        return True, "Share dialog opened."
    return False, "I couldn't find the Share button."


# ---------- Formatting (CDP keyboard shortcut-based) ----------
#
# Google Docs keyboard shortcuts (same on Windows/Linux/ChromeOS):
#   Bold:            Ctrl+B
#   Italic:          Ctrl+I
#   Underline:       Ctrl+U
#   Strikethrough:   Alt+Shift+5
#   Heading 1-6:     Ctrl+Alt+1-6
#   Normal text:     Ctrl+Alt+0
#   Bullet list:     Ctrl+Shift+8
#   Numbered list:   Ctrl+Shift+7
#   Insert comment:  Ctrl+Alt+M
#   Insert link:     Ctrl+K
#   Align left:      Ctrl+Shift+L
#   Align center:    Ctrl+Shift+E
#   Align right:     Ctrl+Shift+R
#   Align justify:   Ctrl+Shift+J
#   Clear format:    Ctrl+\
#   Undo:            Ctrl+Z
#   Redo:            Ctrl+Y
#   Increase font:   Ctrl+Shift+.
#   Decrease font:   Ctrl+Shift+,

def toggle_bold() -> Tuple[bool, str]:
    """Toggle bold (Ctrl+B)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    if _cdp_key_combo(tab, 'b', ctrl=True):
        return True, "Toggled bold."
    return False, "Couldn't toggle bold."


def toggle_italic() -> Tuple[bool, str]:
    """Toggle italic (Ctrl+I)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    if _cdp_key_combo(tab, 'i', ctrl=True):
        return True, "Toggled italic."
    return False, "Couldn't toggle italic."


def toggle_underline() -> Tuple[bool, str]:
    """Toggle underline (Ctrl+U)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    if _cdp_key_combo(tab, 'u', ctrl=True):
        return True, "Toggled underline."
    return False, "Couldn't toggle underline."


def toggle_strikethrough() -> Tuple[bool, str]:
    """Toggle strikethrough (Alt+Shift+5)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    if _cdp_key_combo(tab, '5', alt=True, shift=True):
        return True, "Toggled strikethrough."
    return False, "Couldn't toggle strikethrough."


def insert_heading(level: int = 2) -> Tuple[bool, str]:
    """Apply heading style 1-6 (Ctrl+Alt+1-6), or normal text (Ctrl+Alt+0)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    level = max(0, min(6, level))
    if _cdp_key_combo(tab, str(level), ctrl=True, alt=True):
        if level == 0:
            return True, "Applied Normal text."
        return True, f"Applied Heading {level}."
    return False, f"Couldn't apply Heading {level}."


def toggle_bullets() -> Tuple[bool, str]:
    """Toggle bullet list (Ctrl+Shift+8)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    if _cdp_key_combo(tab, '8', ctrl=True, shift=True):
        return True, "Toggled bullet list."
    return False, "Couldn't toggle bullets."


def toggle_numbered_list() -> Tuple[bool, str]:
    """Toggle numbered list (Ctrl+Shift+7)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    if _cdp_key_combo(tab, '7', ctrl=True, shift=True):
        return True, "Toggled numbered list."
    return False, "Couldn't toggle numbered list."


def insert_comment() -> Tuple[bool, str]:
    """Open comment dialog (Ctrl+Alt+M)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    if _cdp_key_combo(tab, 'm', ctrl=True, alt=True):
        return True, "Comment dialog opened — type your comment."
    return False, "Couldn't open comment dialog."


def insert_link() -> Tuple[bool, str]:
    """Open Insert Link dialog (Ctrl+K)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    if _cdp_key_combo(tab, 'k', ctrl=True):
        return True, "Link dialog opened — paste or type the URL."
    return False, "Couldn't open link dialog."


def set_alignment(align: str = "center") -> Tuple[bool, str]:
    """Set text alignment: left (Ctrl+Shift+L), center (E), right (R), justify (J)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    align_keys = {
        "left":    "l",
        "center":  "e",
        "right":   "r",
        "justify": "j",
    }
    key = align_keys.get(align, "e")
    if _cdp_key_combo(tab, key, ctrl=True, shift=True):
        return True, f"Aligned text to {align}."
    return False, f"Couldn't set {align} alignment."


def clear_formatting() -> Tuple[bool, str]:
    r"""Clear formatting (Ctrl+\\)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    if _cdp_key_combo(tab, '\\', ctrl=True):
        return True, "Cleared formatting."
    return False, "Couldn't clear formatting."


def undo() -> Tuple[bool, str]:
    """Undo (Ctrl+Z)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    if _cdp_key_combo(tab, 'z', ctrl=True):
        return True, "Undone."
    return False, "Couldn't undo."


def redo() -> Tuple[bool, str]:
    """Redo (Ctrl+Y)."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    if _cdp_key_combo(tab, 'y', ctrl=True):
        return True, "Redone."
    return False, "Couldn't redo."


def change_font_size(size: int = 0, direction: str = "") -> Tuple[bool, str]:
    """Change font size — increase (Ctrl+Shift+.), decrease (Ctrl+Shift+,),
    or set to a specific number by typing into the font size box."""
    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."

    if direction == "increase":
        if _cdp_key_combo(tab, '.', ctrl=True, shift=True):
            return True, "Increased font size."
        return False, "Couldn't increase font size."

    if direction == "decrease":
        if _cdp_key_combo(tab, ',', ctrl=True, shift=True):
            return True, "Decreased font size."
        return False, "Couldn't decrease font size."

    if size > 0:
        # Use CDP real mouse click on the font size box, then type the value.
        # JS .focus() doesn't work — Google Docs needs a real click.
        font_size_js = """
        (function() {
            // Try aria-label "Font size" first
            var el = document.querySelector('[aria-label="Font size"]');
            if (el) {
                var rect = el.getBoundingClientRect();
                return JSON.stringify({x: rect.x + rect.width/2,
                                       y: rect.y + rect.height/2});
            }
            // Try finding the font size combo button
            var combos = document.querySelectorAll('.goog-toolbar-combo-button');
            for (var c of combos) {
                var lbl = (c.getAttribute('aria-label') || '');
                if (lbl.includes('Font size')) {
                    var rect = c.getBoundingClientRect();
                    return JSON.stringify({x: rect.x + rect.width/2,
                                           y: rect.y + rect.height/2});
                }
            }
            // Last resort: find the element showing the current size (a number)
            var inputs = document.querySelectorAll(
                '.goog-toolbar-combo-button input');
            for (var inp of inputs) {
                if (/^\d+$/.test(inp.value.trim())) {
                    var rect = inp.getBoundingClientRect();
                    return JSON.stringify({x: rect.x + rect.width/2,
                                           y: rect.y + rect.height/2});
                }
            }
            return null;
        })()
        """
        result = _ws_eval(tab, font_size_js)
        if result:
            try:
                pos = json.loads(result)
                # Real CDP click on the font size box
                _cdp_click_at(tab, pos["x"], pos["y"])
                time.sleep(0.3)
                # Select all text in the now-focused input
                _cdp_key_combo(tab, 'a', ctrl=True)
                time.sleep(0.1)
                # Type the new size
                _cdp_type_text(tab, str(size))
                time.sleep(0.1)
                # Press Enter to apply
                _cdp_press_enter(tab)
                time.sleep(0.2)
                # Click back into the document body so future commands work
                doc_body_js = """
                (function() {
                    var page = document.querySelector('.kix-page-content-wrapper');
                    if (page) {
                        var rect = page.getBoundingClientRect();
                        return JSON.stringify({x: rect.x + rect.width/2,
                                               y: rect.y + 100});
                    }
                    return null;
                })()
                """
                body_pos = _ws_eval(tab, doc_body_js)
                if body_pos:
                    bp = json.loads(body_pos)
                    _cdp_click_at(tab, bp["x"], bp["y"])
                return True, f"Font size set to {size}."
            except Exception as e:
                print(f"[gdocs] Font size error: {e}")
        return False, "I couldn't find the font size input."

    return False, "Specify a size (e.g. 'font size 14') or say 'bigger' / 'smaller'."


def set_text_color(color: str) -> Tuple[bool, str]:
    """Change text color using CDP mouse clicks on toolbar button + palette."""
    if not color:
        return False, "What color? Try 'make the text red'."

    tab = _ensure_docs_tab()
    if not tab:
        return False, "No Google Doc is open."

    color_map = {
        "red":     (255, 0, 0),
        "blue":    (0, 0, 255),
        "green":   (0, 128, 0),
        "black":   (0, 0, 0),
        "white":   (255, 255, 255),
        "orange":  (255, 165, 0),
        "purple":  (128, 0, 128),
        "yellow":  (255, 255, 0),
        "pink":    (255, 105, 180),
        "gray":    (128, 128, 128),
        "grey":    (128, 128, 128),
        "brown":   (140, 69, 19),
        "cyan":    (0, 255, 255),
        "magenta": (255, 0, 255),
    }

    if color.lower() not in color_map:
        return False, (f"I don't know the color '{color}'. "
                       "Try red, blue, green, orange, purple, etc.")

    r_int, g_int, b_int = color_map[color.lower()]

    # Step 1: Find the text color button and get its position
    btn_js = """
    (function() {
        var btns = document.querySelectorAll(
            '[role="button"], .goog-toolbar-button,  .goog-toolbar-menu-button');
        for (var btn of btns) {
            var lbl = (btn.getAttribute('aria-label')
                    || btn.getAttribute('data-tooltip') || '').toLowerCase();
            if (lbl.includes('text color') && !lbl.includes('highlight')) {
                var rect = btn.getBoundingClientRect();
                return JSON.stringify({x: rect.x + rect.width/2,
                                       y: rect.y + rect.height/2});
            }
        }
        return null;
    })()
    """
    btn_pos = _ws_eval(tab, btn_js)
    if not btn_pos:
        return False, ("I couldn't find the text color button. "
                       "Use the toolbar's A button with the color bar.")

    try:
        pos = json.loads(btn_pos)
    except Exception:
        return False, "I couldn't locate the text color button."

    # Step 2: CDP real click on the text color button to open the palette
    _cdp_click_at(tab, pos["x"], pos["y"])
    time.sleep(0.6)

    # Step 3: Find the closest color in the palette and get its position
    color_js = f"""
    (function() {{
        var cells = document.querySelectorAll(
            '[role="gridcell"], .goog-palette-cell,  '
            + '.docs-material-colorpalette-cell, '
            + '.docs-colormenuitems [role="listitem"]');
        var target_r = {r_int}, target_g = {g_int}, target_b = {b_int};
        var best = null, bestDist = 999999;
        for (var cell of cells) {{
            var el = cell.querySelector('div') || cell;
            var bg = window.getComputedStyle(el).backgroundColor;
            var match = bg.match(/rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)/);
            if (match) {{
                var r = parseInt(match[1]), g = parseInt(match[2]),
                    b = parseInt(match[3]);
                var dist = Math.abs(r - target_r)
                         + Math.abs(g - target_g)
                         + Math.abs(b - target_b);
                if (dist < bestDist) {{ bestDist = dist; best = cell; }}
            }}
        }}
        if (best && bestDist < 200) {{
            var rect = best.getBoundingClientRect();
            return JSON.stringify({{x: rect.x + rect.width/2,
                                    y: rect.y + rect.height/2,
                                    dist: bestDist}});
        }}
        return null;
    }})()
    """
    color_pos = _ws_eval(tab, color_js)
    if color_pos:
        try:
            cpos = json.loads(color_pos)
            # Step 4: CDP real click on the color cell
            _cdp_click_at(tab, cpos["x"], cpos["y"])
            return True, f"Changed text color to {color}."
        except Exception as e:
            print(f"[gdocs] Color click error: {e}")

    # If palette didn't open or color not found, close by pressing Escape
    _cdp_key_combo(tab, '\\', ctrl=False)  # won't work, use Escape
    # Send Escape key
    ws_url = tab.get("webSocketDebuggerUrl")
    if ws_url:
        try:
            import websocket  # type: ignore
            ws = websocket.create_connection(ws_url, timeout=3)
            for etype in ["rawKeyDown", "keyUp"]:
                ws.send(json.dumps({
                    "id": 1,
                    "method": "Input.dispatchKeyEvent",
                    "params": {"type": etype,
                               "windowsVirtualKeyCode": 27,
                               "key": "Escape", "code": "Escape",
                               "modifiers": 0}
                }))
                ws.recv()
            ws.close()
        except Exception:
            pass

    return False, (f"I couldn't find {color} in the color palette. "
                   f"Try selecting text and using the text color button manually.")