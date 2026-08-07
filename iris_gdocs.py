"""
iris_gdocs.py — M2: Google Docs voice-control via Docs/Drive API + Chrome DevTools.

Hybrid approach:
  • Create, search, find/replace, rename, export → Google Docs & Drive APIs
    (reliable, works without a tab open). Falls back to Chrome navigation
    if OAuth credentials are missing.
  • UI-only actions (heading, bullets, comment, share dialog) → Chrome
    DevTools keyboard shortcut injection (these have no API equivalent that
    maps to a single voice command).

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


# ── Chrome debug helpers (fallback + UI actions) ────────────────────────────

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


# ── Public Google Docs actions ──────────────────────────────────────────────

def create_document(title: str = "") -> Tuple[bool, str]:
    """Create a new Google Doc. Uses API if available."""
    docs = _get_docs()
    if docs:
        try:
            body = {"title": title or "Untitled document"}
            doc = docs.documents().create(body=body).execute()
            doc_id = doc["documentId"]
            url = f"https://docs.google.com/document/d/{doc_id}/edit"
            _navigate_docs(url)
            name = title or "Untitled document"
            return True, f"Created \"{name}\"."
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
                # Open the first result in browser
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

    # Try API — need the document ID from the current tab URL
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
                    count += reply.get("replaceAllText", {}).get("occurrencesChanged", 0)
                # Reload the tab so changes are visible
                _ws_eval(tab, "window.location.reload()")
                return True, f"Replaced {count} occurrence(s) of \"{find_text}\" with \"{replace_text}\"."
            except Exception as e:
                print(f"[gdocs] API find/replace failed: {e}")

    # Fallback: open Find & Replace dialog via keyboard shortcut
    if not tab:
        return False, "No Google Doc is open."
    _debug_json(f"/json/activate/{tab['id']}")
    js = """
    (function() {
        var e = new KeyboardEvent('keydown', {
            key: 'h', code: 'KeyH', keyCode: 72,
            ctrlKey: true, bubbles: true
        });
        document.dispatchEvent(e);
        return 'opened';
    })()
    """
    _ws_eval(tab, js)
    return True, f"Opened Find & Replace — look for \"{find_text}\" → \"{replace_text}\"."


def insert_heading(level: int = 2) -> Tuple[bool, str]:
    """Apply a heading style (1-6) via keyboard shortcut Ctrl+Alt+<N>."""
    tab = _find_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    level = max(1, min(6, level))
    _debug_json(f"/json/activate/{tab['id']}")
    js = f"""
    (function() {{
        var e = new KeyboardEvent('keydown', {{
            key: '{level}', code: 'Digit{level}', keyCode: {48 + level},
            ctrlKey: true, altKey: true, bubbles: true
        }});
        document.dispatchEvent(e);
        return 'heading_{level}';
    }})()
    """
    _ws_eval(tab, js)
    return True, f"Applied Heading {level}."


def toggle_bullets() -> Tuple[bool, str]:
    """Toggle bullet list on selected text (Ctrl+Shift+8)."""
    tab = _find_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    _debug_json(f"/json/activate/{tab['id']}")
    js = """
    (function() {
        var e = new KeyboardEvent('keydown', {
            key: '8', code: 'Digit8', keyCode: 56,
            ctrlKey: true, shiftKey: true, bubbles: true
        });
        document.dispatchEvent(e);
        return 'toggled';
    })()
    """
    _ws_eval(tab, js)
    return True, "Toggled bullet list."


def insert_comment() -> Tuple[bool, str]:
    """Open the comment dialog (Ctrl+Alt+M)."""
    tab = _find_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    _debug_json(f"/json/activate/{tab['id']}")
    js = """
    (function() {
        var e = new KeyboardEvent('keydown', {
            key: 'm', code: 'KeyM', keyCode: 77,
            ctrlKey: true, altKey: true, bubbles: true
        });
        document.dispatchEvent(e);
        return 'comment';
    })()
    """
    _ws_eval(tab, js)
    return True, "Comment dialog opened — type your comment."


def share_document() -> Tuple[bool, str]:
    """Open the Share dialog."""
    tab = _find_docs_tab()
    if not tab:
        return False, "No Google Doc is open."
    _debug_json(f"/json/activate/{tab['id']}")
    js = """
    (function() {
        var btn = document.querySelector('[data-tooltip="Share"]') ||
                  document.querySelector('.docs-titlebar-share-client-button') ||
                  document.querySelector('[aria-label="Share"]');
        if (btn) { btn.click(); return 'opened'; }
        return 'no_btn';
    })()
    """
    result = _ws_eval(tab, js)
    if result == "opened":
        return True, "Share dialog opened."
    return False, "I couldn't find the Share button."


def rename_document(new_name: str) -> Tuple[bool, str]:
    """Rename the current document. Uses API if possible."""
    if not new_name:
        return False, "What should I rename it to?"

    # Try API — need doc ID from current tab
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
                # Reload tab so title updates
                _ws_eval(tab, "window.location.reload()")
                return True, f"Renamed to \"{new_name}\"."
            except Exception as e:
                print(f"[gdocs] API rename failed: {e}")

    # Fallback: click title and type
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
            # Get filename
            meta = drive.files().get(fileId=doc_id, fields="name").execute()
            filename = meta.get("name", "document") + ".pdf"

            # Export as PDF
            from googleapiclient.http import MediaIoBaseDownload  # type: ignore
            import io
            request = drive.files().export_media(
                fileId=doc_id, mimeType="application/pdf"
            )
            # Save to user's Downloads folder
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
        window.open('https://docs.google.com/document/d/{doc_id}/export?format=pdf', '_blank');
        return 'exporting';
    }})()
    """
    result = _ws_eval(tab, js)
    if result == "exporting":
        return True, "Downloading as PDF."
    return False, "I couldn't export this document."