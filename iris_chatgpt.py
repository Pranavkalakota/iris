"""
iris_chatgpt.py — hands-free ChatGPT (chatgpt.com) for IRIS.

Same browser-automation pattern as iris_youtube.py: talk to IRIS's debug Chrome
over the DevTools JSON API (port 9222), find the ChatGPT tab, and run small JS
snippets in it via a DevTools WebSocket (Runtime.evaluate). Opening/closing the
ChatGPT *tab* is still M2's job (open_app "ChatGPT"); this module performs the
in-page actions.

Every action returns (ok: bool, message: str) for IRIS's chat pane.

Reliability note: ChatGPT's page markup changes often, so the DOM-dependent
actions use several selector fallbacks. If a button moves, the copy/regenerate/
upload/search actions may need a selector tweak — the ask/new/continue family is
the most stable.

Requires: websocket-client (same dep iris_youtube uses).
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Optional, Tuple

DEBUG_PORT = 9222
CHATGPT_URL = "https://chatgpt.com/"
_no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ── Chrome debug helpers (same shape as iris_youtube) ────────────────────────
def _debug_json(path: str, method: str = "GET", timeout: float = 2.0):
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{DEBUG_PORT}{path}", method=method)
        with _no_proxy.open(req, timeout=timeout) as r:
            body = r.read()
        return json.loads(body) if body else True
    except Exception:
        return None


def _find_gpt_tab() -> Optional[dict]:
    tabs = _debug_json("/json")
    if not tabs:
        return None
    for t in tabs:
        url = t.get("url", "")
        if t.get("type") == "page" and (
                "chatgpt.com" in url or "chat.openai.com" in url):
            return t
    return None


def _ws_eval(tab: dict, js: str, timeout: float = 5.0):
    """Evaluate JS in a Chrome tab via the DevTools WebSocket."""
    ws_url = tab.get("webSocketDebuggerUrl")
    if not ws_url:
        return None
    try:
        import websocket  # type: ignore
        ws = websocket.create_connection(ws_url, timeout=timeout)
        ws.send(json.dumps({
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {"expression": js, "returnByValue": True,
                       "awaitPromise": True},
        }))
        resp = json.loads(ws.recv())
        ws.close()
        return resp.get("result", {}).get("result", {}).get("value")
    except Exception as e:
        print(f"[chatgpt] ws_eval failed: {e}")
        return None


def _ensure_tab(open_if_missing: bool = True) -> Optional[dict]:
    """Return the ChatGPT tab, opening one (and giving it a moment to load) if
    none exists yet."""
    tab = _find_gpt_tab()
    if tab or not open_if_missing:
        return tab
    _debug_json(f"/json/new?{CHATGPT_URL}", method="PUT")
    for _ in range(15):                       # up to ~6s for the tab to appear
        time.sleep(0.4)
        tab = _find_gpt_tab()
        if tab:
            return tab
    return None


# ── core: send a prompt into the ChatGPT composer ────────────────────────────
def _send_prompt(prompt: str) -> Tuple[bool, str]:
    tab = _ensure_tab()
    if not tab:
        return False, "I couldn't reach a ChatGPT tab (is the debug Chrome running?)."
    _debug_json(f"/json/activate/{tab['id']}")

    p = json.dumps(prompt)                     # safe JS string literal
    set_js = (
        "(function(){"
        "  var box=document.querySelector('#prompt-textarea')"
        "        ||document.querySelector('textarea[data-testid=\"prompt-textarea\"]')"
        "        ||document.querySelector('main textarea')"
        "        ||document.querySelector('textarea');"
        "  if(!box) return 'no-input';"
        "  box.focus();"
        "  if(box.tagName==='TEXTAREA'){"
        "    var set=Object.getOwnPropertyDescriptor("
        "        window.HTMLTextAreaElement.prototype,'value').set;"
        "    set.call(box, PROMPT);"
        "    box.dispatchEvent(new Event('input',{bubbles:true}));"
        "  } else {"                            # contenteditable (ProseMirror)
        "    box.innerHTML='';"
        "    document.execCommand('insertText', false, PROMPT);"
        "    box.dispatchEvent(new Event('input',{bubbles:true}));"
        "  }"
        "  return 'set';"
        "})()"
    ).replace("PROMPT", p)
    r1 = _ws_eval(tab, set_js)
    if r1 == "no-input":
        return False, "I couldn't find the ChatGPT input box on the page."

    time.sleep(0.4)                            # let the send button enable
    click_js = (
        "(function(){"
        "  var b=document.querySelector('[data-testid=\"send-button\"]')"
        "      ||document.querySelector('button[aria-label*=\"Send\" i]')"
        "      ||document.querySelector('form button[type=\"submit\"]');"
        "  if(b){b.click(); return 'sent';}"
        "  return 'no-send';"
        "})()"
    )
    r2 = _ws_eval(tab, click_js)
    if r2 == "sent":
        return True, "Asked ChatGPT."
    # fallback: some builds submit on Enter within the composer
    return True, "Sent your prompt to ChatGPT (if it didn't send, press Enter)."


# ── public actions ───────────────────────────────────────────────────────────
def ask(question: str) -> Tuple[bool, str]:
    if not question or not question.strip():
        return False, "What should I ask ChatGPT?"
    ok, _ = _send_prompt(question.strip())
    return ok, (f"Asked ChatGPT: “{question.strip()[:80]}”." if ok
                else "I couldn't send that to ChatGPT.")


def generate_code(request: str) -> Tuple[bool, str]:
    if not request or not request.strip():
        return False, "What code should ChatGPT write?"
    return _send_prompt(request.strip())


def summarize() -> Tuple[bool, str]:
    ok, _ = _send_prompt("Summarize this conversation concisely.")
    return ok, ("Asked ChatGPT to summarize the conversation." if ok
                else "I couldn't send that to ChatGPT.")


def rewrite(text: str = "") -> Tuple[bool, str]:
    prompt = ("Rewrite this professionally: " + text.strip()) if text.strip() \
        else "Rewrite my previous message professionally."
    ok, _ = _send_prompt(prompt)
    return ok, ("Asked ChatGPT to rewrite it professionally." if ok
                else "I couldn't send that to ChatGPT.")


def new_chat() -> Tuple[bool, str]:
    tab = _ensure_tab()
    if not tab:
        return False, "I couldn't reach a ChatGPT tab."
    _debug_json(f"/json/activate/{tab['id']}")
    # navigating to the root URL starts a fresh conversation
    _ws_eval(tab, f"window.location.href={json.dumps(CHATGPT_URL)}")
    return True, "Started a new ChatGPT conversation."


def continue_chat() -> Tuple[bool, str]:
    """Bring the existing ChatGPT tab forward — the last conversation is there."""
    tab = _find_gpt_tab()
    if not tab:
        return False, "You don't have a ChatGPT chat open to continue."
    _debug_json(f"/json/activate/{tab['id']}")
    return True, "Back to your ChatGPT conversation."


def copy_last() -> Tuple[bool, str]:
    tab = _find_gpt_tab()
    if not tab:
        return False, "No ChatGPT chat open."
    _debug_json(f"/json/activate/{tab['id']}")
    js = (
        "(function(){"
        "  var m=document.querySelectorAll('[data-message-author-role=\"assistant\"]');"
        "  if(!m.length) return '';"
        "  var t=(m[m.length-1].innerText||'').trim();"
        "  try{ if(navigator.clipboard) navigator.clipboard.writeText(t); }catch(e){}"
        "  return t.slice(0,4000);"
        "})()"
    )
    txt = _ws_eval(tab, js)
    if not txt:
        return False, "I couldn't find a ChatGPT answer to copy."
    snippet = txt[:90] + ("…" if len(txt) > 90 else "")
    return True, f"Copied ChatGPT's last answer to your clipboard. (“{snippet}”)"


def regenerate() -> Tuple[bool, str]:
    tab = _find_gpt_tab()
    if not tab:
        return False, "No ChatGPT chat open."
    _debug_json(f"/json/activate/{tab['id']}")
    js = (
        "(function(){"
        "  var b=document.querySelector('button[aria-label*=\"Regenerate\" i]')"
        "      ||document.querySelector('[data-testid=\"regenerate-button\"]');"
        "  if(b){b.click(); return 'ok';}"
        # fallback: open the last turn's action menu, then click a Regenerate item
        "  var items=[].slice.call(document.querySelectorAll('button, [role=\"menuitem\"]'));"
        "  var hit=items.find(function(e){return /regenerate|try again/i.test(e.innerText||e.getAttribute('aria-label')||'');});"
        "  if(hit){hit.click(); return 'ok';}"
        "  return 'no-btn';"
        "})()"
    )
    r = _ws_eval(tab, js)
    return (True, "Asked ChatGPT to regenerate the response.") if r == "ok" \
        else (False, "I couldn't find the regenerate button — it may be under "
                     "the '…' menu on the last answer.")


def search_chats(query: str) -> Tuple[bool, str]:
    tab = _ensure_tab()
    if not tab:
        return False, "I couldn't reach a ChatGPT tab."
    _debug_json(f"/json/activate/{tab['id']}")
    q = json.dumps(query or "")
    js = (
        "(function(){"
        "  var b=document.querySelector('button[aria-label*=\"Search\" i]')"
        "      ||document.querySelector('[data-testid=\"search-chats-button\"]');"
        "  if(b){b.click(); return 'opened';}"
        "  return 'no-btn';"
        "})()"
    )
    r = _ws_eval(tab, js)
    if r != "opened":
        return False, ("I couldn't open ChatGPT's chat search — use the search "
                       "icon in the sidebar.")
    time.sleep(0.4)
    type_js = (
        "(function(){"
        "  var i=document.querySelector('input[type=\"search\"]')"
        "      ||document.querySelector('input[placeholder*=\"Search\" i]');"
        "  if(!i) return 'no-input';"
        "  i.focus();"
        "  var set=Object.getOwnPropertyDescriptor("
        "      window.HTMLInputElement.prototype,'value').set;"
        "  set.call(i, QUERY);"
        "  i.dispatchEvent(new Event('input',{bubbles:true}));"
        "  return 'typed';"
        "})()"
    ).replace("QUERY", q)
    _ws_eval(tab, type_js)
    return True, f"Searching your ChatGPT chats for “{query}”."


def upload_file() -> Tuple[bool, str]:
    """Click the attach button so the OS file picker opens. JS can't choose the
    file for you (browser security), so you pick it once the dialog appears."""
    tab = _find_gpt_tab()
    if not tab:
        return False, "Open a ChatGPT chat first, then say 'upload a file'."
    _debug_json(f"/json/activate/{tab['id']}")
    js = (
        "(function(){"
        "  var b=document.querySelector('button[aria-label*=\"Attach\" i]')"
        "      ||document.querySelector('input[type=\"file\"]')"
        "      ||document.querySelector('[data-testid*=\"attach\" i]');"
        "  if(b){b.click(); return 'ok';}"
        "  return 'no-btn';"
        "})()"
    )
    r = _ws_eval(tab, js)
    return (True, "Opened ChatGPT's file picker — choose your file in the dialog.") \
        if r == "ok" else (False, "I couldn't find the attach button on ChatGPT.")


# ── dispatch by router intent (used by iris_gui) ─────────────────────────────
def handle(intent, entities: Optional[dict] = None) -> Optional[str]:
    """Map a gpt_* intent to an action. Returns a chat-pane message, or None if
    the intent isn't ours."""
    ent = entities or {}
    kind = getattr(intent, "intent", intent) if not isinstance(intent, str) else intent
    q = (ent.get("query") or "").strip()
    fn = {
        "gpt_ask": lambda: ask(q),
        "gpt_code": lambda: generate_code(q),
        "gpt_summarize": summarize,
        "gpt_rewrite": lambda: rewrite(q),
        "gpt_new": new_chat,
        "gpt_continue": continue_chat,
        "gpt_copy": copy_last,
        "gpt_regenerate": regenerate,
        "gpt_search": lambda: search_chats(q),
        "gpt_upload": upload_file,
    }.get(kind)
    if fn is None:
        return None
    ok, msg = fn()
    return msg


if __name__ == "__main__":
    print("iris_chatgpt — actions:", [n for n in dir() if n[0] != "_"
          and callable(globals()[n])])
    print("Live actions need the debug Chrome + a ChatGPT tab.")