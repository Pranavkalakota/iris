"""
iris_email.py — Gmail read-only lookup for IRIS chat commands.

Handles OAuth (via credentials.json / cached token.json), fetching unread
messages, ordinal position within the unread list, and topic search using
Gmail's own query syntax. No send/delete capability — readonly scope only.
"""

from __future__ import annotations

import os
import re
import base64
import html
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

DEFAULT_CREDENTIALS_PATH = "credentials.json"
DEFAULT_TOKEN_PATH = "token.json"


@dataclass
class EmailMessage:
    id: str
    sender: str
    subject: str
    date: str          # raw header string, e.g. "Tue, 1 Jul 2026 10:03:00 -0700"
    internal_ts: int    # ms since epoch, from Gmail's internalDate — used for sorting
    body: str = ""
    snippet: str = ""

    def when(self) -> str:
        try:
            dt = datetime.fromtimestamp(self.internal_ts / 1000)
            return dt.strftime("%b %d, %Y %I:%M %p")
        except Exception:
            return self.date


def _decode_part_data(data: str) -> str:
    """Gmail base64url-encodes body data with possibly missing padding."""
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _clean_body_text(text: str) -> str:
    """Strip zero-width characters marketing emails pad bodies with (used
    to dodge spam-filter text matching), drop the raw tracking URLs that
    plain-text email alternatives auto-append after every link title in
    the form 'Link Title (https://long-tracking-url...)' -- keeping the
    title, dropping the parenthetical -- and collapse the resulting runs
    of whitespace they leave behind."""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # "(https://...)" -- non-greedy up to the next ")" so it doesn't eat
    # past the URL into unrelated following text.
    text = re.sub(r"\(https?://[^)]*\)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_html(text: str) -> str:
    """Minimal HTML→text fallback for messages with no text/plain part."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text,
                  flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p>", "\n\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return _clean_body_text(html.unescape(text))


def _extract_body(payload: dict) -> str:
    """Walk the (possibly nested multipart) MIME payload and return the
    best body text found. Prefers text/plain; falls back to text/html
    stripped of tags; falls back to the top-level body if there are no
    parts at all (simple non-multipart messages)."""
    def walk(part: dict) -> dict:
        """Returns {'plain': str|None, 'html': str|None} found under this part."""
        found = {"plain": None, "html": None}
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")
        if data:
            if mime == "text/plain":
                found["plain"] = _decode_part_data(data)
            elif mime == "text/html":
                found["html"] = _decode_part_data(data)
        for sub in part.get("parts", []) or []:
            sub_found = walk(sub)
            if sub_found["plain"] and not found["plain"]:
                found["plain"] = sub_found["plain"]
            if sub_found["html"] and not found["html"]:
                found["html"] = sub_found["html"]
        return found

    found = walk(payload)
    if found["plain"]:
        return _clean_body_text(found["plain"])
    if found["html"]:
        return _strip_html(found["html"])
    return ""


def _header(headers: list, name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


class EmailStore:
    def __init__(self, credentials_path: str = DEFAULT_CREDENTIALS_PATH,
                 token_path: str = DEFAULT_TOKEN_PATH):
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._service = None

    def _authenticate(self):
        creds = None
        if os.path.exists(self._token_path):
            creds = Credentials.from_authorized_user_file(
                self._token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self._credentials_path):
                    raise FileNotFoundError(
                        f"credentials.json not found at "
                        f"{self._credentials_path!r} — see Google Cloud "
                        f"Console setup steps.")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self._credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self._token_path, "w") as f:
                f.write(creds.to_json())
        return creds

    def _get_service(self):
        if self._service is None:
            creds = self._authenticate()
            self._service = build("gmail", "v1", credentials=creds)
        return self._service

    def _fetch_full(self, msg_id: str) -> EmailMessage:
        svc = self._get_service()
        msg = svc.users().messages().get(
            userId="me", id=msg_id, format="full").execute()
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        return EmailMessage(
            id=msg_id,
            sender=_header(headers, "From"),
            subject=_header(headers, "Subject"),
            date=_header(headers, "Date"),
            internal_ts=int(msg.get("internalDate", 0)),
            body=_extract_body(payload),
            snippet=msg.get("snippet", ""),
        )

    def _list_ids(self, query: str, max_results: int = 10) -> list:
        svc = self._get_service()
        resp = svc.users().messages().list(
            userId="me", q=query, maxResults=max_results).execute()
        return [m["id"] for m in resp.get("messages", [])]

    def unread(self, limit: int = 10) -> list[EmailMessage]:
        """Most recent unread messages, newest first, restricted to the
        Primary tab. Plain 'is:unread in:inbox' ignores Gmail's tab
        categorization entirely and will happily surface Promotions-tab
        marketing mail as the 'latest unread' — not what someone means by
        'read my email'. category:primary matches what's actually shown
        under the Primary tab. Sorted explicitly by internalDate rather
        than trusting API list order — same reasoning as resolving
        'latest video' in Python instead of leaving order to an LLM."""
        ids = self._list_ids("is:unread in:inbox category:primary",
                              max_results=limit)
        msgs = [self._fetch_full(i) for i in ids]
        msgs.sort(key=lambda m: m.internal_ts, reverse=True)
        return msgs

    def search(self, topic: str, limit: int = 5) -> list[EmailMessage]:
        """Gmail's own search (subject + body + sender), newest match first."""
        ids = self._list_ids(topic, max_results=limit)
        msgs = [self._fetch_full(i) for i in ids]
        msgs.sort(key=lambda m: m.internal_ts, reverse=True)
        return msgs

    def latest(self, limit: int = 10) -> list[EmailMessage]:
        """Most recent messages newest first, restricted to the Primary
        tab — same category:primary reasoning as unread() above, but with
        no is:unread filter. This is the "check email" / "read my email"
        default: read state doesn't matter, just what's newest."""
        ids = self._list_ids("in:inbox category:primary", max_results=limit)
        msgs = [self._fetch_full(i) for i in ids]
        msgs.sort(key=lambda m: m.internal_ts, reverse=True)
        return msgs

    def latest_unread(self) -> Optional[EmailMessage]:
        msgs = self.unread(limit=1)
        return msgs[0] if msgs else None

    def by_ordinal(self, index: int) -> Optional[EmailMessage]:
        """index is 0-based, counted from the most recent mail overall
        (read or unread) — matches _EMAIL_ORDINAL_WORDS in iris_query.py
        ('first'->0, 'second'->1...). Only hit when there's no active
        _last_email_list to resolve against in iris_gui.py."""
        msgs = self.latest(limit=max(index + 1, 5))
        return msgs[index] if index < len(msgs) else None