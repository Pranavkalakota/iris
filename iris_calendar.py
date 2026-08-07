"""
iris_calendar.py — hands-free Google Calendar for IRIS (read / add / remove).

Same OAuth pattern as iris_email.py (credentials.json + a cached token), just
with the Google Calendar scope. Drives Calendar by voice OR chat:

    "what's my day?"                       -> reads today's events onto the screen
    "book 30 minutes with Jack Tuesday"    -> creates a timed event
    "I have an exam on Thursday"           -> creates an all-day event
    "cancel my 3pm" / "delete my exam"     -> removes the matching event

Output is text (for IRIS's chat pane) — no spoken reply. Opening/closing the
Calendar *tab* is still M2's job (open_app "Google Calendar"); this module is
the agent that actually reads and edits the calendar itself.

ONE-TIME SETUP (see notes at bottom): enable the Google Calendar API in the same
Google Cloud project your Gmail uses, then delete calendar_token.json (if any)
so the first calendar command re-consents with the calendar scope.

The parsing helpers (parse_when / extract_title) are pure Python and are unit-
tested; the API calls need your OAuth and run on your machine.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time as _time
from typing import Optional

# Google libs are only needed for the live API calls; guard the import so the
# parser helpers still work (and can be tested) without them installed.
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    _HAS_GOOGLE = True
except Exception:
    _HAS_GOOGLE = False

SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]
DEFAULT_CREDENTIALS_PATH = "credentials.json"
DEFAULT_TOKEN_PATH = "calendar_token.json"   # separate from Gmail's token.json


# ─────────────────────────────────────────────────────────────────────────────
# Natural-language date/time parsing (pure Python, testable, no API needed)
# ─────────────────────────────────────────────────────────────────────────────
_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2,
    "wed": 2, "thursday": 3, "thu": 3, "thurs": 3, "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
_NAMED_TIMES = {"noon": (12, 0), "midday": (12, 0), "midnight": (0, 0)}
_PERIODS = {"morning": (9, 0), "afternoon": (14, 0), "evening": (18, 0),
            "tonight": (19, 0), "night": (20, 0)}
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}

DEFAULT_TIMED_HOUR = 12          # timed event with a duration but no stated time
DEFAULT_DURATION_MIN = 30        # meeting with no stated length


def _parse_duration(low: str) -> Optional[int]:
    """Minutes, from '30 min', '1 hour', 'an hour', 'half an hour', '90 minutes'."""
    if re.search(r"\bhalf (an? )?hour\b", low):
        return 30
    if re.search(r"\ban hour\b", low):
        return 60
    m = re.search(r"\b(\d+)\s*(hours?|hrs?|h)\b", low)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"\b(\d+)\s*(minutes?|mins?|m)\b", low)
    if m:
        return int(m.group(1))
    return None


def _parse_day(low: str, now: datetime) -> Optional[date]:
    if re.search(r"\btoday\b|\btonight\b", low):
        return now.date()
    if re.search(r"\btomorrow\b", low):
        return now.date() + timedelta(days=1)
    m = re.search(r"\bin (\d+) days?\b", low)
    if m:
        return now.date() + timedelta(days=int(m.group(1)))
    # explicit "march 3", "3 march"
    for name, num in _MONTHS.items():
        m = re.search(rf"\b{name}\s+(\d{{1,2}})\b", low) or \
            re.search(rf"\b(\d{{1,2}})\s+{name}\b", low)
        if m:
            day = int(m.group(1))
            year = now.year + (1 if num < now.month else 0)
            try:
                return date(year, num, day)
            except ValueError:
                pass
    # weekday name -> the coming occurrence (today counts if it matches)
    for name, wd in _WEEKDAYS.items():
        if re.search(rf"\b{name}\b", low):
            delta = (wd - now.weekday()) % 7
            if "next" in low and delta == 0:
                delta = 7
            return now.date() + timedelta(days=delta)
    return None


def _parse_time(low: str) -> Optional[tuple]:
    for word, hm in _NAMED_TIMES.items():
        if re.search(rf"\b{word}\b", low):
            return hm
    # "at 3", "3pm", "3:30 pm", "at 14:00"
    m = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", low)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or 0)
        ap = m.group(3)
        if ap == "pm" and hh < 12:
            hh += 12
        elif ap == "am" and hh == 12:
            hh = 0
        elif ap is None and hh <= 7:
            hh += 12          # bare "at 3" -> 3pm, the friendlier default
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return (hh, mm)
    for word, hm in _PERIODS.items():
        if re.search(rf"\b{word}\b", low):
            return hm
    return None


def parse_when(text: str, now: Optional[datetime] = None) -> dict:
    """Return {'start': datetime, 'end': datetime, 'all_day': bool}.

    Rule: a stated time (or morning/afternoon/evening) -> timed event; a stated
    duration but no time -> timed at a default hour; neither -> all-day."""
    now = now or datetime.now()
    low = " " + text.lower().strip() + " "

    day = _parse_day(low, now) or now.date()
    dur = _parse_duration(low)
    # strip the duration span before reading a clock time so "30 minutes" can't
    # be misread as "30 o'clock".
    low_no_dur = re.sub(r"\b\d+\s*(hours?|hrs?|minutes?|mins?|h|m)\b", " ", low)
    low_no_dur = re.sub(r"\b(half (an? )?hour|an hour)\b", " ", low_no_dur)
    tm = _parse_time(low_no_dur)

    if tm is None and dur is None:
        start = datetime.combine(day, _time(0, 0))
        return {"start": start, "end": start + timedelta(days=1), "all_day": True}

    hh, mm = tm if tm is not None else (DEFAULT_TIMED_HOUR, 0)
    start = datetime.combine(day, _time(hh, mm))
    length = dur if dur is not None else DEFAULT_DURATION_MIN
    return {"start": start, "end": start + timedelta(minutes=length),
            "all_day": False}


_CMD_STRIP = re.compile(
    r"\b(book|schedule|set ?up|add|create|put|make|new|event|meeting|appointment|"
    r"remind me to|remind me|i have|i've got|i got|there'?s|a|an|the|for|on|at|"
    r"my calendar|calendar|to)\b", re.I)
_WHEN_STRIP = re.compile(
    r"\b(today|tonight|tomorrow|next|this|monday|tuesday|tues|wednesday|wed|"
    r"thursday|thurs|friday|saturday|sunday|mon|tue|thu|fri|sat|sun|morning|"
    r"afternoon|evening|night|noon|midnight|midday|in \d+ days?|\d+\s*"
    r"(hours?|hrs?|minutes?|mins?)|half (an? )?hour|an hour|at \d[\d:apm ]*|"
    r"\d{1,2}(:\d{2})?\s*(am|pm)?)\b", re.I)


def extract_title(text: str) -> str:
    """Best-effort event title: strip command + time words, keep the subject.
    'book 30 min with Jack Tuesday' -> 'Meeting with Jack';
    'I have an exam on Thursday'      -> 'Exam'."""
    low = text.strip()
    has_with = re.search(
        r"\bwith\s+(?:the\s+|my\s+|a\s+|an\s+)?([A-Za-z][\w'-]*)", low, re.I)
    t = _WHEN_STRIP.sub(" ", low)
    t = _CMD_STRIP.sub(" ", t)
    t = re.sub(r"\bwith\b.*$", "", t, flags=re.I)     # drop trailing "with Jack"
    t = re.sub(r"\s+", " ", t).strip(" .,-")
    if has_with:
        who = has_with.group(1).strip().title()
        return (f"{t.title()} with {who}").strip() if t else f"Meeting with {who}"
    return t.title() if t else "Event"


# ─────────────────────────────────────────────────────────────────────────────
# Calendar API service
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CalEvent:
    id: str
    summary: str
    start: datetime
    end: datetime
    all_day: bool
    location: str = ""

    def when_str(self) -> str:
        if self.all_day:
            return self.start.strftime("%a %b %d") + " (all day)"
        return self.start.strftime("%a %b %d, %-I:%M %p") if os.name != "nt" \
            else self.start.strftime("%a %b %d, %I:%M %p").replace(" 0", " ")


def _to_dt(node: dict) -> tuple:
    """(datetime, all_day) from a Calendar API start/end node."""
    if "date" in node:
        return datetime.fromisoformat(node["date"]), True
    raw = node.get("dateTime", "")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None), False
    except Exception:
        return datetime.now(), False


class CalendarService:
    def __init__(self, credentials_path: str = DEFAULT_CREDENTIALS_PATH,
                 token_path: str = DEFAULT_TOKEN_PATH):
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._service = None

    # -- OAuth (identical shape to iris_email.EmailStore._authenticate) --------
    def _authenticate(self):
        if not _HAS_GOOGLE:
            raise RuntimeError("Google API libraries not installed "
                               "(google-api-python-client, google-auth-oauthlib).")
        creds = None
        if os.path.exists(self._token_path):
            creds = Credentials.from_authorized_user_file(self._token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self._credentials_path):
                    raise FileNotFoundError(
                        f"credentials.json not found at {self._credentials_path!r} "
                        "— use the same one Gmail uses.")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self._credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self._token_path, "w") as f:
                f.write(creds.to_json())
        return creds

    def _svc(self):
        if self._service is None:
            self._service = build("calendar", "v3", credentials=self._authenticate())
        return self._service

    # -- reads -----------------------------------------------------------------
    def list_events(self, start: datetime, end: datetime, max_results: int = 25):
        svc = self._svc()
        res = svc.events().list(
            calendarId="primary", timeMin=start.astimezone().isoformat(),
            timeMax=end.astimezone().isoformat(), singleEvents=True,
            orderBy="startTime", maxResults=max_results).execute()
        out = []
        for it in res.get("items", []):
            s, allday = _to_dt(it.get("start", {}))
            e, _ = _to_dt(it.get("end", {}))
            out.append(CalEvent(id=it.get("id", ""),
                                summary=it.get("summary", "(no title)"),
                                start=s, end=e, all_day=allday,
                                location=it.get("location", "")))
        return out

    # -- writes ----------------------------------------------------------------
    def create_event(self, summary, start, end, all_day=False, description=""):
        svc = self._svc()
        if all_day:
            body = {"summary": summary, "description": description,
                    "start": {"date": start.date().isoformat()},
                    "end": {"date": end.date().isoformat()}}
        else:
            body = {"summary": summary, "description": description,
                    "start": {"dateTime": start.astimezone().isoformat()},
                    "end": {"dateTime": end.astimezone().isoformat()}}
        return svc.events().insert(calendarId="primary", body=body).execute()

    def delete_event(self, event_id: str):
        self._svc().events().delete(calendarId="primary", eventId=event_id).execute()


# ─────────────────────────────────────────────────────────────────────────────
# High-level handlers (return a screen-ready message string)
# ─────────────────────────────────────────────────────────────────────────────
_singleton: Optional[CalendarService] = None


def get_calendar() -> CalendarService:
    global _singleton
    if _singleton is None:
        _singleton = CalendarService()
    return _singleton


def _fmt_event(ev: CalEvent) -> str:
    if ev.all_day:
        return f"• {ev.summary} (all day)"
    t = ev.start.strftime("%I:%M %p").lstrip("0")
    return f"• {t} — {ev.summary}"


def handle_read(text: str, now: Optional[datetime] = None) -> str:
    """'what's my day', 'what's on my calendar tomorrow', etc."""
    now = now or datetime.now()
    day = _parse_day(" " + text.lower() + " ", now) or now.date()
    start = datetime.combine(day, _time(0, 0))
    end = start + timedelta(days=1)
    label = "today" if day == now.date() else day.strftime("%A, %b %d")
    try:
        events = get_calendar().list_events(start, end)
    except Exception as e:
        return _setup_hint(e)
    if not events:
        return f"You have nothing on your calendar {label}."
    lines = "\n".join(_fmt_event(ev) for ev in events)
    n = len(events)
    return f"You have {n} thing{'s' if n != 1 else ''} {label}:\n{lines}"


def handle_create(text: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    when = parse_when(text, now)
    title = extract_title(text)
    try:
        get_calendar().create_event(title, when["start"], when["end"],
                                    all_day=when["all_day"])
    except Exception as e:
        return _setup_hint(e)
    if when["all_day"]:
        return f"Added “{title}” on {when['start'].strftime('%A, %b %d')} (all day)."
    s = when["start"].strftime("%A, %b %d at %I:%M %p").replace(" 0", " ")
    mins = int((when["end"] - when["start"]).total_seconds() // 60)
    return f"Booked “{title}” — {s} for {mins} min."


def handle_delete(text: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    # search a wide window (today .. +30 days) for a title/time match
    start = datetime.combine(now.date(), _time(0, 0))
    end = start + timedelta(days=30)
    try:
        events = get_calendar().list_events(start, end, max_results=100)
    except Exception as e:
        return _setup_hint(e)
    title = extract_title(text).lower()
    tm = _parse_time(re.sub(r"\b\d+\s*(minutes?|mins?)\b", " ", text.lower()))
    matches = []
    for ev in events:
        if title and title not in ev.summary.lower() and \
                ev.summary.lower() not in title:
            if not (tm and not ev.all_day and (ev.start.hour, ev.start.minute) == tm):
                continue
        matches.append(ev)
    if not matches:
        return "I couldn't find a matching event to remove."
    if len(matches) > 1:
        opts = "\n".join(f"• {m.summary} ({m.when_str()})" for m in matches[:5])
        return f"I found a few — which one?\n{opts}"
    ev = matches[0]
    try:
        get_calendar().delete_event(ev.id)
    except Exception as e:
        return _setup_hint(e)
    return f"Removed “{ev.summary}” ({ev.when_str()})."


def _setup_hint(err: Exception) -> str:
    msg = str(err)
    low = msg.lower()
    print("[cal] error:", msg)          # full error to the terminal for debugging
    if "credentials.json" in msg:
        return ("I can't reach your calendar — credentials.json is missing "
                "(the same file Gmail uses).")
    # API not turned on for the project (most common first-run cause). Enabling
    # the API is different from granting OAuth consent — deleting the token
    # won't help here.
    if ("has not been used" in low or "is disabled" in low
            or "accessnotconfigured" in low or "serviceusage" in low):
        return ("The Google Calendar API isn't enabled for your project yet. "
                "In Google Cloud Console → APIs & Services → Library, search "
                "'Google Calendar API' and click Enable, wait ~2 minutes, then "
                "try again. (No need to delete the token.)")
    # Consent granted, but not the edit scope.
    if ("insufficient" in low or "invalid_scope" in low
            or "insufficientpermissions" in low or "forbidden" in low):
        return ("I don't have edit access to your calendar. Delete "
                "calendar_token.json, then when the Google sign-in appears "
                "make sure BOTH calendar checkboxes are ticked before you Allow.")
    return f"I couldn't reach your calendar: {msg[:220]}"


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: exercises the parsers (no Google account needed)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    now = datetime(2026, 7, 27, 10, 0)   # a Monday, for deterministic output
    print("parse/title self-test (today = Mon 2026-07-27)\n" + "-" * 60)
    cases = [
        "book 30 minutes with Jack Tuesday",
        "book 30 min with Jack Tuesday afternoon",
        "I have an exam on Thursday",
        "schedule a dentist appointment tomorrow at 3pm",
        "set up a 1 hour meeting with the team Friday at 2",
        "remind me to call mom tonight",
    ]
    for c in cases:
        w = parse_when(c, now)
        print(f'"{c}"')
        print(f"    title = {extract_title(c)!r}")
        print(f"    when  = {w['start']:%a %b %d %I:%M%p} -> "
              f"{w['end']:%a %b %d %I:%M%p}  all_day={w['all_day']}")
    print("-" * 60)
    print("Parsing OK. (Live read/create/delete need OAuth — see setup notes.)")