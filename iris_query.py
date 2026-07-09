"""
iris_query.py — natural-language → recording-request engine for IRIS.

Pure Python, no Qt, no Ollama. Everything here is deterministic and unit
testable. The chat tab feeds in the user's text plus the list of known
recordings and gets back a structured `Intent` describing what to do.

Design goals (driven by real user transcripts):
  * Tolerate misspellings ("yesterady", "summaruze", "recodring", bad month
    spellings) via fuzzy token correction.
  * Understand many date grammars: 2026-06-17, 6/17, 6/17/2026, "june 17",
    "17 june", "17th of june", "the 6th", plus relative days (today,
    yesterday, day before yesterday, N days ago, weekday names, tomorrow).
  * Understand ranges: "from June 6 to June 10", "june 6 - 10", and index
    ranges "day 3 to day 7" / "recordings 3 to 7".
  * Match recordings by partial / messy names: "testing upload" ==
    testing_upload.wav; treat every clip as an "audio file".
  * Recognize "latest / most recent", "a random audio file", "list them".
  * Content search: "the recording where I talked about X".
  * In-recording lookups: find when a topic was discussed, or what was said
    at a given timestamp.

The chat tab keeps owning the "pending pick" disambiguation; this module is
called for everything that isn't a direct reply to a shown list.
"""

from __future__ import annotations

import os
import re
import difflib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────
Date = tuple  # (year|None, month, day)


@dataclass
class Intent:
    """What the user wants. `kind` selects the handler in the chat tab."""
    kind: str = "none"
    # kinds: latest | random | list | name | date | date_range | index_range
    #        | month | time | content_search | photo | photo_query | none
    dates: list = field(default_factory=list)          # list[Date] (single date)
    date_range: Optional[tuple] = None                  # (Date, Date)
    index_range: Optional[tuple] = None                 # (start, end) 1-based
    time: Optional[tuple] = None                        # (h, m, s|None)
    name_matches: list = field(default_factory=list)    # recordings, best first
    content_query: str = ""                             # topic for search
    summarize_all: bool = False                         # plural / "all" cue
    photo_action: str = ""                              # photo_query sub-kind:
    # "latest" | "date" | "time" | "range" | "all"
    capture_mode: str = "camera"                         # for kind=="photo":
    # "camera" (default — a photo of the person) | "screen" (screenshot)
    corrected_text: str = ""                            # after spell-fix
    note: str = ""                                      # optional debug/info


# ─────────────────────────────────────────────────────────────────────────────
# Recording datetime — mirrors ChatTab._rec_datetime so this module is
# self-contained. Accepts anything with .name and .mtime.
# ─────────────────────────────────────────────────────────────────────────────
def rec_dt(rec) -> datetime:
    name = getattr(rec, "name", "") or ""
    m = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})[ _T-]+"
                  r"(\d{2})[-:](\d{2})[-:](\d{2})", name)
    if m:
        try:
            return datetime(*[int(x) for x in m.groups()])
        except Exception:
            pass
    m = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", name)
    if m:
        try:
            return datetime(int(m[1]), int(m[2]), int(m[3]))
        except Exception:
            pass
    try:
        return datetime.fromtimestamp(getattr(rec, "mtime", 0) or 0)
    except Exception:
        return datetime.fromtimestamp(0)


# ─────────────────────────────────────────────────────────────────────────────
# Spell tolerance
# ─────────────────────────────────────────────────────────────────────────────
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}
MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
ALL_MONTHS = {**MONTHS, **MONTH_ABBR}

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

# Words we try to repair when a token is *close* to one of them. Kept small and
# specific so we never "correct" an ordinary word into one of these by mistake.
_VOCAB = [
    "summarize", "summary", "recap", "recording", "recordings", "transcript",
    "transcribe", "transcription", "audio", "yesterday", "today", "tomorrow",
    "latest", "recent", "newest", "earliest", "between", "from", "about",
    "mentioned", "talked", "discussed", "before",
] + list(MONTHS) + list(WEEKDAYS)


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def correct_text(text: str) -> str:
    """Repair near-miss spellings token by token. Only swaps a token when it is
    clearly a typo of a known word (high similarity, similar length), so real
    words and filenames pass through untouched."""
    out = []
    for tok in re.split(r"(\W+)", text):          # keep separators
        if not tok or not tok.isalpha() or len(tok) < 3:
            out.append(tok)
            continue
        low = tok.lower()
        if low in _VOCAB or low in ALL_MONTHS or low in WEEKDAYS:
            out.append(tok)
            continue
        best, best_r = None, 0.0
        for cand in _VOCAB:
            if abs(len(cand) - len(low)) > 3:
                continue
            r = _similar(low, cand)
            if r > best_r:
                best, best_r = cand, r
        # Require a strong match; longer words can tolerate a slightly lower bar.
        thresh = 0.82 if len(low) >= 6 else 0.86
        if best is not None and best_r >= thresh:
            out.append(best)
        else:
            out.append(tok)
    return "".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Name normalization + matching
# ─────────────────────────────────────────────────────────────────────────────
_EXT_WORDS = {"wav", "mp3", "m4a", "flac", "ogg", "aac", "wma", "webm", "mp4"}
_NAME_STOP = {
    "recording", "recordings", "recorded", "audio", "file", "files", "clip",
    "clips", "chunk", "the", "a", "an", "of", "for", "this", "that", "my",
    "our", "summarize", "summary", "recap", "what", "whats", "is", "in", "it",
    "transcript", "transcribe", "transcription", "and", "ch", "part",
    "conversation", "voice", "memo", "can", "you", "please", "called", "named",
    "titled", "about", "from", "show", "me", "give", "tell", "with",
}


def normalize_name(s: str) -> str:
    s = os.path.splitext(s)[0] if "." in s else s
    s = re.sub(r"[_\-.]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _name_tokens(s: str) -> list:
    toks = [t for t in re.split(r"[\s_\-.]+", s.lower()) if t]
    return [t for t in toks if t not in _EXT_WORDS]


def match_name(text: str, recordings: Sequence) -> list:
    """Return recordings whose name the text plausibly refers to, best first.

    Strategy (most to least confident):
      1. The recording's normalized name appears as a phrase in the text.
      2. The text (minus filler) appears inside the normalized name.
      3. Strong fuzzy ratio between the two normalized strings.
      4. Distinctive shared tokens (non-stopword, length >= 4, or numeric runs).
    """
    norm_text = normalize_name(text)
    text_tokens = [t for t in _name_tokens(text) if t not in _NAME_STOP]
    scored = []
    for r in recordings:
        nm = normalize_name(getattr(r, "name", ""))
        if not nm:
            continue
        score = 0.0
        if nm and nm in norm_text:
            score = max(score, 0.95 + min(len(nm), 40) / 1000)
        # text-minus-filler inside the name
        stripped = " ".join(text_tokens)
        if stripped and len(stripped) >= 3 and stripped in nm:
            score = max(score, 0.9)
        ratio = _similar(nm, norm_text)
        if ratio >= 0.6:
            score = max(score, 0.5 + ratio / 2)
        # token overlap
        nm_tokens = set(t for t in _name_tokens(nm) if t not in _NAME_STOP)
        if nm_tokens and text_tokens:
            shared = nm_tokens & set(text_tokens)
            distinct = {t for t in shared if len(t) >= 4 or t.isdigit()}
            if distinct:
                frac = len(distinct) / max(1, len(nm_tokens))
                score = max(score, 0.55 + 0.4 * frac)
        if score > 0:
            scored.append((score, rec_dt(r), r))
    if not scored:
        return []
    # best score first; newest first on ties
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = scored[0][0]
    # Keep only clearly-comparable matches (within a small band of the best).
    return [r for s, _, r in scored if s >= max(0.6, top - 0.08)]


# ─────────────────────────────────────────────────────────────────────────────
# Date / time / duration parsing
# ─────────────────────────────────────────────────────────────────────────────
def _month_from_word(word: str) -> Optional[int]:
    w = word.lower().strip(".")
    if w in ALL_MONTHS:
        return ALL_MONTHS[w]
    # fuzzy (handles "jne", "septmber")
    best, best_r = None, 0.0
    for name, num in MONTHS.items():
        r = _similar(w, name)
        if r > best_r:
            best, best_r = num, r
    return best if best_r >= 0.8 else None


def _clean_year(s) -> Optional[int]:
    if s and str(s).isdigit() and len(str(s)) == 4:
        return int(s)
    return None


def parse_relative_day(text: str, today: datetime) -> Optional[Date]:
    low = text.lower()
    if re.search(r"\bday before yesterday\b", low) or \
       re.search(r"\b2 days ago\b", low) or re.search(r"\btwo days ago\b", low):
        d = today - timedelta(days=2)
        return (d.year, d.month, d.day)
    m = re.search(r"\b(\d{1,2})\s+days?\s+ago\b", low)
    if m:
        d = today - timedelta(days=int(m.group(1)))
        return (d.year, d.month, d.day)
    if re.search(r"\byesterday\b", low):
        d = today - timedelta(days=1)
        return (d.year, d.month, d.day)
    if re.search(r"\btoday\b", low) or "my day" in low:
        return (today.year, today.month, today.day)
    if re.search(r"\btomorrow\b", low):
        d = today + timedelta(days=1)
        return (d.year, d.month, d.day)
    # "last tuesday" / bare weekday → most recent past occurrence
    m = re.search(r"\b(?:last\s+)?(" + "|".join(WEEKDAYS) + r")\b", low)
    if m:
        target = WEEKDAYS[m.group(1)]
        delta = (today.weekday() - target) % 7
        delta = delta or 7
        d = today - timedelta(days=delta)
        return (d.year, d.month, d.day)
    return None


def parse_one_date(text: str, today: Optional[datetime] = None) -> Optional[Date]:
    """Parse the first absolute date in `text`. Returns (year|None, mo, day)."""
    low = text.lower()
    # ISO 2026-06-17 / 2026/6/17
    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", low)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return (y, mo, d)
    # month-name + day (+year):  june 17, jun 17th, sept 9 2026
    m = re.search(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?"
                  r"(?:\s*,?\s*(\d{4}))?", low)
    if m and _looks_month(m[1]):
        mo = _month_from_word(m[1])
        d = int(m[2])
        if mo and 1 <= d <= 31:
            return (_clean_year(m[3]), mo, d)
    # day + (of) + month:  17 june, 17th of june, the 6th of jun
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\b", low)
    if m and _looks_month(m[2]):
        mo = _month_from_word(m[2])
        d = int(m[1])
        if mo and 1 <= d <= 31:
            return (None, mo, d)
    # numeric M/D/Y or M/D
    m = re.search(r"\b(\d{1,2})[/](\d{1,2})(?:[/](\d{2,4}))?\b", low)
    if m:
        mo, d = int(m[1]), int(m[2])
        if 1 <= mo <= 12 and 1 <= d <= 31:
            y = m[3]
            year = int(y) if y and len(y) == 4 else None
            return (year, mo, d)
    # numeric M-D (avoid matching inside an ISO date already handled)
    m = re.search(r"\b(\d{1,2})-(\d{1,2})\b", low)
    if m:
        mo, d = int(m[1]), int(m[2])
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return (None, mo, d)
    return None


def _looks_month(word: str) -> bool:
    w = word.lower().strip(".")
    if w in ALL_MONTHS:
        return True
    return any(_similar(w, name) >= 0.8 for name in MONTHS)


def parse_time(text: str) -> Optional[tuple]:
    m = re.search(r"\b(\d{1,2})[:_](\d{2})(?:[:_](\d{2}))?\b", text)
    if m:
        h, mi = int(m[1]), int(m[2])
        s = int(m[3]) if m[3] else None
        if 0 <= h <= 23 and 0 <= mi <= 59 and (s is None or 0 <= s <= 59):
            return (h, mi, s)
    m = re.search(r"\b(\d{1,2})\s*([ap])\.?m\.?\b", text.lower())
    if m:
        h = int(m[1]) % 12
        if m[2] == "p":
            h += 12
        return (h, 0, None)
    return None


def parse_duration(text: str) -> Optional[int]:
    """Seconds, from '6 seconds', '6s', '1 min', '1:38'."""
    m = re.search(r"\b(\d{1,3})\s*-?\s*(?:seconds?|secs?|s)\b", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d{1,3})\s*-?\s*(?:minutes?|mins?|m)\b", text)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"\b(\d{1,2}):([0-5]\d)\b", text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Ranges
# ─────────────────────────────────────────────────────────────────────────────
_RANGE_SEP = r"(?:\s*(?:-|–|—|to|thru|through|till|until|and)\s*)"


def parse_index_range(text: str) -> Optional[tuple]:
    """'day 3 to day 7', 'recordings 3-7', 'from 3 to 7', '3 through 7'.
    Returns (start, end) 1-based, inclusive."""
    low = text.lower()
    m = re.search(r"\bdays?\s+(\d{1,3})" + _RANGE_SEP + r"days?\s+(\d{1,3})\b", low)
    if not m:
        m = re.search(
            r"\b(?:recordings?|clips?|files?|items?|number|no\.?|#)\s*"
            r"(\d{1,3})" + _RANGE_SEP + r"(?:recordings?|clips?|files?|items?|"
            r"number|no\.?|#)?\s*(\d{1,3})\b", low)
    if not m:
        m = re.search(r"\bfrom\s+(\d{1,3})" + _RANGE_SEP + r"(\d{1,3})\b", low)
    if m:
        a, b = int(m[1]), int(m[2])
        if a >= 1 and b >= 1:
            return (min(a, b), max(a, b))
    return None


def parse_date_range(text: str, today: Optional[datetime] = None):
    """'from June 6 to June 10', 'june 6 - june 10', 'between 6/6 and 6/10'.
    Returns (Date, Date) or None."""
    low = text.lower()
    # split on a range word and parse each side as a date
    parts = re.split(r"\s+(?:to|thru|through|till|until)\s+|\s*(?:–|—)\s*|"
                     r"\s+(?:and)\s+(?=\d|[a-z]{3,9}\s+\d)|"
                     r"(?<=\d)\s*-\s*(?=\d|[a-z])", low)
    if len(parts) >= 2:
        left, right = parts[0], parts[1]
        d1 = parse_one_date(left, today)
        d2 = parse_one_date(right, today)
        # "june 6 to 10" — right side has only a day; inherit left's month/year
        if d1 and not d2:
            m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", right)
            if m:
                d2 = (d1[0], d1[1], int(m.group(1)))
        if d1 and d2:
            return (_fill_year(d1, d2), _fill_year(d2, d1))
    return None


def _fill_year(d: Date, other: Date) -> Date:
    y = d[0] if d[0] is not None else other[0]
    return (y, d[1], d[2])


# ─────────────────────────────────────────────────────────────────────────────
# Candidate selection helpers (used by the chat tab via classify)
# ─────────────────────────────────────────────────────────────────────────────
def _date_matches(rec_datetime: datetime, d: Date) -> bool:
    y, mo, day = d
    if rec_datetime.month != mo or rec_datetime.day != day:
        return False
    return y is None or rec_datetime.year == y


def candidates_for_date(recordings, d: Date, include_empty=False) -> list:
    out = [(rec_dt(r), r) for r in recordings
           if _date_matches(rec_dt(r), d)
           and (include_empty or not is_empty(r))]
    out.sort(key=lambda x: x[0])
    return [r for _, r in out]


def candidates_for_range(recordings, start: Date, end: Date,
                         include_empty=False) -> list:
    def to_dt(d):
        y = d[0] if d[0] is not None else (start[0] or end[0]
                                           or datetime.now().year)
        return datetime(y, d[1], d[2])
    lo, hi = to_dt(start), to_dt(end)
    if lo > hi:
        lo, hi = hi, lo
    hi = hi + timedelta(days=1)        # inclusive of the end day
    out = []
    for r in recordings:
        dt = rec_dt(r)
        if lo <= dt < hi and (include_empty or not is_empty(r)):
            out.append((dt, r))
    out.sort(key=lambda x: x[0])
    return [r for _, r in out]


def candidates_for_month(recordings, year, month, include_empty=False) -> list:
    out = []
    for r in recordings:
        dt = rec_dt(r)
        if dt.month == month and (year is None or dt.year == year) \
                and (include_empty or not is_empty(r)):
            out.append((dt, r))
    out.sort(key=lambda x: x[0])
    return [r for _, r in out]


def candidates_for_time(recordings, tm, include_empty=False) -> list:
    h, mi, s = tm
    out = []
    for r in recordings:
        dt = rec_dt(r)
        if dt.hour == h and dt.minute == mi and (s is None or dt.second == s) \
                and (include_empty or not is_empty(r)):
            out.append((dt, r))
    out.sort(key=lambda x: x[0])
    return [r for _, r in out]


def is_empty(r) -> bool:
    d = getattr(r, "duration_sec", None)
    return d is not None and d <= 0


def _word_count(s: str) -> int:
    return len((s or "").split())


def is_meaningful(r) -> bool:
    """A clip worth offering for 'latest'/'random': not zero-length, and if it
    has a transcript that transcript isn't trivially short."""
    if is_empty(r):
        return False
    tr = getattr(r, "transcript", "") or ""
    if tr and _word_count(tr) < 4:
        return False
    dur = getattr(r, "duration_sec", None)
    if (not tr) and dur is not None and dur < 2:
        return False
    return True


def latest(recordings, prefer_meaningful=True):
    pool = [r for r in recordings if not is_empty(r)]
    if not pool:
        return None
    if prefer_meaningful:
        rich = [r for r in pool if is_meaningful(r)]
        if rich:
            return max(rich, key=rec_dt)
    return max(pool, key=rec_dt)


# ─────────────────────────────────────────────────────────────────────────────
# Content search ("the recording where I talked about X")
# ─────────────────────────────────────────────────────────────────────────────
_TOPIC_TRIGGERS = (
    "talked about", "talk about", "talking about", "discussed", "discuss",
    "mentioned", "mention", "about when", "where i said", "where we said",
    "where i talk", "where we talk", "that mentions", "where i discussed",
    "regarding", "to do with", "related to",
)


def extract_topic(text: str) -> str:
    low = text.lower()
    for trig in _TOPIC_TRIGGERS:
        idx = low.find(trig)
        if idx != -1:
            tail = text[idx + len(trig):].strip(" ?.!\"'")
            tail = re.sub(r"^(the|a|an|that|some|my|our)\s+", "", tail,
                          flags=re.I)
            if tail:
                return tail
    # "recording about X" / "recording on X"
    m = re.search(r"recording\s+(?:about|on|regarding)\s+(.+)$", low)
    if m:
        return m.group(1).strip(" ?.!\"'")
    return ""


def _topic_tokens(topic: str) -> list:
    return [t for t in re.split(r"\W+", topic.lower())
            if len(t) >= 3 and t not in _NAME_STOP]


def content_search(topic: str, recordings) -> list:
    """Rank recordings by how well their transcript/summary matches the topic.
    Returns recordings (best first) that have any match."""
    toks = _topic_tokens(topic)
    if not toks:
        return []
    scored = []
    for r in recordings:
        hay = ((getattr(r, "transcript", "") or "") + " "
               + (getattr(r, "summary", "") or "")).lower()
        if not hay.strip():
            continue
        hits = sum(hay.count(t) for t in toks)
        distinct = sum(1 for t in toks if t in hay)
        if hits:
            score = distinct * 10 + hits
            scored.append((score, rec_dt(r), r))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for _, _, r in scored]


# ─────────────────────────────────────────────────────────────────────────────
# In-recording lookups (issue #6: forgot a time / find when a topic came up)
# ─────────────────────────────────────────────────────────────────────────────
def _segments(rec) -> list:
    segs = getattr(rec, "segments", None)
    return segs if isinstance(segs, list) else []


def fmt_offset(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def find_topic_in_recording(topic: str, rec) -> list:
    """Find where a topic is discussed inside one recording.
    Returns list of (start_seconds|None, speaker, text)."""
    toks = _topic_tokens(topic)
    segs = _segments(rec)
    out = []
    if segs:
        for seg in segs:
            txt = (seg.get("text") or "").lower()
            if txt and (not toks or any(t in txt for t in toks)):
                if toks and any(t in txt for t in toks):
                    out.append((seg.get("start"), seg.get("speaker"),
                                seg.get("text", "").strip()))
        return out
    # No segments — fall back to a transcript text scan (no timestamps).
    tr = getattr(rec, "transcript", "") or ""
    for line in tr.splitlines():
        if toks and any(t in line.lower() for t in toks):
            out.append((None, None, line.strip()))
    return out


def lookup_offset(rec, seconds: float) -> Optional[dict]:
    """What was being said at `seconds` into the recording. Returns the segment
    dict, or None if there are no timestamped segments / it's out of range."""
    segs = _segments(rec)
    if not segs:
        return None
    best = None
    for seg in segs:
        st = seg.get("start")
        en = seg.get("end")
        if st is None:
            continue
        if en is not None and st <= seconds <= en:
            return seg
        if st <= seconds:
            best = seg
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Photo trigger — "hey iris, take a photo" / "iris, take a picture" / bare
# "take a screenshot". Completely separate domain from recordings, so this is
# checked first in classify() and short-circuits everything else below.
# ─────────────────────────────────────────────────────────────────────────────
_PHOTO_PATTERN = re.compile(
    r"\b(?:take|snap|grab|capture)\s+(?:a\s+|my\s+|the\s+)?"
    r"(?:photo|picture|pic|snap(?:shot)?|screenshot)\b"
    r"|\bcapture\s+(?:the\s+|a\s+|my\s+)?screen\b",
    re.IGNORECASE,
)


def is_photo_trigger(text: str) -> bool:
    """True for an imperative to capture a photo/screenshot, e.g.
    'hey iris, take a photo', 'iris take a picture', 'take a screenshot',
    'snap a photo of my screen'. Independent of the 'iris'/'hey' prefix —
    that part is optional filler the regex just searches past."""
    return bool(_PHOTO_PATTERN.search((text or "").lower()))


_SCREEN_CAPTURE_RE = re.compile(
    r"\bscreenshot\b"
    r"|\bcapture\s+(?:the\s+|a\s+|my\s+)?screen\b"
    r"|\b(?:photo|picture|pic|snap(?:shot)?)\b[^.!?]{0,20}\bscreen\b",
    re.IGNORECASE,
)


def photo_capture_mode(text: str) -> str:
    """Only meaningful once is_photo_trigger() is already True. Returns
    'screen' for anything that explicitly names the screen/screenshot
    ('take a screenshot', 'take a photo of my screen', 'capture my
    screen'), and 'camera' otherwise — the default for a bare 'take a
    photo'/'take a picture', which means a photo of the person, not the
    display."""
    low = (text or "").lower()
    return "screen" if _SCREEN_CAPTURE_RE.search(low) else "camera"


_PHOTO_QUERY_NOUN_RE = re.compile(r"\b(?:photos?|pictures?|pics?|screenshots?|snaps?)\b")
_PHOTO_QUERY_VERB_CUES = (
    "show me", "show", "see", "find", "pull up", "open", "what's", "whats",
    "what is", "give me", "get me", "display",
)
_PHOTO_LATEST_CUES = ("latest", "last", "most recent", "newest")


def is_photo_query(text: str) -> bool:
    """True for a request to look up an EXISTING photo/photos — as opposed
    to is_photo_trigger's 'capture a new one now'. e.g. 'show me my latest
    photo', 'photos from yesterday', 'what's the picture from 2:15'.
    Always requires a photo-specific noun, so it never collides with
    recording phrasing ('show me recordings', 'latest recording', etc.)."""
    low = (text or "").lower()
    if not _PHOTO_QUERY_NOUN_RE.search(low):
        return False
    if any(c in low for c in _PHOTO_QUERY_VERB_CUES):
        return True
    if any(c in low for c in _PHOTO_LATEST_CUES):
        return True
    if re.search(r"\bphotos?\b.{0,15}\b(from|on|at|taken)\b", low):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# The classifier — the single entry point the chat tab uses
# ─────────────────────────────────────────────────────────────────────────────
_LIST_CUES = (
    "list", "show me", "show all", "what recordings", "which recordings",
    "how many recordings", "all recordings", "all the recordings",
    "all my recordings", "my recordings", "every recording",
    "recordings do you", "recordings you have", "recordings you can",
)
_ALL_CUES = ("all", "every", "each", "everything")
_ACTION_CUES = (
    "summarize", "summary", "summarise", "recap", "play", "open", "transcribe",
    "what's in", "whats in", "what is in", "tell me about", "go over",
    "pull up", "describe",
)
_RECORDING_NOUNS = (
    "recording", "recordings", "audio", "clip", "clips", "file", "files",
    "conversation", "conversations", "voice memo", "transcript", "call",
)


def _has_any(low: str, words) -> bool:
    return any(w in low for w in words)


def classify(text: str, recordings, today: Optional[datetime] = None,
             has_active: bool = False) -> Intent:
    """Map raw user text to a structured Intent. `recordings` is the full known
    set (each item exposes .name/.mtime/.duration_sec/.transcript/.summary)."""
    today = today or datetime.now()
    raw = text or ""
    corrected = correct_text(raw)
    low = corrected.lower().strip()
    intent = Intent(corrected_text=corrected)

    # Photo capture is an imperative command in a completely different domain
    # from recordings — check it first so it can never be mistaken for, or
    # swallowed by, recording-related logic below (e.g. an active recording).
    if is_photo_trigger(low):
        intent.kind = "photo"
        intent.capture_mode = photo_capture_mode(low)
        return intent

    # Looking up an EXISTING photo is also its own domain, checked right
    # after the trigger. Reuses the same date/time/range parsers as
    # recordings — they're generic, not recording-specific.
    if is_photo_query(low):
        intent.kind = "photo_query"

        dr = parse_date_range(corrected, today)
        if dr:
            intent.photo_action = "range"
            intent.date_range = dr
            return intent

        d = parse_one_date(corrected, today)
        rel = parse_relative_day(corrected, today)
        if d is None and rel is not None:
            d = rel
        if d is not None:
            intent.photo_action = "date"
            intent.dates = [d]
            tm = parse_time(_strip_dates(low))
            if tm is not None:
                intent.time = tm
            return intent

        tm = parse_time(low)
        if tm is not None:
            intent.photo_action = "time"
            intent.time = tm
            return intent

        if any(c in low for c in _PHOTO_LATEST_CUES):
            intent.photo_action = "latest"
            return intent

        intent.photo_action = "all"
        return intent

    plural = bool(re.search(r"\brecordings\b", low)) or _has_any(low, _ALL_CUES)

    # 1) Explicit list-everything request.
    if _has_any(low, _LIST_CUES) and not _wants_latest(low) \
            and "this recording" not in low and "this call" not in low:
        # "summarize all my recordings" is an all-summary, not just a listing.
        if _has_any(low, _ACTION_CUES) and _has_any(low, _ALL_CUES):
            intent.kind = "list"          # chat tab still shows the list to pick
            intent.summarize_all = True
            return intent
        intent.kind = "list"
        return intent

    # 2) Content search: "the recording where I talked about X".
    topic = extract_topic(corrected)
    if topic and _has_any(low, _RECORDING_NOUNS):
        intent.kind = "content_search"
        intent.content_query = topic
        return intent

    # 3) Date range (absolute) — always summarizes the whole span.
    dr = parse_date_range(corrected, today)
    if dr:
        intent.kind = "date_range"
        intent.date_range = dr
        intent.summarize_all = True
        return intent

    # 4) Index range "day 3 to day 7" / "recordings 3-7".
    ir = parse_index_range(low)
    if ir:
        intent.kind = "index_range"
        intent.index_range = ir
        intent.summarize_all = True
        return intent

    # 5) Latest / most recent.
    if _wants_latest(low):
        intent.kind = "latest"
        return intent

    # 6) Random audio file.
    if re.search(r"\b(random|any|some|a)\b[^.]*\b(audio|recording|clip|file|"
                 r"conversation)\b", low) and "random" in low:
        intent.kind = "random"
        return intent

    # 7) Absolute / relative single date (+ optional time).
    d = parse_one_date(corrected, today)
    rel = parse_relative_day(corrected, today)
    if d is None and rel is not None:
        d = rel
    if d is not None:
        intent.kind = "date"
        intent.dates = [d]
        tm = parse_time(_strip_dates(low))
        if tm is not None:
            intent.time = tm
        intent.summarize_all = plural or _has_any(low, _ALL_CUES)
        return intent

    # 8) Month ("recordings in june", "june recordings").
    mo = _parse_month(low)
    if mo is not None:
        intent.kind = "month"
        intent.dates = [(mo[0], mo[1], 0)]   # day 0 = whole month sentinel
        intent.summarize_all = plural
        return intent

    # 9) Time only ("the 17:09 one") — only if it matches a real recording.
    tm = parse_time(low)
    if tm is not None and candidates_for_time(recordings, tm):
        intent.kind = "time"
        intent.time = tm
        return intent

    # 10) Name match — partial / messy filenames, "testing upload", etc.
    #     Require either a recording noun/action cue OR a confident match so we
    #     don't hijack ordinary chat.
    matches = match_name(corrected, recordings)
    if matches and (_has_any(low, _RECORDING_NOUNS)
                    or _has_any(low, _ACTION_CUES) or not has_active):
        intent.kind = "name"
        intent.name_matches = matches
        return intent

    # 11) Bare "audio file" / "a recording" with an action but no specifier →
    #     treat as "pick something" so it never falls through to raw chat.
    if _has_any(low, _ACTION_CUES) and _has_any(low, _RECORDING_NOUNS):
        intent.kind = "list"
        return intent

    intent.kind = "none"
    return intent


def _wants_latest(low: str) -> bool:
    return bool(re.search(
        r"\b(last|latest|most recent|newest|recent)\b[^.]*"
        r"\b(recording|audio|clip|chat|conversation|file)\b", low)) or \
        bool(re.search(
            r"\b(recording|audio|clip|file)\b[^.]*"
            r"\b(last|latest|newest|recent)\b", low))


def _strip_dates(low: str) -> str:
    names = "|".join(ALL_MONTHS)
    s = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", " ", low)
    s = re.sub(rf"\b({names})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s*\d{{4}})?",
               " ", s)
    s = re.sub(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?({names})\b", " ", s)
    s = re.sub(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", " ", s)
    return s


def _parse_month(low: str):
    names = "|".join(ALL_MONTHS)
    # don't treat "june 17" (has a day) as a whole-month request
    if re.search(rf"\b({names})\.?\s+\d{{1,2}}\b", low):
        return None
    if re.search(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?({names})\b", low):
        return None
    patterns = [
        rf"\b(?:in|during|from|for|this)\s+({names})\b(?:\s+(\d{{4}}))?",
        rf"\b({names})\s+(?:recordings?|clips?|files?|audio)\b",
        rf"\brecordings?\s+in\s+({names})\b",
        rf"\b({names})\s+(\d{{4}})\b",
    ]
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            mo = _month_from_word(m.group(1))
            year = None
            for g in m.groups()[1:]:
                yy = _clean_year(g)
                if yy:
                    year = yy
            if mo:
                return (year, mo)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# M7: memory recall intent classifier
# ─────────────────────────────────────────────────────────────────────────────
# Person-anchored and semantic queries against the ChromaDB memory store.
# Routed BEFORE the recording classifier so things like
# "give me all conversations with Pranav" map to memory recall rather than
# to a flat list of WAV files.
#
# Kinds:
#   memory_person   — conversations about/with a named person
#   memory_who      — "who did I talk to yesterday / this week"
#   memory_semantic — "what did we discuss about <topic>"
#   none            — fall through to the recording classifier or Llama


@dataclass
class MemoryIntent:
    kind: str = "none"
    person_name: str = ""           # resolved name from known_names
    query: str = ""                  # topic / semantic query, may be ""
    date_start: Optional[float] = None   # unix seconds
    date_end: Optional[float] = None     # unix seconds
    corrected_text: str = ""


# Phrases that indicate "I want my own conversation history". These have to be
# specific enough that a casual chat message ("what did you say earlier")
# doesn't accidentally trigger recall.
_MEMORY_PERSON_CUES = (
    "conversations with", "conversation with", "talks with", "talk with",
    "talked with", "talked to", "spoke with", "spoken with", "met with",
    "meetings with", "meeting with", "chats with", "chat with",
    "what did", "what has", "what's", "whats",
    "tell me about my", "tell me about the",
    "history with", "interactions with", "discussion with",
    "what we discussed",
)

_MEMORY_WHO_CUES = (
    "who did i", "who have i", "who was i", "who did we",
    "who's been", "who has been", "who spoke", "who talked",
)

_MEMORY_TOPIC_CUES = (
    "what did we discuss", "what did i discuss", "what did we talk about",
    "what was discussed", "when did we discuss", "when did i talk about",
    "when did we talk about", "remind me about", "find a conversation about",
    "find conversations about", "do i have any conversation about",
    "any conversation about", "what came up about", "anything about",
    "search for", "look up", "look for",
)

# Patterns that explicitly anchor a name even without the cue phrases.
# E.g. "Pranav said" / "with Humza" / "did Ali mention X".
_NAME_AFTER_PREP_RE = re.compile(
    r"\b(?:with|about|to|from|by|of|on)\s+([A-Z][a-zA-Z\-']{1,30}"
    r"(?:\s+[A-Z][a-zA-Z\-']{1,30})?)\b"
)
_NAME_BEFORE_VERB_RE = re.compile(
    r"\b([A-Z][a-zA-Z\-']{1,30}(?:\s+[A-Z][a-zA-Z\-']{1,30})?)\s+"
    r"(?:said|told|mentioned|talked|spoke|asked|discussed|"
    r"says|talks|speaks|mentions|asks|discusses)\b"
)


def _date_range_to_ts(today: datetime, *,
                      days_back: Optional[int] = None,
                      this_week: bool = False) -> tuple[float, float]:
    """Convert relative date refs into (start_ts, end_ts) inclusive."""
    if this_week:
        # Past 7 days ending now.
        start = today - timedelta(days=7)
        return (start.timestamp(), today.timestamp() + 1)
    if days_back is not None:
        start = (today - timedelta(days=days_back)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return (start.timestamp(), end.timestamp())
    return (0.0, today.timestamp() + 1)


def _parse_memory_date(low: str, today: datetime
                       ) -> tuple[Optional[float], Optional[float]]:
    """Pull a date window out of the message. Returns (start, end) or
    (None, None) if no date hint found. Reuses parse_relative_day for the
    common cases."""
    if re.search(r"\b(this week|past week|last 7 days|past 7 days)\b", low):
        s, e = _date_range_to_ts(today, this_week=True)
        return (s, e)
    if re.search(r"\b(this month|past month|last 30 days|past 30 days)\b", low):
        start = today - timedelta(days=30)
        return (start.timestamp(), today.timestamp() + 1)
    if re.search(r"\b(today)\b", low):
        return _date_range_to_ts(today, days_back=0)
    if re.search(r"\b(yesterday)\b", low):
        return _date_range_to_ts(today, days_back=1)
    m = re.search(r"\b(\d{1,2})\s+days?\s+ago\b", low)
    if m:
        return _date_range_to_ts(today, days_back=int(m.group(1)))
    # Absolute single date — reuse parse_one_date.
    d = parse_one_date(low, today)
    if d is not None:
        y, mo, day = d
        year = y or today.year
        try:
            start = datetime(year, mo, day)
            end = start + timedelta(days=1)
            return (start.timestamp(), end.timestamp())
        except Exception:
            pass
    return (None, None)


def _resolve_person_name(text: str,
                         known_names: list[str]) -> str:
    """Find the best matching known person name in the message. Two
    strategies: (1) direct case-insensitive substring against known
    names, then (2) heuristic capitalised-noun extraction near
    person-related prepositions/verbs. Returns "" if no match."""
    if not known_names or not text:
        return ""
    low = text.lower()
    # Direct substring against the full name first (longest first so
    # "Humza Malik" wins over "Humza" when both are known).
    sorted_names = sorted(known_names, key=lambda n: -len(n))
    for name in sorted_names:
        if not name:
            continue
        nl = name.lower().strip()
        if not nl:
            continue
        # Word-boundary match so "Pranav" doesn't match in "pranav-like".
        if re.search(rf"\b{re.escape(nl)}\b", low):
            return name
    # Heuristic: a name-shaped token after a preposition like "with X".
    for m in _NAME_AFTER_PREP_RE.finditer(text):
        cand = m.group(1).strip()
        # Match against known names case-insensitive.
        for name in sorted_names:
            if name.lower() == cand.lower():
                return name
    # Same heuristic, name-before-verb pattern.
    for m in _NAME_BEFORE_VERB_RE.finditer(text):
        cand = m.group(1).strip()
        for name in sorted_names:
            if name.lower() == cand.lower():
                return name
    return ""


def _extract_memory_topic(text: str) -> str:
    """Pull the topic phrase out of "what did we discuss about X" / "any
    conversation about X" / etc. Returns "" if no clear topic anchor."""
    low = text.lower()
    for cue in (
        "discuss about", "discussed about", "talked about", "talk about",
        "talking about", "conversation about", "conversations about",
        "anything about", "came up about", "remind me about",
        "search for", "look up", "look for",
    ):
        idx = low.find(cue)
        if idx != -1:
            tail = text[idx + len(cue):].strip(" ?.!\"'")
            tail = re.sub(r"^(the|a|an|that|some|my|our)\s+", "", tail,
                          flags=re.I)
            if tail:
                return tail
    return ""


def classify_memory(text: str,
                    known_names: Optional[list[str]] = None,
                    today: Optional[datetime] = None) -> MemoryIntent:
    """Route a chat message to a memory-recall intent if it looks like
    one. Called BEFORE the recording classifier in ChatTab so memory
    queries win over recording-name fuzzy matches.

    Returns MemoryIntent(kind='none') if the message isn't a recall.
    The Chat tab then falls through to the existing recording classifier
    and finally to raw Llama chat."""
    today = today or datetime.now()
    raw = text or ""
    corrected = correct_text(raw)
    low = corrected.lower().strip()
    intent = MemoryIntent(corrected_text=corrected)
    if not low:
        return intent

    known_names = [n for n in (known_names or []) if n]

    # Date window (optional — applies to all memory kinds).
    date_start, date_end = _parse_memory_date(low, today)

    # ── 1. "Who did I talk to / who spoke" — pure people-listing query ──
    if _has_any(low, _MEMORY_WHO_CUES):
        intent.kind = "memory_who"
        intent.date_start = date_start
        intent.date_end = date_end
        return intent

    # ── 2. Person-anchored recall ──────────────────────────────────────
    resolved = _resolve_person_name(corrected, known_names)
    has_person_cue = _has_any(low, _MEMORY_PERSON_CUES)
    if resolved and has_person_cue:
        intent.kind = "memory_person"
        intent.person_name = resolved
        intent.date_start = date_start
        intent.date_end = date_end
        # If the message also asks "about X", keep the topic for semantic
        # narrowing within that person's records.
        intent.query = _extract_memory_topic(corrected)
        return intent

    # ── 3. Semantic topic search ───────────────────────────────────────
    if _has_any(low, _MEMORY_TOPIC_CUES):
        topic = _extract_memory_topic(corrected)
        if topic:
            intent.kind = "memory_semantic"
            intent.query = topic
            intent.person_name = resolved        # optional refinement
            intent.date_start = date_start
            intent.date_end = date_end
            return intent

    # ── 4. Bare "conversations with X" without cue but with strong name
    # match — still treat as person recall.
    if resolved and re.search(
            rf"\b(conversations?|talks?|chats?|meetings?)\b.*\b{re.escape(resolved.lower())}\b",
            low):
        intent.kind = "memory_person"
        intent.person_name = resolved
        intent.date_start = date_start
        intent.date_end = date_end
        return intent

    return intent


# ─────────────────────────────────────────────────────────────────────────────
# Action intent classifier — chat-driven UI navigation
# ─────────────────────────────────────────────────────────────────────────────
# Detects when the user is asking IRIS to *do* something in the UI rather
# than answer a question. Right now this covers three flows:
#   action_start_video — switch to Stream tab + click Start Listening
#   action_start_audio — switch to Audio tab + click Start Recording
#   action_open_email  — launch the default email client / webmail
#
# Both are tab-navigation shortcuts: nothing about the underlying tabs
# changes, the chat just clicks the button for you. The chat reply names
# the specific limit of the system (e.g. the camera records 35s clips,
# not a user-picked duration) so the user knows what to expect.


@dataclass
class ActionIntent:
    kind: str = "none"
    # kinds: action_start_video | action_start_audio | action_open_email | none
    corrected_text: str = ""


# Phrases that trigger video / stream actions. Wide net but specific enough
# that "what did we record yesterday" doesn't get pulled in.
_ACTION_VIDEO_CUES = (
    "record a video", "record video", "take a video", "take video",
    "shoot a video", "shoot video", "film something", "film me",
    "start the camera", "start camera", "open the camera",
    "start listening", "start the stream", "start stream",
    "start recording video", "begin recording video",
    "record some video", "capture a video", "capture video",
    "record some footage",
)

# Phrases that trigger audio recording. Deliberately narrower than the
# video set so "record a video" doesn't accidentally land here. We also
# require either "audio" or no qualifier at all (bare "record me" /
# "start recording" map to audio because that's the most common case).
_ACTION_AUDIO_CUES = (
    "start recording", "start an audio recording", "start audio recording",
    "record audio", "record some audio", "record me",
    "begin recording", "begin audio recording",
    "start listening to me", "start the audio",
    "record a conversation", "record this conversation",
    "start the recording",
)

# Phrases that trigger launching email. Kept narrow on purpose — "email"
# alone is too common in ordinary conversation ("I'll email him later")
# to be a safe trigger, so we require an explicit open/check verb next
# to the noun.
_ACTION_EMAIL_CUES = (
    "open email", "open up email", "open my email", "open the email",
    "check email", "check my email", "open gmail", "check gmail",
    "open my inbox", "open inbox", "pull up my email", "pull up email",
)


def classify_action(text: str) -> ActionIntent:
    """Decide whether the message is asking IRIS to start the camera /
    audio recorder. Returns ActionIntent(kind='none') if not.

    Video cues are checked first so 'record a 10 second video' lands on
    action_start_video, not on action_start_audio (which would match
    'start recording' as a substring of 'start recording video').
    """
    corrected = correct_text(text or "")
    low = corrected.lower().strip()
    intent = ActionIntent(corrected_text=corrected)
    if not low:
        return intent

    # Order matters: any video cue overrides any audio cue, because some
    # audio cues are substrings of video phrases.
    if _has_any(low, _ACTION_VIDEO_CUES):
        intent.kind = "action_start_video"
        return intent

    # Broader pattern: 'record' (any tense) + a video-ish noun anywhere.
    # Catches things like 'record a 10 second video' / 'record me 30s of
    # footage' / 'can you record a video' that the fixed phrase list above
    # misses because of intervening words.
    # Excluded when the message is really a LOOKUP about existing clips
    # ("give me the latest video recording", "how many videos do I have")
    # rather than a command to start a new one — those must never hijack
    # the tab switch.
    _LOOKUP_CUES = ("latest", "last video", "last recording", "recent",
                    "show me", "give me", "list", "which video", "how many",
                    "who was", "who's in", "whos in", "what happened",
                    "can you see", "do you have", "access")
    if (re.search(r"\brecord(?:ing|ed|s)?\b", low)
            and ("video" in low or "camera" in low or "footage" in low)
            and not low.strip().endswith("?")
            and not _has_any(low, _LOOKUP_CUES)):
        intent.kind = "action_start_video"
        return intent

    if _has_any(low, _ACTION_AUDIO_CUES):
        # But not if the message also says 'video' anywhere — defensive
        # check in case our phrase lists missed a combination.
        if "video" in low or "camera" in low or "footage" in low:
            intent.kind = "action_start_video"
            return intent
        intent.kind = "action_start_audio"
        return intent

    if _has_any(low, _ACTION_EMAIL_CUES):
        intent.kind = "action_open_email"
        return intent

    return intent


# ─────────────────────────────────────────────────────────────────────────────
# Email read classifier — chat-driven Gmail content lookup
# ─────────────────────────────────────────────────────────────────────────────
# Distinct from action_open_email (classify_action above), which only
# launches the browser tab with no content. This classifier fires when the
# user wants an email's actual content surfaced in chat. Three ways a
# request resolves:
#   email_topic          — "read the email with handshake in it",
#                           "email about the internship" -> search term
#                           handed straight to Gmail's own query search
#   email_ordinal         — "the second email", "third one" -> position
#                           within the current unread list
#   email_latest          — no topic, no ordinal -> most recent email,
#                           regardless of read/unread status

@dataclass
class EmailIntent:
    kind: str = "none"
    # kinds: email_topic | email_ordinal | email_latest | none
    corrected_text: str = ""
    topic: str = ""
    ordinal_index: int = -1
    # --- IRIS email-sender: ADD ---
    # A "from X" / "by X" / "sent by X" query. When set alongside topic,
    # they are combined into Gmail's `from:X topic` search. When set with
    # topic empty, the search is a bare `from:X`. Kept as a plain string
    # so the GUI can hand it straight to Gmail's from: operator, which
    # already substring-matches both display name and address.
    sender: str = ""
    # --- IRIS email-sender: END ---


_EMAIL_READ_CUES = (
    "check email", "check my email", "check the email", "check gmail",
    "read my email", "read email", "read my emails", "read the email",
    "what's in my inbox", "whats in my inbox",
    "what's in my email", "whats in my email",
    "any new email", "any new emails",
    "do i have email", "do i have any email", "do i have new email",
    "pull up my email", "pull up email",
)

_EMAIL_ORDINAL_WORDS = {
    "first": 0, "1st": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
}

# Words that anchor topic extraction to actually being about email — without
# this, generic cues like bare "about " would false-positive on ordinary
# conversation ("I was thinking about lunch").
_EMAIL_CONTEXT_WORDS = ("email", "emails", "inbox", "gmail")


def _extract_email_topic(text: str) -> str:
    """Pull the search topic out of email-content lookups: 'read the email
    with handshake in it', 'email about the internship', 'find the email
    regarding X'. Returns "" if no clear topic anchor -- caller should fall
    back to ordinal/latest resolution instead."""
    low = text.lower()
    if not any(w in low for w in _EMAIL_CONTEXT_WORDS):
        return ""
    # Wrap pattern: "... with X in it"
    m = re.search(r"\bwith\s+(.+?)\s+in it\b", low)
    if m:
        start, end = m.span(1)
        topic = text[start:end].strip(" ?.!\"'")
        if topic:
            return topic
    # Prefix cues: tail after the cue phrase is the topic. Longer / more
    # specific cues first so "email about " wins over a bare "about ".
    for cue in ("email about ", "emails about ", "email regarding ",
                "email containing ", "email mentioning ",
                "where it says ", "that says ", "which says ",
                "about ", "regarding ", "containing ", "mentioning ",
                "says "):
        idx = low.find(cue)
        if idx != -1:
            tail = text[idx + len(cue):].strip(" ?.!\"'")
            tail = re.sub(r"^(the|a|an|that|some|my)\s+", "", tail,
                          flags=re.I)
            tail = re.sub(r"\s+in (it|the email|my (inbox|email))\s*$", "",
                          tail, flags=re.I)
            # Trailing filler that isn't part of the topic itself --
            # "give any email about deepgram listed out" should search
            # for "deepgram", not "deepgram listed out". Applied
            # repeatedly since more than one filler phrase can stack
            # ("... deepgram listed out please").
            trailing_filler = re.compile(
                r"\s+(listed out|listed|out loud|out|shown|displayed|"
                r"please|for me|right now|now)\s*$", re.I)
            while True:
                new_tail = trailing_filler.sub("", tail)
                if new_tail == tail:
                    break
                tail = new_tail
            if tail:
                return tail
    return ""


# Verbs that, combined with an email noun nearby, signal a read command
# even when intervening words break the fixed phrase list -- e.g. "read
# THE SECOND email" doesn't contain the literal substring "read email".
# Same class of bug as classify_action's mid-phrase problem; fixed the
# same way, with a word-distance check instead of a growing phrase list.
# --- IRIS email-verbs: CHANGE ---
# Widened past the original ("read", "check", "pull", "look") because
# real users ask with a much larger set of verbs and every miss here
# dropped the message all the way through to plain chat: "give me the
# latest email", "find me emails from prani", "get my inbox", "show me
# any email about deepgram" all failed classify_email and had to be
# rescued (or, more often, weren't). New additions: give/find/get/
# fetch/show/search. Each still needs to sit within 4 words of an
# email/inbox/gmail noun via _has_email_command_pattern, so ordinary
# conversation like "give me an update" isn't dragged into email mode.
_EMAIL_VERB_WORDS = (
    "read", "check", "pull", "look",
    "give", "find", "get", "fetch", "show", "search",
)
# --- IRIS email-verbs: END ---
_EMAIL_WHATS_IN_RE = re.compile(r"\bwhat'?s in\b|\bwhat is in\b")


def _has_email_command_pattern(low: str) -> bool:
    words = re.findall(r"[a-z']+", low)
    noun_idxs = [i for i, w in enumerate(words)
                 if w in ("email", "emails", "inbox", "gmail")]
    verb_idxs = [i for i, w in enumerate(words) if w in _EMAIL_VERB_WORDS]
    return any(abs(ni - vi) <= 4 for ni in noun_idxs for vi in verb_idxs)


def _has_email_read_cue(low: str) -> bool:
    if _has_any(low, _EMAIL_READ_CUES):
        return True
    if not any(w in low for w in _EMAIL_CONTEXT_WORDS):
        return False  # no email noun at all -- never a read command
    if _EMAIL_WHATS_IN_RE.search(low):
        return True
    return _has_email_command_pattern(low)


# --- IRIS email-sender: ADD ---
# Sender extraction — "email from prani", "emails by prani k", "the
# email sent by John", "email from prani@example.com". Requires an
# email context word (email/inbox/gmail) to be present so ordinary
# chat "from" phrases ("a message from Bob") don't false-positive.
# The captured name is stripped of trailing filler ("in it", "please",
# "listed out", punctuation) the same way _extract_email_topic does.
_EMAIL_SENDER_CUES = (
    r"\bsent\s+by\s+",
    r"\bemails?\s+from\s+",
    r"\bemails?\s+by\s+",
    r"\bfrom\s+the\s+sender\s+",
    r"\bfrom\s+sender\s+",
    r"\bfrom\s+",   # last: broadest, only fires if none above matched
    r"\bby\s+",     # last: broadest, only fires if none above matched
)

_SENDER_TRAILING_FILLER_RE = re.compile(
    r"\s+(in it|in the email|in my (?:inbox|email)|"
    r"listed out|listed|out loud|out|shown|displayed|"
    r"please|for me|right now|now|thanks|thank you)\s*$",
    re.I,
)


def _extract_email_sender(text: str) -> str:
    """Return the sender name/address the user is asking about, or "".
    Anchored to an email-context word so 'a text from bob' doesn't fire.
    """
    low = text.lower()
    if not any(w in low for w in _EMAIL_CONTEXT_WORDS):
        return ""
    for pat in _EMAIL_SENDER_CUES:
        m = re.search(pat, low)
        if not m:
            continue
        tail = text[m.end():].strip(" ?.!\"'")
        tail = re.sub(r"^(the|a|an|that|some|my|user|person)\s+", "",
                       tail, flags=re.I)
        # cut at the next clause boundary so "from prani in my inbox"
        # yields "prani", not "prani in my inbox".
        tail = re.split(
            r"\s+(?:in|about|regarding|containing|mentioning|with|"
            r"that|which|where|when|and)\s+",
            tail, maxsplit=1, flags=re.I,
        )[0]
        while True:
            new_tail = _SENDER_TRAILING_FILLER_RE.sub("", tail)
            if new_tail == tail:
                break
            tail = new_tail
        tail = tail.strip(" ?.!\"'")
        # Guard against catching a topic word like "email from work" ->
        # "work" would be a plausible sender name; keep it. But an empty
        # or too-long capture is discarded.
        if 1 <= len(tail) <= 60 and tail.lower() not in _EMAIL_CONTEXT_WORDS:
            return tail
    return ""


# Date/time question detector — routes bare "what day is it today" style
# queries away from the recording classifier (which happily interprets
# any "today"/"tomorrow" as a date lookup and prints "I don't see a
# recording on ..."). Restricted to questions ABOUT the date/time, not
# statements that merely contain a date word.
_DATE_QUESTION_RE = re.compile(
    r"^\s*(?:hey\s+iris[,\s]+|iris[,\s]+)?"
    r"(?:can you |could you |please )?"
    r"(?:tell me |remind me )?"
    r"(?:what(?:'s| is| s)?|whats)"
    r"\s+(?:the\s+)?"
    r"(?:current\s+)?"
    r"(?:date|day|time|month|year)"
    r"(?:\s+is\s+it)?"
    r"(?:\s+(?:today|right now|now|currently))?"
    r"\s*\??\s*$",
    re.I,
)

_DATE_QUESTION_EXTRA = (
    "what day is it",
    "what day is today",
    "what is today's date",
    "whats today's date",
    "what's today's date",
    "what is the date today",
    "whats the date today",
    "what's the date today",
    "what's the current date",
    "whats the current date",
    "what is the current date",
    "current date",
    "todays date",
    "today's date",
    "what time is it",
    "what's the time",
    "whats the time",
    "what is the time",
    "current time",
    "what day of the week is it",
    "what day of the week",
    "what month is it",
    "what year is it",
)


def is_date_question(text: str) -> bool:
    """True for questions asking IRIS what the date/time is right now.
    Kept narrow: has to look like a bare date/time question, not a
    sentence that happens to contain 'today'."""
    low = (text or "").lower().strip().rstrip("?!. ")
    if not low:
        return False
    if _DATE_QUESTION_RE.match(text.strip()):
        return True
    return low in _DATE_QUESTION_EXTRA
# --- IRIS email-sender: END ---


def classify_email(text: str) -> EmailIntent:
    """Decide whether the message wants an email's content read out in
    chat. Fires on either an explicit read/check cue OR clear topic
    phrasing ('email about X') -- the topic extractor is anchored to an
    email-context word so it won't false-positive on ordinary chat like
    'thinking about lunch'."""
    corrected = correct_text(text or "")
    low = corrected.lower().strip()
    intent = EmailIntent(corrected_text=corrected)
    if not low:
        return intent

    topic = _extract_email_topic(corrected)
    # --- IRIS email-sender: ADD ---
    sender = _extract_email_sender(corrected)
    # --- IRIS email-sender: END ---
    has_cue = _has_email_read_cue(low)
    if not (topic or sender or has_cue):
        return intent

    # --- IRIS email-sender: ADD ---
    # Sender search fires as email_topic so the existing multi-hit
    # pick-list flow keeps working. The GUI checks intent.sender first
    # and hands `from:<sender>` (optionally with the topic appended) to
    # Gmail's own search.
    if sender:
        intent.kind = "email_topic"
        intent.topic = topic  # may be "" — that's fine, sender-only
        intent.sender = sender
        return intent
    # --- IRIS email-sender: END ---

    if topic:
        intent.kind = "email_topic"
        intent.topic = topic
        return intent

    for word, idx in _EMAIL_ORDINAL_WORDS.items():
        if word in low:
            intent.kind = "email_ordinal"
            intent.ordinal_index = idx
            return intent

    intent.kind = "email_latest"
    return intent

def is_bare_email_check(text: str) -> bool:
    """True if text is (close to) an exact 'check my email' style phrase
    with nothing else in it that could be a topic. Used by iris_gui.py to
    decide whether it's worth paying for a llama3.2:1b topic-extraction
    call when classify_email's own cue-based extraction comes up empty —
    no point asking the model to look for a topic in a message that is,
    word for word, one of the recognized bare read/check cues."""
    low = correct_text(text or "").lower().strip()
    return low in _EMAIL_READ_CUES