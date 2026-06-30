"""
Summarizer for Phase 9.

Sends each chunk's transcript to a local Ollama instance running
Llama 3.2 1B (or whatever OLLAMA_MODEL is set to). Writes a
.summary.txt next to the WAV containing 3-5 sentences of summary.

If Ollama is unreachable (not installed, daemon not running), this
quietly fails and just doesn't produce a summary. The rest of the
system keeps working.
"""

import json
import os
import queue
import threading
from typing import Optional

import requests


PROMPT_TEMPLATE = """Summarize the following conversation in 3-4 short sentences.
Be specific and concise. Mention the speakers by their tag if useful.
Do not add any preamble or commentary. Just the summary text.

Transcript:
{transcript}

Summary:"""


class Summarizer(threading.Thread):
    def __init__(self,
                 ollama_url: str,
                 model: str,
                 timeout_s: float = 120.0,
                 event_queue: Optional[queue.Queue] = None):
        super().__init__(daemon=True, name="Summarizer")
        self._url = ollama_url.rstrip("/")
        self._model = model
        self._timeout = timeout_s
        self._event_queue = event_queue

        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._available = False

        self.jobs_completed = 0
        self.jobs_failed = 0

    def submit(self, wav_path: str) -> None:
        self._queue.put(wav_path)

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def run(self) -> None:
        # Probe Ollama once on startup
        try:
            resp = requests.get(f"{self._url}/api/tags", timeout=3.0)
            if resp.ok:
                names = [m["name"] for m in resp.json().get("models", [])]
                if self._model in names:
                    self._available = True
                    print(f"[summarizer] Ollama OK, model '{self._model}' ready")
                else:
                    print(f"[summarizer] Ollama OK but model '{self._model}' "
                          f"not found. Run: ollama pull {self._model}")
            else:
                print(f"[summarizer] Ollama responded with {resp.status_code}")
        except Exception as e:
            print(f"[summarizer] Ollama not reachable ({e}). "
                  f"Summarization disabled.")

        while not self._stop_event.is_set():
            try:
                path = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if path is None:
                break
            if not self._available:
                self.jobs_failed += 1
                continue
            self._process(path)

    def _process(self, wav_path: str) -> None:
        json_path = os.path.splitext(wav_path)[0] + ".json"
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[summarizer] could not read {json_path}: {e}")
            self.jobs_failed += 1
            return

        segments = data.get("segments", [])
        if not segments:
            return

        # Build transcript with speaker tags if present.
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "Speaker")
            text = seg["text"].strip()
            if text:
                lines.append(f"{speaker}: {text}")
        transcript_text = "\n".join(lines)
        if not transcript_text:
            return

        prompt = PROMPT_TEMPLATE.format(transcript=transcript_text)

        try:
            resp = requests.post(
                f"{self._url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 200,
                    },
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            summary = resp.json().get("response", "").strip()
        except Exception as e:
            print(f"[summarizer] error on {os.path.basename(wav_path)}: {e}")
            self.jobs_failed += 1
            return

        if not summary:
            print(f"[summarizer] empty summary for {os.path.basename(wav_path)}")
            self.jobs_failed += 1
            return

        # Save next to the WAV
        summary_path = os.path.splitext(wav_path)[0] + ".summary.txt"
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary + "\n")
        except Exception as e:
            print(f"[summarizer] could not save: {e}")
            self.jobs_failed += 1
            return

        print(f"[summarizer] {os.path.basename(wav_path)} done  "
              f"({len(summary)} chars)")
        self.jobs_completed += 1

        if self._event_queue is not None:
            try:
                self._event_queue.put_nowait({
                    "type": "summary_done", "wav": wav_path
                })
            except queue.Full:
                pass
