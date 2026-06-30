"""
Phase 9 GUI.

Adds to Phase 8:
  - Summary panel at top of transcript area
  - Speaker tags inline in transcript ([YOU], [Mom], [Unknown 1 - 78%])
  - Click a speaker tag to assign/rename
  - "Manage Speakers" button opens profile management dialog
  - Recording row shows diarization and summary status icons

M1 change:
  AudioStreamGUI now inherits CTkFrame instead of CTk. This makes it
  embeddable inside the IRIS tabbed parent window (iris_gui.py).
  The original standalone-window behaviour is preserved by the
  AudioStreamWindow helper at the bottom of this file — main_phase9.py
  uses that, so nothing about the standalone app changes.
"""

import os
import json
import queue
import math
import wave
import threading
from typing import Optional, List, Tuple

import customtkinter as ctk
from tkintermapview import TkinterMapView

from location_phase8 import load_location_sidecar


COLOR_STATUS_ON    = "#10b981"
COLOR_STATUS_OFF   = "#6b7280"
COLOR_DANGER       = "#ef4444"
COLOR_RECORDING    = "#dc2626"


class VUMeter(ctk.CTkFrame):
    def __init__(self, master, height=24, **kwargs):
        super().__init__(master, height=height, **kwargs)
        self.configure(fg_color=("#e5e7eb", "#1f2937"))
        self._level = 0.0
        self._peak  = 0.0
        self._canvas = ctk.CTkCanvas(self, height=height,
                                     highlightthickness=0, bg="#1f2937")
        self._canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self.bind("<Configure>", lambda e: self._redraw())

    def setLevel(self, lvl: float):
        lvl = max(0.0, min(1.0, lvl))
        self._level = lvl
        self._peak = lvl if lvl > self._peak else max(lvl, self._peak * 0.92)
        self._redraw()

    def _redraw(self):
        c = self._canvas
        c.delete("all")
        w = c.winfo_width(); h = c.winfo_height()
        if w <= 1 or h <= 1:
            return
        seg_count = 30
        seg_w = w / seg_count
        for i in range(seg_count):
            x0 = int(i * seg_w) + 1
            x1 = int((i + 1) * seg_w) - 1
            if i / seg_count > self._level:         color = "#374151"
            elif i / seg_count > 0.85:              color = "#ef4444"
            elif i / seg_count > 0.7:               color = "#f59e0b"
            else:                                    color = "#10b981"
            c.create_rectangle(x0, 2, x1, h - 2, fill=color, outline="")
        if self._peak > 0.02:
            px = int(self._peak * w)
            c.create_rectangle(px - 2, 1, px, h - 1, fill="#ffffff", outline="")


class StatusDot(ctk.CTkFrame):
    def __init__(self, master, text: str, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._dot = ctk.CTkLabel(self, text="●", text_color=COLOR_STATUS_OFF,
                                 font=("Segoe UI", 14, "bold"))
        self._dot.pack(side="left", padx=(0, 6))
        self._label = ctk.CTkLabel(self, text=text, font=("Segoe UI", 12))
        self._label.pack(side="left")

    def set(self, *, on=False, text=None, color=None):
        self._dot.configure(
            text_color=color if color else (COLOR_STATUS_ON if on else COLOR_STATUS_OFF))
        if text is not None:
            self._label.configure(text=text)


# ---------- Speaker label dialog ----------
class SpeakerDialog(ctk.CTkToplevel):
    """
    Pops up when user clicks a speaker tag. Lets them:
      - Assign an existing known name
      - Type a new name
      - Leave as unknown
    """
    def __init__(self, master, speaker_db, cluster_emb: Optional[object],
                 current_label: str, wav_path: str, on_confirmed):
        super().__init__(master)
        self.title("Identify Speaker")
        self.geometry("400x320")
        self.grab_set()  # modal
        self._db = speaker_db
        self._emb = cluster_emb
        self._wav = wav_path
        self._on_confirmed = on_confirmed

        ctk.CTkLabel(self, text=f"Current label: {current_label}",
                     font=("Segoe UI", 13, "bold")).pack(padx=20, pady=(16, 4))
        ctk.CTkLabel(self, text="Assign this voice to a known speaker:",
                     font=("Segoe UI", 11), text_color="#9ca3af").pack(padx=20)

        # Existing profiles
        names = self._db.list_names()
        if names:
            ctk.CTkLabel(self, text="Known speakers:",
                         font=("Segoe UI", 11), anchor="w").pack(
                padx=20, pady=(10, 2), fill="x")
            scroll = ctk.CTkScrollableFrame(self, height=80)
            scroll.pack(padx=20, fill="x")
            for name in names:
                ctk.CTkButton(scroll, text=name, height=28,
                              font=("Segoe UI", 12),
                              command=lambda n=name: self._confirm(n)
                              ).pack(fill="x", pady=1)

        # New name
        ctk.CTkLabel(self, text="Or enter a new name:",
                     font=("Segoe UI", 11), anchor="w").pack(
            padx=20, pady=(10, 2), fill="x")
        self._entry = ctk.CTkEntry(self, placeholder_text="e.g. Mom, Sarah, ...")
        self._entry.pack(padx=20, fill="x")
        self._entry.bind("<Return>", lambda e: self._confirm_new())

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(padx=20, pady=12, fill="x")
        ctk.CTkButton(row, text="Save", command=self._confirm_new,
                      fg_color=COLOR_STATUS_ON,
                      hover_color="#059669").pack(side="left", padx=(0, 8))
        ctk.CTkButton(row, text="Cancel", command=self.destroy,
                      fg_color="transparent",
                      border_width=1).pack(side="left")

    def _confirm_new(self):
        name = self._entry.get().strip()
        if name:
            self._confirm(name)

    def _confirm(self, name: str):
        if self._emb is not None:
            self._db.add_to(name, self._emb)
        self._on_confirmed(name)
        self.destroy()


# ---------- Manage speakers dialog ----------
class ManageSpeakersDialog(ctk.CTkToplevel):
    def __init__(self, master, speaker_db, recordings_dir, on_changed):
        super().__init__(master)
        self.title("Manage Speaker Profiles")
        self.geometry("620x520")
        self.minsize(500, 400)
        self.configure(fg_color=("#f3f4f6", "#1f2937"))
        self.grab_set()
        self._db             = speaker_db
        self._recordings_dir = recordings_dir
        self._on_changed     = on_changed
        self._build()

    def _build(self):
        for w in self.winfo_children():
            w.destroy()

        # Header
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(padx=20, pady=(16, 4), fill="x")
        ctk.CTkLabel(hdr, text="👤  Saved Speaker Profiles",
                     font=("Segoe UI", 16, "bold"),
                     anchor="w").pack(side="left")
        ctk.CTkLabel(hdr,
                     text="Tag a speaker in a transcript to enroll them.",
                     font=("Segoe UI", 11), text_color="#9ca3af",
                     anchor="e").pack(side="right")

        profiles = self._db.all_info()

        if not profiles:
            empty = ctk.CTkFrame(self, fg_color=("#e5e7eb", "#374151"),
                                 corner_radius=8)
            empty.pack(padx=20, pady=20, fill="x")
            ctk.CTkLabel(empty,
                         text="No speakers enrolled yet.\n\n"
                              "How to add one:\n"
                              "  1. Record something and wait for transcription\n"
                              "  2. Click  👤 Tag Speaker  in the transcript panel\n"
                              "  3. Type your name and press Save\n"
                              "  4. Come back here to manage saved profiles",
                         font=("Segoe UI", 12),
                         text_color=("#374151", "#d1d5db"),
                         justify="left",
                         wraplength=520).pack(padx=20, pady=20, anchor="w")
            ctk.CTkButton(self, text="Close", command=self.destroy,
                          width=100).pack(pady=12)
            return

        # Count how many recordings each speaker appears in
        recording_counts = self._count_appearances()

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(padx=16, pady=4, fill="both", expand=True)
        scroll.grid_columnconfigure(0, weight=1)

        for i, info in enumerate(profiles):
            card = ctk.CTkFrame(scroll, fg_color=("#ffffff", "#2d3748"),
                                corner_radius=8)
            card.grid(row=i, column=0, padx=4, pady=4, sticky="ew")
            card.grid_columnconfigure(1, weight=1)

            # Avatar circle placeholder
            ctk.CTkLabel(card, text="👤", font=("Segoe UI", 22), width=44
                         ).grid(row=0, column=0, rowspan=2,
                                padx=(12, 8), pady=10, sticky="ns")

            # Name + stats
            name_lbl = ctk.CTkLabel(card, text=info["name"],
                                    font=("Segoe UI", 13, "bold"), anchor="w")
            name_lbl.grid(row=0, column=1, padx=4, pady=(10, 0), sticky="w")

            appears = recording_counts.get(info["name"], 0)
            stats = (f"{info['sample_count']} voice sample"
                     f"{'s' if info['sample_count'] != 1 else ''}  •  "
                     f"appears in {appears} recording"
                     f"{'s' if appears != 1 else ''}")
            ctk.CTkLabel(card, text=stats,
                         font=("Segoe UI", 10), text_color="#9ca3af",
                         anchor="w").grid(row=1, column=1, padx=4,
                                          pady=(0, 10), sticky="w")

            # Action buttons
            btns = ctk.CTkFrame(card, fg_color="transparent")
            btns.grid(row=0, column=2, rowspan=2, padx=8, pady=8)

            ctk.CTkButton(btns, text="Rename", width=80,
                          font=("Segoe UI", 11),
                          fg_color=("#3b82f6", "#1d4ed8"),
                          hover_color=("#2563eb", "#1e40af"),
                          command=lambda n=info["name"]: self._rename(n)
                          ).pack(pady=2)

            ctk.CTkButton(btns, text="Delete", width=80,
                          font=("Segoe UI", 11),
                          fg_color=COLOR_DANGER,
                          hover_color="#b91c1c",
                          command=lambda n=info["name"]: self._delete(n)
                          ).pack(pady=2)

        # Bottom bar
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(padx=20, pady=(8, 16), fill="x")
        ctk.CTkLabel(bar,
                     text=f"{len(profiles)} speaker"
                          f"{'s' if len(profiles) != 1 else ''} enrolled",
                     font=("Segoe UI", 11), text_color="#9ca3af"
                     ).pack(side="left")
        ctk.CTkButton(bar, text="Close", width=90,
                      command=self.destroy).pack(side="right")

    def _count_appearances(self) -> dict[str, int]:
        """Count how many WAV files each speaker name appears in."""
        import glob
        counts: dict[str, int] = {}
        for json_path in glob.glob(
                os.path.join(self._recordings_dir, "recording_*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                names_in_file = {
                    seg.get("speaker")
                    for seg in data.get("segments", [])
                    if seg.get("speaker")
                }
                for name in names_in_file:
                    counts[name] = counts.get(name, 0) + 1
            except Exception:
                pass
        return counts

    def _rename(self, old_name: str):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Rename Speaker")
        dlg.geometry("380x200")
        dlg.grab_set()
        dlg.configure(fg_color=("#f3f4f6", "#1f2937"))

        ctk.CTkLabel(dlg, text=f"Rename  \"{old_name}\"",
                     font=("Segoe UI", 13, "bold")).pack(padx=20, pady=(16, 4))
        ctk.CTkLabel(dlg, text="New name:",
                     font=("Segoe UI", 11), anchor="w").pack(
            padx=20, fill="x")
        entry = ctk.CTkEntry(dlg, placeholder_text="Enter new name...")
        entry.pack(padx=20, pady=(4, 12), fill="x")
        entry.insert(0, old_name)
        entry.select_range(0, "end")
        entry.focus()

        def _do_rename():
            new_name = entry.get().strip()
            if not new_name or new_name == old_name:
                dlg.destroy()
                return
            self._db.rename(old_name, new_name)
            self._rename_in_transcripts(old_name, new_name)
            dlg.destroy()
            self._on_changed()
            self._build()

        entry.bind("<Return>", lambda e: _do_rename())
        row = ctk.CTkFrame(dlg, fg_color="transparent")
        row.pack(padx=20, fill="x")
        ctk.CTkButton(row, text="Save",
                      fg_color=COLOR_STATUS_ON,
                      hover_color="#059669",
                      command=_do_rename).pack(side="left", padx=(0, 8))
        ctk.CTkButton(row, text="Cancel",
                      fg_color="transparent", border_width=1,
                      command=dlg.destroy).pack(side="left")

    def _rename_in_transcripts(self, old_name: str, new_name: str):
        """Update every transcript JSON that contains the old speaker name."""
        import glob
        for json_path in glob.glob(
                os.path.join(self._recordings_dir, "recording_*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                changed = False
                for seg in data.get("segments", []):
                    if seg.get("speaker") == old_name:
                        seg["speaker"] = new_name
                        changed = True
                if changed:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
            except Exception:
                pass

    def _delete(self, name: str):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Confirm Delete")
        dlg.geometry("360x160")
        dlg.grab_set()
        dlg.configure(fg_color=("#f3f4f6", "#1f2937"))

        ctk.CTkLabel(dlg,
                     text=f"Delete \"{name}\" and all their voice samples?",
                     font=("Segoe UI", 12),
                     wraplength=320).pack(padx=20, pady=(20, 8))
        ctk.CTkLabel(dlg,
                     text="Transcript labels using this name will remain.",
                     font=("Segoe UI", 10), text_color="#9ca3af"
                     ).pack(padx=20)

        row = ctk.CTkFrame(dlg, fg_color="transparent")
        row.pack(padx=20, pady=16, fill="x")

        def _confirm_delete():
            self._db.delete(name)
            dlg.destroy()
            self._on_changed()
            self._build()

        ctk.CTkButton(row, text="Delete",
                      fg_color=COLOR_DANGER,
                      hover_color="#b91c1c",
                      command=_confirm_delete).pack(side="left", padx=(0, 8))
        ctk.CTkButton(row, text="Cancel",
                      fg_color="transparent", border_width=1,
                      command=dlg.destroy).pack(side="left")


# ---------- Main GUI frame ----------
# Now inherits CTkFrame so it can live inside any parent (tab, window, ...).
# For standalone use, see AudioStreamWindow below — used by main_phase9.py.
class AudioStreamGUI(ctk.CTkFrame):
    def __init__(self, master, controller, config):
        super().__init__(master)
        self.controller = controller
        self.config = config

        self._selected_path: Optional[str] = None
        self._map_markers: List[Tuple[object, List[str]]] = []
        self._recording_rows: list[ctk.CTkButton] = []

        # Make this frame fill its parent
        self.pack_propagate(False) if hasattr(self, "pack_propagate") else None

        self._build_layout()
        self._bind_hotkeys()
        self._poll_events()
        self._poll_vu()
        self._refresh_recordings()
        self._refresh_map()

    # ---------- layout ----------
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=3)
        self._build_status_panel()
        self._build_transcript_panel()
        self._build_recordings_panel()
        self._build_map_panel()

    def _build_status_panel(self):
        outer = ctk.CTkFrame(self)
        outer.grid(row=0, column=0, padx=(12, 6), pady=(12, 6), sticky="nsew")
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(0, weight=1)

        frame = ctk.CTkScrollableFrame(outer, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="Status",
                     font=("Segoe UI", 16, "bold"), anchor="w"
                     ).grid(row=0, column=0, padx=16, pady=(14, 4), sticky="ew")

        self.dot_wifi     = StatusDot(frame, "Wi-Fi: waiting for ESP32")
        self.dot_stream   = StatusDot(frame, "Audio stream: idle")
        self.dot_monitor  = StatusDot(frame, "Monitoring: off")
        self.dot_location = StatusDot(frame, "Location: fetching...")

        for i, dot in enumerate([self.dot_wifi, self.dot_stream,
                                  self.dot_monitor, self.dot_location], 1):
            dot.grid(row=i, column=0, padx=16, pady=2, sticky="w")

        ctk.CTkLabel(frame, text="Input level",
                     font=("Segoe UI", 11), text_color="#9ca3af", anchor="w"
                     ).grid(row=5, column=0, padx=16, pady=(12, 2), sticky="w")
        self.vu = VUMeter(frame, height=22)
        self.vu.grid(row=6, column=0, padx=16, pady=(0, 12), sticky="ew")

        self.btn_record = ctk.CTkButton(
            frame, text="● Start Recording",
            font=("Segoe UI", 14, "bold"), height=48,
            command=self._on_record_clicked,
            fg_color=COLOR_DANGER, hover_color="#b91c1c")
        self.btn_record.grid(row=7, column=0, padx=16, pady=(4, 4), sticky="ew")

        self.btn_monitor = ctk.CTkButton(
            frame, text="🔊 Start Monitoring",
            font=("Segoe UI", 13), height=40,
            command=self._on_monitor_clicked)
        self.btn_monitor.grid(row=8, column=0, padx=16, pady=(4, 8), sticky="ew")

        ctk.CTkButton(frame, text="👤 Manage Speakers",
                      font=("Segoe UI", 12), height=36,
                      command=self._open_manage_speakers,
                      fg_color=("#4b5563", "#374151"),
                      hover_color=("#6b7280", "#4b5563")
                      ).grid(row=9, column=0, padx=16, pady=(0, 12), sticky="ew")

        info = ctk.CTkFrame(frame, fg_color="transparent")
        info.grid(row=10, column=0, padx=16, pady=(4, 12), sticky="ew")
        info.grid_columnconfigure(1, weight=1)
        rows = [
            ("Recording:",        "lbl_rec_duration",  "--:--"),
            ("Chunk:",            "lbl_rec_chunk",     "--"),
            ("Transcribe queue:", "lbl_queue",         "0"),
            ("Diarize queue:",    "lbl_diarize_queue", "0"),
            ("Summarize queue:",  "lbl_sum_queue",     "0"),
            ("Packet loss:",      "lbl_loss",          "--"),
        ]
        for i, (label_text, attr, default) in enumerate(rows):
            ctk.CTkLabel(info, text=label_text, font=("Segoe UI", 11),
                         text_color="#9ca3af").grid(row=i, column=0, sticky="w")
            lbl = ctk.CTkLabel(info, text=default,
                               font=("Segoe UI", 11, "bold"))
            lbl.grid(row=i, column=1, sticky="w", padx=(8, 0))
            setattr(self, attr, lbl)

    def _build_transcript_panel(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=1, padx=(6, 12), pady=(12, 6), sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.grid(row=0, column=0, padx=16, pady=(14, 4), sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text="Transcript",
                     font=("Segoe UI", 16, "bold"), anchor="w"
                     ).grid(row=0, column=0, sticky="w")
        self.lbl_transcript_target = ctk.CTkLabel(
            header, text="(no recording selected)",
            font=("Segoe UI", 11), text_color="#9ca3af", anchor="e")
        self.lbl_transcript_target.grid(row=0, column=1, sticky="e")

        self.lbl_summary_header = ctk.CTkLabel(
            frame, text="Summary",
            font=("Segoe UI", 13, "bold"), anchor="w",
            text_color="#d1d5db")
        self.lbl_summary_header.grid(row=1, column=0,
                                     padx=16, pady=(4, 2), sticky="w")
        self.txt_summary = ctk.CTkTextbox(frame, height=80,
                                          font=("Segoe UI", 12), wrap="word")
        self.txt_summary.grid(row=1, column=0, padx=12, pady=(28, 4),
                              sticky="ew")
        self.txt_summary.configure(state="disabled")

        self.txt_transcript = ctk.CTkTextbox(
            frame, font=("Consolas", 12), wrap="word")
        self.txt_transcript.grid(row=2, column=0, padx=12, pady=(4, 12),
                                 sticky="nsew")
        self.txt_transcript.configure(state="disabled")

        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.grid(row=3, column=0, padx=12, pady=(0, 12), sticky="ew")
        btn_row.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(btn_row, text="👤 Tag Speaker", height=28,
                      font=("Segoe UI", 11),
                      fg_color=("#4b5563", "#374151"),
                      hover_color=("#6b7280", "#4b5563"),
                      command=self._on_tag_speaker_manual).grid(
            row=0, column=0, sticky="w")
        ctk.CTkButton(btn_row, text="↻ Re-summarize", height=28,
                      font=("Segoe UI", 11),
                      command=self._on_resummarize).grid(
            row=0, column=1, sticky="e")

    def _build_recordings_panel(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, padx=(12, 6), pady=(6, 12), sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.grid(row=0, column=0, padx=16, pady=(14, 4), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="Recordings",
                     font=("Segoe UI", 16, "bold"), anchor="w"
                     ).grid(row=0, column=0, sticky="w")

        btns = ctk.CTkFrame(header, fg_color="transparent")
        btns.grid(row=0, column=1, sticky="e")
        for text, cmd in [("↻", self._refresh_all),
                           ("▶", self._on_play_clicked),
                           ("📝", self._on_transcribe_clicked),
                           ("📂", self._on_open_folder),
                           ("⬆", self._on_import_file)]:
            ctk.CTkButton(btns, text=text, width=36,
                          command=cmd).pack(side="left", padx=2)

        self.list_frame = ctk.CTkScrollableFrame(frame, label_text="")
        self.list_frame.grid(row=1, column=0, padx=12, pady=(4, 12),
                             sticky="nsew")
        self.list_frame.grid_columnconfigure(0, weight=1)

    def _build_map_panel(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=1, padx=(6, 12), pady=(6, 12), sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.grid(row=0, column=0, padx=16, pady=(14, 4), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="Map",
                     font=("Segoe UI", 16, "bold"), anchor="w"
                     ).grid(row=0, column=0, sticky="w")
        self.lbl_location = ctk.CTkLabel(
            header, text="", font=("Segoe UI", 11),
            text_color="#9ca3af", anchor="e")
        self.lbl_location.grid(row=0, column=1, sticky="e")

        self.map_widget = TkinterMapView(frame, corner_radius=4)
        self.map_widget.grid(row=1, column=0, padx=12, pady=(4, 12),
                             sticky="nsew")
        self.map_widget.set_tile_server(self.config.MAP_TILE_URL, max_zoom=19)
        self.map_widget.set_position(self.config.MAP_FALLBACK_LAT,
                                     self.config.MAP_FALLBACK_LON)
        self.map_widget.set_zoom(self.config.MAP_DEFAULT_ZOOM)

    # ---------- hotkeys ----------
    def _bind_hotkeys(self):
        # Bind to top-level window so keys work no matter who has focus
        try:
            toplevel = self.winfo_toplevel()
        except Exception:
            toplevel = self
        bindings = {
            "r": self._on_record_clicked, "R": self._on_record_clicked,
            "m": self._on_monitor_clicked, "M": self._on_monitor_clicked,
            "p": self._on_play_clicked,    "P": self._on_play_clicked,
            "t": self._on_transcribe_clicked,
            "T": self._on_transcribe_clicked,
        }
        for ch, handler in bindings.items():
            toplevel.bind(f"<Key-{ch}>", lambda e, h=handler: h())

    # ---------- event polling ----------
    def _poll_events(self):
        try:
            while True:
                evt = self.controller.event_queue.get_nowait()
                self._handle_event(evt)
        except queue.Empty:
            pass
        self.after(self.config.GUI_POLL_MS, self._poll_events)

    def _handle_event(self, evt: dict):
        et = evt.get("type")
        if et == "esp32_connected":
            self.dot_stream.set(on=True, text="Audio stream: receiving")
            self.dot_wifi.set(on=True, text="Wi-Fi: ESP32 connected")
        elif et == "recording_started":
            self.btn_record.configure(text="■ Stop Recording",
                                      fg_color="#7f1d1d",
                                      hover_color="#991b1b")
            self.dot_stream.set(color=COLOR_RECORDING,
                                text=f"RECORDING ({evt.get('session', '')})")
        elif et == "recording_stopped":
            self.btn_record.configure(text="● Start Recording",
                                      fg_color=COLOR_DANGER,
                                      hover_color="#b91c1c")
            self.dot_stream.set(on=True, text="Audio stream: receiving")
            self.lbl_rec_duration.configure(text="--:--")
            self.lbl_rec_chunk.configure(text="--")
            self._refresh_all()
        elif et == "recording_tick":
            dur = evt.get("duration", 0.0)
            m, s = divmod(int(dur), 60)
            self.lbl_rec_duration.configure(text=f"{m:02d}:{s:02d}")
            self.lbl_rec_chunk.configure(text=str(evt.get("chunk", "--")))
        elif et == "monitor_started":
            self.btn_monitor.configure(text="🔇 Stop Monitoring")
            self.dot_monitor.set(on=True, text="Monitoring: on")
        elif et == "monitor_stopped":
            self.btn_monitor.configure(text="🔊 Start Monitoring")
            self.dot_monitor.set(on=False, text="Monitoring: off")
        elif et == "chunk_finalized":
            self._refresh_all()
        elif et == "transcribe_done":
            self._refresh_recordings()
            if self._selected_path == evt.get("wav"):
                self._show_content(self._selected_path)
        elif et == "diarize_done":
            self._refresh_recordings()
            if self._selected_path == evt.get("wav"):
                self._show_content(self._selected_path)
        elif et == "summary_done":
            self._refresh_recordings()
            if self._selected_path == evt.get("wav"):
                self._show_content(self._selected_path)
        elif et == "transcribe_queue":
            self.lbl_queue.configure(text=str(evt.get("depth", 0)))
        elif et == "diarize_queue":
            self.lbl_diarize_queue.configure(text=str(evt.get("depth", 0)))
        elif et == "summarize_queue":
            self.lbl_sum_queue.configure(text=str(evt.get("depth", 0)))
        elif et == "net_stats":
            self.lbl_loss.configure(text=f"{evt.get('loss_pct', 0.0):.2f}%")
        elif et == "location_ready":
            loc = evt.get("location")
            if loc:
                place = f"{loc['city']}, {loc['region']}"
                self.dot_location.set(on=True, text=f"Location: {place}")
                self.lbl_location.configure(text=place)
                self.map_widget.set_position(loc["lat"], loc["lon"])
            else:
                self.dot_location.set(on=False, text="Location: unavailable")

    # ---------- VU ----------
    def _poll_vu(self):
        try:
            self.vu.setLevel(self.controller.peek_level())
        except Exception:
            pass
        self.after(self.config.GUI_VU_DECAY_MS, self._poll_vu)

    # ---------- button handlers ----------
    def _on_record_clicked(self):   self.controller.toggle_recording()
    def _on_monitor_clicked(self):  self.controller.toggle_monitoring()
    def _on_play_clicked(self):
        if self._selected_path: self.controller.play_file(self._selected_path)
    def _on_transcribe_clicked(self):
        if self._selected_path:
            self.controller.transcribe_file(self._selected_path)
    def _on_resummarize(self):
        if self._selected_path:
            self.controller.summarize_file(self._selected_path)

    def _on_tag_speaker_manual(self):
        if not self._selected_path:
            return
        json_path = os.path.splitext(self._selected_path)[0] + ".json"
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        segments = data.get("segments", [])
        if not segments:
            return

        labels = list(dict.fromkeys(
            seg.get("speaker", "Unknown")
            for seg in segments
            if seg.get("speaker")
        ))
        if not labels:
            for seg in segments:
                seg["speaker"] = "Speaker 1"
                seg["speaker_kind"] = "unknown"
                seg["speaker_confidence"] = 0.0
            labels = ["Speaker 1"]

        dialog = ctk.CTkToplevel(self)
        dialog.title("Tag Speaker")
        dialog.geometry("420x360")
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Who is speaking in this recording?",
                     font=("Segoe UI", 14, "bold")).pack(padx=20, pady=(16, 4))
        ctk.CTkLabel(dialog,
                     text="Pick the current label, then enter the real name.",
                     font=("Segoe UI", 11), text_color="#9ca3af"
                     ).pack(padx=20, pady=(0, 8))

        ctk.CTkLabel(dialog, text="Current label in transcript:",
                     font=("Segoe UI", 11), anchor="w").pack(
            padx=20, fill="x")
        label_var = ctk.StringVar(value=labels[0])
        label_menu = ctk.CTkOptionMenu(dialog, values=labels,
                                       variable=label_var)
        label_menu.pack(padx=20, pady=(4, 12), fill="x")

        ctk.CTkLabel(dialog, text="Real name (who this actually is):",
                     font=("Segoe UI", 11), anchor="w").pack(
            padx=20, fill="x")
        name_entry = ctk.CTkEntry(dialog,
                                   placeholder_text="e.g. Humza, Mom, ...")
        name_entry.pack(padx=20, pady=(4, 12), fill="x")
        name_entry.focus()

        def _save():
            old_label = label_var.get()
            new_name  = name_entry.get().strip()
            if not new_name:
                return

            for seg in segments:
                if seg.get("speaker") == old_label:
                    seg["speaker"]            = new_name
                    seg["speaker_kind"]       = "strict"
                    seg["speaker_confidence"] = 1.0

            data["diarized"] = True
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"[gui] could not save speaker tag: {e}")
                dialog.destroy()
                return

            if new_name not in self.controller.speaker_db.list_names():
                try:
                    import numpy as _np
                    placeholder = _np.zeros(192, dtype=_np.float32)
                    self.controller.speaker_db.create(new_name, placeholder)
                    print(f"[gui] enrolled '{new_name}' (placeholder, "
                          f"no voice sample yet)")
                except Exception as e:
                    print(f"[gui] could not create placeholder profile: {e}")

            emb_path = os.path.splitext(self._selected_path)[0] \
                       + ".embeddings.npz"
            if os.path.exists(emb_path):
                try:
                    import numpy as np
                    npz = np.load(emb_path)
                    cluster_ids = list({
                        seg.get("_cluster", -1)
                        for seg in segments
                        if seg.get("speaker") == new_name
                           and seg.get("_cluster", -1) >= 0
                    })
                    for cid in cluster_ids:
                        key = f"cluster_{cid}"
                        if key in npz:
                            self.controller.speaker_db.add_to(
                                new_name, npz[key])
                            print(f"[gui] added real voiceprint for '{new_name}'")
                except Exception as e:
                    print(f"[gui] could not save voiceprint: {e}")

            dialog.destroy()
            self._show_content(self._selected_path)
            self._refresh_recordings()

        name_entry.bind("<Return>", lambda e: _save())
        row = ctk.CTkFrame(dialog, fg_color="transparent")
        row.pack(padx=20, pady=8, fill="x")
        ctk.CTkButton(row, text="Save",
                      fg_color=COLOR_STATUS_ON,
                      hover_color="#059669",
                      command=_save).pack(side="left", padx=(0, 8))
        ctk.CTkButton(row, text="Cancel",
                      fg_color="transparent",
                      border_width=1,
                      command=dialog.destroy).pack(side="left")

    def _on_open_folder(self):
        os.startfile(self.config.RECORDINGS_DIR)

    def _on_import_file(self):
        import shutil
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(
            title="Import audio files",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not paths:
            return
        imported = 0
        for src in paths:
            dst = os.path.join(self.config.RECORDINGS_DIR, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
                imported += 1
        if imported:
            self._refresh_recordings()

    def _open_manage_speakers(self):
        ManageSpeakersDialog(
            self, self.controller.speaker_db,
            recordings_dir=self.config.RECORDINGS_DIR,
            on_changed=self._refresh_all)

    # ---------- recordings list ----------
    def _refresh_all(self):
        self._refresh_recordings()
        self._refresh_map()

    def _refresh_recordings(self):
        for row in self._recording_rows:
            row.destroy()
        self._recording_rows.clear()

        import glob
        files = sorted(glob.glob(
            os.path.join(self.config.RECORDINGS_DIR, "*.wav")
        ), reverse=True)

        for i, path in enumerate(files):
            row = self._make_row(path)
            row.grid(row=i, column=0, padx=4, pady=2, sticky="ew")
            self._recording_rows.append(row)

        if self._selected_path and self._selected_path in files:
            self._show_content(self._selected_path)
        elif files:
            self._select(files[0])
        else:
            self._show_content(None)

    def _make_row(self, path: str) -> ctk.CTkButton:
        base = os.path.splitext(path)[0]
        has_txt  = os.path.exists(base + ".txt")
        has_dia  = os.path.exists(base + ".embeddings.npz")
        has_sum  = os.path.exists(base + ".summary.txt")
        has_loc  = os.path.exists(base + ".location.json")

        duration = self._wav_duration(path)
        m, s = divmod(int(duration), 60)

        name = os.path.basename(path)
        parts = name.replace("recording_", "").replace(".wav", "").split("_chunk")
        ts_part    = parts[0] if len(parts) == 2 else name
        chunk_part = f"ch{parts[1]}" if len(parts) == 2 else ""

        flags = ""
        if has_txt: flags += "✓"
        if has_dia: flags += "👤"
        if has_sum: flags += "📋"
        if has_loc: flags += "📍"
        if not flags: flags = "⋯"

        label = f"  {ts_part.replace('_',' ')}  {chunk_part}  {m:02d}:{s:02d}  {flags}  "
        btn = ctk.CTkButton(
            self.list_frame, text=label, anchor="w",
            height=32, font=("Consolas", 11),
            fg_color="transparent",
            hover_color=("#e5e7eb", "#374151"),
            text_color=("#111827", "#f3f4f6"),
            command=lambda p=path: self._select(p),
        )
        return btn

    def _select(self, path: str):
        self._selected_path = path
        self._show_content(path)
        for row in self._recording_rows:
            try:
                base = os.path.basename(path).replace(".wav", "")
                row.configure(fg_color=(
                    "#dbeafe" if base.split("_chunk")[0].replace("recording_", "")
                    in row.cget("text") and
                    (len(base.split("_chunk")) < 2 or
                     f"ch{base.split('_chunk')[1]}" in row.cget("text"))
                    else "transparent",
                    "#1e3a8a" if base.split("_chunk")[0].replace("recording_", "")
                    in row.cget("text") and
                    (len(base.split("_chunk")) < 2 or
                     f"ch{base.split('_chunk')[1]}" in row.cget("text"))
                    else "transparent",
                ))
            except Exception:
                pass
        loc = load_location_sidecar(path)
        if loc:
            self.map_widget.set_position(loc["lat"], loc["lon"])

    # ---------- content display ----------
    def _show_content(self, path: Optional[str]):
        self._show_summary(path)
        self._show_transcript(path)

    def _show_summary(self, path: Optional[str]):
        self.txt_summary.configure(state="normal")
        self.txt_summary.delete("1.0", "end")
        if path is None:
            self.txt_summary.configure(state="disabled")
            return
        summary_path = os.path.splitext(path)[0] + ".summary.txt"
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                self.txt_summary.insert("end", text)
            except Exception:
                self.txt_summary.insert("end", "(error reading summary)")
        else:
            self.txt_summary.insert("end",
                                    "No summary yet. Auto-summarize will run "
                                    "after transcription, or click ↻ Re-summarize.")
        self.txt_summary.configure(state="disabled")

    def _show_transcript(self, path: Optional[str]):
        self.txt_transcript.configure(state="normal")
        self.txt_transcript.delete("1.0", "end")
        if path is None:
            self.lbl_transcript_target.configure(text="(no recording selected)")
            self.txt_transcript.configure(state="disabled")
            return

        self.lbl_transcript_target.configure(text=os.path.basename(path))
        json_path = os.path.splitext(path)[0] + ".json"

        if not os.path.exists(json_path):
            self.txt_transcript.insert(
                "end",
                "No transcript yet. Click 📝 to generate one.")
            self.txt_transcript.configure(state="disabled")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.txt_transcript.insert("end", f"(error reading JSON: {e})")
            self.txt_transcript.configure(state="disabled")
            return

        for seg in data.get("segments", []):
            speaker = seg.get("speaker")
            conf    = seg.get("speaker_confidence", 0.0)
            kind    = seg.get("speaker_kind", "unknown")
            text    = seg["text"].strip()

            if self.config.GUI_SHOW_TIMESTAMPS:
                ts = (f"[{self._fmt_ts(seg['start'])} → "
                      f"{self._fmt_ts(seg['end'])}]  ")
                self.txt_transcript.insert("end", ts)

            if speaker:
                if kind == "strict":
                    tag = f"[{speaker}]"
                elif kind == "weak":
                    tag = f"[{speaker} — {conf:.0%}]"
                else:
                    tag = f"[{speaker}]"

                self.txt_transcript.insert("end", tag + "  ")

            self.txt_transcript.insert("end", text + "\n\n")

        self.txt_transcript.configure(state="disabled")

    # ---------- map ----------
    def _refresh_map(self):
        for marker, _ in self._map_markers:
            try: marker.delete()
            except Exception: pass
        self._map_markers.clear()

        import glob
        files = sorted(glob.glob(
            os.path.join(self.config.RECORDINGS_DIR, "*.wav")))
        located = []
        for path in files:
            loc = load_location_sidecar(path)
            if loc:
                located.append((loc["lat"], loc["lon"], path))

        if not located:
            return

        clusters = self._cluster_pins(located, self.config.MAP_CLUSTER_RADIUS_M)
        for cluster in clusters:
            avg_lat = sum(c[0] for c in cluster) / len(cluster)
            avg_lon = sum(c[1] for c in cluster) / len(cluster)
            paths   = [c[2] for c in cluster]
            text    = (f"{len(cluster)} recordings" if len(cluster) > 1
                       else os.path.basename(paths[0]).split("_chunk")[0].replace(
                           "recording_", ""))
            marker = self.map_widget.set_marker(
                avg_lat, avg_lon, text=text,
                command=lambda m, ps=paths: self._select(max(ps, key=os.path.getmtime)))
            self._map_markers.append((marker, paths))

    @staticmethod
    def _cluster_pins(points, radius_m):
        unassigned = list(points)
        clusters = []
        while unassigned:
            seed = unassigned.pop(0)
            cluster = [seed]
            remaining = []
            for p in unassigned:
                if any(AudioStreamGUI._hav_m(p[0], p[1], q[0], q[1]) <= radius_m
                       for q in cluster):
                    cluster.append(p)
                else:
                    remaining.append(p)
            unassigned = remaining
            clusters.append(cluster)
        return clusters

    @staticmethod
    def _hav_m(lat1, lon1, lat2, lon2):
        R = 6_371_000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return 2*R*math.asin(math.sqrt(a))

    # ---------- helpers ----------
    @staticmethod
    def _fmt_ts(s: float) -> str:
        m = int(s // 60); sec = s - m * 60
        return f"{m:02d}:{sec:05.2f}"

    @staticmethod
    def _wav_duration(path: str) -> float:
        try:
            with wave.open(path, "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0


# ---------- Standalone window wrapper ----------
# Lets main_phase9.py keep working as-is. It builds a real top-level
# window, sets its title/size, and drops the AudioStreamGUI frame inside.
class AudioStreamWindow(ctk.CTk):
    """
    Standalone window that hosts the AudioStreamGUI frame.
    Used by main_phase9.py — preserves the exact pre-M1 behaviour:
    window title, size, color theme, and close-shuts-down-controller.
    """
    def __init__(self, controller, config):
        super().__init__()
        ctk.set_appearance_mode(config.GUI_APPEARANCE)
        ctk.set_default_color_theme(config.GUI_COLOR_THEME)

        self.title("ESP32 Audio Stream — Phase 9")
        self.geometry(f"{config.GUI_WINDOW_W}x{config.GUI_WINDOW_H}")
        self.minsize(1200, 750)

        self.gui = AudioStreamGUI(self, controller, config)
        self.gui.pack(fill="both", expand=True)
        self.controller = controller
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.controller.shutdown()
        self.destroy()