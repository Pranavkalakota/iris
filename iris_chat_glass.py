"""
iris_chat_glass.py — Project IRIS chat tab, PyQt6 glass version.
 
What it does:
  Standalone chat window with a real Windows acrylic/mica glass backdrop
  (via PyQt-Frameless-Window's DWM hooks). Replicates the Figma chat tab:
  cyan AI avatar, purple user avatar, face/voice/location pills, a snapshot
  card placeholder, suggestion chips, and a glass input bar. Replies come
  from local Ollama (llama3.2:3b) on a background thread; if Ollama isn't
  reachable it falls back to a stub reply so the UI still demos.
 
Why it exists:
  Proves the glass look on the target Windows machine and isolates the chat
  tab so the ChatTab widget can later be dropped into the full Qt IRIS shell.
  customtkinter can't do real backdrop blur; Qt + DWM can.
 
Run:
  pip install PyQt6 PyQt6-Frameless-Window pywin32 requests
  python iris_chat_glass.py
"""
 
import sys
 
try:
    from qframelesswindow import AcrylicWindow
except ImportError:
    print("Missing dependency. Run: pip install PyQt6-Frameless-Window pywin32")
    raise
 
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QFrame, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QScrollArea, QSizePolicy,
)
 
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
SYSTEM_PROMPT = "You are IRIS, a concise wearable assistant. Keep replies short."
 
# acrylic tint as RRGGBBAA — dark indigo, kept fairly opaque for readability
ACRYLIC_TINT = "201A40CC"
 
STYLE = """
#chatRoot { background: transparent; }
QScrollArea { border: none; background: transparent; }
 
QFrame[role="ai"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(255,255,255,0.10), stop:1 rgba(255,255,255,0.045));
    border: 1px solid rgba(255,255,255,0.10);
    border-top: 1px solid rgba(255,255,255,0.20);
    border-radius: 14px;
}
QFrame[role="user"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(167,139,250,0.22), stop:1 rgba(167,139,250,0.12));
    border: 1px solid rgba(167,139,250,0.30);
    border-top: 1px solid rgba(167,139,250,0.45);
    border-radius: 14px;
}
QLabel[role="msg"] { color: rgba(255,255,255,0.92); font-size: 14px; background: transparent; }
 
QLabel[role="aiAvatar"] {
    background: rgba(34,211,238,0.18); border: 1px solid rgba(34,211,238,0.5);
    border-radius: 17px; color: #67e8f9; font-size: 15px;
}
QLabel[role="userAvatar"] {
    background: rgba(167,139,250,0.20); border: 1px solid rgba(167,139,250,0.5);
    border-radius: 17px; color: #c4b5fd; font-size: 14px;
}
 
QLabel[role="pillFace"] {
    background: rgba(34,211,238,0.14); border: 1px solid rgba(34,211,238,0.35);
    color: #67e8f9; border-radius: 9px; padding: 2px 8px; font-size: 11px;
}
QLabel[role="pillVoice"] {
    background: rgba(167,139,250,0.14); border: 1px solid rgba(167,139,250,0.35);
    color: #c4b5fd; border-radius: 9px; padding: 2px 8px; font-size: 11px;
}
QLabel[role="pillLoc"] {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.18);
    color: rgba(255,255,255,0.70); border-radius: 9px; padding: 2px 8px; font-size: 11px;
}
 
QFrame[role="snap"] {
    background: rgba(0,0,0,0.28); border: 1px dashed rgba(255,255,255,0.18);
    border-radius: 10px;
}
QLabel[role="snapText"] { color: rgba(255,255,255,0.40); font-size: 12px; background: transparent; }
 
QPushButton[role="chip"] {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.14);
    color: rgba(255,255,255,0.78); border-radius: 13px; padding: 6px 13px; font-size: 12px;
}
QPushButton[role="chip"]:hover { background: rgba(255,255,255,0.10); }
 
QFrame[role="inputBar"] {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14);
    border-radius: 22px;
}
QLineEdit[role="input"] {
    background: transparent; border: none; color: rgba(255,255,255,0.92);
    font-size: 14px; padding: 0 4px;
}
QPushButton[role="send"] {
    background: rgba(34,211,238,0.85); color: #08222a; border: none;
    border-radius: 18px; font-size: 17px; font-weight: bold;
}
QPushButton[role="send"]:hover { background: rgba(34,211,238,1.0); }
"""
 
 
class OllamaWorker(QThread):
    """Calls Ollama off the UI thread so the window never freezes."""
    result = pyqtSignal(str)
 
    def __init__(self, messages):
        super().__init__()
        self.messages = messages
 
    def run(self):
        try:
            import requests
            r = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "messages": self.messages, "stream": False},
                timeout=60,
            )
            r.raise_for_status()
            self.result.emit(r.json()["message"]["content"].strip())
        except Exception:
            self.result.emit("(Ollama not reachable — start it with `ollama serve` "
                             "and pull llama3.2:3b. This is a placeholder reply.)")
 
 
def _avatar(letter, role):
    a = QLabel(letter)
    a.setProperty("role", role)
    a.setFixedSize(34, 34)
    a.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return a
 
 
def _pill(text, role):
    p = QLabel(text)
    p.setProperty("role", role)
    return p
 
 
class ChatTab(QWidget):
    """The reusable chat panel. Drop this into the full IRIS Qt shell later."""
 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chatRoot")
        self.setStyleSheet(STYLE)
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.worker = None
 
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 8, 14, 14)
        outer.setSpacing(10)
 
        # scrollable message area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.viewport().setStyleSheet("background: transparent;")
        self.msg_holder = QWidget()
        self.msg_holder.setStyleSheet("background: transparent;")
        self.msg_layout = QVBoxLayout(self.msg_holder)
        self.msg_layout.setContentsMargins(0, 0, 0, 0)
        self.msg_layout.setSpacing(16)
        self.msg_layout.addStretch()
        self.scroll.setWidget(self.msg_holder)
        outer.addWidget(self.scroll, 1)
 
        # suggestion chips
        chips = QHBoxLayout()
        chips.setSpacing(8)
        for label in ("Who else was there?", "Summarize the talk", "Jump to audio"):
            c = QPushButton(label)
            c.setProperty("role", "chip")
            c.setCursor(Qt.CursorShape.PointingHandCursor)
            c.clicked.connect(lambda _, t=label: self._send(t))
            chips.addWidget(c)
        chips.addStretch()
        outer.addLayout(chips)
 
        # input bar
        bar = QFrame()
        bar.setProperty("role", "inputBar")
        bar.setFixedHeight(44)
        bar_l = QHBoxLayout(bar)
        bar_l.setContentsMargins(16, 4, 4, 4)
        self.input = QLineEdit()
        self.input.setProperty("role", "input")
        self.input.setPlaceholderText("Ask IRIS anything…")
        self.input.returnPressed.connect(lambda: self._send(self.input.text()))
        send = QPushButton("↑")
        send.setProperty("role", "send")
        send.setFixedSize(36, 36)
        send.setCursor(Qt.CursorShape.PointingHandCursor)
        send.clicked.connect(lambda: self._send(self.input.text()))
        bar_l.addWidget(self.input, 1)
        bar_l.addWidget(send)
        outer.addWidget(bar)
 
        self._seed_demo()
 
    def _seed_demo(self):
        self.add_message(
            "You were talking with Ali about the ESP32 video transfer around 2:40pm.",
            "ai",
            pills=[("face: Ali", "pillFace"), ("voice match", "pillVoice"), ("lab", "pillLoc")],
        )
        self.add_message("Show me the snapshot from that moment.", "user")
        self.add_message("", "ai", snapshot="snapshot · 14:41:08")
 
    def add_message(self, text, sender, pills=None, snapshot=None):
        row = QHBoxLayout()
        row.setSpacing(10)
 
        if sender == "ai":
            avatar = _avatar("◉", "aiAvatar")
        else:
            avatar = _avatar("H", "userAvatar")
 
        bubble = QFrame()
        bubble.setProperty("role", sender)
        bubble.setMaximumWidth(300)
        bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        b_l = QVBoxLayout(bubble)
        b_l.setContentsMargins(14, 11, 14, 11)
        b_l.setSpacing(10)
 
        if text:
            lbl = QLabel(text)
            lbl.setProperty("role", "msg")
            lbl.setWordWrap(True)
            b_l.addWidget(lbl)
 
        if snapshot:
            snap = QFrame()
            snap.setProperty("role", "snap")
            snap.setFixedHeight(120)
            s_l = QVBoxLayout(snap)
            s_l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            st = QLabel("▢  " + snapshot)
            st.setProperty("role", "snapText")
            st.setAlignment(Qt.AlignmentFlag.AlignCenter)
            s_l.addWidget(st)
            bubble.setMaximumWidth(360)
            b_l.addWidget(snap)
 
        if pills:
            prow = QHBoxLayout()
            prow.setSpacing(6)
            for ptext, prole in pills:
                prow.addWidget(_pill(ptext, prole))
            prow.addStretch()
            b_l.addLayout(prow)
 
        if sender == "ai":
            row.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
            row.addWidget(bubble, 0, Qt.AlignmentFlag.AlignTop)
            row.addStretch()
        else:
            row.addStretch()
            row.addWidget(bubble, 0, Qt.AlignmentFlag.AlignTop)
            row.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
 
        self.msg_layout.insertLayout(self.msg_layout.count() - 1, row)
        QTimer.singleShot(0, self._scroll_bottom)
 
    def _scroll_bottom(self):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
 
    def _send(self, text):
        text = text.strip()
        if not text:
            return
        self.input.clear()
        self.add_message(text, "user")
        self.history.append({"role": "user", "content": text})
 
        self.worker = OllamaWorker(list(self.history))
        self.worker.result.connect(self._on_reply)
        self.worker.start()
 
    def _on_reply(self, reply):
        self.history.append({"role": "assistant", "content": reply})
        self.add_message(reply, "ai")
 
 
class GlassChatWindow(AcrylicWindow):
    """Standalone test host — applies the real DWM glass behind ChatTab."""
 
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IRIS — chat")
        self.resize(460, 780)
        # strong acrylic blur with dark indigo tint
        self.windowEffect.setAcrylicEffect(self.winId(), ACRYLIC_TINT)
        # On Win11 you can swap the line above for mica instead:
        # self.windowEffect.setMicaEffect(self.winId(), isDarkMode=True)
 
        self.tab = ChatTab(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, self.titleBar.height(), 0, 0)
        layout.addWidget(self.tab)
        self.titleBar.raise_()
 
 
def main():
    app = QApplication(sys.argv)
    win = GlassChatWindow()
    win.show()
    sys.exit(app.exec())
 
 
if __name__ == "__main__":
    main()