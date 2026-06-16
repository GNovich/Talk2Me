"""Lightweight attendant dashboard — stdlib http.server, no external deps.

Feature 16: real-time visibility and control for the gallery attendant.

Endpoints:
    GET  /             — serve the single-page dashboard HTML
    GET  /status       — JSON snapshot of session state (updated each turn)
    POST /control      — {"action": "reset"|"panic"|"shutdown"} → sends signal

The server runs in a background daemon thread so it never blocks the pipeline.
State is written by run_loop() via update_status(); reads are lock-protected.

Usage (app.py wires this):
    from talk2me.ui.server import DashboardServer
    ui_server = DashboardServer(port=cfg["ui_port"])
    ui_server.start()
    ui_server.update_status(phase=1, turn=2, ...)
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

# Path to the bundled HTML dashboard
_HTML_PATH = Path(__file__).parent / "dashboard.html"


class _Status:
    """Thread-safe shared state dict updated each turn by run_loop()."""

    _lock = threading.Lock()
    _data: dict = {
        "phase": 0,
        "turn": 0,
        "tier": 0,
        "alpha": 1.0,
        "health": "—",
        "session_active": False,
        "last_transcript": "",
        "last_question": "",
        "voiced_s": 0.0,
        "stt_s": 0.0,
        "tts_s": 0.0,
        "total_s": 0.0,
        "pid": os.getpid(),
        "uptime_s": 0.0,
        "transcript_history": [],   # list of {transcript, question} dicts
        "started_at": time.time(),
    }

    @classmethod
    def update(cls, **kwargs: Any) -> None:
        with cls._lock:
            cls._data.update(kwargs)
            cls._data["uptime_s"] = time.time() - cls._data["started_at"]

    @classmethod
    def snapshot(cls) -> dict:
        with cls._lock:
            return dict(cls._data)


class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for the dashboard server."""

    def log_message(self, fmt, *args):
        pass  # suppress access log noise in production

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._serve_dashboard()
        elif self.path == "/status":
            self._serve_json(_Status.snapshot())
        else:
            self._send(404, "text/plain", b"Not found")

    def do_POST(self) -> None:
        if self.path == "/control":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._send(400, "text/plain", b"Bad JSON")
                return
            action = payload.get("action", "")
            pid = _Status.snapshot()["pid"]
            try:
                if action == "reset":
                    os.kill(pid, signal.SIGUSR1)
                    self._send_json({"ok": True, "action": "reset"})
                elif action == "panic":
                    os.kill(pid, signal.SIGUSR2)
                    self._send_json({"ok": True, "action": "panic"})
                elif action == "shutdown":
                    os.kill(pid, signal.SIGTERM)
                    self._send_json({"ok": True, "action": "shutdown"})
                else:
                    self._send(400, "text/plain", b"Unknown action")
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)})
        else:
            self._send(404, "text/plain", b"Not found")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _serve_dashboard(self) -> None:
        if _HTML_PATH.exists():
            html = _HTML_PATH.read_bytes()
        else:
            html = _FALLBACK_HTML.encode()
        self._send(200, "text/html; charset=utf-8", html)

    def _serve_json(self, data: dict) -> None:
        self._send_json(data)

    def _send_json(self, data: dict) -> None:
        body = json.dumps(data).encode()
        self._send(200, "application/json", body)

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


class DashboardServer:
    """Background HTTP server hosting the attendant dashboard.

    Start with start(); update state each turn with update_status().
    """

    def __init__(self, port: int = 8765, host: str = "127.0.0.1"):
        self.port = port
        self.host = host
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the server in a background daemon thread."""
        try:
            self._server = HTTPServer((self.host, self.port), _Handler)
        except OSError as exc:
            print(f"[ui] Warning: could not bind to {self.host}:{self.port} — {exc}")
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="talk2me-ui",
            daemon=True,  # dies automatically when main process exits
        )
        self._thread.start()
        print(f"[ui] Attendant dashboard → http://{self.host}:{self.port}/")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()

    def update_status(
        self,
        *,
        phase: int = 0,
        turn: int = 0,
        tier: int = 0,
        alpha: float = 1.0,
        health: str = "—",
        session_active: bool = True,
        last_transcript: str = "",
        last_question: str = "",
        voiced_s: float = 0.0,
        stt_s: float = 0.0,
        tts_s: float = 0.0,
        total_s: float = 0.0,
    ) -> None:
        """Push a fresh status snapshot. Called from run_loop() after each turn."""
        existing = _Status.snapshot()
        history = list(existing.get("transcript_history", []))
        if last_transcript or last_question:
            history.append({"transcript": last_transcript, "question": last_question})
            history = history[-20:]  # keep last 20 turns
        _Status.update(
            phase=phase,
            turn=turn,
            tier=tier,
            alpha=alpha,
            health=health,
            session_active=session_active,
            last_transcript=last_transcript,
            last_question=last_question,
            voiced_s=voiced_s,
            stt_s=stt_s,
            tts_s=tts_s,
            total_s=total_s,
            transcript_history=history,
        )

    def reset_session(self) -> None:
        """Clear per-session state (called when run_loop resets)."""
        _Status.update(
            phase=0, turn=0, tier=0, alpha=1.0, health="—",
            session_active=False, last_transcript="", last_question="",
            voiced_s=0.0, stt_s=0.0, tts_s=0.0, total_s=0.0,
            transcript_history=[],
        )


# ── minimal fallback if dashboard.html is missing ─────────────────────────────

_FALLBACK_HTML = """<!DOCTYPE html>
<html><head><title>Talk2Me — Dashboard</title></head>
<body><h2>Talk2Me Attendant Dashboard</h2>
<p>dashboard.html not found — status available at <a href="/status">/status</a></p>
</body></html>"""
