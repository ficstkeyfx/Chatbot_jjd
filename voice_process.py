from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class LocalVoiceProcess:
    """Start and stop the loopback voice service owned by this bot process."""

    def __init__(self, base_url: str, *, startup_timeout: float | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.startup_timeout = startup_timeout or float(os.getenv("LOCAL_VOICE_STARTUP_TIMEOUT", "300"))
        self.autostart = os.getenv("LOCAL_VOICE_AUTOSTART", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def health(self, timeout: float = 5) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}/health")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {"error": f"Voice backend trả về HTTP {exc.code}."}
            payload.setdefault("status", "error")
            return payload
        except urllib.error.URLError as exc:
            return {"status": "unreachable", "error": f"Không kết nối được voice backend: {exc.reason}"}
        except TimeoutError:
            return {"status": "unreachable", "error": "Voice backend phản hồi quá thời gian."}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"status": "error", "error": "Health response của voice backend không hợp lệ."}

    def _local_address(self) -> tuple[str, int] | None:
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return None
        return parsed.hostname, parsed.port or 80

    def ensure_started(self) -> dict[str, Any]:
        health = self.health(timeout=1)
        if health.get("status") == "ok":
            return health
        if not self.autostart:
            health["error"] = f"{health.get('error', '')} LOCAL_VOICE_AUTOSTART đang tắt.".strip()
            return health

        address = self._local_address()
        if address is None:
            health["error"] = (
                f"{health.get('error', '')} Bot chỉ tự khởi động backend dùng URL loopback HTTP.".strip()
            )
            return health

        with self._lock:
            health = self.health(timeout=1)
            if health.get("status") == "ok":
                return health
            if self._process is None or self._process.poll() is not None:
                server_path = Path(__file__).with_name("local_voice_server.py")
                host, port = address
                self._process = subprocess.Popen(
                    [sys.executable, str(server_path), "--host", host, "--port", str(port)],
                    cwd=str(server_path.parent),
                )

            deadline = time.monotonic() + self.startup_timeout
            last_health = health
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    return {
                        "status": "error",
                        "error": f"Voice backend đã thoát với mã {self._process.returncode}. Xem log console để biết chi tiết.",
                    }
                last_health = self.health(timeout=2)
                if last_health.get("status") in {"ok", "error"}:
                    return last_health
                time.sleep(0.25)
            return {
                "status": "unreachable",
                "error": (
                    f"Voice backend chưa sẵn sàng sau {self.startup_timeout:g} giây. "
                    f"Chi tiết cuối: {last_health.get('error', 'không có')}"
                ),
            }

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
