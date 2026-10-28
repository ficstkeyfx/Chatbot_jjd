import json
import mimetypes
import os
import secrets
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class VoiceBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class SynthesizedAudio:
    data: bytes
    mime_type: str
    filename: str


class VoiceBackend(ABC):
    @abstractmethod
    def transcribe(self, audio: bytes, filename: str, mime_type: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def synthesize(self, text: str, speed: float = 1.0) -> SynthesizedAudio:
        raise NotImplementedError

    def health(self) -> bool:
        return True


def _multipart_audio(
    fields: dict[str, str], audio: bytes, filename: str, mime_type: str
) -> tuple[bytes, str]:
    boundary = f"----HealthBot{secrets.token_hex(12)}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")
    safe_filename = os.path.basename(filename).replace('"', "") or "voice.ogg"
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'.encode()
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode())
    body.extend(audio)
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _open_response(
    request: urllib.request.Request, timeout: int = 120
) -> tuple[bytes, str]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get_content_type()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise VoiceBackendError(f"Voice API trả về HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        reason = str(exc.reason)
        if isinstance(exc.reason, ConnectionRefusedError):
            raise VoiceBackendError(
                f"Voice backend chưa chạy tại {request.full_url}: {reason}"
            ) from exc
        raise VoiceBackendError(
            f"Không kết nối được voice backend tại {request.full_url}: {reason}"
        ) from exc
    except TimeoutError as exc:
        raise VoiceBackendError(
            f"Voice backend phản hồi quá thời gian tại {request.full_url}."
        ) from exc


def _open(request: urllib.request.Request, timeout: int = 120) -> bytes:
    return _open_response(request, timeout)[0]


class OpenAIVoiceBackend(VoiceBackend):
    """STT/TTS qua API tương thích OpenAI (OpenAI, Groq, máy chủ tùy chỉnh...)."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        *,
        stt_model: str | None = None,
        tts_model: str | None = None,
        voice: str | None = None,
    ):
        self.api_key = api_key
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.stt_model = stt_model or os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
        self.tts_model = tts_model or os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        self.voice = voice or os.getenv("OPENAI_TTS_VOICE", "marin")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def transcribe(self, audio: bytes, filename: str, mime_type: str) -> str:
        body, content_type = _multipart_audio(
            {"model": self.stt_model, "language": "vi"},
            audio,
            filename,
            mime_type or mimetypes.guess_type(filename)[0] or "audio/ogg",
        )
        request = urllib.request.Request(
            f"{self.base_url}/audio/transcriptions",
            data=body,
            headers={**self.headers, "Content-Type": content_type},
            method="POST",
        )
        try:
            result = json.loads(_open(request))
            return str(result["text"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise VoiceBackendError("Phản hồi STT không đúng định dạng.") from exc

    def synthesize(self, text: str, speed: float = 1.0) -> SynthesizedAudio:
        payload = json.dumps(
            {
                "model": self.tts_model,
                "voice": self.voice,
                "input": text[:4096],
                "speed": speed,
                "response_format": "mp3",
                "instructions": "Nói tiếng Việt bình tĩnh, ấm áp, rõ ràng và không kịch tính.",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/audio/speech",
            data=payload,
            headers={**self.headers, "Content-Type": "application/json"},
            method="POST",
        )
        return SynthesizedAudio(_open(request), "audio/mpeg", "response.mp3")


class LocalVoiceBackend(VoiceBackend):
    def __init__(self, base_url: str, voice: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.voice = voice or None

    def list_voices(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}/voices")
        try:
            return json.loads(_open(request, timeout=5))
        except (VoiceBackendError, json.JSONDecodeError, TypeError):
            return {"default": None, "installed": [], "downloadable": []}

    def health(self) -> bool:
        return self.health_info().get("status") == "ok"

    def health_info(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}/health")
        try:
            return json.loads(_open(request, timeout=5))
        except VoiceBackendError as exc:
            return {"status": "unreachable", "error": str(exc)}
        except (json.JSONDecodeError, TypeError):
            return {"status": "error", "error": "Health response không đúng JSON."}

    def transcribe(self, audio: bytes, filename: str, mime_type: str) -> str:
        body, content_type = _multipart_audio({}, audio, filename, mime_type)
        request = urllib.request.Request(
            f"{self.base_url}/transcribe",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            result = json.loads(_open(request))
            return str(result["text"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise VoiceBackendError("Phản hồi STT local không đúng định dạng.") from exc

    def synthesize(self, text: str, speed: float = 1.0) -> SynthesizedAudio:
        request = urllib.request.Request(
            f"{self.base_url}/speech",
            data=json.dumps({"text": text, "speed": speed, **({"voice": self.voice} if self.voice else {})}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        data, mime_type = _open_response(request)
        mime_type = mime_type or "audio/wav"
        extension = ".wav" if mime_type == "audio/wav" else ".mp3"
        return SynthesizedAudio(data, mime_type, f"response{extension}")
