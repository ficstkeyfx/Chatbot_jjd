from __future__ import annotations

import argparse
import ctypes
import io
import json
import logging
import os
import re
import tempfile
import threading
import wave
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tts_text import prepare_for_speech
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("health_bot.local_voice")

MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))
MAX_TEXT_CHARS = int(os.getenv("LOCAL_TTS_MAX_CHARS", "4096"))
# Quãng nghỉ giữa hai câu (giây) ở tốc độ bình thường.
SENTENCE_PAUSE_SECONDS = float(os.getenv("LOCAL_TTS_SENTENCE_PAUSE", "0.35"))
VOICE_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,80}$")
PIPER_VOICE_ID = re.compile(r"^[a-z]{2}_[A-Z]{2}-[A-Za-z0-9_]+-(?:x_low|low|medium|high)$")
# Giọng tiếng Việt của Piper có thể tải về khi được chọn.
KNOWN_VI_VOICES = ("vi_VN-vais1000-medium", "vi_VN-vivos-x_low", "vi_VN-25hours_single-low")


class LocalVoiceEngine:
    """Own the local STT/TTS models inside the isolated voice process."""

    def __init__(self) -> None:
        self.model_dir = Path(os.getenv("LOCAL_VOICE_MODEL_DIR", ".models/voice")).resolve()
        self.stt_model_name = os.getenv("LOCAL_STT_MODEL", "small")
        self.stt_device = os.getenv("LOCAL_STT_DEVICE", "cuda")
        self.stt_compute_type = os.getenv("LOCAL_STT_COMPUTE_TYPE", "float16")
        self.tts_provider = os.getenv("LOCAL_TTS_PROVIDER", "piper").strip().lower()
        if self.tts_provider not in {"piper", "vieneu"}:
            raise ValueError("LOCAL_TTS_PROVIDER phải là piper hoặc vieneu.")
        self.tts_voice_name = os.getenv(
            "LOCAL_TTS_VOICE", "vieneu-default" if self.tts_provider == "vieneu" else "vi_VN-vais1000-medium"
        )
        self._stt: Any = None
        self._tts: Any = None
        self._extra_voices: dict[str, Any] = {}
        self._stt_error: str | None = None
        self._tts_error: str | None = None
        self._stt_lock = threading.Lock()
        self._tts_lock = threading.Lock()
        self._dll_directories: list[Any] = []

    def _check_cuda_runtime(self) -> None:
        if self.stt_device != "cuda" or os.name != "nt":
            return

        import ctranslate2

        search_dirs = [Path(ctranslate2.__file__).resolve().parent]
        configured_dir = os.getenv("LOCAL_CUDA_DLL_DIR", "").strip()
        if configured_dir:
            search_dirs.insert(0, Path(configured_dir).resolve())
        for directory in search_dirs:
            if directory.is_dir():
                self._dll_directories.append(os.add_dll_directory(str(directory)))

        missing: list[str] = []
        for dll_name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"):
            try:
                ctypes.WinDLL(dll_name)
            except OSError:
                missing.append(dll_name)
        if missing:
            raise RuntimeError(
                "thiếu "
                + ", ".join(missing)
                + ". Cài CUDA 12/cuDNN 9 hoặc đặt LOCAL_CUDA_DLL_DIR tới thư mục chứa các DLL."
            )

    def initialize(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        try:
            from faster_whisper import WhisperModel

            try:
                self._check_cuda_runtime()
                self._stt = self._load_stt(WhisperModel)
            except Exception as exc:
                if not self._should_fall_back_to_cpu(exc):
                    raise
                logger.warning(
                    "CUDA không dùng được (%s). Chuyển sang CPU (int8); "
                    "đặt LOCAL_STT_DEVICE=cpu để tắt cảnh báo này, hoặc LOCAL_STT_STRICT_CUDA=1 để bắt buộc dùng GPU.",
                    str(exc).strip() or exc.__class__.__name__,
                )
                self.stt_device, self.stt_compute_type = "cpu", "int8"
                self._stt = self._load_stt(WhisperModel)
            self._stt_error = None
            logger.info("Local STT is ready")
        except Exception as exc:  # Import, model download and CUDA failures are health data.
            self._stt_error = self._describe_inference_error(exc)
            logger.exception("Local STT initialization failed")
        if self.tts_provider == "vieneu":
            try:
                self._ensure_vieneu()
            except RuntimeError:
                logger.exception("VieNeu initialization failed")

    def _load_stt(self, model_class: Any) -> Any:
        logger.info(
            "Loading STT model=%s device=%s compute_type=%s",
            self.stt_model_name,
            self.stt_device,
            self.stt_compute_type,
        )
        return model_class(
            self.stt_model_name,
            device=self.stt_device,
            compute_type=self.stt_compute_type,
            download_root=str(self.model_dir / "faster-whisper"),
        )

    def _should_fall_back_to_cpu(self, exc: Exception) -> bool:
        if self.stt_device != "cuda" or os.getenv("LOCAL_STT_STRICT_CUDA", "").strip().lower() in {"1", "true", "yes"}:
            return False
        return any(word in str(exc).lower() for word in ("cuda", "cudnn", "cublas", "dll", "cufft", "nvrtc"))

    @staticmethod
    def _describe_inference_error(exc: Exception) -> str:
        detail = (str(exc).strip() or exc.__class__.__name__).rstrip(".")
        lowered = detail.lower()
        if "cuda" in lowered or "cudnn" in lowered or "cublas" in lowered:
            return (
                f"CUDA không sẵn sàng: {detail}. "
                "Cần CUDA 12 và cuDNN tương thích (đang bật LOCAL_STT_STRICT_CUDA nên không chuyển sang CPU)."
            )
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            return f"Thiếu dependency voice local: {detail}. Hãy cài requirements.txt."
        return f"Không khởi tạo được STT local: {detail}"

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self._stt is not None else "error",
            "stt": {
                "ready": self._stt is not None,
                "model": self.stt_model_name,
                "device": self.stt_device,
                "compute_type": self.stt_compute_type,
                "error": self._stt_error,
            },
            "tts": {
                "ready": self._tts is not None,
                "provider": self.tts_provider,
                "model": self.tts_voice_name,
                "state": "ready" if self._tts is not None else "error" if self._tts_error else "lazy",
                "error": self._tts_error,
            },
        }

    def transcribe(self, audio: bytes, filename: str) -> str:
        if self._stt is None:
            raise RuntimeError(self._stt_error or "STT local chưa sẵn sàng.")

        suffix = Path(filename).suffix or ".ogg"
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
                audio_file.write(audio)
                temp_path = audio_file.name
            with self._stt_lock:
                segments, _ = self._stt.transcribe(
                    temp_path,
                    language="vi",
                    beam_size=5,
                    vad_filter=True,
                )
                return " ".join(
                    segment.text.strip() for segment in segments if segment.text.strip()
                )
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    def _ensure_tts(self, voice_name: str | None = None) -> Any:
        """Nạp giọng Piper (mặc định hoặc theo tên). Giọng chưa có trên máy sẽ được tải về lần đầu dùng."""
        name = voice_name or self.tts_voice_name
        if name == self.tts_voice_name and self._tts is not None:
            return self._tts
        if name in self._extra_voices:
            return self._extra_voices[name]
        if not VOICE_NAME.match(name) or ("/" in name or "\\" in name):
            raise ValueError("Tên giọng không hợp lệ.")
        with self._tts_lock:
            if name == self.tts_voice_name and self._tts is not None:
                return self._tts
            if name in self._extra_voices:
                return self._extra_voices[name]
            try:
                from piper import PiperVoice
                from piper.download_voices import download_voice

                model_path = self.model_dir / f"{name}.onnx"
                config_path = self.model_dir / f"{name}.onnx.json"
                if not model_path.exists() or not config_path.exists():
                    if not PIPER_VOICE_ID.match(name):
                        raise RuntimeError(f"Giọng {name!r} chưa được cài trên máy chủ.")
                    logger.info("Downloading Piper voice %s", name)
                    download_voice(name, self.model_dir)
                voice = PiperVoice.load(model_path, config_path=config_path)
                if name == self.tts_voice_name:
                    self._tts = voice
                    self._tts_error = None
                else:
                    self._extra_voices[name] = voice
                logger.info("Local TTS voice %s is ready", name)
                return voice
            except Exception as exc:
                message = self._describe_inference_error(exc).replace("STT", "TTS")
                if name == self.tts_voice_name:
                    self._tts_error = message
                logger.exception("Local TTS initialization failed")
                raise RuntimeError(message) from exc

    def _ensure_vieneu(self) -> Any:
        with self._tts_lock:
            if self._tts is None:
                try:
                    from vieneu_tts import VieneuSpeech

                    self._tts = VieneuSpeech()
                    self._tts_error = None
                    logger.info("VieNeu TTS is ready (CPU ONNX)")
                except Exception as exc:
                    self._tts_error = f"Không khởi tạo được VieNeu: {exc}"
                    raise RuntimeError(self._tts_error) from exc
            return self._tts

    def list_voices(self) -> dict[str, Any]:
        """Giọng Piper đã cài trên máy chủ và giọng mặc định."""
        if self.tts_provider == "vieneu":
            installed = list(self._tts.voices) if self._tts is not None else []
            return {"provider": "vieneu", "default": self.tts_voice_name, "installed": installed, "downloadable": []}
        installed = sorted(
            path.name[: -len(".onnx")]
            for path in self.model_dir.glob("*.onnx")
            if (self.model_dir / f"{path.name}.json").exists()
        ) if self.model_dir.is_dir() else []
        return {"default": self.tts_voice_name, "installed": installed, "downloadable": list(KNOWN_VI_VOICES)}

    def synthesize(self, text: str, speed: float = 1.0, voice_name: str | None = None) -> bytes:
        """Đọc văn bản: chuẩn hóa -> tách câu -> tổng hợp từng câu -> nối với quãng nghỉ."""
        if self.tts_provider == "vieneu":
            engine = self._ensure_vieneu()
            with self._tts_lock:
                return engine.synthesize(text, speed, voice_name or self.tts_voice_name)
        voice = self._ensure_tts(voice_name)
        sentences = prepare_for_speech(text)
        if not sentences:
            raise ValueError("Không có nội dung có thể đọc thành tiếng.")
        syn_config = None
        if abs(speed - 1.0) > 0.01:
            try:
                from piper import SynthesisConfig

                # length_scale lớn hơn 1 thì giọng đọc chậm hơn.
                syn_config = SynthesisConfig(length_scale=1.0 / speed)
            except ImportError:
                logger.warning("Phiên bản piper hiện tại không hỗ trợ chỉnh tốc độ đọc")

        params = None
        frames: list[bytes] = []
        for sentence in sentences:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                if syn_config is not None:
                    voice.synthesize_wav(sentence, wav_file, syn_config=syn_config)
                else:
                    voice.synthesize_wav(sentence, wav_file)
            buffer.seek(0)
            with wave.open(buffer, "rb") as reader:
                if params is None:
                    params = reader.getparams()
                    # Quãng nghỉ giữa các câu: lâu hơn khi người dùng chọn đọc chậm.
                    pause_seconds = SENTENCE_PAUSE_SECONDS / max(0.5, min(1.5, speed))
                    silence = b"\x00" * (
                        int(params.framerate * pause_seconds) * params.sampwidth * params.nchannels
                    )
                elif reader.getparams()[:3] != params[:3]:
                    raise RuntimeError("Các đoạn âm thanh TTS có định dạng không khớp nhau.")
                frames.append(reader.readframes(reader.getnframes()))

        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(params.nchannels)
            wav_file.setsampwidth(params.sampwidth)
            wav_file.setframerate(params.framerate)
            for index, chunk in enumerate(frames):
                wav_file.writeframes(chunk)
                if index < len(frames) - 1:
                    wav_file.writeframes(silence)
        return output.getvalue()


def parse_multipart_audio(content_type: str, body: bytes) -> tuple[bytes, str]:
    envelope = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
        + body
    )
    message = BytesParser(policy=policy.default).parsebytes(envelope)
    if not message.is_multipart():
        raise ValueError("Yêu cầu transcribe phải là multipart/form-data.")
    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") != "file":
            continue
        data = part.get_payload(decode=True) or b""
        filename = part.get_filename() or "voice.ogg"
        if not data:
            raise ValueError("File audio rỗng.")
        return data, os.path.basename(filename)
    raise ValueError("Thiếu trường file trong yêu cầu transcribe.")


class VoiceRequestHandler(BaseHTTPRequestHandler):
    server_version = "ZaloLocalVoice/1.0"

    @property
    def engine(self) -> LocalVoiceEngine:
        return self.server.engine  # type: ignore[attr-defined, no-any-return]

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self, limit: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length không hợp lệ.") from exc
        if length <= 0:
            raise ValueError("Request body rỗng.")
        if length > limit:
            raise OverflowError("Request body vượt quá giới hạn cho phép.")
        return self.rfile.read(length)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/voices":
            self._json(HTTPStatus.OK, self.engine.list_voices())
            return
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        health = self.engine.health()
        status = HTTPStatus.OK if health["status"] == "ok" else HTTPStatus.SERVICE_UNAVAILABLE
        self._json(status, health)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/transcribe":
                body = self._read_body(MAX_AUDIO_BYTES + 64 * 1024)
                audio, filename = parse_multipart_audio(self.headers.get("Content-Type", ""), body)
                if len(audio) > MAX_AUDIO_BYTES:
                    raise OverflowError("Voice note vượt quá giới hạn 10 MB.")
                self._json(HTTPStatus.OK, {"text": self.engine.transcribe(audio, filename)})
                return
            if self.path == "/speech":
                body = self._read_body(64 * 1024)
                payload = json.loads(body)
                text = str(payload.get("text", "")).strip()
                if not text:
                    raise ValueError("Thiếu nội dung cần đọc.")
                speed = min(1.5, max(0.5, float(payload.get("speed", 1.0))))
                voice_name = str(payload.get("voice") or "").strip()
                if voice_name and not VOICE_NAME.match(voice_name):
                    raise ValueError("Tên giọng không hợp lệ.")
                if voice_name:
                    audio = self.engine.synthesize(text[:MAX_TEXT_CHARS], speed, voice_name)
                elif abs(speed - 1.0) > 0.01:
                    audio = self.engine.synthesize(text[:MAX_TEXT_CHARS], speed)
                else:
                    audio = self.engine.synthesize(text[:MAX_TEXT_CHARS])
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(audio)))
                self.end_headers()
                self.wfile.write(audio)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except OverflowError as exc:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except RuntimeError as exc:
            logger.warning("Voice inference unavailable: %s", exc)
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
        except Exception as exc:  # Keep implementation details in server logs.
            logger.exception("Voice request failed")
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Voice inference thất bại: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.client_address[0], format % args)


class VoiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], engine: LocalVoiceEngine):
        super().__init__(address, VoiceRequestHandler)
        self.engine = engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Local STT/TTS backend for health-bot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prepare", action="store_true", help="Download/load STT and TTS, then exit")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Local voice server chỉ được phép bind vào loopback.")

    engine = LocalVoiceEngine()
    engine.initialize()
    if args.prepare:
        if engine.health()["status"] != "ok":
            raise SystemExit(engine.health()["stt"]["error"])
        if engine.tts_provider == "vieneu":
            engine._ensure_vieneu()
        else:
            engine._ensure_tts()
        logger.info("Voice models are prepared")
        return

    server = VoiceHTTPServer((args.host, args.port), engine)
    logger.info("Local voice server listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
