"""Danh mục nhà cung cấp mô hình và cấu hình theo từng tài khoản.

Mỗi tài khoản chọn nhà cung cấp + model cho 4 vai trò:
  llm    mô hình trả lời
  vision đọc ảnh (VLM); có thể là "ocr" (chỉ đọc chữ) hoặc "llm" (dùng luôn mô hình trả lời)
  stt    nhận dạng giọng nói
  tts    đọc thành tiếng

Hầu hết nhà cung cấp dùng API tương thích OpenAI nên chung một đường gọi. Module này không gọi mạng
(trừ kiểm tra địa chỉ tùy chỉnh) để dễ kiểm thử.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import urllib.parse
from dataclasses import dataclass, field

ROLES = ("llm", "vision", "stt", "tts")
PRIVATE_URL_ENV = "ALLOW_PRIVATE_BASE_URL"


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    base_url: str | None  # None: lấy từ cấu hình máy chủ (local) hoặc do người dùng nhập (custom)
    roles: frozenset[str]
    needs_key: bool = True
    custom_url: bool = False
    env_key: str | None = None
    note: str = ""
    models: dict[str, str] = field(default_factory=dict)  # model mặc định theo vai trò
    suggestions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    voices: tuple[str, ...] = ()


_OPENAI_VOICES = ("alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse", "marin", "cedar")

PROVIDERS: dict[str, Provider] = {
    p.id: p
    for p in (
        Provider(
            "local", "Chạy trên máy chủ (Ollama / Whisper / VieNeu hoặc Piper)", None,
            frozenset(ROLES), needs_key=False,
            note="Dữ liệu ở lại máy chủ. Chỉ dùng được khi quản trị viên đã cài sẵn các dịch vụ này.",
        ),
        Provider(
            "openai", "OpenAI", "https://api.openai.com/v1", frozenset(ROLES), env_key="OPENAI_API_KEY",
            models={"llm": "gpt-4.1-mini", "vision": "gpt-4.1-mini", "stt": "gpt-4o-mini-transcribe", "tts": "gpt-4o-mini-tts"},
            suggestions={
                "llm": ("gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"),
                "vision": ("gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"),
                "stt": ("gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"),
                "tts": ("gpt-4o-mini-tts", "tts-1", "tts-1-hd"),
            },
            voices=_OPENAI_VOICES,
        ),
        Provider(
            "gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
            frozenset({"llm", "vision"}), env_key="GEMINI_API_KEY",
            note="Dùng lớp tương thích OpenAI của Gemini. Chưa hỗ trợ giọng nói.",
            models={"llm": "gemini-2.5-flash", "vision": "gemini-2.5-flash"},
            suggestions={
                "llm": ("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"),
                "vision": ("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"),
            },
        ),
        Provider(
            "anthropic", "Anthropic Claude (thử nghiệm)", "https://api.anthropic.com/v1",
            frozenset({"llm", "vision"}), env_key="ANTHROPIC_API_KEY",
            note="Dùng lớp tương thích OpenAI của Anthropic (thử nghiệm). Chưa hỗ trợ giọng nói.",
            models={"llm": "claude-sonnet-5-5", "vision": "claude-sonnet-5-5"},
            suggestions={
                "llm": ("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"),
                "vision": ("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"),
            },
        ),
        Provider(
            "openrouter", "OpenRouter", "https://openrouter.ai/api/v1", frozenset({"llm", "vision"}),
            env_key="OPENROUTER_API_KEY",
            note="Một key dùng được nhiều model; nhập đúng tên model theo trang của OpenRouter.",
            models={"llm": "openai/gpt-4.1-mini", "vision": "openai/gpt-4.1-mini"},
            suggestions={"llm": ("openai/gpt-4.1-mini", "google/gemini-2.5-flash"), "vision": ("openai/gpt-4.1-mini",)},
        ),
        Provider(
            "groq", "Groq", "https://api.groq.com/openai/v1", frozenset({"llm", "stt"}), env_key="GROQ_API_KEY",
            models={"llm": "llama-3.3-70b-versatile", "stt": "whisper-large-v3-turbo"},
            suggestions={
                "llm": ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
                "stt": ("whisper-large-v3-turbo", "whisper-large-v3"),
            },
        ),
        Provider(
            "custom", "Tùy chỉnh (API tương thích OpenAI)", None, frozenset(ROLES), needs_key=False, custom_url=True,
            env_key=None,
            note="Nhập địa chỉ gốc API (ví dụ https://host/v1), tên model và API key. Dùng cho Azure OpenAI, vLLM, LM Studio...",
        ),
    )
}

# Giá trị đặc biệt của từng vai trò (ngoài nhà cung cấp).
SPECIAL = {
    "vision": {"ocr": "Chỉ đọc chữ (OCR)", "llm": "Dùng luôn mô hình trả lời"},
    "stt": {"off": "Tắt"},
    "tts": {"off": "Tắt"},
}

MODEL_NAME = re.compile(r"^[\w.:/@+\-]{1,120}$")
VOICE_NAME = re.compile(r"^[\w.\-]{1,80}$")
API_KEY = re.compile(r"^[\x21-\x7E]{8,400}$")  # ký tự in được, không khoảng trắng/xuống dòng (chống chèn header)
USERNAME = re.compile(r"^[A-Za-z0-9_.\-]{3,32}$")


class ProfileError(ValueError):
    """Cấu hình không hợp lệ; thông điệp hiển thị trực tiếp cho người dùng."""


def provider(provider_id: str) -> Provider:
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        raise ProfileError(f"Nhà cung cấp không hợp lệ: {provider_id!r}.") from None


def valid_choices(role: str) -> list[str]:
    return [*SPECIAL.get(role, {}), *(p.id for p in PROVIDERS.values() if role in p.roles)]


def private_urls_allowed() -> bool:
    return os.getenv(PRIVATE_URL_ENV, "").strip().lower() in {"1", "true", "yes"}


def validate_base_url(url: str) -> str:
    """Chỉ nhận http(s) tới máy chủ công khai (chặn SSRF vào mạng nội bộ) trừ khi quản trị viên cho phép."""
    value = url.strip().rstrip("/")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or len(value) > 300:
        raise ProfileError("Địa chỉ API phải là URL http(s) hợp lệ.")
    if parsed.username or parsed.password:
        raise ProfileError("Địa chỉ API không được chứa tên đăng nhập/mật khẩu.")
    if not private_urls_allowed():
        if parsed.scheme != "https":
            raise ProfileError("Địa chỉ API tùy chỉnh phải dùng https.")
        assert_public_host(parsed.hostname, parsed.port or 443)
    return value


def assert_public_host(host: str, port: int) -> None:
    """Từ chối địa chỉ nội bộ/loopback/link-local (kể cả khi tên miền trỏ về chúng)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ProfileError(f"Không phân giải được tên miền {host}.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ProfileError("Địa chỉ API trỏ tới mạng nội bộ nên bị chặn.")


def default_profile(
    *, default_backend: str, local_vision_model: str, default_voice_mode: str
) -> dict[str, dict[str, str]]:
    """Cấu hình mặc định cho tài khoản mới, suy ra từ cấu hình máy chủ (.env)."""
    llm_provider = default_backend if default_backend in PROVIDERS else "local"
    if local_vision_model:
        vision = "local"
    else:
        vision = "llm" if llm_provider != "local" else "ocr"
    voice = {"local": "local", "cloud": "openai", "off": "off"}.get(default_voice_mode, "local")
    return {
        "llm": {"provider": llm_provider, "model": "", "base_url": ""},
        "vision": {"provider": vision, "model": local_vision_model if vision == "local" else "", "base_url": ""},
        "stt": {"provider": voice, "model": "", "base_url": ""},
        "tts": {"provider": voice, "model": "", "base_url": "", "voice": ""},
    }


def normalize_role(
    role: str, value: dict[str, object], current: dict[str, str], check_url: bool = True
) -> dict[str, str]:
    """Ghép thay đổi vào cấu hình hiện tại của một vai trò và kiểm tra hợp lệ."""
    if role not in ROLES:
        raise ProfileError(f"Vai trò không hợp lệ: {role!r}.")
    if not isinstance(value, dict):
        raise ProfileError("Cấu hình phải là một đối tượng.")
    unknown = set(value) - {"provider", "model", "base_url", "voice"}
    if unknown:
        raise ProfileError(f"Trường không hợp lệ: {', '.join(sorted(unknown))}.")
    merged = {**current}
    for key in ("provider", "model", "base_url", "voice"):
        if key in value:
            if not isinstance(value[key], str):
                raise ProfileError(f"{key} phải là chuỗi.")
            merged[key] = value[key].strip()

    choice = merged.get("provider", "")
    if choice not in valid_choices(role):
        raise ProfileError("Nhà cung cấp này không hỗ trợ chức năng đã chọn.")
    if "provider" in value and value["provider"].strip() != current.get("provider"):
        # Đổi nhà cung cấp thì model/giọng cũ không còn ý nghĩa: về mặc định trừ khi người dùng nhập mới.
        merged.setdefault("model", "")
        if "model" not in value:
            merged["model"] = ""
        if "voice" not in value:
            merged["voice"] = ""
        if "base_url" not in value:
            merged["base_url"] = ""
    if merged.get("model") and not MODEL_NAME.match(merged["model"]):
        raise ProfileError("Tên model không hợp lệ (chỉ gồm chữ, số và . _ : / @ + -).")
    if merged.get("voice") and not VOICE_NAME.match(merged["voice"]):
        raise ProfileError("Tên giọng không hợp lệ.")
    if choice == "custom":
        if not merged.get("base_url"):
            raise ProfileError("Nhà cung cấp tùy chỉnh cần địa chỉ API.")
        merged["base_url"] = (
            validate_base_url(merged["base_url"]) if check_url else merged["base_url"].strip().rstrip("/")
        )
        if role in {"llm", "vision", "stt", "tts"} and not merged.get("model"):
            raise ProfileError("Nhà cung cấp tùy chỉnh cần tên model.")
    else:
        merged["base_url"] = ""
    if role != "tts":
        merged.pop("voice", None)
    return merged


def validate_api_key(key: str) -> str:
    value = key.strip()
    if not API_KEY.match(value):
        raise ProfileError("API key không hợp lệ (8-400 ký tự, không chứa khoảng trắng).")
    return value


def public_catalog() -> list[dict[str, object]]:
    """Danh mục gửi cho giao diện (không chứa bí mật)."""
    return [
        {
            "id": p.id,
            "label": p.label,
            "roles": sorted(p.roles),
            "needs_key": p.needs_key,
            "custom_url": p.custom_url,
            "note": p.note,
            "models": p.models,
            "suggestions": {role: list(items) for role, items in p.suggestions.items()},
            "voices": list(p.voices),
        }
        for p in PROVIDERS.values()
    ]
