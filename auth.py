"""Xác thực: băm mật khẩu (scrypt, chỉ dùng thư viện chuẩn), kiểm tra đầu vào và chống dò mật khẩu."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field

from providers import USERNAME

MIN_PASSWORD = 8
MAX_PASSWORD = 200
_N, _R, _P = 2**14, 8, 1

# Mật khẩu hay bị đoán nhất; không chấp nhận. Danh sách ngắn có chủ đích, chỉ chặn trường hợp hiển nhiên.
COMMON_PASSWORDS = frozenset(
    "12345678 123456789 1234567890 password password1 passw0rd qwertyui qwerty123 11111111 00000000 "
    "abcd1234 iloveyou admin123 matkhau123 12341234".split()
)


class AuthError(ValueError):
    """Lỗi xác thực có thông điệp tiếng Việt hiển thị cho người dùng."""


def validate_username(username: str) -> str:
    value = username.strip()
    if not USERNAME.match(value):
        raise AuthError("Tên đăng nhập gồm 3-32 ký tự: chữ không dấu, số, dấu chấm, gạch dưới hoặc gạch ngang.")
    return value


def validate_password(password: str, username: str = "") -> str:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        raise AuthError(f"Mật khẩu cần ít nhất {MIN_PASSWORD} ký tự.")
    if len(password) > MAX_PASSWORD:
        raise AuthError("Mật khẩu quá dài.")
    if password.lower() in COMMON_PASSWORDS or (username and password.lower() == username.lower()):
        raise AuthError("Mật khẩu quá dễ đoán. Hãy chọn mật khẩu khác.")
    return password


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(
        _N, _R, _P, base64.b64encode(salt).decode(), base64.b64encode(digest).decode()
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
            dklen=len(base64.b64decode(expected)), maxmem=128 * 1024 * 1024,
        )
        return hmac.compare_digest(digest, base64.b64decode(expected))
    except (ValueError, TypeError):
        return False


# Băm giả để thời gian phản hồi không tiết lộ tên đăng nhập có tồn tại hay không.
_DUMMY_HASH = hash_password("khong-ai-biet-mat-khau-nay")


def verify_or_burn(password: str, stored: str | None) -> bool:
    if stored is None:
        verify_password(password, _DUMMY_HASH)
        return False
    return verify_password(password, stored)


@dataclass
class LoginThrottle:
    """Khóa tạm thời sau nhiều lần sai liên tiếp (theo IP + tên đăng nhập), lưu trong RAM."""

    max_failures: int = 5
    window_seconds: int = 300
    lock_seconds: int = 300
    _failures: dict[str, list[float]] = field(default_factory=dict)
    _locked_until: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def key(ip: str, username: str) -> str:
        return f"{ip}|{username.lower()}"

    def remaining_lock(self, key: str, now: float | None = None) -> int:
        now = now if now is not None else time.monotonic()
        until = self._locked_until.get(key, 0.0)
        if until <= now:
            self._locked_until.pop(key, None)
            return 0
        return int(until - now) + 1

    def record_failure(self, key: str, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        recent = [t for t in self._failures.get(key, []) if now - t < self.window_seconds]
        recent.append(now)
        self._failures[key] = recent
        if len(recent) >= self.max_failures:
            self._locked_until[key] = now + self.lock_seconds
            self._failures.pop(key, None)

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)
        self._locked_until.pop(key, None)
