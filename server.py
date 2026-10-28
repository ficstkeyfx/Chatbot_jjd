"""Máy chủ web cho trợ lý sức khỏe: REST API + SSE + phục vụ giao diện Vue đã build."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr

import auth
import core
from storage import Store, StorageError
from lesson_content import MAX_FILE_BYTES, MAX_TEXT_CHARS, LessonContentError
from voice_backends import VoiceBackendError
from zalo_bridge import ZaloBridge
from zalo_accounts import AccountBots

logger = logging.getLogger("health_bot.server")

WEB_DIST = Path(__file__).parent / "web" / "dist"
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
MAX_MESSAGE_CHARS = 4000
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
# Để DB_PATH trống để tắt lưu trữ (chỉ giữ trong RAM).
DB_PATH = os.getenv("DB_PATH", "data/healthbot.db").strip()


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    return default if not value else value not in {"0", "false", "no", "off"}


# Đăng nhập bắt buộc theo mặc định. Đặt AUTH_REQUIRED=0 để quay lại chế độ ẩn danh (chỉ nên dùng trên máy cá nhân).
AUTH_REQUIRED = _flag("AUTH_REQUIRED", True)
ALLOW_REGISTRATION = _flag("ALLOW_REGISTRATION", True)
TRUST_PROXY = _flag("TRUST_PROXY", False)  # tin X-Forwarded-For/Proto khi đứng sau nginx/Caddy
MAX_ACCOUNTS = int(os.getenv("MAX_ACCOUNTS", "0"))  # 0 = không giới hạn
SESSION_COOKIE = "hb_session"
SESSION_DAYS = int(os.getenv("SESSION_DAYS", "30"))
throttle = auth.LoginThrottle()
_registrations: dict[str, deque[float]] = defaultdict(deque)


# ---------------------------------------------------------------------------
# Phiên làm việc và sự kiện thời gian thực
# ---------------------------------------------------------------------------

class EventHub:
    """Đẩy sự kiện hẹn giờ tới mọi tab đang mở của cùng một phiên."""

    def __init__(self) -> None:
        self._queues: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)

    def subscribe(self, sid: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=50)
        self._queues[sid].add(queue)
        return queue

    def unsubscribe(self, sid: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._queues[sid].discard(queue)
        if not self._queues[sid]:
            self._queues.pop(sid, None)

    async def publish(self, sid: str, event: dict[str, Any]) -> None:
        for queue in list(self._queues.get(sid, ())):
            if queue.full():
                continue
            queue.put_nowait(event)


hub = EventHub()
_hits: dict[str, deque[float]] = defaultdict(deque)


def client_ip(request: Request) -> str:
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "?"


def _account_from_cookie(request: Request) -> dict[str, Any] | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token or core.store is None:
        return None
    return core.store.get_session_account(token)


def _identify(request: Request, x_session_id: str | None, query_sid: str | None = None) -> str | None:
    """Danh tính người dùng: tài khoản đăng nhập (mặc định) hoặc mã phiên ẩn danh (AUTH_REQUIRED=0)."""
    if AUTH_REQUIRED:
        account = _account_from_cookie(request)
        return f"acct-{account['id']}" if account else None
    sid = x_session_id or query_sid
    return sid if sid and SESSION_ID.match(sid) else None


def session_id(request: Request, x_session_id: str | None = Header(default=None)) -> str:
    uid = _identify(request, x_session_id)
    if uid is None:
        if AUTH_REQUIRED:
            raise HTTPException(status_code=401, detail="Bạn cần đăng nhập.")
        raise HTTPException(status_code=400, detail="Thiếu hoặc sai mã phiên (X-Session-Id).")
    core.hydrate(uid)
    return uid


def limited(sid: str = Depends(session_id)) -> str:
    """Giới hạn số yêu cầu mỗi phút của một người dùng để tránh lạm dụng tác vụ nặng."""
    now = time.monotonic()
    hits = _hits[sid]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Bạn gửi quá nhanh. Vui lòng chờ một chút.")
    hits.append(now)
    return sid


async def _purge_loop(store: Store) -> None:
    """Mỗi giờ xóa dữ liệu của người dùng không hoạt động quá RETENTION_DAYS ngày."""
    while True:
        try:
            removed = await asyncio.to_thread(store.purge, core.RETENTION_DAYS)
            if removed:
                logger.info("đã xóa dữ liệu hết hạn của %d phiên", removed)
        except Exception as exc:
            logger.error("dọn dữ liệu hết hạn thất bại: %s", exc)
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    store = Store(DB_PATH) if DB_PATH else None
    core.configure_storage(store)
    bridge = ZaloBridge(store)
    app.state.zalo = bridge
    account_bots = AccountBots(store, bridge)
    app.state.zalo_bots = account_bots

    async def notify(uid, event):
        await hub.publish(uid, event)
        await bridge.notify(uid, event)
        await account_bots.notify(uid, event)

    core.set_notifier(notify)
    if AUTH_REQUIRED and store is None:
        raise RuntimeError("Đăng nhập cần cơ sở dữ liệu: hãy đặt DB_PATH hoặc đặt AUTH_REQUIRED=0.")
    if AUTH_REQUIRED and store and store.count_accounts() == 0 and not ALLOW_REGISTRATION:
        logger.warning("Chưa có tài khoản nào và đăng ký đang đóng: chạy `python manage.py create-user TÊN`.")
    if core.DEFAULT_BACKEND == "openai" and not os.getenv("OPENAI_API_KEY"):
        logger.warning(
            "DEFAULT_BACKEND=openai nhưng chưa đặt OPENAI_API_KEY: mỗi người dùng phải tự nhập API key trong Cài đặt."
        )
    purge_task = asyncio.create_task(_purge_loop(store)) if store else None
    logger.info(
        "đăng nhập: %s (đăng ký: %s) | lưu trữ: %s (giữ %d ngày) | LLM mặc định: %s",
        "bắt buộc" if AUTH_REQUIRED else "TẮT (chế độ ẩn danh)",
        "mở" if ALLOW_REGISTRATION else "đóng",
        DB_PATH or "chỉ RAM",
        core.RETENTION_DAYS,
        core.DEFAULT_BACKEND,
    )
    try:
        await account_bots.start()
        if _flag("PRELOAD_LOCAL_VOICE", False):
            voice_health = await core._ensure_local_voice("stt")
            if not voice_health.get("tts", {}).get("ready"):
                raise RuntimeError("Local TTS chưa sẵn sàng: " + str(voice_health.get("tts", {}).get("error")))
            logger.info("Local STT/TTS models are loaded; web is ready")
        yield
    finally:
        core.set_notifier(None)
        await account_bots.stop()
        await bridge.stop()
        if purge_task:
            purge_task.cancel()
        await core.shutdown()
        core.configure_storage(None)
        if store:
            store.close()


app = FastAPI(title="Trợ lý sức khỏe", version="2.0", lifespan=lifespan)

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(self), geolocation=(), microphone=(self)",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    ),
}


@app.middleware("http")
async def protect(request: Request, call_next):
    """Chống CSRF (yêu cầu thay đổi dữ liệu phải có header riêng, trình duyệt không tự thêm được từ trang khác)
    và thêm header bảo mật cho mọi phản hồi."""
    if (
        request.method in {"POST", "PUT", "PATCH", "DELETE"}
        and request.url.path.startswith("/api/")
        and request.headers.get("x-hb-client") != "1"
    ):
        response: Response = JSONResponse(status_code=403, content={"error": "Yêu cầu không hợp lệ."})
    else:
        response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


# ---------------------------------------------------------------------------
# Xử lý lỗi: luôn trả {"error": "..."} bằng tiếng Việt
# ---------------------------------------------------------------------------

def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    message = "Không tìm thấy." if exc.status_code == 404 else str(exc.detail)
    return _error(exc.status_code, message)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _error(422, "Dữ liệu gửi lên không hợp lệ (quá dài, trống hoặc sai định dạng).")


@app.exception_handler(core.BusyError)
async def busy_error(_: Request, exc: core.BusyError) -> JSONResponse:
    return _error(429, str(exc))


@app.exception_handler(LessonContentError)
async def content_error(_: Request, exc: LessonContentError) -> JSONResponse:
    return _error(400, str(exc))


@app.exception_handler(VoiceBackendError)
async def voice_error(_: Request, exc: VoiceBackendError) -> JSONResponse:
    return _error(503, f"Không thể xử lý voice: {exc}")


@app.exception_handler(ValueError)
async def value_error(_: Request, exc: ValueError) -> JSONResponse:
    return _error(400, str(exc))


@app.exception_handler(RuntimeError)
async def runtime_error(_: Request, exc: RuntimeError) -> JSONResponse:
    logger.warning("request failed: %s", exc)
    return _error(502, f"Không thể tạo câu trả lời: {exc}")


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unexpected failure")
    return _error(500, "Đã xảy ra lỗi hệ thống. Vui lòng thử lại.")


async def read_upload(file: UploadFile, limit: int = MAX_FILE_BYTES) -> bytes:
    data = await file.read(limit + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Tệp tải lên trống.")
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Tệp vượt quá giới hạn {limit // (1024 * 1024)} MB.")
    return data


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class PomodoroRequest(BaseModel):
    work_minutes: int = core.POMODORO_DEFAULTS[0]
    break_minutes: int = core.POMODORO_DEFAULTS[1]
    cycles: int = core.POMODORO_DEFAULTS[2]


class EyeRequest(BaseModel):
    enabled: bool


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def linked_account(request: Request) -> dict[str, Any]:
    account = _account_from_cookie(request) if AUTH_REQUIRED else None
    if account is None or core.store is None:
        raise HTTPException(status_code=401, detail="Hãy đăng nhập tài khoản web để liên kết Zalo.")
    return account


@app.get("/api/integrations/zalo")
async def zalo_status(request: Request, account=Depends(linked_account)):
    link = core.store.zalo_link(account["id"])
    bridge = request.app.state.zalo_bots.bridge_for(account['id'])
    return {"enabled": bridge.enabled, "linked": link is not None,
            "linked_at": int(link["created_at"] * 1000) if link else None,
            "mirror_web": bridge.mirror, "bot": core.store.zalo_bot(account['id'])}


class ZaloBotConfig(BaseModel):
    token: SecretStr


@app.put("/api/integrations/zalo/bot")
async def configure_zalo_bot(body: ZaloBotConfig, request: Request, account=Depends(linked_account), sid=Depends(limited)):
    try:
        await request.app.state.zalo_bots.configure(account['id'], body.token.get_secret_value())
    except StorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return await zalo_status(request, account)


@app.delete("/api/integrations/zalo/bot")
async def remove_zalo_bot(request: Request, account=Depends(linked_account), sid=Depends(limited)):
    await request.app.state.zalo_bots.remove(account['id'])
    return await zalo_status(request, account)


@app.post("/api/integrations/zalo/code")
async def zalo_code(request: Request, account=Depends(linked_account), sid=Depends(limited)):
    bridge = request.app.state.zalo_bots.bridge_for(account['id'])
    if not bridge.enabled:
        raise HTTPException(status_code=503, detail="Máy chủ chưa bật Zalo. Cần cấu hình ZALO_ENABLED=1 và ZALO_BOT_TOKEN.")
    return core.store.create_zalo_code(account["id"], bot_id=bridge.bot_id)


@app.delete("/api/integrations/zalo")
async def zalo_unlink(request: Request, account=Depends(linked_account), sid=Depends(limited)):
    core.store.unlink_zalo(account["id"])
    await core._emit(sid, {"type": "sync", "source": "zalo"})
    return {"ok": True}


@app.post("/zalo/webhook")
async def zalo_webhook(request: Request):
    bridge = request.app.state.zalo
    supplied = request.headers.get("X-Bot-Api-Secret-Token", "")
    if not bridge.enabled or bridge.mode != "webhook" or not bridge.secret:
        raise HTTPException(status_code=404)
    if not secrets.compare_digest(supplied, bridge.secret):
        raise HTTPException(status_code=403)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 64 * 1024:
            raise HTTPException(status_code=413)
    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(status_code=400)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400)
    bridge.receive(payload)
    return {"ok": True}


@app.get("/zalo/media/{ticket}.aac")
async def zalo_media(request: Request, ticket: str):
    for bridge in [request.app.state.zalo, *request.app.state.zalo_bots.bots.values()]:
        item = bridge.media.get(ticket) if bridge.enabled else None
        if item and item[0] > time.time():
            return Response(item[1], media_type="audio/aac", headers={"Cache-Control": "no-store"})
        bridge.media.pop(ticket, None)
    raise HTTPException(status_code=404)


@app.get("/api/session")
async def session_info(sid: str = Depends(session_id)) -> dict[str, Any]:
    return {"settings": core.get_settings(sid), "timers": core.timer_state(sid)}


# ---------------------------------------------------------------------------
# Tài khoản
# ---------------------------------------------------------------------------

class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=auth.MAX_PASSWORD + 1)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=auth.MAX_PASSWORD + 1)
    new_password: str = Field(min_length=1, max_length=auth.MAX_PASSWORD + 1)


def _set_session_cookie(request: Request, response: Response, token: str) -> None:
    secure = request.url.scheme == "https" or (
        TRUST_PROXY and request.headers.get("x-forwarded-proto", "").lower() == "https"
    )
    response.set_cookie(
        SESSION_COOKIE, token, max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax", secure=secure, path="/"
    )


def _user_info(account: dict[str, Any] | None) -> dict[str, Any] | None:
    return {"username": account["username"]} if account else None


@app.get("/api/auth/me")
async def auth_me(request: Request) -> dict[str, Any]:
    account = _account_from_cookie(request) if AUTH_REQUIRED else None
    return {
        "required": AUTH_REQUIRED,
        "registration": AUTH_REQUIRED and ALLOW_REGISTRATION,
        "user": _user_info(account),
    }


@app.post("/api/auth/register")
async def register(body: Credentials, request: Request, response: Response) -> dict[str, Any]:
    if not AUTH_REQUIRED or core.store is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy.")
    if not ALLOW_REGISTRATION:
        raise HTTPException(status_code=403, detail="Đăng ký đang đóng. Hãy liên hệ quản trị viên để có tài khoản.")
    ip = client_ip(request)
    now = time.monotonic()
    recent = _registrations[ip]
    while recent and now - recent[0] > 3600:
        recent.popleft()
    if len(recent) >= 10:
        raise HTTPException(status_code=429, detail="Bạn đã tạo quá nhiều tài khoản. Hãy thử lại sau.")
    username = auth.validate_username(body.username)
    auth.validate_password(body.password, username)
    if MAX_ACCOUNTS and core.store.count_accounts() >= MAX_ACCOUNTS:
        raise HTTPException(status_code=403, detail="Hệ thống đã đạt số tài khoản tối đa.")
    password_hash = await asyncio.to_thread(auth.hash_password, body.password)
    try:
        account_id = core.store.create_account(username, password_hash)
    except StorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    recent.append(now)
    token = core.store.create_session(account_id, SESSION_DAYS * 86400)
    _set_session_cookie(request, response, token)
    return {"required": True, "user": {"username": username}}


@app.post("/api/auth/login")
async def login(body: Credentials, request: Request, response: Response) -> dict[str, Any]:
    if not AUTH_REQUIRED or core.store is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy.")
    username = body.username.strip()
    key = throttle.key(client_ip(request), username)
    wait = throttle.remaining_lock(key)
    if wait:
        raise HTTPException(status_code=429, detail=f"Sai quá nhiều lần. Hãy thử lại sau {wait} giây.")
    account = core.store.get_account(username)
    ok = await asyncio.to_thread(auth.verify_or_burn, body.password, account["password_hash"] if account else None)
    if not ok or account is None:
        throttle.record_failure(key)
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu.")
    throttle.reset(key)
    token = core.store.create_session(account["id"], SESSION_DAYS * 86400)
    _set_session_cookie(request, response, token)
    return {"required": True, "user": {"username": account["username"]}}


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response) -> dict[str, bool]:
    token = request.cookies.get(SESSION_COOKIE)
    if token and core.store is not None:
        core.store.delete_session(token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@app.post("/api/auth/password")
async def change_password(body: PasswordChange, request: Request, sid: str = Depends(limited)) -> dict[str, bool]:
    account = _account_from_cookie(request)
    if not AUTH_REQUIRED or account is None or core.store is None:
        raise HTTPException(status_code=400, detail="Chỉ dùng được khi đã đăng nhập.")
    stored = core.store.get_account_by_id(account["id"])
    if stored is None or not await asyncio.to_thread(auth.verify_password, body.current_password, stored["password_hash"]):
        raise HTTPException(status_code=401, detail="Mật khẩu hiện tại không đúng.")
    auth.validate_password(body.new_password, account["username"])
    core.store.set_password_hash(account["id"], await asyncio.to_thread(auth.hash_password, body.new_password))
    # Đăng xuất mọi thiết bị khác (giữ phiên hiện tại).
    core.store.delete_other_sessions(account["id"], request.cookies.get(SESSION_COOKIE))
    return {"ok": True}


@app.post("/api/chat")
async def chat(body: ChatRequest, sid: str = Depends(limited)) -> dict[str, Any]:
    return await core.handle_text(sid, body.message)


@app.post("/api/image")
async def image(
    file: UploadFile = File(...),
    caption: str = Form(default="", max_length=MAX_MESSAGE_CHARS),
    expression: bool = Form(default=False),
    expression_profile: Literal['', 'balanced', 'camera', 'patt-lite'] = Form(default=''),
    sid: str = Depends(limited),
) -> dict[str, Any]:
    data = await read_upload(file)
    return await core.process_image(
        sid, data, file.filename or "photo.jpg", file.content_type or "", caption,
        expression=expression,
        expression_profile=expression_profile,
    )


@app.post("/api/voice")
async def voice(file: UploadFile = File(...), sid: str = Depends(limited)) -> dict[str, Any]:
    data = await read_upload(file, core.MAX_AUDIO_BYTES)
    return await core.process_voice(
        sid, data, file.filename or "voice.webm", file.content_type or ""
    )


@app.get("/api/voice/status")
async def voice_status(sid: str = Depends(session_id)) -> dict[str, Any]:
    return await core.voice_status(sid)


@app.post("/api/tts")
async def tts(body: SpeechRequest, sid: str = Depends(limited)) -> Response:
    audio = await core.synthesize_speech(sid, body.text)
    return Response(content=audio.data, media_type=audio.mime_type)


@app.post("/api/summarize")
async def summarize(
    text: str = Form(default="", max_length=MAX_TEXT_CHARS),
    file: UploadFile | None = File(default=None),
    sid: str = Depends(limited),
) -> dict[str, Any]:
    if file is not None and file.filename:
        data = await read_upload(file)
        return await core.summarize_upload(sid, data, file.filename, file.content_type or "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Hãy dán văn bản hoặc chọn một tệp để tóm tắt.")
    return await core.summarize_text(sid, text)


@app.get("/api/settings")
async def get_settings(sid: str = Depends(session_id)) -> dict[str, Any]:
    return core.get_settings(sid)


@app.put("/api/settings")
async def put_settings(body: dict[str, Any], sid: str = Depends(limited)) -> dict[str, Any]:
    return await asyncio.to_thread(core.update_settings, sid, body)


class RoleTest(BaseModel):
    role: str


@app.post("/api/settings/test")
async def test_settings(body: RoleTest, sid: str = Depends(limited)) -> dict[str, Any]:
    """Thử kết nối thật tới nhà cung cấp đã chọn (dùng API key của chính người dùng)."""
    return await core.check_role(sid, body.role)


@app.get("/api/providers/local")
async def local_resources(sid: str = Depends(session_id)) -> dict[str, Any]:
    """Model Ollama và giọng Piper đang có sẵn trên máy chủ (nếu đã cài)."""
    return await core.local_resources()


@app.get("/api/messages")
async def messages(sid: str = Depends(session_id)) -> dict[str, Any]:
    return {"messages": core.list_messages(sid), "persistent": core.store is not None}


@app.get("/api/messages/{message_id}/image")
async def message_image(message_id: int, request: Request, sid: str | None = Query(default=None)) -> Response:
    uid = _identify(request, request.headers.get("x-session-id"), sid)
    if uid is None:
        raise HTTPException(status_code=401, detail="Bạn cần đăng nhập.")
    data = await asyncio.to_thread(core.store.message_image, uid, message_id) if core.store else None
    if data is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh.")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


@app.delete("/api/me")
async def delete_me(request: Request, response: Response, sid: str = Depends(limited)) -> dict[str, bool]:
    """Xóa mọi dữ liệu của người dùng (và tài khoản nếu đang đăng nhập)."""
    if AUTH_REQUIRED and sid.startswith('acct-'):
        await request.app.state.zalo_bots.remove(int(sid[5:]))
    core.forget_me(sid)
    if AUTH_REQUIRED and core.store is not None:
        account = _account_from_cookie(request)
        if account:
            core.store.delete_account(account["id"])
        response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@app.delete("/api/history")
async def delete_history(sid: str = Depends(limited)) -> dict[str, bool]:
    core.clear_history(sid)
    return {"ok": True}


@app.get("/api/timers")
async def timers(sid: str = Depends(session_id)) -> dict[str, Any]:
    return core.timer_state(sid)


@app.post("/api/timers/pomodoro")
async def start_pomodoro(body: PomodoroRequest, sid: str = Depends(limited)) -> dict[str, Any]:
    core.start_pomodoro(
        sid, [str(body.work_minutes), str(body.break_minutes), str(body.cycles)]
    )
    return core.timer_state(sid)


@app.post("/api/timers/eye")
async def toggle_eye(body: EyeRequest, sid: str = Depends(limited)) -> dict[str, Any]:
    if body.enabled:
        core.start_eye_reminder(sid)
    else:
        core.stop_timers(sid, "eye")
    return core.timer_state(sid)


@app.delete("/api/timers/{target}")
async def stop_timers(target: str, sid: str = Depends(limited)) -> dict[str, Any]:
    core.stop_timers(sid, target)
    return core.timer_state(sid)


@app.get("/api/events")
async def events(request: Request, sid: str | None = Query(default=None)) -> StreamingResponse:
    """Luồng SSE. Chế độ đăng nhập dùng cookie; chế độ ẩn danh dùng mã phiên qua query (EventSource không đặt được header)."""
    uid = _identify(request, None, sid)
    if uid is None:
        raise HTTPException(status_code=401 if AUTH_REQUIRED else 400, detail="Bạn cần đăng nhập." if AUTH_REQUIRED else "Mã phiên không hợp lệ.")
    queue = hub.subscribe(uid)

    async def stream() -> AsyncIterator[str]:
        try:
            yield "retry: 3000\n\n"
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            hub.unsubscribe(uid, queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Giao diện Vue đã build (npm run build trong thư mục web/)
# ---------------------------------------------------------------------------

if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Không tìm thấy.")
        candidate = (WEB_DIST / path).resolve()
        if path and candidate.is_file() and WEB_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
else:

    @app.get("/", include_in_schema=False)
    async def missing_build() -> JSONResponse:
        return JSONResponse(
            {"message": "Chưa build giao diện. Chạy: cd web && npm install && npm run build"},
            status_code=200,
        )


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("LOG_LEVEL", "info").lower())


if __name__ == "__main__":
    main()
