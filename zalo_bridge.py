"""Zalo Bot transport sharing the web app's core, account IDs and SQLite store."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import subprocess
import time
from urllib.parse import urlparse

import httpx

import core
from providers import assert_public_host
from storage import StorageError

logger = logging.getLogger("health_bot.zalo")
# The Bot API embeds its token in URLs. Never log HTTP request URLs.
for _http_logger in {"httpx", "httpx2", httpx.__name__}:
    logging.getLogger(_http_logger).setLevel(logging.WARNING)


def chunks(text: str):
    part, units = "", 0
    for char in text:
        cost = 2 if ord(char) > 0xFFFF else 1
        if units + cost > 1900:
            yield part
            part, units = "", 0
        part += char
        units += cost
    if part.strip():
        yield part


class ZaloAPI:
    def __init__(self, token: str):
        self.token = token
        self.http = httpx.AsyncClient(timeout=40)

    async def call(self, method: str, payload: dict | None = None):
        try:
            response = await self.http.post(f"https://bot-api.zaloplatforms.com/bot{self.token}/{method}", json=payload or {})
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError):
            raise RuntimeError(f"Zalo API {method} chưa kết nối được") from None
        if not isinstance(data, dict):
            raise RuntimeError(f"Zalo API {method} trả về dữ liệu không hợp lệ")
        if not data.get("ok"):
            # Zalo ends an idle long-poll with HTTP 200 + error_code=408.
            # This is an empty poll, not a lost connection or invalid token.
            if method == "getUpdates" and data.get("error_code") == 408:
                return None
            code = data.get("error_code", response.status_code)
            raise RuntimeError(f"Zalo API {method} từ chối yêu cầu ({code})")
        return data.get("result")

    async def close(self):
        await self.http.aclose()


class ZaloBridge:
    def __init__(self, store, *, api=None, owner_id=None, bot_id="", token=None):
        self.store = store
        token = token if token is not None else os.getenv("ZALO_BOT_TOKEN", "").strip()
        self.owner_id, self.bot_id = owner_id, bot_id
        self.enabled = bool(token) and (owner_id is not None or os.getenv("ZALO_ENABLED", "0").lower() in {"1", "true", "yes"})
        self.mode = "polling" if owner_id is not None else os.getenv("ZALO_MODE", "polling").strip().lower()
        self.secret = os.getenv("ZALO_WEBHOOK_SECRET", "").strip()
        self.public_url = os.getenv("ZALO_PUBLIC_BASE_URL", "").rstrip("/")
        self.mirror = os.getenv("ZALO_MIRROR_WEB", "1").lower() not in {"0", "false", "no"}
        self.api = api
        self.token = token
        self.tasks = []
        self.wakeup = asyncio.Event()
        self.media = {}

    async def start(self):
        if not self.enabled:
            return
        if self.store is None:
            raise RuntimeError("Zalo đồng bộ cần DB_PATH.")
        if self.mode not in {"polling", "webhook"}:
            raise RuntimeError("ZALO_MODE phải là polling hoặc webhook.")
        if self.mode == "webhook" and not 8 <= len(self.secret) <= 256:
            raise RuntimeError("Webhook Zalo cần ZALO_WEBHOOK_SECRET dài 8-256 ký tự.")
        self.api = self.api or ZaloAPI(self.token)
        self.tasks.append(asyncio.create_task(self.worker()))
        if self.mode == "polling":
            self.tasks.append(asyncio.create_task(self.poll()))
        logger.info("Zalo bridge enabled (%s); shared web database", self.mode)

    async def stop(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        if self.api:
            await self.api.close()
        self.media.clear()

    def receive(self, envelope: dict) -> bool:
        event = envelope.get("result", envelope)
        if not isinstance(event, dict):
            return False
        message = event.get("message")
        if not isinstance(message, dict):
            return False
        chat, sender = message.get("chat") or {}, message.get("from") or {}
        if not isinstance(chat, dict) or not isinstance(sender, dict):
            return False
        if chat.get("chat_type") != "PRIVATE" or sender.get("is_bot"):
            return False
        external_id, chat_id, message_id = sender.get("id"), chat.get("id"), message.get("message_id")
        if not all(isinstance(value, str) and 0 < len(value) <= 200 for value in (external_id, chat_id, message_id)):
            return False
        if self.bot_id:
            external_id = f"{self.bot_id}/{external_id}"
        text = str(message.get("text") or "")[:4000]
        payload = {"external_id": external_id, "chat_id": chat_id,
                   "event_name": event.get("event_name"), "text": text,
                   "caption": str(message.get("caption") or "")[:4000],
                   "photo": message.get("photo"), "voice_url": message.get("voice_url")}
        if text.startswith("/link "):
            payload["link_hash"] = hashlib.sha256(text.split(maxsplit=1)[1].strip().encode()).hexdigest()
            payload["text"] = "/link"  # Never persist the bearer linking code.
        event_id = hashlib.sha256(f"{external_id}:{chat_id}:{message_id}".encode()).hexdigest()
        added = self.store.enqueue_zalo_event(event_id, payload, bot_id=self.bot_id)
        self.wakeup.set()
        return added

    async def poll(self):
        while True:
            try:
                result = await self.api.call("getUpdates", {"timeout": 30})
                if isinstance(result, dict):
                    self.receive(result)
                elif isinstance(result, list):
                    for event in result:
                        self.receive(event)
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Zalo polling unavailable (%s); retrying", type(exc).__name__)
                await asyncio.sleep(5)

    async def download(self, url: str, limit: int) -> bytes:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            for _ in range(4):
                parsed = urlparse(url)
                if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError("URL media Zalo phải dùng HTTPS công khai.")
                await asyncio.to_thread(assert_public_host, parsed.hostname, parsed.port or 443)
                async with client.stream("GET", url) as response:
                    if response.is_redirect:
                        url = str(response.url.join(response.headers["location"]))
                        continue
                    response.raise_for_status()
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > limit:
                            raise ValueError("Tệp gửi từ Zalo quá lớn.")
                    return bytes(data)
        raise ValueError("URL media chuyển hướng quá nhiều lần.")

    async def handle(self, payload: dict) -> tuple[str, str]:
        external_id, chat_id = payload["external_id"], payload["chat_id"]
        text = payload.get("text", "")
        if payload.get("link_hash"):
            try:
                sid = self.store.consume_zalo_code(payload["link_hash"], external_id, chat_id, bot_id=self.bot_id, owner_id=self.owner_id)
                await core._emit(sid, {"type": "sync", "source": "zalo"})
                return "Đã liên kết Zalo với tài khoản web. Hội thoại, ngữ cảnh, cài đặt và nhắc nghỉ được dùng chung. Gõ /history để xem lịch sử chung.", ""
            except StorageError as exc:
                return str(exc), ""
        sid = self.store.zalo_sid(external_id)
        if self.owner_id is not None and sid != f"acct-{self.owner_id}":
            sid = None
        if not sid:
            return "Chưa xác nhận tài khoản Zalo nên mình chưa thể đọc lịch sử hoặc đồng bộ với web. Vào web → Cài đặt → Tài khoản → Liên kết Zalo, bấm Tạo mã xác nhận Zalo, sao chép nguyên lệnh /link có mã thật và gửi vào chat này. Chỉ cần làm một lần; không gửi token bot hay mật khẩu vào đây.", ""
        core.hydrate(sid)
        if text in {"/start", "/help"}:
            return "Bạn đang dùng chung tài khoản với web. Có thể nhắn văn bản, gửi ảnh hoặc voice; /history xem lịch sử, /status xem cấu hình, /clear xóa lịch sử chung. Hủy liên kết tại Cài đặt trên web.", sid
        if text in {"/history", "/lichsu"}:
            rows = self.store.list_messages(sid, 12)
            return "\n\n".join(f"{'Bạn' if row['role'] == 'user' else 'Trợ lý'}: {row['text']}" for row in rows) or "Lịch sử chung đang trống.", sid
        event = payload.get("event_name")
        commands = {"clear", "status", "local", "vision", "voice", "voice_status", "pomodoro", "eye", "timer_status", "timer_stop", "transcript", "forget_key"}
        if event == "message.text.received" and text.startswith("/"):
            parts = text[1:].split()
            if not parts or parts[0] not in commands:
                return "Lệnh chưa được hỗ trợ. Gõ /help hoặc nhắn yêu cầu bằng tiếng Việt.", sid
            name, args = parts[0], tuple(parts[1:])
            if name == "voice" and not args:
                args = ("local",)
            if name == "voice" and args[0] not in {"local", "cloud", "off"}:
                return "Cú pháp: /voice local|cloud|off", sid
            if name in {"eye", "transcript"} and (not args or args[0] not in {"on", "off"}):
                return f"Cú pháp: /{name} on|off", sid
            if name == "vision" and not args:
                return "Cú pháp: /vision MODEL hoặc /vision off", sid
            if name == "timer_stop" and not args:
                args = ("all",)
            result = await core.handle_command(sid, name, args, channel="zalo")
        elif event == "message.image.received":
            data = await self.download(str(payload.get("photo") or ""), 10 * 1024 * 1024)
            result = await core.process_image(sid, data, "zalo.jpg", "image/jpeg", payload.get("caption", ""), channel="zalo")
        elif event == "message.voice.received":
            data = await self.download(str(payload.get("voice_url") or ""), core.MAX_AUDIO_BYTES)
            result = await core.process_voice(sid, data, "zalo.ogg", "audio/ogg", channel="zalo")
        elif event == "message.text.received" and text:
            result = await core.handle_text(sid, text, channel="zalo")
        else:
            return "Hiện bot hỗ trợ văn bản, ảnh và tin nhắn thoại.", sid
        return str(result["reply"]), sid

    async def notify(self, sid: str, event: dict):
        if not self.enabled or not sid.startswith("acct-"):
            return
        if self.owner_id is not None and sid != f"acct-{self.owner_id}":
            return
        link = self.store.zalo_link(int(sid[5:]))
        if not link:
            return
        if self.bot_id:
            if not link['external_id'].startswith(self.bot_id + '/'):
                return
        elif '/' in link['external_id']:
            return
        if event.get("type") == "reminder":
            text = str(event["text"])
        elif self.mirror and event.get("type") == "conversation" and event.get("source") == "web":
            text = f"Trên web, bạn: {event.get('user_text', '')}\n\nTrợ lý: {event.get('reply', '')}"
        else:
            return
        payload = {"outgoing": True, "external_id": link["external_id"], "chat_id": link["chat_id"]}
        event_id = "out-" + secrets.token_hex(16)
        self.store.enqueue_zalo_event(event_id, payload, sid, bot_id=self.bot_id)
        self.store.save_zalo_reply(event_id, text, sid)
        self.wakeup.set()

    async def process_next(self) -> bool:
        row = self.store.next_zalo_event(bot_id=self.bot_id)
        if row is None:
            return False
        payload = json.loads(row["payload"])
        if row["reply"] is None:
            try:
                reply, sid = await self.handle(payload)
            except (ValueError, RuntimeError, httpx.HTTPError) as exc:
                reply, sid = "Mình chưa xử lý được tin nhắn này. Bạn thử lại hoặc kiểm tra cấu hình trên web nhé.", ""
                logger.warning("Zalo message processing failed (%s)", type(exc).__name__)
            self.store.save_zalo_reply(row["event_id"], reply, sid)
        else:
            reply, sid = row["reply"], row["sid"]
        if sid and self.store.zalo_sid(payload["external_id"]) != sid:
            self.store.finish_zalo_event(row["event_id"])
            return True
        for part in chunks(reply):
            await self.api.call("sendMessage", {"chat_id": payload["chat_id"], "text": part})
        if sid and payload.get("event_name") == "message.voice.received" and self.public_url.startswith("https://") and core.voice_enabled(sid, "tts"):
            try:
                await self.send_voice(payload["chat_id"], sid, reply)
            except Exception as exc:
                logger.warning("Zalo voice delivery unavailable (%s); text was delivered", type(exc).__name__)
        self.store.finish_zalo_event(row["event_id"])
        return True

    async def worker(self):
        while True:
            try:
                if await self.process_next():
                    continue
                self.wakeup.clear()
                try:
                    await asyncio.wait_for(self.wakeup.wait(), 2)
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Zalo delivery deferred (%s)", type(exc).__name__)
                await asyncio.sleep(5)

    async def send_voice(self, chat_id: str, sid: str, text: str):
        audio = await core.synthesize_speech(sid, text)
        def convert():
            import imageio_ffmpeg
            result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-vn", "-c:a", "aac", "-ar", "44100", "-ac", "1", "-f", "adts", "pipe:1"], input=audio.data, capture_output=True, timeout=90, check=True)
            return result.stdout
        data = await asyncio.to_thread(convert)
        now = time.time()
        self.media = {key: item for key, item in self.media.items() if item[0] > now}
        while len(self.media) >= 32:
            self.media.pop(next(iter(self.media)))
        ticket = secrets.token_urlsafe(24)
        self.media[ticket] = (now + 300, data)
        await self.api.call("sendVoice", {"chat_id": chat_id, "voice_url": f"{self.public_url}/zalo/media/{ticket}.aac"})
