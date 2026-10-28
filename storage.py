"""Lưu trữ bền vững bằng SQLite: hội thoại, ngữ cảnh LLM, cài đặt và API key (mã hóa).

Nguyên tắc riêng tư:
- Lưu văn bản và ảnh đính kèm (ảnh mã hóa). Không lưu âm thanh.
- API key được mã hóa (Fernet); khóa mã hóa lấy từ STORAGE_SECRET hoặc tệp khóa cục bộ.
- Dữ liệu tự xóa sau RETENTION_DAYS ngày không hoạt động; người dùng có thể xóa ngay bất cứ lúc nào.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("health_bot.storage")

SCHEMA_VERSION = 5
DAY_SECONDS = 86400

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    sid TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    settings TEXT NOT NULL DEFAULT '{}',
    api_key_enc BLOB
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sid TEXT NOT NULL REFERENCES users(sid) ON DELETE CASCADE,
    role TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'text',
    text TEXT NOT NULL,
    alert TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_sid ON messages(sid, id);
CREATE TABLE IF NOT EXISTS context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sid TEXT NOT NULL REFERENCES users(sid) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_context_sid ON context(sid, id);
"""

# Phiên bản 2: tài khoản đăng nhập và phiên đăng nhập.
_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_login_at REAL
);
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash TEXT PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_account ON auth_sessions(account_id);
"""

_SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS zalo_links (
    external_id TEXT PRIMARY KEY,
    account_id INTEGER NOT NULL UNIQUE REFERENCES accounts(id) ON DELETE CASCADE,
    chat_id TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS zalo_link_codes (
    code_hash TEXT PRIMARY KEY,
    account_id INTEGER NOT NULL UNIQUE REFERENCES accounts(id) ON DELETE CASCADE,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS zalo_events (
    event_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    reply TEXT,
    sid TEXT NOT NULL DEFAULT '',
    done INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
"""


class StorageError(RuntimeError):
    pass


def _load_fernet(key_file: Path) -> Fernet:
    secret = os.getenv("STORAGE_SECRET", "").strip()
    if secret:
        try:
            return Fernet(secret.encode())
        except ValueError as exc:
            raise StorageError(
                "STORAGE_SECRET không hợp lệ. Tạo khóa mới bằng: "
                'python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
            ) from exc
    if key_file.exists():
        return Fernet(key_file.read_bytes().strip())
    key = Fernet.generate_key()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    # Tạo với quyền chỉ chủ sở hữu đọc/ghi (có hiệu lực trên Linux/macOS).
    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key)
    logger.info("Đã tạo khóa mã hóa mới tại %s (đừng đưa tệp này lên git)", key_file)
    return Fernet(key)


class Store:
    """Lớp truy cập SQLite an toàn khi gọi từ nhiều luồng."""

    def __init__(self, path: str | Path, *, key_file: str | Path | None = None) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        if str(path) != ":memory:":
            self._db.execute("PRAGMA journal_mode = WAL")
        if str(path) == ":memory:" and not os.getenv("STORAGE_SECRET"):
            self._fernet = Fernet(Fernet.generate_key())  # chỉ dùng trong test
        else:
            self._fernet = _load_fernet(
                Path(key_file) if key_file else self.path.with_name(".storage.key")
            )
        self._migrate()

    # --- hạ tầng ---

    def _migrate(self) -> None:
        with self._lock:
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise StorageError("Cơ sở dữ liệu được tạo bởi phiên bản mới hơn của ứng dụng.")
            if version < SCHEMA_VERSION:
                self._db.executescript(_SCHEMA)  # CREATE ... IF NOT EXISTS: an toàn khi nâng cấp từ v1
                self._db.executescript(_SCHEMA_V2)
                self._db.executescript(_SCHEMA_V3)
                if version < 4:
                    self._db.executescript("""BEGIN IMMEDIATE;
                        ALTER TABLE zalo_events ADD COLUMN bot_id TEXT NOT NULL DEFAULT '';
                        ALTER TABLE zalo_link_codes ADD COLUMN bot_id TEXT NOT NULL DEFAULT '';
                        CREATE TABLE zalo_bots (account_id INTEGER PRIMARY KEY REFERENCES accounts(id) ON DELETE CASCADE, bot_id TEXT NOT NULL UNIQUE, bot_name TEXT NOT NULL, token_enc BLOB NOT NULL);
                        CREATE INDEX zalo_events_pending_bot ON zalo_events(bot_id, created_at) WHERE done = 0;
                        PRAGMA user_version = 4;
                        COMMIT;""")
                self._db.executescript("""
                    CREATE TABLE IF NOT EXISTS message_images (
                        message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
                        data_enc BLOB NOT NULL
                    );
                """)
                self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def create_zalo_code(self, account_id: int, ttl_seconds: int = 600, *, bot_id: str = "") -> dict[str, Any]:
        code = secrets.token_urlsafe(16)
        expires = time.time() + ttl_seconds
        with self._lock:
            self._db.execute("DELETE FROM zalo_link_codes WHERE account_id = ? OR expires_at < ?", (account_id, time.time()))
            self._db.execute("INSERT INTO zalo_link_codes(code_hash, account_id, expires_at, bot_id) VALUES (?, ?, ?, ?)", (self._token_hash(code), account_id, expires, bot_id))
        return {"code": code, "expires_at": int(expires * 1000)}

    def consume_zalo_code(self, code_hash: str, external_id: str, chat_id: str, *, bot_id: str = "", owner_id: int | None = None) -> str:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT account_id FROM zalo_link_codes WHERE code_hash = ? AND expires_at > ? AND bot_id = ?", (code_hash, time.time(), bot_id)).fetchone()
                if not row:
                    raise StorageError("Mã liên kết sai hoặc đã hết hạn. Hãy tạo mã mới trên web.")
                if owner_id is not None and row[0] != owner_id:
                    raise StorageError("Mã không thuộc tài khoản đã cấu hình bot này.")
                existing = self._db.execute("SELECT external_id, account_id FROM zalo_links WHERE external_id = ? OR account_id = ?", (external_id, row[0])).fetchall()
                if any(item["external_id"] != external_id or item["account_id"] != row[0] for item in existing):
                    raise StorageError("Tài khoản đã liên kết. Hãy hủy liên kết cũ trên web trước.")
                self._db.execute("INSERT OR REPLACE INTO zalo_links VALUES (?, ?, ?, ?)", (external_id, row[0], chat_id, time.time()))
                self._db.execute("DELETE FROM zalo_link_codes WHERE code_hash = ?", (code_hash,))
                self._db.execute("COMMIT")
                return f"acct-{row[0]}"
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def zalo_sid(self, external_id: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT account_id FROM zalo_links WHERE external_id = ?", (external_id,)).fetchone()
        return f"acct-{row[0]}" if row else None

    def zalo_link(self, account_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT external_id, chat_id, created_at FROM zalo_links WHERE account_id = ?", (account_id,)).fetchone()
        return dict(row) if row else None

    def unlink_zalo(self, account_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM zalo_links WHERE account_id = ?", (account_id,))
            self._db.execute("DELETE FROM zalo_link_codes WHERE account_id = ?", (account_id,))
            self._db.execute("DELETE FROM zalo_events WHERE sid = ? AND done = 0", (f"acct-{account_id}",))

    def enqueue_zalo_event(self, event_id: str, payload: dict[str, Any], sid: str = "", *, bot_id: str = "") -> bool:
        with self._lock:
            cursor = self._db.execute("INSERT OR IGNORE INTO zalo_events(event_id, payload, sid, created_at, bot_id) VALUES (?, ?, ?, ?, ?)", (event_id, json.dumps(payload, ensure_ascii=False), sid, time.time(), bot_id))
        return cursor.rowcount == 1

    def next_zalo_event(self, *, bot_id: str = "") -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM zalo_events WHERE done = 0 AND bot_id = ? ORDER BY created_at LIMIT 1", (bot_id,)).fetchone()
        return dict(row) if row else None

    def zalo_bot(self, account_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT account_id, bot_id, bot_name FROM zalo_bots WHERE account_id = ?", (account_id,)).fetchone()
        return dict(row) if row else None

    def list_zalo_bots(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM zalo_bots").fetchall()
        return [{"account_id": r['account_id'], "bot_id": r['bot_id'], "bot_name": r['bot_name'], "token": self._fernet.decrypt(bytes(r['token_enc'])).decode()} for r in rows]

    def save_zalo_bot(self, account_id: int, bot_id: str, bot_name: str, token: str) -> None:
        encrypted = self._fernet.encrypt(token.encode())
        with self._lock:
            existing = self._db.execute("SELECT account_id FROM zalo_bots WHERE bot_id = ?", (bot_id,)).fetchone()
            if existing and existing[0] != account_id:
                raise StorageError("Bot này đã thuộc một tài khoản khác.")
            old = self.zalo_bot(account_id)
            if not old or old['bot_id'] != bot_id:
                self.unlink_zalo(account_id)
                if old:
                    self._db.execute("DELETE FROM zalo_events WHERE bot_id = ?", (old['bot_id'],))
            self._db.execute("INSERT INTO zalo_bots VALUES (?, ?, ?, ?) ON CONFLICT(account_id) DO UPDATE SET bot_id=excluded.bot_id, bot_name=excluded.bot_name, token_enc=excluded.token_enc", (account_id, bot_id, bot_name, encrypted))

    def remove_zalo_bot(self, account_id: int) -> None:
        with self._lock:
            old = self.zalo_bot(account_id)
            if old:
                self.unlink_zalo(account_id)
                self._db.execute("DELETE FROM zalo_events WHERE bot_id = ?", (old['bot_id'],))
                self._db.execute("DELETE FROM zalo_bots WHERE account_id = ?", (account_id,))

    def save_zalo_reply(self, event_id: str, reply: str, sid: str = "") -> None:
        with self._lock:
            self._db.execute("UPDATE zalo_events SET reply = ?, sid = ? WHERE event_id = ?", (reply, sid, event_id))

    def finish_zalo_event(self, event_id: str) -> None:
        with self._lock:
            self._db.execute("UPDATE zalo_events SET done = 1, payload = '{}', reply = NULL WHERE event_id = ?", (event_id,))

    def _touch(self, sid: str) -> None:
        now = time.time()
        self._db.execute(
            "INSERT INTO users(sid, created_at, last_seen_at) VALUES (?, ?, ?) "
            "ON CONFLICT(sid) DO UPDATE SET last_seen_at = excluded.last_seen_at",
            (sid, now, now),
        )

    # --- người dùng và cài đặt ---

    def touch(self, sid: str) -> None:
        with self._lock:
            self._touch(sid)

    def load_settings(self, sid: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT settings FROM users WHERE sid = ?", (sid,)).fetchone()
        if not row:
            return {}
        try:
            value = json.loads(row["settings"])
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def save_settings(self, sid: str, settings: dict[str, Any]) -> None:
        with self._lock:
            self._touch(sid)
            self._db.execute(
                "UPDATE users SET settings = ? WHERE sid = ?",
                (json.dumps(settings, ensure_ascii=False), sid),
            )

    def set_api_keys(self, sid: str, keys: dict[str, str]) -> None:
        """Lưu toàn bộ key của người dùng (nhà cung cấp -> key), mã hóa nguyên khối."""
        encrypted = (
            self._fernet.encrypt(json.dumps(keys, ensure_ascii=False).encode("utf-8")) if keys else None
        )
        with self._lock:
            self._touch(sid)
            self._db.execute("UPDATE users SET api_key_enc = ? WHERE sid = ?", (encrypted, sid))

    def get_api_keys(self, sid: str) -> dict[str, str]:
        with self._lock:
            row = self._db.execute("SELECT api_key_enc FROM users WHERE sid = ?", (sid,)).fetchone()
        if not row or row["api_key_enc"] is None:
            return {}
        try:
            plain = self._fernet.decrypt(bytes(row["api_key_enc"])).decode("utf-8")
        except InvalidToken:
            # Khóa mã hóa đã đổi: coi như chưa có key thay vì làm hỏng phiên.
            logger.warning("Không giải mã được API key đã lưu (khóa mã hóa đã thay đổi)")
            return {}
        try:
            value = json.loads(plain)
        except json.JSONDecodeError:
            return {"openai": plain}  # định dạng cũ: một key OpenAI duy nhất
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items() if isinstance(v, str) and v}
        return {"openai": plain}

    # --- nhật ký hội thoại hiển thị trên giao diện ---

    def add_message(
        self, sid: str, role: str, text: str, *, kind: str = "text", alert: str | None = None,
        image: bytes | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        encrypted = self._fernet.encrypt(image) if image else None
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._touch(sid)
                cursor = self._db.execute(
                    "INSERT INTO messages(sid, role, kind, text, alert, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (sid, role, kind, text, alert, now),
                )
                if encrypted is not None:
                    self._db.execute("INSERT INTO message_images(message_id, data_enc) VALUES (?, ?)",
                                     (cursor.lastrowid, encrypted))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        return {
            "id": cursor.lastrowid, "role": role, "kind": kind, "text": text,
            "alert": alert, "ts": int(now * 1000),
            "image_url": f"/api/messages/{cursor.lastrowid}/image" if encrypted else None,
        }

    def list_messages(self, sid: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, role, kind, text, alert, created_at, "
                "EXISTS(SELECT 1 FROM message_images WHERE message_id=messages.id) AS has_image FROM messages "
                "WHERE sid = ? ORDER BY id DESC LIMIT ?",
                (sid, limit),
            ).fetchall()
        return [
            {
                "id": row["id"], "role": row["role"], "kind": row["kind"], "text": row["text"],
                "alert": row["alert"], "ts": int(row["created_at"] * 1000),
                "image_url": f"/api/messages/{row['id']}/image" if row["has_image"] else None,
            }
            for row in reversed(rows)
        ]

    def message_image(self, sid: str, message_id: int) -> bytes | None:
        with self._lock:
            row = self._db.execute(
                "SELECT data_enc FROM message_images JOIN messages ON messages.id=message_id "
                "WHERE messages.sid=? AND messages.id=?", (sid, message_id),
            ).fetchone()
        return self._fernet.decrypt(row[0]) if row else None

    # --- ngữ cảnh gửi cho LLM ---

    def load_context(self, sid: str, limit: int) -> list[dict[str, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM context WHERE sid = ? ORDER BY id DESC LIMIT ?",
                (sid, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def replace_context(self, sid: str, history: list[dict[str, str]]) -> None:
        with self._lock:
            self._touch(sid)
            self._db.execute("BEGIN")
            try:
                self._db.execute("DELETE FROM context WHERE sid = ?", (sid,))
                self._db.executemany(
                    "INSERT INTO context(sid, role, content) VALUES (?, ?, ?)",
                    [(sid, item["role"], item["content"]) for item in history],
                )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    # --- tài khoản và phiên đăng nhập ---

    def create_account(self, username: str, password_hash: str) -> int:
        try:
            with self._lock:
                cursor = self._db.execute(
                    "INSERT INTO accounts(username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, password_hash, time.time()),
                )
        except sqlite3.IntegrityError as exc:
            raise StorageError("Tên đăng nhập đã tồn tại.") from exc
        return int(cursor.lastrowid)

    def get_account(self, username: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, username, password_hash FROM accounts WHERE username = ?", (username,)
            ).fetchone()
        return dict(row) if row else None

    def get_account_by_id(self, account_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, username, password_hash FROM accounts WHERE id = ?", (account_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_accounts(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, username, created_at, last_login_at FROM accounts ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def count_accounts(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]

    def set_password_hash(self, account_id: int, password_hash: str) -> None:
        with self._lock:
            self._db.execute("UPDATE accounts SET password_hash = ? WHERE id = ?", (password_hash, account_id))

    def delete_account(self, account_id: int) -> None:
        with self._lock:
            self.remove_zalo_bot(account_id)
            self._db.execute("DELETE FROM zalo_events WHERE sid = ? OR json_extract(payload, '$.external_id') IN (SELECT external_id FROM zalo_links WHERE account_id = ?)", (f"acct-{account_id}", account_id))
            self._db.execute("DELETE FROM accounts WHERE id = ?", (account_id,))

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self, account_id: int, ttl_seconds: int) -> str:
        """Tạo phiên đăng nhập; trả về token (chỉ lưu bản băm trong DB)."""
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO auth_sessions(token_hash, account_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (self._token_hash(token), account_id, now, now + ttl_seconds),
            )
            self._db.execute("UPDATE accounts SET last_login_at = ? WHERE id = ?", (now, account_id))
        return token

    def get_session_account(self, token: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            row = self._db.execute(
                "SELECT a.id, a.username FROM auth_sessions s JOIN accounts a ON a.id = s.account_id "
                "WHERE s.token_hash = ? AND s.expires_at > ?",
                (self._token_hash(token), now),
            ).fetchone()
        return dict(row) if row else None

    def delete_session(self, token: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (self._token_hash(token),))

    def delete_other_sessions(self, account_id: int, keep_token: str | None) -> None:
        keep = self._token_hash(keep_token) if keep_token else ""
        with self._lock:
            self._db.execute(
                "DELETE FROM auth_sessions WHERE account_id = ? AND token_hash != ?", (account_id, keep)
            )

    # --- xóa dữ liệu ---

    def clear_conversation(self, sid: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM messages WHERE sid = ?", (sid,))
            self._db.execute("DELETE FROM context WHERE sid = ?", (sid,))

    def delete_user(self, sid: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM users WHERE sid = ?", (sid,))

    def purge(self, retention_days: int) -> int:
        """Xóa phiên ẩn danh không hoạt động quá retention_days và tin nhắn quá hạn; trả về số phiên bị xóa."""
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * DAY_SECONDS
        with self._lock:
            # Phiên ẩn danh hết hạn thì xóa hẳn; tài khoản đăng nhập giữ lại cài đặt và API key
            # (chỉ tin nhắn cũ bị xóa ở bên dưới).
            removed = self._db.execute(
                "DELETE FROM users WHERE last_seen_at < ? AND sid NOT LIKE 'acct-%'", (cutoff,)
            ).rowcount
            self._db.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (time.time(),))
            # Tin nhắn cũ của người dùng vẫn hoạt động cũng hết hạn theo cùng thời hạn.
            self._db.execute("DELETE FROM messages WHERE created_at < ?", (cutoff,))
            self._db.execute("DELETE FROM zalo_link_codes WHERE expires_at < ?", (time.time(),))
            self._db.execute("DELETE FROM zalo_events WHERE created_at < ?", (cutoff,))
        return removed

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                name: self._db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                for name in ("users", "messages", "context")
            }
