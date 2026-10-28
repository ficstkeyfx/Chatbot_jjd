"""Quản lý tài khoản từ dòng lệnh (dùng khi đóng đăng ký hoặc để tạo tài khoản quản trị ban đầu).

    python manage.py create-user TÊN            # hỏi mật khẩu (không hiện trên màn hình)
    python manage.py reset-password TÊN
    python manage.py delete-user TÊN            # xóa tài khoản và toàn bộ dữ liệu của họ
    python manage.py list-users
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

import auth
from storage import Store, StorageError

DB_PATH = os.getenv("DB_PATH", "data/healthbot.db").strip()


def read_password(args: argparse.Namespace, username: str) -> str:
    if args.password_env:
        password = os.getenv(args.password_env, "")
        if not password:
            raise SystemExit(f"Biến môi trường {args.password_env} đang trống.")
    else:
        password = getpass.getpass("Mật khẩu: ")
        if password != getpass.getpass("Nhập lại mật khẩu: "):
            raise SystemExit("Hai lần nhập mật khẩu không khớp.")
    return auth.validate_password(password, username)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("create-user", "reset-password", "delete-user"):
        command = sub.add_parser(name)
        command.add_argument("username")
        if name != "delete-user":
            command.add_argument("--password-env", help="Đọc mật khẩu từ biến môi trường này (thay vì hỏi)")
    sub.add_parser("list-users")
    args = parser.parse_args(argv)

    if not DB_PATH:
        raise SystemExit("DB_PATH đang để trống nên không có cơ sở dữ liệu để quản lý.")
    store = Store(DB_PATH)
    try:
        if args.command == "list-users":
            for row in store.list_accounts():
                print(f"{row['id']:>4}  {row['username']}")
            return 0
        username = auth.validate_username(args.username)
        if args.command == "create-user":
            store.create_account(username, auth.hash_password(read_password(args, username)))
            print(f"Đã tạo tài khoản {username}.")
        elif args.command == "reset-password":
            account = store.get_account(username)
            if not account:
                raise SystemExit("Không có tài khoản này.")
            store.set_password_hash(account["id"], auth.hash_password(read_password(args, username)))
            store.delete_other_sessions(account["id"], None)  # đăng xuất mọi thiết bị
            print(f"Đã đổi mật khẩu của {username} và đăng xuất mọi thiết bị.")
        else:
            account = store.get_account(username)
            if not account:
                raise SystemExit("Không có tài khoản này.")
            store.delete_user(f"acct-{account['id']}")
            store.delete_account(account["id"])
            print(f"Đã xóa tài khoản {username} và dữ liệu của họ.")
        return 0
    except (auth.AuthError, StorageError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
