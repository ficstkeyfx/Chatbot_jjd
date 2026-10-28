"""Tải model dò khuôn mặt YuNet (OpenCV Zoo, ~230 KB, giấy phép MIT) về .models/yunet/.

Chạy một lần:  python setup_yunet.py
Sau đó fer_model.py tự dùng YuNet thay cho Haar (đặt FACE_DETECTOR=haar để quay lại Haar).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from fer_model import DEFAULT_YUNET_PATH

URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MIN_BYTES = 100_000  # tệp lỗi/HTML thường nhỏ hơn nhiều


def main(target: Path = DEFAULT_YUNET_PATH) -> int:
    if target.is_file() and target.stat().st_size >= MIN_BYTES:
        print(f"Đã có {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Đang tải {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    if len(data) < MIN_BYTES:
        print("Tệp tải về quá nhỏ, có thể bị lỗi.", file=sys.stderr)
        return 1
    target.write_bytes(data)
    try:
        import cv2

        cv2.FaceDetectorYN.create(str(target), "", (320, 320))
    except Exception as exc:  # noqa: BLE001
        target.unlink(missing_ok=True)
        print(f"Model tải về không dùng được với OpenCV hiện tại: {exc}", file=sys.stderr)
        return 1
    print(f"Đã lưu {target} ({len(data) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
