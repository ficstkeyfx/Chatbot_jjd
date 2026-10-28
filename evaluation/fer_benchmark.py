"""Đo chất lượng nhận dạng nét mặt bằng ảnh có nhãn và gợi ý ngưỡng tin cậy.

Cấu trúc thư mục dữ liệu (giống FER2013 dạng ảnh):

    DATASET/
      angry/*.png   disgust/*.png   fear/*.png   happy/*.png
      sad/*.png     surprise/*.png  neutral/*.png

Chạy:
    python -m evaluation.fer_benchmark DATASET --mode cropped --sweep --json ket_qua.json

--mode cropped : ảnh đã là khuôn mặt (như FER2013): đo riêng mô hình CNN.
--mode full    : ảnh chụp thật, đi qua cả bộ dò khuôn mặt (cần OpenCV): đo cả đường ống như khi chạy thật.
                 Dùng --detector haar / yunet để so sánh hai bộ dò mặt trên cùng tập ảnh.

Ảnh chỉ được xử lý trên máy; không gửi đi đâu.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from fer_model import (
    DEFAULT_MODEL_PATH, LABELS, MIN_CONFIDENCE, MIN_MARGIN, crop_face, decide_label, detect_faces,
    predict_probabilities, resolve_detector,
)

ABSTAIN = "unclear"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_samples(directory: Path) -> list[tuple[str, Path]]:
    samples: list[tuple[str, Path]] = []
    for label in LABELS:
        folder = directory / label
        if not folder.is_dir():
            continue
        samples.extend(
            (label, path) for path in sorted(folder.rglob("*")) if path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not samples:
        raise SystemExit(f"Không tìm thấy ảnh nào trong {directory} (cần thư mục con theo nhãn: {', '.join(LABELS)}).")
    return samples


def probabilities_for(path: Path, mode: str, model_path: Path, detector: str | None = None):
    """Trả về mảng xác suất, hoặc None nếu đường ống từ chối (không/nhiều khuôn mặt)."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if mode == "cropped":
        return predict_probabilities(image, model_path)
    faces = detect_faces(np.asarray(image), detector)
    if len(faces) != 1:
        return None
    return predict_probabilities(crop_face(image, faces[0]), model_path)


def metrics(records: list[tuple[str, np.ndarray | None]], min_confidence: float, min_margin: float) -> dict:
    """Tính các chỉ số cho một cặp ngưỡng. records: (nhãn thật, xác suất hoặc None nếu bị từ chối sớm)."""
    predictions = [
        ABSTAIN if probabilities is None else decide_label(probabilities, min_confidence, min_margin)
        for _, probabilities in records
    ]
    truths = [truth for truth, _ in records]
    total = len(records)
    answered = [(t, p) for t, p in zip(truths, predictions) if p != ABSTAIN]
    correct = sum(t == p for t, p in answered)

    per_class = {}
    f1_scores = []
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(truths, predictions))
        fp = sum(t != label and p == label for t, p in zip(truths, predictions))
        fn = sum(t == label and p != label for t, p in zip(truths, predictions))
        support = sum(t == label for t in truths)
        if not support:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_scores.append(f1)

    columns = [*LABELS, ABSTAIN]
    confusion = {
        truth: {pred: sum(t == truth and p == pred for t, p in zip(truths, predictions)) for pred in columns}
        for truth in LABELS
        if truth in truths
    }
    return {
        "min_confidence": min_confidence,
        "min_margin": min_margin,
        "total": total,
        "coverage": len(answered) / total if total else 0.0,
        "accuracy_overall": correct / total if total else 0.0,  # từ chối tính là sai
        "accuracy_answered": correct / len(answered) if answered else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "per_class": per_class,
        "confusion": confusion,
    }


def sweep(records, min_coverage: float) -> tuple[list[dict], dict | None]:
    grid = [
        metrics(records, confidence, margin)
        for confidence, margin in itertools.product(
            (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (0.0, 0.05, 0.12, 0.2, 0.3)
        )
    ]
    eligible = [m for m in grid if m["coverage"] >= min_coverage]
    best = max(eligible, key=lambda m: (m["accuracy_answered"], m["coverage"]), default=None)
    return grid, best


def print_report(result: dict, current: dict, best: dict | None, grid: list[dict] | None) -> None:
    out = sys.stdout
    if hasattr(out, "reconfigure"):
        out.reconfigure(encoding="utf-8")
    print(f"Số ảnh: {current['total']}")
    print(
        f"Ngưỡng hiện tại (tin cậy ≥ {current['min_confidence']}, chênh ≥ {current['min_margin']}): "
        f"độ phủ {current['coverage']:.1%} | chính xác trên ảnh được trả lời {current['accuracy_answered']:.1%} | "
        f"chính xác tổng (từ chối = sai) {current['accuracy_overall']:.1%} | macro-F1 {current['macro_f1']:.3f}"
    )
    print("\nTheo nhãn (precision / recall / F1 / số ảnh):")
    for label, row in current["per_class"].items():
        print(f"  {label:9s} {row['precision']:.2f} / {row['recall']:.2f} / {row['f1']:.2f} / {row['support']}")
    print("\nMa trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):")
    columns = [*LABELS, ABSTAIN]
    print("  " + " " * 9 + " ".join(f"{c[:5]:>5s}" for c in columns))
    for truth, row in current["confusion"].items():
        print(f"  {truth:9s}" + " ".join(f"{row[c]:5d}" for c in columns))
    if grid is not None:
        print("\nĐánh đổi giữa độ phủ và độ chính xác (một số ngưỡng):")
        for m in sorted(grid, key=lambda m: -m["coverage"])[::4]:
            print(
                f"  tin cậy ≥ {m['min_confidence']:.2f}, chênh ≥ {m['min_margin']:.2f}: "
                f"phủ {m['coverage']:.0%}, chính xác {m['accuracy_answered']:.1%}, F1 {m['macro_f1']:.3f}"
            )
        if best:
            print(
                f"\nGợi ý (độ phủ tối thiểu theo --min-coverage): MIN_CONFIDENCE={best['min_confidence']}, "
                f"MIN_MARGIN={best['min_margin']} -> chính xác {best['accuracy_answered']:.1%}, phủ {best['coverage']:.0%}"
            )
        else:
            print("\nKhông có ngưỡng nào đạt độ phủ tối thiểu.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--mode", choices=("cropped", "full"), default="cropped")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--detector", choices=("auto", "yunet", "haar"), default="auto",
        help="Bộ dò mặt cho --mode full (so sánh yunet với haar)",
    )
    parser.add_argument("--sweep", action="store_true", help="Quét ngưỡng tin cậy để tìm điểm cân bằng")
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument("--json", type=Path, help="Ghi kết quả đầy đủ ra tệp JSON")
    args = parser.parse_args()

    samples = load_samples(args.dataset)
    detector = None if args.detector == "auto" else args.detector
    if args.mode == "full":
        print(f"Bộ dò mặt: {resolve_detector(detector)}")
    records = [(label, probabilities_for(path, args.mode, args.model, detector)) for label, path in samples]
    current = metrics(records, MIN_CONFIDENCE, MIN_MARGIN)
    grid, best = sweep(records, args.min_coverage) if args.sweep else (None, None)
    print_report({}, current, best, grid)
    if args.json:
        args.json.write_text(
            json.dumps({"current": current, "sweep": grid, "best": best}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
