"""Kiểm toán bộ dữ liệu fine-tune trước khi gửi chuyên gia duyệt hoặc huấn luyện.

Chạy: python -m fine_tuning.audit_dataset [--json báo_cáo.json]

Script chỉ đọc dữ liệu; không gọi LLM. Các kiểm tra mang tính sàng lọc, không thay thế chuyên gia.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from safety import classify_crisis, classify_eye_red_flags

ROOT = Path(__file__).resolve().parent
SCENARIOS = ROOT / "scenarios.jsonl"
RESPONSES = ROOT / "initial_responses.jsonl"
REVIEW = ROOT / "expert_review_template.csv"

EMERGENCY_TERMS = ("115", "cap cuu", "co so y te", "benh vien", "khoa mat", "di kham ngay", "tim tro giup")
# Cụm từ cho thấy câu trả lời khẳng định chẩn đoán hoặc kê đơn (không được phép).
DIAGNOSIS_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\bban (bi|mac) (benh|viem|ung thu|tram cam|roi loan)\b",
        r"\bchac chan (la|bi)\b",
        r"\bchan doan (la|chinh xac)\b",
        r"\bday la (benh|bieu hien cua benh)\b",
    )
)
DOSAGE_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:mg|ml|vien|goi)\b")
PII_PATTERNS = {
    "so_dien_thoai": re.compile(r"\b0\d{9,10}\b"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
}
MAX_WORDS = 140
MAX_BULLETS = 5


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", value).strip()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize(text)))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def audit() -> dict:
    scenarios = read_jsonl(SCENARIOS)
    responses = {row["scenario_id"]: row["initial_chatbot_response"] for row in read_jsonl(RESPONSES)}
    findings: dict[str, list[dict]] = defaultdict(list)

    def flag(kind: str, scenario: dict, detail: str) -> None:
        findings[kind].append({"scenario_id": scenario["scenario_id"], "detail": detail})

    for scenario in scenarios:
        text = responses.get(scenario["scenario_id"], "")
        norm = normalize(text)
        if not text.strip():
            flag("thieu_tra_loi", scenario, "Không có câu trả lời mẫu")
            continue

        # must_include/must_avoid là hướng dẫn cho chuyên gia duyệt (diễn đạt tự do, không so khớp
        # máy móc được) nên không kiểm tra tự động ở đây.
        for pattern in DIAGNOSIS_PATTERNS:
            if pattern.search(norm):
                flag("khang_dinh_chan_doan", scenario, f"Khớp mẫu: {pattern.pattern}")
        if scenario["category"] == "medication_chronic" and DOSAGE_PATTERN.search(norm):
            flag("lieu_thuoc_cu_the", scenario, "Có liều thuốc cụ thể; cần chuyên gia xác nhận")

        words = len(text.split())
        if words > MAX_WORDS:
            flag("qua_dai", scenario, f"{words} từ (giới hạn gợi ý {MAX_WORDS})")
        bullets = len(re.findall(r"^\s*[-*•]\s+", text, re.M))
        if bullets > MAX_BULLETS:
            flag("qua_nhieu_gach_dau_dong", scenario, f"{bullets} gạch đầu dòng")
        if re.search(r"[*#`]{2,}|^#{1,6}\s", text, re.M):
            flag("con_markdown", scenario, "Còn ký hiệu Markdown")

        if scenario["risk_level"] == "immediate_risk" or scenario["category"] == "urgent_physical":
            if not any(term in norm for term in EMERGENCY_TERMS):
                flag("thieu_huong_dan_khan_cap", scenario, "Tình huống nguy cấp nhưng không hướng dẫn cấp cứu/y tế")

        for name, pattern in PII_PATTERNS.items():
            if pattern.search(scenario["draft_user_message"] + " " + text):
                flag("co_the_chua_pii", scenario, name)

    # Lớp an toàn có nhận ra các tình huống đã gán nhãn nguy cơ không?
    safety_matrix: Counter = Counter()
    for scenario in scenarios:
        message = scenario["draft_user_message"]
        predicted = classify_crisis(message)
        if classify_eye_red_flags(message) and predicted == "normal":
            predicted = "eye_red_flag"
        safety_matrix[(scenario["risk_level"], predicted)] += 1
        if scenario["risk_level"] == "immediate_risk" and predicted != "immediate_risk":
            flag("an_toan_bo_sot", scenario, f"Gán nhãn immediate_risk nhưng safety.py trả '{predicted}'")
        if scenario["risk_level"] == "normal" and predicted == "immediate_risk":
            flag("an_toan_bao_nham", scenario, "Gán nhãn normal nhưng safety.py báo immediate_risk")

    # Trùng lặp: gần giống nhau trong cùng tập hoặc rò rỉ giữa train và validation.
    token_sets = [(s, tokens(s["draft_user_message"])) for s in scenarios]
    for i, (a, ta) in enumerate(token_sets):
        for b, tb in token_sets[i + 1:]:
            similarity = jaccard(ta, tb)
            if similarity >= 0.8:
                kind = "ro_ri_train_validation" if a["dataset_split"] != b["dataset_split"] else "trung_lap"
                flag(kind, a, f"Giống {b['scenario_id']} ({similarity:.0%})")

    review_status: Counter = Counter()
    if REVIEW.exists():
        with REVIEW.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                review_status[row.get("review_status") or "(trống)"] += 1

    return {
        "so_kich_ban": len(scenarios),
        "phan_bo": {
            key: dict(Counter(s[key] for s in scenarios))
            for key in ("category", "risk_level", "modality", "dataset_split")
        },
        "trang_thai_duyet": dict(review_status),
        "ma_tran_an_toan": {f"{label} -> {pred}": n for (label, pred), n in sorted(safety_matrix.items())},
        "phat_hien": {kind: items for kind, items in sorted(findings.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Ghi báo cáo đầy đủ ra tệp JSON")
    args = parser.parse_args()
    report = audit()
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out = sys.stdout
    if hasattr(out, "reconfigure"):
        out.reconfigure(encoding="utf-8")
    print(f"Số kịch bản: {report['so_kich_ban']}")
    for key, value in report["phan_bo"].items():
        print(f"  {key}: {value}")
    print(f"Trạng thái duyệt chuyên gia: {report['trang_thai_duyet']}")
    print("Lớp an toàn (nhãn -> dự đoán):")
    for key, count in report["ma_tran_an_toan"].items():
        print(f"  {key}: {count}")
    print("Phát hiện:")
    if not report["phat_hien"]:
        print("  (không có)")
    for kind, items in report["phat_hien"].items():
        print(f"  [{kind}] {len(items)} mục")
        for item in items[:8]:
            print(f"    - {item['scenario_id']}: {item['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
