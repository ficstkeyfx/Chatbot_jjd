from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "scenario_id",
    "dataset_split",
    "expert_rewritten_user_message",
    "expert_approved_assistant_response",
    "naturalness_score",
    "reviewer_name",
    "review_date",
    "safety_review",
    "review_status",
}
VALID_SPLITS = {"train", "validation"}


class ReviewDataError(ValueError):
    pass


def _normalized(text: str) -> str:
    value = unicodedata.normalize("NFD", text.casefold())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", value).strip()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ReviewDataError(f"Thiếu cột bắt buộc: {', '.join(missing)}")
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]


def approved_examples(
    rows: Iterable[dict[str, str]], *, split: str
) -> list[dict[str, list[dict[str, str]]]]:
    examples: list[dict[str, list[dict[str, str]]]] = []
    seen_questions: set[str] = set()
    seen_ids: set[str] = set()

    for row_number, row in enumerate(rows, 2):
        case_id = row.get("scenario_id", "")
        if not case_id:
            raise ReviewDataError(f"Dòng {row_number}: thiếu scenario_id")
        if case_id in seen_ids:
            raise ReviewDataError(f"Dòng {row_number}: scenario_id trùng: {case_id}")
        seen_ids.add(case_id)

        row_split = row.get("dataset_split", "")
        if row_split not in VALID_SPLITS:
            raise ReviewDataError(
                f"Dòng {row_number} ({case_id}): dataset_split không hợp lệ: {row_split!r}"
            )
        if split != "all" and row_split != split:
            continue
        if row.get("review_status", "").casefold() != "approved":
            continue

        if row.get("safety_review", "").casefold() != "pass":
            raise ReviewDataError(
                f"Dòng {row_number} ({case_id}): ca approved phải có safety_review=pass"
            )
        question = row.get("expert_rewritten_user_message", "")
        answer = row.get("expert_approved_assistant_response", "")
        reviewer = row.get("reviewer_name", "")
        review_date = row.get("review_date", "")
        score = row.get("naturalness_score", "")
        if not question or not answer or not reviewer or not review_date:
            raise ReviewDataError(
                f"Dòng {row_number} ({case_id}): ca approved thiếu câu hỏi, câu trả lời, "
                "người duyệt hoặc ngày duyệt"
            )
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", review_date):
            raise ReviewDataError(
                f"Dòng {row_number} ({case_id}): review_date phải theo YYYY-MM-DD"
            )
        if score not in {"1", "2", "3", "4", "5"}:
            raise ReviewDataError(
                f"Dòng {row_number} ({case_id}): naturalness_score phải từ 1 đến 5"
            )

        normalized_question = _normalized(question)
        if normalized_question in seen_questions:
            raise ReviewDataError(
                f"Dòng {row_number} ({case_id}): câu hỏi chuyên gia bị trùng"
            )
        seen_questions.add(normalized_question)
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    return examples


def write_jsonl(path: Path, examples: Iterable[dict]) -> int:
    materialized = list(examples)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in materialized:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(materialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Xuất các mẫu đã được chuyên gia duyệt sang JSONL hội thoại."
    )
    parser.add_argument("input", type=Path, help="CSV UTF-8 do chuyên gia cập nhật")
    parser.add_argument("output", type=Path, help="Tệp JSONL đầu ra")
    parser.add_argument(
        "--split", choices=("train", "validation", "all"), default="train"
    )
    parser.add_argument("--min-approved", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        examples = approved_examples(load_rows(args.input), split=args.split)
        if len(examples) < args.min_approved:
            raise ReviewDataError(
                f"Chỉ có {len(examples)} mẫu đã duyệt cho split={args.split}; "
                f"cần ít nhất {args.min_approved}."
            )
        count = write_jsonl(args.output, examples)
    except (OSError, ReviewDataError) as exc:
        print(f"Không thể xuất dữ liệu: {exc}")
        return 2
    print(f"Đã xuất {count} mẫu vào {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
