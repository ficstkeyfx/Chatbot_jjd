"""Generate a transparent, source-aware benchmark chart for facial expression recognition."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "evaluation" / "results" / "facial_expression_benchmark.csv"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "facial_expression_benchmark.png"

WIDTH, HEIGHT = 1800, 1200
COLORS = {
    "background": "#F3F7FB",
    "panel": "#FFFFFF",
    "ink": "#10243E",
    "muted": "#60738A",
    "grid": "#D8E2EC",
    "blue": "#2474E5",
    "blue_light": "#DCEBFF",
    "green": "#1B9A73",
    "green_light": "#DDF5EC",
    "orange": "#EF8F35",
    "orange_light": "#FDEAD7",
    "red": "#D85656",
    "purple": "#7857C5",
}


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


FONTS = {
    "title": font(38, bold=True),
    "subtitle": font(21),
    "panel": font(25, bold=True),
    "card_value": font(38, bold=True),
    "card_label": font(18),
    "body": font(18),
    "body_bold": font(18, bold=True),
    "small": font(15),
    "small_bold": font(15, bold=True),
}


def text_width(draw: ImageDraw.ImageDraw, text: str, selected_font) -> int:
    box = draw.textbbox((0, 0), text, font=selected_font)
    return box[2] - box[0]


def wrap(draw: ImageDraw.ImageDraw, text: str, selected_font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or text_width(draw, candidate, selected_font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def rounded_panel(draw: ImageDraw.ImageDraw, box, *, fill=None, outline=None, radius=24):
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill or COLORS["panel"],
        outline=outline or COLORS["grid"],
        width=2,
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def draw_card(draw: ImageDraw.ImageDraw, x: int, value: str, label: str, color: str):
    rounded_panel(draw, (x, 132, x + 390, 270), fill="#FFFFFF", outline="#E0E8F0")
    draw.rounded_rectangle((x + 20, 152, x + 29, 250), radius=5, fill=color)
    draw.text((x + 50, 151), value, font=FONTS["card_value"], fill=color)
    lines = wrap(draw, label, FONTS["card_label"], 310)
    for index, line in enumerate(lines[:2]):
        draw.text((x + 51, 207 + index * 23), line, font=FONTS["card_label"], fill=COLORS["muted"])


def draw_published_panel(draw: ImageDraw.ImageDraw, rows: list[dict[str, str]]):
    x, y, w, h = 70, 305, 1030, 535
    rounded_panel(draw, (x, y, x + w, y + h))
    draw.text((x + 30, y + 24), "1. Độ chính xác công bố / tham chiếu", font=FONTS["panel"], fill=COLORS["ink"])
    draw.text(
        (x + 30, y + 61),
        "Top-1 accuracy; các thanh không hoàn toàn cùng tập kiểm thử",
        font=FONTS["small"],
        fill=COLORS["muted"],
    )

    chart_left, chart_right = x + 300, x + w - 38
    chart_top, chart_bottom = y + 120, y + h - 58
    minimum, maximum = 55.0, 80.0

    band_left = chart_left + int((60 - minimum) / (maximum - minimum) * (chart_right - chart_left))
    band_right = chart_left + int((70 - minimum) / (maximum - minimum) * (chart_right - chart_left))
    draw.rectangle((band_left, chart_top - 10, band_right, chart_bottom + 10), fill="#F4F7FA")
    draw.text((band_left + 6, chart_top - 31), "Human ~65±5%", font=FONTS["small"], fill=COLORS["muted"])

    for tick in range(55, 81, 5):
        tx = chart_left + int((tick - minimum) / (maximum - minimum) * (chart_right - chart_left))
        draw.line((tx, chart_top, tx, chart_bottom), fill=COLORS["grid"], width=1)
        draw.text((tx - 12, chart_bottom + 17), str(tick), font=FONTS["small"], fill=COLORS["muted"])

    row_height = 68
    for index, row in enumerate(rows):
        center_y = chart_top + 28 + index * row_height
        value = float(row["value"])
        bar_end = chart_left + int((value - minimum) / (maximum - minimum) * (chart_right - chart_left))
        is_deployed = "đang dùng" in row["metric"]
        color = COLORS["blue"] if is_deployed else COLORS["purple"]
        draw.text((x + 30, center_y - 19), row["metric"], font=FONTS["body_bold"] if is_deployed else FONTS["body"], fill=COLORS["ink"])
        draw.rounded_rectangle((chart_left, center_y - 15, chart_right, center_y + 15), radius=15, fill="#EEF3F8")
        draw.rounded_rectangle((chart_left, center_y - 15, bar_end, center_y + 15), radius=15, fill=color)
        draw.text((bar_end + 9, center_y - 14), f"{value:.1f}%".replace(".", ","), font=FONTS["body_bold"], fill=color)

    draw.text(
        (x + 30, y + h - 34),
        "Web-app 69,8% = số liệu công bố của trọng số đang được chuyển sang ONNX trong project.",
        font=FONTS["small"],
        fill=COLORS["blue"],
    )


def draw_internal_panel(draw: ImageDraw.ImageDraw, internal: list[dict[str, str]], distribution: list[dict[str, str]]):
    x, y, w, h = 1130, 305, 600, 535
    rounded_panel(draw, (x, y, x + w, y + h))
    draw.text((x + 28, y + 24), "2. Kiểm tra nội bộ", font=FONTS["panel"], fill=COLORS["ink"])
    draw.text((x + 28, y + 61), "15 ảnh thực tế · không có nhãn cảm xúc chuẩn", font=FONTS["small"], fill=COLORS["muted"])

    measured = [row for row in internal if row["status"] == "measured"]
    start_y = y + 108
    for index, row in enumerate(measured):
        top = start_y + index * 62
        value = float(row["value"])
        ratio = value / 100.0
        draw.text((x + 28, top), row["metric"], font=FONTS["body"], fill=COLORS["ink"])
        draw.rounded_rectangle((x + 300, top + 2, x + 510, top + 24), radius=11, fill="#EAF0F5")
        draw.rounded_rectangle((x + 300, top + 2, x + 300 + int(210 * ratio), top + 24), radius=11, fill=COLORS["green"])
        draw.text((x + 520, top - 1), f'{row["numerator"]}/{row["denominator"]}', font=FONTS["body_bold"], fill=COLORS["green"])

    badge_y = y + 300
    draw.rounded_rectangle((x + 28, badge_y, x + w - 28, badge_y + 66), radius=16, fill=COLORS["orange_light"])
    draw.text((x + 45, badge_y + 10), "Accuracy biểu cảm nội bộ: N/A", font=FONTS["body_bold"], fill=COLORS["orange"])
    draw.text((x + 45, badge_y + 37), "Cần ground truth độc lập để tính đúng/sai.", font=FONTS["small"], fill=COLORS["ink"])

    draw.text((x + 28, y + 390), "Phân bố dự đoán trên 11 ảnh một mặt", font=FONTS["body_bold"], fill=COLORS["ink"])
    total = sum(int(float(row["value"])) for row in distribution)
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    bar_x, bar_y, bar_w = x + 28, y + 429, w - 56
    cursor = bar_x
    for row, color in zip(distribution, colors):
        count = int(float(row["value"]))
        segment = round(bar_w * count / total)
        draw.rectangle((cursor, bar_y, cursor + segment, bar_y + 25), fill=color)
        cursor += segment
    legend_y = bar_y + 40
    cursor_x = x + 28
    for row, color in zip(distribution, colors):
        count = int(float(row["value"]))
        label = f'{row["metric"]} {count} ({count / total:.0%})'
        draw.ellipse((cursor_x, legend_y + 3, cursor_x + 14, legend_y + 17), fill=color)
        draw.text((cursor_x + 20, legend_y), label, font=FONTS["small"], fill=COLORS["ink"])
        cursor_x += 175


def draw_config_panel(draw: ImageDraw.ImageDraw):
    x, y, w, h = 70, 870, 1660, 255
    rounded_panel(draw, (x, y, x + w, y + h))
    draw.text((x + 30, y + 23), "3. Tiêu chí và cấu hình benchmark", font=FONTS["panel"], fill=COLORS["ink"])
    left = [
        ("Đầu vào", "crop khuôn mặt 48×48, ảnh xám"),
        ("Số lớp", "7: angry, disgust, fear, happy, sad, surprise, neutral"),
        ("TTA", "2 lượt: ảnh gốc + lật ngang, lấy trung bình xác suất"),
    ]
    right = [
        ("Ngưỡng tin cậy", "xác suất cao nhất ≥ 0,50"),
        ("Ngưỡng cách biệt", "top-1 − top-2 ≥ 0,12"),
        ("Quy tắc từ chối", "khác đúng 1 mặt hoặc dưới ngưỡng → unclear"),
    ]
    for column, rows in enumerate((left, right)):
        base_x = x + 30 + column * 810
        for index, (key, value) in enumerate(rows):
            row_y = y + 74 + index * 48
            draw.text((base_x, row_y), key, font=FONTS["small_bold"], fill=COLORS["blue"])
            lines = wrap(draw, value, FONTS["small"], 590)
            draw.text((base_x + 180, row_y), lines[0], font=FONTS["small"], fill=COLORS["ink"])

    footer = (
        "Định nghĩa: accuracy = số dự đoán đúng / tổng mẫu có ground truth. "
        "Tỷ lệ trả kết quả hợp lệ và tỷ lệ abstain không phải accuracy biểu cảm."
    )
    draw.text((x + 30, y + h - 35), footer, font=FONTS["small"], fill=COLORS["muted"])


def generate(data_path: Path, output_path: Path) -> None:
    rows = load_rows(data_path)
    published = [row for row in rows if row["section"] == "published"]
    internal = [row for row in rows if row["section"] == "internal"]
    distribution = [row for row in rows if row["section"] == "distribution"]

    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)
    draw.text((70, 36), "BENCHMARK NHẬN DIỆN BIỂU CẢM KHUÔN MẶT", font=FONTS["title"], fill=COLORS["ink"])
    draw.text(
        (70, 86),
        "Phân biệt rõ độ chính xác công bố, độ ổn định pipeline và phần chưa được đo",
        font=FONTS["subtitle"],
        fill=COLORS["muted"],
    )

    draw_card(draw, 70, "69,8%", "Accuracy công bố của web-app model", COLORS["blue"])
    draw_card(draw, 490, "15/15", "Kết quả pipeline hợp lệ", COLORS["green"])
    draw_card(draw, 910, "7 lớp", "Nhãn biểu cảm FER2013", COLORS["purple"])
    draw_card(draw, 1330, "N/A", "Accuracy nội bộ chưa có ground truth", COLORS["orange"])

    draw_published_panel(draw, published)
    draw_internal_panel(draw, internal, distribution)
    draw_config_panel(draw)
    draw.text(
        (70, 1160),
        "Nguồn: Khanzada, Bai & Celepcikay (2020); FACIAL_SUPPORT.md. Tạo ngày 14/10/2026.",
        font=FONTS["small"],
        fill=COLORS["muted"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.data, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    cli()
