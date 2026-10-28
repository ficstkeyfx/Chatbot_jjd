"""Chuẩn hóa văn bản tiếng Việt trước khi đọc thành tiếng.

Mô hình TTS đọc rất tệ số, đơn vị, ký hiệu Markdown và emoji. Module này đổi chúng thành chữ và
tách văn bản thành các câu ngắn để engine chèn quãng nghỉ tự nhiên giữa các câu.
"""

from __future__ import annotations

import re
import unicodedata

_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

# Số đặc biệt đọc từng chữ số (số cấp cứu, số điện thoại ngắn).
_SPOKEN_AS_DIGITS = {"113", "114", "115"}

_UNITS = (
    (r"°\s*C\b", " độ xê"),
    (r"%", " phần trăm"),
    (r"\bkm\b", " ki-lô-mét"),
    (r"\bkg\b", " ki-lô-gam"),
    (r"\bmg\b", " mi-li-gam"),
    (r"\bml\b", " mi-li-lít"),
    (r"\bcm\b", " xen-ti-mét"),
    (r"\bmmHg\b", " mi-li-mét thủy ngân"),
    (r"\bSpO2\b", " ét-pi-ô-hai"),
    (r"\bBMI\b", " bi-em-ai"),
    (r"(?<=\d)\s*m\b", " mét"),
    (r"(?<=\d)\s*s\b", " giây"),
    (r"(?<=\d)\s*h\b", " giờ"),
)


def number_to_words(value: int) -> str:
    """Đọc số nguyên 0..999999 theo cách nói tiếng Việt."""
    if value < 0 or value > 999_999:
        return " ".join(_DIGITS[int(d)] for d in str(abs(value)))
    if value < 10:
        return _DIGITS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        head = "mười" if tens == 1 else f"{_DIGITS[tens]} mươi"
        if unit == 0:
            return head
        if unit == 1 and tens > 1:
            return f"{head} mốt"
        if unit == 5:
            return f"{head} lăm"
        if unit == 4 and tens > 1:
            return f"{head} tư"
        return f"{head} {_DIGITS[unit]}"
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        head = f"{_DIGITS[hundreds]} trăm"
        if rest == 0:
            return head
        if rest < 10:
            return f"{head} linh {_DIGITS[rest]}"
        return f"{head} {number_to_words(rest)}"
    thousands, rest = divmod(value, 1000)
    head = f"{number_to_words(thousands)} nghìn"
    if rest == 0:
        return head
    if rest < 100:
        return f"{head} không trăm {number_to_words(rest)}" if rest >= 10 else f"{head} không trăm linh {_DIGITS[rest]}"
    return f"{head} {number_to_words(rest)}"


def _read_number(match: re.Match[str]) -> str:
    text = match.group()
    if text in _SPOKEN_AS_DIGITS or (len(text) > 1 and text.startswith("0")) or len(text) > 6:
        return " ".join(_DIGITS[int(d)] for d in text)
    return number_to_words(int(text))


def _read_decimal(match: re.Match[str]) -> str:
    whole, fraction = match.group(1), match.group(2)
    return f"{number_to_words(int(whole))} phẩy {' '.join(_DIGITS[int(d)] for d in fraction)}"


def _strip_symbols(text: str) -> str:
    # Bỏ emoji và ký hiệu hình vẽ (nhóm So/Sk) nhưng giữ chữ và dấu câu.
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if category in {"So", "Sk", "Cs", "Co"} or ord(char) in {0xFE0F, 0x200D}:
            continue
        kept.append(char)
    return "".join(kept)


def normalize_for_speech(text: str) -> str:
    """Đổi ký hiệu, số và đơn vị thành chữ để đọc được tự nhiên."""
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    # Ký hiệu độ phải đổi trước khi bước lọc ký hiệu loại bỏ nó.
    value = re.sub(r"°\s*C\b", " độ xê", value)
    value = value.replace("°", " độ")
    value = _strip_symbols(value)
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"[`*_#>~]+", "", value)
    value = re.sub(r"^\s*(?:[-•+]|\d{1,2}[.)])\s+", "", value, flags=re.M)

    # Quy tắc 20-20-20 và dãy số nối bằng gạch: đọc từng số, ngắt nhẹ giữa các số.
    value = re.sub(
        r"\b(\d{1,3})(?:\s?[-–—]\s?(\d{1,3})){2,}\b",
        lambda m: ", ".join(number_to_words(int(n)) for n in re.findall(r"\d+", m.group())),
        value,
    )
    # Khoảng giá trị: 20–45 giây -> hai mươi đến bốn mươi lăm giây.
    value = re.sub(r"\b(\d+)\s?[-–—]\s?(\d+)\b", lambda m: f"{m.group(1)} đến {m.group(2)}", value)
    value = re.sub(r"(\d+)\s?/\s?(\d+)", r"\1 trên \2", value)

    for pattern, replacement in _UNITS:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE if pattern.islower() else 0)

    value = re.sub(r"\b(\d+)[.,](\d+)\b", _read_decimal, value)
    value = re.sub(r"\b\d+\b", _read_number, value)
    value = value.replace("&", " và ").replace("+", " cộng ").replace("=", " bằng ")
    value = re.sub(r"[()\[\]{}<>|\\]", ", ", value)
    value = re.sub(r"[ \t]+", " ", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def split_sentences(text: str, max_chars: int = 180) -> list[str]:
    """Tách thành câu; câu quá dài được tách tiếp ở dấu phẩy để mỗi lần tổng hợp ngắn và ổn định."""
    sentences: list[str] = []
    for chunk in _SENTENCE_END.split(text):
        chunk = chunk.strip(" ,;:-")
        if not chunk:
            continue
        while len(chunk) > max_chars:
            cut = max(chunk.rfind(", ", 0, max_chars), chunk.rfind("; ", 0, max_chars))
            if cut < max_chars // 3:
                cut = chunk.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            sentences.append(chunk[: cut + 1].strip(" ,;:"))
            chunk = chunk[cut + 1 :].strip(" ,;:")
        if chunk:
            sentences.append(chunk)
    return sentences


def prepare_for_speech(text: str) -> list[str]:
    """Văn bản bất kỳ -> danh sách câu đã chuẩn hóa, sẵn sàng đưa cho engine TTS."""
    return [s for s in split_sentences(normalize_for_speech(text)) if re.search(r"\w", s)]
