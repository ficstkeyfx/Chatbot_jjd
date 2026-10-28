import re
import unicodedata


EMERGENCY_NUMBER = "115"
IMMEDIATE_RISK_MESSAGE = (
    "Mình rất tiếc vì bạn đang phải chịu đựng điều này. "
    "Nếu bạn có thể làm hại bản thân hoặc người khác ngay lúc này, hãy gọi 115 "
    "hoặc đến cơ sở y tế gần nhất ngay. Hãy rời xa vật có thể gây hại và gọi một "
    "người bạn tin cậy đến ở cùng bạn. Bạn có đang gặp nguy hiểm ngay lúc này không?"
)
EYE_URGENT_MESSAGE = (
    "Đau mắt dữ dội hoặc thị lực thay đổi/giảm có thể cần được khám khẩn. "
    "Hãy liên hệ cơ sở y tế hoặc chuyên khoa mắt ngay; nếu mất thị lực đột ngột, "
    "hãy đến cấp cứu. Đừng chỉ áp dụng quy tắc nghỉ mắt 20-20-20."
)


_FACE_SUPPORT_STEPS = {
    "angry": "Nếu lúc này bạn thấy bực bội, hãy thử tạm dừng một phút và thở chậm trước khi quyết định việc tiếp theo.",
    "disgust": "Nếu lúc này bạn thấy khó chịu, hãy thử tạm dừng một lát và thả lỏng vai.",
    "frowning": "Nếu bạn đang căng thẳng, hãy thử thả lỏng vai và thở chậm vài nhịp.",
    "fear": "Nếu bạn đang thấy bất an, hãy thử đặt hai bàn chân chạm sàn và chú ý vài thứ xung quanh.",
    "sad": "Nếu bạn đang trải qua chuyện khó khăn, bạn có thể nhắn một người bạn tin cậy để được lắng nghe.",
    "tears_visible": "Nếu có điều gì đang khó với bạn, bạn có thể nhắn một người bạn tin cậy để được lắng nghe.",
    "happy": "Nếu có điều gì vừa giúp bạn thấy dễ chịu, bạn có thể dành một phút ghi nhận điều đó.",
    "smiling": "Nếu có điều gì vừa giúp bạn thấy dễ chịu, bạn có thể dành một phút ghi nhận điều đó.",
    "surprise": "Bạn có thể dừng lại một phút, chú ý hơi thở và xem mình cần gì lúc này.",
    "neutral": "Bạn có thể dừng lại một phút, chú ý hơi thở và xem mình cần gì lúc này.",
    "unclear": "Bạn có thể dừng lại một phút, chú ý hơi thở và xem mình cần gì lúc này.",
}


def face_support_step(expression: str) -> str:
    """Offer a small optional step; a facial label is not a mental-state assessment."""
    return _FACE_SUPPORT_STEPS.get(expression, _FACE_SUPPORT_STEPS["unclear"])


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


_NON_IMMEDIATE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(khong|chua) (co y dinh |dinh |muon )?"
        r"(tu sat|tu tu|tu lam hai|lam hai (ban than|minh|ai|nguoi khac))\b",
        r"\b(khong|chua) (muon|se|sap|dang|dinh) (nhay lau|nhay xuong|cat co tay)\b",
        r"\b(tung co|da tung co).*(nhieu nam truoc|truoc day).*(hien gio|bay gio).*(khong co|khong con)\b",
        r"\b(cum tu|cau noi).*(co nghia la gi|danh gia tam ly)\b",
        r"\b(trong phim|nhan vat|tieu thuyet).*(tu sat|tu tu|toi se chet)\b",
    )
)


_IMMEDIATE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(muon|se|sap|dang|dinh|quyet dinh) (tu sat|tu tu|chet|ket thuc cuoc doi)\b",
        r"\b(tu lam hai|lam hai) (ban than|minh|nguoi khac)\b",
        r"\bco (ke hoach|dao|sung|thuoc|day)\b.*\b(tu sat|tu tu|lam hai|chet)\b",
        r"\b(chuan bi|co) (dao|sung|thuoc|day)\b.*\b(dinh|se|muon)?\s*(lam hai|minh|chet)\b",
        r"\b(uong|nuot)\b.*\b(thuoc|ca vi)\b.*\b(muon chet|de chet|tu sat|tu tu)\b",
        r"\b(muon chet|bien mat mai mai|khong the tiep tuc song)\b",
        r"\bkhong muon song\b",
        r"\bkhong con ly do de song\b",
        r"\btoi se (giet|lam hai)\b",
        r"\b(da|vua|moi) (uong|nuot)\b.{0,60}\b(qua lieu|ca vi thuoc|nhieu thuoc)\b",
        r"\b(muon|se|sap|dang|dinh) (nhay lau|nhay xuong|cat co tay)\b",
    )
)

_DISTRESS_TERMS = (
    "hoang loan",
    "tuyet vong",
    "khung hoang",
    "khong chiu noi",
    "rat so hai",
    "muon bien mat",
)

_EYE_RED_FLAG_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(dau|nhuc) mat (du doi|rat nhieu|khong chiu noi)\b",
        r"\bmat(?: toi| minh| em)? (dau|nhuc) (du doi|rat nhieu|khong chiu noi)\b",
        r"\b(mat thi luc|giam thi luc|thay doi thi luc|mat nhin)\b",
        r"\bthi luc (?:cua (?:toi|minh|em) )?(?:thay doi|giam|mat)\b",
        r"\b(nhin mo|mo mat) (dot ngot|bong nhien)\b",
        r"\bkhong nhin thay\b",
    )
)


def classify_eye_red_flags(text: str) -> bool:
    """Conservatively identify self-reported severe eye pain or vision change."""
    normalized = re.sub(r"\s+", " ", _normalize(text)).strip()
    for pattern in _EYE_RED_FLAG_PATTERNS:
        for match in pattern.finditer(normalized):
            if match.group() == "khong nhin thay":
                return True
            prefix = normalized[max(0, match.start() - 25):match.start()]
            if re.search(r"\b(khong|chua)(?:\s+\w+){0,2}\s+$", prefix):
                continue
            return True
    return False


def classify_crisis(text: str) -> str:
    """Conservative rule layer; never replaces professional assessment."""
    normalized = _normalize(text)
    screened = normalized
    for pattern in _NON_IMMEDIATE_PATTERNS:
        screened = pattern.sub(" ", screened)
    if any(pattern.search(screened) for pattern in _IMMEDIATE_PATTERNS):
        return "immediate_risk"
    if any(term in normalized for term in _DISTRESS_TERMS):
        return "distress"
    return "normal"
