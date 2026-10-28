"""ONNX facial-expression inference: legacy CNN and calibrated ViT/CNN ensemble.

Legacy weights are converted by ``setup_fer_model.py``; ViT weights and explicit
label/normalization metadata are pinned by ``setup_fer_vit.py``. Images are not persisted.

Phát hiện khuôn mặt: ưu tiên YuNet (OpenCV, ``setup_yunet.py`` để tải model) vì ít báo nhầm hơn Haar và
cho tọa độ mắt để căn chỉnh khuôn mặt; tự dùng Haar cascade khi chưa có model YuNet.
"""

from __future__ import annotations

import base64
import io
import math
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
DEFAULT_MODEL_PATH = Path(__file__).parent / ".models" / "fer2013" / "webapp.onnx"
DEFAULT_YUNET_PATH = Path(__file__).parent / ".models" / "yunet" / "face_detection_yunet_2023mar.onnx"
MIN_CONFIDENCE = 0.50
MIN_MARGIN = 0.12
MIN_FACE_PIXELS = 32
YUNET_SCORE_THRESHOLD = 0.7
# Chỉ căn chỉnh khi mặt nghiêng vừa phải; nghiêng quá nhiều thì landmark thường không đáng tin.
MAX_ALIGN_DEGREES = 45.0


@dataclass(frozen=True)
class Face:
    """Một khuôn mặt dò được. ``eyes`` là (mắt trái trong ảnh, mắt phải trong ảnh) hoặc None."""

    box: tuple[int, int, int, int]  # x, y, rộng, cao
    score: float = 1.0
    eyes: tuple[tuple[float, float], tuple[float, float]] | None = None


@lru_cache(maxsize=4)
def _session(path: str):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(8, int(os.getenv('FER_CPU_THREADS', '2'))))
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=4)
def model_metadata(path: str) -> dict | None:
    sidecar = Path(path + '.json')
    if not sidecar.is_file():
        return None
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    architecture = metadata.get('architecture')
    if metadata.get('format') != 1 or architecture not in {'vit', 'efficientnet_b2', 'patt_lite'}:
        raise ValueError('Unsupported FER model metadata')
    labels = metadata.get('labels', [])
    if len(labels) != len(LABELS) or set(labels) != set(LABELS):
        raise ValueError('FER label mapping must contain each of the seven labels once')
    if metadata.get('input_size') != (260 if architecture == 'efficientnet_b2' else 224) or metadata.get('tta') not in {'none', 'flip'}:
        raise ValueError('Invalid FER input configuration')
    if architecture == 'patt_lite' and metadata.get('tta') != 'none':
        raise ValueError('PAtt-Lite requires independent-photo inference without batched TTA')
    if metadata.get('tta_reduction', 'mean_probabilities') not in {'mean_probabilities', 'mean_logits'}:
        raise ValueError('Invalid FER augmentation reduction')
    companion = metadata.get('companion')
    if companion and (architecture != 'efficientnet_b2' or companion.get('filename') != 'vit-fer2013-fp32.onnx' or not 0 <= companion.get('weight', -1) <= 1):
        raise ValueError('Invalid FER companion model')
    mean, std = np.asarray(metadata.get('mean')), np.asarray(metadata.get('std'))
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid FER normalization')
    if not 0 <= metadata.get('ensemble_legacy_weight', 0) <= 0.5:
        raise ValueError('Invalid FER ensemble weight')
    if not all(0 <= metadata.get(key, default) <= 1 for key, default in [('min_confidence', MIN_CONFIDENCE), ('min_margin', MIN_MARGIN)]):
        raise ValueError('Invalid FER decision threshold')
    fusion = metadata.get('fusion')
    if fusion:
        if fusion.get('labels') != list(LABELS) or metadata['tta'] != 'flip':
            raise ValueError('FER fusion label order or augmentation mismatch')
        for name, shape in [('mean', (14,)), ('scale', (14,)), ('coefficients', (7,14)), ('intercept', (7,))]:
            values = np.asarray(fusion.get(name), dtype=np.float64)
            if values.shape != shape or not np.isfinite(values).all() or (name == 'scale' and (values <= 0).any()):
                raise ValueError('Invalid FER fusion parameters')
    return metadata


def decision_thresholds(model_path: str | Path) -> tuple[float, float]:
    metadata = model_metadata(str(Path(model_path).resolve())) or {}
    return metadata.get('min_confidence', MIN_CONFIDENCE), metadata.get('min_margin', MIN_MARGIN)


def yunet_path() -> Path:
    return Path(os.getenv("YUNET_MODEL_PATH", "").strip() or DEFAULT_YUNET_PATH)


def resolve_detector(detector: str | None = None) -> str:
    """'yunet' hoặc 'haar'. Mặc định (FACE_DETECTOR=auto): YuNet nếu đã tải model, ngược lại Haar."""
    choice = (detector or os.getenv("FACE_DETECTOR", "auto")).strip().lower()
    if choice == "haar":
        return "haar"
    try:
        import cv2

        available = hasattr(cv2, "FaceDetectorYN") and yunet_path().is_file()
    except ImportError:
        available = False
    return "yunet" if available else "haar"


def _detect_haar(rgb: np.ndarray) -> list[Face]:
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    if detector.empty():
        raise RuntimeError("OpenCV face detector is unavailable")
    boxes = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_PIXELS, MIN_FACE_PIXELS)
    )
    return [Face(tuple(map(int, box))) for box in boxes]


def _detect_yunet(rgb: np.ndarray) -> list[Face]:
    import cv2

    height, width = rgb.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(yunet_path()), "", (width, height), YUNET_SCORE_THRESHOLD, 0.3, 5000
    )
    # YuNet nhận ảnh BGR.
    _, rows = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    faces: list[Face] = []
    for row in (rows if rows is not None else []):
        x, y, w, h = (int(round(float(v))) for v in row[:4])
        x, y = max(0, x), max(0, y)
        w, h = min(w, width - x), min(h, height - y)
        if w < MIN_FACE_PIXELS or h < MIN_FACE_PIXELS:
            continue
        # Landmark: mắt phải (bên trái ảnh), mắt trái (bên phải ảnh), mũi, hai khóe miệng.
        eyes = ((float(row[4]), float(row[5])), (float(row[6]), float(row[7])))
        faces.append(Face((x, y, w, h), float(row[14]), eyes))
    return faces


def detect_faces(rgb: np.ndarray, detector: str | None = None) -> list[Face]:
    """Dò mọi khuôn mặt trong ảnh RGB bằng YuNet hoặc Haar (xem resolve_detector)."""
    return _detect_yunet(rgb) if resolve_detector(detector) == "yunet" else _detect_haar(rgb)


def crop_face(image: Image.Image, face: Face, align: bool | None = None) -> Image.Image:
    """Cắt khuôn mặt; nếu có tọa độ mắt thì xoay cho hai mắt nằm ngang."""
    x, y, w, h = face.box
    if align is None:
        align = os.getenv("FACE_ALIGN", "1").strip().lower() not in {"0", "false", "no"}
    if align and face.eyes:
        (x1, y1), (x2, y2) = face.eyes
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if 1.0 <= abs(angle) <= MAX_ALIGN_DEGREES:
            # Cắt vùng rộng hơn (đệm viền) rồi xoay quanh tâm khuôn mặt để không mất góc ảnh.
            cx, cy = x + w / 2, y + h / 2
            side = max(w, h) * 1.6
            region = image.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
            region = region.rotate(angle, resample=Image.Resampling.BICUBIC)
            left, top = (region.width - w) / 2, (region.height - h) / 2
            return region.crop((round(left), round(top), round(left) + w, round(top) + h))
    return image.crop((x, y, x + w, y + h))


def detect_expression(
    data_url: str, model_path: str | Path = DEFAULT_MODEL_PATH, detector: str | None = None
) -> tuple[str, str]:
    """Return face status and FER2013 label; abstain on multiple/unclear faces."""
    encoded = data_url.partition(",")[2]
    if not encoded:
        return "unclear", "unclear"
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    faces = detect_faces(np.asarray(image), detector)
    if len(faces) != 1:
        return ("none" if len(faces) == 0 else "unclear"), "unclear"
    probabilities = predict_probabilities(crop_face(image, faces[0]), model_path)
    return "one", decide_label(probabilities, *decision_thresholds(model_path))


def probability_views(face: Image.Image, model_path: str | Path, *, augment: bool = True):
    """Probability vectors in canonical LABELS order, before optional ensembling."""
    path = str(Path(model_path).resolve())
    metadata = model_metadata(path)
    session = _session(path)
    input_name = session.get_inputs()[0].name
    if metadata and metadata['architecture'] == 'patt_lite' and session.get_inputs()[0].shape[0] != 1:
        raise ValueError('PAtt-Lite must use the fixed batch-one export')
    if metadata:
        if metadata['architecture'] == 'efficientnet_b2':
            import cv2
            pixels = cv2.resize(np.asarray(face.convert('RGB')), (260, 260)).astype(np.float32) / 255.0
        else:
            pixels = np.asarray(face.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        pixels = (pixels - np.asarray(metadata['mean'], dtype=np.float32)) / np.asarray(metadata['std'], dtype=np.float32)
        batch = pixels.transpose(2, 0, 1)[None].astype(np.float32)
        if augment:
            batch = np.concatenate([batch, batch[:, :, :, ::-1]], axis=0)
        first_dim = session.get_inputs()[0].shape[0]
        if isinstance(first_dim, int) and first_dim == 1 and len(batch) > 1:
            logits = np.concatenate([session.run(None, {input_name: item[None]})[0] for item in batch], axis=0)
        else:
            logits = session.run(None, {input_name: batch})[0]
        if logits.shape != (len(batch), 7) or not np.isfinite(logits).all():
            raise ValueError('Invalid FER model output')
        if metadata.get('tta_reduction') == 'mean_logits':
            logits = logits.mean(axis=0, keepdims=True)
        scores = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores /= scores.sum(axis=1, keepdims=True)
        return scores[:, [metadata['labels'].index(label) for label in LABELS]]
    gray = face.convert("L").resize((48, 48), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32) / 255.0
    batch = pixels[None, :, :, None]
    direct = session.run(None, {input_name: batch})[0][0]
    if not augment:
        return np.asarray([direct])
    flipped = session.run(None, {input_name: batch[:, :, ::-1, :].copy()})[0][0]
    return np.asarray([direct, flipped])


def predict_probabilities(face: Image.Image, model_path: str | Path = DEFAULT_MODEL_PATH):
    """Classify a crop with the configured ONNX backend; do not infer mental state."""
    metadata = model_metadata(str(Path(model_path).resolve()))
    probabilities = probability_views(face, model_path, augment=metadata is None or metadata['tta'] == 'flip').mean(axis=0)
    weight = metadata.get('ensemble_legacy_weight', 0) if metadata else 0
    fusion = metadata.get('fusion') if metadata else None
    if weight or fusion:
        legacy_path = Path(model_path).with_name('webapp.onnx')
        if legacy_path.resolve() == Path(model_path).resolve():
            raise ValueError('FER ensemble cannot refer to itself')
        legacy = probability_views(face, legacy_path).mean(axis=0)
        if fusion:
            features = np.log(np.clip(np.concatenate([probabilities, legacy]).astype(np.float64), 1e-6, 1.0))
            logits = ((features - np.asarray(fusion['mean'])) / np.asarray(fusion['scale'])) @ np.asarray(fusion['coefficients']).T + np.asarray(fusion['intercept'])
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
        else:
            probabilities = (1 - weight) * probabilities + weight * legacy
    if not np.isfinite(probabilities).all():
        raise ValueError('Non-finite FER probabilities')
    companion = metadata.get('companion') if metadata else None
    if companion:
        other_path = Path(model_path).with_name(companion['filename'])
        other_metadata = model_metadata(str(other_path.resolve()))
        if not other_metadata or other_metadata.get('architecture') != 'vit' or other_metadata.get('companion'):
            raise ValueError('FER companion must be the standalone ViT fusion model')
        probabilities = (1 - companion['weight']) * probabilities + companion['weight'] * predict_probabilities(face, other_path)
    return probabilities


def decide_label(
    probabilities, min_confidence: float = MIN_CONFIDENCE, min_margin: float = MIN_MARGIN
) -> str:
    """Chọn nhãn hoặc từ chối ('unclear') khi độ tin cậy hay độ chênh với nhãn thứ hai quá thấp."""
    best = int(np.argmax(probabilities))
    runner_up = float(np.partition(probabilities, -2)[-2])
    if probabilities[best] < min_confidence or probabilities[best] - runner_up < min_margin:
        return "unclear"
    return LABELS[best]


def classify_expression(
    data_url: str, model_path: str | Path = DEFAULT_MODEL_PATH, detector: str | None = None
) -> str:
    """Return a label only when the detector finds one face confidently."""
    return detect_expression(data_url, model_path, detector)[1]
