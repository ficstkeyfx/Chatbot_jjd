"""Install pinned ViT FER weights and explicit preprocessing metadata on the model drive."""
from __future__ import annotations
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REPOSITORY = 'onnx-community/vit-face-expression-ONNX'
REVISION = '164ea84136a47af225c54ece15fd9e8b900bef7e'
SHA256 = '44980c02cecbdc22a59f9c4d97edf02a7e6785ac71d6a5bce67dcf0949781b9e'
URL = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/onnx/model.onnx'
DEFAULT_PATH = Path(__file__).resolve().parent / '.models/fer2013/vit-fer2013-fp32.onnx'
# Deployment defaults are selected only on public calibration, never privateTest.
FLIP = True
LEGACY_WEIGHT = 0.0
MIN_CONFIDENCE = 0.6
MIN_MARGIN = 0.3


def prepare(target: Path = DEFAULT_PATH) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != SHA256:
        temporary = target.with_suffix('.download')
        urllib.request.urlretrieve(URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('FER ViT model checksum mismatch')
        temporary.replace(target)
    metadata = {'format': 1, 'architecture': 'vit', 'input_size': 224,
                'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5],
                'labels': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                'tta': 'flip' if FLIP else 'none', 'ensemble_legacy_weight': LEGACY_WEIGHT,
                'min_confidence': MIN_CONFIDENCE, 'min_margin': MIN_MARGIN,
                'repository': REPOSITORY, 'revision': REVISION, 'sha256': SHA256,
                'upstream': 'https://huggingface.co/trpakov/vit-face-expression',
                'license': 'Apache-2.0 (upstream model)',
                'calibration': 'FER2013 publicTest; exact privateTest overlaps excluded'}
    fusion_path = Path(__file__).resolve().parent / 'fer_vit_fusion.json'
    metadata['fusion'] = json.loads(fusion_path.read_text(encoding='utf-8'))
    metadata['fusion_sha256'] = hashlib.sha256(fusion_path.read_bytes()).hexdigest()
    sidecar = Path(str(target) + '.json')
    temporary = sidecar.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    temporary.replace(sidecar)
    if LEGACY_WEIGHT or metadata.get('fusion'):
        from setup_fer_model import convert
        legacy = target.with_name('webapp.onnx')
        if not legacy.is_file():
            convert(legacy)
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_PATH)
    print(prepare(p.parse_args().output))
