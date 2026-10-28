"""Prepare and verify every local image/audio model before Compose starts the app."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def configuration() -> dict[str, str]:
    return {
        "version": "4",
        "whisper": os.getenv("LOCAL_STT_MODEL", "small"),
        "vieneu": importlib.metadata.version("vieneu"),
        "voice": os.getenv("LOCAL_TTS_VOICE", "vieneu-default"),
        "fer": os.getenv("FER_MODEL_PATH", "/models/fer2013/webapp.onnx"),
        "fer_model_id": os.getenv("FER_MODEL_ID", "legacy"),
        "fer_setup_sha256": hashlib.sha256((ROOT / "setup_fer_vit.py").read_bytes()).hexdigest(),
        "fer_fusion_sha256": hashlib.sha256((ROOT / "fer_vit_fusion.json").read_bytes()).hexdigest(),
        "fer_camera": os.getenv('FER_CAMERA_MODEL_PATH', '/models/fer2013/fer-camera.onnx'),
        "fer_camera_setup_sha256": hashlib.sha256((ROOT / 'setup_fer_camera.py').read_bytes()).hexdigest(),
        "fer_camera_config_sha256": hashlib.sha256((ROOT / 'fer_camera_config.json').read_bytes()).hexdigest(),
        "fer_patt": os.getenv('FER_PATT_MODEL_PATH', '/models/fer2013/patt-lite-raf.onnx'),
        "fer_patt_config_sha256": hashlib.sha256((ROOT / 'fer_patt_config.json').read_bytes()).hexdigest(),
        "fer_patt_setup_sha256": hashlib.sha256((ROOT / 'setup_fer_patt.py').read_bytes()).hexdigest(),
        "yunet": os.getenv("YUNET_MODEL_PATH", "/models/yunet/face_detection_yunet_2023mar.onnx"),
        "setup_sha256": hashlib.sha256((ROOT / "setup_vieneu.py").read_bytes()).hexdigest(),
    }


def is_prepared(manifest: Path, config: dict[str, str]) -> bool:
    try:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        return saved["configuration"] == config and bool(saved["files"]) and all(
            (manifest.parent / item["path"]).is_file()
            and (manifest.parent / item["path"]).stat().st_size == item["bytes"]
            for item in saved["files"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main() -> None:
    models = Path(os.getenv("MODEL_ROOT", "/models")).resolve()
    models.mkdir(parents=True, exist_ok=True)
    manifest = models / "ready.json"
    config = configuration()
    if is_prepared(manifest, config):
        print("Model cache verified; no downloads needed.", flush=True)
        return

    from setup_yunet import main as setup_yunet
    from setup_fer_model import convert
    from setup_vieneu import prepare as setup_vieneu
    from local_voice_server import LocalVoiceEngine
    from fer_model import _session

    if setup_yunet(Path(config["yunet"])) != 0:
        raise RuntimeError("YuNet preparation failed")
    fer = Path(config["fer"])
    if config['fer_model_id'] == 'vit-fer2013':
        from setup_fer_vit import prepare
        prepare(fer)
    elif config['fer_model_id'] != 'legacy':
        raise RuntimeError('Unknown FER_MODEL_ID')
    elif not fer.is_file():
        convert(fer)
    _session(str(fer))
    from PIL import Image
    from fer_model import predict_probabilities
    scores = predict_probabilities(Image.new('RGB', (48, 48), (127, 127, 127)), fer)
    if len(scores) != 7 or abs(float(scores.sum()) - 1) > .001:
        raise RuntimeError('FER inference verification failed')
    from setup_fer_camera import prepare as prepare_camera
    camera = prepare_camera(Path(config['fer_camera']))
    scores = predict_probabilities(Image.new('RGB', (100, 100), (127, 127, 127)), camera)
    if len(scores) != 7 or abs(float(scores.sum()) - 1) > .001:
        raise RuntimeError('Camera FER inference verification failed')
    from setup_fer_patt import prepare as prepare_patt
    patt = prepare_patt(Path(config['fer_patt']))
    scores = predict_probabilities(Image.new('RGB', (100, 100), (127, 127, 127)), patt)
    if len(scores) != 7 or abs(float(scores.sum()) - 1) > .001:
        raise RuntimeError('PAtt-Lite inference verification failed')
    setup_vieneu()

    engine = LocalVoiceEngine()
    engine.initialize()
    health = engine.health()
    if not all(health[role]["ready"] for role in ("stt", "tts")):
        raise RuntimeError(f"Speech model preparation failed: {health}")
    audio = engine.synthesize("Xin chào bạn. Mình đã sẵn sàng hỗ trợ.", speed=1.0)
    if not engine.transcribe(audio, "prepare.wav").strip():
        raise RuntimeError("Speech round-trip verification returned no transcript")

    files = []
    for path in sorted(models.rglob("*")):
        relative = path.relative_to(models)
        if not path.is_file() or path == manifest or any(part in {".locks", ".cache", "cache", "xet"} for part in relative.parts):
            continue
        files.append({"path": relative.as_posix(), "bytes": path.stat().st_size})
    payload = {"configuration": config, "files": files, "health": health}
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest)
    print("YuNet, FER, Whisper and VieNeu prepared and speech round-trip verified.", flush=True)


if __name__ == "__main__":
    main()
