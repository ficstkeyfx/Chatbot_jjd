"""Download VieNeu v3 CPU models and materialize ONNX external data paths.

ONNX Runtime requires external tensor data to remain beside its graph. HF cache
symlinks can resolve into separate blob directories; hard links retain deduplication
while giving ONNX a regular file in the model directory.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def prepare() -> None:
    from huggingface_hub import snapshot_download

    models = {
        "pnnbao-ump/VieNeu-TTS-v3-Turbo": ["onnx_update/*", "voices_v3_turbo.json", "config.json", "denoiser.onnx"],
        "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX": ["*.onnx", "*.data", "*.json"],
    }
    for repo, patterns in models.items():
        snapshot = Path(snapshot_download(repo, allow_patterns=patterns))
        count = 0
        for path in snapshot.rglob("*"):
            if not path.is_symlink() or not path.is_file():
                continue
            source = path.resolve(strict=True)
            temporary = path.with_name(path.name + ".local-" + uuid.uuid4().hex)
            try:
                try:
                    os.link(source, temporary)
                except OSError:
                    shutil.copyfile(source, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            count += 1
        print(f"Prepared {repo}: {count} cache links materialized", flush=True)


if __name__ == "__main__":
    prepare()
