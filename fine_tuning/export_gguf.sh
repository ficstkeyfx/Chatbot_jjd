#!/usr/bin/env bash
# Run from the server project root, after train_qwen.py completes.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export UV_CACHE_DIR="$PWD/.cache/uv"
export OMP_NUM_THREADS=4
run=${1:-runs/qwen-draft-v2}
test -s "$run/report.json"
.venv-train/bin/python fine_tuning/merge_qwen.py "$run"
# Converter dependencies are isolated so the training environment stays intact.
test -d .venv-convert || uv venv --python .venv-train/bin/python --system-site-packages .venv-convert
uv pip install --python .venv-convert/bin/python 'transformers==4.57.6' 'numpy~=2.2.6' 'sentencepiece>=0.1.98,<0.3' 'protobuf>=4.21,<5'
# Save tokenizer using the converter's version; v5 special-token metadata differs.
.venv-convert/bin/python - "$run" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
run = Path(sys.argv[1])
report = json.loads((run / 'report.json').read_text())
AutoTokenizer.from_pretrained(report['base_model'], revision=report['base_revision']).save_pretrained(run / 'merged')
PY
.venv-convert/bin/python tools/llama.cpp/convert_hf_to_gguf.py "$run/merged" --outfile "$run/model-f16.gguf" --outtype f16
tools/llama.cpp/build/bin/llama-quantize "$run/model-f16.gguf" "$run/healthbot-qwen-q4_k_m.gguf" Q4_K_M 4
(cd "$run" && sha256sum healthbot-qwen-q4_k_m.gguf > healthbot-qwen-q4_k_m.gguf.sha256)
tar -czf "$run/adapter.tar.gz" -C "$run" adapter
test -f "$run/training.log" || cp setup-training.log "$run/training.log"
git -C tools/llama.cpp rev-parse HEAD > "$run/converter_revision.txt"
uv pip freeze --python .venv-convert/bin/python > "$run/converter_environment.txt"
echo EXPORT_COMPLETE
