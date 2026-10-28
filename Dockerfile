# --- Giai đoạn 1: build giao diện Vue ---
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# --- Giai đoạn 2: máy chủ Python ---
FROM python:3.12-slim
# Tesseract (OCR tiếng Việt) và Poppler (đọc PDF scan) cho tính năng tóm tắt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng poppler-utils libglib2.0-0 libgomp1 libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-vieneu.txt ./
RUN pip install --no-cache-dir -r requirements-vieneu.txt
COPY *.py ./
COPY fer_vit_fusion.json ./
COPY fer_camera_config.json ./
COPY fer_patt_config.json ./
COPY model_assets ./model_assets
COPY evaluation ./evaluation
COPY fine_tuning ./fine_tuning
COPY deploy/*.py ./deploy/
COPY --from=web /web/dist ./web/dist

# Trong container: nghe mọi địa chỉ (cổng chỉ được publish ra 127.0.0.1 trong docker-compose.yml),
# lưu dữ liệu ở /data và model ở /models (cả hai là volume), STT chạy CPU.
ENV HOST=0.0.0.0 PORT=8000 \
    DB_PATH=/data/healthbot.db \
    LOCAL_VOICE_MODEL_DIR=/models/voice \
    FER_MODEL_PATH=/models/fer2013/webapp.onnx \
    LOCAL_STT_DEVICE=cpu LOCAL_STT_COMPUTE_TYPE=int8 \
    HF_HOME=/models/huggingface \
    PYTHONUNBUFFERED=1
RUN useradd --system --create-home --uid 10001 app && mkdir -p /data /models && chown app /data /models
USER app
VOLUME ["/data", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["python", "server.py"]
