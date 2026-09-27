# FakeSTormer Video Deepfake Detection Service
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential libgl1 libglib2.0-0 ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV QT_QPA_PLATFORM=offscreen

RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    MMCV_WITH_OPS=0 pip install --no-cache-dir --no-build-isolation mmcv==1.7.2 && \
    pip install --no-cache-dir --no-deps retinaface-pytorch==0.0.8 iglovikov-helper-functions==0.0.53

# Upstream code at the pinned commit; the checkpoints are mounted at /app/weights (docker-compose.yml)
RUN git init -q model_code && \
    git -C model_code fetch -q --depth 1 https://github.com/10Ring/FakeSTormer c5b5f75100b878212a192c00013275f275f86182 && \
    git -C model_code checkout -q FETCH_HEAD

# RetinaFace-R50 weights into the torch hub cache at build time
RUN python -c "from retinaface.pre_trained_models import get_model; get_model('resnet50_2020-07-20', max_size=2048)"

COPY app.py .

ENV MODEL_PORT=7002
ENV PRELOAD_MODEL="false"
ENV MODEL_TIMEOUT="900"
ENV FRAMES_PER_VIDEO="32"

EXPOSE ${MODEL_PORT}
CMD ["python", "app.py"]
