# FakeSTormer Video Deepfake Detection Service
FROM python:3.8-slim

WORKDIR /app

# Install system dependencies for OpenCV, dlib, video processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    libgl1-mesa-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV OPENCV_IO_ENABLE_NONFREE=0
ENV QT_QPA_PLATFORM=offscreen

# Install PyTorch CPU version first (FakeSTormer requires PyTorch 1.8.0)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch==1.11.0 \
    torchvision==0.12.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install OpenCV
RUN pip install --no-cache-dir opencv-python-headless==4.6.0.66

# Install dlib (requires cmake)
RUN pip install --no-cache-dir dlib

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone FakeSTormer repository
RUN git clone https://github.com/10Ring/FakeSTormer.git model_code

# Create weights directory
RUN mkdir -p /app/model_code/weights

# Download shape predictor for dlib face detection
RUN mkdir -p /app/model_code/dlib_models && \
    wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O /tmp/shape_predictor.dat.bz2 && \
    bunzip2 /tmp/shape_predictor.dat.bz2 && \
    mv /tmp/shape_predictor.dat /app/model_code/dlib_models/shape_predictor_68_face_landmarks.dat

# Copy application files
COPY app.py .

# Set Python path for model imports
ENV PYTHONPATH=/app/model_code:/app/model_code/network:/app/model_code/dataset:${PYTHONPATH}

# Environment variables
ENV MODEL_PORT=7002
ENV PRELOAD_MODEL="false"
ENV MODEL_TIMEOUT="900"
ENV USE_GPU="false"
ENV FAKESTORMER_MODEL_PATH="/app/model_code/weights/best.pth"
ENV FAKESTORMER_CONFIG_PATH="/app/model_code/configs/temporal/fakestormer_sbi.yaml"
ENV DLIB_SHAPE_PREDICTOR_PATH="/app/model_code/dlib_models/shape_predictor_68_face_landmarks.dat"
ENV FRAMES_PER_VIDEO="32"

EXPOSE ${MODEL_PORT}
CMD ["python", "app.py"]
