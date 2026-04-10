FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-paho-mqtt \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1 \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt
COPY . .
CMD ["python3", "yoloTrafficDetectionSystem_PC_GPU.py"]