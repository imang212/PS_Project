# Simulace Raspberry Pi OS (Debian 12 Bookworm) prostředí
FROM debian:bookworm-slim
# Nastavení proměnných prostředí
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11
ENV PATH="/venv/bin:$PATH"

# Instalace základních závislostí
RUN apt-get update && apt-get install -y \
    python3.11 \ 
    python3.11-venv \ 
    python3-pip \
    build-essential \ 
    cmake \ 
    git \ 
    wget \
    libopenblas-dev \ 
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \ 
    gfortran \
    libjpeg-dev \ 
    libpng-dev \ 
    libavcodec-dev \ 
    libavformat-dev \
    libswscale-dev \ 
    libv4l-dev \ 
    libxvidcore-dev \ 
    libx264-dev \
    libhdf5-dev \ 
    libhdf5-serial-dev \
    && rm -rf /var/lib/apt/lists/*
# Vytvoření virtuálního prostředí
RUN python3.11 -m venv /venv
# Aktivace virtuálního prostředí a instalace základních balíčků
RUN . /venv/bin/activate && \
    pip install --upgrade pip setuptools wheel
# Instalace PyTorch pro ARM64 (Raspberry Pi)
# Pro RPi je potřeba použít speciální verzi nebo build z source
RUN . /venv/bin/activate && \
    pip install numpy Pillow && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Instalace Ultralytics (YOLO11) a dalších potřebných knihoven
RUN /venv/bin/pip install --no-cache-dir ultralytics fastapi "uvicorn[standard]" websockets opencv-python numpy scipy filterpy pydantic

# Vytvoření testovacího skriptu
RUN echo '#!/bin/bash\n\
source /venv/bin/activate\n\
echo "Python verze:"\n\
python --version\
echo "PyTorch verze:"\n\
python -c "import torch; print(f\"PyTorch: {torch.__version__}\")"\n\
echo "Ultralytics verze:"\n\
python -c "import ultralytics; print(f\"Ultralytics: {ultralytics.__version__}\")"\n\
echo "Test YOLO11:"\n\
python -c "from ultralytics import YOLO; print(\"YOLO import úspěšný!\")"\n\
' > /test_install.sh && chmod +x /test_install.sh
# Vytvoření skriptu pro spuštění bashu s aktivovaným venv
RUN echo '#!/bin/bash\n\
source /venv/bin/activate\n\
exec bash "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh
# Nastavení entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/test_install.sh"]