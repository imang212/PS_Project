### Docker Simulace Simulace Raspberry Pi OS prostředí (pro testování)
#### 1. Build Docker image
```bash
docker build -t rpi-pytorch .
```

#### 2. Spuštění kontejneru s testem
```bash
docker run --rm rpi-pytorch
```

#### 3. Interaktivní režim
```bash
docker run -it --rm rpi-pytorch
```
### Instalace na Raspberry Pi 5
#### Krok 1: Příprava systému
```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python3.11-venv python3-pip \
    build-essential cmake git wget \
    libopenblas-dev libopencv-dev python3-opencv \
    libatlas-base-dev gfortran \
    libjpeg-dev libpng-dev
```

#### Krok 2: Vytvoření virtuálního prostředí
```bash
cd ~
python3.11 -m venv yolo_env
source yolo_env/bin/activate
```

#### Krok 3: Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

#### Krok 4: Instalace PyTorch
**Varianta A - CPU verze (rychlejší instalace):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Varianta B - Pre-built wheel pro ARM (doporučeno pro RPi):**
```bash
pip install numpy Pillow
wget https://github.com/pytorch/pytorch/releases/download/v2.1.0/torch-2.1.0-cp311-cp311-linux_aarch64.whl
pip install torch-2.1.0-cp311-cp311-linux_aarch64.whl
pip install torchvision torchaudio
```

**Varianta C - Pokud selžou předchozí:**
```bash
# Instalace ze source (trvá 2-3 hodiny)
pip install numpy pyyaml setuptools cffi typing_extensions
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export USE_CUDA=0
export USE_CUDNN=0
export USE_MKLDNN=0
python setup.py install
```

#### Krok 5: Instalace Ultralytics
```bash
pip install ultralytics
```

#### Krok 6: Ověření instalace
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "from ultralytics import YOLO; print('YOLO OK!')"
```

### Test YOLO11

#### Vytvoření testovacího skriptu
```bash
cat > test_yolo.py << 'EOF'
from ultralytics import YOLO

# Načtení modelu
model = YOLO('yolo11n.pt')  # nano model pro rychlé testování

# Predikce na testovacím obrázku
results = model.predict('https://ultralytics.com/images/bus.jpg')

# Výpis výsledků
for result in results:
    print(f"Nalezeno {len(result.boxes)} objektů")
    print(result.boxes)
EOF
```

#### Spuštění testu
```bash
python test_yolo.py
```

### Optimalizace pro Raspberry Pi AI HAT+
Pro využití AI HAT+ (26 TOPS) budete potřebovat:

1. **Hailo Driver** pro AI HAT+:
```bash
# Instalace Hailo runtime
wget https://hailo.ai/downloads/hailo-runtime-arm64.deb
sudo dpkg -i hailo-runtime-arm64.deb
```

2. **Export modelu pro Hailo**:
```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
# Export pro Hailo accelerátor
model.export(format='hailo', imgsz=640)
```

### Řešení problémů

#### Problém: "No module named 'torch'"
- Ujistěte se, že je virtuální prostředí aktivované: `source yolo_env/bin/activate`
- Zkontrolujte instalaci: `pip list | grep torch`

#### Problém: Instalace PyTorch trvá věčnost
- Použijte pre-built wheel (Varianta B)
- Nebo CPU verzi (Varianta A)

#### Problém: Nedostatek paměti
- Zvyšte swap na RPi:
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # nastavte CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### Problém: Import chyby
```bash
# Přeinstalace závislostí
pip install --force-reinstall --no-cache-dir ultralytics
```

### Kontrola instalace
Spusťte kompletní test:
```bash
python << EOF
import sys
import torch
import ultralytics
from ultralytics import YOLO

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"CUDA dostupné: {torch.cuda.is_available()}")
print("\n✅ Všechny knihovny úspěšně nainstalovány!")
EOF
```