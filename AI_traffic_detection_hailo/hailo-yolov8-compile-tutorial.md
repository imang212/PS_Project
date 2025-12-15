# Hailo YOLOv8 Kompilace - Kompletní Návod
Tento návod popisuje proces konverze YOLOv8 modelu do Hailo HEF formátu pro inference na Hailo-8L akcelerátoru.

## Předpoklady
- Nainstalované Hailo Dataflow Compiler (DFC) pro Hailo-8/8L (odkaz:`https://hailo.ai/developer-zone/software-downloads/`)
- Linux OS x86 (Ubuntu,... )
- Vyexportovaný ONNX model (např. `yolov8m.onnx`)
- Python 3.11 prostředí s aktivovaným venv

```bash
python3.11 -m venv hailo_env
source hailo_env/bin/activate
pip install ultralytics
pip install hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
```

Modely pro hailo se dají stáhnout z repozitáře `https://github.com/hailo-ai/hailo_model_zoo/tree/master`, ale ne pro všechny, proto ho budem exportovat.


## Vyexportování modelu
```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
model.export(format='onnx')
```



## Krok 1: Parsing ONNX modelu
Převod ONNX modelu do Hailo HAR (Hailo Archive) formátu.

### Základní příkaz
```bash
hailo parser onnx yolov8m.onnx --hw-arch hailo8l
```

### Doporučený příkaz s end nodes
Parser často doporučí použít specifické end nodes pro lepší kompatibilitu s HailoRT post-processingem:

```bash
hailo parser onnx yolov8m.onnx --hw-arch hailo8l \
  --end-node-names /model.22/cv2.0/cv2.0.2/Conv \
                   /model.22/cv3.0/cv3.0.2/Conv \
                   /model.22/cv2.1/cv2.1.2/Conv \
                   /model.22/cv3.1/cv3.1.2/Conv \
                   /model.22/cv2.2/cv2.2.2/Conv \
                   /model.22/cv3.2/cv3.2.2/Conv \
                   /model.22/Concat_3
```

### Co se stane
- Parser analyzuje strukturu ONNX modelu
- Detekuje YOLOv8 architekturu
- Vytvoří `yolov8m.har` soubor
- Automaticky přidá NMS post-processing (pokud potvrdíte doporučení)

### Výstup
```
yolov8m.har
```

## Krok 2: Optimalizace (Kvantizace)
Kvantizace modelu z FP32 na INT8 pro efektivní běh na Hailo hardware.

### S kalibračními daty (DOPORUČENO)
Pro nejlepší přesnost použijte skutečné obrázky:
```bash
# 1. Připravte kalibrační dataset
mkdir calibration_dataset
cp /cesta/k/obrazkum/*.jpg calibration_dataset/
# 2. Spusťte optimalizaci
hailo optimize yolov8m.har --calib-set-path calibration_dataset/
```

**Doporučení pro kalibrační data:**
- 100-500 reprezentativních obrázků
- Formáty: JPG, PNG
- Obrázky podobné produkčním datům (dopravní scény)
- Různé světelné podmínky a úhly

### S náhodnými daty (testování)
Pro rychlé testování bez kalibračních dat:
```bash
hailo optimize yolov8m.har --use-random-calib-set
```

**Poznámka:** Přesnost modelu bude výrazně nižší, ale pro testování zatím stačí. Pro produkci se může potom použít kalibrační dataset.

### Pokročilé parametry
```bash
hailo optimize yolov8m.har \
  --calib-set-path calibration_dataset/ \
  --output-har-path yolov8m_quantized.har \
  --work-dir ./optimization_work
```

### Co se stane
- Model je kvantizován z FP32 na INT8
- Použijí se kalibrační data pro zachování přesnosti
- Vytvoří se optimalizovaný HAR soubor

### Výstup
```
yolov8m_optimized.har
```

**Čas:** 5-15 minut (závisí na velikosti kalibračního datasetu)

## Krok 3: Kompilace do HEF
Finální kompilace optimalizovaného modelu do Hailo Executable Format.
V tomto příkazu se výsledný hex soubor se uloží do aktuální složky.

```bash
hailo compiler yolov8m_optimized.har \
  --output-dir ./
```

### Co se stane
- Model je rozdělен na optimální partition pro Hailo hardware
- Generují se instrukce pro Neural Network Core
- Vytváří se binární HEF soubor

### Výstup
```
yolov8m.hef
```

**Čas:** 15-45 minut pro YOLOv8m (závisí na CPU a RAM)

## Kompletní Workflow
```bash
# 1. PARSING
hailo parser onnx yolov8m.onnx --hw-arch hailo8l
# Odpovězte 'y' na doporučení použít specifické end nodes

# 2. OPTIMALIZACE
hailo optimize yolov8m.har \
  --calib-set-path calibration_dataset/

# 3. KOMPILACE
hailo compiler yolov8m_optimized.har \
  --output-path yolov8m.hef
```

## Časté problémy a jejich řešení
### Parsing selhává
**Problém:** Parser nedokáže zpracovat ONNX model

**Řešení:**
1. Použijte doporučené end nodes z parser výstupu
2. Re-exportujte ONNX model bez post-processingu:

```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
model.export(format='onnx', simplify=True, opset=11)
```

### Optimalizace vyžaduje kalibrační data
**Řešení:**
- Pro testování: použijte `--use-random-calib-set`
- Pro produkci: připravte kalibrační dataset

### Kompilace trvá příliš dlouho
**Problém:** Kompilace běží více než 60 minut

**Řešení:**
- Zkontrolujte dostupnou RAM (minimum 8GB)
- Ukončete jiné náročné procesy
- Pro větší modely (YOLOv8l/x) může kompilace trvat až 90 minut

### Nedostatek paměti
**Problém:** `MemoryError` během optimalizace nebo kompilace

**Řešení:**
- Uvolněte RAM (zavřete ostatní aplikace)
- Použijte menší model (YOLOv8n místo YOLOv8m)
- Zmenšete kalibrační dataset (100-200 obrázků)

## Výstupní Soubory
Po dokončení všech kroků budete mít:

```
yolov8m.onnx              # Původní ONNX model
yolov8m.har               # Parsed model (HAR)
yolov8m_optimized.har     # Kvantizovaný model
yolov8m.hef               # Finální spustitelný soubor
```

## Další Kroky
Po vytvoření HEF souboru můžete:

1. **Testovat model:**
```bash
hailortcli run yolov8m.hef
```
2. **Integrovat do aplikace** pomocí HailoRT API (Python/C++)
   
3. **Benchmark výkonu:**
```bash
hailortcli benchmark yolov8m.hef
```

### Analýza HAR souboru
```bash
hailo visualizer yolov8m.har
```

### Kontrola HEF souboru
```bash
hailortcli scan
```

## Odkazy
- [Hailo Documentation](https://hailo.ai/developer-zone/documentation/)
- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)
- [YOLOv8 Ultralytics](https://docs.ultralytics.com/)
