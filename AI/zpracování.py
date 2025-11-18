import kagglehub
import os
from PIL import Image

# 1️⃣ Stáhnout dataset přes KaggleHub
dataset_path = kagglehub.dataset_download("farzadnekouei/top-view-vehicle-detection-image-dataset")

print("Path to dataset files:", dataset_path)

# 2️⃣ Výpis několika souborů ve složce datasetu
print("Prvních 10 souborů v datasetu:")
for file in os.listdir(dataset_path)[:10]:
    print(file)

# 3️⃣ Výpis složek a souborů
print("Složky v datasetu:")
print(os.listdir(dataset_path))

# 4️⃣ Najít první obrázek
first_image = None
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            first_image = os.path.join(root, file)
            break
    if first_image:
        break

if first_image:
    print("První obrázek:", first_image)

    # 5️⃣ Zobrazit obrázek
    img = Image.open(first_image)
    img.show()
else:
    print("Žádný obrázek nebyl nalezen v datasetu.")



import cv2
import numpy as np
import os
import urllib.request
import time
from datetime import datetime

# ----------------------------
# YOLO setup
# ----------------------------
YOLO_DIR = "yolo_files"
os.makedirs(YOLO_DIR, exist_ok=True)

weights = os.path.join(YOLO_DIR, "yolov3.weights")
config = os.path.join(YOLO_DIR, "yolov3.cfg")
labels_path = os.path.join(YOLO_DIR, "coco.names")

# Stáhnout soubory pokud neexistují
if not os.path.exists(weights):
    urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", weights)
if not os.path.exists(config):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", config)
if not os.path.exists(labels_path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", labels_path)

with open(labels_path) as f:
    labels = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(weights, config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

# ----------------------------
# Funkce detekce
# ----------------------------
def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    people_count, vehicle_count = 0, 0

    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                w = int(det[2]*width)
                h = int(det[3]*height)
                x = int(det[0]*width - w/2)
                y = int(det[1]*height - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = labels[class_ids[i]]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame, label, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2)
            if label == "person":
                people_count += 1
            elif label in ["car","bus","truck","motorbike"]:
                vehicle_count += 1

    return frame, people_count, vehicle_count

# ----------------------------
# Hlavní smyčka pro webkameru
# ----------------------------
cap = cv2.VideoCapture(0)
print("[INFO] Spouštím detekci z webkamery. Stiskni ESC pro ukončení.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        start = time.time()
        frame, people, vehicles = detect_objects(frame)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"{timestamp}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        cv2.putText(frame, f"Lidi: {people} | Vozidel: {vehicles}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

        cv2.imshow("Webkamera Detekce", frame)

        # čekat 2 sekundy mezi detekcemi
        elapsed = time.time() - start
        if elapsed < 2:
            time.sleep(2 - elapsed)

        if cv2.waitKey(1) == 27:
            print("[INFO] Ukončeno uživatelem.")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import urllib.request
import time
from datetime import datetime

# ----------------------------
# YOLO setup
# ----------------------------
YOLO_DIR = "yolo_files"
os.makedirs(YOLO_DIR, exist_ok=True)

weights = os.path.join(YOLO_DIR, "yolov3.weights")
config = os.path.join(YOLO_DIR, "yolov3.cfg")
labels_path = os.path.join(YOLO_DIR, "coco.names")

if not os.path.exists(weights):
    print("[INFO] Stahuji YOLO weights (poprvé může chvíli trvat)...")
    urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", weights)
if not os.path.exists(config):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", config)
if not os.path.exists(labels_path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", labels_path)

with open(labels_path) as f:
    labels = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(weights, config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

# ----------------------------
# Funkce detekce
# ----------------------------
def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    people_count, vehicle_count = 0, 0

    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                w = int(det[2]*width)
                h = int(det[3]*height)
                x = int(det[0]*width - w/2)
                y = int(det[1]*height - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = labels[class_ids[i]]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            if label=="person":
                people_count += 1
            elif label in ["car","bus","truck","motorbike"]:
                vehicle_count += 1

    return frame, people_count, vehicle_count

# ----------------------------
# Hlavní smyčka pro webkameru
# ----------------------------
cap = cv2.VideoCapture(0)
print("[INFO] Spouštím detekci z webkamery. Stiskni ESC pro ukončení.")

frame_id = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        start = time.time()
        frame, people, vehicles = detect_objects(frame)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame,f"{timestamp}",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.putText(frame,f"Lidi: {people} | Vozidel: {vehicles}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        cv2.imshow("Webkamera Detekce", frame)

        frame_id += 1
        print(f"[INFO] Zpracováno {frame_id} snímků | Lidi: {people} | Vozidel: {vehicles}")

        elapsed = time.time() - start
        if elapsed < 2:
            time.sleep(2 - elapsed)

        if cv2.waitKey(1) == 27:
            print(f"[STOP] Ukončeno uživatelem po {frame_id} snímcích.")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detekce z webkamery dokončena.")


import os
import cv2
import numpy as np
import urllib.request
import yt_dlp
from datetime import datetime
from threading import Event

# =====================
# === 1️⃣ YOLO SETUP ===
# =====================
YOLO_DIR = "yolo_files"
os.makedirs(YOLO_DIR, exist_ok=True)

weights = os.path.join(YOLO_DIR, "yolov3.weights")
config = os.path.join(YOLO_DIR, "yolov3.cfg")
labels_path = os.path.join(YOLO_DIR, "coco.names")

# Stažení YOLO souborů jen pokud neexistují
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"[INFO] Stahuji {filename} ...")
        urllib.request.urlretrieve(url, filename)
        print(f"[INFO] {filename} stažen.")
    else:
        print(f"[INFO] {filename} už existuje, přeskočeno.")

download_file("https://pjreddie.com/media/files/yolov3.weights", weights)
download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", config)
download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", labels_path)

with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(weights, config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

# =====================
# === 2️⃣ DETEKCE FUNKCE ===
# =====================
def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    people_count = 0
    vehicle_count = 0

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(detection[0]*width - w/2)
                y = int(detection[1]*height - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = labels[class_ids[i]]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if label in ["person"]:
                people_count += 1
            if label in ["car","bus","truck","motorbike"]:
                vehicle_count += 1
    return frame, people_count, vehicle_count

# =====================
# === 3️⃣ YOUTUBE STREAM ===
# =====================
YOUTUBE_URL = "https://www.youtube.com/watch?v=Xs_872Hah3o"
MAX_FRAMES = 50
MAX_DURATION = 120  # sec
OUTPUT_DIR = "output_youtube"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_youtube_stream(url):
    ydl_opts = {"format":"best","quiet":True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]

def process_youtube(stop_event):
    try:
        stream_url = get_youtube_stream(YOUTUBE_URL)
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise RuntimeError("Nelze otevřít YouTube stream.")
        print("[INFO] YouTube stream připraven.")

        frame_id = 0
        start_time = datetime.now().timestamp()

        while not stop_event.is_set():
            elapsed = datetime.now().timestamp() - start_time
            if frame_id >= MAX_FRAMES or elapsed > MAX_DURATION:
                print(f"[STOP] Ukončuji po {frame_id} snímcích / {int(elapsed)} s.")
                break

            ret, frame = cap.read()
            if not ret:
                cv2.waitKey(1)
                continue

            frame, people, vehicles = detect_objects(frame)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"{timestamp}", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(frame, f"Lidi: {people} | Vozidel: {vehicles}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow("YouTube Detekce", frame)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.jpg"), frame)
            frame_id += 1

            print(f"[INFO] Zpracováno {frame_id} snímků | Lidi: {people} | Vozidel: {vehicles}")

            if cv2.waitKey(1) == 27:
                stop_event.set()
                break

    except KeyboardInterrupt:
        stop_event.set()
        print("[INFO] Přerušeno uživatelem (KeyboardInterrupt).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] YouTube detekce ukončena.")

# =====================
# === 4️⃣ SPUŠTĚNÍ ===
# =====================
if __name__ == "__main__":
    stop_event = Event()
    process_youtube(stop_event)

import cv2
import numpy as np
import os
import urllib.request
import time
from datetime import datetime
import yt_dlp

# ----------------------------
# YOLO setup
# ----------------------------
YOLO_DIR = "yolo_files"
os.makedirs(YOLO_DIR, exist_ok=True)

weights = os.path.join(YOLO_DIR, "yolov3.weights")
config = os.path.join(YOLO_DIR, "yolov3.cfg")
labels_path = os.path.join(YOLO_DIR, "coco.names")

if not os.path.exists(weights):
    print("[INFO] Stahuji YOLO weights (poprvé může chvíli trvat)...")
    urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", weights)
if not os.path.exists(config):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", config)
if not os.path.exists(labels_path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", labels_path)

with open(labels_path) as f:
    labels = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(weights, config)
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(labels),3))

# ----------------------------
# Funkce detekce
# ----------------------------
def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    people_count, vehicle_count = 0, 0

    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                w = int(det[2]*width)
                h = int(det[3]*height)
                x = int(det[0]*width - w/2)
                y = int(det[1]*height - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = labels[class_ids[i]]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            if label=="person":
                people_count += 1
            elif label in ["car","bus","truck","motorbike"]:
                vehicle_count += 1

    return frame, people_count, vehicle_count

# ----------------------------
# YouTube stream
# ----------------------------
YOUTUBE_URL = "https://www.youtube.com/watch?v=Xs_872Hah3o"

def get_stream(url):
    ydl_opts = {"format":"best", "quiet":True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]

try:
    stream_url = get_stream(YOUTUBE_URL)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError("Nelze otevřít YouTube stream.")
    print("[INFO] YouTube stream připraven.")
except Exception as e:
    print(f"[WARN] Nelze načíst YouTube: {e}")
    cap = None

if cap:
    MAX_FRAMES = 50
    frame_id = 0
    print(f"[INFO] Zpracovávám YouTube stream (max {MAX_FRAMES} snímků).")
    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue

            frame, people, vehicles = detect_objects(frame)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame,f"{timestamp}",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.putText(frame,f"Lidi: {people} | Vozidel: {vehicles}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

            cv2.imshow("YouTube Detection", frame)

            frame_id += 1
            print(f"[INFO] Zpracováno {frame_id} snímků | Lidi: {people} | Vozidel: {vehicles}")

            elapsed = time.time() - start
            if elapsed < 5:
                time.sleep(5 - elapsed)

            if cv2.waitKey(1) == 27 or frame_id >= MAX_FRAMES:
                print(f"[STOP] Ukončuji po {frame_id} snímcích / {int(time.time()-start)} s.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Detekce YouTube dokončena.")

