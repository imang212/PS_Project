import cv2 as cv
import numpy as np
import csv
import os
import time
import json
from datetime import datetime
from typing import Optional, List, Tuple

# Import funkce pro odesílání na server
try:
    from api_integration import post_vehicle_data
except ImportError:
    # Pokud by soubor chyběl, vytvoříme náhradní funkci, aby kód nespadl
    def post_vehicle_data(label, confidence):
        pass

# 1. Definice datové třídy pro výsledky
class Detection:
    def __init__(self, bbox: Tuple[int,int,int,int], label: str, confidence: float):
        self.bbox = bbox
        self.label = label
        self.confidence = confidence

# 2. Definice modelu s filtrem na vozidla
class AIModel:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.mode = "CPU/GPU (Ultralytics)"
        except ImportError:
            self.model = None
            self.mode = "HAILO_PREP"
        
        print(f"[AIModel] Inicializováno v režimu: {self.mode}")

    def process(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None:
            return []
        
        # FILTR: Zajímají nás jen tato vozidla
        allowed_labels = ["car", "truck", "bus", "motorcycle"]
        
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        detections = []
        
        for r in results[0].boxes:
            label = self.model.names[int(r.cls[0])]
            
            # Pokud to není vozidlo, ignorujeme ho
            if label not in allowed_labels:
                continue
                
            conf = float(r.conf[0])
            bbox = tuple(map(int, r.xyxy[0]))
            detections.append(Detection(bbox, label, conf))
            
        return detections

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"{det.label} {det.confidence:.2f}", (x1, y1-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return frame

# 3. Listener zajišťující logování a odesílání přes API
class AIOutputListener:
    def __init__(self, ai_model: AIModel, output_csv: str = "log_detekce.csv", output_json: str = "log_detekce.json"):
        self.ai = ai_model
        self.output_csv = output_csv
        self.output_json = output_json
        self.results_list = []

    def run_demo(self, source: str, duration_seconds: Optional[int] = None):
        cap = cv.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Nelze otevřít zdroj: {source}")
            return

        start_time = time.time()
        
        with open(self.output_csv, mode='w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(['timestamp', 'label', 'conf', 'bbox'])

            print(f"[DEMO] Spouštím detekci. Cíl: API + LOG")
            while cap.isOpened():
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    print(f"[INFO] Limit vypršel.")
                    break

                ret, frame = cap.read()
                if not ret: break

                detections = self.ai.process(frame)
                ts = datetime.now().isoformat()
                
                for d in detections:
                    # 1. Zápis do CSV
                    writer.writerow([ts, d.label, d.confidence, d.bbox])
                    
                    # 2. Uložení pro JSON
                    self.results_list.append({
                        "timestamp": ts,
                        "label": d.label,
                        "confidence": round(d.confidence, 2),
                        "bbox": d.bbox
                    })

                    # 3. NOVINKA: Odeslání live na tvůj server
                    post_vehicle_data(d.label, d.confidence)

                annotated = self.ai.draw_detections(frame, detections)
                cv.imshow("Hailo RPi 5 Demo - Live API", annotated)
                
                if cv.waitKey(1) == ord('q'): break

        with open(self.output_json, 'w', encoding='utf-8') as f_json:
            json.dump(self.results_list, f_json, indent=4)
            print(f"[SUCCESS] JSON uložen a data odeslána na server.")

        cap.release()
        cv.destroyAllWindows()

# 4. Spuštění celého systému
if __name__ == "__main__":
    MODEL_PATH = "yolov8n.pt" 
    SOURCE = 0 # 0 pro webkameru, nebo cesta k videu
    
    model_instance = AIModel(MODEL_PATH)
    listener = AIOutputListener(model_instance)
    listener.run_demo(SOURCE, duration_seconds=None) 