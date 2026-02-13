import sys
import os
import cv2 as cv
import time
from ultralytics import YOLO

# --- AUTOMATICKÉ CESTY K MODELU ---
# Tato cesta míří do kořenové složky PS_Project, kde máš uložený model
model_filename = 'yolov8n.pt'
full_model_path = os.path.abspath(os.path.join(project_root, model_filename))

# Kontrolní výpis pro tebe (v prezentaci to vypadá profesionálně)
if os.path.exists(full_model_path):
    print(f"OK: Model nalezen v: {full_model_path}")
else:
    print(f"POZOR: Model v '{full_model_path}' nebyl nalezen, použije se výchozí cesta.")

# 1. API integrace
try:
    from api_integration import post_vehicle_data
except ImportError:
    def post_vehicle_data(l, c): pass

# 2. Servo (ošetření lgpio chyby na Windows)
try:
    from api.ServoControl import ServoControl
except (ImportError, ModuleNotFoundError):
    class ServoControl:
        def __init__(self): print("SERVO: Inicializace (DEMO - lgpio není na PC)")
        def open_gate(self): print("SERVO: Otevírám závoru (DEMO)")

# 3. Databáze
try:
    from api.DatabaseManagerPostgre import DatabaseManager
except (ImportError, ModuleNotFoundError):
    class DatabaseManager:
        def __init__(self): print("DB: Připojení (DEMO)")
        def insert_detection(self, l, c): print(f"DB: Zápis {l} (DEMO)")

class AIOutputListener:
    def __init__(self, model):
        self.model = model
        self.servo = ServoControl() 
        self.db = DatabaseManager()
        self.car_count = 0  # Celkový počet aut
        self.line_y = 400   # Výška čáry v obraze (uprav podle potřeby)
        self.counted_ids = set()

    def run_demo(self, source=0):
        cap = cv.VideoCapture(source)
        print("--- Spouštím SYSTÉM S POČÍTADLEM ---")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            height, width, _ = frame.shape
            self.line_y = int(height * 0.7) # Čára v 70% výšky obrazu

            # 1. Detekce (používáme tracking pro počítání)
            results = self.model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
            
            # Kreslení čáry
            cv.line(frame, (0, self.line_y), (width, self.line_y), (0, 0, 255), 3)
            cv.putText(frame, "DETECTION LINE", (10, self.line_y - 10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, obj_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    center_y = int((y1 + y2) / 2)
                    
                    # 2. Logika počítání: Pokud střed auta protne čáru
                    if center_y > self.line_y and obj_id not in self.counted_ids:
                        self.car_count += 1
                        self.counted_ids.add(obj_id)
                        
                        # Akce při průjezdu (Servo, DB, API)
                        self.servo.open_gate()
                        self.db.insert_detection("car", 0.9)
                        post_vehicle_data("car", 0.9)

                    # Kreslení boxu
                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv.putText(frame, f"ID: {obj_id}", (int(x1), int(y1)-5), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 3. Statistiky na obrazovce
            cv.rectangle(frame, (0, 0), (250, 80), (0, 0, 0), -1)
            cv.putText(frame, f"TOTAL CARS: {self.car_count}", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.putText(frame, "STATUS: ONLINE", (10, 60), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv.imshow("Smart Traffic AI", frame)
            if cv.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    listener = AIOutputListener(model)
    listener.run_demo(0)