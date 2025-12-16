#~~~{"variant":"standard","title":"YOLO pipeline s video výstupem a CSV logem","id":"72157"}
import cv2 as cv
import numpy as np
import csv
from datetime import datetime
from typing import Optional, List, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("[WARNING] Ultralytics YOLO nenalezen. Nainstaluj: pip install ultralytics")


class Detection:
    """Data class pro jeden detekovaný objekt."""
    def __init__(self, bbox: Tuple[int,int,int,int], label: str, confidence: float):
        self.bbox = bbox
        self.label = label
        self.confidence = confidence


class AIModel:
    """Obal pro YOLO model."""
    def __init__(self, model_path: Optional[str] = "yolov8n.pt"):
        if YOLO is None:
            raise ImportError("Ultralytics YOLO není nainstalovaný.")
        self.model = YOLO(model_path)
        print(f"[AIModel] Načten model: {model_path}")

    def process(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Detection]:
        """Zpracuje frame a vrátí seznam detekcí."""
        results = self.model(frame, verbose=False)
        detections = []
        for r in results[0].boxes:
            conf = float(r.conf[0])
            if conf < conf_threshold:
                continue
            bbox = tuple(map(int, r.xyxy[0]))
            label = self.model.names[int(r.cls[0])]
            detections.append(Detection(bbox, label, conf))
        return detections

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Vykreslí bounding boxy a labely na frame."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"{det.label} {det.confidence:.2f}", (x1, y1-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return frame


class AIOutputListener:
    """Listener pro VideoStream, logování a video output."""
    def __init__(self, ai_model: AIModel, output_video: Optional[str] = None, output_csv: Optional[str] = None, show_window: bool = True, fps: int = 30):
        self.ai = ai_model

class AIModel:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.25, img_size=640):
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.model = YOLO(model_path)

    def process(self, frame: np.ndarray):
        results = self.model(frame, imgsz=self.img_size, verbose=False)
        detections = []
        for r in results[0].boxes:
            conf = float(r.conf[0])
            if conf < self.conf_threshold:
                continue
            bbox = tuple(map(int, r.xyxy[0]))
            label = self.model.names[int(r.cls[0])]
            detections.append(Detection(bbox, label, conf))
        return detections
frame_time = datetime.now().isoformat()
writer.writerow([frame_time, frame_id, det.label, det.confidence, *det.bbox])
        self.output_video = output_video
        self.output_csv = output_csv
        self.show_window = show_window
        self.fps = fps
        self.video_writer = None
        self.csv_file = None
        self.csv_writer = None

    def start(self, video_source: str):
        cap = cv.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[ERROR] Nelze otevřít video zdroj: {video_source}")
            return

        if self.output_video:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            self.video_writer = cv.VideoWriter(self.output_video, fourcc, self.fps, (width, height))
            print(f"[INFO] Video výstup bude uložen do: {self.output_video}")

        if self.output_csv:
            self.csv_file = open(self.output_csv, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'frame_id', 'label', 'confidence', 'x1', 'y1', 'x2', 'y2'])
            print(f"[INFO] CSV log bude uložen do: {self.output_csv}")

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.ai.process(frame)
            annotated_frame = self.ai.draw_detections(frame.copy(), detections)

            if self.show_window:
                cv.imshow("Detekce", annotated_frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.video_writer:
                self.video_writer.write(annotated_frame)

            if self.csv_writer:
                for det in detections:
                    frame_time = datetime.now().isoformat()
                    self.csv_writer.writerow([frame_time, frame_id, det.label, det.confidence, *det.bbox])