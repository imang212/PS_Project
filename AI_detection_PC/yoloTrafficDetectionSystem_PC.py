import cv2
import numpy as np
import asyncio
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import threading
import time
from ultralytics import YOLO
import torch
from fastapi import WebSocket
from MQTTClient import MQTTPublisher

# PART 1: DETECTION CLASS
class Detection:
    """
    Data class representing a single detected object.
    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        label: Object class name (e.g., "car", "truck", "person")
        confidence: Detection confidence score (0.0 - 1.0)
        track_id: Unique tracking ID assigned by SORT tracker
    """    
    def __init__(self, bbox: Tuple[int,int,int,int], label: str, confidence: float, track_id: int = -1):
        self.bbox = bbox
        self.label = label
        self.confidence = confidence
        self.track_id = track_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary for JSON serialization."""
        return {
            'bbox': list(self.bbox),
            'label': self.label,
            'confidence': round(self.confidence, 3),
            'track_id': self.track_id
        }

# PART 2: YOLO11 TRAFFIC DETECTOR
class TrafficDetector:
    """
    YOLO11-based object detector optimized for PC GPU (CUDA).
    Key features:
    - Filters for traffic-relevant objects only
    - CUDA acceleration when available, falls back to CPU
    - Configurable inference resolution
    - With Bytetrack
    ByteTrack parameters (bytetrack.yaml):
    - track_high_thresh: 0.5   — detections above this are matched first (high confidence)
    - track_low_thresh:  0.1   — detections above this are used in second matching pass
    - new_track_thresh:  0.6   — minimum confidence to start a brand new track
    - track_buffer:      30    — frames to keep a lost track alive (like max_age)
    - match_thresh:      0.8   — IoU threshold for Kalman-predicted bbox matching
    """    
    # Traffic-relevant classes from COCO dataset
    TRAFFIC_CLASSES = {'person', 'car', 'motorcycle', 'bus', 'truck'}
    # Initialize YOLO11 model
    def __init__(self, model_path: str = "yolo11n.pt", tracker_config: str = "bytetrack.yaml", conf_threshold: float = 0.3, img_size: Tuple[int, int] = (1280, 720), device: str = "auto"):
        """
        Initialize YOLO11 detector.
        Args:
            model_path: Path to YOLO11 weights file
            conf_threshold: Minimum confidence threshold (0.3 = 30%)
            img_size: Input image size - MUST match RPi hardware accelerator format
        """
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.tracker_config = tracker_config
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"[TrafficDetector] Using device: {self.device}")
        print(f"[TrafficDetector] Tracker    : {tracker_config}")
        print(f"[TrafficDetector] Loading YOLO11 model: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()  # Fuse conv + BN layers for speed
        # Warm-up inference to compile model / CUDA kernels
        dummy = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        _ = self.model.track(dummy, imgsz=img_size, verbose=False, conf=self.conf_threshold,
            device=self.device, tracker=self.tracker_config, persist=True)
        print(f"[TrafficDetector] Ready  resolution={img_size}  device={self.device}")
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO11 + ByteTrack on a single frame.
        Uses model.track() with persist=True so ByteTrack's internal Kalman filter state (velocity, predicted position) is preserved across calls. Without persist=True every frame would restart tracking from scratch.
        Args:
            frame: BGR image from camera (numpy array)
        Returns:
            List of Detection objects with stable ByteTrack IDs
        """
        # YOLO11 inference with hardware-accelerated resolution
        results = self.model.track(
            frame,
            imgsz=self.img_size,
            verbose=False,
            conf=self.conf_threshold,
            device=self.device,
            tracker=self.tracker_config,  
            persist=True               # keeps track state between calls — critical
        )
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None or boxes.id is None:
                continue
            for box in boxes:
                label = self.model.names[int(box.cls[0])]
                if label not in self.TRAFFIC_CLASSES:
                    continue
                track_id = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(bbox=(x1, y1, x2, y2), label=label, confidence=float(box.conf[0]), track_id=track_id))
        return detections
    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection], show_tracking: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        Color coding:
        - Green: Vehicles (car, truck, bus, motorcycle)
        - Blue: Pedestrians
        """
        # Create a copy of the frame to draw on
        annotated = frame.copy()
        # Draw each detection
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0) if det.label in ['car', 'truck', 'bus', 'motorcycle'] else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det.label} {det.confidence:.2f}"
            if show_tracking and det.track_id != -1:
                label_text += f" ID:{det.track_id}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return annotated

# PART 4: TRAFFIC COUNTER USING SIMPLE TRACKER
class TrafficCounter:
    """
    Traffic counter using simple IoU-based tracking.
    Counts unique objects that pass through the scene.
    """
    def __init__(self):
        """
        Initialize traffic counter.
        Args:
            max_age: Max frames to keep track alive without detection
            min_hits: Min detections before track is confirmed
        """
        self.object_counts: Dict[str, int] = defaultdict(int)
        self.tracked_ids: set = set()
        self.total_count: int = 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Register new track_ids and increment per-class counters."""
        for det in detections:
            if det.track_id != -1 and det.track_id not in self.tracked_ids:
                self.tracked_ids.add(det.track_id)
                self.object_counts[det.label] += 1
                self.total_count += 1
        return detections
    
    def get_counts(self) -> Dict[str, int]:
        """Get count of each object type."""
        return dict(self.object_counts)
    
    def get_total_count(self) -> int:
        """Get total count of all tracked objects."""
        return self.total_count

# PART 5: VIDEO STREAMER WITH RASPBERRY PI OPTIMIZATION
class VideoStreamer:
    """
    Threaded video capture for RTSP streams or local cameras.
    Usage examples:
        RTSP camera : VideoStreamer(source="rtsp://user:pass@192.168.1.100:554/stream1")
        Local webcam: VideoStreamer(source=0)
        Video file  : VideoStreamer(source="/path/to/video.mp4")
    """
    def __init__(self, source: Any = 0, resolution: Tuple[int,int] = (1280, 720), fps: int = 25):
        """
        Args:
            source:     RTSP URL (str), device index (int), or video file path (str)
            resolution: Desired output resolution (width, height).
                        Frames are resized to this after capture.
            fps:        Target frames-per-second (used only to throttle reading speed)
        """
        self.source = source
        self.resolution = resolution
        self.target_fps = fps
        self.frame_time = 1.0 / fps

        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.camera_started = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        print(f"[VideoStreamer] Initialized with resolution: {resolution}")
    
    def _build_capture(self) -> cv2.VideoCapture:
        """Open VideoCapture with RTSP-friendly settings."""
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp"):
            # Use FFMPEG backend for RTSP; set TCP transport to avoid UDP packet loss
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # Minimal buffer – always fresh frame
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        return cap

    def _capture_loop(self):
        """Background capture loop."""
        print("[VideoStreamer] Opening video source...")
        cap = self._build_capture()
        if not cap.isOpened():
            print(f"[VideoStreamer] ERROR: Cannot open source: {self.source}")
            return
        self.camera_started = True
        print("[VideoStreamer] Source opened, capturing frames...")
        reconnect_delay = 2  # seconds to wait before reconnect attempt
        while self.running:
            start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[VideoStreamer] Frame read failed – attempting reconnect in {reconnect_delay}s…")
                cap.release()
                time.sleep(reconnect_delay)
                cap = self._build_capture()
                if not cap.isOpened():
                    print("[VideoStreamer] Reconnect failed, retrying…")
                continue
            # Resize if the captured size differs from target
            h, w = frame.shape[:2]
            if (w, h) != self.resolution:
                frame = cv2.resize(frame, self.resolution)
            with self.lock:
                self.frame = frame
            # Throttle to target FPS
            elapsed = time.time() - start
            sleep_time = max(0.0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        cap.release()
        print("[VideoStreamer] Capture loop exited.")

    def start(self):
        """Start the capture thread (non-blocking)."""
        if self.thread is not None and self.thread.is_alive():
            print("[VideoStreamer] Already running"); return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("[VideoStreamer] Capture thread started")

    def stop(self):
        """Stop the capture thread gracefully."""
        print("[VideoStreamer] Stopping...")
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5)
        print("[VideoStreamer] Stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest captured frame (thread-safe copy)."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def is_ready(self) -> bool:
        """True once the camera has started and at least one frame is available."""
        with self.lock:
            return self.camera_started and self.frame is not None

# PART 6: WEBSOCKET MANAGER FOR REAL-TIME STREAMING
class WebSocketManager:
    """
    Manages WebSocket connections for real-time streaming to frontend.    
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
    async def broadcast(self, message: Dict[str, Any]):
        disconnected = []
        for connection in self.active_connections:
            try: 
                await connection.send_json(message)
            except Exception as e:
                print(f"[WebSocket] Error sending to client: {e}")
                disconnected.append(connection)        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

# PART 7: MAIN TRAFFIC MONITORING PIPELINE
class TrafficMonitoringPipeline:
    """
    Main pipeline integrating detection, tracking, WebSocket streaming, and MQTT publishing.
    Architecture:
        VideoStreamer   – captures RTSP / local camera frames in a background thread
        TrafficDetector – runs YOLO11 on GPU
        TrafficCounter  – IoU tracker assigns persistent IDs
        WebSocketManager– streams detection JSON to connected browser clients
        MQTTPublisher   – publishes detections to the MQTT broker
    """
    def __init__(self, 
                 model_path: str = "yolo11n.pt", 
                 tracker_config: str = "bytetrack.yaml",
                 video_source: Any = 0, 
                 resolution: Tuple[int,int] = (1280, 720), 
                 fps: int = 25, 
                 device: str = "auto", 
                 confidence_threshold: float = 0.25,
                 websocket_enabled: bool = True,
                 mqtt_enabled: bool = True
                ):
        """
        Initialize complete pipeline.
        Args:
            model_path: Path to YOLO11 model weights
            video_source: Camera index or video file path
            db_path: SQLite database path
            resolution: Camera resolution (MUST be RPi hardware-accelerated)
            fps: Target frames per second
        """
        # Initialize components
        self.detector = TrafficDetector(model_path, img_size=resolution, device=device, conf_threshold=confidence_threshold ,tracker_config=tracker_config)
        self.counter = TrafficCounter()
        self.streamer = VideoStreamer(source=video_source, resolution=resolution, fps=fps)
        self.ws_manager = None
        self.websocket_enabled = websocket_enabled
        if websocket_enabled:
            self.ws_manager = WebSocketManager()
        self.mqtt_enabled = mqtt_enabled
        self.mqtt = None
        if self.mqtt_enabled:
            self.mqtt = MQTTPublisher(broker_host="mqtt.portabo.cz", broker_port=8883, topic="patrik/traffic_detection", client_id="PC_traffic_detection", username="videoanalyza", password="phdA9ZNW1vfkXdJkhhbP")
        self.mqtt_publish_every_n_frames = 1
        # Pipeline state
        self.running = False
        self.fps = fps
        self.resolution = resolution
        self.frame_id = 0
        self.current_frame: Optional[np.ndarray] = None
        self.current_detections: List[Detection] = []
        # Performance metrics
        self.fps_actual = 0.0
        self.last_fps_update = time.time()
        self.frame_times: List[float] = []
        print("[Pipeline] Initialization complete")
    
    def _publish_mqtt(self, timestamp: str, tracked_detections: List[Detection]):
        """Build the MQTT payload and publish (non-blocking – called from async context via to_thread)."""
        if not self.mqtt or not self.mqtt.connected:
            return
        counts = self.counter.get_counts()
        payload = {
            "type": "detections",
            "timestamp": timestamp,
            "frame_id": self.frame_id,
            "detections": [det.to_dict() for det in tracked_detections],
            "statistics": {
                "total_count": self.counter.get_total_count(),
                "counts_by_type": counts,
                "fps": round(self.fps_actual, 1),
            },
        }
        self.mqtt.publish(payload)

    async def process_frame(self):
        """
        Process single frame through complete pipeline.
        Pipeline flow:
        1. Get frame from streamer
        2. Run YOLO11 detection
        3. Update SORT tracker
        4. Prepare data for WebSocket
        5. Broadcast to frontend
        6. Store in database
        """
        start_time = time.time()
        # Step 1: Get frame
        #print("[Pipeline] Waiting for frame...")
        frame = await asyncio.to_thread(self.streamer.get_frame)
        if frame is None:
            await asyncio.sleep(0.01)
            print("[Pipeline] No frame available yet")
            return
        self.current_frame = frame
        # Step 2: Detect
        detections = await asyncio.to_thread(self.detector.detect, frame)
        # Step 3: Track
        tracked = self.counter.update(detections)
        self.current_detections = tracked
        # Step 4: Prepare data for WebSocket and database
        timestamp = datetime.now().isoformat()
        total_count = self.counter.get_total_count()
        counts = self.counter.get_counts()
        # Prepare detection data for WebSocket and database
        detection_data_list = []
        for det in tracked_detections:
            detection_dict = {
                'timestamp': timestamp,
                'frame_id': self.frame_id,
                'label': det.label,
                'confidence': det.confidence,
                'x1': det.bbox[0],
                'y1': det.bbox[1],
                'x2': det.bbox[2],
                'y2': det.bbox[3],
                'track_id': det.track_id,
                'total_count': total_count
            }
            detection_data_list.append(detection_dict)
        # Step 5: Broadcast to WebSocket clients
        if self.websocket_enabled:
            ws_message = {
                'type': 'detections',
                'timestamp': timestamp,
                'frame_id': self.frame_id,
                'detections': [det.to_dict() for det in tracked_detections],
                'statistics': {
                    'total_count': total_count,
                    'counts_by_type': counts,
                    'fps': round(self.fps_actual, 1)
                }
            }
            # send broadcast message of each frame to all connected clients for real-time updates 
            await self.ws_manager.broadcast(ws_message)
        # Step 6 – MQTT publish (rate-limited by mqtt_publish_every_n_frames)
        if self.mqtt_enabled and (self.frame_id % self.mqtt_publish_every_n_frames == 0):
            await asyncio.to_thread(self._publish_mqtt, timestamp, tracked_detections)

        # Update metrics (id, time, FPS)
        self.frame_id += 1
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)
        # Calculate FPS every second
        if time.time() - self.last_fps_update > 1.0:
            if self.frame_times:
                avg_time = sum(self.frame_times) / len(self.frame_times)
                self.fps_actual = 1.0 / avg_time if avg_time > 0 else 0
                self.frame_times = []
            self.last_fps_update = time.time()
    
    async def start(self):
        """Start the processing pipeline (runs until stopped)."""
        print("[Pipeline] Starting traffic monitoring...")
        self.streamer.start()
        print("[Pipeline] Waiting for camera/stream to initialise...")
        max_wait = 10; waited = 0
        while not self.streamer.is_ready() and waited < max_wait:
            await asyncio.sleep(0.5)
            waited += 0.5
        if not self.streamer.is_ready():
            print("[Pipeline] ERROR: Camera/stream failed to initialise!")
            self.streamer.stop()
            return
        print("[Pipeline] Stream ready, starting detection...")
        self.running = True
        try:
            while self.running:
                await self.process_frame()
        except Exception as e:
            print(f"[Pipeline] Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Stop pipeline and cleanup resources."""
        print("[Pipeline] Stopping...")
        self.running = False
        self.streamer.stop()
        if self.mqtt:
            self.mqtt.disconnect()
        print("[Pipeline] Stopped")
    
    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Return current frame with bounding boxes and stats overlay."""
        if self.current_frame is None:
            return None
        annotated = self.detector.draw_detections(self.current_frame, self.current_detections, show_tracking=True)
        counts = self.counter.get_counts()
        y_offset = 30
        cv2.putText(annotated, f"Total: {self.counter.get_total_count()}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        for label, count in counts.items():
            cv2.putText(annotated, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        cv2.putText(annotated, f"FPS: {self.fps_actual:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated

# bytetrack config
def create_bytetrack_config(output_path: str = "bytetrack.yaml",track_high_thresh: float = 0.5,track_low_thresh: float = 0.1,new_track_thresh: float = 0.6,track_buffer: int = 30,match_thresh: float = 0.8, fuse_score: bool = True) -> str:
    """
    Write a custom bytetrack.yaml so you can tune without editing
    Ultralytics internals. Returns the output path for convenience.
    Parameters explained:
        track_high_thresh  — detections above this confidence are matched in the first (primary) association pass. Raise this if you have many false positives.
        track_low_thresh   — detections between low and high thresh are used in the second pass to recover lost tracks. Lower this to recover more detections at cost of noise.
        new_track_thresh   — a brand-new track is only created if detection confidence exceeds this. Raise to kill ghost tracks.
        track_buffer       — frames a lost track is kept alive before deletion. Increase for slow-moving or temporarily occluded vehicles.
        match_thresh       — IoU distance threshold for Kalman-predicted bbox matching. Lower = stricter spatial matching.
    """
    config = (
        "# ByteTrack configuration\n"
        "# Generated by TrafficDetectionSystem\n\n"
        "tracker_type: bytetrack\n\n"
        f"track_high_thresh: {track_high_thresh}\n"
        f"track_low_thresh: {track_low_thresh}\n"
        f"new_track_thresh: {new_track_thresh}\n"
        f"track_buffer: {track_buffer}\n"
        f"match_thresh: {match_thresh}\n"
        f"fuse_score: {fuse_score}\n"
    )
    with open(output_path, "w") as f:
        f.write(config)
    print(f"[Config] ByteTrack config saved -> {output_path}")
    return output_path

# PART 8: LOCAL TESTING WITHOUT FASTAPI SERVER ON DEVICE
async def main():
    """
    Standalone test – displays an OpenCV window with live annotated frames.
    Press 'q' to quit.
    """
    print("=== YOLO11 Traffic Detection Test ===")
    print("Press 'q' to exit")
    print()
    # Configuration for local testing (adjust resolution for your camera)
    RTSP_URL    = "rtsp://admin:Dcuk.123456@192.168.37.99/Stream"  # ← your camera URL
    # VIDEO_SOURCE = 0
    RESOLUTION = (1280, 736)
    VIDEO_SOURCE = RTSP_URL
    VIDEO_SOURCE = "/home/patrik/videos/test_outputx264-202601091552_000.mp4"
    # Bytetrack
    CUSTOM_BYTETRACK = True
    TRACKER_CONFIG = create_bytetrack_config(
        output_path="bytetrack.yaml",
        track_high_thresh=0.35,   # lower if losing detections on real camera
        track_low_thresh=0.05,    # lower to recover more occluded vehicles
        new_track_thresh=0.45,    # raise if getting ghost/duplicate tracks
        track_buffer=60,          # raise for slow traffic or long occlusions
        match_thresh=0.7,         # lower slightly if IDs keep switching
        fuse_score=True           # multiply confidence into matching distance
    ) if CUSTOM_BYTETRACK else "bytetrack.yaml"
    # Create pipeline instance
    pipeline = TrafficMonitoringPipeline(
        model_path="yolov8l.pt", 
        tracker_config=TRACKER_CONFIG,
        video_source=VIDEO_SOURCE, 
        resolution=RESOLUTION, 
        fps=15, 
        device="auto", 
        websocket_enabled=False, 
        mqtt_enabled=False    
    )
    # Start video streamer thread
    print("[Test] Starting video streamer...")
    pipeline.streamer.start()
    print("Waiting for camera initialization...")
    max_wait = 10; waited = 0
    while not pipeline.streamer.is_ready() and waited < max_wait:
        await asyncio.sleep(0.5)
        waited += 0.5
    if not pipeline.streamer.is_ready():
        print("[Test] ERROR: Stream failed to initialise. Check your RTSP URL / camera.")
        pipeline.stop()
        return
    print("[Pipeline] Detection running...")
    frame_id = 0
    try:
        while True:
            start_time = time.time()
            # Step 1: Get frame from camera
            frame = await asyncio.to_thread(pipeline.streamer.get_frame)
            if frame is None:
                await asyncio.sleep(0.01)
                return
            # Step 2: Detect
            detections = await asyncio.to_thread(pipeline.detector.detect, frame)
            # Step: 3 Update object tracker
            tracked = pipeline.counter.update(detections)
            pipeline.current_frame = frame
            pipeline.current_detections = tracked
            timestamp = datetime.now().isoformat()
            # MQTT publish
            if pipeline.mqtt_enabled and (pipeline.frame_id % pipeline.mqtt_publish_every_n_frames == 0):
                await asyncio.to_thread(pipeline._publish_mqtt, timestamp, tracked)
            # Annotated display
            annotated = pipeline.get_annotated_frame()
            if annotated is not None:
                cv2.imshow("Traffic Detection", annotated)
            # FPS bookkeeping
            pipeline.frame_id += 1
            elapsed = time.time() - start_time
            pipeline.frame_times.append(elapsed)
            now = time.time()
            if now - pipeline.last_fps_update > 1.0:
                if pipeline.frame_times:
                    avg = sum(pipeline.frame_times) / len(pipeline.frame_times)
                    pipeline.fps_actual = 1.0 / avg if avg > 0 else 0
                    pipeline.frame_times = []
                pipeline.last_fps_update = now
                print(f"[Test] Frame {pipeline.frame_id:5d}  FPS: {pipeline.fps_actual:.1f}  "
                      f"Tracked: {len(tracked)}  Total: {pipeline.counter.get_total_count()}")
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Test] 'q' pressed – exiting…")
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        pipeline.stop()
        cv2.destroyAllWindows()
        #cv2.destroyAllWindows()
        print("\nDone!")
        # Print final statistics
        print("\n=== FINAL STATISTICS ===")
        print(f"Total objects detected: {pipeline.counter.get_total_count()}")
        print(f"Count by type: {pipeline.counter.get_counts()}")
        print(f"Total frames processed: {pipeline.frame_id}")

if __name__ == "__main__": asyncio.run(main())