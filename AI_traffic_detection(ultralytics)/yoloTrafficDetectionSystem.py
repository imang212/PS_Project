"""
YOLO11 Traffic Detection System with FastAPI Backend
Features:
- Real-time WebSocket streaming for frontend
- REST API endpoints for data retrieval
- Optimized for Raspberry Pi hardware acceleration (1536x864)
"""
import cv2
import numpy as np
import asyncio
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import threading
from picamera2 import Picamera2
import time
from ultralytics import YOLO
from fastapi import WebSocket

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
    YOLO11-based object detector optimized for Raspberry Pi.    
    Key features:
    - Filters for traffic-relevant objects only
    - Uses hardware-accelerated resolution (1536x864)
    - Optimized inference settings for real-time performance
    """    
    # Traffic-relevant classes from COCO dataset
    TRAFFIC_CLASSES = {'person', 'car', 'motorcycle', 'bus', 'truck'}
    # Initialize YOLO11 model
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.3, img_size: Tuple[int, int] = (1536, 864)):
        """
        Initialize YOLO11 detector.
        Args:
            model_path: Path to YOLO11 weights file
            conf_threshold: Minimum confidence threshold (0.3 = 30%)
            img_size: Input image size - MUST match RPi hardware accelerator format
        """
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        print(f"[TrafficDetector] Loading YOLO11 model: {model_path}")
        self.model = YOLO(model_path)
        #self.model.export(format='onnx')  # Export to ONNX for potential optimizations
        self.model.fuse()  # Fuse conv and batch norm layers for speed
        # Warm-up inference to compile model
        dummy_frame = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8) 
        # Warm-up with a dummy frame
        _ = self.model(dummy_frame, imgsz=img_size, verbose=False)
        print(f"[TrafficDetector] Model loaded. Classes: {len(self.model.names)}")
        print(f"[TrafficDetector] Using hardware-accelerated resolution: {img_size}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame using YOLO11.        
        Process:
        1. Run YOLO11 inference on frame
        2. Extract bounding boxes, classes and confidence scores
        3. Filter by confidence threshold and traffic classes
        4. Create Detection objects
        Args:
            frame: Input BGR image from camera (numpy array)
        Returns:
            List of Detection objects
        """
        # YOLO11 inference with hardware-accelerated resolution
        results = self.model(
            frame, 
            imgsz=self.img_size,  # Use RPi hardware-accelerated size
            verbose=False, 
            conf=self.conf_threshold,
            device='cpu'  # RPi doesn't have CUDA
        )
        # Parse results into Detection objects
        detections: List[Detection] = []
        # Process each detection from YOLO results
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None: continue
            for box in boxes:
                # Extract detection data
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                # Filter: only traffic-relevant objects
                if label not in self.TRAFFIC_CLASSES: continue
                # Get bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(bbox=(x1, y1, x2, y2), label=label, confidence=conf))

        return detections
    
    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection], show_tracking: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame for visualization.
        Color coding:
        - Green: Vehicles (car, truck, bus, motorcycle)
        - Blue: Pedestrians and cyclists
        Args:
            frame: Input frame to draw on
            detections: List of Detection objects
            show_tracking: Whether to display track_id
        Returns: Annotated frame with drawn detections
        """
        # Create a copy of the frame to draw on
        annotated = frame.copy()
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Color based on object type
            if det.label in ['car', 'truck', 'bus', 'motorcycle']: color = (0, 255, 0)  # Green for vehicles
            else: color = (255, 0, 0)  # Blue for pedestrians
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # Prepare label text
            label_text = f"{det.label} {det.confidence:.2f}"
            if show_tracking and det.track_id != -1:
                label_text += f" ID:{det.track_id}"
            # Draw text background for readability
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-text_height-4), (x1+text_width, y1), color, -1)
            # Draw label text
            cv2.putText(annotated, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return annotated

# PART 3: SIMPLE IOU-BASED TRACKER
class SimpleTracker:
    """
    Simple object tracker using IoU (Intersection over Union) matching.    
    Algorithm:
        1. For each new detection, calculate IoU with all existing tracks
        2. Match detection to track with highest IoU (if above threshold)
        3. Create new track for unmatched detections
        4. Remove tracks that haven't been updated for max_age frames
    """
    class Track:
        """Represents a single tracked object."""
        def __init__(self, track_id: int, detection: Detection):
            self.track_id = track_id
            self.bbox = detection.bbox
            self.label = detection.label
            self.confidence = detection.confidence
            self.age = 0  # Frames since last update
            self.hits = 1  # Number of consecutive detections
            
        def update(self, detection: Detection):
            """Update track with new detection."""
            self.bbox = detection.bbox
            self.confidence = detection.confidence
            self.age = 0
            self.hits += 1
            
        def predict(self):
            """Increment age (no motion prediction in simple version)."""
            self.age += 1
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 3):
        """
        Initialize simple tracker.
        Args:
            iou_threshold: Minimum IoU for matching detection to track
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before track is confirmed
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[SimpleTracker.Track] = []
        self.next_id = 1
    
    @staticmethod
    def calculate_iou(bbox1: Tuple[int,int,int,int], bbox2: Tuple[int,int,int,int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        Args:
            bbox1, bbox2: Bounding boxes in format (x1, y1, x2, y2)
        Returns: IoU score (0.0 - 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        # No overlap
        if x2_i < x1_i or y2_i < y1_i: return 0.0
        # Calculate intersection area
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        # return IoU
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Update tracker with new detections.
        Args:
            detections: List of Detection objects from current frame
        Returns: List of Detection objects with assigned track_ids
        """
        # Predict existing tracks (just increment age)
        for track in self.tracks: track.predict()
        # Match detections to existing tracks
        matched_detections = []
        unmatched_detections = list(detections)
        matched_tracks = set()        
        # For each detection, find best matching track
        for detection in detections:
            best_iou = self.iou_threshold
            best_track = None
            for track in self.tracks:
                # Only match same object types
                if track.label != detection.label: continue
                # Skip already matched tracks
                if track.track_id in matched_tracks: continue
                # Calculate IoU
                iou = self.calculate_iou(detection.bbox, track.bbox)
                # Find best IoU
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            if best_track is not None:
                # Match found - update track
                best_track.update(detection)
                matched_tracks.add(best_track.track_id)
                unmatched_detections.remove(detection)
                # Only return confirmed tracks (min_hits requirement)
                if best_track.hits >= self.min_hits:
                    matched_detections.append(Detection(
                        bbox=detection.bbox,
                        label=detection.label,
                        confidence=detection.confidence,
                        track_id=best_track.track_id
                    ))
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track = self.Track(self.next_id, detection)
            self.tracks.append(new_track)
            self.next_id += 1
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        # Return matched detections with track IDs
        return matched_detections

# PART 4: TRAFFIC COUNTER USING SIMPLE TRACKER
class TrafficCounter:
    """
    Traffic counter using simple IoU-based tracking.
    Counts unique objects that pass through the scene.
    """
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize traffic counter.
        Args:
            max_age: Max frames to keep track alive without detection
            min_hits: Min detections before track is confirmed
        """
        self.tracker = SimpleTracker(iou_threshold=iou_threshold, max_age=max_age, min_hits=min_hits)
        self.object_counts = defaultdict(int)  # Count per object type
        self.tracked_ids = set()  # Set of all unique track IDs seen
        self.total_count = 0
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Update counter with new detections.
        Args:
            detections: List of Detection objects from YOLO
        Returns: List of Detection objects enriched with track_id
        """
        # Update tracker
        tracked_detections = self.tracker.update(detections)
        # Count new unique objects
        for det in tracked_detections:
            if det.track_id not in self.tracked_ids:
                self.tracked_ids.add(det.track_id)
                self.object_counts[det.label] += 1
                self.total_count += 1
        # return tracked detections with IDs
        return tracked_detections
    
    def get_counts(self) -> Dict[str, int]:
        """Get count of each object type."""
        return dict(self.object_counts)
    
    def get_total_count(self) -> int:
        """Get total count of all tracked objects."""
        return self.total_count

# PART 5: VIDEO STREAMER WITH RASPBERRY PI OPTIMIZATION
class VideoStreamer:
    """
    Threaded video capture optimized for Raspberry Pi. 
    This turn on video capture in a separate thread to avoid blocking the main processing loop.
    Features:
    - Runs in separate thread to prevent blocking
    - Uses hardware-accelerated resolution (1536x864)
    - Configurable for USB or CSI camera
    """
    def __init__(self, resolution: Tuple[int,int] = (1536, 864), fps: int = 15):
        """
        Initialize video streamer.
        Args:
            source: Camera index (0 for default) or video file path
            resolution: Camera resolution - USE RPi hardware-accelerated format
            fps: Target frames per second
        """
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        # Inicialization IMX708
        self.picam2 = Picamera2()
        # Configuration
        config = self.picam2.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"}, # Use RPi hardware-accelerated format
            controls={"FrameRate": fps},
            buffer_count=4  # Increased for stability
        )
        self.picam2.configure(config)
        # FPS control
        self.target_fps = fps
        self.frame_time = 1.0 / fps
        print(f"[VideoStreamer] Initialized with resolution: {resolution}")
    
    def _capture_loop(self):
        """Internal capture loop running in separate thread"""
        print("[VideoStreamer] Starting camera hardware...")
        self.picam2.start()
        time.sleep(2)  # Stabilizace
        self.camera_started = True
        print("[VideoStreamer] Camera ready, beginning frame capture...")
        while self.running:
            start = time.time()
            try:
                frame = self.picam2.capture_array()
                with self.lock:
                    self.frame = frame
            except Exception as e:
                print(f"[VideoStreamer] Failed to read frame: {e}")
                time.sleep(0.1)
                continue
            # Frame rate control
            elapsed = time.time() - start
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.picam2.stop()

    def start(self):
        """Start capture thread (non-blocking)"""
        if self.thread is not None and self.thread.is_alive(): 
            print("[VideoStreamer] Already running"); return
        # Start capture thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("[VideoStreamer] Capture thread started")
    
    def stop(self):
        """Stop capture thread gracefully"""
        print("[VideoStreamer] Stopping...")
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5)
        print("[VideoStreamer] Stopped")

    def get_frame(self):
        """Get latest frame (thread-safe)"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def is_ready(self):
        with self.lock:
            has_frame = self.frame is not None
            ready = has_frame and self.camera_started
            print(has_frame, ready)
            return ready
    
# PART 6: WEBSOCKET MANAGER FOR REAL-TIME STREAMING
class WebSocketManager:
    """
    Manages WebSocket connections for real-time streaming to frontend.    
    Features:
    - Broadcasts detection data to all connected clients
    - Handles client connections/disconnections
    - Non-blocking async operations
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
    Main pipeline integrating detection, tracking, and real-time streaming.
    Architecture:
    1. VideoStreamer: Captures frames in separate thread
    2. TrafficDetector: Runs YOLO11 detection
    3. TrafficCounter: Tracks objects with SORT
    4. WebSocketManager: Streams data to frontend
    5. DatabaseManager: Stores all detections
    """
    def __init__(self, model_path: str = "yolo11n.pt", video_source: int = 0, db_path: str = "traffic_data.db", resolution: Tuple[int,int] = (1536, 864), fps: int = 15):
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
        self.detector = TrafficDetector(model_path, img_size=resolution)
        self.counter = TrafficCounter()
        self.streamer = VideoStreamer(resolution, fps)
        #self.db = DatabaseManager(db_path)
        self.ws_manager = WebSocketManager()
        # Pipeline state
        self.running = False
        self.fps = fps
        self.resolution = resolution
        self.frame_id = 0
        self.current_frame = None
        self.current_detections = []
        # Performance metrics
        self.fps_actual = 0.0
        self.last_fps_update = time.time()
        self.frame_times = []
        print("[Pipeline] Initialization complete")
    
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
        # Step 2: YOLO11 detection process
        detections = self.detector.detect(frame)
        # Step 3: SORT tracking process
        tracked_detections = self.counter.update(detections)
        self.current_detections = tracked_detections
        # Step 4: Prepare data for WebSocket and database
        timestamp = datetime.now().isoformat()
        total_count = self.counter.get_total_count()
        counts = self.counter.get_counts()
        # Prepare detection data for WebSocket and database
        #detection_data_list = []
        #for det in tracked_detections:
        #    detection_dict = {
        #        'timestamp': timestamp,
        #        'frame_id': self.frame_id,
        #        'label': det.label,
        #        'confidence': det.confidence,
        #        'x1': det.bbox[0],
        #        'y1': det.bbox[1],
        #        'x2': det.bbox[2],
        #        'y2': det.bbox[3],
        #        'track_id': det.track_id,
        #        'total_count': total_count
        #    }
        #    detection_data_list.append(detection_dict)
        # Step 5: Broadcast to WebSocket clients
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
        
        ## Step 6: Store in database (batch insert)
        #if detection_data_list:
        #    self.db.insert_batch_detections(detection_data_list)
        
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
        """
        Start the processing pipeline.
        Runs continuously processing frames until stopped.
        """
        print("[Pipeline] Starting traffic monitoring...")
        # Start video streamer video streamer in background thread
        self.streamer.start()
        # Wait for first frame to be available
        print("[Pipeline] Waiting for camera initialization...")
        # wait up to 10 seconds for camera to be ready
        max_wait = 5; waited = 0
        while not self.streamer.is_ready() and waited < max_wait:
            await asyncio.sleep(0.5)
            waited += 0.5
        if not self.streamer.is_ready():
            print("[Pipeline] ERROR: Camera failed to initialize!")
            self.streamer.stop()
            return
        print("[Pipeline] Camera ready, starting processing...")
        self.running = True
        print("[Pipeline] Detection running...")
        try:
            while self.running:
                await self.process_frame()
        except Exception as e:
            print(f"[Pipeline] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """Stop pipeline and cleanup resources."""
        print("[Pipeline] Stopping...")
        self.running = False
        self.streamer.stop()
        #self.db.close()
        print("[Pipeline] Stopped")
    
    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """
        Get current frame with drawn detections.
        Returns: Annotated frame or None
        """
        # Check if frame is available
        if self.current_frame is None:
            return None
        # Draw detections on frame
        annotated = self.detector.draw_detections(self.current_frame, self.current_detections, show_tracking=True)
        # Add statistics overlay
        counts = self.counter.get_counts()
        # Draw total count and per-type counts
        y_offset = 30
        cv2.putText(annotated, f"Total: {self.counter.get_total_count()}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        # Draw per-type counts
        for label, count in counts.items():
            cv2.putText(annotated, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        # Draw FPS
        cv2.putText(annotated, f"FPS: {self.fps_actual:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated

# PART 8: LOCAL TESTING WITHOUT FASTAPI SERVER ON DEVICE
async def main():
    """
    Local camera test without FastAPI server.
    Press 'q' to exit.
    """
    print("=== YOLO11 Traffic Detection Test ===")
    print("Press 'q' to exit")
    print()
    # Configuration for local testing (adjust resolution for your camera)
    RESOLUTION = (1536, 864)  # Use RPi hardware-accelerated resolution
    FPS = 15
    MODEL_PATH = "yolo11n.pt"  # Ensure model is downloaded
    # Create pipeline instance
    pipeline = TrafficMonitoringPipeline(model_path=MODEL_PATH, video_source=0, resolution=RESOLUTION, fps=FPS)
    # Start video streamer thread
    print("[Test] Starting video streamer...")
    pipeline.streamer.start()
    print("Waiting for camera initialization...")
    max_wait = 4; waited = 0
    while not pipeline.streamer.is_ready() and waited < max_wait:
        await asyncio.sleep(0.5)
        waited += 0.5
    if not pipeline.streamer.is_ready():
        print("[Pipeline] ERROR: Camera failed to initialize!")
        pipeline.streamer.stop()
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
            # Step 2: Run YOLO11 detection
            detections = pipeline.detector.detect(frame)
            # Step: 3 Update object tracker
            tracked_detections = pipeline.counter.update(detections)
            pipeline.current_detections = tracked_detections
            # Step 4: Prepare data for WebSocket and database
            timestamp = datetime.now().isoformat()
            total_count = pipeline.counter.get_total_count()
            counts = pipeline.counter.get_counts()
            # Update frame counter and FPS metrics
            pipeline.frame_id += 1
            print("Frame id", pipeline.frame_id)
            current_time = time.time()
            if current_time - pipeline.last_fps_update > 1.0:
                if pipeline.frame_times:
                    avg_time = sum(pipeline.frame_times) / len(pipeline.frame_times)
                    pipeline.fps_actual = 1.0 / avg_time if avg_time > 0 else 0
                    pipeline.frame_times = []
                pipeline.last_fps_update = current_time
            print(tracked_detections)
            detection_data_list = []
            for det in tracked_detections:
                detection_dict = {
                    'timestamp': timestamp,
                    'frame_id': pipeline.frame_id,
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
            print("Actual detections: ", detection_data_list)
            elapsed = time.time() - start_time
            pipeline.frame_times.append(elapsed)
            # Calculate FPS every second
            if time.time() - pipeline.last_fps_update > 1.0:
                if pipeline.frame_times:
                    avg_time = sum(pipeline.frame_times) / len(pipeline.frame_times)
                    pipeline.fps_actual = 1.0 / avg_time if avg_time > 0 else 0
                    pipeline.frame_times = []
                pipeline.last_fps_update = time.time()
            print(f"FPS:", pipeline.fps_actual)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        pipeline.streamer.stop()
        #cv2.destroyAllWindows()
        print("\nDone!")
        # Print final statistics
        print("\n=== FINAL STATISTICS ===")
        print(f"Total objects detected: {pipeline.counter.get_total_count()}")
        print(f"Count by type: {pipeline.counter.get_counts()}")
        print(f"Total frames processed: {pipeline.frame_id}")

if __name__ == "__main__": asyncio.run(main())