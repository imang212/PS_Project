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
import time
from ultralytics import YOLO
from sort import Sort  # Ensure SORT tracker is available
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
    TRAFFIC_CLASSES = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign'}
    
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
        _ = self.model(dummy_frame, imgsz=img_size, verbose=False)
        
        print(f"[TrafficDetector] Model loaded. Classes: {len(self.model.names)}")
        print(f"[TrafficDetector] Using hardware-accelerated resolution: {img_size}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame using YOLO11.        
        Process:
        1. Run YOLO11 inference on frame
        2. Extract bounding boxes, classes, and confidence scores
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
        
        detections = []
        # Process each detection from YOLO results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract detection data
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                # Filter: only traffic-relevant objects
                if label not in self.TRAFFIC_CLASSES:
                    continue
                # Get bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    label=label,
                    confidence=conf
                ))
        
        return detections
    
    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Detection], show_tracking: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame for visualization.
        Color coding:
        - Green: Vehicles (car, truck, bus, motorcycle)
        - Blue: Pedestrians and cyclists
        - Red: Traffic signs and lights
        Args:
            frame: Input frame to draw on
            detections: List of Detection objects
            show_tracking: Whether to display track_id
        Returns:
            Annotated frame with drawn detections
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Color based on object type
            if det.label in ['car', 'truck', 'bus', 'motorcycle']:
                color = (0, 255, 0)  # Green for vehicles
            elif det.label in ['person', 'bicycle']:
                color = (255, 0, 0)  # Blue for pedestrians/cyclists
            else:
                color = (0, 0, 255) # Red for traffic infrastructure
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

# PART 3: SORT TRACKER FOR VEHICLE COUNTING
class TrafficCounter:
    """
    Traffic counter using SORT (Simple Online Realtime Tracking).    
    SORT algorithm:
    1. Predicts object positions using Kalman filter
    2. Associates detections with existing tracks using Hungarian algorithm
    3. Creates new tracks for unmatched detections
    4. Maintains unique IDs for each object throughout video
    This enables accurate counting even when objects temporarily disappear.
    """
    def __init__(self, max_age: int = 30, min_hits: int = 3):
        """
        Initialize SORT tracker. 
        Args:
            max_age: Max frames to keep track alive without detection
            min_hits: Min detections before track is confirmed
        """
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=0.3)
        self.object_counts = defaultdict(int)  # Count per object type
        self.tracked_ids = set()  # Set of all unique track IDs seen
        self.total_count = 0
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Update tracker with new detections from current frame.
        Process:
        1. Convert detections to SORT format: [x1,y1,x2,y2,conf]
        2. Call SORT tracker update
        3. SORT returns tracked objects with IDs
        4. Match tracked objects back to original detections
        5. Count new unique IDs
        Args:
            detections: List of Detection objects from YOLO
        Returns:
            List of Detection objects enriched with track_id
        """
        if not detections:
            # Update tracker with empty detections to maintain tracks
            self.tracker.update(np.empty((0, 5)))
            return []
        # Convert detections to numpy array for SORT: [[x1,y1,x2,y2,conf], ...]
        det_array = np.array([
            [*det.bbox, det.confidence] 
            for det in detections
        ])
        # SORT update returns: [[x1,y1,x2,y2,track_id], ...]
        tracked_objects = self.tracker.update(det_array)
        # Match tracked objects back to original detections
        tracked_detections = []
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            # Find closest matching detection by comparing bounding box centers
            best_match = None
            min_dist = float('inf')
            for det in detections:
                # Calculate distance between bbox centers
                track_center = ((x1+x2)/2, (y1+y2)/2)
                det_center = ((det.bbox[0]+det.bbox[2])/2, (det.bbox[1]+det.bbox[3])/2)
                dist = np.sqrt((track_center[0]-det_center[0])**2 + (track_center[1]-det_center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = det
            
            if best_match:
                # Create new detection with track_id
                tracked_det = Detection(
                    bbox=tuple(map(int, [x1, y1, x2, y2])),
                    label=best_match.label,
                    confidence=best_match.confidence,
                    track_id=track_id
                )
                tracked_detections.append(tracked_det)
                # Count new unique objects
                if track_id not in self.tracked_ids:
                    self.tracked_ids.add(track_id)
                    self.object_counts[best_match.label] += 1
                    self.total_count += 1
        
        return tracked_detections
    
    def get_counts(self) -> Dict[str, int]:
        """Get count of each object type."""
        return dict(self.object_counts)
    
    def get_total_count(self) -> int:
        """Get total count of all tracked objects."""
        return self.total_count

# PART 4: VIDEO STREAMER WITH RASPBERRY PI OPTIMIZATION
class VideoStreamer:
    """
    Threaded video capture optimized for Raspberry Pi.    
    Features:
    - Runs in separate thread to prevent blocking
    - Uses hardware-accelerated resolution (1536x864)
    - Configurable for USB or CSI camera
    """
    def __init__(self, source: int = 0, resolution: Tuple[int,int] = (1536, 864), fps: int = 15):
        """
        Initialize video streamer.
        Args:
            source: Camera index (0 for default) or video file path
            resolution: Camera resolution - USE RPi hardware-accelerated format
            fps: Target frames per second
        """
        self.source = source
        self.resolution = resolution
        self.target_fps = fps
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        print(f"[VideoStreamer] Initializing with resolution: {resolution}")
    
    def start(self):
        """Start video capture in separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self
    
    def _capture_loop(self):
        """
        Main capture loop running in separate thread.
        This prevents the main processing loop from being blocked by camera read operations.
        """
        cap = cv2.VideoCapture(self.source)
        # Configure camera for Raspberry Pi hardware acceleration
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        # Reduce buffer size for lower latency (critical for real-time)
        if self.source == 0:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # For RPi Camera Module (if using libcamera)
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        frame_time = 1.0 / self.target_fps
        # Capture loop
        while self.running:
            start = time.time()
            ret, frame = cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("[VideoStreamer] Failed to read frame")
            # Frame rate control
            elapsed = time.time() - start
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0: time.sleep(sleep_time)
        # Release resources
        cap.release()
        print("[VideoStreamer] Capture stopped")
    
    def read(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame.
        Returns: 
            Copy of current frame or None if not available
        """
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Stop video capture thread."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

# PART 5: WEBSOCKET MANAGER FOR REAL-TIME STREAMING
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

# PART 6: MAIN TRAFFIC MONITORING PIPELINE
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
        self.streamer = VideoStreamer(video_source, resolution, fps)
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
        frame = self.streamer.read()
        if frame is None:
            await asyncio.sleep(0.01)
            return
        self.current_frame = frame
        # Step 2: YOLO11 detection
        detections = self.detector.detect(frame)
        # Step 3: SORT tracking
        tracked_detections = self.counter.update(detections)
        self.current_detections = tracked_detections
        # Step 4: Prepare data
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
        await self.ws_manager.broadcast(ws_message)
        ## Step 6: Store in database (batch insert)
        #if detection_data_list:
        #    self.db.insert_batch_detections(detection_data_list)
        
        # Update metrics
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
        # Start video streamer
        self.streamer.start()
        await asyncio.sleep(2)  # Wait for camera initialization
        self.running = True
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
        if self.current_frame is None:
            return None

        annotated = self.detector.draw_detections(
            self.current_frame, 
            self.current_detections, 
            show_tracking=True
        )
        # Add statistics overlay
        counts = self.counter.get_counts()
        y_offset = 30
        cv2.putText(annotated, f"Total: {self.counter.get_total_count()}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        for label, count in counts.items():
            cv2.putText(annotated, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        cv2.putText(annotated, f"FPS: {self.fps_actual:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated


