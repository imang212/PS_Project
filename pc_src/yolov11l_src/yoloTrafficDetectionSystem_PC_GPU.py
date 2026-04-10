import av
import cv2
import numpy as np
import asyncio
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import threading
import time
import subprocess
import shlex
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
            if not torch.cuda.is_available():
                print("[TrafficDetector] CUDA not available — exiting.")
                raise SystemExit(1)
            self.device = "cuda"
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
    Crop behaviour (crop_rect):
        Pass crop_rect=(x, y, w, h) to extract a 640×640 (or any size) region
        from the raw decoded frame BEFORE it is stored.  The crop is applied
        after decoding but before the resize step, so `resolution` is the final
        output size of the cropped region.  If crop_rect is None the full frame
        is resized to `resolution` as before.
    Example – crop a 640×640 patch starting at pixel (320, 180):
        VideoStreamer(source=..., resolution=(640, 640), crop_rect=(320, 180, 640, 640))
    """
    def __init__(self, source: Any = 0, resolution: Tuple[int,int] = (1280, 720), fps: int = 25, crop_rect: Optional[Tuple[int, int, int, int]] = None,):
        """
        Args:
            source:      RTSP URL (str), device index (int), or video file path (str)
            resolution:  Desired *output* resolution (width, height).
                         If crop_rect is given this is the size of the cropped tile.
                         If crop_rect is None the full frame is resized to this.
            fps:         Target frames-per-second (throttles read loop speed)
            crop_rect:   Optional (x, y, w, h) crop window applied to the raw
                         decoded frame before resize.  Set to None to disable.
        """
        self.source = source
        self.resolution = resolution
        self.target_fps = fps
        self.frame_time = 1.0 / fps
        self.crop_rect = crop_rect 

        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.eof = False 
        self.camera_started = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        if crop_rect:
            cx, cy, cw, ch = crop_rect
            print(f"[VideoStreamer] Crop enabled: x={cx} y={cy} w={cw} h={ch}  → output {resolution}")
        else:
            print(f"[VideoStreamer] Initialized  resolution: {resolution}  (no crop)")
    
    def _open_stream(self):
        return av.open(
            self.source,
            options={
                "rtsp_transport": "tcp",
                "fflags": "nobuffer",
                "flags": "low_delay",
                "max_delay": "0",
                "reorder_queue_size": "0",
                "stimeout": "5000000",    # 5s timeout
            }
        )
    
    def _apply_crop(self, img: np.ndarray) -> np.ndarray:
        """Crop img to self.crop_rect then resize to self.resolution."""
        cx, cy, cw, ch = self.crop_rect
        h, w = img.shape[:2]
        # Clamp so the crop never exceeds frame boundaries
        x1 = max(0, cx)
        y1 = max(0, cy)
        x2 = min(w, cx + cw)
        y2 = min(h, cy + ch)
        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            # Fallback – return full frame resized (shouldn't happen in normal use)
            return cv2.resize(img, self.resolution)
        if (x2 - x1, y2 - y1) != self.resolution:
            cropped = cv2.resize(cropped, self.resolution)
        return cropped

    def _capture_loop(self):
        frame_time = 1.0 / self.target_fps 
        while self.running:
            try:
                container = self._open_stream()
                self.camera_started = True
                print("[VideoStreamer] Stream opened via PyAV")
                for packet in container.demux(video=0):
                    if not self.running:
                        break
                    # Check for EOF flush packet
                    if packet.size == 0:
                        print("[VideoStreamer] End of file reached — stopping.")
                        with self.lock:
                            self.frame = None
                        self.eof = True
                        self.running = False
                        break
                    loop_start = time.time()
                    for av_frame in packet.decode():
                        # Resize if needed
                        img = av_frame.to_ndarray(format="bgr24")
                        if self.crop_rect is not None:
                            img = self._apply_crop(img)
                        else:
                            h, w = img.shape[:2]
                            if (w, h) != self.resolution:
                                img = cv2.resize(img, self.resolution)
                        with self.lock:
                            self.frame = img
                    # Throttle to target FPS
                    elapsed = time.time() - loop_start
                    sleep_time = frame_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                container.close()
            except Exception as e:
                print(f"[VideoStreamer] Unexpected error: {e} — reconnecting in 3s...")
                time.sleep(3)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def is_ready(self) -> bool:
        with self.lock:
            return self.camera_started and self.frame is not None
    
    def is_eof(self) -> bool:
        """True once a file source has played to the end."""
        return self.eof
    
# PART 6a: WEBSOCKET MANAGER FOR REAL-TIME STREAMING
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

# PART 6b: SELF-HOSTED RTSP SERVER USING GSTREAMER
class RTSPBroadcaster:
    """
    Self-contained RTSP server uses GStreamer's gst-rtsp-server (Python bindings: python3-gi +
    gir1.2-gst-rtsp-server-1.0) to host an RTSP endpoint directly inside
    this process. Frames are fed via an appsrc element.
    Install on Ubuntu/Debian:
        sudo apt install python3-gi gir1.2-gst-rtsp-server-1.0 \
                         gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
                         gstreamer1.0-plugins-bad gstreamer1.0-libav
    Usage:
        broadcaster = RTSPBroadcaster(port=8554, path="/traffic", width=640, height=640, fps=25)
        broadcaster.start()
        broadcaster.push_frame(bgr_frame)   # call from any thread
        broadcaster.stop()
    Connect with:
        ffplay  rtsp://127.0.0.1:8554/traffic
        vlc     rtsp://127.0.0.1:8554/traffic
        gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/traffic latency=0 \\
            ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
    """

    def __init__(self, port: int = 8554, path: str = "/traffic", width: int = 640, height: int = 640, fps: int = 25, bitrate: int = 2000,):
        self.port    = port
        self.path    = path
        self.width   = width
        self.height  = height
        self.fps     = fps
        self.bitrate = bitrate      # kbps 
        self._appsrc  = None        # GStreamer appsrc element
        self._loop    = None        # GLib main loop
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock    = threading.Lock()
        self._pts     = 0           # presentation timestamp counter
        self._frame_duration_ns = int(1e9 / fps)
        rtsp_url = f"rtsp://127.0.0.1:{port}{path}"
        print(f"[RTSPBroadcaster] Configured → {rtsp_url}  {width}×{height} @ {fps} fps")

    def start(self):
        """Start the GStreamer RTSP server in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        # Give the server a moment to bind the port
        time.sleep(0.5)

    def _run_server(self):
        try:
            import gi
            gi.require_version("Gst",      "1.0")
            gi.require_version("GstRtspServer", "1.0")
            from gi.repository import Gst, GstRtspServer, GLib
        except Exception as e:
            print(f"[RTSPBroadcaster] GStreamer import failed: {e}")
            print("[RTSPBroadcaster] Install with:")
            print("  sudo apt install python3-gi gir1.2-gst-rtsp-server-1.0 "
                  "gstreamer1.0-plugins-base gstreamer1.0-plugins-good "
                  "gstreamer1.0-plugins-bad gstreamer1.0-libav")
            return
        Gst.init(None)
        # Pipeline string fed to the RTSP media factory.
        # appsrc  → videoconvert → x264enc (low-latency) → rtph264pay
        pipeline = (
            f"( appsrc name=src is-live=true block=false format=time "
            f"  caps=video/x-raw,format=BGR,width={self.width},height={self.height},"
            f"framerate={self.fps}/1 "
            f"! videoconvert "
            f"! video/x-raw,format=I420 "
            f"! x264enc tune=zerolatency bitrate={self.bitrate} speed-preset=ultrafast "
            f"  key-int-max={self.fps} "
            f"! rtph264pay name=pay0 pt=96 config-interval=1 )"
        )
        server  = GstRtspServer.RTSPServer.new()
        server.set_service(str(self.port))
        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(pipeline)
        factory.set_shared(True)        # one pipeline shared across all clients
        # Grab appsrc once the pipeline is constructed
        factory.connect("media-configure", self._on_media_configure)
        mounts = server.get_mount_points()
        mounts.add_factory(self.path, factory)
        server.attach(None)
        print(f"[RTSPBroadcaster] RTSP server listening on "f"rtsp://127.0.0.1:{self.port}{self.path}")
        self._loop = GLib.MainLoop()
        self._loop.run()

    def _on_media_configure(self, factory, media):
        """Called by GStreamer when a client connects and the pipeline starts."""
        pipeline = media.get_element()
        self._appsrc = pipeline.get_by_name("src")
        # Debug: print the actual caps being negotiated
        print(f"[RTSPBroadcaster] appsrc caps: {self._appsrc.get_property('caps')}")
        print("[RTSPBroadcaster] appsrc ready — streaming active")
    
    def push_frame(self, frame: np.ndarray):
        """Push one BGR frame into the RTSP stream (thread-safe)."""
        with self._lock:
            appsrc = self._appsrc
        if appsrc is None:
            return      # no client connected yet — silently drop
        # Resize if needed
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))
        try:
            import gi
            from gi.repository import Gst
            data   = frame.tobytes()
            buf    = Gst.Buffer.new_wrapped(data)
            buf.pts      = self._pts
            buf.duration = self._frame_duration_ns
            self._pts   += self._frame_duration_ns
            appsrc.emit("push-buffer", buf)
        except Exception as e:
            print(f"[RTSPBroadcaster] push_frame error: {e}")
    
    def stop(self):
        """Shut down the RTSP server."""
        self._running = False
        if self._loop:
            self._loop.quit()
        if self._thread:
            self._thread.join(timeout=5)
        print("[RTSPBroadcaster] Stopped")

# PART 6c: VIDEO RECORDER
class VideoStatistics:
    """
    Tracks and calculates statistics for a single recording session.
    Owned by VideoRecorder — one instance per recording.
    reset() is called automatically on VideoRecorder.start() so the same
    VideoRecorder object can be reused across multiple sessions.
    """

    def __init__(self):
        self.detections_per_class: Dict[str, int]       = defaultdict(int)
        self.confidence_sum_per_class: Dict[str, float] = defaultdict(float)
        self.total_frames: int           = 0
        self.frames_with_detections: int = 0
        self.object_counts: Dict[str, int] = {
            "person": 0, "car": 0, "truck": 0,
            "bus": 0, "motorcycle": 0, "bicycle": 0,
        }
        self.unique_track_ids: set          = set()
        self.start_time: Optional[datetime] = None
        self.end_time:   Optional[datetime] = None

    def reset(self):
        """Clear all counters — called at the start of every new recording."""
        self.__init__()

    def add_detection(self, class_name: str, confidence: float, track_id: Optional[int] = None):
        """Register one detected object in the current frame."""
        self.detections_per_class[class_name] += 1
        self.confidence_sum_per_class[class_name] += confidence
        if track_id is not None and track_id not in self.unique_track_ids:
            self.unique_track_ids.add(track_id)
            if class_name in self.object_counts:
                self.object_counts[class_name] += 1

    def add_frame(self, has_detections: bool = False):
        """Record that one frame was processed."""
        self.total_frames += 1
        if has_detections:
            self.frames_with_detections += 1

    def get_summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable statistics dictionary."""
        summary: Dict[str, Any] = {
            "total_frames":           self.total_frames,
            "frames_with_detections": self.frames_with_detections,
            "unique_objects_tracked": len(self.unique_track_ids),
            "processing_time":        None,
            "fps":                    "n/a",
            "detections_by_class":    {},
        }
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            summary["recording_start"] = self.start_time.isoformat()
            summary["recording_end"]   = self.end_time.isoformat()
            summary["processing_time"] = f"{duration:.2f} seconds"
            summary["fps"] = (
                f"{self.total_frames / duration:.2f}" if duration > 0 else "N/A"
            )
        for class_name, count in self.detections_per_class.items():
            avg_conf = (
                self.confidence_sum_per_class[class_name] / count if count > 0 else 0.0
            )
            summary["detections_by_class"][class_name] = {
                "count":              count,
                "real_object_count":  self.object_counts.get(class_name, 0),
                "average_confidence": f"{avg_conf:.4f}",
            }
        return summary

class VideoRecorder:
    """
    Records annotated pipeline frames to a single MP4 file per session.
    Usage:
        recorder = VideoRecorder(output_dir="recordings", fps=25, resolution=(640, 640))
        recorder.start()            # opens the file, begins writing
        recorder.write(bgr_frame)  # call for every annotated frame
        recorder.stop()             # flushes & closes the file
    File naming:
        <output_dir>/recording_YYYYMMDD_HHMMSS.mp4
    Codec:
        mp4v (OpenCV built-in, no extra dependencies).
        Swap fourcc to 'avc1' if your OpenCV build supports it for smaller files.
    Thread safety:
        write() acquires a lock so it is safe to call from any thread.
    """
    def __init__(self, output_dir: str = "recordings", fps: int = 25, resolution: Tuple[int, int] = (640, 640),):
        """
        Args:
            output_dir:  Directory where the MP4 file will be saved (created if absent).
            fps:         Frame rate written into the video container.
            resolution:  (width, height) of every frame that will be passed to write().
        """
        self.output_dir = output_dir
        self.fps = fps
        self.resolution = resolution  # (width, height)
        self._writer: Optional[cv2.VideoWriter] = None
        self._lock = threading.Lock()
        self._output_path: Optional[str] = None
        self._frame_count: int = 0
        self.stats: VideoStatistics = VideoStatistics()


    def start(self) -> str:
        """
        Open a new MP4 file and start recording.
        Returns the full path of the file being written.
        Raises RuntimeError if a recording is already active.
        """
        with self._lock:
            if self._writer is not None:
                raise RuntimeError("[VideoRecorder] Already recording – call stop() first.")
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename  = f"recording_{timestamp}.mp4"
            self._output_path = os.path.join(self.output_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self._output_path, fourcc, self.fps, self.resolution
            )
            if not self._writer.isOpened():
                self._writer = None
                raise RuntimeError(
                    f"[VideoRecorder] cv2.VideoWriter failed to open '{self._output_path}'. "
                    "Check that OpenCV was built with video-write support."
                )
            self._frame_count = 0
            self.stats.reset()
            self.stats.start_time = datetime.now()
            print(f"[VideoRecorder] Recording started → {self._output_path}")
            return self._output_path

    def write(self, frame: np.ndarray) -> bool:
        """
        Write one BGR frame to the open recording.
        Silently skipped when no recording is active (returns False).
        Resizes the frame if it does not match self.resolution.
        Returns True if the frame was written successfully.
        """
        with self._lock:
            if self._writer is None:
                return False
            w, h = self.resolution
            fh, fw = frame.shape[:2]
            if (fw, fh) != (w, h):
                frame = cv2.resize(frame, (w, h))
            self._writer.write(frame)
            self._frame_count += 1
            return True

    def stop(self) -> Optional[str]:
        """
        Flush and close the current recording, then write a companion
        JSON statistics file with the same base name as the MP4.
        Returns the path of the finished MP4, or None if nothing was recording.
        """
        with self._lock:
            if self._writer is None:
                return None
            self._writer.release()
            self._writer = None
            path = self._output_path
            self._output_path = None
            print(
                f"[VideoRecorder] Recording stopped → {path}  "
                f"({self._frame_count} frames)"
            )
            self._frame_count = 0
        # Finalise statistics (outside lock — no writer state needed)
        self.stats.end_time = datetime.now()
        self._write_stats_json(path)
        return path

    def _write_stats_json(self, video_path: Optional[str]):
        """Serialise VideoStatistics.get_summary() to a .json file beside the MP4."""
        if video_path is None:
            return
        import json, os
        json_path = os.path.splitext(video_path)[0] + "_stats.json"
        summary = self.stats.get_summary()
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[VideoRecorder] Statistics saved  → {json_path}")
        except Exception as e:
            print(f"[VideoRecorder] Failed to write statistics JSON: {e}")

    def feed_detections(self, detections: List["Detection"]):
        """
        Update statistics for the current frame.
        Call once per frame (whether or not detections is empty) so that
        total_frames and frames_with_detections stay in sync with the video.
        Safe to call even when not recording — silently ignored.
        """
        if not self.is_recording:
            return
        self.stats.add_frame(has_detections=bool(detections))
        for det in detections:
            self.stats.add_detection(class_name=det.label, confidence=det.confidence, track_id=det.track_id if det.track_id != -1 else None,)

    @property
    def is_recording(self) -> bool:
        """True while a file is open and frames are being written."""
        with self._lock:
            return self._writer is not None
    @property
    def output_path(self) -> Optional[str]:
        """Path of the file currently being written (None when idle)."""
        with self._lock:
            return self._output_path
    @property
    def frame_count(self) -> int:
        """Number of frames written in the current (or last) recording."""
        return self._frame_count

# PART 7: MAIN TRAFFIC MONITORING PIPELINE
class TrafficMonitoringPipeline:
    """
    Main pipeline integrating detection, tracking, WebSocket streaming, and MQTT publishing.
    Architecture:
        VideoStreamer   – captures RTSP / local camera frames in a background thread with optional 640×640 crop (crop_rect)
        TrafficDetector – runs YOLO11 on GPU
        TrafficCounter  – IoU tracker assigns persistent IDs
        WebSocketManager– streams detection JSON to connected browser clients
        MQTTPublisher   – publishes detections to the MQTT broker
        RTSPBroadcaster  – (optional) re-streams annotated frames as an RTSP feed
        ROI zone         – (optional) rectangle that restricts which detections count
    """
    def __init__(self, model_path: str = "yolo11n.pt", tracker_config: str = "bytetrack.yaml", video_source: Any = 0, 
                 resolution: Tuple[int,int] = (1280, 720), fps: int = 25, device: str = "auto", 
                 confidence_threshold: float = 0.25, crop_rect: Optional[Tuple[int, int, int, int]] = None, roi_rect: Optional[Tuple[int, int, int, int]] = None,
                 websocket_enabled: bool = True, mqtt_enabled: bool = True, rtsp_out_enabled: bool = False,
                 recording_enabled: bool = False, recording_dir: str = "recordings",):
        """
        Initialize complete pipeline.
        Args:
            model_path: Path to YOLO11 model weights
            video_source: Camera index or video file path
            db_path: SQLite database path
            resolution: Camera resolution (MUST be RPi hardware-accelerated)
            fps: Target frames per second
            device              : "auto" | "cuda" | "cpu"
            confidence_threshold: YOLO confidence threshold
            crop_rect           : (x, y, w, h) crop in raw-frame pixels; None = no crop
            roi_rect            : (x, y, w, h) detection-active rectangle in output-frame pixels; None = whole frame
            websocket_enabled   : Enable WebSocket JSON streaming
            mqtt_enabled        : Enable MQTT publishing
            rtsp_out_enabled    : Broadcast annotated frames as an RTSP stream
            recording_enabled   : Set True to record annotated frames to MP4 from session start
            recording_dir       : Directory where recording_YYYYMMDD_HHMMSS.mp4 files are saved 
        """
        # Initialize components
        self.detector = TrafficDetector(model_path, img_size=resolution, device=device, conf_threshold=confidence_threshold ,tracker_config=tracker_config)
        self.counter = TrafficCounter()
        self.streamer = VideoStreamer(source=video_source, resolution=resolution, fps=fps, crop_rect=crop_rect,)
        # websocket
        self.ws_manager = None
        self.websocket_enabled = websocket_enabled
        if websocket_enabled:
            self.ws_manager = WebSocketManager()
        # mqtt
        self.mqtt = None
        self.mqtt_enabled = mqtt_enabled
        if self.mqtt_enabled:
            self.mqtt = MQTTPublisher(broker_host="mqtt.portabo.cz", broker_port=8883, topic="patrik/traffic_detection", client_id="PC_traffic_detection", username="videoanalyza", password="phdA9ZNW1vfkXdJkhhbP")
        self.mqtt_publish_every_n_frames = 1
        # rtsp
        self.rtsp_out_enabled = rtsp_out_enabled
        self.rtsp_broadcaster: Optional[RTSPBroadcaster] = None
        if rtsp_out_enabled:
            w, h = resolution
            self.rtsp_broadcaster = RTSPBroadcaster(port=8554, path="/traffic", width=w, height=h, fps=fps, bitrate="2000",)
            print(f"[Pipeline] RTSP output enabled → rtsp://127.0.0.1:8554/traffic")
        # recording
        self.recording_enabled: bool = recording_enabled
        self.recorder: VideoRecorder = VideoRecorder(
            output_dir=recording_dir, fps=fps, resolution=resolution
        )
        if recording_enabled:
            self.recorder.start()
        else:
            print("[Pipeline] Video recording disabled (set recording_enabled=True to activate)")
        # roi
        self.roi_rect: Optional[Tuple[int, int, int, int]] = roi_rect
        if roi_rect:
            rx, ry, rw, rh = roi_rect
            print(f"[Pipeline] ROI zone  x={rx} y={ry} w={rw} h={rh}")
        else:
            print("[Pipeline] ROI zone  disabled (whole frame)")
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
    
    def _detection_in_roi(self, det: Detection) -> bool:
        """Return True if the bbox centre of `det` lies inside self.roi_rect."""
        if self.roi_rect is None:
            return True
        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx, ry, rw, rh = self.roi_rect
        return rx <= cx <= rx + rw and ry <= cy <= ry + rh
    
    def _filter_by_roi(self, detections: List[Detection]) -> List[Detection]:
        """Keep only detections whose centre falls inside the ROI."""
        if self.roi_rect is None:
            return detections
        return [d for d in detections if self._detection_in_roi(d)]

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
        1. Get frame from streamer (already cropped if crop_rect set)
        2. Run YOLO11 detection
        3. Filter detections by ROI rectangle
        4. Update ByteTrack counter
        5. Broadcast detection JSON via WebSocket
        6. Publish to MQTT (rate-limited)
        7. Push annotated frame to RTSP broadcaster (if enabled) 
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
        # Step 2: Detection
        detections = await asyncio.to_thread(self.detector.detect, frame)
        # Step 3: Filter by ROI
        detections = self._filter_by_roi(detections)
        self.current_detections = tracked
        # Step 4: Track
        tracked = self.counter.update(detections)
        self.current_detections = tracked
        # Step 5: Prepare data for WebSocket and database
        timestamp = datetime.now().isoformat()
        total_count = self.counter.get_total_count()
        counts = self.counter.get_counts()
        # Prepare detection data for WebSocket and database
        detection_data_list = []
        for det in tracked:
            detection_dict = {
                'timestamp': timestamp,
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
        # Step 6 – MQTT publish (rate-limited by mqtt_publish_every_n_frames)
        if self.mqtt_enabled and (self.frame_id % self.mqtt_publish_every_n_frames == 0):
            await asyncio.to_thread(self._publish_mqtt, timestamp, detection_data_list)
        if (self.frame_id % 30 == 0):
            print(f"Detected objects: {detection_data_list}")
        # Step 7: RTSP output push
        if self.rtsp_out_enabled and self.rtsp_broadcaster:
            annotated = self.get_annotated_frame()
            if annotated is not None:
                await asyncio.to_thread(self.rtsp_broadcaster.push_frame, annotated)
        # Step 8: Video recording
        if self.recording_enabled and self.recorder.is_recording:
            annotated = self.get_annotated_frame()
            if annotated is not None:
                await asyncio.to_thread(self.recorder.write, annotated)
            self.recorder.feed_detections(tracked)
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
        if self.rtsp_out_enabled and self.rtsp_broadcaster:
            self.rtsp_broadcaster.start()
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
        if self.rtsp_out_enabled and self.rtsp_broadcaster:
            self.rtsp_broadcaster.stop()
        if self.recording_enabled:
            self.recorder.stop()
        if self.mqtt:
            self.mqtt.disconnect()
        print("[Pipeline] Stopped")
    
    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Return current frame with bounding boxes and stats overlay."""
        if self.current_frame is None:
            return None
        annotated = self.detector.draw_detections(self.current_frame, self.current_detections, show_tracking=True)
        if self.roi_rect is not None:
            rx, ry, rw, rh = self.roi_rect
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            cv2.putText(annotated, "ROI", (rx + 4, ry + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        counts = self.counter.get_counts()
        y_offset = 30
        cv2.putText(annotated, f"Bytetrack - yolo11l", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(annotated, f"Total: {self.counter.get_total_count()}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 25
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
    New feature flags
    -----------------
    CROP_RECT        : (x, y, w, h) in raw-frame pixels.  Set to None to disable.
    RTSP_OUT_ENABLED : True → push annotated frames to RTSP_OUT_URL via FFmpeg. Requires a running RTSP server (e.g. MediaMTX).
    ROI_RECT         : (x, y, w, h) in output-frame pixels.  Only detections whose bbox centre falls inside this box are counted / published. Set to None to use the whole frame.
    """
    print("=== YOLO11 Traffic Detection Test ===")
    print("Press 'q' to exit")
    print()
    # Configuration for local testing (adjust resolution for your camera)
    RTSP_URL    = "rtsp://admin:Dcuk.123456@192.168.37.99/Stream"  # ← your camera URL
    # VIDEO_SOURCE = 0
    VIDEO_SOURCE = "/home/patrik/videos/test_outputx264-202601091552_000.mp4"
    #VIDEO_SOURCE = RTSP_URL
    CROP_RECT  : Optional[Tuple[int,int,int,int]] = (200, 80, 640, 640)  # (x,y,w,h) | None
    RESOLUTION = (640, 640)
    # roi rect
    ROI_RECT : Optional[Tuple[int,int,int,int]] = (int(640*0.15), int(640*0.85), int(640*0.35), int(640*0.14)) # (x,y,w,h) | None
    # Bytetrack
    TRACKER_CONFIG = create_bytetrack_config(
        output_path="bytetrack.yaml",
        track_high_thresh=0.35,   # lower if losing detections on real camera
        track_low_thresh=0.05,    # lower to recover more occluded vehicles
        new_track_thresh=0.45,    # raise if getting ghost/duplicate tracks
        track_buffer=60,          # raise for slow traffic or long occlusions
        match_thresh=0.7,         # lower slightly if IDs keep switching
        fuse_score=True           # multiply confidence into matching distance
    )
    # Create pipeline instance
    pipeline = TrafficMonitoringPipeline(
        model_path="yolo11l.pt", 
        tracker_config=TRACKER_CONFIG,
        video_source=VIDEO_SOURCE, 
        resolution=RESOLUTION, 
        fps=25, 
        device="auto", 
        crop_rect=CROP_RECT,
        roi_rect=ROI_RECT,
        websocket_enabled=False, 
        mqtt_enabled=False,
        rtsp_out_enabled=False,    
        recording_enabled=True,
        recording_dir="recordings",
    )
    # Start video streamer thread
    print("[Test] Starting video streamer...")
    pipeline.streamer.start()
    if pipeline.rtsp_out_enabled and pipeline.rtsp_broadcaster:
        pipeline.rtsp_broadcaster.start()
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
            # Step 3: Filter by ROI
            detections = pipeline._filter_by_roi(detections)
            # Step: 3 Update object tracker
            tracked = pipeline.counter.update(detections)
            pipeline.current_frame = frame
            pipeline.current_detections = tracked
            timestamp = datetime.now().isoformat()
            # MQTT publish
            if pipeline.mqtt_enabled and (pipeline.frame_id % pipeline.mqtt_publish_every_n_frames == 0):
                await asyncio.to_thread(pipeline._publish_mqtt, timestamp, tracked)
            # Push annotated frame to RTSP if enabled (no local window)
            if pipeline.rtsp_out_enabled and pipeline.rtsp_broadcaster:
                annotated = pipeline.get_annotated_frame()
                if annotated is not None:
                    await asyncio.to_thread(pipeline.rtsp_broadcaster.push_frame, annotated)
            # Record annotated frame to MP4 if enabled
            if pipeline.recording_enabled and pipeline.recorder.is_recording:
                annotated = pipeline.get_annotated_frame()
                if annotated is not None:
                    await asyncio.to_thread(pipeline.recorder.write, annotated)
                pipeline.recorder.feed_detections(tracked)
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
                #print(f"[Test] Frame {pipeline.frame_id:5d}  FPS: {pipeline.fps_actual:.1f}  "f"Tracked: {len(tracked)}  Total: {pipeline.counter.get_total_count()}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        pipeline.stop()
        #cv2.destroyAllWindows()
        print("\nDone!")
        # Print final statistics
        print("\n=== FINAL STATISTICS ===")
        print(f"Total objects detected: {pipeline.counter.get_total_count()}")
        print(f"Count by type: {pipeline.counter.get_counts()}")
        print(f"Total frames processed: {pipeline.frame_id}")

if __name__ == "__main__": asyncio.run(main())