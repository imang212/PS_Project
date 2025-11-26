#!/usr/bin/env python3
"""
AI Traffic Detection System for Raspberry Pi AI HAT+ (26 TOPS)
Uses YOLOv8m model with ONNX + Hailo acceleration and HailoTracker
"""
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import websockets
from picamera2 import Picamera2
from hailo_platform import (VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)

#0: SYSTEM CONFIGURATION
class Config:
    """Global application configuration"""
    # Camera and inference resolution
    CAMERA_WIDTH = 1536
    CAMERA_HEIGHT = 864
    # YOLOv8m Model
    MODEL_PATH = "/path/to/yolov8m.hef"  # HEF = Hailo Executable Format
    # WebSocket configuration
    WEBSOCKET_HOST = "0.0.0.0"
    WEBSOCKET_PORT = 8765
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4  # Non-Maximum Suppression
    # COCO classes for traffic (YOLOv8 is trained on COCO dataset)
    TRAFFIC_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',1: 'bicycle', 0: 'person'}

#1: HAILO INFERENCE ENGINE
class HailoInferenceEngine:
    """
    Class for managing Hailo inference with YOLOv8m model.
    Handles model loading, VStreams configuration, and inference execution.
    """
    def __init__(self, model_path: str):
        """
        Initialize Hailo inference engine. 
        Args:
            model_path: Path to YOLOv8m model in HEF format
        """
        self.model_path = model_path
        self.device = None
        self.network_group = None
        self.input_vstreams = None
        self.output_vstreams = None
        self.tracker = None
        
    def initialize(self):
        """
        Initialize Hailo device and load model.
        Steps:
        1. Create VDevice object (virtual Hailo device)
        2. Load HEF file with YOLOv8m model
        3. Configure network group
        4. Set up input/output VStreams for data flow
        5. Initialize HailoTracker for object tracking
        """
        print("[INIT] Initializing Hailo AI HAT+...")
        # Step 1: Create virtual Hailo device
        self.device = VDevice()
        # Step 2: Load model
        print(f"[INIT] Loading YOLOv8m model from: {self.model_path}")
        hef = self.device.create_hef_file(self.model_path) # Load HEF
        # Step 3: Configure network group
        network_group_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.device.configure(hef, network_group_params)[0]
        # Step 4: Set up VStreams
        # Input stream - data from camera to Hailo chip
        input_vstream_params = InputVStreamParams.make_from_network_group(
            self.network_group,
            format_type=FormatType.UINT8  # 8-bit unsigned integer (RGB)
        )
        # Output stream - detections from Hailo chip
        output_vstream_params = OutputVStreamParams.make_from_network_group(
            self.network_group,
            format_type=FormatType.FLOAT32  # 32-bit float (bounding boxes + confidence)
        )
        self.input_vstreams = InferVStreams(self.network_group, input_vstream_params)
        self.output_vstreams = InferVStreams(self.network_group, output_vstream_params)
        # Step 5: Initialize HailoTracker
        from hailo_tracker import HailoTracker
        self.tracker = HailoTracker(
            max_lost_frames=30,  # Object "disappears" after 30 frames without detection
            iou_threshold=0.3     # Threshold for matching objects between frames
        )
        print("[INIT] Hailo engine initialized successfully")
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for YOLOv8m model.
        Args:
            frame: RGB frame from camera (1536x864x3)
        Returns: Preprocessed frame ready for inference
        Steps:
        1. Normalize pixels (0-255 -> 0-1)
        2. Change dimension order (HWC -> CHW) for ONNX
        3. Add batch dimension
        """
        # Normalize to range 0-1
        frame_normalized = frame.astype(np.float32) / 255.0
        # Change from (Height, Width, Channels) to (Channels, Height, Width)
        frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
        # Add batch dimension: (1, C, H, W)
        frame_batched = np.expand_dims(frame_transposed, axis=0)
        # return preprocessed frame
        return frame_batched
    
    def postprocess_detections(self, output: np.ndarray, frame_id: int) -> List[Dict[str, Any]]:
        """
        Postprocess output from YOLOv8m model.
        Args:
            output: Raw output from Hailo chip
            frame_id: Current frame ID
        Returns: List of detections with tracking ID
        Steps:
        1. Parse output tensor (bounding boxes, confidence, classes)
        2. Apply confidence threshold
        3. Non-Maximum Suppression (remove duplicates)
        4. Track objects using HailoTracker
        5. Filter only traffic classes
        """
        detections = []
        # Step 1: Parse YOLOv8 output format
        # YOLOv8 output: [batch, num_detections, 85]
        # where 85 = [x, y, w, h, confidence, class_0_prob, ..., class_79_prob]
        for detection in output[0]:  # batch=1, so we take [0]
            x_center, y_center, width, height = detection[0:4]
            confidence = detection[4]
            class_probs = detection[5:]
            # Step 2: Confidence filtering
            if confidence < Config.CONFIDENCE_THRESHOLD:
                continue
            # Get class with highest probability
            class_id = int(np.argmax(class_probs))
            class_confidence = class_probs[class_id]
            # Step 5: Filter traffic classes
            if class_id not in Config.TRAFFIC_CLASSES:
                continue
            # Convert from center format to corner format (x1, y1, x2, y2)
            x1 = int((x_center - width / 2) * Config.CAMERA_WIDTH)
            y1 = int((y_center - height / 2) * Config.CAMERA_HEIGHT)
            x2 = int((x_center + width / 2) * Config.CAMERA_WIDTH)
            y2 = int((y_center + height / 2) * Config.CAMERA_HEIGHT)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence * class_confidence),
                'class_id': class_id,
                'class_name': Config.TRAFFIC_CLASSES[class_id]
            })
        # Step 3: Non-Maximum Suppression
        detections = self._apply_nms(detections)
        # Step 4: Tracking
        tracked_detections = self.tracker.update(detections, frame_id)
        return tracked_detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Non-Maximum Suppression - remove overlapping detections.
        Args:
            detections: List of all detections
        Returns: Filtered list without duplicates
        """
        if len(detections) == 0: return []
        # Extract bounding boxes and confidence scores
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        # Calculate areas
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        # Sort by confidence (descending)
        order = scores.argsort()[::-1]
        # Apply NMS
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # Calculate IoU (Intersection over Union)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            # Keep only detections with IoU < threshold
            inds = np.where(iou <= Config.NMS_THRESHOLD)[0]
            # Update order
            order = order[inds + 1]
        # return filtered detections
        return [detections[i] for i in keep]
    
    def infer(self, frame: np.ndarray, frame_id: int) -> List[Dict[str, Any]]:
        """
        Perform inference on a single frame.
        Args:
            frame: RGB frame from camera
            frame_id: Frame ID
        Returns: List of detections with tracking information
        """
        # Preprocessing
        processed_frame = self.preprocess_frame(frame)
        # Inference on Hailo chip (HW acceleration!)
        with self.input_vstreams as input_stream, self.output_vstreams as output_stream:
            # Send data to Hailo chip
            input_stream.send(processed_frame)
            # Receive results from Hailo chip
            output = output_stream.recv()
        # Postprocessing and tracking
        detections = self.postprocess_detections(output, frame_id)
        return detections
    
    def cleanup(self):
        """Shutdown and release resources"""
        if self.network_group:
            self.network_group.release()
        if self.device:
            self.device.release()
        print("[CLEANUP] Hailo engine terminated")

#2: CAMERA MANAGER
class CameraManager:
    """
    IMX708 camera management using picamera2.
    Provides continuous frame capture at optimal resolution.
    """
    def __init__(self):
        """Initialize camera manager"""
        self.camera = None
        self.is_running = False
        
    def initialize(self):
        """
        Initialize camera with optimized configuration. 
        Steps:
        1. Create Picamera2 instance
        2. Configure for 1536x864 resolution (Hailo HW acceleration)
        3. Set frame rate and format
        4. Start camera
        """
        print("[CAMERA] Initializing IMX708 camera...")
        self.camera = Picamera2()
        # Configure camera for Hailo-optimized resolution
        config = self.camera.create_still_configuration(
            main={ "size": (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT), "format": "RGB888"},
            buffer_count=4  # Buffering for smooth capture
        )
        self.camera.configure(config)
        # start camera
        self.camera.start()
        self.is_running = True
        print(f"[CAMERA] Camera started: {Config.CAMERA_WIDTH}x{Config.CAMERA_HEIGHT}")
        
    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from camera.
        Returns: RGB frame as numpy array (1536x864x3)
        """
        if not self.is_running:
            raise RuntimeError("Camera is not running")
        frame = self.camera.capture_array()
        return frame
    
    def cleanup(self):
        """Shutdown camera"""
        if self.camera and self.is_running:
            self.camera.stop()
            self.is_running = False
            print("[CAMERA] Camera terminated")

#3: WEBSOCKET SERVER
class WebSocketServer:
    """
    WebSocket server for real-time detection streaming.
    Sends detected object data to clients.
    """    
    def __init__(self, host: str, port: int):
        """
        Initialize WebSocket server. 
        Args:
            host: Server IP address
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        
    async def register_client(self, websocket):
        """
        Register a new client.
        Args:
            websocket: WebSocket connection object
        """
        self.clients.add(websocket)
        print(f"[WEBSOCKET] New client connected. Total clients: {len(self.clients)}")
        
    async def unregister_client(self, websocket):
        """Unregister a client"""
        self.clients.remove(websocket)
        print(f"[WEBSOCKET] Client disconnected. Total clients: {len(self.clients)}")
        
    async def broadcast_detections(self, detections_data: Dict[str, Any]):
        """
        Send detections to all connected clients.
        Args:
            detections_data: Dictionary with detection data
        Data format:
        {
            'timestamp': ISO timestamp,
            'frame_id': Frame ID,
            'detections': [
                {
                    'track_id': Tracked object ID,
                    'label': Class name,
                    'confidence': Confidence score (0-1),
                    'x1', 'y1', 'x2', 'y2': Bounding box coordinates
                }
            ],
            'count': Total number of detected objects
        }
        """
        if not self.clients: return
        # Serialize to JSON
        message = json.dumps(detections_data)
        # Broadcast to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        # Remove disconnected clients
        for client in disconnected:
            await self.unregister_client(client)
    
    async def handler(self, websocket, path):
        """
        Handler for WebSocket connections.
        Args:
            websocket: WebSocket connection
            path: URL path (endpoint)
        """
        await self.register_client(websocket)
        try:
            # Keep connection alive
            async for message in websocket:
                # Can receive configuration messages from clients
                pass
        finally:
            await self.unregister_client(websocket)
    
    async def start(self):
        """Start WebSocket server"""
        self.server = await websockets.serve(
            self.handler,
            self.host,
            self.port
        )
        print(f"[WEBSOCKET] Server started at ws://{self.host}:{self.port}")
        
#4: MAIN APPLICATION
class TrafficDetectionApp:
    """
    Main application for traffic detection.
    Combines camera, Hailo inference, and WebSocket communication.
    """    
    def __init__(self):
        """Initialize application"""
        self.camera = CameraManager()
        self.inference = HailoInferenceEngine(Config.MODEL_PATH)
        self.websocket = WebSocketServer(Config.WEBSOCKET_HOST, Config.WEBSOCKET_PORT)
        self.frame_id = 0
        self.is_running = False
        
    async def initialize(self):
        """
        Initialize all components. 
        Steps:
        1. Initialize camera
        2. Initialize Hailo inference engine
        3. Start WebSocket server
        """
        print("=" * 70)
        print("TRAFFIC DETECTION AI - Raspberry Pi AI HAT+ (26 TOPS)")
        # Initialize components
        self.camera.initialize()
        self.inference.initialize()
        await self.websocket.start()
        # Mark application as running
        self.is_running = True
        print("\n[APP] System ready for traffic detection\n")
    
    def process_frame(self) -> Dict[str, Any]:
        """
        Process a single frame.
        Returns: Dictionary with detections ready for sending
        Steps:
        1. Capture frame from camera
        2. Perform inference on Hailo chip
        3. Prepare data for WebSocket
        """
        # Step 1: Capture frame
        frame = self.camera.capture_frame()
        # Step 2: Inference
        detections = self.inference.infer(frame, self.frame_id)
        # Step 3: Prepare data
        timestamp = datetime.utcnow().isoformat() + "Z"
        # Format detections for output
        detections_list = []
        for det in detections:
            detections_list.append({
                'track_id': det.get('track_id', -1),
                'label': det['class_name'],
                'confidence': round(det['confidence'], 3),
                'x1': det['bbox'][0],
                'y1': det['bbox'][1],
                'x2': det['bbox'][2],
                'y2': det['bbox'][3]
            })
        # Compile result
        result = {
            'timestamp': timestamp,
            'frame_id': self.frame_id,
            'detections': detections_list,
            'count': len(detections_list)
        }
        # Increment frame ID
        self.frame_id += 1
        return result
    
    async def run(self):
        """
        Main application loop.  
        Continuously:
        1. Processes frames from camera
        2. Performs detection using YOLOv8m
        3. Sends results via WebSocket
        4. Logs statistics
        """
        print("[APP] Starting detection loop...\n")
        fps_counter = 0
        fps_start = time.time()
        try:
            while self.is_running:
                # Process frame
                detections_data = self.process_frame()
                # Send via WebSocket
                await self.websocket.broadcast_detections(detections_data)
                # FPS monitoring
                fps_counter += 1
                if fps_counter % 30 == 0:  # Every 30 frames
                    fps = 30 / (time.time() - fps_start)
                    print(f"[STATS] Frame: {self.frame_id} | "
                          f"FPS: {fps:.1f} | "
                          f"Detections: {detections_data['count']} | "
                          f"Clients: {len(self.websocket.clients)}")
                    fps_start = time.time()
                # Minimal delay for stability (can be removed for max FPS)
                await asyncio.sleep(0.001)
        except KeyboardInterrupt:
            print("\n[APP] Shutting down on user request...")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Shutdown application and release resources"""
        print("\n[APP] Shutting down application...")
        self.is_running = False
        self.camera.cleanup()
        self.inference.cleanup()
        print("[APP] Application terminated")

#5: ENTRY POINT
async def main():
    """
    Main entry point for the application.
    """
    app = TrafficDetectionApp()
    try:
        await app.initialize() # Initialize
        await app.run() # Run main loop
    except Exception as e:
        print(f"[ERROR] Critical error: {e}")
        import traceback
        traceback.print_exc()

#6: RUN APPLICATION
if __name__ == "__main__": asyncio.run(main()) # Run async application