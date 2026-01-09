"""
API Integration Module
Bridges FastAPI backend (client_API.py) with NiceGUI frontend (PiManagerApp)
Allows NiceGUI to proxy/expose FastAPI endpoints and share state
"""
from fastapi import HTTPException, FastAPI
from fastapi.responses import StreamingResponse, FileResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from nicegui import app, ui
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import subprocess
from time import sleep

from contextlib import asynccontextmanager

from hardware.ServoControl import ContinuousServo

pipeline = None

servo_L = ContinuousServo(chip=0, pin=18) # FIRST SERVO ON PIN 18

## CONFIGURATION FOR CAPTURE STORAGE
CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(exist_ok=True)

## MODELS
# Servo control request
class ServoRequest(BaseModel):
    angle: int = Field(..., ge=-180, le=180, description="Servo position (-180 to 180 degrees)")
    speed: int = Field(50, ge=0, le=100, description="Speed percentage (0 to 100%)")
# Camera settings for capture
class CameraSettings(BaseModel):
    width: int = Field(1536, ge=1536, le=4608)
    height: int = Field(864, ge=864, le=2592)
# Detection response model
class DetectionResponse(BaseModel):
    """Response model for detection data via REST API."""
    timestamp: str
    frame_id: int
    label: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    track_id: int
    total_count: int
# Statistics response model
class StatisticsResponse(BaseModel):
    """Response model for traffic statistics."""
    label: str
    count: int
    avg_confidence: float
# Health check response model
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    fps: float
    total_tracked: int
    timestamp: str

class APIIntegration:
    """
    Integrates FastAPI backend endpoints into NiceGUI application.
    Provides proxy methods and direct access to backend functionality.
    """
    def __init__(self):
        self.pipeline = pipeline
        self.servo = servo_L
        self.capture_dir = CAPTURE_DIR
        self.capture_dir.mkdir(exist_ok=True)
    
    def register_endpoints(self):
        """
        Register all FastAPI endpoints from client_API.py into NiceGUI's FastAPI app.
        This makes them accessible through the same server instance.
        """ 
        # SERVO ENDPOINTS
        @app.post("/servo/move")
        async def move_servo(data: ServoRequest):
            """Move servo from -180 to 180 degrees"""
            if not (-180 <= data.angle <= 180):
                raise HTTPException(400, "Angle must be between -180–180°")
            try:
                pulse = await self.servo.rotate_degrees(data.angle, data.speed)
                return {"status": "ok", "angle_set": data.angle, "speed_value": pulse}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Servo error: {str(e)}")
        
        # AI TRAFFIC DETECTION ENDPOINTS
        @app.get("/health/ai", response_model=HealthResponse)
        async def health_check_ai():
            """Health check for AI pipeline"""
            if self.pipeline is None:
                raise HTTPException(status_code=503, detail="Pipeline not initialized")
            return HealthResponse(
                status="running" if self.pipeline else "stopped",
                fps=self.pipeline.fps_actual,
                total_tracked=self.pipeline.counter.get_total_count(),
                timestamp=datetime.now().isoformat()
            )
        
        @app.get("/api/detections", response_model=List[DetectionResponse])
        async def get_detections(minutes: int = 5, limit: int = 100):
            """Get recent detections from database"""
            if self.pipeline is None:
                raise HTTPException(status_code=503, detail="Pipeline not initialized")    
            detections = MQTTListener.get_recent_detections(minutes, limit)
            return [
                DetectionResponse(
                    timestamp=d['timestamp'],
                    frame_id=d['frame_id'],
                    label=d['label'],
                    confidence=d['confidence'],
                    bbox=[d['x1'], d['y1'], d['x2'], d['y2']],
                    track_id=d['track_id'],
                    total_count=d['total_count']
                )
                for d in detections
            ]
        
        @app.get("/api/statistics", response_model=Dict[str, StatisticsResponse])
        async def get_statistics(hours: int = 1):
            """Get aggregated traffic statistics"""
            if self.pipeline is None:
                raise HTTPException(status_code=503, detail="Pipeline not initialized")
            stats = MQTTListener.get_statistics(hours)
            return {
                label: StatisticsResponse(
                    label=label,
                    count=data['count'],
                    avg_confidence=data['avg_confidence']
                )
                for label, data in stats.items()
            }
        
        @app.get("/api/current/ai")
        async def get_current_state():
            """Get current real-time AI state"""
            if self.pipeline is None:
                raise HTTPException(status_code=503, detail="Pipeline not initialized")
            return {
                "frame_id": self.pipeline.frame_id,
                "timestamp": datetime.now().isoformat(),
                "detections": [det.to_dict() for det in self.pipeline.current_detections],
                "statistics": {
                    "total_count": self.pipeline.counter.get_total_count(),
                    "counts_by_type": self.pipeline.counter.get_counts(),
                    "fps": round(self.pipeline.fps_actual, 1)
                }
            }
        
        @app.post("/api/detections/bulk")
        async def create_bulk_detections(detections: List[Dict[str, Any]]):
            """Manually insert bulk detections into database"""
            if self.pipeline is None:
                raise HTTPException(status_code=503, detail="Pipeline not initialized")
            MQTTListener.insert_batch_detections(detections)
            return {
                "message": "Detections inserted successfully",
                "count": len(detections)
            }
        
        # CAMERA ENDPOINTS
        @app.post("/camera/stream/capture")
        async def camera_capture(settings: Optional[CameraSettings] = None):
            """Capture a single photo from the camera"""
            if settings is None:
                settings = CameraSettings()    
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = self.capture_dir / f"capture_{timestamp}.jpg"
            try:
                result = subprocess.run([
                    "rpicam-still", "-n",
                    "--width", str(settings.width),
                    "--height", str(settings.height),
                    "-o", str(output)
                ], capture_output=True, timeout=10)
                if result.returncode != 0:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Camera capture failed: {result.stderr.decode()}"
                    )
                if not output.exists():
                    raise HTTPException(status_code=500, detail="Output file not created")
                return FileResponse(output, media_type="image/jpeg", filename=output.name)
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=500, detail="Camera capture timeout")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/camera/stream/snapshots")
        def stream_snapshots(width: int = 1536, height: int = 864, framerate: int = 30):
            """Stream individual JPEG snapshots"""
            def generate_snapshots():
                frame_delay = 1.0 / framerate
                try:
                    while True:
                        temp_img = "/tmp/stream_frame.jpg"
                        result = subprocess.run([
                            "rpicam-still", "-n",
                            "--width", str(width),
                            "--height", str(height),
                            "-o", temp_img,
                            "-t", "1"
                        ], capture_output=True, timeout=2)
                        if result.returncode != 0:
                            sleep(frame_delay)
                            continue
                        with open(temp_img, 'rb') as f:
                            frame_bytes = f.read()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' +
                               frame_bytes + b'\r\n')         
                        sleep(frame_delay)
                except Exception as e:
                    print(f"Stream error: {e}")    
            return StreamingResponse(
                generate_snapshots(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        
        @app.get("/camera/stream/hls")
        async def camera_viewer():
            """Simple HTML page for stream display"""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Raspberry Pi Camera Stream</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 1200px; 
                           margin: 50px auto; padding: 20px; background: #1a1a1a; color: white;}
                    .stream-container { margin: 20px 0; border: 2px solid #333; 
                                      border-radius: 8px; overflow: hidden;}
                    img { width: 100%; height: auto; display: block; }
                    h2 { color: #4CAF50; }
                </style>
            </head>
            <body>
                <h1>Raspberry Pi Camera Stream</h1>
                <div class="stream-container">
                    <img id="stream" src="/camera/stream/snapshots" alt="Camera Stream">
                </div>
            </body>
            </html>
            """
            return Response(content=html_content, media_type="text/html")
        
        # UTILITY ENDPOINTS
        @app.get("/captures")
        async def list_captures():
            """List all captured images"""
            captures = sorted(self.capture_dir.glob("*.jpg"), reverse=True)
            return {
                "total": len(captures),
                "captures": [f.name for f in captures[:20]]
            }
        
        @app.delete("/captures/{filename}")
        async def delete_capture(filename: str):
            """Delete a specific capture"""
            filepath = self.capture_dir / filename
            if not filepath.exists():
                raise HTTPException(status_code=404, detail="File not found")
            filepath.unlink()
            return {"status": "deleted", "filename": filename}
        
        @app.get("/health/camera")
        async def health_check_camera():
            """Check if camera is accessible"""
            try:
                result = subprocess.run(
                    ["rpicam-hello", "--list-cameras"],
                    capture_output=True,
                    timeout=5
                )
                return {
                    "status": "healthy",
                    "camera_available": result.returncode == 0
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        @app.get("/test-ws", response_class=HTMLResponse)
        async def test_websocket_page():
            """WebSocket test page"""
            html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8" />
                <title>WebSocket Tester</title>
            </head>
            <body>
                <h2>WebSocket Test Page</h2>
                <button onclick="sendTest()">Send Test Message</button>
                <div id="log"></div>
                <script>
                    let logDiv = document.getElementById("log");
                    function log(msg) { logDiv.textContent += msg + "\\n"; }
                    let ws = new WebSocket("wss://192.168.37.205:8000/ws/ai");
                    ws.onopen = () => log("WebSocket connected!");
                    ws.onmessage = (event) => log("Message: " + event.data);
                    ws.onerror = (error) => log("Error: " + error);
                    ws.onclose = () => log("WebSocket closed.");
                    function sendTest() {
                        if (ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({ test: "Hello!" }));
                            log("Sent: Hello!");
                        } else {
                            log("WebSocket not connected!");
                        }
                    }
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=html)
        
        @app.get("/api")
        def root():
            """API documentation"""
            return {
                "message": "Raspberry Pi Control API (Integrated)",
                "version": "1.0.0",
                "endpoints": {
                    "camera": {
                        "capture": "POST /camera/stream/capture",
                        "stream": "GET /camera/stream/snapshots",
                        "viewer": "GET /camera/stream/hls",
                    },
                    "utility": {
                        "list_captures": "GET /captures",
                        "delete_capture": "DELETE /captures/{filename}",
                        "health": "GET /health/camera",
                        "servo_move": "POST /servo/move",
                    },
                    "AI_traffic_detection": {
                        "health": "/health/ai",
                        "detections": "/api/detections",
                        "statistics": "/api/statistics",
                        "bulk_insert": "/api/detections/bulk",
                        "current_state": "/api/current/ai",
                    }
                }
            }

# Singleton instance
_api_integration = None

def get_api_integration():
    """Get or create the APIIntegration singleton"""
    global _api_integration
    if _api_integration is None:
        _api_integration = APIIntegration()
        _api_integration.register_endpoints()
    return _api_integration