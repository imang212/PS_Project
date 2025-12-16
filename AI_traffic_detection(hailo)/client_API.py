from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import asyncio
from time import sleep
from contextlib import asynccontextmanager
import subprocess

from DetectionWithGStreamer_pipeline import HailoTrackerPipeline
from ServoControl import ContinuousServo

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

# Global pipeline instance for AI process
pipeline: Optional[HailoTrackerPipeline] = None

# INITIALIZE SERVO/S A MONITORING PIPELINE ON STARTUP AND STOP ON SHUTDOWN
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Handles startup and shutdown services.
    - Startup: Initialize and start pipeline in background task
    - Shutdown: Stop pipeline and cleanup resources
    """
    global pipeline
    global servo_L
    # Startup
    print("[FastAPI] Starting up...")
    pipeline = HailoTrackerPipeline(enable_rtsp=True, enable_websocket=False, enable_webrtc=False, enable_mqtt=False, enable_debug=False)
    servo_L = ContinuousServo(chip=0, pin=18) # FIRST SERVO ON PIN 18
    # Start pipeline in background
    pipeline.create_pipeline()
    pipeline.start() 
    yield
    # Shutdown
    print("[FastAPI] Shutting down...")
    if pipeline: pipeline.stop()
    if servo_L: servo_L.cleanup()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

## CONFIGURATION FOR CAPTURE STORAGE
CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(exist_ok=True)

# Allow frontend access
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)

## SERVO MOVE ENDPOINT
@app.post("/servo/move")
async def move_servo(data: ServoRequest):
    """Move servo from -180 to 180 degrees)"""
    if not (-180 <= data.angle <= 180):
        raise HTTPException(400, "Angle must be between -180–180°")
    try:
	    # set angle
        pulse = await servo_L.rotate_degrees(data.angle,data.speed)
        return {"status": "ok", "angle_set": data.angle, "speed_value": pulse}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Servo error: {str(e)}")

## AI TRAFFIC DETECTION ENDPOINTS
# HEALTH AI CHECK ENDPOINT
@app.get("/health/ai", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns current system status:
    - Status (running/stopped)
    - Actual FPS
    - Total tracked objects
    - Current timestamp
    Returns: HealthResponse with system metrics
    """
    if pipeline is None: raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return HealthResponse(
        status="running" if pipeline.running else "stopped",
        fps=pipeline.fps_actual,
        total_tracked=pipeline.counter.get_total_count(),
        timestamp=datetime.now().isoformat()
    )

# SHOW RECENT DETECTIONS FROM DATABASE ENDPOINT
@app.get("/api/detections", response_model=List[DetectionResponse])
async def get_detections(minutes: int = 5, limit: int = 100):
    """
    Get recent detections from database.
    Query parameters:
    - minutes: Look back N minutes (default: 5)
    - limit: Maximum records to return (default: 100)
    Args:
        minutes: Time window in minutes
        limit: Max number of records
    Returns:
        List of recent detections
    Example:
        GET /api/detections?minutes=10&limit=50
    """
    if pipeline is None: raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    detections = pipeline.db.get_recent_detections(minutes, limit)
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

# STATISTICS FROM DB ENDPOINT
@app.get("/api/statistics", response_model=Dict[str, StatisticsResponse])
async def get_statistics(hours: int = 1):
    """
    Get aggregated traffic statistics.
    Returns count and average confidence for each object type
    within the specified time window.
    Args:
        hours: Time window in hours (default: 1)
    Returns:
        Dictionary mapping object labels to statistics
    Example:
        GET /api/statistics?hours=24
        Response:
        {"car": {"count": 150, "avg_confidence": 0.92}, "truck": {"count": 23, "avg_confidence": 0.88}, "person": {"count": 45, "avg_confidence": 0.85}}
    """
    if pipeline is None: raise HTTPException(status_code=503, detail="Pipeline not initialized")
    stats = pipeline.db.get_statistics(hours)
    return {
        label: StatisticsResponse(
            label=label,
            count=data['count'],
            avg_confidence=data['avg_confidence']
        )
        for label, data in stats.items()
    }

# REAL-TIME CURRENT STATE ENDPOINT
@app.get("/api/current/ai")
async def get_current_state():
    """
    Get current real-time state.
    Returns:
    - Current frame_id
    - Active detections in current frame
    - Real-time statistics
    - Current FPS
    Returns:
        Current system state
    This is useful for getting immediate state without WebSocket connection.
    """
    if pipeline is None: raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {
        "frame_id": pipeline.frame_id,
        "timestamp": datetime.now().isoformat(),
        "detections": [det.to_dict() for det in pipeline.current_detections],
        "statistics": {
            "total_count": pipeline.counter.get_total_count(),
            "counts_by_type": pipeline.counter.get_counts(),
            "fps": round(pipeline.fps_actual, 1)
        }
    }

# BULK INSERT MULTIPLE DETECTIONS ENDPOINT
@app.post("/api/detections/bulk")
async def create_bulk_detections(detections: List[Dict[str, Any]]):
    """
    Manually insert bulk detections into database.    
    This endpoint allows external systems to push detection data into the database. Useful for testing or integrating external detectors.
    Args:
        detections: List of detection dictionaries
    Returns:
        Success message with count
    Example request body:
    [{"timestamp": "2024-11-20T10:30:00","frame_id": 0,"label": "car","confidence": 0.95, "x1": 100, "y1": 100, "x2": 200, "y2": 200, "track_id": 1, "total_count": 1}]
    """
    if pipeline is None: raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline.db.insert_batch_detections(detections)
    return {
        "message": "Detections inserted successfully",
        "count": len(detections)
    }

# WEBSOCKET ENDPOINT FOR REAL-TIME STREAMING
@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection streaming.
    Protocol:
    1. Client connects to ws://server:8000/ws
    2. Server accepts connection
    3. Server continuously sends detection data as JSON
    4. Client receives updates in real-time
    Usage (JavaScript):
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Detections:', data.detections);
        console.log('Statistics:', data.statistics);
    };
    ```
    Args:
        websocket: FastAPI WebSocket connection
    """
    if pipeline is None:
        await websocket.close(code=1011, reason="Pipeline not initialized")
        return
    # Accept and register connection
    await pipeline.ws_manager.connect(websocket)
    try:
        # Keep connection alive and handle incoming messages
        while True:
            # Wait for messages from client (e.g., commands, config changes)
            data = await websocket.receive_text()
            # Echo back or process commands
            # For now, just acknowledge
            await websocket.send_json({
                "type": "ack",
                "message": "Message received",
                "data": data
            })
    except WebSocketDisconnect:
        # Client disconnected
        pipeline.ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        pipeline.ws_manager.disconnect(websocket)

# ENDPOINT FOR WEBSOCKET TESTING
@app.get("/test-ws", response_class=HTMLResponse)
async def test_websocket_page():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>WebSocket Tester</title>
        <style>body { font-family: Arial; margin: 20px; } #log { white-space: pre-wrap; background: #eee; padding: 10px; border-radius: 8px; }</style>
    </head>
    <body>
        <h2>WebSocket Test Page</h2>
        <p>Připojuji se na: <b>wss://192.168.37.205:8000/ws/ai</b></p>
        <button onclick="sendTest()">Send Test Message</button>
        <h3>Log:</h3>
        <div id="log"></div>
        <script>
            let logDiv = document.getElementById("log");
            function log(msg) {
                logDiv.textContent += msg + "\\n";
            }
            // Připojení na WebSocket server
            let ws = new WebSocket("wss://192.168.37.205:8000/ws/ai");
            ws.onopen = () => {
                log("WebSocket připojen!");
            };
            ws.onmessage = (event) => {
                log("Přišla zpráva: " + event.data);
            };
            ws.onerror = (error) => {
                log("Chyba: " + error);
            };
            ws.onclose = () => {
                log("WebSocket byl uzavřen.");
            };
            function sendTest() {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ test: "Hello from browser!" }));
                    log("Odesláno: Hello from browser!");
                } else {
                    log("WebSocket není připojen!");
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# MJPEG VIDEO AI STREAM ENDPOINT
@app.get("/stream/ai")
async def video_stream():
    """
    MJPEG video stream endpoint.
    Streams annotated video frames as Motion JPEG.
    Can be viewed directly in browser or embedded in <img> tag.
    Usage:
    <img src="http://localhost:8000/stream" />
    Returns: StreamingResponse with MJPEG video
    Note: This is more resource-intensive than WebSocket.
    Use WebSocket for detection data and render on frontend instead.
    """
    from fastapi.responses import StreamingResponse
    import cv2
    if pipeline is None: raise HTTPException(status_code=503, detail="Pipeline not initialized")
    async def generate_frames():
        """Generate MJPEG frames."""
        while True:
            frame = pipeline.get_annotated_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(1.0 / pipeline.fps)
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

## CAMERA ENDPOINTS
# CAMERA CAPTURE (PHOTO)
@app.post("/camera/stream/capture")
async def camera_capture(settings: Optional[CameraSettings] = None):
    """Capture a single photo from the camera"""
    if settings is None:
        settings = CameraSettings()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = CAPTURE_DIR / f"capture_{timestamp}.jpg"
    try:
        result = subprocess.run([
            "rpicam-still",
            "-n",
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

# CAMERA STREAM (SNAPSHOT STREAM)
@app.get("/camera/stream/snapshots")
def stream_snapshots(width: int = 1536, height: int = 864, framerate: int = 30):
    """
    Stream jednotlivých JPEG snímků
    Spolehlivější než video stream
    """
    def generate_snapshots():
        frame_delay = 1.0 / framerate
        try:
            while True:
                # Zachytíme jeden frame
                temp_img = "/tmp/stream_frame.jpg"               
                result = subprocess.run([
                    "rpicam-still",
                    "-n",
                    "--width", str(width),
                    "--height", str(height),
                    "-o", temp_img,
                    "-t", "1"  # Rychlé zachycení
                ], capture_output=True, timeout=2)
                if result.returncode != 0:
                    sleep(frame_delay)
                    continue
                # Načteme obrázek
                with open(temp_img, 'rb') as f:
                    frame_bytes = f.read()
                # Pošleme jako MJPEG frame
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

#CAMERA VIEWER (IN HTML)
@app.get("/camera/stream/hls")
async def camera_viewer():
    """
    Jednoduchá HTML stránka pro zobrazení streamu
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi Camera Stream</title>
        <style> body { font-family: Arial, sans-serif; max-width: 1200px; margin: 50px auto; padding: 20px; background: #1a1a1a; color: white;} .stream-container { margin: 20px 0; border: 2px solid #333; border-radius: 8px; overflow: hidden;} img { width: 100%; height: auto; display: block; } h2 { color: #4CAF50; } .info { background: #333; padding: 15px; border-radius: 5px; margin: 10px 0; } button { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px;} button:hover { background: #45a049; }</style>
    </head>
    <body>
        <h1>Raspberry Pi Camera Stream</h1>
        <div class="stream-container">
            <img id="stream" src="/camera/stream/snapshots" alt="Camera Stream">
        </div> 
        <script>
            function switchStream(type) {
                const streamImg = document.getElementById('stream');
                const currentStream = document.getElementById('current-stream');
                const streamUrl = `/camera/stream/${type}`;
                streamImg.src = streamUrl;
                currentStream.textContent = streamUrl;
                console.log('Switching to:', streamUrl);
            } 
            // Kontrola stavu streamu
            const streamImg = document.getElementById('stream');
            const status = document.getElementById('status');
            streamImg.onload = () => { status.textContent = 'Running'; status.style.color = '#4CAF50'; };
            streamImg.onerror = () => { status.textContent = 'Error'; status.style.color = '#f44336'; };
        </script>
    </body>
    </html>
    """    
    return Response(content=html_content, media_type="text/html")

# UTILITY ENDPOINTS FOR CAPTURES STORAGE MANAGEMENT
@app.get("/captures")
async def list_captures():
    """List all captured images"""
    captures = sorted(CAPTURE_DIR.glob("*.jpg"), reverse=True)
    return {
        "total": len(captures),
        "captures": [f.name for f in captures[:20]]  # Last 20
    }

@app.delete("/captures/{filename}")
async def delete_capture(filename: str):
    """Delete a specific capture"""
    filepath = CAPTURE_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    filepath.unlink()
    return {"status": "deleted", "filename": filename}

# HEALTH CHECK ENDPOINT
@app.get("/health/camera")
async def health_check():
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
    
# ROOT ENDPOINT
@app.get("/")
def root():
    return {
        "message": "Raspberry Pi Control API",
        "version": "1.0.0",
        "endpoints": {
            "camera": {
                "capture": "POST /camera/capture",
                "stream": "GET /camera/stream",
            },
            "utility": {
                "list_captures": "GET /captures",
                "delete_capture": "DELETE /captures/{filename}",
                "health": "GET /health",
            },
            "AI_traffic_detection": {
                "health": "/health/ai",
                "detections": "/api/detections",
                "statistics": "/api/statistics",
                "websocket": "/ws/ai",
                "video_stream": "/stream/ai"
            }
        }
    }

if __name__ == "__main__":
    """
    Start FastAPI server with Uvicorn.
    Configuration:
    - host: 0.0.0.0 (accessible from other devices on network)
    - port: 8000
    - log_level: info
    Run:
        python script.py
    Or with custom settings:
        uvicorn script:app --host 0.0.0.0 --port 8000 --reload
    Access:
    - API Docs: http://localhost:8000/docs
    - WebSocket: ws://localhost:8000/ws
    - Video Stream: http://localhost:8000/stream
    """
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        log_level="info",
        access_log=True
    )