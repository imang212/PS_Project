from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict, Any
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
from yolo_traffic_detection import TrafficMonitoringPipeline

class DetectionResponse(BaseModel):
    """Response model for detection data via REST API."""
    timestamp: str
    frame_id: int
    label: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    track_id: int
    total_count: int

class StatisticsResponse(BaseModel):
    """Response model for traffic statistics."""
    label: str
    count: int
    avg_confidence: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    fps: float
    total_tracked: int
    timestamp: str

# Global pipeline instance
pipeline: Optional[TrafficMonitoringPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Handles startup and shutdown of the traffic monitoring pipeline.
    - Startup: Initialize and start pipeline in background task
    - Shutdown: Stop pipeline and cleanup resources
    """
    global pipeline
    # Startup
    print("[FastAPI] Starting up...")
    pipeline = TrafficMonitoringPipeline(model_path="yolo11n.pt", video_source=0, db_path="traffic_data.db", resolution=(1536, 864), fps=15)
    # Start pipeline in background
    asyncio.create_task(pipeline.start()) 
    yield
    # Shutdown
    print("[FastAPI] Shutting down...")
    if pipeline: pipeline.stop()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
# Allow frontend access
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)
# REST API ENDPOINTS
@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint - API information.
    Returns: Basic API info and available endpoints
    """
    return {
        "message": "YOLO11 Traffic Monitoring API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "detections": "/api/detections",
            "statistics": "/api/statistics",
            "websocket": "/ws"
        }
    }

@app.get("/health", response_model=HealthResponse)
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

# show recent detections from database
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

# statistics from database
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

# real-time current state
@app.get("/api/current")
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
        port=5000,
        log_level="info",
        access_log=True
    )