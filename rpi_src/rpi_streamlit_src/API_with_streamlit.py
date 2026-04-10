"""
Combined FastAPI + Streamlit Application
Run with: python combined_app.py
"""
import sys
import subprocess
import multiprocessing
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from time import sleep
import threading
from pathlib import Path
    
from DatabaseManagerPostgre import PostgreDatabaseManager
from MQTTClient import MQTTPublisher
from ServoControl import ContinuousServo
try:
    from pipeline.DetectionWithGStreamer_pipeline import HailoTrackerPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("[WARNING] Pipeline module not found")

## MODELS
# Servo control request
class ServoRequest(BaseModel):
    angle: int = Field(..., ge=-180, le=180, description="Servo position (-180 to 180 degrees)")
    speed: int = Field(50, ge=0, le=100, description="Speed percentage (0 to 100%)")
# Camera settings for capture
class CameraSettings(BaseModel):
    width: int = Field(1536, ge=1536, le=4608)
    height: int = Field(864, ge=864, le=2592)
# Pipeline configuration model
class PipelineConfig(BaseModel):
    """Configuration for starting AI pipeline"""
    rpicamera: bool = Field(False, description="Use Raspberry Pi camera")
    video_file: Optional[str] = Field(None, description="Path to video file")
    enable_rtsp: bool = Field(False, description="Enable RTSP streaming")
    enable_websocket: bool = Field(False, description="Enable WebSocket")
    enable_mqtt: bool = Field(True, description="Enable MQTT")
    enable_recording: bool = Field(False, description="Enable video recording")
    output_path: Optional[str] = Field(None, description="Output video path")
    enable_debug: bool = Field(False, description="Enable debug mode")
# Pipeline status response model
class PipelineStatus(BaseModel):
    """Pipeline status response"""
    running: bool
    mode: Optional[str]
    fps: Optional[float]
    total_tracked: Optional[int]
# Detection response model
class DetectionResponse(BaseModel):
    """Response model for detection data via REST API."""
    id: int
    timestamp: datetime
    frame_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    track_id: Optional[int]
# Statistics response model
class StatisticsResponse(BaseModel):
    """Response model for traffic statistics."""
    class_name: str
    total_detections: int
    unique_tracks: int
    avg_confidence: float
    min_confidence: float
    max_confidence: float
#  Health check response model for database
class HealthResponseDB(BaseModel):
    """Health check response."""
    status: str
    total_detections: int
    latest_detection: Optional[str]
    database_size: str
    mqtt_sending: bool
# Database configuration
class DatabaseConfig(BaseModel):
    mqtt_sending: bool = Field(False, description="Enable MQTT sending")

# Global instances
db: Optional[PostgreDatabaseManager] = None
servo_L: Optional[ContinuousServo] = None
pipeline: Optional[HailoTrackerPipeline] = None
pipeline_thread: Optional[threading.Thread] = None
pipeline_config: Optional[Dict[str, Any]] = None
pipeline_running: bool = False
mqtt_client: Optional[MQTTPublisher] = None
mqtt_client_sending: bool = False

DB_CONFIG = {'host': '0.0.0.0', 'port': 5432, 'database': 'hailo_db', 'user': 'hailo_user', 'password': 'hailo_pass', 'min_connections': 2, 'max_connections': 10}
MQTT_CONFIG = {'broker_host': 'mqtt.portabo.cz', 'broker_port': 8883, 'topic': '/videoanalyza', 'client_id': 'patrik_dashboard_visualization_listener', 'username': 'videoanalyza', 'password': 'phdA9ZNW1vfkXdJkhhbP'}

# Initialize FastAPI
app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)

@app.on_event("startup")
async def startup():
    global servo_L, db, mqtt_client, mqtt_client_sending
    # Startup
    print("[FastAPI] Starting up...")
    # Initialize database
    try:
        # Create database manager instance
        db = PostgreDatabaseManager(DB_CONFIG)
        print("[FastAPI] Database connected successfully")
    except Exception as e:
        print(f"[FastAPI] Database connection failed: {e}")
        db = None
    # Initialize servo
    try:
        servo_L = ContinuousServo(chip=0, pin=18) # FIRST SERVO ON PIN 18
        print("[FastAPI] Servo initialized")
    except Exception as e:
        print(f"[FastAPI] Servo initialization failed: {e}")
        servo_L = None
@app.on_event("shutdown")
async def shutdown():
    global servo_L, db, mqtt_client, mqtt_client_sending
    print("[FastAPI] Shutting down...")
    if servo_L: 
        servo_L.cleanup()
    if db: 
        db.close()
        if mqtt_client_sending:
            mqtt_client.disconnect()
            mqtt_client = None
            mqtt_client_sending = False

## CONFIGURATION FOR CAPTURE STORAGE
CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(exist_ok=True)

## HELPER FUNCTIONS
def check_pipeline_instance():
    """Check if pipeline instance exists and is properly initialized"""
    global pipeline_running
    if pipeline_running is False:
        return False, "Pipeline not initialized"
    return True, "Pipeline instance available"
def check_db_instance():
    """Check if database instance exists and is connected"""
    global db
    if db is None:
        return False, "Database not initialized"
    try:
        if db.verify_pool():
            return True, "Database connected"
        else:
            return False, "Database connection pool verification failed"
    except Exception as e:
        return False, f"Database error: {str(e)}"

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

## DATABASE MANAGEMENT ENDPOINTS
@app.get("/database/status")
async def database_status():
    """Check database connection status"""
    global db, mqtt_client, mqtt_client_sending
    is_connected = check_db_instance()
    if is_connected:
        try:
            health = db.get_health_status() 
            if mqtt_client_sending is True:
                health['mqtt_sending'] = mqtt_client_sending
            return health 
        except Exception as e:
            return { "connected": False, "status": "error", "message": f"Health check failed: {str(e)}", "mqtt_sending": mqtt_client_sending }    
    return { "connected": False, "status": "disconnected", "message": "Database not connected", "mqtt_sending": mqtt_client_sending }

@app.post("/database/start")
async def database_start(data: DatabaseConfig):
    """Start/initialize database connection"""
    global db, mqtt_client, mqtt_client_sending    
    if db is not None:
        is_connected = check_db_instance()
        if is_connected: 
            return { "status": "already_running", "message": "Database and MQTT client are already connected", "mqtt": mqtt_client_sending }
    try:
        db = PostgreDatabaseManager(DB_CONFIG) # create new db instance
        if data.mqtt_sending is True:
            mqtt_client = MQTTPublisher(MQTT_CONFIG) # create new mqtt client instance
            mqtt_client.client.user_data_set({"db": db})
            mqtt_client.client.on_message = mqtt_client.on_message
            mqtt_client.connect({"db": db})
            await asyncio.sleep(2)  # Wait for connection
            mqtt_client_sending = True
            print("[FastAPI] MQTT client connected for sending")
        return { "status": "started", "message": "Database connection established successfully", "mqtt": mqtt_client_sending }
    except Exception as e:
        db = None
        if mqtt_client_sending:
            mqtt_client.disconnect()
            mqtt_client = None
            mqtt_client_sending = False
        raise HTTPException( status_code=500, detail=f"Failed to start database connection: {str(e)}" )

@app.post("/database/stop")
async def database_stop():
    """Stop/close database connection"""
    global db, mqtt_client, mqtt_client_sending
    if db is None:
        return { "status": "already_stopped", "message": "Database is not running" }
    try:
        db.close()
        if mqtt_client_sending:
            mqtt_client.disconnect()
        db = None
        mqtt_client = None
        mqtt_client_sending = False
        return { "status": "stopped", "message": "Database connection closed successfully", "mqtt": mqtt_client_sending }
    except Exception as e:
        raise HTTPException( status_code=500, detail=f"Failed to stop database: {str(e)}" )

## DATABASE QUERIES ENDPOINTS
# STATISTICS FROM DB ENDPOINT
@app.get("/api/detections", response_model=List[DetectionResponse])
async def get_detections(minutes: int = 5, limit: int = 100):
    """
    Get recent detections from database.
    Query parameters:
    - minutes: Look back N minutes (default: 5)
    - limit: Maximum records to return (default: 100)
    Example: GET /api/detections?minutes=10&limit=50
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        detections = db.get_recent_detections(minutes=minutes, limit=limit)
        #print(f"[FastAPI] Retrieved {len(detections)} detections from DB")
        #print(detections[0:5])  # Print first 5 detections for debugging
        #print(detections[0:1].keys(), detections[0:1].items())  # Print keys of the first detection
        result = []
        for d in detections:
            try:
                d_dict = dict(d)  # make sure it's a dict
                result.append(
                    DetectionResponse(
                        id=d_dict['id'],
                        timestamp=d_dict['timestamp_'],  # or d_dict['timestamp_'].isoformat()
                        frame_id=d_dict['frame_id'],
                        class_name=d_dict['class_name'],
                        confidence=d_dict['confidence'],
                        bbox=[d_dict['x1'], d_dict['y1'], d_dict['x2'], d_dict['y2']],
                        track_id=d_dict['track_id']
                    )
                )
            except Exception as inner_e:
                print(f"[FastAPI] Error converting detection: {inner_e}, data: {d}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching detections: {str(e)}")

@app.get("/api/detection_counts", response_model=List[Dict[str, Any]])
async def get_detections(minutes: int = 5, per: str = "10 minutes"):
    """
    Get recent detections from database.
    Query parameters:;
    - minutes: Look back N minutes (default: 5)
    - per: Time bucket size (default: "10 minutes")
    Example: GET /api/detections?minutes=10&limit=50
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        detections = db.get_counts_detections(minutes=minutes, per=per)
        #print(f"[FastAPI] Retrieved {len(detections)} detections from DB")
        #print(detections[0:5])  # Print first 5 detections for debugging
        #print(detections[0:1].keys(), detections[0:1].items())  # Print keys of the first detection
        result = []
        for d in detections:
            try:
                d_dict = dict(d)  # make sure it's a dict
                result.append(
                    (d_dict['time_bucket'], d_dict['detection_count'])
                )
            except Exception as inner_e:
                print(f"[FastAPI] Error converting detection: {inner_e}, data: {d}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching detections: {str(e)}")

@app.get("/api/statistics", response_model=Dict[str, StatisticsResponse])
async def get_statistics(hours: int = 1):
    """
    Get aggregated traffic statistics.
    Returns count and average confidence for each object type
    within the specified time window.
    Args:
        hours: Time window in hours (default: 1)
    Example: GET /api/statistics?hours=24
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        stats = db.get_statistics(hours=hours)
        return {
            class_name: StatisticsResponse(
                class_name=class_name,
                total_detections=data['total_detections'],
                unique_tracks=data['unique_tracks'],
                avg_confidence=data['avg_confidence'],
                min_confidence=data['min_confidence'],
                max_confidence=data['max_confidence']
            )
            for class_name, data in stats.items()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching statistics: {str(e)}")

## AI PIPELINE DETECTION ENDPOINTS
@app.get("/pipeline/status", response_model=PipelineStatus)
async def pipeline_status():
    """Get current pipeline status"""
    global pipeline, pipeline_config, pipeline_running
    if not PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pipeline module not available" ) 
    if pipeline is None:
        return PipelineStatus(running=False, mode=None, fps=None, total_tracked=None, config=None )
    try:
        # Determine mode
        mode = "unknown"
        if pipeline_config:
            if pipeline_config.get('video_file'):
                mode = "video_file"
            elif pipeline_config.get('rpicamera'):
                mode = "rpicamera"
            else:
                mode = "rtsp_stream"
        print(f"[FastAPI] Pipeline status requested. Running: {pipeline_running}, Mode: {mode}, FPS: {getattr(pipeline, 'frame_count', 0.0)}, Total Tracked: {getattr(pipeline.detection_data, 'get_count_overlay_text', lambda: 0)() if hasattr(pipeline, 'detection_data') else 0}")
        return PipelineStatus(
            running=pipeline_running,
            mode=mode,
            fps= pipeline.frame_count if hasattr(pipeline, 'frame_count') else 0.0,
            total_tracked=getattr(pipeline.detection_data, 'get_count_overlay_text', lambda: 0)() if hasattr(pipeline, 'detection_data') else 0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting pipeline status: {str(e)}" )

@app.post("/pipeline/start")
async def pipeline_start(config: PipelineConfig):
    """Start AI detection pipeline with specified configuration"""
    global pipeline, pipeline_thread, pipeline_config, pipeline_running
    if not PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Pipeline module not available" )
    # Check if already running
    if pipeline_running:
        raise HTTPException(status_code=409, detail="Pipeline is already running. Stop it first." )
    # Check database connection
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected. Start database first." )
    try:
        # Generate output path if recording enabled
        output_path = config.output_path
        if config.enable_recording and not output_path and not config.video_file:
            if config.video_file:
                import os
                base_name = os.path.splitext(config.video_file)[0]
                output_path = f"{base_name}_processed.mp4"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"hailo_output_{timestamp}.mp4"
        # Store configuration
        pipeline_config = config.dict()
        pipeline_config['output_path'] = output_path
        # Create pipeline instance
        pipeline = HailoTrackerPipeline(
            rpicamera=config.rpicamera,
            enable_rtsp=config.enable_rtsp,
            enable_websocket=config.enable_websocket,
            enable_mqtt=config.enable_mqtt,
            enable_recording=config.enable_recording,
            enable_debug=config.enable_debug,
        )
        # Create and start pipeline
        pipeline.create_pipeline()
        # Start in separate thread
        def run_pipeline():
            try:
                pipeline.start()
            except Exception as e:
                print(f"[Pipeline Thread] Error: {e}")
        pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
        pipeline_thread.start()
        pipeline_running = True
        # Wait a moment to ensure it starts
        await asyncio.sleep(1)
        return { "status": "started", "message": "Pipeline started successfully", "config": pipeline_config, "running": pipeline_running }
    except Exception as e:
        pipeline = None
        pipeline_config = None
        raise HTTPException( status_code=500, detail=f"Failed to start pipeline: {str(e)}" )

@app.post("/pipeline/stop")
async def pipeline_stop():
    """Stop AI detection pipeline"""
    global pipeline, pipeline_config, pipeline_running
    if pipeline is None:
        return { "status": "already_stopped", "message": "Pipeline is not running" }
    try:
        # Stop the pipeline
        if hasattr(pipeline, 'stop') and pipeline_running == True:
            pipeline.stop()
        # Clear references
        pipeline = None
        pipeline_config = None
        pipeline_running = False
        return { "status": "stopped", "message": "Pipeline stopped successfully" }
    except Exception as e:
        raise HTTPException( status_code=500, detail=f"Failed to stop pipeline: {str(e)}" )

@app.get("/pipeline/modes")
async def pipeline_modes():
    """Get available pipeline modes and their descriptions"""
    return {
        "modes": {
            "rpicamera": {
                "description": "Use Raspberry Pi Camera (IMX708)",
                "config": {"rpicamera": True}
            },
            "rtsp_stream": {
                "description": "Use RTSP IP Camera Stream",
                "config": {"rpicamera": False, "video_file": None}
            },
            "video_file": {
                "description": "Analyze video file",
                "config": {"video_file": "/path/to/video.mp4"}
            }
        },
        "features": {
            "enable_rtsp": "Enable RTSP streaming output",
            "enable_websocket": "Enable WebSocket data streaming",
            "enable_mqtt": "Enable MQTT message publishing",
            "enable_recording": "Record processed video output",
            "enable_debug": "Enable debug logging"
        }
    }

@app.get("/api/current/ai")
async def get_current_state():
    """Get current real-time state from running pipeline"""
    if pipeline is None:
        raise HTTPException( status_code=503, detail="Pipeline not running" )
    try:
        if hasattr(pipeline, 'detection_data'):
            pipeline_detections = pipeline.detection_data.get_dict()
        return pipeline_detections if pipeline_detections else []
    except Exception as e:
        raise HTTPException( status_code=500, detail=f"Error getting current state: {str(e)}")

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
    """API root with endpoint documentation"""
    return {
        "message": "Raspberry Pi Control API",
        "version": "2.0.0",
        "endpoints": {
            "servo": {
                "move": "POST /servo/move"
            },
            "database": {
                "status": "GET /database/status",
                "start": "POST /database/start",
                "stop": "POST /database/stop",
                "restart": "POST /database/restart",
                "health": "GET /health/ai",
            },
            "pipeline": {
                "status": "GET /pipeline/status",
                "start": "POST /pipeline/start",
                "stop": "POST /pipeline/stop",
                "restart": "POST /pipeline/restart",
                "modes": "GET /pipeline/modes",
                "current_state": "GET /api/current/ai",
            },
            "ai_detection": {
                "detections": "GET /api/detections?minutes=5&limit=100",
                "statistics": "GET /api/statistics?hours=1",
                "bulk_insert": "POST /api/detections/bulk"
            },
            "camera": {
                "capture": "POST /camera/stream/capture",
                "stream": "GET /camera/stream/snapshots",
                "viewer": "GET /camera/stream/hls",
                "list_captures": "GET /captures",
                "delete_capture": "DELETE /captures/{filename}",
                "health": "GET /health/camera",
            },
        }
    }

# STREAMLIT DASHBOARD
# Streamlit imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import asyncio
def run_streamlit():
    """Streamlit dashboard - runs in separate process"""
    st.set_page_config(page_title="Detection Analytics Dashboard",page_icon="",layout="wide",initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        .main { padding: 0rem 1rem; }
        .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
        [data-testid="stMetricValue"] {
            font-size: 24px;  /* Change the value size */
        }
        [data-testid="stMetricLabel"] {
            font-size: 28px;  /* Change the label size */
        }
        [data-testid="stMetricDelta"] {
            font-size: 24px;  /* Change the delta size */
        }
        </style>
    """, unsafe_allow_html=True)
    API_BASE_URL = "http://localhost:8000"
    
    @st.cache_data(ttl=60)
    def fetch_detections(minutes=60, limit=10000000):
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/detections",
                params={"minutes": minutes, "limit": limit},
                timeout=300
            )
            if response.status_code == 200:
                data = response.json()
                #xprint(data[0:5])  # Print first 5 records for debugging
                #print(f"Fetched {len(data)} detections")
                if data:
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    # Optionally expand bbox into separate columns
                    df[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(df['bbox'].tolist(), index=df.index)
                    #print(df.head())
                    return df
                return pd.DataFrame()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return []
        except requests.Timeout:
            st.error(f"Request timed out fetching {minutes} minutes of data")
            return []
        except Exception as e:
            st.error(f"Error fetching detections: {str(e)}")
            return []
    
    @st.cache_data(ttl=60)
    def fetch_statistics(hours=24):
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/statistics",
                params={"hours": hours},
                timeout=300
            )
            data = response.json()
            if data:
             print(f"Fetched statistics: {data}")
             return data
            #print(f'{response.json()}')
        except:
            return {}
    
    @st.cache_data(ttl=5)
    def fetch_pipeline_status():
        try:
            response = requests.get(f"{API_BASE_URL}/pipeline/status", timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None 
    def start_pipeline(config):
        try:
            response = requests.post(f"{API_BASE_URL}/pipeline/start", json=config, timeout=10)
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    def stop_pipeline():
        try:
            response = requests.post(f"{API_BASE_URL}/pipeline/stop", timeout=10)
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    def fetch_db_status():
        try:
            response = requests.get(f"{API_BASE_URL}/database/status", timeout=5)
            #print(f'DB Status Response: {response.status_code} - {response.text}')
            return response.json() if response.status_code == 200 else None
        except:
            return None     
    def start_database(mqtt_connect):
        try:
            print(f"Starting database with MQTT: {mqtt_connect}")
            response = requests.post(f"{API_BASE_URL}/database/start", json={"mqtt_sending": mqtt_connect}, timeout=10)
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    def stop_database():
        try:
            response = requests.post(f"{API_BASE_URL}/database/stop", timeout=10)
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, {"error": str(e)}

    ## SIDEBAR
    with st.sidebar:
        st.header("System Control")
        # Database Status & Control
        st.subheader("Database")
        db_status = fetch_db_status()
        print(f"DB Status: {db_status}")
        if db_status and db_status.get('status') == "healthy":
            st.success(f"CONNECTED")
            if db_status.get('total_detections') is not None:
                st.metric("Total Detections", f"{db_status['total_detections']:,}")
            if db_status.get('database_size') is not None:
                st.metric("Database Size", db_status['database_size'])
            #if db_status.get('mqtt_sending') is True:
            #    st.metric("MQTT sending", "Enabled")
            #if st.button("Stop Database", use_container_width=True):
            #    success, result = stop_database()
            #    if success:
            #        st.success("Database stopped")
            #        st.cache_data.clear()
            #        time.sleep(1)
            #        st.rerun()
            #    else:
            #        st.error(f"Failed: {result}")
        else:
            st.error(f"Not Connected")
            #col1 = st.columns(1)[0]
            #with col1:
            #    mqtt = st.checkbox("MQTT listening to db", value=True)
            if st.button("Start Database", use_container_width=True):
                with st.spinner("Starting database..."):
                    success, result = start_database(mqtt_connect=False)
                    if success:
                        st.success("Database started!")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Failed: {result}")
        
        time_format_func = lambda x: f"Last {x} minutes" if x < 60 else (f"Last {x//1440} days" if x > 1440 else f"Last {x//60} hours")
        st.subheader("Statistics table interval")
        stats_hours = st.selectbox(
            "Statistics win",
            options=[1, 6, 12, 24, 48, 168],
            format_func=lambda x: f"Last {x} hours" if x < 24 else f"Last {x//24} days",
            index=3
        ) 
        st.subheader("Time Range")
        time_range = st.selectbox(
            "Select time range",
            options=[60, 180, 360, 720, 1440, 2880, 5760, 10080, 20160, 40320],
            format_func=time_format_func,
            index=1
        )
        INTERVAL_MAP = {"1 minute": "1T", "10 minutes": "10T", "30 minutes": "30T", "1 hour": "1H", "1 day": "1D", "1 month": "M1" }
        DTICK_MAP = {"1T": 1 * 60 * 1000,"10T": 10 * 60 * 1000,"30T": 30 * 60 * 1000,"1H": 60 * 60 * 1000,"6H": 6 * 60 * 60 * 1000,"1D": 24 * 60 * 60 * 1000,"1W": 7 * 24 * 60 * 60 * 1000,"1M": "M1","3M": "M3","1Y": "M12"}
        FORMAT_MAP = {"1T": "%H:%M", "10T": "%H:%M", "30T": "%H:%M", "1H": "%H:%M", "6H": "%H:%M", "1D": "%d.%m", "1W": "%d.%m", "M1": "%b %Y", "M3": "%b %Y", "M12": "%Y"}
        st.subheader("Time counts visualisation format")
        per = st.selectbox("Time aggregation", options=["1 minute", "10 minutes", "30 minutes", "1 hour", "1 day", "1 month"], index=1)
        
        st.markdown("---")
        auto_refresh = st.checkbox("Enable auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 60)
        if st.button("Refresh Now"):
            st.cache_data.clear()
            st.rerun()

    ## DASHBOARD
    if db_status and db_status.get('status') == "healthy": 
        df_detections = fetch_detections(minutes=time_range, limit=10000000)
        stats = fetch_statistics(hours=stats_hours)
        # Statistics summary table
        if stats:
            st.subheader(f"Statistics Summary last {stats_hours} hours")    
            col1 = st.columns(1)[0]
            with col1:
                if stats:
                    stats_data = []
                    for class_name, data in stats.items():
                        stats_data.append({
                            'Class': data['class_name'],
                            'Total Detections': data['total_detections'],
                            'Avg Confidence': f"{data['avg_confidence']:.2%}",
                            'Min Confidence': f"{data['min_confidence']:.2%}",
                            'Max Confidence': f"{data['max_confidence']:.2%}"
                        })
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, width='stretch', hide_index=True)
        else:
            st.info("No statistics available for the selected time range.")

        if df_detections.empty:
            st.warning("No detection data available for the selected time range.")
        else:
            st.subheader(f"Detection Analytics Dashboard {time_format_func(time_range).lower()}.")
            # Key metrics0
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Detections", f"{len(df_detections):,}")
            with col2:
                st.metric("Avg Confidence", f"{df_detections['confidence'].mean():.2%}")
            
            # histogram
            col1 = st.columns(1)[0]
            with col1:
                df_timeline = df_detections.copy()
                selected = INTERVAL_MAP[per]
                df_timeline = df_timeline.set_index("timestamp")
                # Group by both time and class
                df_class_timeline = (
                    df_timeline.groupby('class_name')
                    .resample(selected)
                    .size()
                    .reset_index(name="count")
                )
                # Remove incomplete periods at start and end
                if len(df_class_timeline) > 0:
                    # Get second and second-to-last timestamps
                    unique_times = sorted(df_class_timeline['timestamp'].unique())
                    if len(unique_times) > 2:
                        df_class_timeline = df_class_timeline[(df_class_timeline['timestamp'] > unique_times[0]) & (df_class_timeline['timestamp'] < unique_times[-1])]
                # Sort classes by total count (descending)
                class_order = (
                    df_class_timeline.groupby('class_name')['count']
                    .sum()
                    .sort_values(ascending=True)  # Největší první
                    .index.tolist()
                )
                # Convert class_name to categorical with specified order
                df_class_timeline['class_name'] = pd.Categorical(
                    df_class_timeline['class_name'],
                    categories=class_order,
                    ordered=True
                )
                # Create stacked bar chart
                fig_timeline = px.bar(
                    df_class_timeline,
                    x="timestamp",
                    y="count",
                    color="class_name",  # Different color per class
                    title=f"Diagram by detections every {per}",
                    labels={'count': 'Number of Objects', 'timestamp': 'Time', 'class_name': 'Class'},
                    barmode='stack',  # Options: 'stack', 'group', 'overlay'
                    category_orders={'class_name': class_order}
                )
                columns_count = len(sorted(df_class_timeline['timestamp'].unique()))
                fig_timeline.update_xaxes(
                    type="date",
                    title="Time",
                    nticks=DTICK_MAP[selected] if columns_count < 60 else 15,
                    tickformat=FORMAT_MAP[selected], 
                    ticklabelmode="period",
                    ticks="outside",
                    tickangle= -45 if columns_count > 20 else 0
                )
                fig_timeline.update_layout(
                    height=800,
                    hovermode="x unified", 
                    yaxis_title="Number of Objects"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

            # Detection timeline line chart
            col1 = st.columns(1)[0]
            with col1:
                df_timeline = df_detections.copy()
                selected = INTERVAL_MAP[per]
                df_timeline = df_timeline.set_index("timestamp")
                df_hourly = (
                    df_timeline.resample(selected)          # 1-hour buckets
                    .size()                  # count detections
                    .reset_index(name="count")
                )
                df_timeline = df_hourly
                # Remove incomplete periods at start and end
                if len(df_timeline) > 2:
                    # Simply remove first and last time buckets
                    df_timeline = df_timeline.iloc[1:-1]
                #print(df_timeline.head())
                # Create line chart with Plotly
                fig_timeline = px.line(
                    df_timeline,
                    x=df_timeline.columns[0],
                    y="count",
                    title=f"Detection timeline every {per}",
                    markers=True
                )
                fig_timeline.update_xaxes(
                    type="date",
                    title="Time",
                    nticks=10,
                    tickformat=FORMAT_MAP[selected], 
                    ticklabelmode="period",
                    ticks="outside"
                )
                fig_timeline.update_layout(
                    height=400,
                    hovermode="x unified", 
                    yaxis_title="Number of Objects"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

            # Class distribution pie chart
            col1 = st.columns(1)[0]
            with col1:
                class_counts = df_detections['class_name'].value_counts()
                fig_pie = px.pie(
                    values=class_counts.values, 
                    names=class_counts.index, 
                    hole=0.4,     
                    title=f"Class distribution pie",
                )
                fig_pie.update_layout(
                    height=500
                )
                st.plotly_chart(fig_pie, width='stretch')

            # Confidence distribution
            col1 = st.columns(1)[0]
            with col1:
                fig_conf = px.box(
                    df_detections,
                    x='class_name',
                    y='confidence',
                    color='class_name',
                    title='Confidence by Class',
                    labels={'class_name': 'Class', 'confidence': 'Confidence'}
                )
                fig_conf.update_layout(
                    height=800, 
                    showlegend=False
                )
                st.plotly_chart(fig_conf, width='stretch')
            st.markdown("---")

            col1 = st.columns(1)[0]
            with col1:
                # Calculate bbox centers
                df_bbox = df_detections.copy()
                df_bbox['center_x'] = (df_bbox['bbox'].apply(lambda x: x[0]) + df_bbox['bbox'].apply(lambda x: x[2])) / 2
                df_bbox['center_y'] = (df_bbox['bbox'].apply(lambda x: x[1]) + df_bbox['bbox'].apply(lambda x: x[3])) / 2
                fig_scatter = px.scatter(
                    df_bbox,
                    x='center_x',
                    y='center_y',
                    color='class_name',
                    size='confidence',
                    title='Detection Positions (Bbox Centers)',
                    labels={'center_x': 'X Position', 'center_y': 'Y Position', 'class_name': 'Class'},
                    opacity=0.6
                )
                fig_scatter.update_yaxes(autorange="reversed")
                fig_scatter.update_layout(title='Spatial analysis map',height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Bounding box confidence per class visualization
            col1 = st.columns(1)[0]
            with col1:
                df_bbox = df_detections.copy()
                df_bbox['center_x'] = (df_bbox['bbox'].apply(lambda x: x[0]) + df_bbox['bbox'].apply(lambda x: x[2])) / 2
                df_bbox['center_y'] = (df_bbox['bbox'].apply(lambda x: x[1]) + df_bbox['bbox'].apply(lambda x: x[3])) / 2
                # Grid resolution
                bins_x = 40
                bins_y = 40
                # Create histogram
                heatmap, xedges, yedges = np.histogram2d(
                    df_bbox['center_x'], 
                    df_bbox['center_y'], 
                    bins=[bins_x, bins_y]
                )
                # Create heatmap figure
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=heatmap.T,  # Transpose for correct orientation
                    x=xedges[:-1],
                    y=yedges[:-1],
                    colorscale='Hot',  # or 'Viridis', 'Jet', 'Hot', 'Portland'
                    colorbar=dict(title="Density<br>(0-1)")
                ))
                fig_heatmap.update_layout(
                    title='Detection Density Heatmap',
                    xaxis_title='X Position',
                    yaxis_title='Y Position',
                    height=600,
                    yaxis=dict(autorange="reversed")  # Reverse Y-axis to match image coordinates
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

            # Class distributions columns grapf
            col1 = st.columns(1)[0]
            with col1:
                class_dist = df_detections['class_name'].value_counts().reset_index()
                class_dist.columns = ['Class', 'Count']
                fig_bar = px.bar(
                    class_dist,
                    x='Class',
                    y='Count',
                    color='Class',
                    title='Detections distribution by Class',
                    labels={'Count': 'Number of Detections', 'Class': 'Object Class'},
                    text='Count'
                )
                fig_bar.update_traces(
                   texttemplate='%{text}',  
                   textposition='outside'  
                )       
                fig_bar.update_layout(
                    height=600, 
                    showlegend=False
                )
                st.plotly_chart(fig_bar, width='stretch')
    else:
        st.info("No database connected.")
    if auto_refresh:
        time.sleep(refresh_interval)
        st.cache_data.clear()
        st.rerun()

# MAIN LAUNCHER
def run_fastapi():
    """Run FastAPI with uvicorn"""
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "API_with_streamlit:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--lifespan", "on",
        "--log-level", "debug",
    ])

    
def run_streamlit_process():
    """Run Streamlit in subprocess"""
    # Create temporary Streamlit script
    streamlit_script = Path("_temp_streamlit_app.py")
    # Write the run_streamlit function to a file
    with open(streamlit_script, 'w') as f:
        f.write("""
import sys
sys.path.insert(0, '.')
from API_with_streamlit import run_streamlit
if __name__ == '__main__':
    run_streamlit()
""")
    # Run Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(streamlit_script),
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])

def main():
    """Main launcher - starts both FastAPI and Streamlit"""
    print("Services:")
    print("  - FastAPI:   http://localhost:8000")
    print("  - API Docs:  http://localhost:8000/docs")
    print("  - Streamlit: http://localhost:8501")
    print()
    # Start FastAPI in separate process
    multiprocessing.set_start_method("spawn", force=True)
    fastapi_process = multiprocessing.Process(target=run_fastapi, daemon=False)
    fastapi_process.start()
    # Give FastAPI time to start
    time.sleep(5)
    # Start Streamlit in main process (so we can see output)
    try:
        run_streamlit_process()
    except KeyboardInterrupt:
        print("\nShutting down...")
        fastapi_process.terminate()
        fastapi_process.join()

if __name__ == "__main__": main()