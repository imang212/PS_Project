from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import asyncio
import time
import subprocess
import io
import os

app = FastAPI()

## CONFIGURATION
CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(exist_ok=True)

## MODELS
# Servo control request
class ServoRequest(BaseModel):
    angle: float = Field(..., ge=-1.0, le=1.0, description="Servo position (-1.0 to 1.0)")

# Camera settings for capture
class CameraSettings(BaseModel):
    width: int = Field(1280, ge=320, le=1920)
    height: int = Field(720, ge=240, le=1080)
    framerate: int = Field(30, ge=1, le=60)

# Stream settings    
class StreamSettings(BaseModel):
    width: int = Field(640, ge=320, le=1920)
    height: int = Field(480, ge=240, le=1080)
    framerate: int = Field(15, ge=1, le=30)
    stream_port: int = Field(8554, ge=1024, le=65535)

# AI Analysis Result Models
class DetectionData(BaseModel):
    label: str
    confidence: float
    bbox: List[int]  # [x, y, width, height]

class AIAnalysisResult(BaseModel):
    timestamp: str
    device_id: str
    analysis_type: str  # "object_detection", "face_detection", "motion_detection"
    detections: List[DetectionData]
    frame_number: Optional[int] = None
    metadata: Optional[dict] = None

## In-memory storage for analysis results (for demo purposes)
analysis_history = []

## SERVO CONTROL 
# from gpiozero import Servo
# from time import sleep
# servo = Servo(17)

# @app.post("/servo/move")
# async def move_servo(data: ServoRequest):
#     """Move servo to normalized position (-1.0 to +1.0)"""
#     try:
#         servo.value = data.angle
#         await asyncio.sleep(0.4)
#         return {"status": "ok", "angle_set": data.angle}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Servo error: {str(e)}")

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

# MJPEG STREAM
@app.get("/camera/stream/mjpeg")
def stream_mjpeg(width: int = 1280, height: int = 720, quality: int = 10):
    """
    MJPEG stream - funguje přímo v prohlížeči v <img> tagu
    Příklad: <img src="http://raspi:8000/camera/stream/mjpeg">
    """
    
    def generate_frames():
        # Spustíme rpicam-vid s MJPEG výstupem
        cmd = [
            "rpicam-vid",
            "-t", "0",              # Nekonečný stream
            "-n",                   # Bez preview
            "--width", str(width),
            "--height", str(height),
            "--codec", "mjpeg",     # MJPEG codec!
            "-o", "-"               # Stdout
        ]
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        try:
            while True:
                # Čteme data po velkých kusech
                chunk = proc.stdout.read(4096)
                
                if not chunk:
                    break
                
                # Multipart MIME boundary pro MJPEG
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n')
        
        finally:
            proc.kill()
            proc.wait()
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# CAMERA STREAM (SNAPSHOT STREAM)
@app.get("/camera/stream/snapshots")
def stream_snapshots(width: int = 1280, height: int = 720, fps: int = 10):
    """
    Stream jednotlivých JPEG snímků
    Spolehlivější než video stream
    """
    def generate_snapshots():
        frame_delay = 1.0 / fps
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
                    time.sleep(frame_delay)
                    continue
                # Načteme obrázek
                with open(temp_img, 'rb') as f:
                    frame_bytes = f.read()
                # Pošleme jako MJPEG frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       frame_bytes + b'\r\n')
                time.sleep(frame_delay)
        except Exception as e:
            print(f"Stream error: {e}")
    return StreamingResponse(
        generate_snapshots(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

#CAMERA VIEWER
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
        <div class="info">
            <h2>MJPEG Stream (doporučeno pro prohlížeč)</h2>
            <button onclick="switchStream('mjpeg')">Zapnout MJPEG</button>
            <button onclick="switchStream('snapshots')">Zapnout Snapshots</button>
        </div> 
        <div class="stream-container">
            <img id="stream" src="/camera/stream/mjpeg" alt="Camera Stream">
        </div> 
        <div class="info">
            <h3>Aktuální stream:</h3>
            <p id="current-stream">/camera/stream/mjpeg</p>
            <p><strong>Rozlišení:</strong> <span id="resolution">640x480</span></p>
            <p><strong>Status:</strong> <span id="status">Running</span></p>
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
            streamImg.onload = () => {
                status.textContent = 'Running';
                status.style.color = '#4CAF50';
            };
            streamImg.onerror = () => {
                status.textContent = 'Error';
                status.style.color = '#f44336';
            };
        </script>
    </body>
    </html>
    """    
    return Response(content=html_content, media_type="text/html")

## ENDPOINT TO RECEIVE AI ANALYSIS RESULTS
@app.post("/api/analysis/objects")
async def receive_object_detection(data: AIAnalysisResult):
    """Příjem výsledků z object detection"""
    try:
        #analysis_history.append(data.dict())
        
        # Zde můžeš přidat logiku pro:
        # - uložení do databáze
        # - triggery (např. alarm při detekci osoby)
        # - notifikace
        # - logging
        print(f"[{data.timestamp}] Obdržena analýza od {data.device_id}: {len(data.detections)} objektů")
        return {
            "status": "success",
            "message": "Analysis received",
            "detections_count": len(data.detections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

## ENDPOINT TO RECEIVE MOTION DETECTION RESULTS
@app.get("/api/analysis/stats")
async def get_statistics():
    """Statistiky analýz"""
    return {"total_analyses": len(analysis_history)}
        
# UTILITY ENDPOINTS
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

@app.get("/health")
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
                "health": "GET /health"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
