from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
#from gpiozero import Servo
from time import sleep
import subprocess
import io
import os

app = FastAPI()

# -------------------------
# SERVO INITIALIZATION
# -------------------------
# Servo on GPIO17 (change if needed)
#servo = Servo(17)

#class ServoRequest(BaseModel):
#    angle: float   # -1.0 až 1.0 (nebo 0–180 po úpravě)

#@app.post("/servo/move")
#def move_servo(data: ServoRequest):
#    """
#    Move servo to normalized position (-1.0 to +1.0).
#    For 0–180° support, add mapping later.
#    """
#    angle = max(-1.0, min(1.0, data.angle))
#    servo.value = angle
#    sleep(0.4)
#    return {"status": "ok", "angle_set": angle}

# -------------------------
# CAMERA CAPTURE (PHOTO)
# -------------------------
@app.get("/camera/capture")
def camera_capture():
    output = "capture.jpg"
    # Use libcamera to take a picture
    subprocess.run([
        "rpicam-still",
        "-n",
        "--width", "1280",
        "--height", "720",
        "-o", output
    ])
    if not os.path.exists(output):
        return {"error": "Camera capture failed"}
    return FileResponse(output)

# -------------------------
# CAMERA STREAM (SNAPSHOT STREAM)
# -------------------------
@app.get("/camera/stream")
def camera_stream():
    """
    Returns continuous JPEG snapshots – works everywhere.
    """
    cmd = ["rpicam-jpeg", "-n", "--width", "640", "--height", "480", "-o", "-"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    return StreamingResponse(proc.stdout, media_type="image/jpeg")

# -------------------------
# ROOT ENDPOINT
# -------------------------
@app.get("/")
def root():
    return {
        "message": "Raspberry Pi Control API running",
        "capture": "/camera/capture",
        "stream": "/camera/stream"
    }
