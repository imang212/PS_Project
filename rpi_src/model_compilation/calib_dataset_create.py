import cv2
import os
import time
import numpy as np
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
# --- CONFIGURATION ---
RTSP_URL = "rtsp://admin:Dcuk.123456@192.168.37.99/Stream"
OUTPUT_DIR = "calib_dataset"
TARGET_COUNT = 1050  # Number of photos to capture (100-300 is sufficient for Hailo)
SAVE_EVERY_N_FRAME = 35 
# Crop settings (same as your pipeline)
TOP, LEFT = 80, 200
RIGHT, BOTTOM = 440, 0

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize camera (try index 0 or 1 depending on connection)
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Error: Cannot connect to RTSP stream. Check IP and password.")
    exit()

print(f"Starting data collection. Target: {TARGET_COUNT} frames.")
saved_count = 0
frame_count = 0
try:
    while saved_count < TARGET_COUNT:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Lost connection to stream...")
            break
        if frame_count % SAVE_EVERY_N_FRAME == 0:
            # Calculate dimensions (in case camera sends different resolution)
            h, w, _ = frame.shape
            y_end = h - BOTTOM
            x_end = w - RIGHT
            # Crop
            cropped = frame[TOP:y_end, LEFT:x_end]
            # Check resolution (should result in 640x640)
            final_img = cv2.resize(cropped, (640, 640))
            # Save
            filename = f"{OUTPUT_DIR}/img_{saved_count:03d}.jpg"
            cv2.imwrite(filename, final_img)
            saved_count += 1
            print(f"Saved: {filename} ({saved_count}/{TARGET_COUNT})")
        frame_count += 1
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"Done! Photos can be found in folder: {OUTPUT_DIR}")
