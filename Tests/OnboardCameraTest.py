# Include:
# - RaspberryPi/Input.py

def install_library_linux(name):
    import sys
    import subprocess
    if not sys.platform.startswith("linux"):
        return
    # prefer the apt package name for known pip->apt differences
    apt_name = "python3-opencv" if name == "opencv-python" else name
    try:
        subprocess.check_call(["sudo", "apt-get", "update"])
        subprocess.check_call(["sudo", "apt-get", "install", "-y", apt_name])
    except subprocess.CalledProcessError:
        print(f"Failed to install {apt_name} via apt-get.")

install_library_linux("opencv-python")
import cv2 as cv
from Input import OnboardCameraVideoProvider, VideoStream, VideoStreamFormatterStrategy, VideoStreamListener, FrameBuffer


def test_onboard_camera():
    video_provider = OnboardCameraVideoProvider()
    video_stream = VideoStream(video_provider)

    class PrintFrameSizeListener(VideoStreamListener):
        def on_new_frame(self, frame_buffer: FrameBuffer):
            print(f"Received frame of size: {frame_buffer.width}x{frame_buffer.height}")

    listener = PrintFrameSizeListener()
    video_stream.add_listener(listener)

    video_stream.start_streaming()

    try:
        # Stream for 10 seconds
        import time
        import sys, shutil, subprocess
        time.sleep(10)
    finally:
        video_stream.stop_streaming()