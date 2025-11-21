# Include:
# - RaspberryPi/Input.py

import cv2 as cv
from Input import VideoStream, VideoStreamFormatterStrategy, OnboardRaspberryPiCameraProvider, VideoStreamListener

class DisplayListener(VideoStreamListener):
    def __init__(self, window_name: str = "Video"):
        super().__init__()
        self.window_name = window_name

    def on_frame(self, frame, formatted_frame, stream):
        # formatted_frame is BGR (OpenCV). Convert to RGB if you need RGB array for processing:
        rgb = cv.cvtColor(formatted_frame, cv.COLOR_BGR2RGB)
        # Display the formatted (BGR) frame so colors appear correct in OpenCV window:
        cv.imshow(self.window_name, formatted_frame)

if __name__ == "__main__":
    strategy = VideoStreamFormatterStrategy.resize_strategy((160, 90), interpolation=cv.INTER_LINEAR)
    strategy.append_chain(VideoStreamFormatterStrategy.gray_scale_strategy())
    #video_provider = YouTubeVideoProvider("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    video_provider = OnboardRaspberryPiCameraProvider(raspberry_ip="192.168.37.205", raspberry_user="imang", raspberry_password="imang", stream_port=8554, resolution=(640, 480))
    stream = VideoStream(video_provider, buffer_size=10, format_strategy=strategy)
    

    listener = DisplayListener("Formatted Stream")
    stream.add_listener(listener)

    try:
        while True:
            # Pull a frame from the provider and notify listeners
            stream.update()

            # Refresh GUI and handle quit key
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

            # Stop if source ended (file/stream finished)
            if stream.has_source_ended:
                break
    finally:
        # Attempt to release provider resources if available, then close windows
        try:
            if hasattr(video_provider, "release"):
                video_provider.release()
        except Exception:
            pass
        cv.destroyAllWindows()
