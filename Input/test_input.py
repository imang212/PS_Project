from __future__ import annotations
import pygame
from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
import threading
import time
from Yui import Yui, YuiRoot, Graphics, Stack, Color
from Input import VideoStream, VideoStreamListener, CameraOpenError, VideoStreamFormatterStrategy, CameraVideoProvider, YouTubeVideoProvider, RemoteRaspberryPiStreamProvider

class VideoYui(Yui, VideoStreamListener):
    def __init__(self, parent: Yui, stream: VideoStream):
        super().__init__(parent)
        self.width = self.parent.width
        self.height = self.parent.height
        self.stream = stream
        self.image = None
        self.buffer = None

        print("Initialized")

    def on_frame(self, frame: np.ndarray, formatted_frame: np.ndarray, stream: VideoStream) -> None:
        image = cv.cvtColor(formatted_frame, cv.COLOR_BGR2RGB)

        # image is now RGB with shape (height, width, 3)
        h, w = image.shape[:2]
        try:
            # create a PyGame Surface directly from the buffer (fast)
            surf = pygame.image.frombuffer(image.tobytes(), (w, h), "RGB")
            # convert for faster blitting on the display surface
            self.image = surf.convert()
        except Exception:
            # fallback: use surfarray (requires numpy array shape (width, height, 3))
            self.image = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        
        self.buffer = stream.frame_buffer
    
    def on_draw(self, graphics: Graphics) -> None:
        self.stream.update()
        if self.image is None:
            graphics.text_align = (0.5, 0.5)
            graphics.fill_color = Color(255, 255, 255, 255)
            graphics.text_size = 24
            graphics.text("No signal.", self.width / 2, self.height / 2)
            return
        iw, ih = self.image.get_width(), self.image.get_height()
        if iw == 0 or ih == 0:
            graphics.text_align = (0.5, 0.5)
            graphics.fill_color = Color(255, 255, 255, 255)
            graphics.text_size = 24
            graphics.text("Invalid size.", self.width / 2, self.height / 2)
            return
        # scale to fit inside the available area while preserving aspect ratio
        scale = min(self.width / iw, self.height / ih)
        w, h = iw * scale, ih * scale
        x, y = (self.width - w) / 2, (self.height - h) / 2
        graphics.image_mode = "corner"
        graphics.image(self.image, x, y, w, h)

        # draw buffer info in the top-left corner
        graphics.text_align = (0, 0)
        graphics.fill_color = Color(255, 255, 255, 255)
        graphics.text_size = 24
        x, y = 5, 5

        if not self.buffer:
            graphics.text("buffer: empty", x, y)
        else:
            for i, elem in enumerate(self.buffer):
                if isinstance(elem, np.ndarray):
                    s = f"buffer[{i}]: shape={elem.shape}, dtype={elem.dtype}, bytes={elem.nbytes}"
                else:
                    try:
                        length = len(elem)
                    except Exception:
                        length = None
                    s = f"buffer[{i}]: type={type(elem).__name__}, len={length}"
                graphics.text(s, x, y)
                y += 16
        print(self.stream._frame_buffer)
    
def main():
    root = YuiRoot(name="Video Stream Test", width=800, height=600)

    strategy = VideoStreamFormatterStrategy.resize_strategy((160, 90), interpolation=cv.INTER_LINEAR)
    strategy.append_chain(VideoStreamFormatterStrategy.gray_scale_strategy())
    #video_provider = YouTubeVideoProvider("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    video_provider = RemoteRaspberryPiStreamProvider(raspberry_ip="raspberrypi.local", raspberry_user="imang", raspberry_password="imang", stream_port=8554, resolution=(640, 480))
    stream = VideoStream(video_provider, buffer_size=10, format_strategy=strategy)
    video_yui = VideoYui(root, stream)
    stream.add_listener(video_yui)
    root.auto_background = Color(0, 0, 0, 0)
    root.init()

if __name__ == "__main__":
    main()