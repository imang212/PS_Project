from __future__ import annotations
from typing import List, Optional, Iterator
from copy import deepcopy
import cv2
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
import threading
import time

class FrameBuffer:
    """
    FrameBuffer — A fixed-length, circular buffer for storing image frames.
    
    Always maintains the same length (capacity), even if not yet full.
    Returns frames in logical order (oldest → newest).
    Overwrites oldest frames when new ones are added beyond capacity.
    Designed for real-time video streams and AI frame parsing.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.frames: List[Optional[object]] = [None] * capacity  # Underlying storage
        self.head = 0   # Index to write next frame
        self.count = 0  # Number of valid frames

    # -----------------------------------------------------------------------
    # ADDING FRAMES
    # -----------------------------------------------------------------------
    def add(self, frame: Optional[object]):
        """Add a new frame to the buffer, copying it for safety."""
        self.frames[self.head] = None if frame is None else deepcopy(frame)
        self.head = (self.head + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    # -----------------------------------------------------------------------
    # ACCESSING FRAMES
    # -----------------------------------------------------------------------
    def get(self, index: int) -> Optional[object]:
        """Return the frame at a logical index (0 = oldest, capacity-1 = newest)."""
        if index < 0 or index >= self.capacity:
            raise IndexError(f"{index} out of bounds for capacity {self.capacity}")
        if self.count < self.capacity and index >= self.count:
            return None
        logical_base = (self.head - self.count + self.capacity) % self.capacity
        real_index = (logical_base + index) % self.capacity
        return self.frames[real_index]

    def set(self, index: int, frame: Optional[object]):
        """Set or replace a frame at a specific logical position."""
        if index < 0 or index >= self.capacity:
            raise IndexError(f"{index} out of bounds for capacity {self.capacity}")
        logical_base = (self.head - self.count + self.capacity) % self.capacity
        real_index = (logical_base + index) % self.capacity
        self.frames[real_index] = None if frame is None else deepcopy(frame)

    def get_newest(self) -> Optional[object]:
        return None if self.count == 0 else self.get(self.count - 1)

    def get_oldest(self) -> Optional[object]:
        return None if self.count == 0 else self.get(0)

    def get_relative(self, offset: int) -> Optional[object]:
        """Offset 0 = newest, 1 = one before newest, etc."""
        if offset < 0 or offset >= self.capacity:
            return None
        idx = (self.head - 1 - offset + self.capacity) % self.capacity
        return self.frames[idx]

    # -----------------------------------------------------------------------
    # INFORMATION
    # -----------------------------------------------------------------------
    def size(self) -> int:
        return self.capacity

    def count_frames(self) -> int:
        return self.count

    def is_full(self) -> bool:
        return self.count == self.capacity

    def is_empty(self) -> bool:
        return self.count == 0

    def clear(self):
        self.frames = [None] * self.capacity
        self.head = 0
        self.count = 0

    # -----------------------------------------------------------------------
    # ITERATION SUPPORT
    # -----------------------------------------------------------------------
    def __iter__(self) -> Iterator[Optional[object]]:
        for i in range(self.capacity):
            yield self.get(i)

    # -----------------------------------------------------------------------
    # ARRAY CONVERSION
    # -----------------------------------------------------------------------
    def to_list(self) -> List[Optional[object]]:
        """Returns frames in logical order (oldest → newest)."""
        return [self.get(i) for i in range(self.capacity)]

class VideoStreamThread:
    """
    Continuously updates a VideoStream in a background thread.
    """

    def __init__(self, stream: VideoStream, update_interval: float = 0.01):
        """
        :param stream: VideoStream instance to update
        :param update_interval: seconds between updates (default 0.01 = 100 FPS)
        """
        self.stream = stream
        self.update_interval = update_interval
        self._running = False
        self._thread: threading.Thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if not self._running:
            self._running = True
            self._thread.start()

    def _run(self):
        while self._running and self.stream.is_running():
            self.stream.update()
            time.sleep(self.update_interval)  # control update frequency

    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join()
        self.stream.stop()

class VideoStreamListener(ABC):
    """
    Listener class for VideoStream.
    """

    def __init__(self):
        pass
    
    @abstractmethod
    def onFrame(last_scaled: object, full_buffer: FrameBuffer, scaled_buffer: FrameBuffer):
        """
        A trigger method for when a new frame is captured.
        """
        pass

class VideoStream:
    """
    Represents a single video input source using OpenCV.
    Maintains two frame buffers:
      - full_buffer: full-resolution frames
      - scaled_buffer: 320x180 scaled frames (center-cropped)
    """

    def __init__(self, name: str, buffer_size: int, capture_index: int = 0, width: int = 640, height: int = 480, update_interval: float = 0.01):
        self.name = name
        self.capture_index = capture_index
        self.capture = cv2.VideoCapture(capture_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.full_buffer = FrameBuffer(buffer_size)
        self.scaled_buffer = FrameBuffer(buffer_size)

        self.scaled_width = 320
        self.scaled_height = 180

        self.listeners : List[VideoStreamListener] = []
        self.thread : VideoStreamThread = VideoStreamThread(self, update_interval)

        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open camera {capture_index}")

    def update(self):
        """
        Grab a new frame from the camera and update buffers.
        Should be called continuously in a loop.
        """
        ret, frame = self.capture.read()
        if ret:
            # OpenCV frames are NumPy arrays (BGR)
            self.full_buffer.add(frame)

            scaled = self._crop_and_scale(frame, self.scaled_width, self.scaled_height)
            self.scaled_buffer.add(scaled)
            for listener in self.listeners:
                listener.on_frame(self.get_latest_scaled(), self.full_buffer, self.scaled_buffer)

    def _crop_and_scale(self, frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """
        Center-crop the frame to maintain the target aspect ratio, then resize.
        """
        src_h, src_w = frame.shape[:2]
        src_aspect = src_w / src_h
        target_aspect = target_w / target_h

        crop_w, crop_h = src_w, src_h

        if src_aspect > target_aspect:
            # Too wide: crop width
            crop_w = int(src_h * target_aspect)
        else:
            # Too tall: crop height
            crop_h = int(src_w / target_aspect)

        x = (src_w - crop_w) // 2
        y = (src_h - crop_h) // 2

        cropped = frame[y:y + crop_h, x:x + crop_w]
        resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized

    # === Accessors ===
    def get_name(self) -> str:
        return self.name

    def get_capture(self) -> cv2.VideoCapture:
        return self.capture

    def get_full_buffer(self) -> FrameBuffer:
        return self.full_buffer

    def get_scaled_buffer(self) -> FrameBuffer:
        return self.scaled_buffer

    def get_latest_full(self) -> Optional[np.ndarray]:
        return self.full_buffer.get_newest()

    def get_latest_scaled(self) -> Optional[np.ndarray]:
        return self.scaled_buffer.get_newest()

    def is_ready(self) -> bool:
        return self.full_buffer.count_frames() > 0

    def is_running(self) -> bool:
        return self.capture.isOpened()

    def stop(self):
        if self.capture.isOpened():
            self.capture.release()
            self.thread.stop()

    def __str__(self):
        return f"VideoStream[{self.name}, {int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}, full={self.full_buffer.count_frames()}, scaled={self.scaled_buffer.count_frames()}]"

