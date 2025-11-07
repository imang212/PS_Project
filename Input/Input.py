from __future__ import annotations
import numpy as np
import cv2
from typing import Iterator, Optional, List, Tuple
import threading
import time
from abc import ABC, abstractmethod

class FrameBuffer:
    def __init__(self, capacity: int, frame_shape: tuple[int, ...], frame_dtype: type = np.uint8) -> None:
        """
        Initializes the buffer.
        
        :param capacity: Number of frames to keep
        :param frame_shape: Shape of each frame (height, width, channels)
        :param frame_dtype: Data type of the frames (default uint8 for CV2)
        """
        self.capacity: int = capacity
        self.frame_shape: tuple[int, ...] = frame_shape
        self.frame_dtype: type = frame_dtype
        self.buffer: np.ndarray = np.zeros((capacity, *frame_shape), dtype=frame_dtype)
        self.index: int = 0
        self.full: bool = False
        self._lock = threading.Lock()

    def add_frame(self, frame: np.ndarray) -> None:
        """
        Adds a new frame to the buffer, overwriting the oldest if full.
        """
        with self._lock:
            if frame.shape != self.frame_shape:
                raise ValueError(f"Frame shape {frame.shape} does not match buffer shape {self.frame_shape}")
            
            self.buffer[self.index] = frame
            self.index = (self.index + 1) % self.capacity
            if self.index == 0:
                self.full = True

    def get(self, i: int) -> np.ndarray:
        """
        Returns the i-th oldest frame (0 is oldest).
        """
        with self._lock:
            if self.full:
                if i < 0 or i >= self.capacity:
                    raise IndexError("Index out of range")
                real_index: int = (self.index + i) % self.capacity
                return self.buffer[real_index]
            else:
                if i < 0 or i >= self.index:
                    raise IndexError("Index out of range")
                return self.buffer[i]

    def __len__(self) -> int:
        """
        Returns the number of frames currently in the buffer.
        """
        return self.capacity if self.full else self.index

    def __getitem__(self, i: int) -> np.ndarray:
        """
        Allows indexing like buffer[i] to get the i-th oldest frame.
        """
        return self.get(i)

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterates over frames from oldest to newest.
        """
        for i in range(len(self)):
            yield self.get(i)

class VideoStreamListener(ABC):
    """Listener interface triggered when a new frame is available."""
    @abstractmethod
    def on_frame(self, frame: np.ndarray) -> None:
        pass

class VideoStreamThread(threading.Thread):
    """Thread that continuously updates a VideoStream."""
    def __init__(self, stream: 'VideoStream', interval: float = 0.0) -> None:
        """
        :param stream: The VideoStream to update
        :param interval: Sleep interval between updates (0 = no delay)
        """
        super().__init__(daemon=True)
        self.stream = stream
        self.interval = interval
        self._running = True

    def run(self) -> None:
        while self._running:
            self.stream.update()
            if self.interval > 0:
                time.sleep(self.interval)

    def stop(self) -> None:
        self._running = False


class VideoStream(ABC):
    """Abstract base class for video streams (capture or file)."""
    
    def __init__(self, frame_shape: Tuple[int, ...], scaled_shape: Tuple[int, ...], name: str = None, full_buffer_size: int = 10, scaled_buffer_size: int = 10) -> None:
        """
        :param name: Name of the stream
        :param full_buffer_size: Number of frames in full resolution buffer
        :param scaled_buffer_size: Number of frames in scaled buffer
        :param scaled_shape: Optional shape to scale frames for scaled_buffer (width, height)
        """
        self.name: str = name
        self.listeners: List[VideoStreamListener] = []
        
        self.full_buffer = FrameBuffer(full_buffer_size, frame_shape)
        self.scaled_buffer = FrameBuffer(scaled_buffer_size, scaled_shape)
        self.frame_shape = frame_shape
        self.scaled_shape = scaled_shape

        # Start update thread
        self._thread = VideoStreamThread(self)

    @abstractmethod
    def read(self):
        pass

    def update(self) -> None:
        """Update buffers with a new frame and notify listeners."""
        frame = self.read()
        if frame is None:
            return  # No new frame

        # Add to buffers
        self.full_buffer.add_frame(frame)
        scaled_frame = self.resize(frame, self.scaled_shape)
        self.scaled_buffer.add_frame(scaled_frame)

        # Notify listeners
        for listener in self.listeners:
            listener.on_frame(frame)
    
    def resize(self, frame: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Crop and rescale a frame to the target_shape, keeping the center.

        :param frame: Input frame (H, W, C)
        :param target_shape: Desired output shape (width, height)
        :return: Cropped and resized frame
        """
        target_w, target_h = target_shape
        h, w = frame.shape[:2]

        # Compute aspect ratios
        frame_ratio = w / h
        target_ratio = target_w / target_h

        # Determine cropping dimensions
        if frame_ratio > target_ratio:
            # Frame is wider -> crop width
            new_w = int(h * target_ratio)
            new_h = h
            x1 = (w - new_w) // 2
            y1 = 0
        else:
            # Frame is taller -> crop height
            new_w = w
            new_h = int(w / target_ratio)
            x1 = 0
            y1 = (h - new_h) // 2

        # Crop the frame
        cropped = frame[y1:y1+new_h, x1:x1+new_w]

        # Resize to target shape
        resized = cv2.resize(cropped, (target_w, target_h))
        return resized

    @property
    def last_frame(self) -> np.ndarray:
        """Return the newest frame from the full buffer."""
        if len(self.full_buffer) == 0:
            raise RuntimeError("No frames in buffer")
        return self.full_buffer[len(self.full_buffer) - 1]

    def add_listener(self, listener: VideoStreamListener) -> None:
        """Add a listener to be notified of new frames."""
        self.listeners.append(listener)

    def remove_listener(self, listener: VideoStreamListener) -> None:
        """Remove a listener."""
        self.listeners.remove(listener)
    
    def start(self):
        """Start the update thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the update thread."""
        self._thread.stop()
        self._thread.join()
        self.on_stop()

    @abstractmethod
    def on_stop(self):
        """Hook called when the stream is stopped."""
        pass

class VideoPlayer(VideoStream):
    """Video stream from a video file."""
    
    def __init__(self, filename: str, scaled_shape: Tuple[int,int], name: str = None, full_buffer_size: int = 10, scaled_buffer_size: int = 10):
        self.filename = filename
        self.cap = cv2.VideoCapture(filename)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {filename}")
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from video file")

        if scaled_shape is None:
            scaled_shape = (frame.shape[1] // 2, frame.shape[0] // 2)

        super().__init__(
            name=name or f"VideoPlayer:{filename}",
            frame_shape=frame.shape,
            scaled_shape=scaled_shape,
            full_buffer_size=full_buffer_size,
            scaled_buffer_size=scaled_buffer_size
        )
        # add first frame manually
        self.full_buffer.add_frame(frame)
        self.scaled_buffer.add_frame(self.resize(frame, self.scaled_buffer.frame_shape[:2][::-1]))

        self.start()

    def read(self) -> Optional[np.ndarray]:
        """Read the next frame from the video."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def on_stop(self):
        super().stop()
        self.cap.release()

class VideoCapture(VideoStream):
    """Video stream from a camera device."""
    
    def __init__(self, device_index: int = 0, name: str = None, full_buffer_size: int = 10, scaled_buffer_size: int = 10, scaled_shape: Optional[Tuple[int,int]] = None):
        self.device_index = device_index
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {device_index}")
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from camera")

        if scaled_shape is None:
            scaled_shape = (frame.shape[1] // 2, frame.shape[0] // 2)

        super().__init__(
            name=name or f"VideoCapture:{device_index}",
            frame_shape=frame.shape,
            scaled_shape=scaled_shape,
            full_buffer_size=full_buffer_size,
            scaled_buffer_size=scaled_buffer_size
        )
        # add first frame manually
        self.full_buffer.add_frame(frame)
        self.scaled_buffer.add_frame(self.resize(frame, self.scaled_buffer.frame_shape[:2][::-1]))

        self.start()

    def read(self) -> Optional[np.ndarray]:
        """Try reading a frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def on_stop(self):
        super().stop()
        self.cap.release()

player = VideoPlayer("sample_video.mp4")
