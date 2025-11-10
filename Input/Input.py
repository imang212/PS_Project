from __future__ import annotations
import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod
import threading
import time
import yt_dlp

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

class CameraOpenError(Exception):
    """Raised when the requested camera/video source cannot be opened.

    Stores details about the attempt so callers can inspect/log them.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        src: int | str | None = None,
        api: int | None = None,
        last_error: Exception | str | None = None,
    ):
        # Allow legacy use: CameraOpenError("Error: cannot open camera")
        self.message = message or "Failed to open camera/video source"
        self.src = src
        self.api = api
        self.last_error = last_error
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        parts = [self.message]
        if self.src is not None:
            parts.append(f"src={self.src!r}")
        if self.api is not None:
            parts.append(f"api={self.api!r}")
        if self.last_error is not None:
            parts.append(f"last_error={self.last_error!r}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        """Return a serializable representation of the error details."""
        return {
            "message": self.message,
            "src": self.src,
            "api": self.api,
            "last_error": repr(self.last_error),
        }

    def __str__(self) -> str:
        return self._build_message()

class VideoStreamListener(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def on_frame(self, frame: np.ndarray, formatted_frame: np.ndarray, stream: VideoStream) -> None:
        """Called when a new frame is available."""
        pass

class VideoStreamFormatterStrategy(ABC):
    def __init__(self):
        self.next: VideoStreamFormatterStrategy | None = None

    def append_chain(self, strategy: VideoStreamFormatterStrategy) -> VideoStreamFormatterStrategy:
        """Appends a strategy to the end of the chain."""
        if self.next is None:
            self.next = strategy
        else:
            self.next.append(strategy)
        return self
    
    def remove_next(self) -> None:
        """Removes the next strategy in the chain."""
        if self.next is not None:
            self.next = self.next.next
    
    def insert_chain(self, strategy: VideoStreamFormatterStrategy) -> VideoStreamFormatterStrategy:
        """Inserts a strategy immediately after this one in the chain."""
        strategy.next = self.next
        self.next = strategy
        return strategy
    
    def apply(self, frame: np.ndarray, stream: VideoStream) -> np.ndarray:
        image = self.format(frame, stream)
        if self.next is not None:
            image = self.next.format(image, stream)
        return image

    @abstractmethod
    def format(self, frame: np.ndarray, stream: VideoStream) -> np.ndarray:
        pass

    @classmethod
    def resize_strategy(cls, size: int | tuple[int, int], interpolation: int = cv.INTER_AREA) -> '_ResizeStrategy':
        return _ResizeStrategy(size, interpolation)

    @classmethod
    def gray_scale_strategy(cls) -> '_GrayScaleStrategy':
        return _GrayScaleStrategy()

class _ResizeStrategy(VideoStreamFormatterStrategy):
    """
    Resize frames to a fixed (width, height) provided in the constructor,
    scaling to cover the target area and center-cropping the excess.

    :param size: (width, height) tuple or single int (square size)
    :param interpolation: OpenCV interpolation flag (default cv.INTER_AREA)
    """
    def __init__(self, size: int | tuple[int, int], interpolation: int = cv.INTER_AREA) -> None:
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
            if len(self.size) != 2:
                raise ValueError("size must be an int or a (width, height) tuple")
        # store as (width, height) to match cv.resize dsize
        self.interpolation = interpolation

    def format(self, frame: np.ndarray, stream: VideoStream) -> np.ndarray:
        # frame.shape -> (h, w) or (h, w, c)
        h, w = frame.shape[:2]
        target_w, target_h = self.size

        # scale to cover the target area (like CSS 'cover')
        scale_w = target_w / w
        scale_h = target_h / h
        scale = max(scale_w, scale_h)

        new_w = int(np.ceil(w * scale))
        new_h = int(np.ceil(h * scale))

        # resize first
        resized = cv.resize(frame, dsize=(new_w, new_h), interpolation=self.interpolation)

        # center-crop to the exact target size
        x0 = (new_w - target_w) // 2
        y0 = (new_h - target_h) // 2
        x1 = x0 + target_w
        y1 = y0 + target_h

        # handle possible 2D (grayscale) or 3D arrays
        if resized.ndim == 2:
            cropped = resized[y0:y1, x0:x1]
        else:
            cropped = resized[y0:y1, x0:x1, ...]

        return cropped

class _GrayScaleStrategy(VideoStreamFormatterStrategy):
    """
    Convert frames to grayscale.
    """
    def __init__(self) -> None:
        super().__init__()

    def format(self, frame: np.ndarray, stream: VideoStream) -> np.ndarray:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

class VideoStream:
    WINDOWS_CAMERA: int = cv.CAP_DSHOW
    def __init__(self, video_provider: VideoProvider, buffer_size: int = 10, threaded: bool = False, thread_frequency: float = 0.01, format_strategy: VideoStreamFormatterStrategy = None):
        self._video_provider: VideoProvider = video_provider
        self._frame_shape: np.ndarray = None
        self._frame_buffer: FrameBuffer = None
        self._formatted_buffer: FrameBuffer = None
        self._format_strategy: VideoStreamFormatterStrategy = format_strategy
        self._listeners: list[VideoStreamListener] = []
        self._buffer_size: int = buffer_size
        self._thread_frequency: float = thread_frequency
        self._thread: threading.Thread = None if not threaded else threading.Thread(target=self._threaded_update, daemon=True)
        if threaded:
            self._thread.start()
    
    @property
    def is_file(self):
        return isinstance(self.device_name, str) and self.device_name != ""
    
    def is_threaded(self):
        return self._thread is not None
    
    def thread_frequency(self):
        return self._thread_frequency

    @property
    def frame_shape(self):
        return self._frame_shape
    
    @property
    def has_source_ended(self):
        return not self._video_provider.is_active()
    
    @property
    def device_name(self):
        return self.cap.getBackendName()
    
    @property
    def frame_buffer(self):
        return self._frame_buffer
    
    @property
    def formatted_buffer(self):
        return self._formatted_buffer
    
    def _threaded_update(self):
        while True:
            self.update()
            time.sleep(self._thread_frequency)

    def update(self):
        frame = self._video_provider.read()

        if frame is None:
            return
        formatted_frame = self._format_strategy.apply(frame, self) if self._format_strategy else frame
        if self._formatted_buffer is None:
            self._formatted_buffer = FrameBuffer(capacity=self._buffer_size, frame_shape=formatted_frame.shape)
        if self._frame_shape is None:
            self._frame_shape = frame.shape
        if self._frame_buffer is None:
            self._frame_buffer = FrameBuffer(capacity=self._buffer_size, frame_shape=self.frame_shape)

        self._frame_buffer.add_frame(frame)
        self.formatted_buffer.add_frame(formatted_frame)
        for listener in self._listeners:
            if isinstance(listener, VideoStreamListener):
                listener.on_frame(frame, formatted_frame, self)
            else:
                self._listeners.remove(listener)
    
    def add_listener(self, listener: VideoStreamListener) -> None:
        self._listeners.append(listener)
    
    def remove_listener(self, listener: VideoStreamListener) -> None:
        self._listeners.remove(listener)

    def release(self):
        self.cap.release()
        cv.destroyAllWindows()

class VideoProvider(ABC):
    @abstractmethod
    def read(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def is_active(self) -> bool:
        pass

class CameraVideoProvider(VideoProvider):
    def __init__(self, src: int | str = 0, api: int = VideoStream.WINDOWS_CAMERA):
        self.cap = cv.VideoCapture(src, api)
        if not self.cap.isOpened():
            raise CameraOpenError(src=src, api=api)
    
    def get_name(self) -> str:
        return self.cap.getBackendName()
    
    def is_active(self) -> bool:
        return self.cap.isOpened()
    
    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

class FileVideoProvider(VideoProvider):
    def __init__(self, filepath: str):
        self.cap = cv.VideoCapture(filepath)
        if not self.cap.isOpened():
            raise CameraOpenError(src=filepath)
    
    def get_name(self) -> str:
        return self.cap.getBackendName()
    
    def is_active(self) -> bool:
        return self.cap.isOpened()
    
    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

class YouTubeVideoProvider(VideoProvider):
    def __init__(self, url: str):
        self.url = url
        self.video_url = self._get_stream_url(url)
        self.cap = cv.VideoCapture(self.video_url)
        if not self.cap.isOpened():
            raise CameraOpenError(src=url)
    
    def _get_stream_url(self, url: str) -> str:
        """Extract the direct video stream URL using yt_dlp."""
        ydl_opts = {'format': 'best', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    
    def get_name(self) -> str:
        return f"YouTube: {self.url}"
    
    def is_active(self) -> bool:
        return self.cap.isOpened()
    
    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame