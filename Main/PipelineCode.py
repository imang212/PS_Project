"""
PipelineCode.py
Header:
    High-level, extensible pipeline framework for camera input and inference on
    Hailo accelerators. Provides a dual-branch GStreamer input node (zero-copy
    hardware path + CPU copy path for visualization), an example inference node
    (HailoYoloNode) and a simple Pipeline controller to chain nodes together.
Overview:
    This module defines a small data-flow pipeline pattern where a central
    mutable object (PipelineData) flows through a chain of Node instances. Each
    Node updates PipelineData in-place and then forwards it to the next node.
    The design focuses on avoiding unnecessary CPU copies by maintaining
    hardware-backed buffer references while still providing an optional
    CPU-visible frame for visualization or other CPU-only processing.
Primary classes and responsibilities:
    - PipelineData
        Central data carrier for a single pipeline iteration.
        Fields include:
          * request_frame_copy: bool - indicates caller requests a CPU copy this iteration
          * latest_buffer: optional hardware-backed buffer (e.g., GstBuffer)
          * history: list of recent hardware-backed buffer references
          * frame_copy: optional numpy array (CPU copy of frame)
          * timestamp: float - time of node completion
          * detections: optional list of detection dicts produced by compute nodes
          * meta: dict - place for diagnostics and per-node information
    - Node (abstract)
        Base class for pipeline nodes. Subclasses implement _process(pipeline_data)
        and use the public process(pipeline_data) to ensure the node's work runs
        and that the pipeline chaining contract (always pass to next node) is honored.
        Attributes:
          * next_node: optional Node - next node in the chain
    - GStreamerInputNode (Node)
        Builds and runs a GStreamer pipeline that tees the camera stream onto two
        branches:
          1) hailo_sink: intended to provide zero-copy, hardware-backed buffers
             (kept in a small in-process history for downstream inference).
          2) python_sink: CPU branch where frames are copied to user-space and can
             be converted to numpy arrays for visualization/CPU processing.
        Key behavior:
          * build_element(): constructs and starts the Gst pipeline; configures appsinks
          * _process(pipeline_data): pulls from both appsinks (non-blocking with a timeout),
            updates pipeline_data.latest_buffer, pipeline_data.history, optionally
            fills pipeline_data.frame_copy (if request_frame_copy is True), and
            records diagnostics in pipeline_data.meta.
        Notes:
          * _gstbuffer_to_numpy() contains a simple conversion assuming raw RGB or GRAY8,
            real deployments may need format-specific conversion (NV12/YUV etc.).
          * The class attempts to probe supported camera modes (via picamera2) and falls
            back to safe defaults when introspection is not available.
    - HailoYoloNode (Node)
        Illustrative computation node that demonstrates where Hailo SDK integration
        would occur. Intended to consume a hardware-backed buffer reference and run
        inference without copying to CPU when possible.
        Key hooks:
          * _load_model(model_path): placeholder to create device/context and load HEF
          * _prepare_input_from_gstbuffer(gst_buffer): placeholder to wrap or map the
            GstBuffer into a form the Hailo runtime accepts
          * _run_inference_on_hailo(prepared_input): placeholder for runtime invocation
        Expected output:
          * pipeline_data.detections: list of detection dicts with keys 'bbox', 'class', 'conf'
        Notes:
          * The current implementation returns empty detections and must be replaced
            with concrete Hailo SDK calls and result parsing.
    - Pipeline
        Controller for connecting nodes and executing the chain.
        Key methods:
          * add_node(node): append a node to the end of the chain (returns self)
          * tick(request_frame_copy=True): run one pipeline iteration, returns PipelineData
          * run(callback=None, interval=0.0): convenience loop to continuously call tick and
            optionally pass each PipelineData to a callback (e.g., for rendering/logging)
Usage (conceptual):
    1) Instantiate Pipeline().
    2) Create and configure a GStreamerInputNode(width, height, ...) and call build_element().
    3) Add a HailoYoloNode(model_path, ...) (or other nodes) to the pipeline via add_node().
    4) Call pipeline.tick(request_frame_copy=True/False) to process a single frame or pipeline.run()
       to loop continuously. The returned PipelineData contains buffers, optional CPU frame,
       detections and metadata.
Best practices and notes:
    - Avoid requesting a CPU copy unless needed: set request_frame_copy=False for pure
      accelerator pipelines to minimize CPU overhead.
    - The history mechanism stores references to GstBuffer objects to allow inference to
      fall back to recent frames if no new hardware buffer is available on a tick.
    - The Gst pipeline string and memory hints (e.g., memory:NVMM) may need to be adjusted
      depending on the platform (Raspberry Pi, Jetson, etc.) and the actual camera source
      element (libcamerasrc vs v4l2src).
    - Replace placeholder Hailo methods with concrete SDK calls, taking care to map or
      share DMA buffers when possible to keep the zero-copy data path.
Requirements and environment:
    - Python bindings for GStreamer (PyGObject / gi.repository.Gst)
    - numpy
    - picamera2 (optional — used for probing camera modes; code falls back on defaults)
    - Hailo SDK (or other accelerator SDK) to implement the placeholders in HailoYoloNode
Extensibility:
    - Add additional Node subclasses to perform post-processing, drawing, logging,
      or to integrate other compute backends (TensorRT, OpenVINO, CPU).
    - Node._process should be kept side-effect-limited to modifying the supplied
      PipelineData and should not swallow exceptions silently; consider adding explicit
      error handling/logging per node.
Limitations:
    - The provided Hailo integration is a placeholder; real applications must implement
      device lifecycle, error handling, and efficient zero-copy buffer mapping specific
      to the target hardware/SDK.
    - Gst buffer to numpy conversion implemented here assumes planar contiguous layout
      appropriate for RGB/GRAY raw formats; conversion for other formats is non-trivial.
Example (pseudo):
    pipeline = Pipeline()
    input_node = GStreamerInputNode(width=640, height=640)
    input_node.build_element()
    pipeline.add_node(input_node)
    pipeline.add_node(HailoYoloNode("/path/to/model.hef"))
    pdata = pipeline.tick(request_frame_copy=True)
    # use pdata.frame_copy for visualization, pdata.detections for results
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional
import subprocess
import os
import sys
import time
from abc import ABC, abstractmethod
import logging
from collections import deque
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import logging
import picamera2

log = logging.getLogger(__name__)

Gst.init(None)

@dataclass
class PipelineData:
    """
    Central data object that flows through the pipeline each iteration.
    Nodes read and write fields they are responsible for.
    """
    # Request flags (set by main loop / caller to request optional behaviour)
    request_frame_copy: bool = False  # whether a CPU-copy (OpenCV) is required this iteration

    # Camera/hardware data
    latest_buffer: Optional[Any] = None   # hardware-backed buffer (GstBuffer or opaque handle)
    history: List[Any] = field(default_factory=list)  # list of recent hardware-backed buffers (references)

    # CPU-visible data
    frame_copy: Optional[Any] = None  # numpy / OpenCV frame (copied from python branch)
    timestamp: float = field(default_factory=time.time)

    # AI results
    detections: Optional[List[dict]] = None  # list of detection dicts: {'bbox':(...), 'class':id, 'conf':float}

    # Diagnostics / misc
    meta: dict = field(default_factory=dict)

    def __str__(self):
        return (f"PipelineData(timestamp={self.timestamp}, "
                f"latest_buffer={'set' if self.latest_buffer else 'None'}, "
                f"history_size={len(self.history)}, "
                f"frame_copy_shape={self.frame_copy.shape if self.frame_copy is not None else 'None'}, "
                f"detections={self.detections}, "
                f"meta_keys={list(self.meta.keys())})")

class Node(ABC):
    """
    Base class for pipeline nodes.

    Each node implements _process(pipeline_data) which performs the node-specific work.
    The public method process(pipeline_data) calls _process then forwards the data to next_node.
    """
    def __init__(self):
        self.next_node: Optional["Node"] = None

    @abstractmethod
    def _process(self, pipeline_data: PipelineData) -> None:
        """
        Implement node-specific behaviour here.
        Must update pipeline_data in-place.
        """
        raise NotImplementedError

    def process(self, pipeline_data: PipelineData) -> None:
        """
        Process the pipeline_data and then pass it to the next node (if any).
        This enforces the 'every node passes data to the next' contract.
        """
        # Node-specific work
        self._process(pipeline_data)

        # Pass on (chain)
        if self.next_node is not None:
            self.next_node.process(pipeline_data)

    def chain(self):
        """Return the next node in the chain (for traversal)."""
        return self.next_node

class GStreamerInputNode(Node):
    """
    Camera input node that:
      - builds a dual-branch GStreamer pipeline (hardware path for Hailo, python path for CPU copy)
      - maintains a small history of hardware-backed buffers (references)
      - optionally places a CPU-copy (OpenCV numpy array) into pipeline_data.frame_copy
    """

    def __init__(self, width=640, height=640, history_size=5, filter_type="RGB", timeout_ns=Gst.SECOND // 10):
        super().__init__()
        self.width = width
        self.height = height
        self.history_size = history_size
        self.filter_type = filter_type.upper()
        self.timeout_ns = timeout_ns

        # local ring/history of GstBuffer references (avoid copying)
        self._history = deque(maxlen=history_size)

        # appsinks will be set when build_element() is called
        self._hailo_sink = None    # hardware-backed appsink (zero-copy)
        self._python_sink = None   # CPU branch appsink (for copies)

        # Query supported modes and validate (placeholder)
        self.supported_modes = self._query_camera_modes()
        if not self._is_supported(self.width, self.height, self.filter_type):
            raise ValueError(f"Requested {self.width}x{self.height} / {self.filter_type} not supported. "
                             f"Supported: {self.supported_modes}")

        # Keep a reference to the launched Gst.Pipeline if needed
        self._pipeline = None

    def _query_camera_modes(self):
        """
        Query camera for supported modes using pycamera2 (best-effort).
        Returns a list of dicts: {'width': int, 'height': int, 'format': 'RGB'|'GRAY'}
        If introspection of the pycamera2 API fails, returns a sensible default set
        including the requested resolution so the rest of the pipeline can continue.
        """
        modes = []
        try:
            pc2 = picamera2.Picamera2()
            # Candidate Picamera2 pixel formats mapped to our simple filter types
            candidates = [("RGB", "RGB888"), ("GRAY", "GRAY8")]

            # Probe the requested resolution first
            for fmt_name, fmt_str in candidates:
                try:
                    pc2.create_preview_configuration(main={"size": (self.width, self.height), "format": fmt_str})
                    modes.append({"width": self.width, "height": self.height, "format": fmt_name})
                except Exception:
                    # Not supported for this format/size combination
                    continue

            # Probe a few common resolutions to expose what the camera can do
            common_res = [(640, 480), (1280, 720), (1920, 1080)]
            for w, h in common_res:
                for fmt_name, fmt_str in candidates:
                    try:
                        pc2.create_preview_configuration(main={"size": (w, h), "format": fmt_str})
                        entry = {"width": w, "height": h, "format": fmt_name}
                        if entry not in modes:
                            modes.append(entry)
                    except Exception:
                        continue
        except Exception:
            # Any introspection error -> fall through to fallback below
            pass

        # Ensure we always include at least the requested resolution (fallback)
        if not modes:
            modes = [
                {"width": self.width, "height": self.height, "format": self.filter_type},
                {"width": 640, "height": 480, "format": "RGB"},
            ]

        return modes
    
    def _is_supported(self, width, height, filter_type):
        for m in self.supported_modes:
            if m["width"] == width and m["height"] == height and m["format"] == filter_type:
                return True
        return False

    def build_element(self):
        """
        Construct and start the GStreamer pipeline.
        The pipeline uses tee to provide:
          - hailo_sink: zero-copy hardware-backed buffers
          - python_sink: CPU-copy branch for visualization (OpenCV)
        """
        # Build the pipeline string. Adjust elements if your Pi backend requires them (libcamera vs v4l2).
        # Ensure NVMM / memory hints are correct for the target hardware.
        pipeline_str = (
            f"libcamerasrc ! video/x-raw, width={self.width}, height={self.height} ! videoconvert "
            f"! tee name=t "
            f"t. ! queue ! video/x-raw(memory:NVMM) ! appsink name=hailo_sink "
            f"t. ! queue ! video/x-raw ! appsink name=python_sink"
        )

        self._pipeline = Gst.parse_launch(pipeline_str)

        # Configure hailo_sink
        self._hailo_sink = self._pipeline.get_by_name("hailo_sink")
        self._hailo_sink.set_property("emit-signals", True)
        self._hailo_sink.set_property("max-buffers", self.history_size)
        self._hailo_sink.set_property("drop", True)

        # Configure python_sink
        self._python_sink = self._pipeline.get_by_name("python_sink")
        self._python_sink.set_property("emit-signals", True)
        self._python_sink.set_property("max-buffers", 2)
        self._python_sink.set_property("drop", True)

        # Start the pipeline
        self._pipeline.set_state(Gst.State.PLAYING)
        log.info("GStreamer pipeline started")

        return self._pipeline

    def _pull_hailo_sample(self):
        if self._hailo_sink is None:
            return None
        return self._hailo_sink.emit("try-pull-sample", self.timeout_ns)

    def _pull_python_sample(self):
        if self._python_sink is None:
            return None
        return self._python_sink.emit("try-pull-sample", self.timeout_ns)

    def _gstbuffer_to_numpy(self, gst_buffer):
        """
        Convert a GstBuffer (CPU branch) into a numpy array (OpenCV format).
        This forces a copy (extract_dup). Only used for visualization.
        """
        # Note: for complex formats (NV12, YUV) you must do proper conversion.
        # This implementation assumes the python branch provides RGB or GRAY8 raw frames.
        size = gst_buffer.get_size()
        if size == 0:
            return None
        raw = gst_buffer.extract_dup(0, size)
        # shape depends on filter_type
        channels = 1 if self.filter_type == "GRAY" else 3
        try:
            arr = np.frombuffer(raw, dtype=np.uint8)
            arr = arr.reshape((self.height, self.width, channels))
            return arr
        except Exception as e:
            log.exception("Failed to convert GstBuffer to numpy: %s", e)
            return None

    def _process(self, pipeline_data: PipelineData):
        """
        Pull from the hailo_sink (hardware) and python_sink (CPU copy) as requested.
        Update pipeline_data in-place:
          - pipeline_data.latest_buffer -> latest hailo-backed GstBuffer
          - pipeline_data.history -> list of recent buffer references
          - pipeline_data.frame_copy -> numpy array if request_frame_copy is True
          - pipeline_data.timestamp -> time.time() at pull
        """
        # Pull hardware sample (hailo path). Keep reference, do not copy
        hailo_sample = self._pull_hailo_sample()
        if hailo_sample:
            gst_buffer = hailo_sample.get_buffer()
            # append to local history (references)
            self._history.append(gst_buffer)
            pipeline_data.latest_buffer = gst_buffer
            # copy history references into pipeline_data.history (shallow references)
            pipeline_data.history = list(self._history)
        else:
            # No new hailo buffer this iteration
            pipeline_data.latest_buffer = None

        # If caller requested a python copy, pull CPU branch and convert to numpy
        if pipeline_data.request_frame_copy:
            python_sample = self._pull_python_sample()
            if python_sample:
                gst_buffer = python_sample.get_buffer()
                numpy_frame = self._gstbuffer_to_numpy(gst_buffer)
                pipeline_data.frame_copy = numpy_frame
            else:
                pipeline_data.frame_copy = None
        else:
            # Clear any stale frame_copy unless we want to keep previous
            pipeline_data.frame_copy = None

        # Timestamp when input node completed its work
        pipeline_data.timestamp = __import__("time").time()
        # Optionally add diagnostics
        pipeline_data.meta.setdefault("input_info", {}) 
        pipeline_data.meta["input_info"].update({
            "width": self.width,
            "height": self.height,
            "filter_type": self.filter_type,
            "history_size": len(self._history)
        })

class HailoYoloNode(Node):
    """
    Computation node that accepts hardware-backed frames (via pipeline_data.latest_buffer
    or pipeline_data.history) and runs YOLO inference on the Hailo accelerator.

    The Hailo SDK integration points are placeholders; replace them with your actual Hailo calls.
    """

    def __init__(self, model_path: str, input_shape=(640, 640, 3), conf_threshold=0.25):
        super().__init__()
        self.model_path = model_path
        self.input_shape = input_shape
        self.conf_threshold = conf_threshold
        # Placeholder for the actual Hailo runtime / context object
        self.hailo_runtime = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Load compiled HEF model and prepare runtime.
        Replace with actual Hailo SDK code:
          - create device/context
          - load HEF
          - allocate tensors/buffers
        """
        log.info("Loading Hailo model (placeholder): %s", model_path)
        return None  # replace with real runtime

    def _prepare_input_from_gstbuffer(self, gst_buffer):
        """
        Convert or wrap the GstBuffer for the Hailo SDK input.
        The goal: avoid copying where possible and pass a handle or DMA-backed region directly.
        Implementation will depend on the Hailo SDK API.
        """
        # Placeholder: return the buffer as-is (your SDK may require mapping to a special structure)
        return gst_buffer

    def _run_inference_on_hailo(self, prepared_input):
        """
        Execute the model on the Hailo chip and return parsed results.
        Replace this with real runtime.run(...) calls and parsing.
        """
        # Placeholder returns empty detections:
        return []

    def _process(self, pipeline_data: PipelineData):
        """
        Use the most appropriate source:
          - Prefer the latest hardware buffer (pipeline_data.latest_buffer)
          - If None and history exists, optionally use history[-1]
        Populate pipeline_data.detections with a list of dicts.
        """
        gst_buffer = pipeline_data.latest_buffer
        if gst_buffer is None and pipeline_data.history:
            gst_buffer = pipeline_data.history[-1]

        if gst_buffer is None:
            pipeline_data.detections = []
            return

        # Wrap or convert buffer for Hailo input (avoid copies if possible)
        hailo_input = self._prepare_input_from_gstbuffer(gst_buffer)

        # Run inference (placeholder)
        detections = self._run_inference_on_hailo(hailo_input)

        # Post-process (filter by conf_threshold, convert coordinates if needed)
        # Expected format: [{'bbox': (x1,y1,x2,y2), 'class': id, 'conf': float}, ...]
        pipeline_data.detections = [d for d in detections if d.get("conf", 0) >= self.conf_threshold]

        # Optionally add diagnostics
        pipeline_data.meta.setdefault("computation_info", {})
        pipeline_data.meta["computation_info"].update({
            "model": self.model_path,
            "num_detections": len(pipeline_data.detections)
        })

class Pipeline:
    """
    High-level controller for a chain of processing nodes.
    This object is what your main code interacts with.
    """

    def __init__(self):
        # The first node in the chain
        self.head_node = None

        # The last node in the chain (for easy append)
        self.tail_node = None

    def __str__(self):
        nodes = []
        current = self.head_node
        while current is not None:
            nodes.append(current.__class__.__name__)
            current = current.chain()
        return " -> ".join(nodes) if nodes else "Empty Pipeline"

    # ------------------------------------------------------------
    # Chainable method: add a new node to the pipeline
    # ------------------------------------------------------------
    def add_node(self, node: Node):
        """
        Append a node to the internal chain.
        Returns self so additional nodes can be chained fluently.
        """
        if self.head_node is None:
            # First node in pipeline
            self.head_node = node
            self.tail_node = node
        else:
            # Link new node at the end
            self.tail_node.next_node = node
            self.tail_node = node
        
        return self  # for chaining

    # ------------------------------------------------------------
    # Execute one pipeline step
    # ------------------------------------------------------------
    def tick(self, request_frame_copy=True):
        """
        Executes one complete pipeline pass:
        1. Creates a fresh PipelineData object for this frame.
        2. Passes it through the chain of nodes.
        3. Returns the fully populated PipelineData.
        """
        pdata = PipelineData(request_frame_copy=request_frame_copy)

        if self.head_node is not None:
            self.head_node.process(pdata)

        return pdata

    # ------------------------------------------------------------
    # Run continuously (optional helper)
    # ------------------------------------------------------------
    def run(self, callback=None, interval=0.0):
        """
        Repeatedly executes the pipeline.
        If callback is provided, it receives each PipelineData object.

        callback signature:
            callback(pdata: PipelineData)
        """
        import time

        try:
            while True:
                pdata = self.tick()

                # User-specified callback for drawing, logging, etc.
                if callback is not None:
                    callback(pdata)

                if interval > 0:
                    time.sleep(interval)
        except KeyboardInterrupt:
            print("Pipeline stopped.")