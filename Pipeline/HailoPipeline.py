import cv2
import numpy as np
import paho.mqtt.client as mqtt
import gi
import logging
import json
import os
import hailo # Required for metadata extraction
from abc import ABC, abstractmethod

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class VideoSource:
    def __init__(self, location):
        self.location = location

    def get_source_str(self):
        return f"filesrc location={self.location} ! decodebin ! videoconvert ! video/x-raw, format=NV12"

class FileSource(VideoSource):
    def __init__(self, location, fps=30):
        super().__init__(location)
        self.fps = fps

    def get_source_str(self):
        # 'videorate' ensures the stream follows the specified framerate
        return (
            f"filesrc location={self.location} ! decodebin ! "
            f"videoconvert ! videorate ! "
            f"video/x-raw, framerate={self.fps}/1 ! queue"
        )

class RTSPSource(VideoSource):
    def __init__(self, url, latency=200):
        super().__init__(url)
        self.latency = latency

    def get_source_str(self):
        # Added a queue after depay to stabilize the 'segment' event
        return (
            f"rtspsrc location={self.location} latency={self.latency} use-pipeline-clock=true ! "
            f"rtph264depay ! h264parse ! decodebin ! queue max-size-buffers=10 ! "
            f"videoconvert ! videorate ! video/x-raw, format=NV12, framerate=30/1"
        )

class FrameAdapter(ABC):
    @abstractmethod
    def get_adapter_str(self): pass

class LetterboxAdapter(FrameAdapter):
    def __init__(self, width=640, height=640):
        self.width, self.height = width, height

    def get_adapter_str(self):
        return (
            f"videoconvert ! videoscale add-borders=true ! "
            f"video/x-raw, width={self.width}, height={self.height}, pixel-aspect-ratio=1/1 ! videoconvert"
        )

class HailoInference:
    def __init__(self, hef_path, post_so_path, batch_size=1):
        self.hef_path = hef_path
        self.post_so_path = post_so_path
        self.batch_size = batch_size

    def get_inference_str(self):
        # Mandatory function-name=filter for standard TAPPAS libraries
        return (
            f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} ! "
            f"hailofilter so-path={self.post_so_path} function-name=filter ! queue"
        )

class Sink(ABC):
    @abstractmethod
    def get_sink_str(self): pass

class BoxSink(Sink):
    def __init__(self, output_path):
        self.output_path = output_path

    def get_sink_str(self):
        return (
            f"hailooverlay ! videoconvert ! "
            f"x264enc tune=zerolatency bitrate=5000 speed-preset=ultrafast ! "
            f"h264parse ! matroskamux ! filesink location={self.output_path}"
        )

class JSONDataSink(Sink):
    """
    Extracts Hailo metadata and saves it to a JSON file.
    """
    def __init__(self, json_output_path):
        self.json_path = json_output_path
        self.data_log = []

    def get_sink_str(self):
        # fakesink with handoff signal enabled to catch buffers in Python
        return "fakesink name=json_sink signal-handoffs=True sync=false"

    def on_new_frame(self, sink, buffer, pad):
        # Get the Region of Interest (ROI) which holds the metadata
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        frame_results = []
        for det in detections:
            bbox = det.get_bbox()
            frame_results.append({
                "label": det.get_label(),
                "confidence": round(det.get_confidence(), 2),
                "bbox": [bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()]
            })
        
        if frame_results:
            self.data_log.append({"pts": buffer.pts, "detections": frame_results})

    def finalize(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.data_log, f, indent=4)
        logger.info(f"Metadata saved to {self.json_path}")

class HailoPipeline:
    def __init__(self, source, adapter, inference, listener):
        self.source = source
        self.adapter = adapter
        self.inference = inference
        self.listener = listener
        
        # We use a single branch for simplicity:
        # source -> inference -> overlay -> videoconvert -> fakesink
        self.pipeline_str = (
            f"{self.source.get_source_str()} ! "
            f"{self.adapter.get_adapter_str()} ! "
            f"{self.inference.get_inference_str()} ! "
            f"hailooverlay ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"fakesink name=unified_sink signal-handoffs=True sync=false"
        )
        
        self.pipeline = None
        self.loop = GLib.MainLoop()

    def _on_new_buffer(self, sink, buffer, pad):
        """Extracts pixels and metadata, then notifies the listener."""
        # 1. Extract Metadata
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        # 2. Extract Pixel Data
        caps = pad.get_current_caps()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            height = caps.get_structure(0).get_value("height")
            width = caps.get_structure(0).get_value("width")
            
            # Create a numpy array (BGR for OpenCV) from the buffer
            frame = np.ndarray((height, width, 3), buffer=map_info.data, dtype=np.uint8)
            
            # Send combined data to the listener
            if self.listener:
                self.listener.on_data_received(frame.copy(), detections)
            
            buffer.unmap(map_info)

    def run(self):
        logger.info("Launching Unified Frame/Data Pipeline...")
        self.pipeline = Gst.parse_launch(self.pipeline_str)
        
        # Connect the unified handoff signal
        sink = self.pipeline.get_by_name("unified_sink")
        sink.connect("handoff", self._on_new_buffer)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)
        self.pipeline.set_state(Gst.State.PLAYING)
        self.loop.run()

    def stop(self):
        self.pipeline.send_event(Gst.Event.new_eos())

    def _on_message(self, bus, message):
        if message.type == Gst.MessageType.EOS:
            self._finalize_shutdown()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"GStreamer Error: {err.message}")
            self._finalize_shutdown()

    def _finalize_shutdown(self):
        self.pipeline.set_state(Gst.State.NULL)
        if self.listener:
            self.listener.on_stop()
        self.loop.quit()

class HailoListener(ABC):
    @abstractmethod
    def on_data_received(self, frame, detections):
        """Called when a new frame and its associated metadata are ready."""
        pass

    def on_stop(self):
        """Called when the pipeline finishes."""
        pass

class SaveListener(HailoListener):
    def __init__(self, output_path, width=640, height=640, fps=30):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def on_data_received(self, frame, detections):
        # The frame already has boxes burned in by hailooverlay
        self.writer.write(frame)

    def on_stop(self):
        if self.writer:
            self.writer.release()
            logger.info(f"Saved video to: {self.output_path}")

class MQTTListener(HailoListener):
    """Sends inference results to an MQTT broker as JSON strings with optional authentication."""
    def __init__(self, broker="broker.hivemq.com", port=1883, topic="hailo/detections", username=None, password=None):
        self.topic = topic
        
        # Initialize MQTT client with compatibility for paho-mqtt 2.x
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            self.client = mqtt.Client() 
            
        # Set credentials if provided
        if username is not None:
            self.client.username_pw_set(username, password)
            logger.info(f"MQTT authentication enabled for user: {username}")
        
        if port == 8883:
            self.client.tls_set()

        try:
            self.client.connect(broker, port, 60)
            self.client.loop_start()
            logger.info(f"Connected to MQTT Broker: {broker} on port {port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")

    def on_data_received(self, frame, detections):
        payload = []
        for det in detections:
            bbox = det.get_bbox()
            payload.append({
                "label": det.get_label(),
                "confidence": round(det.get_confidence(), 2),
                "bbox": [bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()]
            })
        
        if payload:
            logger.info(f"Publishing {len(payload)} detections to {self.topic}")
            self.client.publish(self.topic, json.dumps(payload))

    def on_stop(self):
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("MQTT Listener disconnected.")

if __name__ == "__main__":
    # --- Configuration ---
    URL = "rtsp://192.168.0.105:8080/h264_ulaw.sdp"
    VIDEO_OUT = "/home/nightshadearia/output_boxed.mkv"
    JSON_OUT = "/home/nightshadearia/detections.json"
    HEF_PATH = "/home/nightshadearia/hailo-rpi5-examples/resources/yolov8n.hef"
    POST_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"

    src = RTSPSource(URL)
    inf = HailoInference(HEF_PATH, POST_SO)
    
    # Create the listener that handles the saving logic
    save_listener = MQTTListener()
    
    # Pass the listener directly to the pipeline
    pipeline = HailoPipeline(src, LetterboxAdapter(), inf, save_listener)
    
    # Run for 10 seconds
    GLib.timeout_add_seconds(10, lambda: (pipeline.stop(), False)[1])
    pipeline.run()



