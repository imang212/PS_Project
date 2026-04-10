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

class VideoSource(ABC):
    def __init__(self, location, rotation=0, flip_h=False, flip_v=False):
        self.location = location
        self.rotation = rotation
        self.flip_h = flip_h
        self.flip_v = flip_v

    def get_transform_str(self):
        """Generates the GStreamer videoflip string based on config."""
        transforms = []
        rot = int(self.rotation)
        
        # Using strict integer enums forces GStreamer to override the camera's 
        # orientation tags without triggering string parsing crashes.
        # 1=90r, 2=180, 3=90l
        if rot == 90:
            transforms.append("videoflip video-direction=1")
        elif rot == 180:
            transforms.append("videoflip video-direction=2")
        elif rot == 270:
            transforms.append("videoflip video-direction=3")
        
        # 4=horiz, 5=vert
        if self.flip_h:
            transforms.append("videoflip video-direction=4")
        if self.flip_v:
            transforms.append("videoflip video-direction=5")
            
        if transforms:
            # We keep the RGBA conversion to prevent the hardware memory lock!
            return " ! videoconvert ! video/x-raw, format=RGBA ! " + " ! ".join(transforms)
            
        return ""

    @abstractmethod
    def get_source_str(self): pass


class FileSource(VideoSource):
    def __init__(self, location, fps=30, rotation=0, flip_h=False, flip_v=False):
        super().__init__(location, rotation, flip_h, flip_v)
        self.fps = fps

    def get_source_str(self):
        return (
            f"filesrc location={self.location} ! decodebin"
            f"{self.get_transform_str()} ! "
            f"videoconvert ! videorate ! "
            f"video/x-raw, framerate={self.fps}/1 ! queue"
        )

class CameraSource(VideoSource):
    """
    Source for hardware cameras using libcamerasrc.
    """
    def __init__(self, device_index=0, width=640, height=480, fps=30, rotation=0, flip_h=False, flip_v=False):
        super().__init__(str(device_index), rotation, flip_h, flip_v)
        self.width = width
        self.height = height
        self.fps = fps

    def get_source_str(self):
        # Omitting the name for index 0 allows auto-binding to the primary camera
        cam_prop = "" if self.location == "0" else f"camera-name={self.location}"
        print(self.location)
        
        return (
            f"libcamerasrc {cam_prop} ! "
            # FIX: We explicitly demand format=NV12 here so the Pi hardware ISP kicks in!
            f"video/x-raw, format=NV12, width={self.width}, height={self.height}, framerate={self.fps}/1"
            f"{self.get_transform_str()} ! "
            f"videoconvert ! video/x-raw, format=NV12 ! queue"
        )

class RTSPSource(VideoSource):
    def __init__(self, url, latency=200, rotation=0, flip_h=False, flip_v=False):
        super().__init__(url, rotation, flip_h, flip_v)
        self.latency = latency

    def get_source_str(self):
        return (
            f"rtspsrc location={self.location} latency={self.latency} protocols=4 ! "
            f"application/x-rtp, media=video ! " 
            f"rtph264depay ! h264parse ! decodebin"
            f"{self.get_transform_str()} ! queue max-size-buffers=5 ! "
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
            f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} vdevice-group-id=1 ! "
            f"hailofilter so-path={self.post_so_path} function-name=filter ! queue"
        )

class HailoTracker:
    def __init__(self, keep_tracked_frames=30, keep_lost_frames=10, class_id=-1):
        self.keep_tracked_frames = keep_tracked_frames
        self.keep_lost_frames = keep_lost_frames
        self.class_id = class_id

    def get_tracker_str(self):
        # Hardware-accelerated tracking element with custom permanence settings
        return (
            f"hailotracker class-id={self.class_id} "
            f"keep-tracked-frames={self.keep_tracked_frames} "
            f"keep-lost-frames={self.keep_lost_frames} ! "
        )

class Sink(ABC):
    @abstractmethod
    def get_sink_str(self): pass

class BoxSink(Sink):
    def __init__(self, output_path):
        self.output_path = output_path

    def get_sink_str(self):
        return (
            f"videoconvert ! "
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

class AppSink(Sink):
    """
    Modular sink that extracts metadata (JSON) and optional frames.
    Ensures a consistent structure with or without tracking.
    """
    def __init__(self, on_data_cb=None, include_frame=True):
        self.on_data_cb = on_data_cb # Callback to pipeline.notify_listeners
        self.include_frame = include_frame
        self.pipeline = None

    def get_sink_str(self):
        return "videoconvert ! video/x-raw, format=BGR ! fakesink name=app_sink signal-handoffs=True sync=false"
    
    def on_new_buffer(self, sink, buffer, pad):
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        results = []
        for det in detections:
            bbox = det.get_bbox()
            track_id = -1
            tracking_info = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if tracking_info: track_id = tracking_info[0].get_id()

            results.append({
                "label": det.get_label(),
                "confidence": round(det.get_confidence(), 2),
                "bbox": [bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()],
                "track_id": track_id
            })
            
        frame = None
        if self.include_frame:
            caps = pad.get_current_caps()
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                height = caps.get_structure(0).get_value("height")
                width = caps.get_structure(0).get_value("width")
                frame = np.ndarray((height, width, 3), buffer=map_info.data, dtype=np.uint8).copy()
                buffer.unmap(map_info)
        
        # Notify the pipeline's list of listeners
        if self.on_data_cb:
            self.on_data_cb(frame, results)

class HailoPipeline:
    def __init__(self, source, adapter, inference, sink, tracker=None):
        self.source = source
        self.adapter = adapter
        self.inference = inference
        self.sink = sink
        self.tracker = tracker 
        self.listeners = []  
        self.raw_frame = None 
        
        # 1. Get the segment strings
        inf_str = self.inference.get_inference_str()
        # Ensure tracker_str does NOT end with a '!' here to avoid doubles
        tracker_str = self.tracker.get_tracker_str().strip().rstrip('!') if self.tracker else ""
        
        # 2. Build the pipeline string carefully
        # Note: We removed the hardcoded '!' after tracker_str and replaced it with logic
        elements = [
            f"{self.source.get_source_str()} ! tee name=t",
            "t. ! queue ! videoconvert ! video/x-raw, format=BGR ! fakesink name=raw_sink signal-handoffs=True sync=false",
            f"t. ! queue ! {self.adapter.get_adapter_str()}",
            f"{inf_str}"
        ]

        if tracker_str:
            elements.append(f"{tracker_str}")

        elements.append(f"{self.sink.get_sink_str()}")

        # Join with ' ! ' only where appropriate
        # We must be careful because 't.' branches don't always follow a linear '!'
        self.pipeline_str = (
            f"{self.source.get_source_str()} ! tee name=t "
            f"t. ! queue leaky=2 max-size-buffers=5 ! videoconvert ! video/x-raw, format=BGR ! fakesink name=raw_sink signal-handoffs=True sync=false "
            f"t. ! queue leaky=2 max-size-buffers=5 ! {self.adapter.get_adapter_str()} ! "
            f"{inf_str} ! "
            f"{tracker_str + ' ! ' if tracker_str else ''}"
            f"{self.sink.get_sink_str()}"
        )
        
        self.pipeline = None
        self.loop = GLib.MainLoop()

    def add_listener(self, listener):
        self.listeners.append(listener)

    def get_raw_frame(self):
        """Returns the latest raw frame from the camera."""
        return self.raw_frame

    def notify_listeners(self, frame, detections):
        for listener in self.listeners:
            listener.on_data_received(frame, detections)

    def _on_raw_buffer(self, sink, buffer, pad):
        """Callback for the raw_sink tap."""
        caps = pad.get_current_caps()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            height = caps.get_structure(0).get_value("height")
            width = caps.get_structure(0).get_value("width")
            self.raw_frame = np.ndarray((height, width, 3), buffer=map_info.data, dtype=np.uint8).copy()
            buffer.unmap(map_info)
    
    def _on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("End-Of-Stream reached.")
            self.pipeline.set_state(Gst.State.NULL)
            self.loop.quit()
            for listener in self.listeners:
                listener.on_stop()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Error: {err}, Debug info: {debug}")
            self.pipeline.set_state(Gst.State.NULL)
            self.loop.quit()
            for listener in self.listeners:
                listener.on_stop()
        print(f"Message: {t}")

    def stop(self):
        """Stops the GStreamer pipeline and releases hardware resources."""
        if self.pipeline:
            print("Stopping Hailo Pipeline and releasing resources...")
            # This is the crucial step that frees the Hailo device
            self.pipeline.set_state(Gst.State.NULL)
            
        if self.loop and self.loop.is_running():
            self.loop.quit()
            
        print("Pipeline stopped.")

    def run(self):
        print("Starting Hailo Pipeline with the following configuration:")
        print(f"{self.pipeline_str}")
        self.sink.on_data_cb = self.notify_listeners
        self.pipeline = Gst.parse_launch(self.pipeline_str)
        
        # Connect the primary detection sink
        gst_sink = self.pipeline.get_by_name("app_sink")
        if gst_sink:
            gst_sink.connect("handoff", self.sink.on_new_buffer)

        # Connect the raw frame tap sink
        gst_raw_sink = self.pipeline.get_by_name("raw_sink")
        if gst_raw_sink:
            gst_raw_sink.connect("handoff", self._on_raw_buffer)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)
        self.pipeline.set_state(Gst.State.PLAYING)
        self.loop.run()
    
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

class LogListener(HailoListener):
    """Logs detection results to the console."""
    def on_data_received(self, frame, detections):
        # 'detections' is now a list of dicts from AppSink
        logger.info(f"Received {len(detections)} detections:")
        for det in detections:
            # Access keys directly from the dictionary
            label = det["label"]
            conf = det["confidence"]
            bbox = det["bbox"]
            track_id = det.get("track_id", -1) # Handles object permanence
            
            logger.info(f" - {label} (ID: {track_id}, Conf: {conf}) at {bbox}")


if __name__ == "__main__":
    # Testing the pipeline with a file source, letterbox adapter, Hailo inference, and MQTT sink
    source = RTSPSource("rtsp://192.168.73.17:8080/h264_pcm.sdp")
    adapter = LetterboxAdapter()
    inference = HailoInference("/home/nightshadearia/DetectionUI/yolov8s_h8l.hef", "/home/nightshadearia/DetectionUI/libyolo_hailortpp_post.so")
    tracker = HailoTracker(keep_tracked_frames=30, keep_lost_frames=10)
    sink = AppSink()
    
    pipeline = HailoPipeline(source, adapter, inference, sink, tracker)
    log = LogListener()
    pipeline.add_listener(log)
    pipeline.run()