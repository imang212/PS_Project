"""
GStreamer Pipeline with Hailo Tracker for Raspberry Pi AI HAT+ (26 TOPS)
=========================================================================
This script demonstrates object detection and tracking using:
- IMX708 camera via rpicam-apps or IP camera RTSP stream
- YOLOv8m model for detection
- Hailo HW accelerator for inference
- hailotracker for object tracking
- RTSP streaming (legacy support)
- WebRTC streaming (browser-compatible)
- MQTT protocol for detection data sending
- WebSocket for real-time data sending
- Video recording mode for video analysis with statistics generation
"""
import os
os.environ["GST_PLUGIN_PATH"] = ""
os.environ["GST_PLUGIN_SYSTEM_PATH"] = "/usr/lib/aarch64-linux-gnu/gstreamer-1.0"
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer
import json
import threading
import sys
from datetime import datetime
from collections import defaultdict
from MQTTClient import MQTTPublisher
from WebSocket import WebSocketServerWithDetectionData
import numpy as np
try:
    from hailo import get_roi_from_buffer, HAILO_DETECTION, HAILO_UNIQUE_ID
except ImportError as e:
    print(f"Error message: {e}") # The error message tells you exactly which .so version it's looking for

# Initialize GStreamer
Gst.init(None)
Gst.version()

class VideoStatistics:
    """
    Tracks and calculates statistics for video analysis
    """
    def __init__(self):
        self.detections_per_class = defaultdict(int)
        self.confidence_sum_per_class = defaultdict(float)
        self.total_frames = 0
        self.frames_with_detections = 0
        self.object_counts = {"person": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}
        self.unique_track_ids = set()
        self.start_time = None
        self.end_time = None
        
    def add_detection(self, class_name, confidence, track_id=None):
        """Add a detection to statistics"""
        self.detections_per_class[class_name] += 1
        self.confidence_sum_per_class[class_name] += confidence
        if (track_id is not None and track_id not in self.unique_track_ids):
            self.object_counts[class_name] += 1
            self.unique_track_ids.add(track_id)
    
    def add_frame(self, has_detections=False):
        """Record a processed frame"""
        self.total_frames += 1
        if has_detections:
            self.frames_with_detections += 1
    
    def get_summary(self):
        """Generate statistics summary"""
        summary = {
            'total_frames': self.total_frames,
            'frames_with_detections': self.frames_with_detections,
            'unique_objects_tracked': len(self.unique_track_ids),
            'processing_time': None,
            'fps': "n/a",
            'detections_by_class': {}
        }        
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            summary['processing_time'] = f"{duration:.2f} seconds"
            summary['fps'] = f"{self.total_frames / duration:.2f}" if duration > 0 else "N/A"
        for class_name in self.detections_per_class:
            count = self.detections_per_class[class_name]
            avg_confidence = self.confidence_sum_per_class[class_name] / count if count > 0 else 0
            summary['detections_by_class'][class_name] = {
                'count': count,
                'real_object_count': self.object_counts[class_name],
                'average_confidence': f"{avg_confidence:.4f}"
            }
        return summary
    
    def print_summary(self):
        """Print formatted statistics summary"""
        summary = self.get_summary()
        print("\n" + "="*70)
        print("VIDEO ANALYSIS STATISTICS")
        print("="*70)
        print(f"Total Frames Processed: {summary['total_frames']}")
        print(f"Frames with Detections: {summary['frames_with_detections']}")
        print(f"Unique Objects Tracked: {summary['unique_objects_tracked']}")
        if summary['processing_time']:
            print(f"Processing Time: {summary['processing_time']}")
            print(f"Average FPS: {summary['fps']}")
        
        print("\nDetections by Class:")
        print("-" * 70)
        for class_name, stats in summary['detections_by_class'].items():
            print(f"  {class_name:20s}: {stats['count']:6d} detections, "
                  f"avg confidence: {stats['average_confidence']}")
        print("="*70 + "\n")
        
        return summary

class DetectionData:
    """
    Stores actual detection and tracking data
    """
    def __init__(self, debug=True):
        self.detections = []
        self.frame_count = 0
        self.timestamp = None
        self.lock = threading.Lock()
        self.has_new_data = False  # Flag for new data
        self.debug = debug
        # Track active objects by class
        self.object_counts = {"person": 0, "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}
        self.counted_track_ids = set()
        self.total_count = 0
    
    def update(self, detections, frame_count):
        """
        Updates detection data in a thread-safe manner        
        Args:
            detections: List of detection objects
            frame_count: Current frame number
        """
        with self.lock:
            self.detections = detections
            self.frame_count = frame_count
            self.timestamp = datetime.now().isoformat()
            self.has_new_data = True
            # Update active object counts
            # Count unique tracked objects
            for det in detections:
                track_id = det.get('track_id')
                if track_id is not None and track_id not in self.counted_track_ids:
                    self.object_counts[det['class_name']] += 1
                    self.counted_track_ids.add(track_id)
                    self.total_count += 1
            if self.debug == True and self.frame_count % 60 == 0:
                self._print_summary()
    
    def _print_summary(self):
        """
        Prints detection summary (called from within locked context)
        """
        print(f"\n[DETECTION SUMMARY Frame {self.frame_count}]")
        print(f"  Total active objects: {self.total_count}")
        print(f"  Current frame detections: {len(self.detections)}")
        for idx, det in enumerate(self.detections):
            bbox_tuple = det['bbox']
            tracking_info = f"[ID:{det['track_id']}]" if det['track_id'] is not None else "[No ID]"
            print(f"  [{idx}] {det['class_name']}: {det['confidence']:.2f} "f"@ bbox{bbox_tuple} {tracking_info}, time: {det['timestamp']}")
    
    def get_count_overlay_text(self):
        """Returns formatted text for overlay display"""
        with self.lock:
            lines = [f"Total Objects: {self.total_count}"]
            #for class_name in sorted(self.object_counts.keys()):
            #    if self.object_counts[class_name] > 0:
            #        count = self.object_counts[class_name]
            #        lines.append(f"{class_name}: {count}")
            return " | ".join(lines)
    
    def get_json(self):
        """
        Returns detection data as JSON string
        """
        with self.lock:
            data = {
                'timestamp': self.timestamp,
                'frame_count': self.frame_count,
                'detections': self.detections,
                'active_counts': {
                    'total': self.total_count,
                    'by_class': {k: v for k, v in self.object_counts.items()}
                }
            }
            return json.dumps(data)
    
    def get_dict(self):
        with self.lock:
            return {
                'timestamp': self.timestamp, 
                'frame_count': self.frame_count, 
                'detections': self.detections,
                'active_counts': {
                    'total': self.total_count,
                    'by_class': {k: v for k, v in self.object_counts.items()}
                }
            }

class RTSPServer:
    """
    RTSP server to stream the processed video from pipeline with detections and tracking
    """
    def __init__(self, port=8554, mount_point="/hailo_stream"):
        self.port = port
        self.mount_point = mount_point
        self.server = None
        self.appsrc = None
        self.appsrc_lock = threading.Lock()
        self.is_ready = False
        
    def create_server(self):
        """
        Creates and configures the RTSP server
        """
        # Create RTSP server instance
        self.server = GstRtspServer.RTSPServer.new()
        self.server.set_service(str(self.port))
        # Get the mount points object
        mounts = self.server.get_mount_points()
        # Create factory for the stream
        factory = GstRtspServer.RTSPMediaFactory.new()
        # Define the RTSP pipeline
        # This pipeline receives video from our main pipeline via appsrc and streams it via RTSP with H.264 encoding
        factory.set_launch(
            "( appsrc name=mysrc is-live=true format=time do-timestamp=true "
            "! video/x-raw,format=I420,width=640,height=640,framerate=30/1 "
            "! videoconvert "
            "! videoscale "
            "! video/x-raw,width=1536,height=864 "
            "! x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast "
            "! rtph264pay name=pay0 pt=96 )"
        )
        # Allow multiple clients to connect
        factory.set_shared(True)
        # Connect to media-configure signal to get appsrc
        factory.connect("media-configure", self.on_media_configure)
        # Add the factory to the mount point
        mounts.add_factory(self.mount_point, factory)
        print(f"RTSP server ready at rtsp://localhost:{self.port}{self.mount_point}")
    
    def on_media_configure(self, factory, media):
        """Called when media is configured - get the appsrc element"""
        element = media.get_element()
        appsrc = element.get_child_by_name("mysrc")
        if appsrc:
            with self.appsrc_lock:
                self.appsrc = appsrc
                self.is_ready = False  # Will be ready when it reaches PLAYING state
            # Connect to state-changed signal to detect when ready
            element.connect("pad-added", self.on_pad_added)
            print("[INFO] RTSP appsrc configured")
    
    def on_pad_added(self, element, pad):
        """Called when pad is added - indicates media is ready"""
        with self.appsrc_lock:
            self.is_ready = True
    
    def push_buffer(self, buffer):
        """Thread-safe buffer push with state checking"""
        with self.appsrc_lock:
            if not self.appsrc:
                return Gst.FlowReturn.OK
            # Check appsrc state before pushing
            state = self.appsrc.get_state(0)
            if state[1] != Gst.State.PLAYING:
                return Gst.FlowReturn.OK  # Silently skip if not playing
            
            try:
                ret = self.appsrc.emit("push-buffer", buffer)
                return ret
            except Exception as e:
                print(f"[ERROR] RTSP push-buffer exception: {e}")
                return Gst.FlowReturn.ERROR
    
    def start(self):
        """
        Attaches the server to the main context
        """
        if self.server:
            self.server.attach(None)

class HailoTrackerPipeline:
    """
    Main class for managing the GStreamer pipeline with Hailo detection and tracking.
    """
    def __init__(self, rpicamera=False, rtsp_link=None, video_file=None, enable_rtsp=False, 
                 enable_websocket=False, enable_mqtt=False, enable_recording=False, 
                 output_video_path=None, enable_debug=False):
        self.pipeline = None
        self.loop = None
        self.rpicamera = rpicamera
        self.rtsp_link = rtsp_link
        self.video_file = video_file
        self.enable_rtsp = enable_rtsp
        self.enable_websocket = enable_websocket
        self.enable_mqtt = enable_mqtt
        self.enable_recording = enable_recording
        self.output_video_path = output_video_path
        self.enable_debug = enable_debug
        self.bus = None
        # Video analysis mode
        self.video_mode = output_video_path is not None
        # Statistics for video analysis
        self.statistics = VideoStatistics() if self.video_mode else None
        # Detection data storage
        self.detection_data = DetectionData(debug=enable_debug)
        # RTSP server
        self.rtsp_server = None
        if enable_rtsp: self.rtsp_server = RTSPServer(port=8554, mount_point="/hailo_stream")
        # WebSocket server
        self.websocket_server = None
        if enable_websocket: self.websocket_server = WebSocketServerWithDetectionData(detection_data=self.detection_data, host="0.0.0.0", port=8765)
        # MQTT publisher
        self.mqtt_publisher = None
        if enable_mqtt: self.mqtt_publisher = MQTTPublisher(broker_host="mqtt.portabo.cz", broker_port=8883, topic="/videoanalyza", client_id="hailo_tracker_client", username="videoanalyza", password="phdA9ZNW1vfkXdJkhhbP")
        # frame count
        self.frame_count = 0
        # Set of sent track IDs to avoid duplicates
        self.sent_track_ids = set()
        # Recording
        self.recording_sink = None
        # Dynamic overlay text for count display
        self.count_overlay = ""  
        # Running flag
        self.running = False
        
    def create_pipeline(self):
        """
        Creates the GStreamer pipeline with the following flow:
        1. libcamerasrc, rstpsrc - Captures video from raspberry IMX camera or RTSP source
        2. capsfilter - Sets video format and resolution
        3. videoconvert - Converts color format if needed
        4. hailonet - Runs YOLOv8m detection on Hailo accelerator
        5. hailofilter - Post-processes detection results
        6. hailotracker - Tracks detected objects across frames
        7. hailooverlay - Draws bounding boxes and tracking IDs
        8. videoconvert - Prepares for display/encoding, RSTP/video recording
        """ 
        # Define the video source based on mode
        if self.video_file:
            if not os.path.exists(self.video_file):
                print(f"[ERROR] Video file not found: {self.video_file}")
                return False
            pipeline_str = (
                f"filesrc location={self.video_file} "
                "! decodebin "
                "! videoconvert "
                "! videoscale "
            )
        elif self.rpicamera:
            pipeline_str = (
                "libcamerasrc "
                "! video/x-raw, format=NV12, width=1536,height=864, framerate=30/1 " # for rpicamera
                "! videoconvert "
            )
        else:
            if self.rtsp_link is None:
                self.rtsp_link = "admin:Dcuk.123456@192.168.37.99/Stream"
            pipeline_str = (
                f"rtspsrc location=rtsp://{self.rtsp_link} latency=150 drop-on-latency=true "
                "! rtph264depay "
                "! h264parse "
                "! avdec_h264 "
                "! videoconvert "
                "! videoscale "
                "! video/x-raw,width=1280,height=720,format=RGB " 
                "! videocrop top=80 left=200 right=440 bottom=0 "  # Crop to bottom-left 640x640, top=440 left=320 right=960 bottom=0 for fullhd
            )
        pipeline_str += (
            "! videoscale method=lanczos "
            "! video/x-raw,width=640,height=640,format=RGB "
            "! hailonet hef-path=/home/imang/tappas/apps/detection/resources/h8/yolov8m.hef batch-size=1 "
            "nms-score-threshold=0.6 nms-iou-threshold=0.5 name=hailonet "
            "! queue leaky=downstream max-size-buffers=5 max-size-bytes=0 max-size-time=0 " # allow some leakiness before hailofilter
            "! hailofilter qos=false name=hailofilter function-name=yolov8m "
            "so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so "
            "! queue leaky=no max-size-buffers=5 max-size-bytes=0 max-size-time=0 " # After hailofilter, keep tight 
            "! hailotracker keep-past-metadata=true kalman-dist-thr=0.85 iou-thr=0.5 keep-new-frames=5 keep-tracked-frames=45 keep-lost-frames=20 name=hailotracker "
            "! hailooverlay show-confidence=true line-thickness=2 name=hailooverlay "
            "! textoverlay text='Hailo Tracker - YOLOv8m' valignment=top halignment=left font-desc='Sans 12' "
            "! textoverlay text='Counting...' valignment=bottom halignment=left font-desc='Sans Bold 14' name=count_overlay "
            "! videoconvert "
            "! video/x-raw,format=I420 "
        )
        if self.enable_rtsp or self.enable_recording:
            pipeline_str += "! tee name=t "
            branches_added = False
            # RTSP branch
            if self.enable_rtsp and not self.video_mode:
                self.rtsp_server.create_server()
                self.rtsp_server.start()
                pipeline_str += (
                    "t. "
                    "! queue leaky=downstream max-size-buffers=3 "
                    "! videoconvert "
                    "! video/x-raw,format=I420,width=640,height=640 "  # Keep same resolution
                    "! appsink name=rtsp_sink emit-signals=true sync=false drop=true max-buffers=1 "
                 )
                branches_added = True
            # Recording branch
            if self.enable_recording and self.output_video_path:
                pipeline_str += (
                    "t. "
                    "! queue max-size-buffers=30 "
                    "! videoconvert "
                    "! videorate "
                    "! video/x-raw,format=I420 "
                    "! x264enc tune=zerolatency bitrate=4000 speed-preset=medium key-int-max=30 "
                    "! h264parse "
                    "! mp4mux fragment-duration=1000 streamable=true "
                    f"! filesink location={self.output_video_path} sync=false name=recording_sink "
                )
                branches_added = True
                print(f"[INFO] Recording enabled, output: {self.output_video_path}")

            if not branches_added:
                pipeline_str += "t. ! fakesink sync=false async=false "
        else:
            sync_mode = "false" if self.video_mode else "false"
            pipeline_str += f"! fakesink sync={sync_mode} async=false "
    
        if self.enable_debug:
            print("[DEBUG] Pipeline string:")
            print(pipeline_str)
            print()
        try:
            self.pipeline = Gst.parse_launch(pipeline_str) # Parse and create the pipeline from the string
            if self.pipeline is None:
                print("[ERROR] Pipeline is None after parse_launch")
                return False  # Return False to indicate failure
        except Exception as e:
            print(f"[ERROR] Failed to create pipeline: {e}")
            return
         # Get the count overlay element
        self.count_overlay = self.pipeline.get_by_name('count_overlay')
        # Set up buffer probe on hailofilter to extract detection metadata
        hailotracker = self.pipeline.get_by_name('hailotracker') # Get the hailofilter element to extract detection data
        if hailotracker:
            pad = hailotracker.get_static_pad('src') # Connect to the pad to intercept metadata
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, self.buffer_probe_callback)
        else:
            print("[WARNING] Could not find hailofilter element")
        # Set up bus to handle messages
        self.bus = self.pipeline.get_bus() # Connect to the bus to handle messages
        self.bus.add_signal_watch() 
        self.bus.connect("message", self.on_message)
        # Set running flag to True
        self.running = True

    def buffer_probe_callback(self, pad, info):
        """
        Callback function to extract detection and tracker metadata from GStreamer buffers
        This is called for each frame passing through the pipeline
        Args:
            pad: GStreamer pad object
            info: Probe info containing buffer data
        """
        # Get the buffer
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK
        # Extract Hailo detection metadata from the buffer
        roi = get_roi_from_buffer(buffer)
        # Get detections from the ROI
        hailo_detections = roi.get_objects_typed(HAILO_DETECTION)
        # Prepare detection list
        detection_list = []
        current_timestamp = datetime.now().isoformat()
        # Process each detection
        for detection in hailo_detections:
            class_id = detection.get_class_id()
            label = detection.get_label()
            confidence = detection.get_confidence()
            bbox = detection.get_bbox()
            bbox_coords = [
                float(bbox.xmin()),
                float(bbox.ymin()),
                float(bbox.xmin() + bbox.width()),
                float(bbox.ymin() + bbox.height())
            ]
            # check zone
            cx = (bbox.xmin() + bbox.width() * 0.5)
            cy = (bbox.ymin() + bbox.height() * 0.5) 
            # filter zone
            in_zone = (0.15 <= cx <= 0.5 and 0.85 <= cy <= 0.99)
            if not in_zone:
                #print(f"y={cy} not in zone")
                continue
            # Get tracking ID if available
            tracking_id = None
            track = detection.get_objects_typed(HAILO_UNIQUE_ID)
            if len(track) == 1: tracking_id = track[0].get_id()
            # Filter by class (only vehicles and persons)
            if label not in ["car", "motorcycle", "bicycle", "truck", "bus", "person"]:  # Threshold for valid detections
                #print(f"[DEBUG] Skipping detection with label(not vehicle): {label}")
                continue
            # Avoid sending duplicate track IDs
            if tracking_id in self.sent_track_ids:
                continue
            # Create detection dictionary
            detection_dict = {
                'class_id': class_id,
                'class_name': label,
                'confidence': float(confidence),
                'bbox': bbox_coords,
                'track_id': tracking_id if tracking_id is not None else None,
                'timestamp': current_timestamp
            }
            detection_list.append(detection_dict)
            # Update detection data with tracked objects
            self.detection_data.update(detection_list, self.frame_count)
            # Publish to MQTT
            if detection_list and self.enable_mqtt and self.mqtt_publisher and self.mqtt_publisher.connected:
                data_dict = self.detection_data.get_dict()
                self.mqtt_publisher.publish(data_dict)
                self.sent_track_ids.add(tracking_id)  # ONLY after publish
            # Debug print
            if self.enable_debug:
                print(f"Label: {label}, Confidence: {confidence:.2f}, BBox: {bbox_coords}, Track ID: {tracking_id}")
        
        self.frame_count += 1
        # Update counting overlay every frame
        if self.count_overlay:
            overlay_text = self.detection_data.get_count_overlay_text()
            self.count_overlay.set_property('text', overlay_text)
        # Update statistics if in video mode
        if self.statistics:
            self.statistics.add_frame(has_detections=len(detection_list) > 0)
            for det in detection_list:
                self.statistics.add_detection(det['class_name'], det['confidence'], det.get('track_id'))
        return Gst.PadProbeReturn.OK
    
    def on_message(self, bus, message):
        """
        Handles GStreamer bus messages (errors, warnings, EOS, etc.)
        """
        t = message.type
        if t == Gst.MessageType.EOS: 
            print("[Pipeline] End of stream reached")
            if self.video_file: 
                self.stop()
        elif t == Gst.MessageType.ERROR: 
            err, debug = message.parse_error(); print(f"[Pipeline] Error: {err}"); print(f"[Pipeline] Debug info: {debug}"); self.stop()
        elif t == Gst.MessageType.WARNING: 
            warn, debug = message.parse_warning(); print(f"[Pipeline] Warning: {warn}")        
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed(); print(f"[Pipeline] Pipeline state: {old_state.value_nick} -> {new_state.value_nick}")
    
    def on_new_sample_rtsp(self, appsink):
        """Callback for appsink - pushes buffers to RTSP server"""
        try:
            sample = appsink.emit("pull-sample")
            if sample and self.rtsp_server:
                buffer = sample.get_buffer()
                ret = self.rtsp_server.push_buffer(buffer)
                # Only warn on actual errors, not FLUSHING
                if ret == Gst.FlowReturn.ERROR:
                    print(f"[ERROR] RTSP push-buffer failed")
            return Gst.FlowReturn.OK
        except Exception as e:
            print(f"[ERROR] RTSP sample callback error: {e}")
            return Gst.FlowReturn.ERROR
    
    


    def start(self):
        """
        Starts the pipeline with all configured features
        """
        # Create the pipeline for video analysis or live tracking
        if self.video_mode and self.output_video_path:
            print(f"[Pipeline] Starting Video Analysis Mode...")
            if self.statistics:
                self.statistics.start_time = datetime.now()
        else:
            print("[Pipeline] Starting Hailo Tracker Pipeline...")
        # connect to mqtt broker
        if self.enable_mqtt and self.mqtt_publisher:
            self.mqtt_publisher.connect()
        # Create the pipeline for rtsp streaming
        if self.enable_rtsp and self.rtsp_server:
            rtsp_sink = self.pipeline.get_by_name('rtsp_sink')
            if rtsp_sink:
                rtsp_sink.connect("new-sample", self.on_new_sample_rtsp)
                print("[INFO] RTSP sink connected")
        # Start WebSocket server
        if self.enable_websocket and self.websocket_server:
            ws_thread = threading.Thread(target=self.websocket_server.run, daemon=True)
            ws_thread.start()    
        # Set pipeline to PLAYING state
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] Failed to start pipeline")
            # Check each element's state
            it = self.pipeline.iterate_elements()
            while True:
                result, element = it.next()
                if result != Gst.IteratorResult.OK:
                    break
                state = element.get_state(0)
                print(f"Element {element.get_name()}: {state[1]}")
            self.pipeline.set_state(Gst.State.NULL)
            return False
        print("[pipeline] Pipeline started")
        if self.enable_rtsp: print("RTSP Stream: rtsp://localhost:8554/hailo_stream")
        if self.enable_websocket: print("WebSocket: ws://localhost:8765")
        if self.enable_mqtt: print(f"MQTT Topic: {self.mqtt_publisher.topic}")
        if self.enable_recording: print(f"Recording to: {self.output_video_path}")
        print("="*60)
        print("\n⌨Press Ctrl+C to stop\n")
        # Create and start the main loop
        self.loop = GLib.MainLoop()    
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n\n[Pipeline] Interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
        finally:
            # Always call stop() to save statistics and cleanup
            if not self.video_file:
                self.stop()
        return True
    
    def stop(self):
        """
        Stops the pipeline and cleans up resources
        """ 
        print("Stopping pipeline...")
        # Stop the pipeline
        if self.pipeline:
            # Send EOS to ensure proper file finalization
            print("[INFO] Sending End-of-Stream signal...")
            self.pipeline.send_event(Gst.Event.new_eos())    
            # Wait for EOS to propagate through pipeline
            bus = self.pipeline.get_bus()
            # Set timeout to 5 seconds
            msg = bus.timed_pop_filtered(
                5 * Gst.SECOND,
                Gst.MessageType.EOS | Gst.MessageType.ERROR
            )
            # Check message type
            if msg:
                if msg.type == Gst.MessageType.EOS:
                    print("[INFO] EOS received, file finalized properly")
                elif msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    print(f"[WARNING] Error during shutdown: {err}")
            else:
                print("[WARNING] EOS timeout - forcing shutdown")
            # Now set to NULL state
            self.pipeline.set_state(Gst.State.NULL)
        # Print statistics if in video mode
        if self.statistics:
            self.statistics.end_time = datetime.now()
            summary = self.statistics.print_summary()    
            stats_file = self.output_video_path.rsplit('.', 1)[0]
            # Save statistics to JSON file
            if not self.video_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stats_file += f"_statistics_{timestamp}.json"            
            else:
                stats_file += "_statistics.json"
            try:
                with open(stats_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"Statistics saved to: {stats_file}")
            except Exception as e:
                print(f"[ERROR] Failed to save statistics: {e}")
        # disconnect from MQTT
        if self.mqtt_publisher:
            self.mqtt_publisher.disconnect()
        # Stop WebSocket server
        if self.websocket_server:
            self.websocket_server.stop()
        # Quit the main loop
        if self.loop:
            self.loop.quit()
        # Set running flag to False
        self.running = False
        print("Pipeline stopped")
    
def main():
    """
    Main entry point for the application
    """
    import argparse
    parser = argparse.ArgumentParser(description='Hailo Detection Pipeline')
    parser.add_argument('--rpicamera', action='store_true', help='Use RPi camera stream (IP camera 192.168.37.99 stream default)')
    parser.add_argument('--rtsp-link', type=str, help='Use specific link to IP camera stream etc.: admin:password@192.168.37.99/Stream (if not set, uses default rtsp camera)')
    parser.add_argument('--video-file', type=str, help='Enter path to video file for analysis (if not set, uses camera or RTSP)')
    parser.add_argument('--enable-rtsp', action='store_true', help='Enable RTSP streaming (default False)')
    parser.add_argument('--enable-websocket', action='store_true', help='Enable WebSocket (default False)')
    parser.add_argument('--enable-mqtt', action='store_true', help='Enable MQTT (default False)')
    parser.add_argument('--enable-recording', action='store_true', help='Enable recording output video (default False)')
    parser.add_argument('--output-path', type=str, help='Output video file path if recording is enabled')
    parser.add_argument('--debug', action='store_true', help='Enable debug output (default False)')

    args = parser.parse_args()
    print("=" * 60)
    print("Hailo Tracker Pipeline - Raspberry Pi AI HAT+ (26 TOPS)")
    print("=" * 60)
    rtsp_link = args.rtsp_link
    # Determine source
    video_file = args.video_file
    use_camera = args.rpicamera
    if video_file:
        print(f"Video File: {video_file}")
        print("Mode: Video Analysis")
    elif use_camera:
        print("Camera: IMX708")
    else:
        print("Source: RTSP Stream")
    print("Model: YOLOv8m")
    print("Accelerator: Hailo 26 TOPS")
    print("=" * 60)
    print(f"GStreamer version: {Gst.version_string()}")
    print() 
    # Set output path for recording
    output_path = args.output_path
    if args.enable_recording and not output_path:
        if video_file:
            base_name = os.path.splitext(video_file)[0]
            output_path = f"{base_name}_processed.mp4"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"hailo_output_{timestamp}.mp4"

    # Create and start the pipeline with all features enabled
    tracker = HailoTrackerPipeline(
        rpicamera=use_camera, 
        rtsp_link=rtsp_link,
        video_file=video_file,
        enable_rtsp=args.enable_rtsp, 
        enable_websocket=args.enable_websocket,  
        enable_mqtt=args.enable_mqtt,
        enable_recording=args.enable_recording,
        output_video_path=output_path,
        enable_debug=args.debug,
    )
    tracker.create_pipeline()
    tracker.start()
    return 0

if __name__ == "__main__": sys.exit(main())

"""
CONFIGURATION NOTES:
====================
1. RTSP Streaming:
   - Default port: 8554
   - Mount point: /hailo_stream
   - URL: rtsp://YOUR_PI_IP:8554/hailo_stream
   - View with: ffplay rtsp://YOUR_PI_IP:8554/hailo_stream
   - Or VLC: vlc rtsp://YOUR_PI_IP:8554/hailo_stream
   - Encoded with H.264 for wide compatibility
2. MQTT Publisher:
    - Broker: mqtt.portabo.cz
    - Port: 8883 (TLS)
    - Topic: /videoanalyza
    - Client ID: hailo_tracker_client
    - Username: videoanalyza
    - Publishes JSON data with detection results
3. WebSocket Server for detections sending:
   - Default port: 8765
   - Sends JSON data with detection results
   - Connect with WebSocket client
4. Model Files:
   - Update hef-path and config-path with your actual file locations
   - Ensure Hailo runtime is properly installed
   - post-process shared object path is set correctly
5. Performance Optimization:
   - Adjust bitrate in x264enc (default: 2000 kbps)
   - Modify framerate if needed
   - Lower resolution for better performance
   - Adjust WebSocket update frequency (currently 10 Hz)
6. Firewall Configuration:
   - Open port 8554 for RTSP
   - Open port 8765 for WebSocket
   - Example: sudo ufw allow 8554/tcp
   - Example: sudo ufw allow 8765/tcp
7. Dependencies:
   - GStreamer 1.0+ with Python bindings
   - GStreamer RTSP Server (python3-gst-1.0, gstreamer1.0-rtsp)
   - websockets library: pip install websockets
   - paho-mqtt library: pip install paho-mqtt
   - Hailo GStreamer plugins
   - rpicam-apps and rpicamsrc plugin or rstpsrc for RTSP input
9. Multiple Outputs:
   - Can enable/disable RTSP, video recording, WebSocket, MQTT and display independently
   - Useful for headless operation (disable_display=False)
   - Can run multiple instances with different ports
USAGE:
======
python3 DetectionWithGStreamer_pipeline.py
Then connect to:
- RTSP: rtsp://raspberry-pi-ip:8554/hailo_stream
- MQTT: mqtt.portabo.cz topic /videoanalyza
- WebSocket: ws://raspberry-pi-ip:8765

RUN IN BACKGROUND:
nohup python3 AI_traffic_detection_hailo/DetectionWithGStreamer_pipeline.py --enable-mqtt > detection.log 2>&1 &
"""