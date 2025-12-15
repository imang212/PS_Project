#!/usr/bin/env python3
"""
GStreamer Pipeline with Hailo Tracker for Raspberry Pi AI HAT+ (26 TOPS)
=========================================================================
This script demonstrates object detection and tracking using:
- IMX708 camera via rpicam-apps
- YOLOv8m model for detection
- Hailo HW accelerator for inference
- hailotracker for object tracking
- WebRTC streaming (browser-compatible)
- MQTT protocol for detection data
- RTSP streaming (legacy support)
- WebSocket for real-time data
"""
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
gi.require_version('GstWebRTC', '1.0')
gi.require_version('GstSdp', '1.0')
from gi.repository import Gst, GLib, GstRtspServer, GstWebRTC, GstSdp
import asyncio
import json
import threading
import sys
from datetime import datetime
from MQTT import MQTTPublisher
import websockets
try:
    from hailo import get_roi_from_buffer, HAILO_DETECTION
except ImportError as e:
    print(f"Error message: {e}") # The error message tells you exactly which .so version it's looking for

# Initialize GStreamer
Gst.init(None)

class DetectionData:
    """
    Stores detection and tracking data to be sent via WebSocket
    """
    def __init__(self):
        self.detections = []
        self.frame_count = 0
        self.timestamp = None
        self.lock = threading.Lock()
        self.last_sent_detections = None  # Track last sent data
        self.has_new_data = False  # Flag for new data
    
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
            # Print detection summary every 60 frames
            if self.frame_count % 60 == 0:
                self._print_summary()
    
    def _print_summary(self):
        """
        Prints detection summary (called from within locked context)
        """
        print(f"\n[DETECTION SUMMARY Frame {self.frame_count}]")
        print(f"  Total detections: {len(self.detections)}")
        for idx, det in enumerate(self.detections):
            bbox_tuple = (det['bbox']['x'], det['bbox']['y'], 
                         det['bbox']['width'], det['bbox']['height'])
            tracking_info = f"[ID:{det['tracking_id']}]" if det['tracking_id'] is not None else "[No ID]"
            print(f"  [{idx}] {det['class']}: {det['confidence']:.2f} "
                  f"@ bbox{bbox_tuple} {tracking_info}")
    
    def get_json_if_new(self):
        """
        Returns detection data as JSON string only if it's new data
        Returns None if no new data
        """
        with self.lock:
            if not self.has_new_data:
                return None
            data = {
                'timestamp': self.timestamp,
                'frame_count': self.frame_count,
                'detections': self.detections
            }
            current_json = json.dumps(data)
            # Check if data actually changed
            if current_json == self.last_sent_detections:
                return None
            self.last_sent_detections = current_json
            self.has_new_data = False
            return current_json
        
    def get_json(self):
        """
        Returns detection data as JSON string
        """
        with self.lock:
            data = {
                'timestamp': self.timestamp,
                'frame_count': self.frame_count,
                'detections': self.detections
            }
            return json.dumps(data)
    
    def get_dict(self):
        with self.lock:
            return {'timestamp': self.timestamp, 'frame_count': self.frame_count, 'detections': self.detections}

class WebSocketServerWithDetectionData:
    """
    WebSocket server to send real-time detection data to clients
    """
    def __init__(self, detection_data, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.detection_data = detection_data
        self.clients = set()
        
    async def handler(self, websocket):
        """
        Handles WebSocket client connections and sends detection data 
        Args:
            websocket: WebSocket connection object
            path: Connection path
        """
        # Register new client
        self.clients.add(websocket)
        print(f"New WebSocket client connected from {websocket.remote_address}")
        try:
            # Send initial connection message
            await websocket.send(json.dumps({
                'type': 'connection',
                'message': 'Connected to Hailo Tracker',
                'timestamp': datetime.now().isoformat()
            }))
            # Continuously send detection data
            while True:
                # Get detection data only if it's new
                data_json = self.detection_data.get_json_if_new()
                # Send to client
                if data_json is not None:
                    # Send to WebSocket client
                    await websocket.send(data_json)
                    print(f"Sent detections at frame {self.detection_data.frame_count}")
                # Check frequently but only send when there's new data
                await asyncio.sleep(0.1)  # ~30 FPS check rate
        except websockets.exceptions.ConnectionClosed:
            print(f"Client {websocket.remote_address} disconnected")
        finally:
            # Unregister client
            self.clients.remove(websocket)
    
    async def start_server(self):
        """
        Starts the WebSocket server
        """
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever
    
    def run(self):
        """
        Runs the WebSocket server in an asyncio event loop
        """
        asyncio.run(self.start_server())

class WebRTCStreamer:
    """WebRTC streaming server using GStreamer webrtcbin"""
    def __init__(self, signaling_server="ws://localhost:8443"):
        self.signaling_server = signaling_server
        self.webrtc_pipeline = None
        self.webrtcbin = None
        self.signaling_ws = None
        
    def create_webrtc_pipeline(self):
        """
        Creates a WebRTC streaming pipeline.
        This pipeline receives video via appsrc and streams via WebRTC.
        """
        pipeline_str = (
            "webrtcbin name=webrtc stun-server=stun://stun.l.google.com:19302 "
            "appsrc name=webrtc_src format=time is-live=true "
            "! videoconvert "
            "! video/x-raw,format=I420 "
            "! vp8enc deadline=1 target-bitrate=2000000 "
            "! rtpvp8pay pt=96 "
            "! webrtc. "
        )
        self.webrtc_pipeline = Gst.parse_launch(pipeline_str)
        self.webrtcbin = self.webrtc_pipeline.get_by_name('webrtc')
        # Connect WebRTC signals
        self.webrtcbin.connect('on-negotiation-needed', self.on_negotiation_needed)
        self.webrtcbin.connect('on-ice-candidate', self.on_ice_candidate)
        return self.webrtc_pipeline
    
    def on_negotiation_needed(self, webrtcbin):
        """Handle WebRTC negotiation"""
        print("[WebRTC] Negotiation needed")
        promise = Gst.Promise.new_with_change_func(self.on_offer_created, webrtcbin, None)
        webrtcbin.emit('create-offer', None, promise)
    
    def on_offer_created(self, promise, webrtcbin, _):
        """Handle offer creation"""
        promise.wait()
        reply = promise.get_reply()
        offer = reply['offer']
        promise = Gst.Promise.new()
        webrtcbin.emit('set-local-description', offer, promise)
        promise.interrupt()        
        # Send offer to signaling server
        self.send_sdp_offer(offer)
    
    def on_ice_candidate(self, webrtcbin, mlineindex, candidate):
        """Handle ICE candidate"""
        print(f"[WebRTC] ICE candidate: {candidate}")
        # Send to signaling server
        if self.signaling_ws:
            asyncio.create_task(self.signaling_ws.send(json.dumps({
                'type': 'ice',
                'candidate': candidate,
                'sdpMLineIndex': mlineindex
            })))
    
    def send_sdp_offer(self, offer):
        """Send SDP offer to signaling server"""
        sdp = offer.sdp.as_text()
        print(f"[WebRTC] Sending offer")
        if self.signaling_ws:
            asyncio.create_task(self.signaling_ws.send(json.dumps({
                'type': 'offer',
                'sdp': sdp
            })))
    
    def set_remote_description(self, sdp_type, sdp_text):
        """Set remote SDP description"""
        ret, sdp = GstSdp.SDPMessage.new_from_text(sdp_text)
        if ret != GstSdp.SDPResult.OK:
            print("[WebRTC] Failed to parse SDP")
            return
        if sdp_type == 'answer':
            answer = GstWebRTC.WebRTCSessionDescription.new(
                GstWebRTC.WebRTCSDPType.ANSWER, sdp
            )
            promise = Gst.Promise.new()
            self.webrtcbin.emit('set-remote-description', answer, promise)
            promise.interrupt()
    
    def add_ice_candidate(self, mlineindex, candidate):
        """Add ICE candidate"""
        self.webrtcbin.emit('add-ice-candidate', mlineindex, candidate)

class RTSPServer:
    """
    RTSP server to stream the processed video with detections and tracking
    """
    def __init__(self, port=8554, mount_point="/hailo_stream"):
        self.port = port
        self.mount_point = mount_point
        self.server = None
        
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
            "( appsrc name=mysrc is-live=true format=time "
            "! video/x-raw,format=I420,width=1536,height=864,framerate=30/1 "
            "! x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast "
            "! rtph264pay name=pay0 pt=96 )"
        )
        # Allow multiple clients to connect
        factory.set_shared(True)
        # Add the factory to the mount point
        mounts.add_factory(self.mount_point, factory)
        print(f"RTSP server ready at rtsp://localhost:{self.port}{self.mount_point}")
        
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
    def __init__(self, enable_rtsp=True, enable_websocket=True, enable_webrtc=True, enable_mqtt=True, enable_display=True):
        self.pipeline = None
        self.loop = None
        self.enable_rtsp = enable_rtsp
        self.enable_websocket = enable_websocket
        self.enable_webrtc = enable_webrtc
        self.enable_mqtt = enable_mqtt
        self.bus = None
        # Detection data storage
        self.detection_data = DetectionData()
        # RTSP server
        self.rtsp_server = None
        if enable_rtsp: self.rtsp_server = RTSPServer(port=8554, mount_point="/hailo_stream")
        # WebSocket server
        self.websocket_server = None
        if enable_websocket: self.websocket_server = WebSocketServerWithDetectionData(detection_data=self.detection_data, host="0.0.0.0", port=8765)
        # WebRTC streamer
        self.webrtc_streamer = None
        if enable_webrtc: self.webrtc_streamer = WebRTCStreamer()
        # MQTT publisher
        self.mqtt_publisher = None
        if enable_mqtt: self.mqtt_publisher = MQTTPublisher(broker_host="mqtt.portabo.cz", broker_port=8883, topic="hailo/detections", client_id="hailo_tracker", username="videoanalyza", password="phdA9ZNW1vfkXdJkhhbP")
        # frame count
        self.frame_count = 0

    def create_pipeline(self):
        """
        Creates the GStreamer pipeline with the following flow:
        1. libcamerasrc - Captures video from IMX708 camera
        2. capsfilter - Sets video format and resolution
        3. videoconvert - Converts color format if needed
        4. hailonet - Runs YOLOv8m detection on Hailo accelerator
        5. hailofilter - Post-processes detection results
        6. hailotracker - Tracks detected objects across frames
        7. hailooverlay - Draws bounding boxes and tracking IDs
        8. videoconvert - Prepares for display/encoding
        """ 
        # Define the pipeline string
        pipeline_str = (
            "rtspsrc location=rtsp://admin:Dcuk.123456@192.168.37.99/Stream latency=0 "
            "! rtph264depay "
            "! h264parse "
            "! avdec_h264 "
            #"! video/x-raw, format=NV12, width=1536,height=864, framerate=30/1 " # for rpicamera
            "! videoconvert "
            "! videoscale "
            "! video/x-raw,format=RGB,width=640,height=640 "
            "! hailonet hef-path=/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef batch-size=1 "
            "! queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 "  # Add queue after hailonet
            "! hailofilter function-name=yolov8m qos=false name=hailofilter "
            #"config-path=/home/imang/hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config/yolov8m_nms_config.json "
            "so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so "
            "! queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 " # Add queue after hailofilter
            "! hailotracker kalman-dist-thr=0.7 iou-thr=0.5 keep-tracked-frames=30 keep-new-frames=3 keep-lost-frames=10 name=hailotracker "
            "! hailooverlay show-confidence=true line-thickness=2 name=hailooverlay "
            "! textoverlay text='Hailo Tracker - YOLOv8m' valignment=top halignment=left font-desc='Sans 12' "
            "! videoconvert "
            "! video/x-raw,format=I420 "
        )
        if self.enable_rtsp or self.enable_webrtc:
            pipeline_str += "! tee name=t "
            branches_added = False
            # RTSP branch
            if self.enable_rtsp:
                pipeline_str += (
                    "t. "
                    "! queue leaky=downstream max-size-buffers=3 "
                    "! videoconvert "
                    "! video/x-raw,format=I420,width=1536,height=864 "
                    "! appsink name=rtsp_sink emit-signals=true sync=false drop=true max-buffers=1 "
                 )
                branches_added = True
            # WebRTC branch
            if self.enable_webrtc:
                pipeline_str += (
                    "t. "
                    "! queue leaky=downstream max-size-buffers=3 "
                    "! videoconvert "
                    "! video/x-raw,format=I420 "
                    "! vp8enc deadline=1 target-bitrate=2000000 "
                    "! rtpvp8pay pt=96 "
                    "! udpsink host=127.0.0.1 port=5001 sync=false async=false "
                )
                branches_added = True    
            if not branches_added:
                pipeline_str += "t. ! fakesink sync=false async=false "
        else:
            pipeline_str += "! fakesink sync=false async=false "
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
        hailofilter = self.pipeline.get_by_name('hailofilter') # Get the hailofilter element to extract detection data
        if hailofilter:
            pad = hailofilter.get_static_pad('src') # Connect to the pad to intercept metadata
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, self.buffer_probe_callback)
        else:
            print("[WARNING] Could not find hailofilter element")
    
        self.bus = self.pipeline.get_bus() # Connect to the bus to handle messages
        self.bus.add_signal_watch() 
        self.bus.connect("message", self.on_message)
        
    def buffer_probe_callback(self, pad, info):
        """
        Callback function to extract detection metadata from GStreamer buffers
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
        # Get current timestamp
        current_timestamp = datetime.now().isoformat()
        # Process each detection
        for detection in hailo_detections:
            
            label = detection.get_label()
            confidence = detection.get_confidence()
            bbox = detection.get_bbox()
            # Get tracking ID if available
            tracking_id = detection.get_class_id()
            #if label in ["car", "motorcycle", "truck", "bus"] and confidence > 0.4:  # Threshold for valid detections
            #    label = detection.get_label()
            #    confidence = detection.get_confidence()
            #    bbox = detection.get_bbox()
            for det in roi.get_objects_typed(HAILO_DETECTION):
                print(dir(det))
                break  # print only once
            
            #Create detection dictionary
            if confidence > 0.5:
                detection_dict = {
                    'class': label,
                    'confidence': float(confidence),
                    'bbox': {
                        'x': float(bbox.xmin()),
                        'y': float(bbox.ymin()), 
                        'width': float(bbox.width()),
                        'height': float(bbox.height())
                    },
                    'tracking_id': tracking_id if tracking_id is not None else None,
                    'timestamp': current_timestamp
                }
                detection_list.append(detection_dict)
            #print(f"Label: {label}, Confidence: {confidence:.2f}")
            #print(f"BBox: x={bbox.xmin()}, y={bbox.ymin()}, "
            #      f"width={bbox.width()}, height={bbox.height()}")
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.detection_data.update(detection_list, self.frame_count)
        # Publish to MQTT
        if self.enable_mqtt and self.mqtt_publisher and self.mqtt_publisher.connected:
            if self.frame_count % 30 == 0:  # Publish every 10 frames to reduce load
                data_dict = self.detection_data.get_dict()
                self.mqtt_publisher.publish(data_dict)
        return Gst.PadProbeReturn.OK
    
    def on_message(self, bus, message):
        """
        Handles GStreamer bus messages (errors, warnings, EOS, etc.)
        """
        t = message.type
        if t == Gst.MessageType.EOS: 
            print("[Pipeline] End of stream reached"); self.stop()
        elif t == Gst.MessageType.ERROR: 
            err, debug = message.parse_error(); print(f"[Pipeline] Error: {err}"); print(f"[Pipeline] Debug info: {debug}"); self.stop()
        elif t == Gst.MessageType.WARNING: 
            warn, debug = message.parse_warning(); print(f"[Pipeline] Warning: {warn}")        
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed(); print(f"[Pipeline] Pipeline state: {old_state.value_nick} -> {new_state.value_nick}")
    
    def start(self):
        """
        Starts the pipeline, RTSP server, and WebSocket server
        """
        print("[pipeline] Starting Hailo Tracker Pipeline...")
        if self.enable_mqtt and self.mqtt_publisher:
            self.mqtt_publisher.connect()
        if self.enable_rtsp and self.rtsp_server:
            self.rtsp_server.create_server()
            self.rtsp_server.start()
        if self.enable_websocket and self.websocket_server:
            ws_thread = threading.Thread(target=self.websocket_server.run, daemon=True)
            ws_thread.start()
        if self.enable_webrtc and self.webrtc_streamer:
            print("[WebRTC] Note: Requires separate signaling server")
            print("WebSocket server started")
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
        if self.enable_webrtc: print("WebRTC: Requires signaling server")
        print("="*60)
        print("\n⌨Press Ctrl+C to stop\n")
        # Create and start the main loop
        self.loop = GLib.MainLoop()    
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n\n[Pipeline] Interrupted by user")
        return True
    
    def stop(self):
        """
        Stops the pipeline and cleans up resources
        """
        print("Stopping pipeline...")
        # Stop the pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        # disconnect form mgtt
        if self.mqtt_publisher:
            self.mqtt_publisher.disconnect()
        # Quit the main loop
        if self.loop:
            self.loop.quit()
        print("Pipeline stopped")

def main():
    """
    Main entry point for the application
    """
    print("=" * 60)
    print("Hailo Tracker Pipeline - Raspberry Pi AI HAT+ (26 TOPS)")
    print("=" * 60)
    print("Camera: IMX708")
    print("Model: YOLOv8m")
    print("Accelerator: Hailo 26 TOPS")
    print("Streaming: RTSP + WebSocket")
    print("=" * 60)
    print(f"GStreamer version: {Gst.version_string()}")
    print() 
    # Create and start the pipeline with all features enabled
    tracker = HailoTrackerPipeline(enable_rtsp=False, enable_websocket=True, enable_webrtc=False, enable_mqtt=False)
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
s   - Encoded with H.264 for wide compatibility
2. WebSocket Server:
   - Default port: 8765
   - Sends JSON data with detection results
   - Data format:
     {
       "timestamp": "2025-11-29T10:30:45.123",
       "frame_count": 1234,
       "detections": [
         {
           "class": "person",
           "confidence": 0.95,
           "bbox": {"x": 100, "y": 150, "width": 200, "height": 300},
           "tracking_id": 42
         }
       ]
     }
3. WebSocket Client Example (JavaScript):
   const ws = new WebSocket('ws://YOUR_PI_IP:8765');
   ws.onmessage = (event) => {
     const data = JSON.parse(event.data);
     console.log('Detections:', data.detections);
   };
4. WebSocket Client Example (Python):
   import asyncio
   import websockets
   async def receive_detections():
       uri = "ws://YOUR_PI_IP:8765"
       async with websockets.connect(uri) as websocket:
           while True:
               data = await websocket.recv()
               print(f"Received: {data}")
   asyncio.run(receive_detections())
5. Model Files:
   - Update hef-path and config-path with your actual file locations
   - Ensure Hailo runtime is properly installed
6. Performance Optimization:
   - Adjust bitrate in x264enc (default: 2000 kbps)
   - Modify framerate if needed
   - Lower resolution for better performance
   - Adjust WebSocket update frequency (currently 10 Hz)
7. Firewall Configuration:
   - Open port 8554 for RTSP
   - Open port 8765 for WebSocket
   - Example: sudo ufw allow 8554/tcp
   - Example: sudo ufw allow 8765/tcp
8. Dependencies:
   - GStreamer 1.0+ with Python bindings
   - GStreamer RTSP Server (python3-gst-1.0, gstreamer1.0-rtsp)
   - websockets library: pip install websockets
   - Hailo GStreamer plugins
   - rpicam-apps and rpicamsrc plugin
9. Multiple Outputs:
   - Can enable/disable RTSP, WebSocket, and display independently
   - Useful for headless operation (disable_display=False)
   - Can run multiple instances with different ports
INSTALLATION:
=============
sudo apt-get install gstreamer1.0-tools gstreamer1.0-rtsp
sudo apt-get install python3-gi python3-gst-1.0
pip3 install websockets
USAGE:
======
python3 hailo_tracker.py
Then connect to:
- RTSP: rtsp://raspberry-pi-ip:8554/hailo_stream
- WebSocket: ws://raspberry-pi-ip:8765

GStreamer testing command:
gst-launch-1.0 \
  libcamerasrc ! \
  video/x-raw,format=NV12,width=1536,height=864,framerate=30/1 ! \
  videoconvert ! \
  videoscale ! \
  video/x-raw,format=RGB, width=640,height=640 ! \
  hailonet hef-path=/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef name=hailonet ! \
  hailofilter function-name=yolov8 \
    config-path=/home/imang/hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config/yolov8m_nms_config.json \
    name=hailofilter ! \
  hailotracker kalman-dist-thr=0.7 iou-thr=0.3 keep-tracked-frames=30 keep-new-frames=3 keep-lost-frames=10 name=hailotracker ! \
  hailooverlay show-confidence=true line-thickness=2 name=hailooverlay ! \
  textoverlay text='Hailo Tracker - YOLOv8m' valignment=top halignment=left font-desc='Sans 12' ! \
  videoconvert ! \
  video/x-raw,format=I420 ! \
  tee name=t \
    t. ! queue ! fakesink

Test with rtsp:
gst-launch-1.0 \
  libcamerasrc ! \
  video/x-raw,format=NV12,width=1536,height=864,framerate=30/1 ! \
  videoconvert ! \
  videoscale ! \
  video/x-raw,format=RGB, width=640, height=640 ! \
  hailonet hef-path=/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef name=hailonet ! \
  hailofilter function-name=yolov8 \
    config-path=/home/imang/hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config/yolov8m_nms_config.json \
    name=hailofilter ! \
  hailotracker kalman-dist-thr=0.7 iou-thr=0.3 keep-tracked-frames=30 keep-new-frames=3 keep-lost-frames=10 
  name=hailotracker ! \
  hailooverlay show-confidence=true line-thickness=2 name=hailooverlay ! \
  textoverlay text='Hailo Tracker - YOLOv8m' valignment=top halignment=left font-desc='Sans 12' ! \
  videoconvert ! \
  video/x-raw,format=I420 ! \
  tee name=t \
     t. ! queue ! videoconvert ! video/x-raw,format=I420,width=1536,height=864 ! appsink name=rtsp_sink emit-signals=true sync=false \
      t. ! queue ! fakesink

Small test:  
  gst-launch-1.0 \
  libcamerasrc ! video/x-raw,format=NV12,width=1536,height=864,framerate=30/1 ! \
  videoconvert ! video/x-raw ! videoscale ! video/x-raw,width=640,height=640 ! \
  hailonet hef-path=/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef name=hailonet ! \
  hailofilter function-name=yolov8 \
    config-path=/home/imang/hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config/yolov8m_nms_config.json \
    so-path=/home/imang/hailo-rpi5-examples/resources/so/libyolov8_postprocess.so \
    name=hailofilter ! \
  fakesink

Test with window:
gst-launch-1.0 \
  rtspsrc location="rtsp://admin:Dcuk.123456@192.168.37.99/Stream" latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! \
  videoconvert ! \
  videoscale ! \
  video/x-raw,format=RGB,width=640,height=640 ! \
  hailonet hef-path=/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef batch-size=1 ! \
  queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
  hailofilter function-name=yolov8m \
    so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so \
    qos=false ! \
  queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! \
  hailotracker kalman-dist-thr=0.7 iou-thr=0.3 keep-tracked-frames=30 keep-new-frames=3 keep-lost-frames=10 ! \
  hailooverlay show-confidence=true line-thickness=2 ! \
  textoverlay text='Hailo Tracker - YOLOv8m' valignment=top halignment=left font-desc='Sans 12' ! \
  videoconvert ! \
  autovideosink

GStreamer test
gst-launch-1.0 \
    rtspsrc location="rtsp://admin:Dcuk.123456@192.168.37.99/Stream" latency=0 ! \
    rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! \
    videoscale ! \
    video/x-raw,format=RGB, width=640,height=640 ! \
    hailonet hef-path=/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef name=hailonet ! \
    hailofilter function-name=yolov8 \
        config-path=/home/imang/hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config/yolov8m_nms_config.json \
        name=hailofilter ! \
    hailotracker kalman-dist-thr=0.7 iou-thr=0.3 keep-tracked-frames=30 keep-new-frames=3 keep-lost-frames=10 name=hailotracker ! \
    hailooverlay show-confidence=true line-thickness=2 name=hailooverlay ! \
    textoverlay text='Hailo Tracker - YOLOv8m' valignment=top halignment=left font-desc='Sans 12' ! \
    autovideosink !
"""