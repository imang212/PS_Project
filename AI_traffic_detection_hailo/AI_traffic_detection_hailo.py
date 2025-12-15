import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstWebRTC', '1.0')
gi.require_version('GstSdp', '1.0')
from gi.repository import Gst, GLib, GstWebRTC, GstSdp
import threading
import asyncio
import json
from datetime import datetime
import hailo
#from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
import websockets
from aiohttp import web

# User-defined class to be used in the callback function
#class user_app_callback_class(app_callback_class):
#    def __init__(self, config):
#        super().__init__()
#        self.config = config
#        self.detected_cars = {}
#        self.frame_count = 0
#        print("[INIT] Car tracking initialized with WebSocket & streaming")
#        print(f"[INIT] Detecting classes: {config.vehicle_classes}")

# Configuration Class
class DetectionConfig:
    """Configuration options for car detection and streaming"""
    def __init__(self):
        # Detection settings
        self.vehicle_classes = {3: 'car', 6: 'bus', 8: 'truck'}
        self.track_class_id = 3  # Primary class to track (car)
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        # Video settings
        self.camera_type = 'ip'  # 'rpicam' or 'ip'
        self.ip_camera_url = 'rtsp://admin:password@192.168.1.100:554/stream'
        self.video_width = 1920
        self.video_height = 1080
        self.video_fps = 30
        self.video_format = 'RGB'
        # Network settings (WS, RTSP, WebRTC)
        self.ws_host = '0.0.0.0'
        self.ws_port = 8765
        self.http_port = 8080
        self.rtsp_port = 8554
        # WebRTC settings
        self.webrtc_stun_server = 'stun:stun.l.google.com:19302'
        self.webrtc_video_bitrate = 2000000  # 2 Mbps
        # RTSP settings
        self.rtsp_mount_point = '/stream'
        self.rtsp_latency = 200  # ms
        # Debug settings
        self.debug_mode = True
        self.summary_interval = 30  # frames

    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*60)
        print("DETECTION CONFIGURATION")
        print("="*60)
        print(f"Camera Type: {self.camera_type}")
        if self.camera_type == 'ip':
            print(f"IP Camera URL: {self.ip_camera_url}")
        print(f"Camera: {self.video_width}x{self.video_height} @ {self.video_fps}fps")
        print(f"Vehicle Classes: {self.vehicle_classes}")
        print(f"Track Class ID: {self.track_class_id}")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"WebSocket: ws://{self.ws_host}:{self.ws_port}")
        print(f"HTTP/WebRTC: http://{self.ws_host}:{self.http_port}")
        print(f"RTSP: rtsp://{self.ws_host}:{self.rtsp_port}{self.rtsp_mount_point}")
        print("="*60 + "\n")

# WebSocket Server Class
class WebSocketServer:
    """Manages WebSocket connections and broadcasts detection data"""
    def __init__(self, config):
        self.config = config
        self.connected_clients = set()
        self.loop = None
        self.server = None
        
    async def handle_client(self, websocket, path):
        """Handle individual WebSocket client connection"""
        self.connected_clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"[WS] Client connected: {client_addr}")
        try:
            # Send initial connection message
            await websocket.send(json.dumps({
                'type': 'connection',
                'status': 'connected',
                'message': 'Car tracking WebSocket connected',
                'config': {
                    'width': self.config.video_width,
                    'height': self.config.video_height,
                    'fps': self.config.video_fps
                }
            }))
            # Keep connection alive and handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            print(f"[WS] Client disconnected: {client_addr}")
        finally:
            self.connected_clients.remove(websocket)
    
    async def broadcast(self, data):
        """Broadcast detection data to all connected clients"""
        if self.connected_clients:
            message = json.dumps(data)
            await asyncio.gather(
                *[client.send(message) for client in self.connected_clients],
                return_exceptions=True
            )
    
    def start(self):
        """Start WebSocket server in a separate thread"""
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            start_server = websockets.serve(self.handle_client, self.config.ws_host, self.config.ws_port)
            print(f"[WS] WebSocket server started on ws://{self.config.ws_host}:{self.config.ws_port}")
            self.loop.run_until_complete(start_server)
            self.loop.run_forever()        
        ws_thread = threading.Thread(target=run_server, daemon=True)
        ws_thread.start()

# WebRTC Server Class
class WebRTCServer:
    """Manages WebRTC peer connections and streaming"""    
    def __init__(self, config):
        self.config = config
        self.peers = {}
        self.peer_id_counter = 0
        self.pipeline = None
        
    def set_pipeline(self, pipeline):
        """Set the GStreamer pipeline reference"""
        self.pipeline = pipeline
    
    def create_webrtc_bin(self, peer_id):
        """Create WebRTC bin and add to pipeline"""
        if not self.pipeline:
            print("[WebRTC ERROR] Pipeline not set")
            return None            
        print(f"[WebRTC] Creating WebRTC bin for peer {peer_id}")
        # Create webrtcbin element
        webrtcbin = Gst.ElementFactory.make("webrtcbin", f"webrtc_{peer_id}")
        webrtcbin.set_property("bundle-policy", "max-bundle")
        webrtcbin.set_property("stun-server", self.config.webrtc_stun_server)
        # Add to pipeline
        self.pipeline.add(webrtcbin)
        # Get the tee element from the pipeline
        tee = self.pipeline.get_by_name("video_tee")
        if not tee:
            print("[WebRTC ERROR] No tee element found in pipeline")
            return None
        # Create queue for this WebRTC stream
        queue = Gst.ElementFactory.make("queue", f"queue_webrtc_{peer_id}")
        videoconvert = Gst.ElementFactory.make("videoconvert", f"convert_webrtc_{peer_id}")
        x264enc = Gst.ElementFactory.make("x264enc", f"x264enc_{peer_id}")
        rtph264pay = Gst.ElementFactory.make("rtph264pay", f"rtph264pay_{peer_id}")
        # Configure encoder
        x264enc.set_property("bitrate", self.config.webrtc_video_bitrate // 1000)
        x264enc.set_property("speed-preset", "ultrafast")
        x264enc.set_property("tune", "zerolatency")
        # Configure payloader
        rtph264pay.set_property("config-interval", 1)
        rtph264pay.set_property("pt", 96)
        # Add elements to pipeline
        self.pipeline.add(queue)
        self.pipeline.add(videoconvert)
        self.pipeline.add(x264enc)
        self.pipeline.add(rtph264pay)
        # Link: tee -> queue -> videoconvert -> x264enc -> rtph264pay -> webrtcbin
        tee.link(queue)
        queue.link(videoconvert)
        videoconvert.link(x264enc)
        x264enc.link(rtph264pay)
        rtph264pay.link(webrtcbin)
        # Sync state
        queue.sync_state_with_parent()
        videoconvert.sync_state_with_parent()
        x264enc.sync_state_with_parent()
        rtph264pay.sync_state_with_parent()
        webrtcbin.sync_state_with_parent()
        return webrtcbin
    
    async def handle_offer(self, request):
        """Handle WebRTC offer from client"""
        params = await request.json()
        peer_id = self.peer_id_counter
        self.peer_id_counter += 1
        print(f"[WebRTC] Received offer from peer {peer_id}")
        # Create WebRTC bin
        webrtcbin = self.create_webrtc_bin(peer_id)
        if not webrtcbin: return web.Response(status=500, text="Failed to create WebRTC bin")
        # Store peer connection
        self.peers[peer_id] = {'webrtcbin': webrtcbin, 'peer_id': peer_id}
        # Handle ICE candidates
        def on_ice_candidate(webrtc, mline, candidate):
            print(f"[WebRTC] ICE candidate: {candidate}")
        webrtcbin.connect("on-ice-candidate", on_ice_candidate)
        # Set remote description (offer)
        offer_sdp = params["sdp"]
        ret, sdp_msg = GstSdp.SDPMessage.new()
        GstSdp.sdp_message_parse_buffer(bytes(offer_sdp.encode()), sdp_msg)
        offer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.OFFER, sdp_msg)
        promise = Gst.Promise.new()
        webrtcbin.emit("set-remote-description", offer, promise)
        promise.interrupt()
        # Create answer
        def on_answer_created(promise):
            promise.wait()
            reply = promise.get_reply()
            answer = reply.get_value("answer")
            promise2 = Gst.Promise.new()
            webrtcbin.emit("set-local-description", answer, promise2)
            promise2.interrupt()        
            # Get SDP text
            answer_sdp = answer.sdp.as_text()
            print(f"[WebRTC] Created answer for peer {peer_id}")
            # Store for retrieval
            self.peers[peer_id]['answer_sdp'] = answer_sdp
        promise = Gst.Promise.new_with_change_func(lambda p, _: on_answer_created(p), None)
        webrtcbin.emit("create-answer", None, promise)
        # Wait for answer to be created
        await asyncio.sleep(0.5)
        answer_sdp = self.peers[peer_id].get('answer_sdp', '')
        return web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": answer_sdp, "type": "answer", "peer_id": peer_id})
        )
    
    def cleanup(self):
        """Cleanup all WebRTC connections"""
        for peer_id, peer in self.peers.items():
            peer['webrtcbin'].set_state(Gst.State.NULL)
        self.peers.clear()

# HTTP Server Class
class HTTPServer:
    """Manages HTTP server for WebRTC signaling and web interface"""
    def __init__(self, config, webrtc_server):
        self.config = config
        self.webrtc_server = webrtc_server
        
    async def index(self, request):
        """Serve WebRTC client HTML page"""
        content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Car Detection Stream</title>
            <style>body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; } video { width: 100%; max-width: 1280px; background: #000; border: 2px solid #333; border-radius: 8px; } .container { max-width: 1280px; margin: 0 auto; } .info { background: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 5px; } .detection { background: #3a3a3a; padding: 10px; margin: 5px 0; border-left: 3px solid #4CAF50; } .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 10px 0; } .stat { background: #2a2a2a; padding: 10px; text-align: center; border-radius: 5px; } .stat-value { font-size: 24px; font-weight: bold; color: #4CAF50; } h1 { color: #4CAF50; } button { background: #4CAF50; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 5px; margin: 5px; } button:hover { background: #45a049; } button:disabled { background: #666; cursor: not-allowed; } .controls { margin: 15px 0; } </style>
        </head>
        <body>
            <div class="container">
                <h1>Car Detection & Tracking System</h1>        
                <div class="controls">
                    <button id="startBtn" onclick="startStream()">Start Stream</button>
                    <button id="stopBtn" onclick="stopStream()" disabled>Stop Stream</button>
                </div>
                <video id="video" autoplay playsinline></video>
                <div class="stats">
                    <div class="stat">
                        <div>WebSocket</div>
                        <div class="stat-value" id="wsStatus">❌</div>
                    </div>
                    <div class="stat">
                        <div>WebRTC</div>
                        <div class="stat-value" id="rtcStatus">❌</div>
                    </div>
                    <div class="stat">
                        <div>Detections</div>
                        <div class="stat-value" id="detCount">0</div>
                    </div>
                    <div class="stat">
                        <div>Frame</div>
                        <div class="stat-value" id="frameCount">0</div>
                    </div>
                </div>
                <div class="info">
                    <h3>Live Detections</h3>
                    <div id="detections"></div>
                </div>
            </div>
            <script>
                let pc = null;
                let ws = null;
                let detectionCount = 0;
                function connectWebSocket() {
                    ws = new WebSocket('ws://' + window.location.hostname + ':8765');
                    ws.onopen = () => {
                        console.log('WebSocket Connected');
                        document.getElementById('wsStatus').textContent = '✅';
                    };            
                    ws.onclose = () => {
                        document.getElementById('wsStatus').textContent = '❌';
                        setTimeout(connectWebSocket, 2000);
                    };
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);  
                        if (data.type === 'detection') {
                            detectionCount++;
                            document.getElementById('detCount').textContent = detectionCount;
                            document.getElementById('frameCount').textContent = data.frame;
                            const detectionsDiv = document.getElementById('detections');
                            const det = document.createElement('div');
                            det.className = 'detection';
                            det.innerHTML = `
                                <strong>${data.class_name.toUpperCase()}</strong> | 
                                Track ID: ${data.track_id} | 
                                Confidence: ${(data.confidence * 100).toFixed(1)}% | 
                                BBox: (${data.bbox.x.toFixed(0)}, ${data.bbox.y.toFixed(0)}, ${data.bbox.width.toFixed(0)}x${data.bbox.height.toFixed(0)})
                            `;
                            detectionsDiv.insertBefore(det, detectionsDiv.firstChild);
                            if (detectionsDiv.children.length > 10) {
                                detectionsDiv.removeChild(detectionsDiv.lastChild);
                            }
                        }
                    };
                }
                async function startStream() {
                    const config = {
                        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                    };
                    pc = new RTCPeerConnection(config);
                    pc.ontrack = (event) => {
                        console.log('Received track:', event.track.kind);
                        document.getElementById('video').srcObject = event.streams[0];
                        document.getElementById('rtcStatus').textContent = '✅';
                    };
                    pc.onicecandidate = (event) => {
                        if (event.candidate) {
                            console.log('ICE candidate:', event.candidate);
                        }
                    };
                    pc.onconnectionstatechange = () => {
                        console.log('Connection state:', pc.connectionState);
                        if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
                            document.getElementById('rtcStatus').textContent = '❌';
                        }
                    };
                    // Create offer
                    const offer = await pc.createOffer();
                    await pc.setLocalDescription(offer);
                    // Send offer to server
                    const response = await fetch('/offer', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            sdp: offer.sdp,
                            type: offer.type
                        })
                    });
                    const answer = await response.json();
                    await pc.setRemoteDescription(new RTCSessionDescription(answer));
                    console.log('WebRTC connection established');
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                }
                function stopStream() {
                    if (pc) {
                        pc.close();
                        pc = null;
                        document.getElementById('video').srcObject = null;
                        document.getElementById('rtcStatus').textContent = '❌';
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                    }
                }
                // Connect WebSocket on load
                connectWebSocket();
            </script>
        </body>
        </html>
        """
        return web.Response(content_type="text/html", text=content)
    
    def start(self):
        """Start HTTP server in a separate thread"""
        def run_server():
            app = web.Application()
            app.router.add_get("/", self.index)
            app.router.add_post("/offer", self.webrtc_server.handle_offer)
            print(f"[HTTP] Web interface available at http://0.0.0.0:{self.config.http_port}")
            web.run_app(app, host='0.0.0.0', port=self.config.http_port, print=None)
        http_thread = threading.Thread(target=run_server, daemon=True)
        http_thread.start()

# Detection Pipeline Class
class DetectionPipeline:
    """Manages GStreamer pipeline with Hailo detection and tracking""" 
    def __init__(self, config, websocket_server):
        self.config = config
        self.websocket_server = websocket_server
        self.pipeline = None
        self.detected_cars = {}
        self.frame_count = 0
        self.main_loop = None
        self.detection_queue = asyncio.Queue()  # Queue for async detection processing
        
    def create_ip_camera_source(self):
        """Create IP camera source pipeline string"""
        return f"""
            rtspsrc location={self.config.ip_camera_url} latency={self.config.rtsp_latency} ! 
            rtph264depay ! 
            h264parse ! 
            avdec_h264 ! 
            videoconvert ! 
            videoscale ! 
            video/x-raw,format={self.config.video_format},width={self.config.video_width},height={self.config.video_height} !
        """
    
    def create_rpicam_source(self):
        """Create Raspberry Pi camera source pipeline string"""
        return f"""
            libcamerasrc ! 
            video/x-raw,format={self.config.video_format},width={self.config.video_width},height={self.config.video_height},framerate={self.config.video_fps}/1 !
        """
    
    def detection_callback(self, pad, info):
        """Callback function for processing detections"""
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        self.frame_count += 1
        # Get detections from buffer
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if len(detections) > 0 and self.config.debug_mode:
            print(f"\n[FRAME {self.frame_count}] Detected {len(detections)} object(s)")
        # Process detections
        detection_list = []
        for detection in detections:
            class_id = detection.get_class_id()
            # Only process vehicle detections
            if class_id not in self.config.vehicle_classes:
                continue
            # Get tracking ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) > 0:
                track_id = track[0].get_id()
            # Get detection details
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            # Filter by confidence threshold
            if confidence < self.config.confidence_threshold:
                continue
            class_name = self.config.vehicle_classes[class_id]
            # Debug output
            if self.config.debug_mode:
                print(f"  [DETECTION] {class_name.upper()} | Track ID: {track_id} | "
                      f"Confidence: {confidence:.2f} | "
                      f"BBox: ({bbox.xmin():.0f}, {bbox.ymin():.0f}, "
                      f"{bbox.width():.0f}x{bbox.height():.0f})")
            # Track unique vehicles
            if track_id not in self.detected_cars:
                self.detected_cars[track_id] = {
                    'first_seen': self.frame_count,
                    'class': class_name,
                    'count': 0
                }
                if self.config.debug_mode:
                    print(f"    -> NEW {class_name} tracked (ID: {track_id})")
            self.detected_cars[track_id]['count'] += 1
            self.detected_cars[track_id]['last_seen'] = self.frame_count
            # Prepare data for WebSocket broadcast
            detection_data = {
                'type': 'detection',
                'timestamp': current_time,
                'frame': self.frame_count,
                'track_id': track_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': {
                    'x': bbox.xmin(),
                    'y': bbox.ymin(),
                    'width': bbox.width(),
                    'height': bbox.height()
                }
            }
            detection_list.append(detection_data)
        # Put detections in async queue for processing
        if detection_list:
            try:
                self.detection_queue.put_nowait(detection_list)
            except asyncio.QueueFull:
                print("[WARNING] Detection queue full, dropping frame")
        # Print summary
        if (self.frame_count % self.config.summary_interval == 0 and 
            len(self.detected_cars) > 0 and self.config.debug_mode):
            print(f"\n[SUMMARY] Total unique vehicles tracked: {len(self.detected_cars)}")
            for track_id, info in self.detected_cars.items():
                print(f"  Track {track_id}: {info['class']} - "
                      f"seen {info['count']} times "
                      f"(frames {info['first_seen']}-{info.get('last_seen', info['first_seen'])})")
        return Gst.PadProbeReturn.OK
    
    async def process_detections_async(self):
        """Async task that processes detections from the queue"""
        print("[ASYNC] Starting async detection processor...")
        try:
            while True:
                # Wait for detections from the queue
                detection_list = await self.detection_queue.get()
                # Broadcast each detection via WebSocket
                for detection_data in detection_list:
                    if self.websocket_server.connected_clients:
                        await self.websocket_server.broadcast(detection_data)
                # Mark task as done
                self.detection_queue.task_done()
        except asyncio.CancelledError:
            print("[ASYNC] Detection processor cancelled")
            raise

    def create_pipeline(self):
        """Create and configure the GStreamer pipeline"""
        print("[PIPELINE] Creating GStreamer pipeline...")
        # Create pipeline
        self.pipeline = Gst.Pipeline.new("detection-pipeline")
        # Determine camera source
        if self.config.camera_type == 'ip':
            print(f"[PIPELINE] Using IP camera: {self.config.ip_camera_url}")
            # IP Camera source
            source = Gst.ElementFactory.make("rtspsrc", "camera-source")
            source.set_property("location", self.config.ip_camera_url)
            source.set_property("latency", self.config.rtsp_latency)
            depay = Gst.ElementFactory.make("rtph264depay", "depay")
            parse = Gst.ElementFactory.make("h264parse", "parse")
            decode = Gst.ElementFactory.make("avdec_h264", "decode")
            self.pipeline.add(source)
            self.pipeline.add(depay)
            self.pipeline.add(parse)
            self.pipeline.add(decode)
            # Link depay -> parse -> decode
            depay.link(parse)
            parse.link(decode)
            # Connect source pad dynamically (RTSP creates pads dynamically)
            def on_pad_added(src, pad):
                sink_pad = depay.get_static_pad("sink")
                if not sink_pad.is_linked():
                    pad.link(sink_pad)
            source.connect("pad-added", on_pad_added)
            # Last element to link with the rest of the pipeline
            last_element = decode
        else:  # rpicam
            print("[PIPELINE] Using Raspberry Pi camera")
            source = Gst.ElementFactory.make("libcamerasrc", "camera-source")
            caps_filter = Gst.ElementFactory.make("capsfilter", "caps")
            caps = Gst.Caps.from_string(
                f"video/x-raw,format={self.config.video_format},"
                f"width={self.config.camera_width},"
                f"height={self.config.camera_height},"
                f"framerate={self.config.camera_fps}/1"
            )
            caps_filter.set_property("caps", caps)
            self.pipeline.add(source)
            self.pipeline.add(caps_filter)
            source.link(caps_filter)
            last_element = caps_filter
        # Common pipeline elements
        videoconvert = Gst.ElementFactory.make("videoconvert", "convert")
        videoscale = Gst.ElementFactory.make("videoscale", "scale")
        caps_filter2 = Gst.ElementFactory.make("capsfilter", "caps2")
        caps2 = Gst.Caps.from_string(
            f"video/x-raw,format={self.config.video_format},"
            f"width={self.config.video_width},"
            f"height={self.config.video_height}"
        )
        caps_filter2.set_property("caps", caps2)
        # Add detection elements (placeholder - needs Hailo elements)
        queue1 = Gst.ElementFactory.make("queue", "queue1")
        # Tee for splitting stream
        tee = Gst.ElementFactory.make("tee", "video_tee")
        # Display branch
        queue_display = Gst.ElementFactory.make("queue", "queue_display")
        videoconvert_display = Gst.ElementFactory.make("videoconvert", "convert_display")
        videosink = Gst.ElementFactory.make("autovideosink", "videosink")
        # Add all elements
        self.pipeline.add(videoconvert)
        self.pipeline.add(videoscale)
        self.pipeline.add(caps_filter2)
        self.pipeline.add(queue1)
        self.pipeline.add(tee)
        self.pipeline.add(queue_display)
        self.pipeline.add(videoconvert_display)
        self.pipeline.add(videosink)
        # Link pipeline
        last_element.link(videoconvert)
        videoconvert.link(videoscale)
        videoscale.link(caps_filter2)
        caps_filter2.link(queue1)
        queue1.link(tee)
        # Link display branch
        tee.link(queue_display)
        queue_display.link(videoconvert_display)
        videoconvert_display.link(videosink)
        # Add probe for detection callback on queue1's src pad
        queue1_src_pad = queue1.get_static_pad("src")
        queue1_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            lambda pad, info: self.detection_callback(pad, info)
        )
        print("[PIPELINE] Pipeline created successfully")
        # Set up bus to watch for messages
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)
        return self.pipeline
    
    def on_bus_message(self, bus, message):
        """Handle GStreamer bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("[PIPELINE] End of stream")
            if self.main_loop:
                self.main_loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[PIPELINE ERROR] {err}: {debug}")
            if self.main_loop:
                self.main_loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"[PIPELINE WARNING] {err}: {debug}")
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                print(f"[PIPELINE] State changed: {old_state.value_nick} -> {new_state.value_nick}")
    
    def start_pipeline(self):
        """Start the GStreamer pipeline"""
        if not self.pipeline:
            self.create_pipeline()
        print("[PIPELINE] Starting pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[PIPELINE ERROR] Unable to set pipeline to PLAYING state")
            return False
        print("[PIPELINE] Pipeline is running")
        return True
    
    def stop_pipeline(self):
        """Stop the GStreamer pipeline"""
        if self.pipeline:
            print("[PIPELINE] Stopping pipeline...")
            self.pipeline.set_state(Gst.State.NULL)
            print("[PIPELINE] Pipeline stopped")


# Main Application Class
class CarTrackingApp:
    """Main application class that coordinates all components"""
    def __init__(self, config):
        self.config = config
        # Initialize components
        self.websocket_server = WebSocketServer(config)
        self.webrtc_server = WebRTCServer(config)
        self.http_server = HTTPServer(config, self.webrtc_server)
        self.detection_pipeline = DetectionPipeline(config, self.websocket_server)
        # GStreamer app reference
        self.gst_app = None
        
    async def setup_pipeline_callback(self, pad, info, user_data):
        """Wrapper for detection callback"""
        return self.detection_pipeline.detection_callback(pad, info)
    
    def initialize(self):
        """Initialize all components"""
        print("="*60)
        print("CAR DETECTION AND TRACKING SYSTEM")
        print("WITH IP CAMERA, WEBSOCKET, WebRTC & RTSP")
        print("="*60)
        # Print configuration
        self.config.print_config()
        # Start servers
        self.websocket_server.start()
        self.http_server.start()
        print("[INIT] Initializing Hailo detection pipeline...")
        
    async def detection_loop(self):
        """Async detection loop that processes frames"""
        print("[PIPELINE] Starting async detection loop...")
        # Start the async detection processor
        processor_task = asyncio.create_task(self.detection_pipeline.process_detections_async())
        try:
            # Keep the loop running
            while True:
                await asyncio.sleep(1)
                # Optional: Monitor queue size
                queue_size = self.detection_pipeline.detection_queue.qsize()
                if queue_size > 10:
                    print(f"[WARNING] Detection queue backlog: {queue_size} items")
        except asyncio.CancelledError:
            print("[PIPELINE] Detection loop cancelled")
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
            raise

    async def gstreamer_loop(self):
        """Run GStreamer main loop in async context"""
        print("[GSTREAMER] Starting GStreamer main loop...")
        # Start the pipeline first
        if not self.detection_pipeline.start_pipeline():
            print("[ERROR] Failed to start pipeline")
            return
        def run_gst_loop():
            """Run GLib main loop in a separate thread"""
            self.detection_pipeline.main_loop = GLib.MainLoop()
            self.detection_pipeline.main_loop.run()
        # Start GLib main loop in thread
        gst_thread = threading.Thread(target=run_gst_loop, daemon=True)
        gst_thread.start()
        try:
            # Keep monitoring the pipeline
            while True:
                await asyncio.sleep(1)
                # Check pipeline state
                if self.detection_pipeline.pipeline:
                    state = self.detection_pipeline.pipeline.get_state(0)
                    if state[0] == Gst.StateChangeReturn.FAILURE:
                        print("[ERROR] Pipeline state change failed")
                        break
        except asyncio.CancelledError:
            print("[GSTREAMER] Main loop cancelled")
            self.detection_pipeline.stop_pipeline()
            if self.detection_pipeline.main_loop:
                self.detection_pipeline.main_loop.quit()
            raise
    
    async def async_run(self):
        """Async run method for the application"""
        print("\n[READY] Starting detection pipeline...")
        print(f"[INFO] WebSocket: ws://localhost:{self.config.ws_port}")
        print(f"[INFO] Web Interface: http://localhost:{self.config.http_port}")
        print(f"[INFO] IP Camera: {self.config.ip_camera_url}")
        print("[INFO] Press Ctrl+C to stop\n")
        print("="*60)
        # Create tasks for async operations
        detection_task = asyncio.create_task(self.detection_loop())
        gstreamer_task = asyncio.create_task(self.gstreamer_loop())
        try:
            # Wait for all tasks
            await asyncio.gather(detection_task, gstreamer_task)
        except asyncio.CancelledError:
            print("\n[SHUTDOWN] Cancelling tasks...")
            detection_task.cancel()
            gstreamer_task.cancel()
            try:
                await asyncio.gather(detection_task, gstreamer_task)
            except asyncio.CancelledError:
                pass

    def run(self):
        """Run the application"""
        self.initialize()
        try:
            # Run the async event loop
            asyncio.run(self.async_run())
        except KeyboardInterrupt:
            print("\n\n[SHUTDOWN] Stopping detection...")
            print(f"[STATS] Total frames processed: {self.detection_pipeline.frame_count}")
            print(f"[STATS] Unique vehicles tracked: {len(self.detection_pipeline.detected_cars)}")
            # Cleanup
            self.webrtc_server.cleanup()
            print("\nGoodbye!")
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
            import traceback
            traceback.print_exc()
            raise

# Entry Point
if __name__ == "__main__":
    # Initialize GStreamer
    Gst.init(None)
    # Create configuration
    config = DetectionConfig()
    # Customize configuration here
    config.ip_camera_url = 'rtsp://admin:Dcuk.123456@192.168.37.99:554/stream'
    config.camera_type = 'ip'  # or 'rpicam'
    config.debug_mode = True
    config.confidence_threshold = 0.5
    # Create and run application
    app = CarTrackingApp(config)
    app.run()

"""
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