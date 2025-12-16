import asyncio
import json
import websockets
from typing import Dict

class WebRTCStreamer:
    """WebRTC streaming server using GStreamer webrtcbin"""
    def __init__(self, signaling_port="8443", signaling_server="ws://0.0.0.0:8443"):
        self.signaling_port = signaling_port
        # WebRTC components
        self.appsrc = None
        self.webrtcbin = None
        self.peer_id = None
        # signalling components
        self.signaling_ws = None
        self.signaling_server = signaling_server
        self.signaling_peers: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.loop = None
        self.signaling_thread = None

    ### Signalling server build    
    async def signaling_handler(self, websocket, path):
        """Handle WebSocket connections for built-in signaling server"""
        peer_id = None
        print(f"[Signaling] New connection from {websocket.remote_address}")    
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type')
                peer_id = data.get('peer_id')
                print(f"[Signaling] Received {msg_type} from {peer_id}")
                if msg_type == 'register':
                    self.signaling_peers[peer_id] = websocket
                    print(f"[Signaling] Peer registered: {peer_id} (Total: {len(self.signaling_peers)})")
                    await websocket.send(json.dumps({'type': 'registered', 'peer_id': peer_id}))
                elif msg_type in ['offer', 'answer', 'ice']:
                    # Forward message to all other peers
                    target_peer = data.get('target_peer')
                    if target_peer and target_peer in self.signaling_peers:
                        # Send to specific peer
                        await self.signaling_peers[target_peer].send(message)
                    else:
                        # Broadcast to all except sender
                        for pid, ws in self.signaling_peers.items():
                            if pid != peer_id and ws != websocket:
                                try:
                                    await ws.send(message)
                                except Exception as e:
                                    print(f"[Signaling] Failed to forward to {pid}: {e}")
        except websockets.exceptions.ConnectionClosed:
            print(f"[Signaling] Connection closed: {peer_id}")
        except Exception as e:
            print(f"[Signaling] Error in handler: {e}")
        finally:
            if peer_id and peer_id in self.signaling_peers:
                del self.signaling_peers[peer_id]
                print(f"[Signaling] Peer unregistered: {peer_id}")
    
    async def run_signaling_server(self):
        """Run the built-in signaling server"""
        try:
            async with websockets.serve(
                self.signaling_handler, 
                "0.0.0.0", 
                self.signaling_port
            ):
                print(f"[Signaling] Built-in signaling server started on ws://0.0.0.0:{self.signaling_port}")
                print("[Signaling] Waiting for connections...")
                await asyncio.Future()  # Run forever
        except Exception as e:
            print(f"[Signaling] Server error: {e}")
    
    def start_builtin_signaling_server(self):
        """Start the built-in signaling server in a separate thread"""
        def run_server():
            asyncio.run(self.run_signaling_server())
        
        self.signaling_thread = threading.Thread(target=run_server, daemon=True)
        self.signaling_thread.start()
        print("[Signaling] Built-in signaling server thread started")
    
    ### WebRTC pipeline
    def get_appsrc(self):
        """Returns the appsrc element for pushing frames"""
        return self.appsrc        
        
    def create_webrtc_pipeline(self, pipeline):
        """
        Creates a WebRTC streaming pipeline.
        This pipeline receives video via appsrc and streams via WebRTC.
        """
        self.webrtcbin = pipeline.get_by_name('webrtcbin')
        if not self.webrtcbin:
            print("[ERROR] Could not find webrtcbin element")
            return False
        # Connect WebRTC signals
        self.webrtcbin.connect('on-negotiation-needed', self.on_negotiation_needed)
        self.webrtcbin.connect('on-ice-candidate', self.on_ice_candidate)
        self.webrtcbin.connect('notify::connection-state', self.on_connection_state)
        self.webrtcbin.connect('notify::ice-connection-state', self.on_ice_connection_state)
        print("[WebRTC] WebRTC bin configured")
        return True
    
    def on_negotiation_needed(self, webrtcbin):
        """Handle WebRTC negotiation"""
        print("[WebRTC] Negotiation needed - creating offer")
        promise = Gst.Promise.new_with_change_func(self.on_offer_created, webrtcbin, None)
        webrtcbin.emit('create-offer', None, promise)
    
    def on_offer_created(self, promise, webrtcbin, _):
        """Handle offer creation"""
        promise.wait()
        reply = promise.get_reply()
        if not reply:
            print("[WebRTC] Failed to create offer")
            return
        offer = reply.get_value('offer')
        promise = Gst.Promise.new()
        webrtcbin.emit('set-local-description', offer, promise)
        promise.interrupt()
        # Send offer to signaling server
        print("[WebRTC] Offer created, sending to signaling server")
        self.send_sdp_offer(offer)
    
    def on_ice_candidate(self, webrtcbin, mlineindex, candidate):
        """Handle ICE candidate"""
        print(f"[WebRTC] ICE candidate {mlineindex}: {candidate}")        
        if self.signaling_ws and self.loop:
            # Schedule the coroutine in the event loop
            asyncio.run_coroutine_threadsafe(
                self.send_ice_candidate(mlineindex, candidate),
                self.loop
            )
    
    def on_connection_state(self, webrtcbin, pspec):
        """Monitor WebRTC connection state"""
        state = webrtcbin.get_property('connection-state')
        print(f"[WebRTC] Connection state: {state}")
    
    def on_ice_connection_state(self, webrtcbin, pspec):
        """Monitor ICE connection state"""
        state = webrtcbin.get_property('ice-connection-state')
        print(f"[WebRTC] ICE connection state: {state}")
    
    ## signalling methods
    def send_sdp_offer(self, offer):
        """Send SDP offer to signaling server or peers"""
        sdp = offer.sdp.as_text()
        print(f"[WebRTC] Sending SDP offer")
        # Send to external signaling server
        if self.signaling_ws and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.send_offer(sdp),
                self.loop
            )
    
    async def send_offer(self, sdp):
        """Send offer to external signaling server"""
        if self.signaling_ws:
            try:
                await self.signaling_ws.send(json.dumps({
                    'type': 'offer',
                    'sdp': sdp,
                    'peer_id': self.peer_id
                }))
            except Exception as e:
                print(f"[WebRTC] Failed to send offer: {e}")

    async def send_ice_candidate(self, mlineindex, candidate):
        """Send ICE candidate to signaling server"""
        if self.signaling_ws:
            try:
                await self.signaling_ws.send(json.dumps({
                    'type': 'ice',
                    'candidate': candidate,
                    'sdpMLineIndex': mlineindex,
                    'peer_id': self.peer_id
                }))
            except Exception as e:
                print(f"[WebRTC] Failed to send ICE candidate: {e}")
    
    def set_remote_description(self, sdp_type, sdp_text):
        """Set remote SDP description (answer from peer)"""
        print(f"[WebRTC] Setting remote description: {sdp_type}")
        ret, sdp_msg = GstSdp.SDPMessage.new_from_text(sdp_text)
        if ret != GstSdp.SDPResult.OK:
            print("[WebRTC] Failed to parse SDP")
            return False
        if sdp_type == 'answer':
            answer = GstWebRTC.WebRTCSessionDescription.new( GstWebRTC.WebRTCSDPType.ANSWER, sdp_msg)
            promise = Gst.Promise.new()
            self.webrtcbin.emit('set-remote-description', answer, promise)
            promise.interrupt()
            print("[WebRTC] Remote description set successfully")
            return True    
        return False
    
    def add_ice_candidate(self, mlineindex, candidate):
        """Add ICE candidate from remote peer"""
        print(f"[WebRTC] Adding remote ICE candidate: {mlineindex}")
        self.webrtcbin.emit('add-ice-candidate', mlineindex, candidate)
    
    async def connect_signaling(self, peer_id="hailo-stream"):
        """Connect to WebRTC signaling server"""
        self.peer_id = peer_id
        self.loop = asyncio.get_event_loop()
        try:
            print(f"[WebRTC] Connecting to signaling server: {self.signaling_server}")
            async with websockets.connect(self.signaling_server) as ws:
                self.signaling_ws = ws
                # Register with signaling server
                await ws.send(json.dumps({
                    'type': 'register',
                    'peer_id': self.peer_id
                }))
                print("[WebRTC] Connected to signaling server")
                # Handle incoming messages
                async for message in ws:
                    await self.handle_signaling_message(message)
        except Exception as e:
            print(f"[WebRTC] Signaling connection error: {e}")
        finally:
            self.signaling_ws = None
    
    async def handle_signaling_message(self, message):
        """Handle messages from signaling server"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            print(f"[WebRTC] Received signaling message: {msg_type}")
            if msg_type == 'answer':
                self.set_remote_description('answer', data['sdp'])
            elif msg_type == 'ice':
                self.add_ice_candidate(data['sdpMLineIndex'], data['candidate'])
            elif msg_type == 'error':
                print(f"[WebRTC] Signaling error: {data.get('message')}")
        except Exception as e:
            print(f"[WebRTC] Error handling signaling message: {e}")
    
    def start_signaling(self, peer_id="hailo-stream"):
        """Start signaling in a separate thread"""
        import threading
        def run_signaling():
            asyncio.run(self.connect_signaling(peer_id))
        self.signaling_thread = threading.Thread(target=run_signaling, daemon=True)
        self.signaling_thread.start()
        print("[WebRTC] Signaling thread started")
    
    def start(self, peer_id="hailo-stream"):
        """
        Start WebRTC streaming with signaling
        Uses built-in signaling server by default, or connects to external server if configured
        """
        # Create event loop for async operations
        if self.loop is None:
            def setup_loop():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()
            loop_thread = threading.Thread(target=setup_loop, daemon=True)
            loop_thread.start()
            # Wait for loop to be ready
            import time
            timeout = 5
            start_time = time.time()
            while self.loop is None and (time.time() - start_time) < timeout:
                time.sleep(0.1)
        if self.use_builtin_signaling:
            print("[WebRTC] Starting with built-in signaling server")
            self.start_builtin_signaling_server()
        else:
            print("[WebRTC] Starting with external signaling server")
            self.start_signaling_client(peer_id)
