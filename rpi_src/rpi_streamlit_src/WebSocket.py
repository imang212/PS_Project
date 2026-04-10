import asyncio
import websockets
import json
from typing import Dict, Any
from datetime import datetime

class WebSocketServer:
    """
    WebSocket server for real-time detection streaming.
    Sends detected object data to clients.
    """    
    def __init__(self, host: str, port: int):
        """
        Initialize WebSocket server. 
        Args:
            host: Server IP address
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        
    async def register_client(self, websocket):
        """
        Register a new client.
        Args:
            websocket: WebSocket connection object
        """
        self.clients.add(websocket)
        print(f"[WEBSOCKET] New client connected. Total clients: {len(self.clients)}")
        
    async def unregister_client(self, websocket):
        """Unregister a client"""
        self.clients.remove(websocket)
        print(f"[WEBSOCKET] Client disconnected. Total clients: {len(self.clients)}")
        
    async def broadcast_detections(self, detections_data: Dict[str, Any]):
        """
        Send detections to all connected clients.
        Args:
            detections_data: Dictionary with detection data
        Data format:
        {
            'timestamp': ISO timestamp,
            'frame_id': Frame ID,
            'detections': [
                {
                    'track_id': Tracked object ID,
                    'label': Class name,
                    'confidence': Confidence score (0-1),
                    'x1', 'y1', 'x2', 'y2': Bounding box coordinates
                }
            ],
            'count': Total number of detected objects
        }
        """
        if not self.clients: return
        # Serialize to JSON
        message = json.dumps(detections_data)
        print(message)
        # Broadcast to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except Exception as e:
                print(f"[WEBSOCKET] Error sending to client: {e}")
                disconnected.add(client)
        # Remove disconnected clients
        for client in disconnected:
            await self.unregister_client(client)
    
    async def handler(self, websocket):
        """
        Handler for WebSocket connections.
        Args:
            websocket: WebSocket connection
            path: URL path (endpoint)
        """
        await self.register_client(websocket)
        try:
            # Keep connection alive
            async for message in websocket:
                # Can receive configuration messages from clients
                pass
        except Exception as e:
            print(f"[WEBSOCKET] Handler error: {e}")
        finally:
            await self.unregister_client(websocket)
        
    async def start(self):
        """Start WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.handler,
                self.host,
                self.port
            )
            print(f"[WEBSOCKET] Server started at ws://{self.host}:{self.port}")
        except OSError as e:
            print(f"[WEBSOCKET] Failed to start server: {e}")
            raise
        except Exception as e:
            print(f"[WEBSOCKET] Unexpected error: {e}")
            raise

    async def start_server(self):
        """
        Starts the WebSocket server
        """
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    async def stop(self):
        """Stop WebSocket server"""
        if self.server:
            print(f"[WEBSOCKET] Shutting down server...")
            self.server.close()
            await self.server.wait_closed()
            print(f"[WEBSOCKET] Server stopped")

    def run(self):
        """
        Runs the WebSocket server in an asyncio event loop
        """
        asyncio.run(self.start_server())

class WebSocketServerWithDetectionData:
    """
    WebSocket server to sending real-time detection data to clients from class with detection data
    """
    def __init__(self, detection_data, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.detection_data = detection_data
        self.clients = set()
        self.server = None
        
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

class WebSocketServerWithDebug:
    """
    WebSocket server for real-time detection streaming for version 15.0.1.
    Sends detected object data to clients.
    """    
    def __init__(self, host: str, port: int, debug: bool = True):
        """
        Initialize WebSocket server. 
        Args:
            host: Server IP address
            port: Server port
            debug: Enable detailed debug logging
        """
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        # debug
        self.debug = debug
        # message counting
        self.message_count = 0
        self.total_bytes_sent = 0
        
    async def register_client(self, websocket):
        """
        Register a new client.
        Args:
            websocket: WebSocket connection object
        """
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"[WEBSOCKET]   New client connected: {client_info}")
        print(f"[WEBSOCKET]   Total clients: {len(self.clients)}")    
        if self.debug:
            print(f"[WEBSOCKET]   Active connections: {[f'{c.remote_address[0]}:{c.remote_address[1]}' for c in self.clients]}")
    
    async def unregister_client(self, websocket):
        """Unregister a client"""
        try: client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        except: client_info = "unknown"
        # remove client
        self.clients.remove(websocket)
        print(f"[WEBSOCKET] Client disconnected: {client_info}")
        print(f"[WEBSOCKET] Remaining clients: {len(self.clients)}")
    
    async def broadcast_detections(self, detections_data: Dict[str, Any]):
        """
        Send detections to all connected clients.
        Args:
            detections_data: Dictionary with detection data
        Data format:
        {
            'timestamp': ISO timestamp,
            'frame_id': Frame ID,
            'detections': [
                {
                    'track_id': Tracked object ID,
                    'label': Class name,
                    'confidence': Confidence score (0-1),
                    'x1', 'y1', 'x2', 'y2': Bounding box coordinates
                }
            ],
            'count': Total number of detected objects
        }
        """
        # Check if are clients
        if not self.clients:
            if self.debug and self.message_count % 100 == 0:  # Print every 100 frames
                print(f"[WEBSOCKET] No clients connected (frame {detections_data.get('frame_id', '?')})")
            return
        # Serialize to JSON
        try:
            message = json.dumps(detections_data)
            message_size = len(message.encode('utf-8'))
            if self.debug and self.message_count % 50 == 0:  # Print every 50 messages
                print(f"[WEBSOCKET] Broadcasting message #{self.message_count}")
                print(f"[WEBSOCKET]   Frame: {detections_data.get('frame_id', '?')}")
                print(f"[WEBSOCKET]   Detections: {detections_data.get('count', 0)}")
                print(f"[WEBSOCKET]   Size: {message_size} bytes")
                print(f"[WEBSOCKET]   Recipients: {len(self.clients)}")
        except Exception as e:
            print(f"[WEBSOCKET] JSON serialization error: {e}")
            print(f"[WEBSOCKET] Data: {detections_data}")
            return
        # Broadcast to all clients
        disconnected = set()
        success_count = 0
        for client in self.clients:
            try:
                await client.send(message)
                if self.debug:
                    success_count += 1
                    self.total_bytes_sent += message_size
            except websockets.exceptions.ConnectionClosed as e:
                if self.debug: print(f"[WEBSOCKET] Client connection closed during send: {e}")
                disconnected.add(client)
            except Exception as e:
                print(f"[WEBSOCKET] Error sending to client: {e}")
                disconnected.add(client)
        # Update counter
        if self.debug:
            self.message_count += 1
        if self.debug and disconnected:
            print(f"[WEBSOCKET] Cleaning up {len(disconnected)} disconnected clients")
        # Remove disconnected clients
        for client in disconnected:
            await self.unregister_client(client)
        # Log successful broadcast
        if self.debug and success_count > 0 and self.message_count % 50 == 0:
            print(f"[WEBSOCKET] Broadcast successful to {success_count}/{len(self.clients) + len(disconnected)} clients")
    
    async def handler(self, websocket):
        """
        Handler for WebSocket connections.
        Compatible with websockets v13+ (no path argument)
        Args:
            websocket: WebSocket connection
        """
        if self.debug:
            try:
                print(f"[WEBSOCKET] New connection attempt from {websocket.remote_address[0]}")
            except:
                print(f"[WEBSOCKET] New connection attempt")
        # Register client    
        await self.register_client(websocket)
        try:
            # Keep connection alive and listen for messages
            async for message in websocket:
                # Can receive configuration messages from clients
                if self.debug:
                    print(f"[WEBSOCKET] Received message from client: {message[:100]}")
                pass
        except websockets.exceptions.ConnectionClosed as e:
            if self.debug: print(f"[WEBSOCKET] Connection closed: {e.code} - {e.reason}")
        except Exception as e:
            print(f"[WEBSOCKET] ✗ Handler error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.unregister_client(websocket)
    
    async def start(self):
        """Start WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.handler,
                self.host,
                self.port,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10    # Wait 10 seconds for pong
            )
            print(f"[WEBSOCKET] Server started at ws://{self.host}:{self.port}")
            print(f"[WEBSOCKET]   Debug mode: {'ON' if self.debug else 'OFF'}")
            print(f"[WEBSOCKET]   Waiting for clients...")
        except OSError as e:
            print(f"[WEBSOCKET] Failed to start server: {e}")
            print(f"[WEBSOCKET] Check if port {self.port} is already in use")
            raise
        except Exception as e:
            print(f"[WEBSOCKET] Unexpected error: {e}")
            raise

    async def stop(self):
        """Stop WebSocket server"""
        if self.server:
            print(f"[WEBSOCKET] Shutting down server...")
            print(f"[WEBSOCKET]   Messages sent: {self.message_count}")
            print(f"[WEBSOCKET]   Total data: {self.total_bytes_sent / 1024 / 1024:.2f} MB")
            self.server.close()
            await self.server.wait_closed()
            print(f"[WEBSOCKET] Server stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        if self.debug:
            return {
                'clients_connected': len(self.clients),
                'messages_sent': self.message_count,
                'total_bytes_sent': self.total_bytes_sent,
                'total_mb_sent': round(self.total_bytes_sent / 1024 / 1024, 2)
            }
        else:
            return "No debugging on."

async def test_server():
    """Test the WebSocket server with mock data"""
    server = WebSocketServerWithDebug("192.168.37.205", 8765, debug=True)
    await server.start()
    print("\n[TEST] Server is running. Sending test detections...")    
    # Simulate sending detection data
    frame_id = 0
    try:
        while True:
            await asyncio.sleep(3)
            test_data = {
                'timestamp': datetime.now().isoformat(),
                'frame_id': frame_id,
                'detections': [
                    {
                        'track_id': 1,
                        'label': 'person',
                        'confidence': 0.95,
                        'x1': 100, 'y1': 100,
                        'x2': 200, 'y2': 300
                    },
                    {
                        'track_id': 2,
                        'label': 'car',
                        'confidence': 0.87,
                        'x1': 300, 'y1': 150,
                        'x2': 450, 'y2': 250
                    }
                ],
                'count': 2
            }
            await server.broadcast_detections(test_data)
            if frame_id % 10 == 0:
                print(f"[TEST] Sent test frame {frame_id}")
                stats = server.get_stats()
                print(f"[TEST] Stats: {stats['clients_connected']} clients, {stats['messages_sent']} messages")
            frame_id += 1
    except KeyboardInterrupt:
        print("\n[TEST] Stopping server...")
        await server.stop()
    
if __name__ == "__main__":
    try: asyncio.run(test_server())
    except KeyboardInterrupt: print("\n[TEST] Server stopped by user")