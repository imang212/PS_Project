"""
WebSocket Test Client for Raspberry Pi Detection Server
Tests connection, message reception, and reconnection handling
"""
import asyncio
import websockets
import json
import time
from datetime import datetime
from typing import Optional

class WebSocketTestClient:
    """Test client for WebSocket server validation""" 
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.uri = f"ws://{host}:{port}"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.message_count = 0
        self.start_time = None
        
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            print(f"[TEST] Connecting to {self.uri}...")
            self.ws = await websockets.connect(self.uri)
            print(f"[TEST] Connected successfully!")
            self.start_time = time.time()
            return True
        except Exception as e:
            print(f"[TEST] Connection failed: {e}")
            return False
    
    async def listen(self, duration: int = 30):
        """
        Listen for messages from server
        Args:
            duration: How long to listen (seconds)
        """
        if not self.ws:
            print("[TEST] Not connected!")
            return
        print(f"[TEST] Listening for {duration} seconds...")
        print("-" * 60)
        try:
            end_time = time.time() + duration
            while time.time() < end_time:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                    self.message_count += 1
                    data = json.loads(message)
                    # Display message info
                    print(f"\n[MESSAGE #{self.message_count}]")
                    print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
                    print(f"  Frame count: {data.get('frame_count', 'N/A')}")
                    # Show first 3 detections
                    detections = data.get('detections', [])
                    print(f"  Detections count: {len(detections)}")
                    for i, det in enumerate(detections[:3]):
                        print(f"  Detection {i+1}:")
                        print(f"    - Track ID: {det.get('tracking_id')}")
                        print(f"    - Label: {det.get('class')}")
                        print(f"    - Confidence: {det.get('confidence', 0):.2f}")
                        print(f"    - Timestamp: {det.get('timestamp', 0)}")
                        print(f"    - BBox: ({det.get('bbox')}")
                    if len(detections) > 3:
                        print(f"  ... and {len(detections) - 3} more")
                except asyncio.TimeoutError:
                    # No message received in 1 second, continue
                    continue
        except websockets.exceptions.ConnectionClosed:
            print("\n[TEST] Connection closed by server")
        except KeyboardInterrupt:
            print("\n[TEST] Interrupted by user")
        finally:
            await self.disconnect()
    
    async def send_test_message(self, message: str = "ping"):
        """Send a test message to server"""
        if not self.ws:
            print("[TEST] Not connected!")
            return
        try:
            await self.ws.send(message)
            print(f"[TEST] Sent message: {message}")
        except Exception as e:
            print(f"[TEST] Failed to send: {e}")
    
    async def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
            elapsed = time.time() - self.start_time if self.start_time else 0
            print("\n" + "=" * 60)
            print("[TEST] Connection closed")
            print(f"  Total messages received: {self.message_count}")
            print(f"  Duration: {elapsed:.1f} seconds")
            if elapsed > 0 and self.message_count > 0:
                print(f"  Message rate: {self.message_count/elapsed:.2f} msg/sec")
            print("=" * 60)

async def test_connection_only(host: str, port: int):
    """Quick connection test"""
    print("\n=== CONNECTION TEST ===")
    client = WebSocketTestClient(host, port)
    if await client.connect():
        print("[TEST] Connection test passed!")
        await client.disconnect()
        return True
    else:
        print("[TEST] Connection test failed!")
        return False

async def test_multiple_clients(host: str, port: int, num_clients: int = 3):
    """Test multiple simultaneous connections"""
    print(f"\n=== MULTIPLE CLIENTS TEST ({num_clients} clients) ===")
    clients = []
    # Connect all clients
    for i in range(num_clients):
        client = WebSocketTestClient(host, port)
        if await client.connect():
            clients.append(client)
            await asyncio.sleep(0.5)  # Stagger connections
    print(f"[TEST] Connected {len(clients)}/{num_clients} clients")
    # Listen briefly
    await asyncio.sleep(5)
    # Disconnect all
    for client in clients:
        await client.disconnect()

async def test_reconnection(host: str, port: int):
    """Test reconnection after disconnect"""
    print("\n=== RECONNECTION TEST ===")
    client = WebSocketTestClient(host, port)
    # First connection
    if not await client.connect():
        print("[TEST] ✗ Initial connection failed!")
        return
    await asyncio.sleep(2)
    await client.disconnect()
    # Wait and reconnect
    print("[TEST] Waiting 3 seconds before reconnecting...")
    await asyncio.sleep(3)
    if await client.connect():
        print("[TEST] Reconnection successful!")
        await asyncio.sleep(2)
        await client.disconnect()
    else:
        print("[TEST] Reconnection failed!")

async def main():
    """Run all tests"""
    # Configuration
    HOST = "192.168.37.205"  # Change to Raspberry Pi IP for remote testing
    PORT = 8765
    print("=" * 60)
    print("WebSocket Server Test Suite")
    ##Test 1: Basic connection
    #if not await test_connection_only(HOST, PORT):
    #    print("\n[TEST] Server not running or unreachable. Exiting.")
    #    return
    #await asyncio.sleep(1)
    # Test 2: Listen for messages
    print("\n=== MESSAGE RECEPTION TEST ===")
    client = WebSocketTestClient(HOST, PORT)
    if await client.connect():
        await client.listen(duration=60)  # Listen for 120 seconds
    await asyncio.sleep(1)
    #Test 3: Multiple clients (optional)
    await test_multiple_clients(HOST, PORT, num_clients=3)
    # Test 4: Reconnection (optional)
    await test_reconnection(HOST, PORT)

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\n[TEST] Tests interrupted by user")

