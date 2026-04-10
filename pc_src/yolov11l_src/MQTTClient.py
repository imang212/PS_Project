import paho.mqtt.client as mqtt
import json
import ssl
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading
import time

class MQTTPublisher:
    """MQTT client for publishing detection data"""    
    def __init__(self, broker_host="mqtt.portabo.cz", broker_port=8883, topic="patrik/traffic_detection", client_id="hailo_tracker", username="videoanalyza", password="phdA9ZNW1vfkXdJkhhbP", debug=False):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.client_id = client_id
        self.username = username
        self.password = password
        # Setup MQTT client
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.username_pw_set(self.username, self.password)
        # Setup SSL/TLS
        self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)
        self.client.tls_insecure_set(False)
        # Setup callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        
        self.connected = False
        self.lock = threading.Lock()
        self.debug = False

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            client.subscribe(self.topic)
            print(f"[MQTT] Connected to broker at {self.broker_host}:{self.broker_port}")
        else:
            print(f"[MQTT] Connection failed with code {rc}")
            self.connected = False
    
    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            print(f"[MQTT] Unexpected disconnection")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            #timestamp = datetime.now().isoformat()        
            #detection_data = {
            #    "timestamp": timestamp,
            #    "topic": msg.topic,
            #    "data": payload
            #}
            if userdata and "db" in userdata:
                userdata["db"]._process_mqtt_message(msg)
            with self.lock:
                if self.debug:
                    print(f"MQTT detections received: {payload['detections'] if 'detections' in payload else 'No detections'}")
            #print(f"Received detection: {detection_type} at {timestamp}") 
        except json.JSONDecodeError:
            print(f"Failed to decode message: {msg.payload}")
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def connect(self, userdata: Optional[Dict] = None):
        """Connect to the MQTT broker"""
        try:
            if userdata:
                self.client.user_data_set(userdata)
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            # Wait for connection
            timeout = 10
            start = time.time()
            while not self.connected and (time.time() - start) < timeout:
                time.sleep(0.1)
            if self.connected:
                print("Successfully connected and subscribed")
            else:
                print("Connection timeout")
        except Exception as e:
            print(f"Connection error: {e}")
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print("[MQTT] Disconnected")

    def publish(self, data_dict):
        """Publish detection data to MQTT topic"""
        if self.connected and self.client:
            try:
                payload = json.dumps(data_dict)
                result = self.client.publish(self.topic, payload, qos=1)
                if result.rc != mqtt.MQTT_ERR_SUCCESS:
                    print(f"[MQTT] Publish failed: {result.rc}")
            except Exception as e:
                print(f"[MQTT] Publish error: {e}")
    
    def publish_bulk(self, detections: List[Dict], delay: float = 0.1, batch_size: Optional[int] = None) -> Dict:
        """
        Publish bulk detection data to the MQTT topic        
        Args:
            detections: List of detection dictionaries to publish
            delay: Delay in seconds between each message (default 0.1s to avoid flooding)
            batch_size: If specified, sends detections in batches with this size
        Returns:
            Dictionary with summary of the bulk publish operation:
            - total: Total number of detections to send
            - sent: Number successfully sent
            - failed: Number that failed
            - duration: Time taken in seconds
        """
        if not self.connected:
            print("Not connected. Cannot publish bulk messages.")
            return {"total": len(detections), "sent": 0, "failed": len(detections), "duration": 0, "error": "Not connected" } 
        start_time = time.time()
        sent_count = 0
        failed_count = 0
        try:
            if batch_size:
                # Send as batches
                print(f"Publishing {len(detections)} detections in batches of {batch_size}...")
                for i in range(0, len(detections), batch_size):
                    batch = detections[i:i + batch_size]
                    batch_message = { "type": "bulk_detections", "batch_number": i // batch_size + 1, "detections": batch, "timestamp": datetime.now().isoformat() }
                    try:
                        payload = json.dumps(batch_message)
                        result = self.client.publish(self.topic, payload, qos=1)   
                        if result.rc == mqtt.MQTT_ERR_SUCCESS:
                            sent_count += len(batch)
                            print(f"Batch {i // batch_size + 1} sent ({len(batch)} detections)")
                        else:
                            failed_count += len(batch)
                            print(f"Batch {i // batch_size + 1} failed with code {result.rc}")
                        time.sleep(delay)
                    except Exception as e:
                        failed_count += len(batch)
                        print(f"Error sending batch {i // batch_size + 1}: {e}")
            else:
                # Send individually
                print(f"Publishing {len(detections)} detections individually...")
                for idx, detection in enumerate(detections):
                    try:
                        # Add metadata if not present
                        if "timestamp" not in detection:
                            detection["timestamp"] = datetime.now().isoformat()
                        payload = json.dumps(detection)
                        result = self.client.publish(self.topic, payload, qos=1)
                        if result.rc == mqtt.MQTT_ERR_SUCCESS:
                            sent_count += 1
                            if (idx + 1) % 10 == 0:
                                print(f"Progress: {idx + 1}/{len(detections)} sent")
                        else:
                            failed_count += 1
                            print(f"Failed to send detection {idx + 1} with code {result.rc}")
                        time.sleep(delay)
                    except Exception as e:
                        failed_count += 1
                        print(f"Error sending detection {idx + 1}: {e}")
        except Exception as e:
            print(f"Bulk publish error: {e}")
        duration = time.time() - start_time
        summary = { "total": len(detections), "sent": sent_count, "failed": failed_count, "duration": duration, "rate": sent_count / duration if duration > 0 else 0 }
        print(f"\n--- Bulk Publish Summary ---")
        print(f"Total: {summary['total']}")
        print(f"Sent: {summary['sent']}")
        print(f"Failed: {summary['failed']}")
        print(f"Duration: {summary['duration']:.2f}s")
        print(f"Rate: {summary['rate']:.2f} msg/s")
        return summary

# Test
if __name__ == "__main__":
    # Create and connect client
    client = MQTTPublisher(
        broker_host="mqtt.portabo.cz", 
        broker_port=8883, 
        topic="/videoanalyza", 
        client_id="traffic_detection_client_test", 
        username="videoanalyza", 
        password="phdA9ZNW1vfkXdJkhhbP"
    )
    client.connect()
    try:
        # Keep running and receiving messages
        #bulk_data = [
        #    {"type": "person", "confidence": 0.95, "location": "zone_1"},
        #    {"type": "vehicle", "confidence": 0.88, "location": "zone_2"},
        #    {"type": "person", "confidence": 0.92, "location": "zone_3"},
        #]
        ## Send individually with 0.1s delay
        #summary = client.publish_bulk(bulk_data, delay=0.1)
        print("\nListening for detections... (Press Ctrl+C to stop)")
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.disconnect()