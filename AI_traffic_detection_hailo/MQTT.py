import paho.mqtt.client as mqtt
import json
import ssl

class MQTTPublisher:
    """MQTT client for publishing detection data"""    
    def __init__(self, broker_host="mqtt.portabo.cz", broker_port=8883, topic="hailo/detections", client_id="hailo_tracker", username="videoanalyza", password="phdA9ZNW1vfkXdJkhhbP"):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.client_id = client_id
        self.username = username
        self.password = password
        self.client = None
        self.connected = False
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"[MQTT] Connected to broker at {self.broker_host}:{self.broker_port}")
        else:
            print(f"[MQTT] Connection failed with code {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            print(f"[MQTT] Unexpected disconnection")
    
    def connect(self):
        """Connect to MQTT broker with TLS/SSL"""
        try:
            self.client = mqtt.Client(client_id=self.client_id)
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect         
            # Set username and password
            self.client.username_pw_set(self.username, self.password)
            # Configure TLS/SSL for secure connection (port 8883)
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)
            print(f"[MQTT] Connecting to {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            return True
        except Exception as e:
            print(f"[MQTT] Connection error: {e}")
            return False
    
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
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            print("[MQTT] Disconnected")

