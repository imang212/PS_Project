import paho.mqtt.client as mqtt
import json

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
        print(f"\n[SERVER RECEIVED] Topic: {msg.topic}")
        for item in data:
            # Added a safer way to get confidence and formatting
            conf = item.get('confidence', 0) * 100
            print(f"  - Detected: {item['label']} ({conf:.1f}%)")
    except Exception as e:
        print(f"Error: {e}")

# Use VERSION2 to remove the DeprecationWarning
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set("videoanalyza", "phdA9ZNW1vfkXdJkhhbP")

# Enable TLS because you are using port 8883
client.tls_set() 

client.on_message = on_message
client.connect("mqtt.portabo.cz", 8883, 60)
client.subscribe("hailo/detections")

print("Waiting for Hailo detections on mqtt.portabo.cz (Port 8883)...")
client.loop_forever()