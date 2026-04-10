import json
import logging
import paho.mqtt.client as mqtt
from typing import Optional
from nicegui import ui
from Node import Node, NodeRegistry
from Schema import PipelinePayload

# Configure local logger
logger = logging.getLogger(__name__)

@NodeRegistry.register("MQTT Sink")
class MqttSink(Node):
    has_input = True
    has_output = False 

    def __init__(self):
        super().__init__()
        # Configuration Defaults
        self.broker = "broker.hivemq.com"
        self.port = 1883
        self.topic = "hailo/detections"
        self.username = ""
        self.password = ""
        
        # Runtime internals
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.last_status = "Stopped"
        self.msg_count = 0
        self.status_timer = None 

    def _start(self):
        self.last_status = "Connecting..."
        self._update_status_ui()
        
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            self.client = mqtt.Client()

        if self.username:
            self.client.username_pw_set(self.username, self.password)
            
        if self.port == 8883:
            self.client.tls_set()

        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

        try:
            self.client.connect(self.broker, int(self.port), 60)
            self.client.loop_start() 
            self.status_timer = ui.timer(0.5, self._update_status_ui)
        except Exception as e:
            self.last_status = f"Error: {str(e)}"
            self._update_status_ui()
            self.running = False

    def _stop(self):
        if self.status_timer:
            self.status_timer.cancel()
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        self.connected = False
        self.last_status = "Stopped"
        self._update_status_ui()

    def _input(self, payload: PipelinePayload):
        """Processes incoming JSON and updates the UI log."""
        if self.client and self.connected:
            # Safely serialize right before hitting the network
            json_string = payload.to_json()  # Assume data_json is already a stringified JSON payload
            self.client.publish(self.topic, json_string)
            
            try:
                # Update Message Counter
                if hasattr(self, 'msg_counter_label'):
                    self.msg_count += 1
                    self.msg_counter_label.set_text(f"Sent: {self.msg_count}")

                # Update the Collapsable Log
                if hasattr(self, 'last_input_display'):
                    # Prettify the JSON for the log
                    parsed = json.loads(json_string)
                    pretty_json = json.dumps(parsed, indent=2)
                    self.last_input_display.set_content(f"```json\n{pretty_json}\n```")
                    
            except Exception as e:
                logger.error(f"MQTT Publish/UI update failed: {e}")
        return None

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.connected = True
            self.last_status = f"Connected to {self.broker}"
            self.msg_count = 0
        else:
            self.last_status = f"Failed to connect: RC={rc}"

    def on_disconnect(self, client, userdata, flags, rc=None, properties=None):
        self.connected = False
        self.last_status = "Disconnected"

    def _update_status_ui(self):
        if hasattr(self, 'status_label'):
            self.status_label.set_text(str(self.last_status))
            color = "text-green-600" if self.connected else "text-red-500"
            self.status_label.classes(replace=f"text-xs {color}")

    def create_content(self):
        with ui.column().classes('w-full gap-2'):
            # 1. Connection settings 
            with ui.column().classes('bg-slate-50 border border-slate-200 border-l-4 border-l-orange-500 w-full p-2 shadow-sm gap-1'):
                ui.label("MQTT BROKER").classes('text-[10px] font-bold text-slate-700')
                ui.input(label="Broker Host").bind_value(self, 'broker').classes('w-full text-xs').props('dense')
                
                with ui.row().classes('w-full items-center gap-2'):
                    ui.number(label="Port", format="%d").bind_value(self, 'port').classes('w-1/4 text-xs').props('dense')
                    ui.input(label="Topic").bind_value(self, 'topic').classes('text-xs grow').props('dense')
                
                # Dynamic hints
                ui.label("Port 8883 enables SSL/TLS automatically.").classes('text-[9px] text-orange-600 italic leading-tight').bind_visibility_from(self, 'port', backward=lambda p: p == 8883)
                ui.label("Insecure connection (Standard).").classes('text-[9px] text-slate-400 italic leading-tight').bind_visibility_from(self, 'port', backward=lambda p: p != 8883)

            # 2. Authentication 
            with ui.column().classes('bg-blue-50 border border-slate-200 border-l-4 border-l-blue-500 w-full p-2 shadow-sm gap-1'):
                ui.label("CREDENTIALS").classes('text-[10px] font-bold text-blue-700')
                ui.input(label="Username").bind_value(self, 'username').classes('w-full text-xs').props('dense')
                ui.input(label="Password", password=True).bind_value(self, 'password').classes('w-full text-xs').props('dense')

            # 3. Status and Payload Monitor
            with ui.column().classes('bg-white border border-slate-200 border-l-4 border-l-slate-400 w-full p-2 shadow-sm gap-1'):
                with ui.row().classes('w-full items-center justify-between'):
                    self.status_label = ui.label("Status: Idle").classes("text-[10px] font-mono")
                    self.msg_counter_label = ui.label(f"Sent: {self.msg_count}").classes("text-[10px] font-mono")
                
                with ui.expansion('Live Payload Log', icon='receipt').classes('w-full max-w-full text-xs mt-1 border border-slate-200 overflow-hidden'):
                    self.last_input_display = ui.markdown("Waiting for data...").classes('w-full max-w-full text-[10px] bg-slate-900 text-slate-100 p-2 overflow-x-auto')

    def save(self) -> dict:
        data = super().save()
        data.update({
            "broker": self.broker, "port": self.port, "topic": self.topic,
            "username": self.username, "password": self.password
        })
        return data

    def _load_config(self, data: dict):
        self.broker = data.get("broker", "broker.hivemq.com") #
        self.port = int(data.get("port", 1883)) #
        self.topic = data.get("topic", "hailo/detections") #
        self.username = data.get("username", "") #
        self.password = data.get("password", "") #