import json
import paho.mqtt.client as mqtt
from typing import Optional
from nicegui import ui
from Node import Node, NodeRegistry
from Schema import PipelinePayload

@NodeRegistry.register("MQTT Source")
class MQTTInputNode(Node):
    has_input = False  
    has_output = True

    def __init__(self):
        super().__init__()
        self.broker = "broker.hivemq.com"
        self.port = 1883
        self.topic = "hailo/detections"
        self.username = ""
        self.password = ""
        
        self.client: Optional[mqtt.Client] = None

    def _on_message(self, client, userdata, msg):
        try:
            payload_str = msg.payload.decode()
            # Parse the string into your strongly-typed objects
            payload = PipelinePayload.from_json(payload_str) 
            self.notify(payload)
        except Exception as e:
            print(f"MQTT Source Error: {e}")

    def _start(self):
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            self.client = mqtt.Client()

        if self.username:
            self.client.username_pw_set(self.username, self.password)
        
        # SECURE TOGGLE LOGIC: Port 8883 triggers TLS
        if self.port == 8883:
            self.client.tls_set()
            print(f"Connecting securely to {self.broker}:{self.port}")

        self.client.on_message = self._on_message
        self.client.connect(self.broker, self.port, 60)
        self.client.subscribe(self.topic)
        self.client.loop_start()

    def _stop(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None

    def create_content(self):
        with ui.column().classes('w-full gap-2'):
            # 1. Connection settings
            with ui.column().classes('bg-slate-50 border border-slate-200 border-l-4 border-l-emerald-500 w-full p-2 shadow-sm gap-1'):
                ui.label("MQTT BROKER").classes('text-[10px] font-bold text-slate-700')
                ui.input(label="Broker Host").bind_value(self, 'broker').classes('w-full text-xs').props('dense')
                
                with ui.row().classes('w-full items-center gap-2'):
                    ui.number(label="Port", format="%d").bind_value(self, 'port').classes('w-1/4 text-xs').props('dense')
                    # 'grow' ensures the topic input takes the remaining space neatly
                    ui.input(label="Topic").bind_value(self, 'topic').classes('text-xs grow').props('dense')
                
                # Dynamic hints
                ui.label("Port 8883 enables SSL/TLS automatically.").classes('text-[9px] text-emerald-600 italic leading-tight').bind_visibility_from(self, 'port', backward=lambda p: p == 8883)
                ui.label("Insecure connection (Standard).").classes('text-[9px] text-slate-400 italic leading-tight').bind_visibility_from(self, 'port', backward=lambda p: p != 8883)

            # 2. Authentication
            with ui.column().classes('bg-blue-50 border border-slate-200 border-l-4 border-l-blue-500 w-full p-2 shadow-sm gap-1'):
                ui.label("CREDENTIALS").classes('text-[10px] font-bold text-blue-700')
                ui.input(label="Username").bind_value(self, 'username').classes('w-full text-xs').props('dense')
                ui.input(label="Password", password=True).bind_value(self, 'password').classes('w-full text-xs').props('dense')

    def save(self) -> dict:
        base = super().save()
        base.update({
            "broker": self.broker,
            "port": self.port,
            "topic": self.topic,
            "username": self.username,
            "password": self.password
        })
        return base

    def _load_config(self, data: dict):
        self.broker = data.get("broker", "broker.hivemq.com")
        self.port = data.get("port", 1883)
        self.topic = data.get("topic", "hailo/detections")
        self.username = data.get("username", "")
        self.password = data.get("password", "")

    def _input(self, payload: PipelinePayload):
        return None
