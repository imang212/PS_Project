import json
import logging
from typing import Optional
from nicegui import ui
from Node import Node, NodeRegistry
from Schema import PipelinePayload

logger = logging.getLogger(__name__)

@NodeRegistry.register("Logger")
class LoggerNode(Node):
    node_color = "slate-600"
    has_input = True
    has_output = True 

    def __init__(self):
        super().__init__()
        self.log_format = "Summary" # Options: "Summary" or "Full JSON"
        self.log_count = 0
        
    def _start(self): 
        self.log_count = 0
        logger.info(f"[{self._node_type_name}] Started Logging Session")
        
    def _stop(self): 
        logger.info(f"[{self._node_type_name}] Stopped Logging Session. Total recorded: {self.log_count}")

    def _input(self, payload):
        if not payload:
            return None
            
        self.log_count += 1
        
        # Parse the JSON string representation for clean logging
        if hasattr(payload, 'to_json'):
            payload_dict = json.loads(payload.to_json())
        elif isinstance(payload, str):
            payload_dict = json.loads(payload)
        else:
            payload_dict = payload
            
        if self.log_format == "Summary":
            # Build a readable summary string summarizing ALL attached sensors
            timestamp = payload_dict.get('timestamp', 0.0)
            summary = f"[Payload {self.log_count}] Time: {timestamp:.2f} | "
            
            modules = payload_dict.get('modules', {})
            if not modules:
                summary += "No sensor data."
            else:
                for mod_name, mod_data in modules.items():
                    data_dict = mod_data.get('data', {})
                    count = data_dict.get('count', '?')
                    summary += f"{mod_name}: {count} detections "
                    
            print(summary)
            
        else:
            # Dump the entire nested dictionary structure
            print(f"--- [Payload {self.log_count}] ---\n{json.dumps(payload_dict, indent=2)}\n")
            
        # Update the UI counter dynamically
        if hasattr(self, 'count_label'):
            self.count_label.set_text(f"Logged: {self.log_count} payloads")

        # Crucial: Return the payload untouched so it can continue down the tree!
        return payload
        
    def create_content(self):
        with ui.column().classes('bg-slate-50 border border-slate-200 border-l-4 border-l-slate-600 w-full p-2 mb-2 shadow-sm gap-1'):
            ui.label("LOGGER SETTINGS").classes('text-[10px] font-bold text-slate-800')
            
            ui.radio(['Summary', 'Full JSON'], value=self.log_format) \
                .bind_value(self, 'log_format').props('inline dense size=sm')
                
            self.count_label = ui.label("Logged: 0 payloads") \
                .classes("text-[10px] text-slate-500 font-mono mt-1")

    def save(self) -> dict:
        base = super().save()
        base.update({
            "log_format": self.log_format
        })
        return base

    def _load_config(self, data: dict):
        self.log_format = data.get("log_format", "Summary")