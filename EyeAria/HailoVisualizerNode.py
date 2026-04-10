import cv2
import base64
import numpy as np
from nicegui import ui
from Node import Node, NodeRegistry
from Schema import PipelinePayload

@NodeRegistry.register("Visualizer")
class VisualizerNode(Node):
    node_color = "amber-600"
    has_input = True
    has_output = True 

    def __init__(self):
        super().__init__()
        self.target_module = None
        self.available_modules = []
        
        # We increase the default node width to comfortably fit a video feed
        self.width = 340
        self.height = 400
        
        # Runtime internals
        self.is_rendering = False

    def _start(self):
        self.is_rendering = True
        if hasattr(self, 'status_label'):
            self.status_label.set_text("Waiting for video feed...")

    def _stop(self):
        self.is_rendering = False
        if hasattr(self, 'status_label'):
            self.status_label.set_text("Stopped")

    def _input(self, payload: PipelinePayload):
        if not self.is_rendering or not payload:
            return payload

        # 1. Dynamically update the dropdown with available camera streams
        current_hailo_keys = [k for k in payload.modules.keys() if k.startswith("Hailo")]
        
        if current_hailo_keys != self.available_modules:
            self.available_modules = current_hailo_keys
            if hasattr(self, 'stream_select'):
                self.stream_select.options = self.available_modules
                self.stream_select.update()
                
            # Auto-select the first available camera if none is selected
            if not self.target_module and self.available_modules:
                self.target_module = self.available_modules[0]
                if hasattr(self, 'stream_select'):
                    self.stream_select.value = self.target_module

        # 2. Extract the target camera's data
        if self.target_module and self.target_module in payload.modules:
            mod_data = payload.modules[self.target_module]
            frame = mod_data.data.get('frame')
            detections = mod_data.data.get('detections', [])
            
            if frame is not None:
                # 3. Draw bounding boxes on the frame using OpenCV
                annotated_frame = self._draw_detections(frame, detections)
                
                # 4. Convert the OpenCV frame (BGR) to a Base64 JPEG for NiceGUI
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                b64_str = base64.b64encode(buffer).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{b64_str}"
                
                # 5. Push the image to the UI component
                if hasattr(self, 'video_player'):
                    self.video_player.set_source(image_data)
                    
                if hasattr(self, 'status_label'):
                    self.status_label.set_text(f"Rendering: {len(detections)} objects")

        # Pass the payload downstream completely untouched
        return payload

    def _draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Helper method to draw Hailo bounding boxes on the raw frame."""
        # Create a copy so we don't permanently paint boxes onto the raw frame 
        # that other downstream nodes might want to use cleanly.
        img = frame.copy() 
        h, w = img.shape[:2]

        for det in detections:
            # Handle both dictionary (network) and object (local) formats
            bbox = det.bbox if hasattr(det, 'bbox') else det.get('bbox', [0,0,0,0])
            label = det.label if hasattr(det, 'label') else det.get('label', 'Unknown')
            conf = det.confidence if hasattr(det, 'confidence') else det.get('confidence', 0.0)
            track_id = det.track_id if hasattr(det, 'track_id') else det.get('track_id', -1)

            # Hailo bounding boxes are normalized [0.0 to 1.0]. Convert to absolute pixels.
            xmin = int(bbox[0] * w)
            ymin = int(bbox[1] * h)
            xmax = int(bbox[2] * w)
            ymax = int(bbox[3] * h)

            # Pick a color (Green for tracked, Orange for untracked)
            color = (0, 255, 0) if track_id != -1 else (0, 165, 255) # BGR format

            # Draw Rectangle
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw Label Background
            text = f"{label} {conf:.2f}"
            if track_id != -1:
                text += f" [ID:{track_id}]"
                
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (xmin, ymin - text_h - 4), (xmin + text_w, ymin), color, -1)

            # Draw Label Text
            cv2.putText(img, text, (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return img

    def create_content(self):
        with ui.column().classes('bg-slate-50 border border-slate-200 border-l-4 border-l-amber-500 w-full p-2 mb-2 shadow-sm gap-1'):
            ui.label("DISPLAY CONFIG").classes('text-[10px] font-bold text-slate-800')
            
            # Dropdown to select which camera to view
            self.stream_select = ui.select(
                options=self.available_modules, 
                label="Target Camera Stream"
            ).bind_value(self, 'target_module').classes('w-full text-xs').props('dense')

            # The Image Container
            with ui.card().classes('w-full p-0 mt-2 bg-black flex items-center justify-center').style('min-height: 200px;'):
                # We use an interactive_image or standard image to display the base64 string
                self.video_player = ui.interactive_image().classes('w-full object-contain')

        # Status Bar
        with ui.row().classes('w-full items-center justify-between px-2'):
            self.status_label = ui.label("Status: Idle").classes("text-[10px] text-slate-500 font-mono")

    def save(self) -> dict:
        data = super().save()
        data.update({
            "target_module": self.target_module
        })
        return data

    def _load_config(self, data: dict):
        self.target_module = data.get("target_module", None)