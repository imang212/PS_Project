import traceback
from nicegui import ui
from Node import Node, NodeRegistry
from Schema import PipelinePayload

# ==========================================
# THE TAGGER CONTEXT API
# ==========================================
class TaggerAPI:
    """Provides access to persistent memory and helper functions for tagging."""
    def __init__(self, payload: PipelinePayload, memory: dict):
        self._payload = payload
        self.memory = memory

    def get_value(self, module_prefix: str, key: str, default=None):
        for mod_name, mod in self._payload.modules.items():
            if mod_name.startswith(module_prefix):
                # Handle both dicts and dataclasses
                if isinstance(mod.data, dict):
                    return mod.data.get(key, default)
                elif hasattr(mod.data, key):
                    return getattr(mod.data, key)
        return default

    def add_global_tag(self, module_prefix: str, tag: str):
        """Helper function to easily append a global tag to a specific module."""
        for mod_name, mod in self._payload.modules.items():
            if mod_name.startswith(module_prefix):
                if isinstance(mod.data, dict):
                    if "global_tags" not in mod.data:
                        mod.data["global_tags"] = []
                    if tag not in mod.data["global_tags"]:
                        mod.data["global_tags"].append(tag)
                else:
                    if not hasattr(mod.data, "global_tags"):
                        mod.data.global_tags = []
                    if tag not in mod.data.global_tags:
                        mod.data.global_tags.append(tag)

# ==========================================
# THE DEFAULT USER TAGGER SCRIPT
# ==========================================
DEFAULT_CODE = """def tag_detection(api, detection) -> None:
    # 1. Setup tag lists (handles both dict and dataclass structures)
    tags_list = detection.get("tags", []) if isinstance(detection, dict) else getattr(detection, "tags", [])
    car_id = detection.get("track_id", -1) if isinstance(detection, dict) else getattr(detection, "track_id", -1)
    
    if car_id == -1: return # Ignore untracked objects
        
    # 2. Get current X (center of bounding box)
    bbox = detection.get("bbox") if isinstance(detection, dict) else getattr(detection, "bbox")
    current_x = bbox[0] + (bbox[2] - bbox[0]) / 2 

    # 3. Initialize memory for new cars
    if car_id not in api.memory:
        api.memory[car_id] = {"last_x": current_x}

    # 4. Calculate velocity (dx)
    dx = current_x - api.memory[car_id]["last_x"]

    # 5. Apply Object Tags & Global Tags
    if dx > 5:
        tags_list.append("Moving_Right")
    elif dx < -5:
        tags_list.append("Moving_Left")
        
        # Example of triggering a global violation if moving left in the bottom lane
        # api.add_global_tag("Hailo", "Violation_Wrong_Way")

    # 6. Save state for next frame
    api.memory[car_id]["last_x"] = current_x
"""

@NodeRegistry.register("Tagger")
class TagNode(Node):
    node_color = "amber-600"
    has_input = True
    has_output = True

    def __init__(self):
        super().__init__()
        self.width = 450
        self.height = 420
        self.code_text = DEFAULT_CODE
        self.compiled_func = None
        self.error_msg = ""
        self.node_memory = {} # <--- Persistent memory dictionary
        self._compile_code()

    def _start(self):
        self.node_memory.clear() # Reset memory on start
        self._compile_code()

    def _stop(self):
        if hasattr(self, 'status_label'):
            self.status_label.set_text("Pipeline Stopped")
            self.status_label.classes(replace='text-[10px] font-mono w-full truncate text-slate-500')

    def _compile_code(self):
        self.node_memory.clear() # Reset memory if code changes
        try:
            local_vars = {}
            exec(self.code_text, {}, local_vars)
            
            if 'tag_detection' in local_vars:
                self.compiled_func = local_vars['tag_detection']
                self.error_msg = ""
            else:
                self.error_msg = "Error: Must define 'def tag_detection(api, detection):'"
                self.compiled_func = None
            self._update_ui()
        except Exception as e:
            self.error_msg = f"Syntax Error: {str(e)}"
            self.compiled_func = None
            self._update_ui()

    def _open_code_editor(self):
        """Opens a large modal dialog for easier code editing."""
        with ui.dialog() as dialog, ui.card().classes('w-[800px] max-w-[90vw] h-[80vh] flex flex-col bg-slate-800 p-4'):
            # Header
            with ui.row().classes('w-full justify-between items-center text-white mb-2 shrink-0'):
                ui.label("Advanced Code Editor").classes('text-lg font-bold')
                ui.button(icon='close', on_click=dialog.close).props('flat round dense text-white bg-slate-700 hover:bg-red-500')
            
            ui.label("Changes are saved automatically as you type.").classes('text-xs text-slate-400 mb-2 shrink-0')

            ui.codemirror(language='Python', theme='material') \
                .bind_value(self, 'code_text') \
                .on('change', self._compile_code) \
                .classes('w-full grow rounded text-sm overflow-hidden')
                
        dialog.open()


    def _input(self, payload: PipelinePayload):
        if not self.compiled_func or not payload:
            return payload

        api = TaggerAPI(payload, self.node_memory)
        
        for mod_name, mod in payload.modules.items():
            detections = None
            if isinstance(mod.data, dict):
                detections = mod.data.get("detections")
            elif hasattr(mod.data, "detections"):
                detections = getattr(mod.data, "detections")

            if detections is not None:
                for det in detections:
                    try:
                        self.compiled_func(api, det)
                    except Exception as e:
                        print(f"[Tag Node] Error executing user script: {e}")

        return payload

    def create_content(self):
        with ui.column().classes('w-full h-full gap-1 p-2 bg-slate-50 border border-slate-200 border-l-4 border-l-blue-500 shadow-sm'):
            
            # --- HEADER ---
            with ui.row().classes('w-full justify-between items-center'):
                ui.label("PYTHON LOGIC").classes('text-[10px] font-bold text-blue-800')
                
                # Added a row to group the buttons together
                with ui.row().classes('gap-1'):
                    # The new expand button!
                    ui.button(icon='fullscreen', on_click=self._open_code_editor) \
                        .props('flat dense size=xs color=blue').tooltip("Expand Editor")
                    ui.button(icon='play_arrow', on_click=self._compile_code) \
                        .props('flat dense size=xs color=green').tooltip("Recompile Code")

            # --- THE SMALL SCROLLABLE EDITOR ---
            # Removed 'autogrow', added a fixed height (e.g., h-32 or max-h-40) and overflow-y-auto
            self.editor = ui.codemirror(language='Python', theme='material') \
                .bind_value(self, 'code_text') \
                .on('change', self._compile_code) \
                .classes('w-full h-40 rounded text-[10px] overflow-hidden')
            
            # --- STATUS LABEL ---
            self.status_label = ui.label().classes('text-[10px] font-mono w-full truncate')
            self._update_ui()

    def _on_code_change(self, e):
        self.code_text = e.value
        self._compile_code()

    def _update_ui(self):
        if hasattr(self, 'status_label'):
            if self.error_msg:
                self.status_label.set_text(self.error_msg)
                self.status_label.classes(replace='text-[10px] font-mono w-full truncate text-red-500')
            else:
                self.status_label.set_text("Compiled Successfully")
                self.status_label.classes(replace='text-[10px] font-mono w-full truncate text-green-600')

    def save(self) -> dict:
        base = super().save()
        base.update({"code_text": self.code_text})
        return base

    def _load_config(self, data: dict):
        self.code_text = data.get("code_text", DEFAULT_CODE)
        self._compile_code()