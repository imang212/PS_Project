import traceback
from nicegui import ui
from Node import Node, NodeRegistry
from Schema import PipelinePayload

# ==========================================
# THE READ-ONLY CONTEXT API
# ==========================================
class ReadOnlyAPI:
    """Provides safe, read-only access to the pipeline state without deep copying."""
    def __init__(self, payload: PipelinePayload, memory: dict):
        self._payload = payload
        self.memory = memory  # Expose the persistent dictionary to the user

    def get_value(self, module_prefix: str, key: str, default=None):
        """Safely fetches a value from any sensor."""
        for mod_name, mod in self._payload.modules.items():
            if mod_name.startswith(module_prefix) and isinstance(mod.data, dict):
                return mod.data.get(key, default)
        return default

# ==========================================
# THE DEFAULT USER PREDICATE
# ==========================================
DEFAULT_CODE = """def keep_detection(api, detection) -> bool:
    # --- Example: Using Memory to track total cars seen ---
    if "total_cars" not in api.memory:
        api.memory["total_cars"] = 0
        
    if detection.get("label") == "car":
        api.memory["total_cars"] += 1
        print(f"Total cars seen this session: {api.memory['total_cars']}")
    
    # Context Check: Is the camera sideways?
    if api.get_value("IMU", "rotation", 0) > 90:
        return False # Discard this detection
        
    # Detection Check: Is it a high-confidence person?
    is_person = detection.get("label") == "person"
    is_confident = detection.get("confidence", 0) > 0.5
    
    # Return True to keep, False to discard
    return is_person and is_confident
"""

@NodeRegistry.register("Filter")
class FilterNode(Node):
    node_color = "blue-600"
    has_input = True
    has_output = True

    def __init__(self):
        super().__init__()
        self.width = 400
        self.height = 380
        self.code_text = DEFAULT_CODE
        self.compiled_func = None
        self.error_msg = ""
        self.node_memory = {}
        self._compile_code()

    def _start(self):
        self.node_memory.clear()
        self._compile_code()

    def _stop(self):
        # The filter node has no background threads or hardware to release.
        if hasattr(self, 'status_label'):
            self.status_label.set_text("Pipeline Stopped")
            self.status_label.classes(replace='text-[10px] font-mono w-full truncate text-slate-500')

    def _compile_code(self):
        self.node_memory.clear()
        try:
            local_vars = {}
            exec(self.code_text, {}, local_vars)
            
            if 'keep_detection' in local_vars:
                self.compiled_func = local_vars['keep_detection']
                self.error_msg = ""
            else:
                self.error_msg = "Error: Must define 'def keep_detection(api, detection):'"
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

        api = ReadOnlyAPI(payload, self.node_memory)
        
        for mod_name, mod in payload.modules.items():
            # 1. Safely extract detections whether data is a dictionary or a dataclass
            detections = None
            if isinstance(mod.data, dict):
                detections = mod.data.get("detections")
            elif hasattr(mod.data, "detections"):
                detections = getattr(mod.data, "detections")

            # 2. If we found a detection list, filter it
            if detections is not None:
                valid_dets = []
                for det in detections:
                    try:
                        # Run the user's predicate function
                        if self.compiled_func(api, det) is True:
                            valid_dets.append(det)
                    except Exception as e:
                        print(f"[Filter Node] Error evaluating detection: {e}")
                        
                # 3. Reassign the filtered list AND update the count
                if isinstance(mod.data, dict):
                    mod.data["detections"] = valid_dets
                    mod.data["count"] = len(valid_dets) # <--- UPDATES METADATA
                else:
                    setattr(mod.data, "detections", valid_dets)
                    setattr(mod.data, "count", len(valid_dets)) # <--- UPDATES METADATA

        return payload

    # ==========================================
    # UI CONSTRUCTION
    # ==========================================
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