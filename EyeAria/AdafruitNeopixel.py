import time
import threading
import traceback
from typing import Optional
from nicegui import ui
from Node import Node, NodeRegistry
from Schema import PipelinePayload

# Import the API wrapper we built for the Filter node
try:
    from Filter import ReadOnlyAPI
except ImportError:
    # Fallback if ReadOnlyAPI isn't available
    class ReadOnlyAPI:
        def __init__(self, payload, memory=None):
            self._payload = payload
            self.memory = memory or {}
        def get_value(self, module_prefix, key, default=None):
            for mod_name, mod in self._payload.modules.items():
                if mod_name.startswith(module_prefix):
                    if isinstance(mod.data, dict): return mod.data.get(key, default)
                    elif hasattr(mod.data, key): return getattr(mod.data, key)
            return default

# --- NeoPixel Abstraction ---
try:
    import board
    import neopixel
    NEOPIXEL_AVAILABLE = True
except ImportError:
    NEOPIXEL_AVAILABLE = False

class MockPixels:
    def fill(self, color): pass
    def show(self): pass
    def __setitem__(self, key, value): pass
    def __getitem__(self, key): return (0,0,0)

# ==========================================
# DEFAULT GPIO LOGIC (TRAFFIC LIGHT DEMO)
# ==========================================
DEFAULT_CODE = """def led_loop(api, pixels, memory, current_time, is_new_frame) -> None:
    # 1. State Initialization
    if "state" not in memory:
        memory["state"] = "GREEN"
        memory["flash_toggle"] = False
        memory["last_flash"] = current_time

    # ==========================================
    # EVENT 1: INPUT HANDLING (Runs exactly ONCE per frame)
    # ==========================================
    if is_new_frame and api:
        global_tags = api.get_value("Hailo", "global_tags", [])
        
        # If the camera sees a violation, immediately update our state
        if "Violation_Wrong_Way" in global_tags:
            if memory["state"] != "VIOLATION":
                print("NEW EVENT: Violation Detected!") # This will only print once!
                memory["state"] = "VIOLATION"
                
        # If it's clear, return to green
        elif len(global_tags) == 0:
            memory["state"] = "GREEN"

    # ==========================================
    # EVENT 2: HARDWARE LOOP (Runs at 20 FPS constantly)
    # ==========================================
    if memory["state"] == "VIOLATION":
        # Strobe Logic - independent of camera frame rate!
        if current_time - memory["last_flash"] > 0.1:
            memory["flash_toggle"] = not memory["flash_toggle"]
            memory["last_flash"] = current_time
            
        if memory["flash_toggle"]:
            pixels.fill((255, 0, 0)) # Red
        else:
            pixels.fill((0, 0, 255)) # Blue
            
    elif memory["state"] == "GREEN":
        pixels.fill((0, 255, 0)) # Solid Green
        
    pixels.show()
"""

@NodeRegistry.register("Programmable LED")
class ProgrammableLEDNode(Node):
    node_color = "purple-600"
    has_input = True
    has_output = False 

    def __init__(self):
        super().__init__()
        self.width = 450
        self.height = 420
        
        self.brightness = 0.1 
        self.pin_number = 18
        
        self.code_text = DEFAULT_CODE
        self.compiled_func = None
        self.error_msg = ""
        self.node_memory = {}

        self.stop_signal = False
        self.worker_thread: Optional[threading.Thread] = None
        self.log_messages = []
        self._pixels = None
        self.latest_payload = None
        self.last_processed_timestamp = 0.0

        self._compile_code()

    def _start(self):
        self.stop_signal = False
        self.node_memory.clear()
        self._compile_code()
        self._init_pixels()
        
        # Start the continuous hardware loop
        self.worker_thread = threading.Thread(target=self._logic_loop, daemon=True)
        self.worker_thread.start()
        self.log(f"Hardware Loop Started (Pin D{self.pin_number})")

    def _stop(self):
        self.stop_signal = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        if self._pixels and NEOPIXEL_AVAILABLE:
            self._pixels.fill((0, 0, 0))
            self._pixels.show()
            
        self._pixels = None 
        if hasattr(self, 'status_label'):
            self.status_label.set_text("Pipeline Stopped")
            self.status_label.classes(replace='text-[10px] font-mono w-full truncate text-slate-500')

    def _init_pixels(self):
        if NEOPIXEL_AVAILABLE and not self._pixels:
            try:
                pin_obj = getattr(board, f'D{int(self.pin_number)}')
                self._pixels = neopixel.NeoPixel(
                    pin_obj, 24, brightness=self.brightness, auto_write=False
                )
            except Exception as e:
                self.log(f"ERR: Hardware init failed ({e})")
                self._pixels = MockPixels()
        elif not NEOPIXEL_AVAILABLE:
            self._pixels = MockPixels()

    def _compile_code(self):
        self.node_memory.clear()
        try:
            local_vars = {}
            exec(self.code_text, {}, local_vars)
            
            if 'led_loop' in local_vars:
                self.compiled_func = local_vars['led_loop']
                self.error_msg = ""
            else:
                self.error_msg = "Error: Must define 'def led_loop(api, pixels, memory, current_time):'"
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

    # ==========================================
    # INPUT: JUST UPDATE STATE
    # ==========================================
    def _input(self, payload: PipelinePayload):
        # We don't execute logic here. We just save the latest worldview.
        self.latest_payload = payload
        return payload 

    # ==========================================
    # BACKGROUND HARDWARE THREAD (20 FPS)
    # ==========================================
    def _logic_loop(self):
        while not self.stop_signal:
            if self.compiled_func:
                try:
                    api = None
                    is_new_frame = False # <--- Default to false
                    
                    if self.latest_payload:
                        api = ReadOnlyAPI(self.latest_payload)
                        
                        # Check if this payload is newer than the last one we saw!
                        if self.latest_payload.timestamp > self.last_processed_timestamp:
                            is_new_frame = True
                            self.last_processed_timestamp = self.latest_payload.timestamp
                    
                    # Pass the new flag into the user's function!
                    self.compiled_func(api, self._pixels, self.node_memory, time.time(), is_new_frame)
                    
                except Exception as e:
                    error_msg = traceback.format_exc().splitlines()[-1]
                    self.log(f"Script Error: {error_msg}")
                    
            time.sleep(0.05)

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_messages.insert(0, f"[{ts}] {msg}")
        self.log_messages = self.log_messages[:3]
        
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

    def _update_ui_state(self):
        if hasattr(self, 'log_container'):
            self.log_container.clear()
            with self.log_container:
                for msg in self.log_messages: ui.label(msg)

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
        base.update({
            "code_text": self.code_text,
            "brightness": self.brightness,
            "pin_number": self.pin_number
        })
        return base

    def _load_config(self, data: dict):
        self.code_text = data.get("code_text", DEFAULT_CODE)
        self.brightness = data.get("brightness", 0.1)
        self.pin_number = data.get("pin_number", 18)
        self._compile_code()