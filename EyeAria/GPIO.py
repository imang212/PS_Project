import json
import time
import threading
from typing import List, Dict, Optional, Any
from nicegui import ui
from Node import Node, NodeRegistry
from Schema import PipelinePayload, Detection

# --- Hardware Abstraction ---
try:
    from gpiozero import Servo
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    class Servo:
        def __init__(self, pin): self.pin = pin
        def value(self, v): pass 
        def detach(self): pass

@NodeRegistry.register("GPIO")
class GPIONode(Node):
    node_color = "orange-700"
    has_input = True
    has_output = False 

    def __init__(self):
        super().__init__()
        
        # --- Configuration ---
        self.rules = {"op": "AND", "conditions": []}
        self.state_timeout = 2.0
        self.repeat_enter = False
        self.repeat_exit = False
        
        self.enter_sequence: List[Dict] = []
        self.exit_sequence: List[Dict] = []

        # --- Runtime State ---
        self.last_valid_detection_time = 0.0
        self.current_state = "IDLE" 
        self.stop_signal = False
        self.worker_thread: Optional[threading.Thread] = None
        self.active_servo_map = {} 
        self.log_messages = []

    def _start(self):
        self.stop_signal = False
        self.current_state = "IDLE"
        self.worker_thread = threading.Thread(target=self._logic_loop, daemon=True)
        self.worker_thread.start()
        self.log("GPIO Engine Started")

    def _stop(self):
        self.stop_signal = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        for pin, servo in self.active_servo_map.items():
            try: servo.detach()
            except: pass
        self.active_servo_map.clear()

    # ==========================================
    # INPUT PROCESSING
    # ==========================================
    def _calculate_geometry(self, det: Detection, index: int) -> Dict:
        """Helper to standardize detection data for rule evaluation."""
        xmin, ymin, xmax, ymax = det.bbox
        width = xmax - xmin
        height = ymax - ymin
        return {
            "label": det.label,
            "confidence": det.confidence,
            "track_id": det.track_id,
            "width": width, "height": height,
            "xcenter": xmin + (width/2), "ycenter": ymin + (height/2)
        }

    def _evaluate_condition(self, condition: Dict, data: Dict) -> bool:
        """Recursive rule evaluator with AND/OR support."""
        if "op" in condition:
            results = [self._evaluate_condition(c, data) for c in condition["conditions"]]
            if not results: return True
            # OR Logic implementation
            return all(results) if condition["op"] == "AND" else any(results)
        
        try:
            actual = data.get(condition["key"])
            val = condition["value"]
            op = condition["operator"]
            
            if op == "==": return str(actual) == str(val)
            if op == "!=": return str(actual) != str(val)
            
            a_f, v_f = float(actual), float(val)
            if op == ">": return a_f > v_f
            if op == "<": return a_f < v_f
            if op == ">=": return a_f >= v_f
            if op == "<=": return a_f <= v_f
        except: return False
        return False

    def _input(self, payload: PipelinePayload) -> Optional[PipelinePayload]:
        try:
            has_match = False
            if not self.rules["conditions"]:
                has_match = len(payload.detections) > 0
            else:
                for i, det in enumerate(payload.detections):
                    geo = self._calculate_geometry(det, i)
                    if self._evaluate_condition(self.rules, geo):
                        has_match = True
                        break
            
            if has_match:
                self.last_valid_detection_time = time.time()
                
            return payload 
        except Exception as e:
            print(f"GPIO Input Error: {e}")
            return None

    # ==========================================
    # LOGIC LOOP & HARDWARE
    # ==========================================
    def _logic_loop(self):
        while not self.stop_signal:
            now = time.time()
            time_since_det = now - self.last_valid_detection_time
            is_active_zone = time_since_det < self.state_timeout
            
            if is_active_zone:
                if self.current_state in ["IDLE", "EXITED"]:
                    self.current_state = "ENTERING"
                    self.log("Trigger: Enter Sequence")
                    self._run_sequence(self.enter_sequence)
                    self.current_state = "ENTERED"
                elif self.current_state == "ENTERED" and self.repeat_enter:
                    self._run_sequence(self.enter_sequence)
            else:
                if self.current_state == "ENTERED":
                    self.current_state = "EXITING"
                    self.log("Trigger: Exit Sequence")
                    self._run_sequence(self.exit_sequence)
                    self.current_state = "EXITED"
                elif self.current_state == "EXITED" and self.repeat_exit:
                    self._run_sequence(self.exit_sequence)

            time.sleep(0.1)

    def _run_sequence(self, sequence: List[Dict]):
        for step in sequence:
            if self.stop_signal: break
            action_type = step.get("type")
            
            if action_type == "WAIT":
                dur = float(step.get("duration", 1.0))
                end_time = time.time() + dur
                while time.time() < end_time and not self.stop_signal:
                    time.sleep(0.1)

            elif action_type == "SERVO":
                pin = int(step.get("pin", 17))
                speed = float(step.get("speed", 0.0))
                dur = float(step.get("duration", 1.0))
                self._set_servo(pin, speed)
                
                end_time = time.time() + dur
                while time.time() < end_time and not self.stop_signal:
                    time.sleep(0.1)
                self._set_servo(pin, 0)

    def _set_servo(self, pin, speed):
        if not GPIO_AVAILABLE:
            self.log(f"[MOCK] Pin {pin} -> {speed}")
            return
        try:
            if pin not in self.active_servo_map:
                self.active_servo_map[pin] = Servo(pin)
            servo = self.active_servo_map[pin]
            val = max(-1.0, min(1.0, speed))
            if val == 0: servo.detach()
            else: servo.value = val
        except Exception as e:
            self.log(f"GPIO Error: {e}")

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_messages.insert(0, f"[{ts}] {msg}")
        self.log_messages = self.log_messages[:10]
        
    # ==========================================
    # UI CONSTRUCTION
    # ==========================================
    def create_content(self):
        with ui.column().classes('w-full gap-2'):
            # 1. Status Monitor (Removed 'rounded')
            with ui.column().classes('w-full bg-slate-900 border border-slate-700 border-l-4 border-l-orange-500 p-2 shadow-sm gap-0'):
                with ui.row().classes('w-full justify-between items-center mb-1'):
                    ui.label("ENGINE STATUS").classes('text-[10px] font-bold text-orange-400')
                    self.status_label = ui.label("IDLE").classes('text-[10px] font-mono font-bold text-green-400')
                self.log_container = ui.column().classes('w-full text-[9px] font-mono text-slate-300 max-h-16 overflow-y-auto gap-0')

            # 2. Config Tabs (Removed 'rounded')
            with ui.column().classes('bg-slate-50 border border-slate-200 border-l-4 border-l-slate-500 w-full p-0 shadow-sm overflow-hidden gap-0'):
                with ui.tabs().classes('w-full text-xs bg-slate-200 text-slate-700') as tabs:
                    tab_rules = ui.tab('Rules').classes('p-0 m-0 min-h-0 h-8')
                    tab_enter = ui.tab('Enter').classes('p-0 m-0 min-h-0 h-8')
                    tab_exit = ui.tab('Exit').classes('p-0 m-0 min-h-0 h-8')

                with ui.tab_panels(tabs, value=tab_rules).classes('w-full bg-transparent p-2'):
                    with ui.tab_panel(tab_rules).classes('p-0'):
                        with ui.row().classes('w-full items-center justify-between mb-2'):
                            ui.label("LOGIC:").classes('text-[10px] font-bold text-slate-700')
                            ui.toggle(['AND', 'OR'], value=self.rules['op'], on_change=lambda e: self.update_op(e.value)) \
                                .props('dense unelevated toggle-color=orange color=white text-color=slate-700') \
                                .classes('border border-slate-300 shadow-sm')

                        with ui.row().classes('items-center gap-1 mb-2'):
                            self.key_sel = ui.select(['label', 'confidence', 'track_id', 'xcenter'], value='label').classes('w-20 text-xs').props('dense')
                            self.op_sel = ui.select(['==', '!=', '>', '<'], value='==').classes('w-12 text-xs').props('dense')
                            self.val_input = ui.input(placeholder='val').classes('w-16 text-xs').props('dense')
                            ui.button(icon='add', on_click=self.add_rule).props('flat dense size=sm color=orange')
                            
                        self.rules_display = ui.column().classes('w-full gap-1')
                        ui.number("State Timeout (s)", value=2.0).bind_value(self, 'state_timeout').classes('w-full text-xs mt-2').props('dense')

                    with ui.tab_panel(tab_enter).classes('p-0'):
                        ui.checkbox("Repeat Sequence").bind_value(self, 'repeat_enter').classes('text-[10px] font-bold text-slate-700 mb-2').props('dense size=sm')
                        self._render_sequence_builder(self.enter_sequence, "Enter")

                    with ui.tab_panel(tab_exit).classes('p-0'):
                        ui.checkbox("Repeat Sequence").bind_value(self, 'repeat_exit').classes('text-[10px] font-bold text-slate-700 mb-2').props('dense size=sm')
                        self._render_sequence_builder(self.exit_sequence, "Exit")

        self.ui_timer = ui.timer(0.5, self._update_ui_state)
        self.refresh_rules_ui()

    def refresh_rules_ui(self):
        if hasattr(self, 'rules_display'):
            self.rules_display.clear()
            with self.rules_display:
                ui.label(f"Mode: {self.rules['op']}").classes('text-[8px] text-slate-400 uppercase font-bold')
                for i, c in enumerate(self.rules["conditions"]):
                    # Removed 'rounded'
                    with ui.row().classes('w-full justify-between items-center bg-white border border-slate-200 p-1 shadow-sm'):
                        ui.label(f"{c['key']} {c['operator']} {c['value']}").classes('text-[10px] font-mono text-slate-600')
                        ui.button(icon='close', on_click=lambda idx=i: self.remove_rule(idx)).props('flat dense color=red size=xs')

    def _refresh_sequence_ui(self, context):
        seq_list = self.enter_sequence if context == "Enter" else self.exit_sequence
        container = getattr(self, f"container_{context}")
        container.clear()
        with container:
            for i, act in enumerate(seq_list):
                # Removed 'rounded'
                with ui.row().classes('w-full items-center justify-between bg-white border border-slate-200 p-1 shadow-sm mt-1'):
                    ui.label(f"{act['type']} ({act['duration']}s)").classes('text-[10px] font-mono text-slate-600')
                    ui.button(icon='close', on_click=lambda idx=i: self.remove_action(seq_list, idx, context)).props('flat dense color=red size=xs')

    def update_op(self, val):
        self.rules['op'] = val
        self.refresh_rules_ui()

    def _render_sequence_builder(self, seq_list, context):
        with ui.column().classes('w-full gap-1'):
            with ui.row().classes('w-full items-end gap-1 bg-gray-50 p-1 border'):
                type_sel = ui.select(["SERVO", "WAIT"], value="SERVO").classes('w-16 text-xs')
                with ui.row().bind_visibility_from(type_sel, 'value', value='SERVO').classes('gap-1'):
                    pin_in = ui.number("Pin", value=17).classes('w-10 text-xs')
                    speed_in = ui.number("Spd", value=1.0).classes('w-10 text-xs')
                dur_in = ui.number("Sec", value=1.0).classes('w-10 text-xs')
                
                def add_action():
                    act = {"type": type_sel.value, "duration": dur_in.value}
                    if type_sel.value == "SERVO":
                        act.update({"pin": int(pin_in.value), "speed": float(speed_in.value)})
                    seq_list.append(act)
                    self._refresh_sequence_ui(context)
                ui.button(icon='add', on_click=add_action).props('flat dense')

            attr = f"container_{context}"
            if not hasattr(self, attr): setattr(self, attr, ui.column().classes('w-full'))
            self._refresh_sequence_ui(context)


    def remove_action(self, seq_list, index, context):
        seq_list.pop(index)
        self._refresh_sequence_ui(context)

    def add_rule(self):
        self.rules["conditions"].append({"key": self.key_sel.value, "operator": self.op_sel.value, "value": self.val_input.value})
        self.refresh_rules_ui()

    def remove_rule(self, index):
        self.rules["conditions"].pop(index)
        self.refresh_rules_ui()

    def _update_ui_state(self):
        if hasattr(self, 'status_label'):
            self.status_label.set_text(self.current_state)
            self.log_container.clear()
            with self.log_container:
                for msg in self.log_messages: ui.label(msg)

    def save(self) -> dict:
        base = super().save()
        base.update({
            "rules": self.rules, "enter_sequence": self.enter_sequence,
            "exit_sequence": self.exit_sequence, "state_timeout": self.state_timeout,
            "repeat_enter": self.repeat_enter, "repeat_exit": self.repeat_exit
        })
        return base

    def _load_config(self, data: dict):
        self.rules = data.get("rules", {"op": "AND", "conditions": []})
        self.enter_sequence = data.get("enter_sequence", [])
        self.exit_sequence = data.get("exit_sequence", [])
        self.state_timeout = data.get("state_timeout", 2.0)
        self.repeat_enter = data.get("repeat_enter", False)
        self.repeat_exit = data.get("repeat_exit", False)
