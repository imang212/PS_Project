import time
from typing import Optional
from nicegui import ui
from Node import Node, NodeRegistry
from Schema import PipelinePayload

@NodeRegistry.register("Payload Viewer")
class PayloadViewerNode(Node):
    node_color = "indigo-600"
    has_input = True
    has_output = True

    def __init__(self):
        super().__init__()
        self.width = 340
        self.height = 350
        
        # State variables
        self.frozen = False
        self.tree = None
        self._update_tick = 0
        
        # Thread-safe UI state
        self.pending_tree = None
        self.ui_timer = None

    def _start(self):
        self.pending_tree = {'id': 'root', 'label': 'Waiting for data...'}
        if not self.ui_timer:
            self.ui_timer = ui.timer(0.1, self._refresh_ui)

    def _stop(self):
        if self.ui_timer:
            self.ui_timer.cancel()
            self.ui_timer = None

    def _refresh_ui(self):
        """Forces the UI to re-render using direct Quasar property mutation."""
        if self.pending_tree is not None and self.tree is not None:
            try:
                self.tree._props['nodes'] = [self.pending_tree]
                self.tree.update()
            except Exception as e:
                pass
            finally:
                self.pending_tree = None

    def _safe_extract(self, obj, max_depth=6, current_depth=0):
        """
        Safely traverses data. 
        Crucially: Squashes simple lists (like coords/IMU data) into single readable strings 
        and rounds floats so they don't visually clutter the tree.
        """
        if current_depth > max_depth:
            return "<Max Depth>"
            
        # Unpack Pydantic objects
        if hasattr(obj, 'model_dump'):
            try: obj = obj.model_dump()
            except: pass
        elif hasattr(obj, 'dict'):
            try: obj = obj.dict()
            except: pass
            
        if isinstance(obj, dict):
            # Recursively process dictionaries
            return {str(k): self._safe_extract(v, max_depth, current_depth + 1) for k, v in obj.items()}
            
        elif isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return "[]"
                
            # --- THE FORMATTING MAGIC ---
            # If the list is short and only contains basic data types, SQUASH IT into one line
            is_primitive = all(isinstance(x, (int, float, str, bool, type(None))) for x in obj)
            
            if is_primitive and len(obj) <= 16:
                formatted_items = []
                for x in obj:
                    if isinstance(x, float): formatted_items.append(f"{x:.3f}") # Round floats to 3 decimals
                    elif isinstance(x, str): formatted_items.append(f"'{x}'")
                    else: formatted_items.append(str(x))
                return f"[{', '.join(formatted_items)}]"
            # ----------------------------

            if len(obj) > 30:
                return f"<{type(obj).__name__} len={len(obj)}>"
                
            return [self._safe_extract(v, max_depth, current_depth + 1) for v in obj]
            
        elif type(obj).__name__ == 'ndarray':
            return f"<ndarray shape={getattr(obj, 'shape', '?')}>"
            
        elif isinstance(obj, float):
            return round(obj, 4) # Standalone floats get rounded to 4 decimals
            
        elif isinstance(obj, bytes):
            return f"<bytes len={len(obj)}>"
            
        else:
            return obj

    def _dict_to_tree(self, data, name="Payload", path="root") -> dict:
        """Converts safe dictionaries into Quasar tree nodes."""
        node = {'id': path, 'label': str(name)}
        
        if isinstance(data, dict):
            node['children'] = [
                self._dict_to_tree(v, str(k), f"{path}_{k}") 
                for k, v in data.items()
            ]
        elif isinstance(data, list):
            node['children'] = [
                self._dict_to_tree(v, f"[{i}]", f"{path}_{i}") 
                for i, v in enumerate(data)
            ]
        else:
            # Leaf node representation (Key: Value)
            node['label'] = f"{name}: {data}"
            
        return node

    def _input(self, payload: PipelinePayload) -> Optional[PipelinePayload]:
        if not self.frozen:
            current_time = time.time()
            if current_time - getattr(self, '_last_process_time', 0) > 0.1:
                self._last_process_time = current_time
                try:
                    self._update_tick += 1
                    
                    # Safely pull data apart and apply the new formatting rules
                    safe_data = self._safe_extract(payload)

                    # Build the UI tree
                    unique_path = f"root_{self._update_tick}"
                    new_tree = self._dict_to_tree(
                        safe_data, 
                        name=f"Payload (Tick: {self._update_tick})", 
                        path=unique_path
                    )

                    self.pending_tree = new_tree
                        
                except Exception as e:
                    self.pending_tree = {'id': f'error_{self._update_tick}', 'label': f"Error: {str(e)}"}

        return payload

    def create_content(self):
        with ui.column().classes('w-full h-full gap-2'):
            with ui.row().classes('w-full items-center justify-between px-1'):
                ui.label("PAYLOAD EXPLORER").classes('text-[10px] font-bold text-slate-500 tracking-wider')
                ui.switch('Freeze').bind_value(self, 'frozen').classes('text-xs font-bold text-indigo-800')

            with ui.scroll_area().classes('w-full grow bg-slate-900 border border-slate-700 p-2 shadow-inner rounded'):
                self.tree = ui.tree(
                    [{'id': 'root', 'label': 'Waiting for data...'}], 
                    label_key='label', 
                    children_key='children'
                ).classes('text-[12px] text-green-400 font-mono tracking-tight') \
                 .props('default-expand-all dense dark')

    def save(self) -> dict:
        data = super().save()
        data.update({"frozen": self.frozen})
        return data

    def _load_config(self, data: dict):
        self.frozen = data.get("frozen", False)