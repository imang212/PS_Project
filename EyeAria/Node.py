import abc
import json
import uuid
from typing import List, Dict, Type, Optional
from nicegui import ui, app

from Schema import PipelinePayload

class NodeRegistry:
    _nodes: Dict[str, Type['Node']] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            cls._nodes[name] = subclass
            subclass._node_type_name = name
            return subclass
        return decorator

    @classmethod
    def get_all(cls):
        return cls._nodes

    @classmethod
    def get(cls, name):
        return cls._nodes.get(name)

class Node(abc.ABC):
    _node_type_name: str = "BaseNode"
    has_input: bool = True
    has_output: bool = True
    node_color: str = "slate-800" # Default color
    deletable: bool = True
    is_gateway: bool = False

    _spawn_offset = 0 

    def __init__(self):
        self.id = str(uuid.uuid4())
        
        Node._spawn_offset = (Node._spawn_offset + 30) % 300
        self.x = 50 + Node._spawn_offset
        self.y = 50 + Node._spawn_offset
        self.width = 280  
        self.height = 100 

        self.subscribers: List['Node'] = []
        self.parent: Optional['Node'] = None
        self.running = False
        self.collapsed = False
        
        self.card = None
        self.mini_card = None

    def push_schema(self, template: dict) -> bool:
        """
        Force-feeds the schema down the tree. 
        Returns True if the branch is ready, False if configuration is broken.
        """
        # 1. Evaluate this node's readiness
        is_ready = self.on_schema_update(template)
        if not is_ready:
            print(f"[!] Node {self._node_type_name} ({self.id}) failed schema validation.")
            return False
            
        # 2. Push to all children
        for sub in self.subscribers:
            if not sub.push_schema(template):
                return False
                
        return True

    def on_schema_update(self, template: dict) -> bool:
        """
        OVERRIDE THIS IN CHILD CLASSES.
        Read the template, update UI if needed. 
        Return True if your current config is valid for this schema.
        Default is True (e.g., MQTTSink doesn't care about schema shape).
        """
        return True

    def start(self):
        if self.running: return
        self.running = True
        self._start()
        for sub in self.subscribers: sub.start()

    def stop(self):
        if not self.running: return
        self.running = False
        self._stop()
        for sub in self.subscribers: sub.stop()

    def input(self, payload: PipelinePayload):
        if not self.running: return
        
        # Pass the object directly
        result_payload = self._input(payload)
        
        # If the node returns a modified payload, notify subscribers
        if self.has_output and result_payload is not None:
            self.notify(result_payload)

    def notify(self, payload: PipelinePayload):
        for sub in self.subscribers: 
            # Use your custom copy method to branch the data safely!
            sub.input(payload.copy())
            
    def add_subscriber(self, node: 'Node'):
        node.parent = self
        self.subscribers.append(node)

    def remove_subscriber(self, node: 'Node'):
        if node in self.subscribers:
            self.subscribers.remove(node)
            node.parent = None

    def build_node_ui(self, is_mini: bool = False):
        """Builds either a full draggable node or a tiny minimap dot."""
        if is_mini:
            # Store the reference to the minimap dot
            self.mini_card = ui.element('div').classes(f'absolute bg-{self.node_color} rounded-sm') \
                .style(f'left: {self.x}px; top: {self.y}px; width: {self.width}px; height: {self.height}px;')
            return

                # The main card remains absolutely positioned
        self.card = ui.card().classes('absolute w-72 shadow-lg p-0 border border-slate-300 bg-white select-none touch-none') \
            .style(f'left: {self.x}px; top: {self.y}px; z-index: 10;')
        
        # Stops clicks on the card from panning the canvas
        self.card.on('pointerdown.stop', lambda: None) 

        with self.card:
            ui.element('q-resize-observer').on('resize', self._handle_resize)

            # The Header (Drag Handle)
            handle = ui.row().classes(f'w-full bg-{self.node_color} text-white p-1 cursor-move items-center justify-between no-wrap')
            handle.style('touch-action: none; user-select: none;')

            with handle:
                icon = 'expand_more' if self.collapsed else 'expand_less'
                ui.button(icon=icon, on_click=self.toggle_collapse).props('flat round dense text-white size=sm')
                ui.label(self._node_type_name).classes('font-bold text-[11px] grow cursor-move')
                
                # --- REPLACE THE HARDCODED BUTTONS WITH THIS ---
                self.build_header_buttons()

            if not self.collapsed:
                with ui.column().classes('p-2 w-full'):
                    self.create_content()

                # --- REFINED DRAG LOGIC ---
        self.offset_x = 0
        self.offset_y = 0

        def start_drag(e):
            if hasattr(app, 'logic'):
                app.logic.active_node = self # Tell the canvas WE are moving
                zoom, px, py = app.logic.zoom, app.logic.pan_x, app.logic.pan_y
                
                # Pointer events unify mouse and touch perfectly!
                mx = e.args.get('clientX', 0)
                my = e.args.get('clientY', 0)
                
                self.offset_x = (mx - px) / zoom - self.x
                self.offset_y = (my - py) / zoom - self.y
                self.card.style('z-index: 100;')

        # .stop prevents the canvas from panning when dragging a node
        handle.on('pointerdown.stop', start_drag, args=['clientX', 'clientY'])

    def build_header_buttons(self):
        """Overrideable method for rendering header buttons."""
        if self.has_output:
            ui.button(icon='add', on_click=lambda: app.logic.add_node_dialog(self, 'output')) \
                .props('flat round dense text-green-400 size=sm').tooltip('Add Output')
        if self.deletable:
            ui.button(icon='close', on_click=lambda: app.logic.delete_node(self)) \
                .props('flat round dense text-red-400 size=sm').tooltip('Delete Node')

    def toggle_collapse(self):
        self.collapsed = not self.collapsed
        if hasattr(app, 'logic'):
            app.logic.refresh_ui()

    def _handle_resize(self, e):
        try:
            size = e.args if 'width' in e.args else e.args.get('size', {})
            self.width, self.height = size.get('width', self.width), size.get('height', self.height)
            if hasattr(app, 'logic'): 
                app.logic.redraw_wires()
                # Keep minimap synchronized with collapse/expand size changes
                if self.mini_card:
                    self.mini_card.style(f'width: {self.width}px; height: {self.height}px;')
        except Exception: pass

    @abc.abstractmethod
    def _start(self): pass
    @abc.abstractmethod
    def _stop(self): pass
    @abc.abstractmethod
    def _input(self, payload: PipelinePayload) -> Optional[PipelinePayload]: pass
    @abc.abstractmethod
    def create_content(self): pass

    def save(self) -> dict:
        return {
            "type": self._node_type_name,
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,   
            "height": self.height, 
            "collapsed": self.collapsed,
            # Prevent saving the Gateway inside an input node's subscribers to avoid infinite recursion
            "subscribers": [sub.save() for sub in self.subscribers if not getattr(sub, 'is_gateway', False)]
        }
        
    @classmethod
    def load(cls, data: dict) -> 'Node':
        node_type = data.get("type")
        node_class = NodeRegistry.get(node_type)
        if not node_class:
            raise ValueError(f"Node type '{node_type}' not found.")
            
        instance = node_class()
        instance.id = data.get("id", str(uuid.uuid4()))
        instance.x = data.get("x", instance.x)
        instance.y = data.get("y", instance.y)
        # Restore dimensions
        instance.width = data.get("width", instance.width)
        instance.height = data.get("height", instance.height)
        instance.collapsed = data.get("collapsed", False)
        
        instance._load_config(data)
        
        for sub_data in data.get("subscribers", []):
            child = Node.load(sub_data)
            if child: instance.add_subscriber(child)
        return instance
    @abc.abstractmethod
    def _load_config(self, data: dict): pass