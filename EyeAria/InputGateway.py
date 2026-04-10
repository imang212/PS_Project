from Node import Node, NodeRegistry
from Schema import PipelinePayload, ModuleData
from nicegui import ui, app
import time

@NodeRegistry.register("Input Gateway")
class InputGatewayNode(Node):
    node_color = "purple-800"
    has_input = False 
    has_output = True
    deletable = False       # Cannot be deleted
    is_gateway = True       # Flags this as the root for saving

    def __init__(self):
        super().__init__()
        self.input_nodes = []
        # The master envelope that lives in the Gateway
        self.master_payload = PipelinePayload(timestamp=time.time())

    def generate_template(self) -> dict:
        """Generates the master schema based on connected input nodes."""
        template = {
            "timestamp": "float",
            "modules": {}
        }
        
        # Dynamically build the schema based on whatever is plugged into the left side
        for inp in self.input_nodes:
            # Create a unique but readable key for each connected sensor (e.g., 'Hailo_a1b2c3')
            module_key = f"{inp._node_type_name}_{inp.id[:6]}" 
            
            template["modules"][module_key] = {
                "is_new": "boolean",
                "count": "integer",
                "detections": "list[Detection]"
            }
            
        return template
        
    def build_header_buttons(self):
        """Custom header buttons for the gateway"""
        ui.button(icon='login', on_click=lambda: app.logic.add_node_dialog(self, 'input')) \
            .props('flat round dense text-blue-400 size=sm').tooltip('Add Input')
        ui.button(icon='add', on_click=lambda: app.logic.add_node_dialog(self, 'output')) \
            .props('flat round dense text-green-400 size=sm').tooltip('Add Output')

    def add_input_node(self, node: 'Node'):
        """Connects a node to feed INTO the gateway."""
        self.input_nodes.append(node)
        node.subscribers.append(self)
        
    def remove_input_node(self, node: 'Node'):
        """Disconnects a node feeding the gateway."""
        if node in self.input_nodes:
            self.input_nodes.remove(node)
        if self in node.subscribers:
            node.subscribers.remove(self)

    def _start(self): 
        # Start all left-side sensors before starting the right-side pipeline
        for inp in self.input_nodes: inp.start()
        
    def _stop(self): 
        for inp in self.input_nodes: inp.stop()
        
    def _input(self, chunk):
        """
        Catches `{key: ModuleData}` chunks from sensors on the left,
        updates the master payload, and pushes it to the right.
        """
        if not isinstance(chunk, dict):
            return None
            
        # 1. Update the master payload with the new sensor data
        for module_key, module_data in chunk.items():
            self.master_payload.modules[module_key] = module_data
            
        # 2. Stamp the synchronized time
        self.master_payload.timestamp = time.time()
        
        # 3. Pass a deepcopy of the master payload down the processing tree
        return self.master_payload.copy()
        
    def create_content(self):
        with ui.column().classes('w-full gap-1 p-2 bg-purple-50 border border-purple-200'):
            ui.label("CENTRAL PIPELINE ROUTER").classes('text-[10px] font-bold text-purple-800')
            ui.label("Collects asynchronous inputs and distributes the unified payload downstream.") \
                .classes('text-[9px] text-slate-600 leading-tight italic')

    def save(self) -> dict:
        base = super().save()
        base.update({
            "input_nodes": [inp.save() for inp in self.input_nodes]
        })
        return base

    def _load_config(self, data: dict):
        self.input_nodes = []
        for inp_data in data.get("input_nodes", []):
            child = Node.load(inp_data)
            if child: self.add_input_node(child)