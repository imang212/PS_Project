import abc
import json
import uuid
from typing import List, Dict, Type, Optional
from nicegui import ui, app
from Node import Node, NodeRegistry
import importlib
import os
import sys

# ==========================================
# CONCRETE NODE IMPLEMENTATIONS
# ==========================================

def auto_import_nodes():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    excluded = {"Node.py", "Tree.py", "__init__.py"}
    print(f"Scanning {current_dir} for plugin nodes...")
    
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and filename not in excluded:
            module_name = filename[:-3] 
            try:
                importlib.import_module(module_name)
                print(f"  [+] Loaded module: {module_name}")
            except Exception as e:
                print(f"  [!] Failed to load {module_name}: {e}")

# ==========================================
# UI LAYOUT & APPLICATION LOGIC
# ==========================================

class PipelineApp:
    def __init__(self):
        self.root_node = NodeRegistry.get("Input Gateway")()
        self.container = None
        self.svg_container = None

        self.is_running = False

        # --- CANVAS STATE ---
        self.pan_x = 0
        self.pan_y = 0
        self.zoom = 1.0
        self.is_panning = False
        
        self.active_node = None
        self.view_box = None

        self.root_node.x = 400
        self.root_node.y = 300
    
    def adjust_zoom(self, delta):
        """Helper for on-screen zoom buttons."""
        self.zoom = max(0.2, min(3.0, self.zoom + delta))
        self._update_viewport()


    def add_node_dialog(self, target_node: Optional[Node] = None, mode: str = 'output'):
        with ui.dialog() as dialog, ui.card().classes('w-80 p-4'):
            ui.label(f"Add {mode.capitalize()} Node").classes('text-lg font-bold mb-4')
            
            for name, cls in NodeRegistry.get_all().items():
                if getattr(cls, 'is_gateway', False): continue # Prevent adding a 2nd gateway
                
                show = False
                if mode == 'input' and not cls.has_input:  # Show sources
                    show = True
                elif mode == 'output' and cls.has_input:   # Show processors
                    show = True
                
                if show:
                    ui.button(name, on_click=lambda n=name: self.create_node(n, target_node, dialog, mode))\
                        .props('outline').classes('w-full mb-2')
            
            ui.button('Close', on_click=dialog.close).props('flat').classes('w-full mt-2')
        dialog.open()

    def create_node(self, name, target_node, dialog, mode='output'):
        node = NodeRegistry.get(name)()
        
        if mode == 'output' and target_node:
            node.x = target_node.x + getattr(target_node, 'width', 280) + 50
            node.y = target_node.y
            target_node.add_subscriber(node)
        elif mode == 'input' and target_node:
            # Spawn to the LEFT of the gateway
            node.x = target_node.x - getattr(node, 'width', 280) - 50
            node.y = target_node.y
            if hasattr(target_node, 'add_input_node'):
                target_node.add_input_node(node)
            
        dialog.close()
        self.refresh_ui()

    def delete_node(self, node):
        if not getattr(node, 'deletable', True): 
            ui.notify("The Input Gateway cannot be deleted.", color='warning')
            return
            
        if node.parent: node.parent.remove_subscriber(node)
        
        # Disconnect from Gateway if it was an input
        if hasattr(self.root_node, 'input_nodes') and node in self.root_node.input_nodes:
            self.root_node.remove_input_node(node)
            
        node.stop()
        self.refresh_ui()

    def toggle_pipeline(self, button):
        label = next((c for c in button.default_slot.children if isinstance(c, ui.label)), None)

        if self.is_running:
            # ... STOP PIPELINE LOGIC (unchanged) ...
            if self.root_node: self.root_node.stop()
            self.is_running = False
            if label: label.set_text('START PIPELINE')
            button.props('icon=play_arrow color=emerald')
        else:
            # === THE NEW PRE-FLIGHT CHECK ===
            if self.root_node:
                # 1. Ask the root node to generate the master template
                # (Assuming your root is the new Rendezvous/Gateway node)
                master_template = self.root_node.generate_template()
                
                # 2. Force-feed the template down the tree
                is_valid = self.root_node.push_schema(master_template)
                
                if not is_valid:
                    ui.notify("Pipeline compilation failed! Check node configurations.", color='negative', position='top')
                    return # ABORT STARTUP!

                # 3. If everything is valid, start the pipeline
                self.root_node.start()
                
            self.is_running = True
            if label: label.set_text('STOP PIPELINE')
            button.props('icon=stop color=red')

            
    # ==========================================
    # THE CANVAS RENDERING ENGINE
    # ==========================================
    def refresh_ui(self):
        if not self.container: return
        self.container.clear()
        
        with self.container:
            # Add a grid background and use touch-action: none to STOP the browser from scrolling
            grid_style = (
                'background-size: 40px 40px; '
                'background-image: radial-gradient(circle, #cbd5e1 1px, transparent 1px); '
                'touch-action: none;' 
            )
            outer = ui.element('div').classes('relative w-full h-[85vh] bg-slate-200 overflow-hidden select-none border-2') \
                .style(grid_style)
                
            with outer:
                self.viewport = ui.element('div').classes('absolute w-full h-full').style(
                    f'transform-origin: 0 0; transform: translate({self.pan_x}px, {self.pan_y}px) scale({self.zoom});'
                )
                with self.viewport:
                    self.svg_container = ui.html('<svg width="100%" height="100%" style="overflow: visible;"></svg>', sanitize=False).classes('absolute top-0 left-0 w-full h-full pointer-events-none z-0')
                    
                    if not self.root_node:
                        # Added pointerdown.stop so tapping the button works and doesn't pan
                        ui.button('ADD SOURCE NODE', on_click=lambda: self.add_node_dialog(None)) \
                            .classes('absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 shadow-lg') \
                            .props('icon=add color=green') \
                            .on('pointerdown.stop', lambda: None)
                    else:
                        self._render_nodes(self.root_node, is_mini=False)
                        ui.timer(0.05, self.redraw_wires, once=True)
                
                if self.root_node: self._render_minimap()

                # On-Screen View Controls (Added pointerdown.stop to protect them)
                with ui.row().classes('absolute bottom-4 right-4 z-50 gap-2').on('pointerdown.stop', lambda: None):
                    with ui.button(on_click=lambda: self.adjust_zoom(-0.2)).props('flat round size=lg').classes('bg-white/50 backdrop-blur-md shadow-sm border border-slate-300'):
                        ui.icon('remove').classes('text-slate-800')
                    with ui.button(on_click=lambda: self.adjust_zoom(0.2)).props('flat round size=lg').classes('bg-white/50 backdrop-blur-md shadow-sm border border-slate-300'):
                        ui.icon('add').classes('text-slate-800')
                    with ui.button(on_click=self.reset_view).props('flat round size=lg').classes('bg-white/50 backdrop-blur-md shadow-sm border border-slate-300'):
                        ui.icon('center_focus_strong').classes('text-slate-800')
                        ui.tooltip('Reset View')
            
            # --- CENTRALIZED EVENT MANAGEMENT ---
            def get_coords(e):
                return e.args.get('clientX', 0), e.args.get('clientY', 0)

            def handle_move(e):
                mx, my = get_coords(e)
                
                if self.active_node:
                    self.active_node.x = (mx - self.pan_x) / self.zoom - self.active_node.offset_x
                    self.active_node.y = (my - self.pan_y) / self.zoom - self.active_node.offset_y
                    self.active_node.card.style(f'left: {self.active_node.x}px; top: {self.active_node.y}px;')
                    if self.active_node.mini_card:
                        self.active_node.mini_card.style(f'left: {self.active_node.x}px; top: {self.active_node.y}px; width: {self.active_node.width}px; height: {self.active_node.height}px;')
                    self.redraw_wires()
                elif self.is_panning:
                    self.pan_x += (mx - self.last_mouse_x)
                    self.pan_y += (my - self.last_mouse_y)
                    self.last_mouse_x, self.last_mouse_y = mx, my
                    self._update_viewport()

            def handle_bg_mousedown(e):
                self.is_panning = True
                self.last_mouse_x, self.last_mouse_y = get_coords(e)

            def handle_stop(e=None):
                self.is_panning = False
                if self.active_node:
                    self.active_node.card.style('z-index: 10;')
                    self.active_node = None
            
            def handle_wheel(e):
                self.adjust_zoom(0.05 if e.args.get('deltaY', 0) < 0 else -0.05)

            # Bind Canvas using Modern Pointer Events
            pointer_args = ['clientX', 'clientY']

            outer.on('pointerdown', handle_bg_mousedown, args=pointer_args)
            outer.on('pointermove', handle_move, args=pointer_args)
            outer.on('pointerup', handle_stop)
            outer.on('pointerleave', handle_stop)
            outer.on('pointercancel', handle_stop)
            outer.on('wheel', handle_wheel, args=['deltaY'], throttle=0.01)

    # --- MINIMAP RENDER FIX ---
    def _render_minimap(self):
        """Creates a birds-eye view that tracks movements."""
        # Added pointerdown.stop so tapping the minimap doesn't pan the canvas behind it
        with ui.card().classes('absolute top-4 right-4 w-40 h-40 p-0 overflow-hidden bg-slate-800/90 border border-slate-600 shadow-xl z-50').on('pointerdown.stop', lambda: None):
            with ui.element('div').classes('relative w-full h-full').style('transform: scale(0.05) translate(100px, 100px); transform-origin: 0 0;'):
                self._render_nodes(self.root_node, is_mini=True)

    def _update_viewport(self):
        """Applies the current pan and zoom to the UI."""
        self.viewport.style(f'transform: translate({self.pan_x}px, {self.pan_y}px) scale({self.zoom});')

    def _render_nodes(self, node: Node, is_mini: bool = False, visited=None):
        if visited is None: visited = set()
        if node.id in visited: return
        visited.add(node.id)
        
        node.build_node_ui(is_mini=is_mini)
        for sub in node.subscribers:
            self._render_nodes(sub, is_mini=is_mini, visited=visited)
            
        # Traverse left side
        if hasattr(node, 'input_nodes'):
            for inp in node.input_nodes:
                self._render_nodes(inp, is_mini=is_mini, visited=visited)

    def redraw_wires(self):
        if not self.root_node or not self.svg_container: return
            
        svg_content = '<svg width="100%" height="100%" style="overflow: visible;">'
        
        def draw_branch(node, visited=None):
            nonlocal svg_content
            if visited is None: visited = set()
            if node.id in visited: return
            visited.add(node.id)
            
            parent_w = getattr(node, 'width', 280)
            parent_h = getattr(node, 'height', 100)
            start_x = node.x + parent_w
            start_y = node.y + (parent_h / 2)

            # Draw Output Wires (Right side)
            for child in node.subscribers:
                child_h = getattr(child, 'height', 100)
                end_x = child.x
                end_y = child.y + (child_h / 2)

                ctrl_1_x = start_x + max(50, (end_x - start_x) / 2)
                ctrl_1_y = start_y
                ctrl_2_x = end_x - max(50, (end_x - start_x) / 2)
                ctrl_2_y = end_y

                path = f'<path d="M {start_x} {start_y} C {ctrl_1_x} {ctrl_1_y}, {ctrl_2_x} {ctrl_2_y}, {end_x} {end_y}" fill="none" stroke="#94a3b8" stroke-width="3" />'
                svg_content += path
                draw_branch(child, visited)
                
            # Draw Input Wires (Left side feeding INTO gateway)
            if hasattr(node, 'input_nodes'):
                for inp in node.input_nodes:
                    inp_w = getattr(inp, 'width', 280)
                    inp_h = getattr(inp, 'height', 100)
                    i_start_x = inp.x + inp_w
                    i_start_y = inp.y + (inp_h / 2)
                    
                    i_end_x = node.x
                    i_end_y = node.y + (parent_h / 2)
                    
                    ctrl_1_x = i_start_x + max(50, (i_end_x - i_start_x) / 2)
                    ctrl_1_y = i_start_y
                    ctrl_2_x = i_end_x - max(50, (i_end_x - i_start_x) / 2)
                    ctrl_2_y = i_end_y
                    
                    path = f'<path d="M {i_start_x} {i_start_y} C {ctrl_1_x} {ctrl_1_y}, {ctrl_2_x} {ctrl_2_y}, {i_end_x} {i_end_y}" fill="none" stroke="#94a3b8" stroke-width="3" />'
                    svg_content += path
                    draw_branch(inp, visited)
                
        draw_branch(self.root_node)
        svg_content += '</svg>'
        self.svg_container.set_content(svg_content)
        
    def reset_view(self):
        """Snaps the canvas back to origin."""
        self.pan_x = 0
        self.pan_y = 0
        self.zoom = 1.0
        self._update_viewport()
        ui.notify("View Reset")

    def _render_minimap(self):
        """Creates a birds-eye view that tracks movements."""
        with ui.card().classes('absolute top-4 right-4 w-40 h-40 p-0 overflow-hidden bg-slate-800/90 border border-slate-600 shadow-xl z-50'):
            # The minimap dot scale (0.1) should match the scale of the world
            # We add a small offset so (0,0) isn't hugged against the top-left corner
            with ui.element('div').classes('relative w-full h-full').style('transform: scale(0.05) translate(100px, 100px); transform-origin: 0 0;'):
                self._render_nodes(self.root_node, is_mini=True)

    # ==========================================
    # PERSISTENCE (Export / Import)
    # ==========================================

    def export_pipeline(self):
        if not self.root_node:
            ui.notify("Nothing to export!", color='warning')
            return
        
        # Wrap the node tree and viewport state together
        full_config = {
            "viewport": {
                "pan_x": self.pan_x,
                "pan_y": self.pan_y,
                "zoom": self.zoom
            },
            "root": self.root_node.save()
        }
        content = json.dumps(full_config, indent=2)
        ui.download(content.encode(), "full_pipeline_config.json")
        ui.notify("Entire pipeline exported.")

    async def apply_pipeline_config(self, event, dialog):
        try:
            binary_data = await event.file.read()
            data = json.loads(binary_data.decode('utf-8'))
            
            # Handle legacy files (no 'root' key) vs new files
            pipeline_data = data.get("root", data)
            viewport_data = data.get("viewport", {})

            if self.root_node: self.root_node.stop()
            self.root_node = Node.load(pipeline_data)
            
            # Restore Viewport
            self.pan_x = viewport_data.get("pan_x", 0)
            self.pan_y = viewport_data.get("pan_y", 0)
            self.zoom = viewport_data.get("zoom", 1.0)

            dialog.close()
            self.refresh_ui()
            ui.notify("Pipeline and Viewport restored!", color='green')
        except Exception as e:
            ui.notify(f"Failed to import pipeline: {e}", color='red')

    def import_pipeline(self):
        with ui.dialog() as dialog, ui.card().classes('p-4'):
            ui.label('Import Full Pipeline').classes('text-lg font-bold')
            ui.upload(on_upload=lambda e: self.apply_pipeline_config(e, dialog), 
                    auto_upload=True, label="Select Pipeline JSON").classes('w-full')
        dialog.open()

@ui.page('/')
def main():
    if not hasattr(app, 'logic'):
        auto_import_nodes()
        app.logic = PipelineApp()

    ui.add_head_html('<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">')
    
    with ui.header().classes('bg-slate-900 shadow-md flex justify-between items-center px-4'):
        ui.label('EyeAria').classes('font-mono text-xl font-bold')
        
        with ui.row().classes('items-center gap-2'):
            # Hidden text on mobile, visible on medium screens and up
            ui.button(icon='file_download', on_click=app.logic.export_pipeline)\
                .props('flat color=white size=sm')\
                .classes('px-2 md:px-4')\
                .tooltip('Export Pipeline')
            
            ui.button(icon='file_upload', on_click=app.logic.import_pipeline)\
                .props('flat color=white size=sm')\
                .classes('px-2 md:px-4')\
                .tooltip('Import Pipeline')
            
            ui.separator().props('vertical').classes('mx-1 md:mx-2 bg-slate-700 h-6')
            
            # The play button needs dynamic text, so we use a child label to control visibility
            btn = ui.button(on_click=lambda e: app.logic.toggle_pipeline(e.sender)) \
                .props('icon=play_arrow color=emerald unelevated').classes('font-bold shadow-sm px-2 md:px-4')
            with btn:
                ui.label('START PIPELINE').classes('hidden md:block ml-2')

            ui.separator().props('vertical').classes('mx-1 md:mx-2 bg-slate-700 h-6')
            
            # The Magical Kill Switch
            kill_btn = ui.button(on_click=app.shutdown) \
                .props('icon=power_settings_new color=red-8 unelevated').classes('font-bold shadow-sm px-2 md:px-4')
            with kill_btn:
                ui.label('POWER WORD: KILL').classes('hidden md:block ml-2')

    
    # Render the Canvas Container
    with ui.column().classes('p-0 w-full h-full m-0') as container:
        app.logic.container = container
        app.logic.refresh_ui()

ui.run(port=8082, title="EyeAria")