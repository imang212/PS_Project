from dataclasses import dataclass, field, asdict
import os
import threading
import queue
import json
import time
import uuid
import glob
import subprocess
import re
import copy
from nicegui import ui
from Node import Node, NodeRegistry
from typing import List, Dict, Any, Optional
from Schema import ModuleData

# Import updated components from HailoPipeline
try:
    from HailoPipeline import (
        HailoPipeline, HailoListener, AppSink,
        FileSource, RTSPSource, CameraSource,  # <--- Added here
        HailoTracker, LetterboxAdapter, HailoInference
    )
except ImportError:
    print("Warning: HailoPipeline dependencies not found. Using mocks.")
    class HailoListener: pass
    class HailoPipeline: 
        def __init__(self, *args, **kwargs): pass
        def add_listener(self, listener): pass
        def run(self): pass
        def stop(self): pass
        def get_raw_frame(self): return None

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class HailoDetection:
    """1:1 mapping for the objects inside the 'detections' list."""
    label: str
    confidence: float
    bbox: List[float]  # [xmin, ymin, xmax, ymax]
    track_id: int = -1
    tags: List[str] = field(default_factory=list)

@dataclass
class HailoPayload:
    """1:1 mapping for the root JSON payload."""
    timestamp: float
    config: Dict[str, Any]
    count: int
    model_name: str = ""
    pi_uuid: str = ""
    camera_url: str = ""
    detections: List[HailoDetection] = field(default_factory=list)
    global_tags: List[str] = field(default_factory=list)

    def copy(self) -> 'PipelinePayload':
        """
        Creates a deep copy of the payload. 
        Crucial for branching outputs so nodes don't mutate each other's data.
        The frame is passed by reference to save memory and CPU.
        """
        # Temporarily detach the frame to avoid deepcopying the heavy numpy array
        frame_ref = self.frame
        self.frame = None
        
        # Deepcopy the rest of the lightweight metadata
        cloned = copy.deepcopy(self)
        
        # Restore the frame reference to both the original and the clone
        self.frame = frame_ref
        cloned.frame = frame_ref
        
        return cloned

    def to_json(self, indent: int = None) -> str:
        """Serializes the strictly-typed object back into a JSON string."""
        d = asdict(self)
        # Drop the frame so it is completely ignored by text logs and network sinks
        d.pop('frame', None) 
        return json.dumps(d, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'PipelinePayload':
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_dict(cls, data: dict) -> 'PipelinePayload':
        detections_list = [HailoDetection(**det) for det in data.get("detections", [])]
        config_data = data.get("config", {})
        
        return cls(
            timestamp=data.get("timestamp", 0.0),
            config=config_data,
            count=data.get("count", len(detections_list)),
            # Load explicitly from root dict, or fallback to config dictionary
            model_name=data.get("model_name", config_data.get("model_name", "")),
            pi_uuid=data.get("pi_uuid", config_data.get("pi_uuid", "")),
            camera_url=data.get("camera_url", config_data.get("camera_url", "")),
            detections=detections_list,
            frame=data.get("frame", None)
        )

@NodeRegistry.register("Hailo")
class HailoNode(Node, HailoListener): # Inherit from Listener
    node_color = "blue-900"
    has_input = False  
    has_output = True

    _scan_cache = {'hef': [], 'so': []}
    _is_scanning = False
    _scan_complete = False

    def __init__(self):
        super().__init__()
        # Basic Configuration
        self.hef_path = "yolov8s_h8l.hef"
        self.so_path = "libyolo_hailortpp_post.so"

        self.model_name = "YOLOv8"
        self.camera_url = "0"
        self.pi_uuid = self._get_device_uuid()
        
        # Tracking Configuration (Object Permanence)
        self.tracking_enabled = False
        self.keep_tracked_frames = 30
        self.keep_lost_frames = 10
        
        # Runtime internals
        self.pipeline = None
        self.pipeline_thread = None
        self.data_queue = queue.Queue()
        self.poll_timer = None
        
        # Basic Configuration
        self.source_type = "Camera"  # Can be "Camera" or "Stream/File"
        self.source_path = "detection0.mp4"
        self.camera_index = 0
        self.camera_resolution_str = "640,480,30" # Format: width,height,fps
        
        # Camera Transforms
        self.rotation = 0
        self.flip_h = False
        self.flip_v = False
    
    def get_available_cameras(self):
        """Scans /dev/video* and returns a dictionary for the UI dropdown."""
        cams = glob.glob('/dev/video*')
        indices = []
        for cam in cams:
            try:
                # Extract the numeric index from the path
                num = int(cam.replace('/dev/video', ''))
                indices.append(num)
            except ValueError:
                pass
        
        indices.sort()
        if not indices:
            return {0: '0 (No Camera Detected)'}
        
        # Format for the UI Select: {0: 'Camera 0', 1: 'Camera 1', ...}
        return {i: f"Camera {i}" for i in indices}
    
    def get_real_cameras_and_formats(self):
        """Uses rpicam/libcamera to find Pi Camera modules and their exact physical modes."""
        cameras = {}
        formats = {}

        try:
            # Try Bookworm's rpicam first, fallback to Bullseye's libcamera
            try:
                result = subprocess.run(['rpicam-hello', '--list-cameras'], capture_output=True, text=True, timeout=2.0)
            except FileNotFoundError:
                result = subprocess.run(['libcamera-hello', '--list-cameras'], capture_output=True, text=True, timeout=2.0)
            
            lines = result.stdout.split('\n')
            current_cam_idx = None
            
            for line in lines:
                line = line.strip()
                
                # Match camera definition: "0 : imx708 [4608x2592 10-bit RGGB]"
                cam_match = re.match(r'^(\d+)\s*:\s*(\w+)', line)
                if cam_match:
                    current_cam_idx = int(cam_match.group(1))
                    cam_name = cam_match.group(2)
                    cameras[current_cam_idx] = f"Camera {current_cam_idx} ({cam_name})"
                    formats[current_cam_idx] = {}
                    
                    # Pre-load standard scaled resolutions. The Pi ISP can scale the sensor natively.
                    standard_modes = [("1920", "1080", "30"), ("1280", "720", "30"), ("640", "480", "30")]
                    for w, h, fps in standard_modes:
                        formats[current_cam_idx][f"{w},{h},{fps}"] = f"{w}x{h} @ {fps}fps (Scaled)"
                
                # Match physical sensor modes: "1536x864 [120.13 fps - (0, 0)/0x0 crop]"
                mode_match = re.search(r'(\d+)x(\d+)\s*\[([\d\.]+)\s*fps', line)
                if mode_match and current_cam_idx is not None:
                    w = mode_match.group(1)
                    h = mode_match.group(2)
                    fps = str(int(float(mode_match.group(3))))
                    
                    val_str = f"{w},{h},{fps}"
                    lbl_str = f"{w}x{h} @ {fps}fps (Native)"
                    
                    # Overwrite if it already exists to prefer the "Native" label
                    formats[current_cam_idx][val_str] = lbl_str

        except Exception as e:
            print(f"Error scanning libcamera: {e}")
                
        if not cameras:
            cameras = {0: "Camera 0 (Fallback)"}
            formats = {0: {"640,480,30": "640x480 @ 30fps"}}
            
        return cameras, formats
        
    def trigger_system_scan(self, force=False):
        if self.__class__._is_scanning: return
        if self.__class__._scan_complete and not force: return
            
        self.__class__._is_scanning = True
        
        if force:
            self.__class__._scan_cache = {'hef': [], 'so': []}
            self.__class__._scan_complete = False
        
        # Launch thread without touching the UI
        threading.Thread(target=self._run_scan_worker, daemon=True).start()

    def _run_scan_worker(self):
        exclude_dirs = {'proc', 'sys', 'dev', 'run', 'tmp', 'boot', 'snap', 'var'}
        try:
            for root, dirs, files in os.walk('/'):
                dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
                
                for f in files:
                    if f.endswith('.hef'): 
                        self.__class__._scan_cache['hef'].append(os.path.join(root, f))
                    elif f.endswith('.so'): 
                        self.__class__._scan_cache['so'].append(os.path.join(root, f))
        except Exception as e:
            print(f"File scan interrupted: {e}")
            
        # Purely update state. The dialog's timer will pick this up automatically.
        self.__class__._is_scanning = False
        self.__class__._scan_complete = True

    def _update_scan_ui(self):
        if hasattr(self, 'scan_spinner'): self.scan_spinner.set_visibility(False)
        if hasattr(self, 'scan_btn'): self.scan_btn.props('disable=false')

    def _get_device_uuid(self):
        """Reads the hardware UUID from a file, or generates one if it's the first boot."""
        uuid_file = "device_uuid.txt"
        if os.path.exists(uuid_file):
            with open(uuid_file, "r") as f:
                return f.read().strip()
        else:
            new_uuid = f"pi-{str(uuid.uuid4())[:8]}" # Creates something like pi-a1b2c3d4
            with open(uuid_file, "w") as f:
                f.write(new_uuid)
            return new_uuid

    def on_data_received(self, frame, raw_detections):
        import re 
        
        schema_detections = [HailoDetection(**d) for d in raw_detections]
        derived_model_name = os.path.splitext(os.path.basename(self.hef_path))[0]
        
        if self.source_type == 'Camera':
            active_source_name = f"Camera {self.camera_index} ({self.camera_resolution_str})"
        else:
            active_source_name = re.sub(r'(://)([^:]+):([^@]+)(@)', r'\1***:***\4', self.source_path)
            
        # 1. Package ONLY the sensor's specific data into a dictionary
        sensor_data = {
            "config": {"source": active_source_name, "hef": self.hef_path, "tracking": self.tracking_enabled},
            "count": len(schema_detections),
            "model_name": derived_model_name,
            "pi_uuid": self.pi_uuid,
            "camera_url": active_source_name,
            "detections": schema_detections,
            "frame": frame
        }
        
        # 2. Wrap it in the ModuleData contract
        module_key = f"{self._node_type_name}_{self.id[:6]}"
        module_chunk = ModuleData(
            name=module_key,
            is_new=True,
            data=sensor_data
        )
        
        # 3. Thread-Safe Handshake: Put the chunk in the queue, DO NOT call self.notify() here!
        self.data_queue.put({module_key: module_chunk})

    def check_queue(self):
        """Runs on the UI thread, safely popping data and pushing to the Gateway."""
        try:
            while not self.data_queue.empty():
                chunk = self.data_queue.get_nowait() # chunk is dict: { "Hailo_abc123": ModuleData(...) }
                
                module_key = list(chunk.keys())[0]
                mod_data = chunk[module_key]
                
                # Update UI
                if hasattr(self, 'status_label'):
                    count = mod_data.data.get('count', 0)
                    self.status_label.set_text(f"Detections: {count} (Tracked: {self.tracking_enabled})")
                
                # Push the module chunk to the InputGateway
                self.notify(chunk) 
                
        except queue.Empty:
            pass

    def get_frame(self):
        """Retrieves the raw camera frame directly from the pipeline tap."""
        if self.pipeline:
            return self.pipeline.get_raw_frame()
        return None

    def _start(self):
        # 1. Setup Source based on input type
        if self.source_type == 'Camera':
            try:
                w, h, fps = map(int, self.camera_resolution_str.split(','))
            except:
                w, h, fps = 640, 480, 30
                
            source = CameraSource(
                device_index=self.camera_index,
                width=w,
                height=h,
                fps=fps,
                rotation=self.rotation, 
                flip_h=self.flip_h, 
                flip_v=self.flip_v
            )
        else:
            is_rtsp = any(self.source_path.startswith(prefix) for prefix in ["rtsp://", "rtmp://", "http://"])
            if is_rtsp:
                source = RTSPSource(self.source_path, rotation=self.rotation, flip_h=self.flip_h, flip_v=self.flip_v)
            else:
                source = FileSource(self.source_path, rotation=self.rotation, flip_h=self.flip_h, flip_v=self.flip_v)

        # 2. Setup Inference Components
        adapter = LetterboxAdapter() 
        inference = HailoInference(self.hef_path, self.so_path)
        
        # 3. Setup Optional Tracker
        tracker = None
        if self.tracking_enabled:
            tracker = HailoTracker(
                keep_tracked_frames=self.keep_tracked_frames,
                keep_lost_frames=self.keep_lost_frames
            )

        # 4. Initialize Modular Sink and Pipeline
        sink = AppSink(include_frame=True)
        self.pipeline = HailoPipeline(source, adapter, inference, sink, tracker)
        self.pipeline.add_listener(self) # The node listens to its own pipeline

        # 5. Launch
        self.pipeline_thread = threading.Thread(target=self.pipeline.run, daemon=True)
        self.pipeline_thread.start()
        self.poll_timer = ui.timer(0.1, self.check_queue)

    def _stop(self):
        if self.poll_timer: self.poll_timer.cancel()
        
        if self.pipeline:
            # Use the new stop method to ensure State.NULL is set
            self.pipeline.stop()
            
            # Wait briefly for the thread to finish cleanly
            if self.pipeline_thread and self.pipeline_thread.is_alive():
                self.pipeline_thread.join(timeout=1.0)
                
            self.pipeline = None

    def create_content(self):
        # 1. Source Config Panel
        with ui.column().classes('bg-emerald-50 border border-slate-200 border-l-4 border-l-emerald-500 w-full p-2 mb-2 shadow-sm gap-1'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label("SOURCE CONFIG").classes('text-[10px] font-bold text-emerald-800')
                ui.radio(['Camera', 'Stream/File'], value=self.source_type).bind_value(self, 'source_type').props('inline dense size=sm')

            # --- CAMERA UI ---
            with ui.column().bind_visibility_from(self, 'source_type', lambda t: t == 'Camera').classes('w-full gap-1'):
                
                # Fetch true cameras dynamically
                cameras, formats = self.get_real_cameras_and_formats()
                
                def update_resolutions(e):
                    # Prevent the error if the event fires before self.res_select is created
                    if not hasattr(self, 'res_select'):
                        return
                        
                    cam_idx = e.value
                    # Update the options dictionary of the resolution dropdown
                    self.res_select.options = formats.get(cam_idx, {"640,480,30": "640x480 @ 30fps"})
                    # Auto-select the first available resolution
                    self.res_select.value = list(self.res_select.options.keys())[0] if self.res_select.options else None
                    self.res_select.update()

                cam_select = ui.select(cameras, label="Select Camera Device", on_change=update_resolutions) \
                    .bind_value(self, 'camera_index').classes('w-full text-xs').props('dense')
                
                current_formats = formats.get(self.camera_index, {"640,480,30": "640x480 @ 30fps"})
                
                # Attach to 'self' to bypass the local closure limitation
                self.res_select = ui.select(current_formats, label="Native Resolution & FPS") \
                    .bind_value(self, 'camera_resolution_str').classes('w-full text-xs').props('dense')

            # --- STREAM/FILE UI ---
            with ui.column().bind_visibility_from(self, 'source_type', lambda t: t == 'Stream/File').classes('w-full gap-1'):
                ui.input(label="Source URL/Path (RTSP, MP4)").bind_value(self, 'source_path').classes('w-full text-xs').props('dense')

            # --- SHARED TRANSFORMS ---
            with ui.row().classes('w-full items-center gap-2 no-wrap mt-1'):
                ui.select({0: '0°', 90: '90°', 180: '180°', 270: '270°'}, label="Rotation") \
                    .bind_value(self, 'rotation').classes('w-16 text-xs').props('dense')
                ui.checkbox("Flip H").bind_value(self, 'flip_h').classes('text-[10px] text-slate-700').props('dense size=sm')
                ui.checkbox("Flip V").bind_value(self, 'flip_v').classes('text-[10px] text-slate-700').props('dense size=sm')

        # 2. Hailo Config Panel
        with ui.column().classes('bg-slate-50 border border-slate-200 border-l-4 border-l-slate-500 w-full p-2 mb-2 shadow-sm gap-1'):
            ui.label("HAILO CONFIG").classes('text-[10px] font-bold text-slate-700')
            
            # HEF Search Row
            with ui.row().classes('w-full items-center gap-1 no-wrap'):
                ui.input(label="HEF File").bind_value(self, 'hef_path').classes('grow text-xs').props('dense')
                ui.button(icon='search', on_click=lambda: self.show_search_dialog('hef_path', 'hef')) \
                    .props('flat round dense color=blue size=sm')

            # SO Search Row
            with ui.row().classes('w-full items-center gap-1 no-wrap'):
                ui.input(label="Post-Proc (.so)").bind_value(self, 'so_path').classes('grow text-xs').props('dense')
                ui.button(icon='search', on_click=lambda: self.show_search_dialog('so_path', 'so')) \
                    .props('flat round dense color=blue size=sm')
            
            # Start the background scan silently on boot
            self.trigger_system_scan()
            
        # 3. Object Permanence Panel
        with ui.column().classes('bg-blue-50 border border-slate-200 border-l-4 border-l-blue-500 w-full p-2 mb-2 shadow-sm gap-1'):
            ui.label("OBJECT PERMANENCE").classes('text-[10px] font-bold text-blue-700')
            
            ui.checkbox("Enable Tracker").bind_value(self, 'tracking_enabled').classes('text-[10px] font-bold text-slate-700').props('dense size=sm')
            
            with ui.column().bind_visibility_from(self, 'tracking_enabled').classes('w-full gap-1 mt-1'):
                ui.number(label="Keep Tracked", format="%d").bind_value(self, 'keep_tracked_frames').classes('w-full text-xs').props('dense')
                ui.number(label="Keep Lost", format="%d").bind_value(self, 'keep_lost_frames').classes('w-full text-xs').props('dense')
                ui.label("Higher values keep IDs longer.").classes('text-[9px] text-slate-500 italic leading-tight')
            
        # 4. Status Display
        with ui.row().classes('w-full items-center justify-between px-2'):
            self.status_label = ui.label("Status: Idle").classes("text-[10px] text-slate-500 font-mono")


    def show_search_dialog(self, target_attr, file_type):
        """Displays a live, searchable dialog for cached files."""
        with ui.dialog() as dialog, ui.card().classes('w-[32rem] h-[32rem] p-4'):
            # Force a strict vertical column for the whole card
            with ui.column().classes('w-full h-full gap-2 no-wrap'):
                with ui.row().classes('w-full items-center justify-between'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label(f"SEARCH {file_type.upper()} FILES").classes('text-[12px] font-bold text-slate-700')
                        # Refresh button moved here
                        ui.button(icon='refresh', on_click=lambda: (self.trigger_system_scan(force=True), update_results())) \
                            .props('flat round dense size=xs color=slate').tooltip('Force Re-scan')
                            
                    status_label = ui.label().classes('text-[10px] text-blue-500 italic font-bold')
                
                search_input = ui.input('Type to filter by filename or path...', 
                                        on_change=lambda e: update_results(e.value)) \
                    .classes('w-full text-xs').props('clearable dense autofocus')
                
                results_container = ui.column().classes('w-full grow overflow-y-auto gap-0 border border-slate-200 p-1 shadow-inner bg-slate-50')
                
                def update_results(query=None):
                    if query is None: query = search_input.value
                    results_container.clear()
                    query = query.lower() if query else ""
                    
                    # Copy the list to prevent thread-mutation errors while looping
                    all_files = list(self.__class__._scan_cache[file_type])
                    filtered = [f for f in all_files if query in f.lower()]
                    display_files = filtered[:100]
                    
                    with results_container:
                        if not display_files:
                            msg = "Scanning drive..." if self.__class__._is_scanning else "No matches found."
                            ui.label(msg).classes('text-[10px] text-slate-500 italic p-2')
                        
                        for f in display_files:
                            filename = os.path.basename(f)
                            dir_path = os.path.dirname(f)
                            
                            with ui.button(on_click=lambda f=f: select_file(f)) \
                                .props('flat align=left size=sm color=slate no-caps') \
                                .classes('w-full py-1 border-b border-slate-200'):
                                # FIX: The internal column forces the text to stack vertically
                                with ui.column().classes('w-full gap-0 items-start'):
                                    ui.label(filename).classes('font-bold text-blue-600 text-[11px] leading-tight')
                                    ui.label(dir_path).classes('text-[9px] text-slate-400 break-all leading-tight')
                                
                        if len(filtered) > 100:
                            ui.label(f"... and {len(filtered) - 100} more unseen files. Keep typing to refine search.") \
                                .classes('text-[10px] text-orange-500 italic p-2 text-center w-full')

                    # Update top right status text
                    if self.__class__._is_scanning:
                        status_label.set_text(f"Scanning... ({len(all_files)} found)")
                    else:
                        status_label.set_text(f"Scan Complete ({len(all_files)} total)")

                def select_file(f):
                    setattr(self, target_attr, f)
                    dialog.close()

                update_results()
                ui.button('CANCEL', on_click=dialog.close).props('flat color=grey size=xs').classes('w-full mt-auto')

                # Automatically refresh the results if the background scan is still running
                def auto_refresh():
                    if self.__class__._is_scanning:
                        update_results()
                    else:
                        update_results() # Final update
                        refresh_timer.deactivate()
                        
                refresh_timer = ui.timer(1.0, auto_refresh)
                
        dialog.open()

    def show_file_picker(self, target_attr, extensions):
        """Displays a dialog to select files with specific extensions from the current directory."""
        # Scan current directory for matching files
        try:
            available_files = [f for f in os.listdir('.') if os.path.isfile(f) and any(f.endswith(ext) for ext in extensions)]
        except Exception as e:
            available_files = []
            print(f"Error reading directory: {e}")

        with ui.dialog() as dialog, ui.card().classes('p-4 w-72'):
            ui.label(f"SELECT {' / '.join(extensions).upper()}").classes('text-[12px] font-bold text-slate-700 mb-2')
            
            with ui.column().classes('w-full gap-2 max-h-48 overflow-y-auto'):
                if not available_files:
                    ui.label("No matching files found.").classes('text-[10px] text-red-400 italic text-center w-full')
                else:
                    for f in available_files:
                        # Use default argument n=f to capture the loop variable properly
                        ui.button(f, on_click=lambda n=f: (setattr(self, target_attr, n), dialog.close())) \
                            .props('outline color=blue size=sm').classes('w-full text-[10px] truncate')
            
            ui.button('CANCEL', on_click=dialog.close).props('flat color=grey size=xs').classes('w-full mt-2')
            
        dialog.open()

    def save(self) -> dict:
        base = super().save()
        base.update({
            "source_type": self.source_type,
            "source_path": self.source_path,
            "camera_index": self.camera_index,
            "camera_resolution_str": self.camera_resolution_str, # <--- NEW
            "rotation": self.rotation,
            "flip_h": self.flip_h,
            "flip_v": self.flip_v,
            "hef_path": self.hef_path,
            "so_path": self.so_path,
            "tracking_enabled": self.tracking_enabled,
            "keep_tracked_frames": self.keep_tracked_frames,
            "keep_lost_frames": self.keep_lost_frames
        })
        return base

    def _load_config(self, data: dict):
        self.source_type = data.get("source_type", "Camera")
        self.source_path = data.get("source_path", "")
        self.camera_index = data.get("camera_index", 0)
        self.camera_resolution_str = data.get("camera_resolution_str", "640,480,30") # <--- NEW
        self.rotation = data.get("rotation", 0)
        self.flip_h = data.get("flip_h", False)
        self.flip_v = data.get("flip_v", False)
        self.hef_path = data.get("hef_path", "")
        self.so_path = data.get("so_path", "")
        self.tracking_enabled = data.get("tracking_enabled", False)
        self.keep_tracked_frames = data.get("keep_tracked_frames", 30)
        self.keep_lost_frames = data.get("keep_lost_frames", 10)

    def _input(self, data_json): return None 
