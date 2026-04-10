import sys
import os
import subprocess
import argparse
import socket
import concurrent.futures
import ipaddress
import time
import shutil
from typing import List, Optional, Dict
from pathlib import Path
import zipfile
import tempfile
import asyncio
import ast
import re
import base64
import mimetypes
import socket
from urllib.parse import urlparse
import threading
import requests
import io
from starlette.responses import StreamingResponse, Response
from starlette.background import BackgroundTask
from fastapi import Request
from collections import deque
from datetime import datetime
import importlib.util
import runpy
from unittest.mock import patch
import json
import signal

# --- Dependency Management ---
# dev: phW5uVdLuACNXLBK

class DependencyManager:
    """Handles automatic installation and importing of required libraries."""
    @staticmethod
    def auto_install(package: str, import_name: Optional[str] = None):
        import_name = import_name or package
        try:
            return __import__(import_name)
        except ImportError:
            print(f"[*] Library '{package}' not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                return __import__(import_name)
            except Exception as e:
                print(f"[!] Failed to install {package}: {e}")
                sys.exit(1)

# Initialize dependencies through the manager
deps = DependencyManager()
ng = deps.auto_install("nicegui")
psutil = deps.auto_install("psutil")
paramiko = deps.auto_install("paramiko")
fabric = deps.auto_install("fabric")
cv2 = deps.auto_install("opencv-python", import_name="cv2")
mqtt_client = deps.auto_install("paho-mqtt", import_name="paho.mqtt.client")
httpx = deps.auto_install("httpx")
np = deps.auto_install("numpy")
psycopg2 = deps.auto_install("psycopg2-binary", import_name="psycopg2")
import psycopg2.sql as sql
ui, app, run = ng.ui, ng.app, ng.run,

# --- Logic Layer ---

class PluginManager:
    _instance = None
    SKIP_DIRS = {
        'proc', 'sys', 'dev', 'run', 'boot', 'snap', 'var', 'tmp', 
        '__pycache__', '.git', 'site-packages', 'dist-packages', 'node_modules'
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
            cls._instance.mounted_plugins = {} 
            cls._instance.cached_scan = []
        return cls._instance

    def init_storage(self):
        """Called once storage is ready to load previously mounted plugins."""
        if 'mounted_plugins_paths' not in app.storage.user:
            app.storage.user['mounted_plugins_paths'] = []
            
        saved_paths = app.storage.user['mounted_plugins_paths']
        for path_str in saved_paths:
            path = Path(path_str)
            if path.exists():
                self.mount_plugin(path, save=False)

    def scan_for_plugins(self, root_path: Path) -> List[Dict]:
        """Scans for plugins and ENFORCES a @router.page('/index') entry point."""
        plugins = []
        current_script = Path(__file__).name
        
        try:
            for root, dirs, files in os.walk(str(root_path)):
                dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS and not d.startswith('.')]
                
                for file in files:
                    if file.endswith(".py") and file != current_script:
                        full_path = Path(root) / file
                        try:
                            if full_path.stat().st_size > 1_000_000: continue
                            
                            content = full_path.read_text(errors='ignore')
                            if 'APIRouter' not in content: continue
                            
                            try:
                                tree = ast.parse(content)
                            except SyntaxError: continue

                            has_router = False
                            has_index_route = False 
                            has_unsafe_ui = False

                            for node in ast.walk(tree):
                                if isinstance(node, ast.Assign):
                                    for target in node.targets:
                                        if isinstance(target, ast.Name) and target.id == 'router':
                                            has_router = True
                                
                                if isinstance(node, ast.FunctionDef):
                                    for decorator in node.decorator_list:
                                        if isinstance(decorator, ast.Call):
                                            func = decorator.func
                                            
                                            if isinstance(func, ast.Attribute) and func.attr == 'page':
                                                if isinstance(func.value, ast.Name) and func.value.id == 'ui':
                                                    has_unsafe_ui = True
                                            
                                            if isinstance(func, ast.Attribute) and func.attr == 'page':
                                                if isinstance(func.value, ast.Name) and func.value.id == 'router':
                                                    if decorator.args:
                                                        arg = decorator.args[0]
                                                        val = getattr(arg, 'value', getattr(arg, 's', None))
                                                        if val == '/index':
                                                            has_index_route = True

                            if has_router and has_index_route and not has_unsafe_ui:
                                plugins.append({
                                    'name': full_path.stem.replace('_', ' ').title(),
                                    'filename': file,
                                    'path': full_path,
                                    'id': full_path.stem
                                })

                        except Exception: continue
        except Exception: pass
        self.cached_scan = plugins
        return plugins

    def mount_plugin(self, plugin_input, save=True) -> bool:
        """
        Mounts a plugin. 
        Accepts EITHER a dictionary (from UI scan) OR a Path object (from storage/auto-load).
        """
        # FIX: Handle both input types
        if isinstance(plugin_input, dict):
            path = plugin_input['path']
        elif isinstance(plugin_input, (str, Path)):
            path = Path(plugin_input)
        else:
            print(f"Invalid plugin input: {type(plugin_input)}")
            return False

        p_id = path.stem
        if p_id in self.mounted_plugins: return True

        try:
            spec = importlib.util.spec_from_file_location(f"plugin_{p_id}", str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'router'):
                prefix = f"/plugins/{p_id}"
                app.include_router(module.router, prefix=prefix)
                
                # Extract Routes
                routes_list = []
                try:
                    for route in module.router.routes:
                        rel_path = route.path
                        full_path = f"{prefix}{rel_path}"
                        routes_list.append({
                            'path': full_path,
                            'name': rel_path if rel_path != '/' else '/root'
                        })
                    routes_list.sort(key=lambda x: x['path'])
                except: pass

                self.mounted_plugins[p_id] = {
                    'module': module, 
                    'path': prefix, 
                    'routes': routes_list
                }
                
                if save:
                    current = app.storage.user.get('mounted_plugins_paths', [])
                    if str(path) not in current:
                        current.append(str(path))
                        app.storage.user['mounted_plugins_paths'] = current
                return True
            return False
        except Exception as e:
            print(f"Mount Error: {e}")
            return False
    
    def unmount_plugin(self, p_id: str):
        """
        Surgically removes a plugin's routes from the running server and clears it from storage.
        """
        if p_id not in self.mounted_plugins: return

        try:
            plugin_data = self.mounted_plugins[p_id]
            prefix = plugin_data['path'] # e.g. "/plugins/advanced_metrics"

            # 1. Remove Routes from FastAPI/Starlette
            # We filter the app.routes list to exclude anything starting with the plugin's prefix.
            # This effectively makes the endpoints 404 immediately.
            # Note: We modify the list in-place [:] to ensure the server sees the change.
            app.routes[:] = [
                r for r in app.routes 
                if not (hasattr(r, 'path') and r.path.startswith(prefix))
            ]

            # 2. Clean up Storage (Prevent auto-load on next restart)
            if 'mounted_plugins_paths' in app.storage.user:
                current_paths = app.storage.user['mounted_plugins_paths']
                # Filter out the path corresponding to this plugin ID
                new_paths = [p for p in current_paths if Path(p).stem != p_id]
                app.storage.user['mounted_plugins_paths'] = new_paths

            # 3. Remove from Memory
            del self.mounted_plugins[p_id]
            
            # 4. Optional: Clean sys.modules (Garbage Collection hint)
            # This allows the file to be re-imported if the code changes.
            module_key = f"plugin_{p_id}"
            if module_key in sys.modules:
                del sys.modules[module_key]

            return True

        except Exception as e:
            print(f"Unmount Error: {e}")
            return False

    def is_mounted(self, p_id):
        return p_id in self.mounted_plugins

plugin_manager = PluginManager()

class MediaViewer(ui.dialog):
    """Versatile popup for viewing text, images, and videos."""
    def __init__(self, item: Path):
        super().__init__()
        self.item = item
        self.mime = mimetypes.guess_type(str(item))[0] or ""
        self.build()

    def build(self):
        with self, ui.card().classes('bg-[#050a0f] border border-blue-900 p-0 overflow-hidden').style('width: 80vw; max-width: 1000px;'):
            with ui.row().classes('w-full items-center justify-between bg-[#0d1b2a] px-4 py-2 border-b border-blue-900'):
                ui.label(f'VIEWER: {self.item.name}').classes('text-blue-200 font-mono text-xs font-bold truncate')
                ui.button(icon='close', on_click=self.close).props('flat color=white dense size=sm')

            with ui.column().classes('w-full p-4 items-center justify-center').style('min-height: 400px; max-height: 80vh; overflow-y: auto;'):
                if self.is_image():
                    ui.image(self.get_data_url()).classes('rounded shadow-lg').style('max-width: 100%;')
                elif self.is_video():
                    ui.video(self.get_data_url()).classes('w-full shadow-lg')
                elif self.is_text():
                    try:
                        content = self.item.read_text()
                        ui.label(content).classes('font-mono text-[11px] text-gray-300 whitespace-pre-wrap w-full bg-black/40 p-4 rounded border border-gray-800')
                    except Exception as e:
                        ui.label(f"Error reading text: {e}").classes('text-red-500 font-mono')
                else:
                    ui.icon('help_outline', size='4rem', color='blue-900')
                    ui.label("Binary or unsupported format").classes('text-gray-500 font-bold mt-2')

    def is_image(self):
        return self.mime.startswith('image/') or self.item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']

    def is_video(self):
        return self.mime.startswith('video/') or self.item.suffix.lower() in ['.mp4', '.webm', '.ogv']

    def is_text(self):
        return self.mime.startswith('text/') or self.item.suffix.lower() in ['.txt', '.log', '.py', '.sh', '.js', '.json', '.html', '.css', '.md', '.cfg', '.yaml']

    def get_data_url(self):
        try:
            with open(self.item, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('ascii')
                return f"data:{self.mime};base64,{encoded}"
        except Exception:
            return ""

class GlobalCameraManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GlobalCameraManager, cls).__new__(cls)
                cls._instance.streams = {}  # Store active captures here
        return cls._instance

    def get_frame(self, source):
        """Returns a single JPEG encoded frame with thread locking to prevent crashes."""
        with self._lock: # Use the existing class lock
            if source not in self.streams:
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    return None
                self.streams[source] = cap
            
            cap = self.streams[source]
            success, frame = cap.read()
            if not success:
                # If read fails, try to reset the capture
                cap.release()
                del self.streams[source]
                return None
                
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buffer.tobytes()
    
camera_manager = GlobalCameraManager()

class VideoRecorder:
    def __init__(self, source, save_path, duration_secs):
        # Force .mkv for XVID stability on Linux
        self.save_path = str(Path(save_path).with_suffix('.mkv'))
        self.source = source
        self.duration = duration_secs
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._record_loop, daemon=True).start()

    def _record_loop(self):
        import time
        start_time = time.time()
        writer = None
        
        # Ensure directory exists
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            while time.time() - start_time < self.duration:
                # Use Global Manager's existing frame stream
                frame_bytes = camera_manager.get_frame(self.source)
                
                if frame_bytes:
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if writer is None:
                        h, w, _ = frame.shape
                        # XVID + MKV is the most stable combination for Pi OpenCV
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        writer = cv2.VideoWriter(self.save_path, fourcc, 20.0, (w, h))
                        
                        if not writer.isOpened():
                            print(f"[!] Writer failed to open: {self.save_path}")
                            return

                    writer.write(frame)
                
                time.sleep(0.05) # ~20 FPS
        except Exception as e:
            print(f"[!] Record Error: {e}")
        finally:
            if writer:
                writer.release()
            self.running = False
            print(f"[*] Saved to: {self.save_path}")
            
class GPIOManager:
    """Handles GPIO interactions with safety fallback for non-Pi environments."""
    _setup_done = False
    
    # Standard 40-Pin Header Map (Physical Pin -> BCM GPIO)
    PIN_MAP = {
        1:  {'type': 'pwr', 'label': '3.3V', 'bcm': None},
        2:  {'type': 'pwr', 'label': '5V',   'bcm': None},
        3:  {'type': 'gpio', 'label': 'GPIO 2 (SDA)', 'bcm': 2},
        4:  {'type': 'pwr', 'label': '5V',   'bcm': None},
        5:  {'type': 'gpio', 'label': 'GPIO 3 (SCL)', 'bcm': 3},
        6:  {'type': 'gnd', 'label': 'GND',  'bcm': None},
        7:  {'type': 'gpio', 'label': 'GPIO 4', 'bcm': 4},
        8:  {'type': 'gpio', 'label': 'GPIO 14 (TX)', 'bcm': 14},
        9:  {'type': 'gnd', 'label': 'GND',  'bcm': None},
        10: {'type': 'gpio', 'label': 'GPIO 15 (RX)', 'bcm': 15},
        11: {'type': 'gpio', 'label': 'GPIO 17', 'bcm': 17},
        12: {'type': 'gpio', 'label': 'GPIO 18 (PCM)', 'bcm': 18},
        13: {'type': 'gpio', 'label': 'GPIO 27', 'bcm': 27},
        14: {'type': 'gnd', 'label': 'GND',  'bcm': None},
        15: {'type': 'gpio', 'label': 'GPIO 22', 'bcm': 22},
        16: {'type': 'gpio', 'label': 'GPIO 23', 'bcm': 23},
        17: {'type': 'pwr', 'label': '3.3V', 'bcm': None},
        18: {'type': 'gpio', 'label': 'GPIO 24', 'bcm': 24},
        19: {'type': 'gpio', 'label': 'GPIO 10 (MOSI)', 'bcm': 10},
        20: {'type': 'gnd', 'label': 'GND',  'bcm': None},
        21: {'type': 'gpio', 'label': 'GPIO 9 (MISO)', 'bcm': 9},
        22: {'type': 'gpio', 'label': 'GPIO 25', 'bcm': 25},
        23: {'type': 'gpio', 'label': 'GPIO 11 (SCLK)', 'bcm': 11},
        24: {'type': 'gpio', 'label': 'GPIO 8 (CE0)', 'bcm': 8},
        25: {'type': 'gnd', 'label': 'GND',  'bcm': None},
        26: {'type': 'gpio', 'label': 'GPIO 7 (CE1)', 'bcm': 7},
        27: {'type': 'gpio', 'label': 'GPIO 0 (ID_SD)', 'bcm': 0},
        28: {'type': 'gpio', 'label': 'GPIO 1 (ID_SC)', 'bcm': 1},
        29: {'type': 'gpio', 'label': 'GPIO 5', 'bcm': 5},
        30: {'type': 'gnd', 'label': 'GND',  'bcm': None},
        31: {'type': 'gpio', 'label': 'GPIO 6', 'bcm': 6},
        32: {'type': 'gpio', 'label': 'GPIO 12', 'bcm': 12},
        33: {'type': 'gpio', 'label': 'GPIO 13', 'bcm': 13},
        34: {'type': 'gnd', 'label': 'GND',  'bcm': None},
        35: {'type': 'gpio', 'label': 'GPIO 19', 'bcm': 19},
        36: {'type': 'gpio', 'label': 'GPIO 16', 'bcm': 16},
        37: {'type': 'gpio', 'label': 'GPIO 26', 'bcm': 26},
        38: {'type': 'gpio', 'label': 'GPIO 20', 'bcm': 20},
        39: {'type': 'gnd', 'label': 'GND',  'bcm': None},
        40: {'type': 'gpio', 'label': 'GPIO 21', 'bcm': 21},
    }

    def __init__(self):
        self.simulated = False
        try:
            import RPi.GPIO as GPIO
            self.lib = GPIO
            if not GPIOManager._setup_done:
                self.lib.setmode(self.lib.BCM)
                self.lib.setwarnings(False)
                GPIOManager._setup_done = True
        except ImportError:
            print("[!] RPi.GPIO not found. Using Simulation Mode.")
            self.simulated = True
            self.mock_states = {} # {bcm_pin: value}

    def get_pin_status(self, bcm_pin):
        if self.simulated:
            return self.mock_states.get(bcm_pin, 0)
        try:
            return self.lib.input(bcm_pin)
        except: return 0

    def set_pin_mode(self, bcm_pin, mode_str):
        """mode_str: 'IN' or 'OUT'"""
        if self.simulated: return
        
        mode = self.lib.OUT if mode_str == 'OUT' else self.lib.IN
        self.lib.setup(bcm_pin, mode)

    def set_pin_value(self, bcm_pin, value):
        if self.simulated:
            self.mock_states[bcm_pin] = value
        else:
            self.lib.output(bcm_pin, value)

    def cleanup(self):
        if not self.simulated:
            self.lib.cleanup()

gpio_manager = None

class DynamicApiManager:
    """Dynamically loads FastAPI routers from external Python files."""
    
    @staticmethod
    def load_and_mount(file_path: str, prefix: str):
        path = Path(file_path)
        if not path.exists():
            ui.notify(f"File not found: {file_path}", color='red-10')
            return False

        try:
            # 1. Create a unique module name based on the filename
            module_name = f"dynamic_api_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            module = importlib.util.module_from_spec(spec)
            
            # 2. Execute the module
            spec.loader.exec_module(module)

            # 3. Look for a FastAPI or APIRouter instance
            api_instance = None
            for attr in ['router', 'api', 'app']:
                if hasattr(module, attr):
                    api_instance = getattr(module, attr)
                    break
            
            if not api_instance:
                ui.notify(f"No 'router' or 'app' object found in {path.name}", color='amber-7')
                return False

            # 4. Mount to the NiceGUI FastAPI app
            from fastapi import APIRouter, FastAPI
            if isinstance(api_instance, APIRouter):
                app.include_router(api_instance, prefix=f"/api/{prefix}")
            elif isinstance(api_instance, FastAPI):
                app.mount(f"/api/{prefix}", api_instance)
            
            ui.notify(f"API Mounted at /api/{prefix}", color='emerald-9')
            return True

        except Exception as e:
            ui.notify(f"Import Error: {e}", color='red-10')
            return False

class PiAuth:
    """Handles SSH-based authentication logic."""
    @staticmethod
    def verify(username, password) -> bool:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect('127.0.0.1', username=username, password=password, timeout=2)
            ssh.close()
            return True
        except Exception:
            return False

class NetworkScanner:
    """Handles discovery of Raspberry Pis on the network."""
    def __init__(self, user, password, net_str=None):
        self.user = user
        self.password = password
        self.network = self._parse_network(net_str)

    def _parse_network(self, net_str):
        try:
            if not net_str:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    base = ".".join(s.getsockname()[0].split('.')[:3]) + ".0/24"
                    return ipaddress.IPv4Network(base)
            return ipaddress.IPv4Network(net_str, strict=False)
        except Exception as e:
            print(f"[-] Network Error: {e}")
            return None

    def _check_ip(self, ip):
        ip_str = str(ip)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                if s.connect_ex((ip_str, 22)) != 0: return None
            
            with fabric.Connection(ip_str, user=self.user, connect_kwargs={"password": self.password}, connect_timeout=1) as c:
                c.run("echo 1", hide=True)
                return ip_str
        except:
            return None

    def discover(self) -> List[str]:
        if not self.network: return []
        hosts = list(self.network.hosts())
        found = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            results = executor.map(self._check_ip, hosts)
            for res in results:
                if res: found.append(res)
        return found

class RemoteController:
    """Handles remote deployment and system commands via SSH."""
    def __init__(self, ip, user, password):
        self.ip = ip
        self.user = user
        self.password = password
        self.filename = os.path.basename(__file__)
        self.remote_path = f"/home/{user}/{self.filename}"

    def deploy(self):
        with fabric.Connection(self.ip, user=self.user, connect_kwargs={"password": self.password}) as c:
            c.put(__file__, self.remote_path)
            service = (
                f"[Unit]\nDescription=NiceGUI Manager\nAfter=network.target\n\n"
                f"[Service]\nExecStart=/home/imang/venv/bin/python {self.remote_path} --gui\n"
                f"Restart=always\nUser={self.user}\nWorkingDirectory=/home/{self.user}\n\n"
                f"[Install]\nWantedBy=multi-user.target"
            )
            c.run(f"echo '{service}' | sudo tee /etc/systemd/system/pimanager.service", hide=True)
            c.run("sudo systemctl daemon-reload && sudo systemctl enable pimanager.service --now", hide=True)

    def destroy(self):
        with fabric.Connection(self.ip, user=self.user, connect_kwargs={"password": self.password}) as c:
            c.run("sudo systemctl stop pimanager.service", warn=True, hide=True)
            c.run("sudo rm /etc/systemd/system/pimanager.service", warn=True, hide=True)
            c.run(f"rm {self.remote_path}", warn=True, hide=True)

    def reboot(self):
        with fabric.Connection(self.ip, user=self.user, connect_kwargs={"password": self.password}) as c:
            c.run("sudo reboot", warn=True)
 
class ScriptInspector:
    # List of common argument names that likely refer to a camera source
    CAMERA_KEYWORDS = {'camera', 'source', 'input', 'cam'}

    @staticmethod
    def _get_val(node):
        """Strictly extracts literal values to prevent code leakage."""
        try:
            # Only return actual literals (strings, numbers, etc.)
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            # If it is a Constant node (Python 3.8+), return its value
            if isinstance(node, ast.Constant):
                return node.value
            # If it's code/logic (Name, Attribute), return None so it isn't used as an argument
            return None
        
    @staticmethod
    def extract_args(item: Path):
        if item.suffix != '.py': return None
        args_found = {'positional': [], 'flags': []}
        try:
            # Safely read file with utf-8, ignore errors
            content = item.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument':
                    arg = ScriptInspector._parse_arg_node(node)
                    if not arg: continue
                    if arg['is_flag']: args_found['flags'].append(arg)
                    else: args_found['positional'].append(arg)
            return args_found if any(args_found.values()) else None
        except Exception as e:
            # print(f"Error parsing script {item.name}: {e}")
            return None
    
    @staticmethod
    def is_plugin_capable(item: Path) -> bool:
        """Checks if a script explicitly defines a --remote-utils-gui flag."""
        try:
            args = ScriptInspector.extract_args(item)
            if not args or 'flags' not in args:
                return False
            for flag in args['flags']:
                if 'remote-utils-gui' in flag['actual_name'].lower():
                    return True
            return False
        except: return False
        
    @staticmethod
    def _parse_arg_node(node):
        names = []
        for a in node.args:
            val = ScriptInspector._get_val(a)
            if isinstance(val, str) and val.startswith('-'): names.append(val)
        
        if not names:
            for a in node.args:
                val = ScriptInspector._get_val(a)
                if isinstance(val, str): names.append(val)

        if not names: return None
        
        actual_name = names[-1] 
        kwargs = {}
        for k in node.keywords:
            val = ScriptInspector._get_val(k.value)
            # Handle list/tuple values for choices/metavar
            if k.arg in ['choices', 'metavar'] and isinstance(k.value, (ast.List, ast.Tuple)):
                if val is None: val = [ScriptInspector._get_val(elt) for elt in k.value.elts]
            kwargs[k.arg] = val

        clean_name = re.sub(r'^--?', '', actual_name).lower()
        formatted = clean_name.replace('-', ' ').replace('_', ' ').title()
        is_flag = actual_name.startswith('-')
        
        is_mqtt_bundle = False
        if 'mqtt' in clean_name and kwargs.get('nargs') == 5:
            meta = kwargs.get('metavar')
            if isinstance(meta, (list, tuple)) and len(meta) == 5:
                required_set = {'IP', 'PORT', 'USERNAME', 'PASSWORD', 'TOPIC'}
                current_set = {str(m).upper() for m in meta}
                if required_set == current_set:
                    is_mqtt_bundle = True

        return {
            'actual_name': actual_name,
            'formatted_name': formatted,
            'help': kwargs.get('help', ''),
            'type': kwargs.get('type', 'str'),
            'action': kwargs.get('action', 'store'),
            'default': kwargs.get('default', None),
            'choices': kwargs.get('choices', None),
            'nargs': kwargs.get('nargs', None),
            'metavar': kwargs.get('metavar', None),
            'is_flag': is_flag,
            'is_mqtt_bundle': is_mqtt_bundle,
            'is_camera_arg': clean_name in ScriptInspector.CAMERA_KEYWORDS 
        }

class LogCapture:
    """Intercepts sys.stdout and sys.stderr to provide logs to the UI."""
    def __init__(self, max_logs=500):
        self.terminal = sys.stdout
        self.logs = deque(maxlen=max_logs)
        self.lock = threading.Lock()
        
        # Standard stream attributes expected by libraries like uvicorn/click
        self.encoding = self.terminal.encoding
        self.errors = self.terminal.errors

    def isatty(self):
        """Uvicorn checks this to decide whether to use colors."""
        # Return the original stream's value to maintain correct behavior
        return self.terminal.isatty()

    def fileno(self):
        """Return the underlying file descriptor if it exists."""
        return self.terminal.fileno()

    def write(self, message):
        self.terminal.write(message)
        if message.strip():
            with self.lock:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.logs.append({"time": timestamp, "msg": message.strip()})

    def flush(self):
        self.terminal.flush()

class DatabaseManager:
    def __init__(self):
        self.db_path = Path.home() / "RemoteUtilsDatabase"
        self.admin_user = "admin_user"
        self.admin_pass = "securepassword123"
        self.is_busy = False
        # Persistent state for background harvesting
        self.bridge_client = None
        self.active_broker_info = None 

    def is_bridging(self) -> bool:
        """Returns True if the background MQTT harvester is running."""
        return self.bridge_client is not None

    def stop_bridge_logic(self):
        """Standard disconnect sequence for the background thread."""
        if self.bridge_client:
            try:
                self.bridge_client.loop_stop()
                self.bridge_client.disconnect()
            except: pass
            self.bridge_client = None
            self.active_broker_info = None

    def start_persistent_bridge(self, broker_data):
        """Starts the MQTT loop with TLS and corrected decoding logic."""
        self.stop_bridge_logic()
        self.active_broker_info = broker_data
        topic = broker_data.get('topic', 'hailo/detections')

        try:
            import paho.mqtt.client as mqtt_lib
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                self.bridge_client = mqtt_lib.Client(CallbackAPIVersion.VERSION2)
            except (ImportError, AttributeError):
                self.bridge_client = mqtt_lib.Client()
            
            if broker_data.get('username'):
                self.bridge_client.username_pw_set(broker_data['username'], broker_data['password'])
            
            # CRITICAL: Handle TLS for Port 8883
            if str(broker_data.get('port')) == '8883':
                self.bridge_client.tls_set()
            
            # Pass the raw payload to let insert_detection handle decoding
            self.bridge_client.on_message = lambda c, u, m: self.insert_detection(broker_data, m.payload.decode())
            
            self.bridge_client.connect(broker_data['url'], int(broker_data['port']), 60)
            self.bridge_client.subscribe(topic)
            self.bridge_client.loop_start()
            return True
        except Exception as e:
            print(f"Persistent Bridge Error: {e}")
            self.bridge_client = None
            return False

    def insert_detection(self, broker_data, payload):
        """Standardized decoding and insertion logic."""
        try:
            # We decode HERE once
            detections = json.loads(payload)
            if not isinstance(detections, list): 
                detections = [detections]

            conn = psycopg2.connect(
                dbname='postgres', user=self.admin_user, password=self.admin_pass,
                host=str(self.db_path), port=5433
            )
            cur = conn.cursor()
            for det in detections:
                bbox = det.get('bbox', [0, 0, 0, 0])
                cur.execute(
                    """INSERT INTO detection_debug_logs 
                       (ip, host, broker, object_name, confidence, x1, y1, x2, y2) 
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (broker_data['url'], broker_data['name'], broker_data['name'],
                     det.get('label'), det.get('confidence'),
                     bbox[0], bbox[1], bbox[2], bbox[3])
                )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Harvester DB Insert Error: {e}")
            
    async def get_pg_bin(self):
        """Dynamically finds where PostgreSQL binaries are installed."""
        for path in ["/usr/lib/postgresql/16/bin", "/usr/lib/postgresql/15/bin", "/usr/lib/postgresql/14/bin", "/usr/bin"]:
            if (Path(path) / "pg_ctl").exists():
                return path
        # Final fallback: try which
        proc = await asyncio.create_subprocess_shell("which pg_ctl", stdout=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        return str(Path(stdout.decode().strip()).parent) if stdout else None

    def is_running(self) -> bool:
        return (self.db_path / "postmaster.pid").exists()

    def is_initialized(self) -> bool:
        return (self.db_path / "PG_VERSION").exists()

    def check_system_deps(self) -> str:
        if not self.is_initialized(): return "missing"
        return "running" if self.is_running() else "stopped"

    async def run_setup_sequence(self, log_callback):
        self.is_busy = True
        bin_dir = await self.get_pg_bin()
        
        # Ensure directory is ready and has correct permissions
        self.db_path.mkdir(parents=True, exist_ok=True)
        os.chmod(self.db_path, 0o700) #

        if not self.is_initialized():
            log_callback("Initializing new cluster...")
            cmd = [f"{bin_dir}/initdb", "-D", str(self.db_path)]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            await proc.wait()

        # START SERVICE FIX
        # We redirect the socket to your DB path where you have write permissions
        log_callback("Starting PostgreSQL (Redirecting Sockets)...")
        start_cmd = [
            f"{bin_dir}/pg_ctl", "-D", str(self.db_path), "-l", "/tmp/pg_log", 
            "-o", f"-p 5433 -c unix_socket_directories='{self.db_path}'", 
            "start"
        ]
        proc = await asyncio.create_subprocess_exec(*start_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        await proc.wait()
        
        # CONFIG ADMIN (Must use the new port and socket location)
        log_callback("Configuring Admin...")
        user_sql = f"CREATE USER {self.admin_user} WITH SUPERUSER PASSWORD '{self.admin_pass}';"
        admin_cmd = [f"{bin_dir}/psql", "-p", "5433", "-h", str(self.db_path), "-d", "postgres", "-c", user_sql]
        await asyncio.create_subprocess_exec(*admin_cmd)

        await self.setup_queries(log_callback)
        await self.stop_database(log_callback)

        self.is_busy = False
        log_callback("Deployment Finished Successfully.")
    
    async def purge_database(self, log_callback):
        """Stops the database and wipes the custom data directory."""
        self.is_busy = True
        bin_dir = await self.get_pg_bin()
        
        # 1. Attempt to stop the service if it's running
        if self.is_running():
            log_callback("Stopping active database service...")
            stop_cmd = [f"{bin_dir}/pg_ctl", "-D", str(self.db_path), "stop", "-m", "immediate"]
            await asyncio.create_subprocess_exec(*stop_cmd)
            await asyncio.sleep(2)

        # 2. Delete the directory contents
        if self.db_path.exists():
            log_callback(f"Wiping directory: {self.db_path}...")
            try:
                shutil.rmtree(self.db_path)
                self.db_path.mkdir(parents=True, exist_ok=True)
                os.chmod(self.db_path, 0o700) #
                log_callback("Directory purged.")
            except Exception as e:
                log_callback(f"Error during purge: {e}")

        # 3. Clean up stale logs or sockets
        if os.path.exists("/tmp/pg_log"):
            os.remove("/tmp/pg_log")

        self.is_busy = False
        log_callback("--- PURGE COMPLETE ---")
    
    async def stop_database(self, log_callback):
        """Gracefully shuts down the PostgreSQL service."""
        self.is_busy = True
        bin_dir = await self.get_pg_bin()
        
        log_callback("Initiating shutdown...")
        # Use 'fast' mode to close active connections immediately
        stop_cmd = [f"{bin_dir}/pg_ctl", "-D", str(self.db_path), "-m", "fast", "stop"]
        
        process = await asyncio.create_subprocess_exec(
            *stop_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        await process.wait()
        
        # Cleanup the PID file just in case of a crash
        if (self.db_path / "postmaster.pid").exists():
            (self.db_path / "postmaster.pid").unlink()
            
        self.is_busy = False
        log_callback("Service stopped.")

    async def setup_queries(self, log_callback):
        """Sets up the flattened detection table for debugging."""
        log_callback("Configuring Flattened Debug Table...")
        
        setup_sql = f"""
        CREATE TABLE IF NOT EXISTS detection_debug_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            ip TEXT,
            host TEXT,
            broker TEXT,
            object_name TEXT,
            confidence FLOAT,
            x1 FLOAT, y1 FLOAT, x2 FLOAT, y2 FLOAT
        );
        CREATE INDEX IF NOT EXISTS idx_debug_timestamp ON detection_debug_logs(timestamp DESC);
        GRANT ALL PRIVILEGES ON TABLE detection_debug_logs TO {self.admin_user};
        """
        bin_dir = await self.get_pg_bin()
        cmd = [f"{bin_dir}/psql", "-p", "5433", "-h", str(self.db_path), "-d", "postgres", "-c", setup_sql]
        process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        await process.wait()
        log_callback("Debug table ready.")

    def save_to_debug_db(self, broker_data, topic, payload):
        """Unrolls MQTT arrays into individual detection entries."""
        try:
            detections = json.loads(payload)
            conn = psycopg2.connect(
                dbname='postgres', user=db_manager.admin_user, password=db_manager.admin_pass,
                host=str(db_manager.db_path), port=5433
            )
            cur = conn.cursor()

            for det in detections:
                bbox = det.get('bbox', [0, 0, 0, 0])
                cur.execute(
                    """INSERT INTO detection_debug_logs 
                       (ip, host, broker, object_name, confidence, x1, y1, x2, y2) 
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (broker_data['url'], broker_data['name'], broker_data['name'],
                     det.get('label'), det.get('confidence'),
                     bbox[0], bbox[1], bbox[2], bbox[3])
                )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Harvester DB Insert Error: {e}")

db_manager = DatabaseManager()

# --- UI Layer ---

class SystemHeader:
    """Consolidated Header with last-changed GPIO tracking and unified metrics."""
    def __init__(self, drawer):
        self.drawer = drawer
        self.username = app.storage.user.get('username', 'USER').upper()
        self.cpu = self.ram = self.tmp = self.pwr = self.io_status = None
        self.last_changed_pin = "READY"
        self.last_val = 0
        self.previous_states = {} 
        self.db_status = None
        self.build()

    def _metric_tack(self, icon, tooltip_text, custom_width='w-10'):
        with ui.column().classes(f'items-center gap-0 px-1 {custom_width}'):
            ui.icon(icon, color='blue-300').classes('text-sm')
            lbl = ui.label('--').classes('text-[9px] font-bold text-white uppercase tracking-tighter truncate w-full text-center')
            ui.tooltip(tooltip_text) 
            return lbl

    def build(self):
        with ui.header(elevated=True).classes('items-center py-1 px-4').style('background-color: #0a192f; border-bottom: 2px solid #1a237e'):
            with ui.button(on_click=lambda: self.drawer.toggle()).props('flat color=white dense no-caps').classes('q-pa-xs'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('menu')
                    ui.label(self.username).classes('text-sm font-black tracking-widest text-blue-200')
            
            ui.space()
            
            with ui.row().classes('items-center gap-1'):
                with ui.column().classes('items-center gap-0 px-1 w-14 cursor-pointer hover:bg-white/10 rounded transition-colors') \
                    .on('click', lambda: GPIOMapPopup().open()):
                    ui.icon('settings_input_component', color='blue-300').classes('text-sm')
                    self.io_status = ui.label('READY').classes('text-[9px] font-bold text-amber-400 uppercase tracking-tighter truncate w-full text-center')
                    ui.tooltip('Last Changed GPIO (Click for Map)')

                self.cpu = self._metric_tack('memory', 'CPU Load')
                self.ram = self._metric_tack('storage', 'RAM Usage')
                self.tmp = self._metric_tack('thermostat', 'Core Temp')
                self.pwr = self._metric_tack('bolt', 'Power Status')
                self.db_status = self._metric_tack('database', 'PostgreSQL Status')
        
        ui.timer(1.0, self.refresh_metrics)

    def refresh_metrics(self):
        self.cpu.set_text(f"{psutil.cpu_percent():.0f}%")
        self.ram.set_text(f"{psutil.virtual_memory().percent:.0f}%")
        
        for pin_num, data in GPIOManager.PIN_MAP.items():
            bcm = data['bcm']
            if bcm is not None:
                current_val = gpio_manager.get_pin_status(bcm)
                if bcm in self.previous_states and current_val != self.previous_states[bcm]:
                    self.last_changed_pin = f"BCM{bcm}"
                    self.last_val = current_val
                self.previous_states[bcm] = current_val

        display_text = f"{self.last_changed_pin}:{'H' if self.last_val else 'L'}" if self.last_changed_pin != "READY" else "READY"
        self.io_status.set_text(display_text)

        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                self.tmp.set_text(f"{int(f.read()) / 1000:.0f}°")
        except: self.tmp.set_text("N/A")
        self.pwr.set_text("AC")

        db_state = db_manager.check_system_deps()
        self.db_status.set_text("UP" if db_state == "running" else "DOWN")
        self.db_status.classes(replace='text-emerald-400' if db_state == "running" else 'text-red-500')

class GPIOMapPopup(ui.dialog):
    def __init__(self):
        super().__init__()
        self.pin_elements = {} 
        
        with self, ui.card().classes('bg-[#050a0f] border border-blue-900 p-0 overflow-hidden').style('width: 95vw; max-width: 800px;'):
            with ui.row().classes('w-full items-center justify-between bg-[#0d1b2a] px-4 py-1 border-b border-blue-900'):
                ui.label('PI HARDWARE MAP').classes('text-blue-200 font-mono text-[10px] font-bold')
                ui.button(icon='close', on_click=self.close).props('flat color=white dense size=xs')

            with ui.row().classes('w-full p-4 justify-center gap-1 no-wrap bg-[#050a0f]'):
                with ui.column().classes('gap-1'):
                    for pin_num in range(1, 41, 2):
                        self._render_compact_capsule(pin_num, align_right=True)
                
                with ui.column().classes('gap-1 items-center bg-blue-900/10 px-1 rounded'):
                    for i in range(1, 21):
                        with ui.row().classes('h-7 items-center gap-2 text-[8px] font-mono text-gray-500'):
                            ui.label(str(i*2-1))
                            ui.label(str(i*2))

                with ui.column().classes('gap-1'):
                    for pin_num in range(2, 42, 2):
                        self._render_compact_capsule(pin_num, align_right=False)

        ui.timer(0.5, self.update_map)

    def _render_compact_capsule(self, pin_num, align_right):
        data = GPIOManager.PIN_MAP[pin_num]
        bcm = data['bcm']
        
        bg_colors = {
            'pwr': 'bg-red-950/40 border-red-900/50',
            'gnd': 'bg-slate-900/60 border-slate-800/50',
            'gpio': 'bg-blue-950/30 border-blue-900/40'
        }
        text_colors = {'pwr': 'text-red-400', 'gnd': 'text-slate-500', 'gpio': 'text-blue-300'}
        
        container_class = f'flex items-center gap-2 px-2 h-7 w-44 rounded border transition-all {bg_colors[data["type"]]}'
        if align_right: container_class += ' flex-row-reverse text-right'

        with ui.element('div').classes(container_class) as container:
            dot = ui.label('').classes('w-2 h-2 rounded-full bg-black/40 border border-white/10 shrink-0')
            lbl = ui.label(data['label']).classes(f'text-[9px] font-bold {text_colors[data["type"]]} truncate grow uppercase')
            
            if bcm is not None:
                self.pin_elements[bcm] = {'container': container, 'dot': dot, 'label': lbl}

    def update_map(self):
        for bcm, el in self.pin_elements.items():
            val = gpio_manager.get_pin_status(bcm)
            
            if val:
                el['container'].classes('bg-emerald-500/20 border-emerald-400/60 shadow-[0_0_8px_rgba(52,211,153,0.2)]')
                el['dot'].classes('bg-emerald-400 shadow-[0_0_5px_rgba(52,211,153,1)]')
                el['label'].classes('text-emerald-300', remove='text-blue-300')
            else:
                el['container'].classes(remove='bg-emerald-500/20 border-emerald-400/60 shadow-[0_0_8px_rgba(52,211,153,0.2)]')
                el['dot'].classes(remove='bg-emerald-400 shadow-[0_0_5px_rgba(52,211,153,1)]')
                el['label'].classes('text-blue-300', remove='text-emerald-300')

class DashboardLayout:
    """A reusable layout wrapper that provides the Header and Sidebar without extra titles."""
    def __init__(self):
        ui.query('.q-page').classes('p-0')
        self.drawer = self.setup_sidebar()
        self.header = SystemHeader(self.drawer)

    def setup_sidebar(self):
        with ui.left_drawer().style('background-color: #0d1b2a; color: white').classes('column no-wrap p-0') as dr:
            
            btn_props = 'flat no-caps color=white dense'
            btn_classes = 'w-full justify-start px-4 py-3 text-xs font-medium hover:bg-blue-900/30 rounded-none'

            with ui.column().classes('grow w-full gap-0'):
                ui.label('SYSTEM').classes('text-[10px] font-bold px-5 py-3 text-blue-400 tracking-widest opacity-60')
                
                ui.button('File Manager', icon='folder_open', on_click=lambda: ui.navigate.to('/')) \
                    .props(btn_props).classes(btn_classes)
                
                ui.button('Resources', icon='dns', on_click=lambda: ui.navigate.to('/resources')) \
                    .props(btn_props).classes(btn_classes)
                
                ui.button('Plugins', icon='extension', on_click=lambda: ui.navigate.to('/plugins')) \
                    .props(btn_props).classes(btn_classes)
                
                ui.button('Database', icon='storage', on_click=lambda: ui.navigate.to('/database')) \
                    .props(btn_props).classes(btn_classes)
            
            with ui.column().classes('w-full pb-4 gap-0'):
                ui.separator().classes('bg-blue-900/30 mb-2')
                
                ui.button('System Logs', icon='list_alt', on_click=lambda: ui.navigate.to('/logs')).props(btn_props).classes(btn_classes)

                ui.button('System Update', icon='system_update_alt', on_click=self.open_update_dialog) \
                    .props(btn_props).classes(btn_classes + ' text-emerald-300 hover:bg-emerald-900/20')
                
                ui.button('Reboot Host', icon='restart_alt', on_click=self.confirm_reboot) \
                    .props(btn_props).classes(btn_classes + ' text-red-400 hover:bg-red-900/20')

                ui.button('Logout Session', icon='logout', on_click=self.logout) \
                    .props(btn_props).classes(btn_classes + ' text-blue-300')
                    
        return dr
    
    def open_update_dialog(self):
        """Opens a dialog to upload/update, deploy locally, or uninstall."""
        running_path = Path(__file__).resolve()
        expected_path = Path.home() / 'RemoteUtils.py'
        # Robust comparison using resolved strings
        is_deployed = str(running_path) == str(expected_path)

        with ui.dialog() as dialog, ui.card().classes('bg-[#0a192f] border border-emerald-500 w-96 p-0 overflow-hidden'):
            # Header
            with ui.row().classes('w-full items-center justify-between bg-[#0d1b2a] px-4 py-2 border-b border-emerald-900/50'):
                ui.label('SYSTEM MAINTENANCE').classes('text-emerald-400 font-black tracking-widest text-xs')
                ui.button(icon='close', on_click=dialog.close).props('flat color=white dense size=sm')

            with ui.column().classes('w-full p-4 gap-4'):
                if not is_deployed:
                    # --- DEPLOYMENT VIEW ---
                    ui.label('LOCAL DEPLOYMENT REQUIRED').classes('text-amber-500 font-bold text-xs')
                    ui.label('The script is currently running from a temporary path. To enable updates, move it to your home directory.').classes('text-gray-400 text-[10px]')
                    
                    with ui.column().classes('w-full bg-black/40 p-3 rounded border border-blue-900/20'):
                        ui.label('Current:').classes('text-gray-500 text-[8px] uppercase')
                        ui.label(str(running_path)).classes('text-amber-400 font-mono text-[9px] break-all')
                        ui.icon('arrow_downward', color='blue-900').classes('self-center my-1')
                        ui.label('Target:').classes('text-gray-500 text-[8px] uppercase')
                        ui.label(str(expected_path)).classes('text-emerald-400 font-mono text-[9px] break-all')

                    ui.button('DEPLOY TO HOME', icon='rocket_launch', on_click=lambda: self.do_local_deploy(running_path, expected_path, dialog)) \
                        .props('color=emerald-9 w-full')
                else:
                    # --- UPDATE & UNINSTALL VIEW ---
                    ui.label('UPDATE SOURCE').classes('text-blue-400 font-bold text-[10px] tracking-widest')
                    ui.upload(on_upload=self.handle_update_upload, auto_upload=True, max_files=1) \
                        .props('dark accept=.py color=emerald-9').classes('w-full')
                    
                    ui.separator().classes('bg-red-900/30 my-2')
                    
                    ui.label('DANGER ZONE').classes('text-red-500 font-bold text-[10px] tracking-widest')
                    ui.button('UNINSTALL SYSTEM', icon='delete_forever', on_click=self.confirm_uninstall) \
                        .props('outline color=red-10 w-full').classes('hover:bg-red-900/10')

        dialog.open()

    def do_local_deploy(self, src, dst, dialog):
        """Copies script to home and notifies user."""
        try:
            shutil.copy2(src, dst)
            ui.notify('SUCCESS: Deployed to Home', type='positive', color='emerald-10')
            ui.notify('Restart script from ~/ to enable full features.', color='amber-10', duration=0)
            dialog.close()
        except Exception as e:
            ui.notify(f"Deployment Failed: {e}", color='red-10')

    async def handle_update_upload(self, e):
        """Processes uploaded updates."""
        try:
            binary_data = e.content.read() 
            if e.name != 'RemoteUtils.py':
                ui.notify("Error: Filename must be RemoteUtils.py", color='red-10')
                return
            with open(Path(__file__).resolve(), 'wb') as f:
                f.write(binary_data)
            ui.notify('UPDATE SUCCESSFUL. REBOOTING...', color='emerald-10')
            await asyncio.sleep(1)
            os.system('sudo reboot')
        except Exception as err:
            ui.notify(f"Update Failed: {err}", color='red-10')

    async def confirm_uninstall(self):
        """Warning dialog before system deletion."""
        with ui.dialog() as diag, ui.card().classes('bg-[#0a192f] border border-red-900 p-6'):
            ui.label('PERMANENT UNINSTALL').classes('text-red-500 font-black text-lg')
            ui.label('This will remove the system service and delete the script from home. Continue?').classes('text-gray-400 text-sm')
            with ui.row().classes('w-full justify-end mt-4 gap-2'):
                ui.button('CANCEL', on_click=diag.close).props('flat color=white')
                ui.button('YES, DELETE SYSTEM', on_click=self.run_uninstall_sequence).props('color=red-10')
        diag.open()

    def run_uninstall_sequence(self):
        """Stops service, removes files, and exits."""
        try:
            ui.notify('Stopping service...', color='amber-9')
            # 1. Stop and disable the service
            os.system('sudo systemctl stop pimanager.service')
            os.system('sudo systemctl disable pimanager.service')
            
            # 2. Remove the service file
            os.system('sudo rm /etc/systemd/system/pimanager.service')
            os.system('sudo systemctl daemon-reload')
            
            # 3. Delete the script itself
            script_path = Path.home() / 'RemoteUtils.py'
            if script_path.exists():
                script_path.unlink()
            
            ui.notify('UNINSTALL COMPLETE. Exiting...', color='red-10', duration=0)
            # 4. Kill the current process
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            ui.notify(f"Uninstall failed: {e}", color='red-10')
        
    def logout(self):
        app.storage.user['authenticated'] = False
        ui.navigate.to('/')

    async def confirm_reboot(self):
        with ui.dialog() as dialog, ui.card().classes('bg-[#0a192f] border border-red-900 p-6'):
            ui.label('CRITICAL SYSTEM REBOOT').classes('text-red-500 font-black tracking-tighter text-lg')
            ui.label('Host machine will restart immediately. Confirm?').classes('text-gray-400 text-sm')
            with ui.row().classes('w-full justify-end mt-4 gap-2'):
                ui.button('CANCEL', on_click=dialog.close).props('flat color=white')
                ui.button('EXECUTE REBOOT', on_click=lambda: os.system('sudo reboot')).props('color=red-10')
        dialog.open()

class FileNavigator:
    """Integrated Navigator with Browser/Shell modes and Workspace support."""
    def __init__(self, side_label: str, parent_browser):
        self.side_label = side_label
        self.parent = parent_browser
        self.boundary_path = Path("/")
        self.home_path = Path.home()
        self.current_path = self.home_path
        self.last_path = self.home_path
        self.mode = "browser" 
        
        self.header_row = None
        self.content_area = None
        self.shell_input = None
        self.log_area = None
        
        self.build()

    def build(self):
        with ui.column().classes('w-full h-full gap-0 border-r border-blue-900/40'):
            self.header_row = ui.row().classes('w-full items-center bg-[#0d1b2a] px-2 py-1 border-b border-blue-900/50')
            self.content_area = ui.column().classes('w-full grow overflow-hidden bg-[#050a0f] gap-0')
            self.refresh_ui()

    def refresh_ui(self):
        self.header_row.clear()
        self.content_area.clear()
        
        with self.content_area:
            self._setup_persistent_dialogs()

        if self.mode == "browser":
            self._build_browser_interface()
        else:
            self._build_shell_interface()

    def _build_browser_interface(self):
        with self.header_row:
            with ui.row().classes('gap-1 items-center'):
                with ui.button(icon='workspaces', on_click=self.open_workspace_manager).props('flat color=amber-400 dense size=sm').classes('animate-pulse').tooltip('Workspaces'):
                    pass
                
                ui.button(icon='arrow_back', on_click=self.go_back).props('flat color=blue-200 dense size=sm').tooltip('Go Back')
                ui.button(icon='home', on_click=self.go_home).props('flat color=blue-200 dense size=sm').tooltip('Home/Workspace Root')
                ui.button(icon='refresh', on_click=self.refresh_ui).props('flat color=blue-200 dense size=sm').tooltip('Refresh List')
            
            ui.label(str(self.current_path)).classes('text-[10px] font-mono text-blue-400 grow truncate px-3 tracking-tighter')
            
            with ui.row().classes('gap-1'):
                ui.button(icon='terminal', on_click=self.toggle_mode).props('flat color=blue-400 dense size=sm').tooltip('Open Shell')
                ui.button(icon='create_new_folder', on_click=self.prompt_new_folder).props('flat color=blue-300 dense size=sm').tooltip('New Folder')
                ui.button(icon='upload', on_click=lambda: self.upload_dialog.open()).props('flat color=emerald-400 dense size=sm').tooltip('Upload Files')

        with self.content_area:
            self.list_area = ui.column().classes('w-full h-full overflow-y-auto gap-0')
            self.refresh_file_list()

    def open_workspace_manager(self):
        with ui.dialog() as diag, ui.card().classes('bg-[#0a192f] border border-amber-900 p-4 w-80'):
            ui.label('AUTO-DISCOVERED WORKSPACES').classes('text-amber-400 font-black tracking-widest text-xs mb-2')
            
            with ui.column().classes('w-full gap-1 mb-4'):
                ui.button('Full File System (Anchored to ~/)', on_click=lambda: self.set_workspace(Path("/"), diag, ui_home=Path.home())).props('flat no-caps color=blue-200 dense').classes('text-xs w-full justify-start')
                ui.separator().classes('bg-amber-900/30 my-2')

                tmp_dir = Path("/tmp")
                found_any = False
                if tmp_dir.exists():
                    for folder in tmp_dir.iterdir():
                        if folder.is_dir() and folder.name.startswith("workspace-"):
                            found_any = True
                            display_name = folder.name.replace("workspace-", "")
                            with ui.row().classes('w-full items-center justify-between hover:bg-white/5 px-2 py-1 rounded group'):
                                ui.button(display_name, on_click=lambda p=folder: self.set_workspace(p, diag)).props('flat no-caps color=amber-200 dense').classes('text-xs grow justify-start')
                                ui.button(icon='delete_forever', on_click=lambda p=folder: self.delete_workspace_physical(p, diag)).props('flat color=red-400 dense size=xs').classes('opacity-0 group-hover:opacity-100')
                
                if not found_any: ui.label('No /tmp/ workspaces found').classes('text-[9px] text-gray-600 italic px-2')

            ui.label('INITIALIZE NEW').classes('text-[10px] text-gray-500 font-bold mb-1')
            name_input = ui.input('Name').props('dark dense outline').classes('w-full mb-2')
            
            async def save_and_create():
                if name_input.value:
                    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name_input.value)
                    new_path = Path(f"/tmp/workspace-{safe_name}")
                    try:
                        new_path.mkdir(parents=True, exist_ok=True)
                        ui.notify(f"Initialized: {new_path.name}")
                        diag.close()
                        self.set_workspace(new_path)
                    except Exception as e: ui.notify(f"Failed: {e}", color='red-10')
            
            ui.button('CREATE WORKSPACE', on_click=save_and_create).props('color=amber-9').classes('w-full')
        diag.open()

    def set_workspace(self, path: Path, dialog=None, ui_home: Path = None):
        if path.exists():
            self.boundary_path = path 
            self.home_path = ui_home if ui_home else path
            self.current_path = self.home_path
            if dialog: dialog.close()
            self.refresh_ui()
        else: ui.notify("Path missing", color='red-10')

    def delete_workspace_physical(self, path: Path, dialog):
        try:
            if path.exists() and path.name.startswith('workspace-'):
                shutil.rmtree(path)
                ui.notify(f"Wiped {path.name}")
                dialog.close()
                self.set_workspace(Path("/"), ui_home=Path.home())
                self.open_workspace_manager() 
        except Exception as e: ui.notify(f"Error: {e}", color='red-10')

    def _build_shell_interface(self):
        with self.header_row:
            ui.icon('terminal', color='emerald-500').classes('text-sm ml-1')
            ui.label(f"SHELL: {self.current_path.name}").classes('text-[10px] font-mono text-emerald-400 grow px-2')
            
            with ui.row().classes('gap-1'):
                ui.button(icon='delete_sweep', on_click=lambda: self.log_area.clear()).props('flat color=blue-300 dense size=sm').tooltip('Clear Logs')
                ui.button(icon='close', on_click=self.toggle_mode).props('flat color=red-400 dense size=sm').tooltip('Exit Shell')

        with self.content_area.classes('bg-black'):
            self.log_area = ui.log().classes('w-full grow text-emerald-500 font-mono text-[11px] p-2 bg-black border-0')
            with ui.row().classes('w-full items-center px-3 py-0 bg-black no-wrap border-t border-emerald-950/30'):
                ui.label(f'pi@system:{self.current_path.name}$').classes('text-emerald-800 font-mono font-bold text-[11px] whitespace-nowrap mr-1')
                self.shell_input = ui.input().on('keydown.enter', self.execute_shell_cmd) \
                    .props('dark borderless dense autofocus inputmode=text enterkeyhint=send shadow-0') \
                    .classes('grow text-emerald-400 font-mono text-[11px] caret-emerald-500') \
                    .style('background: transparent; outline: none; border: none;')
                ui.button(icon='send', on_click=self.execute_shell_cmd) \
                    .props('flat dense size=sm') \
                    .classes('text-emerald-600 hover:text-emerald-400 transition-colors') \
                    .tooltip('Execute Command')
                          
    async def handle_upload(self, e):
        try:
            filename = e.file.name 
            dest = self.current_path / filename
            await e.file.save(dest)
            ui.notify(f"SUCCESS: {filename} saved.", color='emerald-9')
            self.refresh_file_list()
        except Exception as err:
            ui.notify(f"Upload failed: {str(err)}", color='red-10')

    def handle_download(self, item: Path):
        try:
            if item.is_file(): ui.download(str(item))
            else:
                target = shutil.make_archive(os.path.join(tempfile.gettempdir(), item.name), 'zip', item)
                ui.download(target, filename=f"{item.name}.zip")
        except Exception as e: ui.notify(f"Error: {e}", color='red-10')

    def handle_copy(self, item: Path):
        dest_pane = self.parent.right_pane if self.side_label == "LEFT" else self.parent.left_pane
        dest_dir = dest_pane.current_path
        dest_file = dest_dir / item.name
        try:
            if item.is_dir(): shutil.copytree(str(item), str(dest_file), dirs_exist_ok=True)
            else: shutil.copy2(str(item), str(dest_file))
            ui.notify("COPIED")
            dest_pane.refresh_file_list()
        except Exception as e: ui.notify(f"Failed: {e}", color='red-10')

    def handle_move(self, item: Path):
        dest_pane = self.parent.right_pane if self.side_label == "LEFT" else self.parent.left_pane
        dest_dir = dest_pane.current_path
        dest_file = dest_dir / item.name
        try:
            shutil.move(str(item), str(dest_file))
            self.parent.left_pane.refresh_file_list()
            self.parent.right_pane.refresh_file_list()
        except Exception as e: ui.notify(f"Failed: {e}", color='red-10')

    def handle_delete_confirm(self, item: Path):
        async def do_delete():
            try:
                if item.is_dir(): shutil.rmtree(str(item))
                else: item.unlink()
                self.refresh_file_list()
                diag.close()
            except Exception as e: ui.notify(f"Error: {e}", color='red-10')

        with ui.dialog() as diag, ui.card().classes('bg-[#0a192f] border border-red-900'):
            ui.label(f'DELETE {item.name}?').classes('text-red-400 font-bold')
            with ui.row().classes('w-full justify-end mt-4'):
                ui.button('CANCEL', on_click=diag.close).props('flat color=white')
                ui.button('DELETE', on_click=do_delete).props('color=red-10')
        diag.open()

    def _setup_persistent_dialogs(self):
        with ui.dialog() as self.upload_dialog, ui.card().classes('bg-[#0a192f] border border-blue-800'):
            ui.label(f'UPLOAD TO: {self.side_label}').classes('text-blue-200 text-xs font-bold')
            ui.upload(on_upload=self.handle_upload, multiple=True, max_file_size=500_000_000).classes('w-80').props('dark')
            ui.button('CLOSE', on_click=self.upload_dialog.close).props('flat color=white').classes('w-full')

    def refresh_file_list(self):
        if not hasattr(self, 'list_area') or self.list_area is None: return
        self.list_area.clear()
        try:
            items = sorted(list(self.current_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
            with self.list_area:
                for item in items: self._build_row(item)
        except Exception as e:
            with self.list_area: ui.label(f"DENIED").classes('text-red-500 text-[10px] p-4')

    def _build_row(self, item: Path):
        with ui.row().classes('w-full items-center hover:bg-blue-900/20 px-3 py-0.5 border-b border-blue-900/10 group'):
            icon = 'folder' if item.is_dir() else 'description'
            ui.icon(icon, color='blue-400' if item.is_dir() else 'slate-500').classes('text-sm')
            lbl = ui.label(item.name).classes('grow text-[12px] cursor-pointer text-gray-300 truncate')
            
            if item.is_dir(): 
                lbl.on('click', lambda: self.navigate_to(item))
                lbl.tooltip('Open Folder')

            with ui.row().classes('opacity-0 group-hover:opacity-100 transition-opacity gap-0'):
                if item.is_file():
                    ui.button(icon='visibility', on_click=lambda: MediaViewer(item).open()).props('flat color=blue-200 dense size=xs').tooltip('View')

                if item.suffix in ['.py', '.sh']:
                    ui.button(icon='play_arrow', on_click=lambda: ExecutionDialog(item, self.current_path).open()).props('flat color=emerald-400 dense size=xs').tooltip('Run Script')
                
                ui.button(icon='download', on_click=lambda: self.handle_download(item)).props('flat color=blue-200 dense size=xs').tooltip('Download')
                ui.button(icon='content_copy', on_click=lambda: self.handle_copy(item)).props('flat color=amber-300 dense size=xs').tooltip('Copy to Other Pane')
                
                ui.button(icon='input', on_click=lambda: self.handle_move(item)).props('flat color=orange-400 dense size=xs').tooltip('Move to Other Pane')
                ui.button(icon='delete', on_click=lambda: self.handle_delete_confirm(item)).props('flat color=red-500 dense size=xs').tooltip('Delete')

    def navigate_to(self, p: Path): self.current_path = p; self.refresh_ui()
    def go_back(self):
        if self.current_path != self.boundary_path and self.current_path.parent != self.current_path: 
            self.navigate_to(self.current_path.parent)
    def go_home(self): self.navigate_to(self.home_path)
    def toggle_mode(self): self.mode = "shell" if self.mode == "browser" else "browser"; self.refresh_ui()
    
    async def execute_shell_cmd(self):
        cmd = self.shell_input.value.strip()
        if not cmd: return
        self.shell_input.value = ''; self.log_area.push(f"\n> {cmd}")
        try:
            process = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=str(self.current_path))
            stdout, _ = await process.communicate()
            if stdout: self.log_area.push(stdout.decode().rstrip())
        except Exception as e: self.log_area.push(f"Error: {e}")

    def prompt_new_folder(self):
        async def create():
            if inp.value:
                try: (self.current_path / inp.value).mkdir(); self.refresh_file_list(); d.close()
                except Exception as e: ui.notify(str(e), color='red-10')
        with ui.dialog() as d, ui.card().classes('bg-[#0a192f] border border-blue-800'):
            ui.label('NEW FOLDER').classes('text-blue-200 text-xs font-bold')
            inp = ui.input('Name').props('dark dense outline autofocus')
            ui.button('CREATE', on_click=create).classes('w-full mt-2')
        d.open()
    
    async def confirm_reboot(self):
        with ui.dialog() as dialog, ui.card().classes('bg-[#0a192f] border border-red-900 p-6'):
            ui.label('CRITICAL SYSTEM REBOOT').classes('text-red-500 font-black tracking-tighter text-lg')
            ui.label('Host machine will restart immediately. Confirm?').classes('text-gray-400 text-sm')
            with ui.row().classes('w-full justify-end mt-4 gap-2'):
                ui.button('CANCEL', on_click=dialog.close).props('flat color=white')
                ui.button('EXECUTE REBOOT', on_click=lambda: os.system('sudo reboot')).props('color=red-10')
        dialog.open()

class FileBrowser:
    def __init__(self):
        with ui.row().classes('w-full h-[calc(100vh-50px)] gap-0 no-wrap flex-col md:flex-row bg-[#050a0f]'):
            with ui.column().classes('w-full md:w-1/2 h-1/2 md:h-full border-b md:border-b-0 md:border-r border-blue-900/40 overflow-hidden'):
                self.left_pane = FileNavigator("LEFT", self)
            with ui.column().classes('w-full md:w-1/2 h-1/2 md:h-full overflow-hidden'):
                self.right_pane = FileNavigator("RIGHT", self)

class ExecutionDialog(ui.dialog):
    def __init__(self, item: Path, current_dir: Path):
        super().__init__()
        self.item = item
        self.current_dir = current_dir
        self.process = None
        self.arg_elements = {}
        self.log_container = None 
        self.log_scroll = None
        
        self.mqtt_states = {} 
        self.topic_rows = {}     
        
        self.on_value_change(lambda e: self.handle_dialog_close(e))
        self.build_input_view()

    def handle_dialog_close(self, e):
        if not e.value: 
            self.kill_process()
            self.stop_all_listeners()

    def build_input_view(self):
        self.clear()
        self.arg_elements = {}
        detected = ScriptInspector.extract_args(self.item)
        
        with self, ui.card().classes('bg-[#0a192f] border border-emerald-900 p-4 w-full max-w-[600px] overflow-hidden') as self.card:
            ui.label(f'RUN: {self.item.name}').classes('text-emerald-400 font-black text-lg mb-2 tracking-wide')
            
            with ui.scroll_area().classes('w-full h-[60vh] pr-4'):
                if detected:
                    self._build_section("REQUIRED PARAMETERS", detected['positional'], is_optional=False)
                    self._build_section("OPTIONAL FLAGS", detected['flags'], is_optional=True)
                else:
                    self.arg_elements['raw'] = {'input': ui.input('Manual Arguments').props('dark outline').classes('w-full'), 'type': 'raw'}
            
            ui.separator().classes('bg-emerald-900/30 my-4')
            
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('CANCEL', on_click=self.close).props('flat color=white')
                ui.button('EXECUTE SCRIPT', on_click=self.assemble_and_run).props('color=emerald-9 icon=play_arrow')

    def _build_section(self, section_label, args_list, is_optional):
        if not args_list: return
        ui.label(section_label).classes('text-[10px] text-emerald-600 font-bold mt-4 mb-2 tracking-widest uppercase')
        
        for arg in args_list:
            nargs = arg.get('nargs')
            metavar = arg.get('metavar')
            is_int_nargs = isinstance(nargs, int) and not isinstance(nargs, bool)
            is_tuple_meta = isinstance(metavar, (list, tuple))
            
            if is_int_nargs:
                initial_count = nargs
                is_fixed_count = True
                is_multi = False 
            elif is_tuple_meta:
                initial_count = len(metavar)
                is_fixed_count = False 
                is_multi = True
            else:
                initial_count = 1
                is_fixed_count = False
                is_multi = nargs in ['+', '*']

            # Styled container for the argument
            with ui.column().classes('w-full gap-1 mb-3 bg-black/20 p-2 rounded border border-emerald-900/20'):
                with ui.row().classes('w-full items-start gap-3 no-wrap'):
                    cb = None
                    if is_optional: 
                        cb = ui.checkbox().props('dark dense color=emerald').style('margin-top: 8px')
                    
                    with ui.column().classes('gap-0 mt-1'):
                        ui.label(arg['formatted_name']).classes('text-xs text-emerald-100 font-bold w-24 truncate').tooltip(arg['actual_name'])
                        if arg['help']:
                            ui.label(arg['help']).classes('text-[9px] text-gray-400 w-24 break-words leading-tight')

                    input_container = ui.column().classes('grow gap-2')
                    self.arg_elements[arg['actual_name']] = {
                        'check': cb, 
                        'container': input_container, 
                        'inputs': [], 
                        'type': 'flag' if is_optional else 'positional', 
                        'meta': arg
                    }

                    # --- INTEGRATED MQTT SELECTOR ---
                    # Placed INSIDE the input container so it respects the optional checkbox
                    if arg.get('is_mqtt_bundle'):
                        self._render_mqtt_integrated_selector(arg, input_container)
                    
                    # Generate Input Fields
                    for i in range(initial_count):
                        field_label = arg['formatted_name']
                        if is_tuple_meta and i < len(metavar): field_label = metavar[i]
                        elif metavar and isinstance(metavar, str):
                            if initial_count > 1: field_label = f"{metavar} {i+1}"
                            else: field_label = metavar
                        self._add_input_row(arg['actual_name'], label=field_label, can_remove=not is_fixed_count)

                    if is_multi and not is_fixed_count:
                        ui.button(icon='add', on_click=lambda a=arg['actual_name']: self._add_input_row(a)) \
                            .props('flat dense size=sm color=emerald-400').tooltip('Add another value') \
                            .classes('self-center')
                
                if cb: input_container.bind_visibility_from(cb, 'value')

    def _render_mqtt_loader(self, arg):
        """Injects a dropdown to auto-fill MQTT bundle fields from storage."""
        brokers = app.storage.user.get('mqtt_servers', [])
        if not brokers: return

        # Map display name to broker object
        options = {i: f"{b['name']} ({b['url']})" for i, b in enumerate(brokers)}

        def apply_broker(e):
            if e.value is None: return
            data = brokers[e.value]
            inputs = self.arg_elements[arg['actual_name']]['inputs']
            
            # Smart Mapping based on input labels (created from metavars)
            # We assume the inputs are created in order: IP, PORT, USER, PASS, TOPIC
            # But checking labels makes it robust against reordering.
            
            mapping = {
                'IP': data.get('url'), 'HOST': data.get('url'),
                'PORT': data.get('port'),
                'USER': data.get('username'), 'USERNAME': data.get('username'),
                'PASS': data.get('password'), 'PASSWORD': data.get('password'), 'PWD': data.get('password'),
                'TOPIC': data.get('topic')
            }

            for inp in inputs:
                # NiceGUI Input 'label' property check
                lbl = inp._props.get('label', '').upper()
                
                # Match label keywords to data
                for key, val in mapping.items():
                    if key in lbl:
                        inp.value = val or '' # Fill value
                        # Visual feedback
                        inp.props('bg-color=emerald-9')
                        ui.timer(0.5, lambda i=inp: i.props(remove='bg-color=emerald-9'))
                        break
            
            ui.notify(f"Loaded credentials: {data['name']}", color='emerald')

        with ui.row().classes('w-full items-center gap-2 mb-2 bg-amber-900/20 p-1 rounded border border-amber-900/50'):
            ui.icon('hub', color='amber-400').classes('text-xs ml-1')
            ui.select(options=options, label='Auto-fill from Saved Broker', on_change=apply_broker) \
                .props('dark dense outlined options-dense borderless').classes('grow text-xs')
            
    def _render_mqtt_integrated_selector(self, arg, container):
        """Renders the dropdown and handles locking/unlocking of inputs."""
        brokers = app.storage.user.get('mqtt_servers', [])
        
        # Build Options: -1 is 'Other', 0..N are broker indices
        options = {-1: 'Other... (Manual Input)'}
        for i, b in enumerate(brokers):
            options[i] = f"{b['name']} ({b['url']})"

        def on_profile_change(e):
            selected_idx = e.value
            inputs = self.arg_elements[arg['actual_name']]['inputs']
            
            if selected_idx == -1:
                # Manual Mode: Enable inputs for editing
                for inp in inputs:
                    inp.enable()
                    # We do NOT clear values here to allow user to tweak previous selection
            else:
                # Profile Mode: Fill and Disable inputs
                data = brokers[selected_idx]
                
                # Robust mapping using keyword matching against the Input Label
                # This ensures IP goes to "IP", User to "Username", etc.
                mapping = {
                    'IP': data.get('url'), 'HOST': data.get('url'),
                    'PORT': data.get('port'),
                    'USER': data.get('username'), 'USERNAME': data.get('username'),
                    'PASS': data.get('password'), 'PASSWORD': data.get('password'), 'PWD': data.get('password'),
                    'TOPIC': data.get('topic')
                }

                for inp in inputs:
                    inp.disable() # Lock the input
                    
                    # Find matching data key
                    lbl = inp._props.get('label', '').upper()
                    matched = False
                    for key, val in mapping.items():
                        if key in lbl:
                            inp.value = val or ''
                            matched = True
                            break
                    
                    # Visual flair to indicate auto-fill
                    if matched:
                        inp.props('bg-color=emerald-9')
                        ui.timer(0.3, lambda i=inp: i.props(remove='bg-color=emerald-9'))

        with container:
            # Render Selector First
            ui.select(options=options, value=-1, label='Credential Profile', on_change=on_profile_change) \
                .props('dark dense outlined options-dense').classes('w-full mb-1 bg-blue-900/20')

    def _add_input_row(self, arg_name, label=None, can_remove=True):
        element = self.arg_elements[arg_name]
        arg_meta = element['meta']
        if not label:
            label = arg_meta.get('metavar') or "Value"
            if isinstance(label, (list, tuple)): label = label[-1]
        
        with element['container']:
            with ui.row().classes('w-full items-center gap-1 no-wrap') as row:
                new_input = self._create_input(arg_meta, label)
                element['inputs'].append(new_input)
                
                if can_remove and (len(element['inputs']) > 1 or arg_meta.get('nargs') in ['*', '+']):
                    ui.button(icon='remove', on_click=lambda: self._remove_input_row(arg_name, row, new_input)) \
                        .props('flat dense size=xs color=red-400')
    
    def _remove_input_row(self, arg_name, row_layout, input_el):
        self.arg_elements[arg_name]['inputs'].remove(input_el)
        self.arg_elements[arg_name]['container'].remove(row_layout)

    def _create_input(self, arg, label):
        # 1. Camera Selector
        if arg.get('is_camera_arg'):
            all_cams = ResourcesManager.get_all_cameras()
            if all_cams:
                options = {c['url']: f"📷 {c['name']}" for c in all_cams}
                return ui.select(options=options, value=all_cams[0]['url'], label=label).props('dark dense outlined').classes('grow')
        
        # 2. Choice Selector
        if arg['choices']:
            opts = {val: str(val) for val in arg['choices']}
            return ui.select(options=opts, value=arg['default'] or arg['choices'][0], label=label) \
                     .props('dark dense outlined').classes('grow')
        
        # 3. Standard Input
        default_val = str(arg['default']) if arg['default'] is not None else ''
        
        # Password Masking
        is_password = False
        lbl_upper = str(label).upper()
        if 'PASS' in lbl_upper or 'PWD' in lbl_upper or 'SECRET' in lbl_upper:
            is_password = True
            
        inp = ui.input(label=label, value=default_val, password=is_password).props('dark dense outlined').classes('grow')
        return inp
    
    def assemble_and_run(self):
        """Gathers values strictly from UI components and prepares for execution."""
        if 'raw' in self.arg_elements: 
            final_args = self.arg_elements['raw']['input'].value.split()
        else:
            cmd_parts = []
            for name, el in self.arg_elements.items():
                # SAFETY: Ensure we are only looking at real UI components
                if not isinstance(el, dict) or 'inputs' not in el or 'meta' not in el:
                    continue
                
                # Gather non-empty values from the UI text fields
                values = []
                for inp in el['inputs']:
                    val = str(inp.value).strip() if inp.value is not None else ""
                    if val:
                        values.append(val)
                
                # Positional arguments
                if el['type'] == 'positional':
                    cmd_parts.extend(values)
                
                # Flags (Options)
                elif el['type'] == 'flag':
                    # Only include if the checkbox is checked in the UI
                    if el.get('check') and el['check'].value:
                        if el['meta']['action'] in ['store_true', 'store_false']:
                            cmd_parts.append(name)
                        elif values:
                            cmd_parts.append(name)
                            cmd_parts.extend(values)
            
            final_args = cmd_parts

        self.build_log_view(final_args)

    def build_log_view(self, args_str: str):
        self.card.clear() 
        with self.card.classes('bg-black border border-emerald-900 w-[800px] max-w-none p-0 overflow-hidden'):
            with ui.row().classes('w-full items-center justify-between bg-[#0d1b2a] px-4 py-2'):
                ui.label(f'TERMINAL: {self.item.name}').classes('text-emerald-500 font-mono text-xs font-bold')
                with ui.row().classes('gap-2'):
                    ui.button(icon='refresh', on_click=self.rerun).props('flat color=blue-400 dense size=sm').tooltip('Stop & Rerun')
                    ui.button(icon='close', on_click=self.close).props('flat color=white dense size=sm').tooltip('Stop & Close')
            
            self.log_scroll = ui.scroll_area().classes('w-full h-96 bg-black p-4')
            with self.log_scroll:
                self.log_container = ui.column().classes('w-full gap-0 font-mono text-[11px]')

        asyncio.create_task(self.run_process(args_str))

    def start_mqtt_listener(self, host, port, topic, user, password):
        try:
            context_client = ui.context.client

            is_visible = True
            if topic in self.mqtt_states:
                is_visible = self.mqtt_states[topic].get('visible', True)
                old_client = self.mqtt_states[topic].get('client')
                if old_client:
                    try: old_client.loop_stop(); old_client.disconnect()
                    except: pass
            
            import paho.mqtt.client as mqtt_lib
            # Version 2.0+ compatibility check
            try:
                from paho.mqtt.enums import CallbackAPIVersion
                client = mqtt_lib.Client(CallbackAPIVersion.VERSION2)
            except (ImportError, AttributeError):
                client = mqtt_lib.Client()

            if user: client.username_pw_set(user, password)
            if int(port) == 8883: client.tls_set()

            if topic not in self.topic_rows: self.topic_rows[topic] = []

            def on_msg(c, u, msg):
                try:
                    payload = msg.payload.decode('utf-8', errors='ignore')
                    def safe_log_update():
                        try:
                            with context_client:
                                self.log_mqtt_message(topic, payload)
                        except Exception: pass
                    app.timer(0.0, safe_log_update, once=True)
                except Exception: pass

            client.on_message = on_msg
            client.connect(host, int(port), 10)
            client.subscribe(topic)
            client.loop_start()
            
            self.mqtt_states[topic] = {'client': client, 'visible': is_visible}
            return True
        except Exception as e:
            self.append_log_line(f"[MQTT ERROR] Failed to connect {host}: {e}")
            return False
        
    def stop_mqtt_listener(self, topic):
        if topic in self.mqtt_states:
            client = self.mqtt_states[topic].get('client')
            if client:
                try:
                    client.loop_stop()
                    client.disconnect()
                except: pass
            self.mqtt_states[topic]['client'] = None

    def stop_all_listeners(self):
        for topic in list(self.mqtt_states.keys()):
            self.stop_mqtt_listener(topic)

    def toggle_mqtt_visibility(self, topic, button):
        if topic in self.mqtt_states:
            curr = self.mqtt_states[topic]['visible']
            new_state = not curr
            self.mqtt_states[topic]['visible'] = new_state
            
            new_icon = 'visibility' if new_state else 'visibility_off'
            new_color = 'emerald-400' if new_state else 'gray-500'
            button.props(f'icon={new_icon} color={new_color}')
            
            if topic in self.topic_rows:
                for row in self.topic_rows[topic]:
                    try:
                        row.visible = new_state
                    except: pass 

    def log_mqtt_message(self, topic, payload):
        import json
        if self.log_container.is_deleted: return
        
        is_visible = self.mqtt_states.get(topic, {}).get('visible', True)

        with self.log_container:
            row = ui.row().classes('w-full items-start gap-2 py-1 hover:bg-white/5 font-mono border-l-2 border-blue-900/50 pl-2 group')
            row.visible = is_visible
            
            is_json = False
            json_obj = None
            try:
                json_obj = json.loads(payload)
                is_json = True
            except: pass

            icon_name = 'chat'
            icon_color = 'blue-400'
            if is_json: 
                icon_name = 'data_object'
                icon_color = 'amber-400'
            elif 'error' in payload.lower() or 'fail' in payload.lower():
                icon_name = 'warning'
                icon_color = 'red-400'

            with row:
                ui.icon(icon_name, color=icon_color).classes('text-[10px] mt-0.5 opacity-70')
                ui.label(f"[{topic}]").classes('text-blue-400 font-bold shrink-0 text-[10px] mt-0.5')
                
                with ui.column().classes('grow gap-0 min-w-0'):
                    raw_view = ui.label(payload).classes('text-gray-300 break-all text-[11px] leading-tight')
                    json_view = None
                    if is_json:
                        formatted_str = json.dumps(json_obj, indent=2)
                        json_view = ui.label(formatted_str).classes('text-amber-300 whitespace-pre font-mono text-[10px] bg-black/50 p-2 rounded border border-amber-900/30 w-full overflow-x-auto')
                        json_view.visible = False

                if is_json and json_view:
                    def toggle(e, r=raw_view, j=json_view):
                        is_raw = r.visible
                        r.visible = not is_raw
                        j.visible = is_raw
                        e.sender.props(f"icon={'expand_less' if is_raw else 'data_object'}")

                    ui.button(icon='data_object', on_click=toggle) \
                        .props('flat dense size=xs color=amber-400').tooltip('Format JSON').classes('opacity-0 group-hover:opacity-100 transition-opacity')

            if topic in self.topic_rows:
                self.topic_rows[topic].append(row)
            
        self.log_scroll.scroll_to(percent=1.0)

    def append_log_line(self, text: str):
        media_regex = r'<(image|video)\s+"([^"]+)">'
        mqtt_start_regex = r'<mqtt start "([^"]+)" "(\d+)" "([^"]+)" "([^"]*)" "([^"]*)">'
        mqtt_end_regex = r'<mqtt end "([^"]+)" "(\d+)" "([^"]+)">'

        media_match = re.search(media_regex, text)
        mqtt_start = re.search(mqtt_start_regex, text)
        mqtt_end = re.search(mqtt_end_regex, text)

        with self.log_container:
            if media_match:
                media_type, path_str = media_match.groups()
                full_path = Path(path_str)
                file_exists = full_path.exists()
                
                base_color = 'amber' if file_exists else 'red'
                icon_name = 'photo_camera' if media_type == 'image' else 'videocam'
                if not file_exists: icon_name = 'broken_image'

                with ui.row().classes(f'w-full items-center gap-2 py-2 my-1 border-l-4 border-{base_color}-500 bg-{base_color}-900/10 px-3 rounded cursor-pointer transition-colors') \
                        .on('click', lambda: MediaViewer(full_path).open() if file_exists else None):
                    
                    ui.icon(icon_name, color=f'{base_color}-400').classes('text-lg')
                    with ui.column().classes('gap-0 grow'):
                        ui.label(f"GENERATED {media_type.upper()}").classes(f'text-{base_color}-500 font-bold text-[10px] leading-none')
                        ui.label(full_path.name).classes('text-gray-300 italic text-[11px] truncate leading-tight')

                    if file_exists:
                        ui.button('OPEN', on_click=lambda: MediaViewer(full_path).open()) \
                            .props(f'flat dense size=sm color={base_color}-300 icon=visibility').classes('px-2')
                    else:
                         ui.label('MISSING').classes('text-red-500 font-bold text-[9px]')

            elif mqtt_start:
                host, port, topic, user, pw = mqtt_start.groups()
                success = self.start_mqtt_listener(host, port, topic, user, pw)
                
                color = 'emerald' if success else 'red'
                status = 'LISTENING' if success else 'FAILED'

                with ui.row().classes(f'w-full items-center gap-2 py-1 my-1 border-l-2 border-{color}-500 bg-{color}-900/10 px-2 rounded'):
                    ui.icon('hub', color=f'{color}-400').classes('text-sm')
                    ui.label(f"MQTT {status}: {host}:{port}").classes(f'text-{color}-400 font-bold text-[10px]')
                    ui.label(topic).classes('text-gray-400 italic text-[10px] truncate grow')
                    
                    if success:
                        btn = ui.button().props('flat dense size=xs icon=visibility color=emerald-400').tooltip('Toggle Log Visibility')
                        btn.on('click', lambda e, t=topic, b=btn: self.toggle_mqtt_visibility(t, b))

            elif mqtt_end:
                host, port, topic = mqtt_end.groups()
                self.stop_mqtt_listener(topic)
                
                with ui.row().classes('w-full items-center gap-2 py-1 my-1 border-l-2 border-gray-600 bg-gray-800/30 px-2 rounded'):
                    ui.icon('cloud_off', color='gray-400').classes('text-sm')
                    ui.label(f"MQTT CLOSED: {topic}").classes('text-gray-400 font-bold text-[10px]')

            else:
                ui.label(text).classes('text-emerald-300 break-all whitespace-pre-wrap leading-tight')

        self.log_scroll.scroll_to(percent=1.0)

    async def run_process(self, cmd_args: List[str]):
        """Runs the script using the precise argument list."""
        abs_path = str(self.item.absolute())
        executable = 'python3' if self.item.suffix == '.py' else 'bash'
        
        # Use -u for unbuffered output so you see errors immediately
        full_cmd = [executable, '-u', abs_path] + cmd_args
        
        try:
            # This will now show a clean command in your UI log
            self.append_log_line(f"--- STARTING: {' '.join(full_cmd)} ---\n")
            
            self.process = await asyncio.create_subprocess_exec(
                *full_cmd,
                stdout=asyncio.subprocess.PIPE, 
                stderr=asyncio.subprocess.STDOUT, 
                cwd=str(self.current_dir)
            )
            
            while True:
                line_bytes = await self.process.stdout.readline()
                if not line_bytes: break
                self.append_log_line(line_bytes.decode().rstrip())
            
            await self.process.wait()
        except Exception as e:
            self.append_log_line(f"\n[SYSTEM ERROR] {str(e)}")
        finally:
            self.stop_all_listeners()
                    
    def rerun(self): 
        self.stop_all_listeners()
        self.kill_process()
        self.build_input_view()

    def kill_process(self):
        if self.process and self.process.returncode is None:
            try: self.process.kill()
            except ProcessLookupError: pass

class ResourcesManager:
    """Manages external resources: Camera Streams and MQTT Brokers."""
    
    @staticmethod
    def get_all_cameras():
        local = []
        for i in [0, 1]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                local.append({'name': f'PI-CAM {i}', 'url': str(i), 'is_local': True})
                cap.release()
        
        remote = app.storage.user.get('cameras', [])
        return local + remote
    
    def __init__(self):
        if 'cameras' not in app.storage.user: app.storage.user['cameras'] = []
        if 'mqtt_servers' not in app.storage.user: app.storage.user['mqtt_servers'] = []
        self.active_recordings = {}
        
        self.camera_stats = {}
        self.build()

    def build(self):
        with ui.column().classes('w-full p-6 gap-8'):
            
            with ui.column().classes('w-full gap-4'):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('CAMERA NETWORK').classes('text-blue-200 text-xl font-black tracking-tighter')
                    ui.button('REGISTER CAM', icon='videocam', on_click=lambda: self.camera_dialog()) \
                        .props('flat color=emerald-400 border dense')

                self.cam_grid = ui.row().classes('w-full gap-4')
                self.refresh_cam_grid()

            ui.separator().classes('bg-blue-900/30')

            with ui.column().classes('w-full gap-4'):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('MQTT SERVER REGISTRY').classes('text-amber-200 text-xl font-black tracking-tighter')
                    ui.button('REGISTER BROKER', icon='hub', on_click=lambda: self.mqtt_dialog()) \
                        .props('flat color=amber-400 border dense')

                self.mqtt_grid = ui.row().classes('w-full gap-4')
                self.refresh_mqtt_grid()

    def refresh_cam_grid(self):
        self.cam_grid.clear()
        all_cameras = self.get_all_cameras()
        
        if not all_cameras:
            with self.cam_grid: ui.label('No cameras detected.').classes('text-gray-500 italic text-sm')
            return

        with self.cam_grid:
            for idx, cam in enumerate(all_cameras):
                self._build_camera_card(idx, cam)

    def _build_camera_card(self, idx, cam):
        is_local = cam.get('is_local', False)
        if idx not in self.camera_stats: self.camera_stats[idx] = {'res': 'Detecting...', 'online': False}

        with ui.card().classes('bg-[#0d1b2a] border border-blue-900 p-0 w-80 overflow-hidden group'):
            with ui.row().classes('w-full justify-between items-center px-3 py-2 bg-blue-900/20'):
                with ui.column().classes('gap-0'):
                    ui.label(cam['name']).classes('text-[10px] font-bold text-blue-300 uppercase')
                    if is_local: ui.label('HARDWARE DIRECT').classes('text-[7px] text-amber-500 font-black')
                
                with ui.row().classes('gap-0 no-wrap'):
                    ui.button(icon='fullscreen', on_click=lambda: self.preview_modal(idx, cam)) \
                        .props('flat dense size=sm color=amber-400')
                    ui.button(icon='videocam', on_click=lambda: self.prompt_timed_record(idx, cam)) \
                        .props('flat dense size=sm color=red-400') \
                        .tooltip('Record Timed Clip')
                    if not is_local:
                        storage_idx = idx - len([c for c in self.get_all_cameras() if c.get('is_local')])
                        ui.button(icon='edit', on_click=lambda i=storage_idx: self.camera_dialog(i)) \
                            .props('flat dense size=sm color=blue-300')
                        ui.button(icon='delete', on_click=lambda i=storage_idx: self.delete_camera(i)) \
                            .props('flat dense size=sm color=red-400')

            ui.interactive_image(f'/camera_proxy/{idx}').classes('w-full h-48 bg-black')
            threading.Thread(target=self.verify_camera_connection, args=(idx, cam['url']), daemon=True).start()

    def verify_camera_connection(self, idx, url):
        if any(p.endswith(f"capture_{idx}.mp4") for p in os.listdir(Path.home()) if ".mp4" in p):
            return

        try:
            if url in camera_manager.streams:
                self.camera_stats[idx] = {'res': "Active", 'online': True}
                return

            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.camera_stats[idx] = {'res': f"{w}x{h}", 'online': True}
                cap.release()
        except:
            self.camera_stats[idx] = {'res': "Offline", 'online': False}

    def preview_modal(self, idx, cam_data):
        with ui.dialog() as diag, ui.card().classes('bg-black border border-blue-900 p-0 overflow-hidden').style('width: 90vw;'):
            with ui.row().classes('w-full items-center justify-between bg-[#0d1b2a] px-4 py-2 border-b border-blue-900'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('videocam', color='red-500').classes('text-xs animate-pulse')
                    ui.label(f"LIVE: {cam_data.get('name')}").classes('text-blue-200 font-mono text-sm font-bold')
                ui.button(icon='close', on_click=diag.close).props('flat color=white dense')
            ui.image(f'/camera_proxy/{idx}').classes('w-full bg-black')
        diag.open()

    def camera_dialog(self, index=None):
        cameras = app.storage.user.get('cameras', [])
        is_edit = index is not None
        curr = cameras[index] if is_edit else {'name': '', 'url': ''}

        with ui.dialog() as diag, ui.card().classes('bg-[#0a192f] border border-blue-800 w-96'):
            ui.label('CONFIG CAMERA').classes('text-blue-200 font-bold mb-2')
            name = ui.input('Name', value=curr['name']).props('dark dense outlined').classes('w-full mb-2')
            url = ui.input('URL / Source', value=curr['url']).props('dark dense outlined').classes('w-full mb-4')
            
            def save():
                data = {'name': name.value, 'url': url.value}
                if is_edit: cameras[index] = data
                else: cameras.append(data)
                app.storage.user['cameras'] = cameras
                self.refresh_cam_grid()
                diag.close()

            with ui.row().classes('w-full gap-2'):
                ui.button('CANCEL', on_click=diag.close).props('flat color=white').classes('grow')
                ui.button('SAVE', on_click=save).props('color=emerald-9').classes('grow')
        diag.open()

    def delete_camera(self, index):
        cameras = app.storage.user.get('cameras', [])
        if 0 <= index < len(cameras):
            cameras.pop(index)
            app.storage.user['cameras'] = cameras
            self.refresh_cam_grid()

    def prompt_timed_record(self, idx, cam_data):
        with ui.dialog() as diag, ui.card().classes('bg-[#0a192f] border border-red-900 p-4 w-96'):
            ui.label('RECORDING PARAMETERS').classes('text-red-500 font-black tracking-widest text-xs')
            
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_path = str(Path.home() / f"capture_{ts}.mkv")
            
            path_input = ui.input('Full Destination Path', value=default_path).props('dark dense outlined').classes('w-full mb-2')
            duration_input = ui.number('Seconds to Record', value=10, format='%.0f').props('dark dense outlined').classes('w-full mb-4')
            
            def start_task():
                raw_url = cam_data['url']
                source = int(raw_url) if str(raw_url).isdigit() else raw_url
                
                target_path = Path(path_input.value)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                recorder = VideoRecorder(source, str(target_path), duration_input.value)
                recorder.start()
                
                diag.close()
                ui.notify(f"Recording {duration_input.value}s to {target_path.name}", color='red-10', icon='fiber_manual_record')

            with ui.row().classes('w-full gap-2 mt-2'):
                ui.button('CANCEL', on_click=diag.close).props('flat color=white').classes('grow')
                ui.button('START', on_click=start_task).props('color=red-10').classes('grow')
        diag.open()

    def refresh_mqtt_grid(self):
        self.mqtt_grid.clear()
        servers = app.storage.user.get('mqtt_servers', [])
        
        if not servers:
            with self.mqtt_grid: ui.label('No brokers registered.').classes('text-gray-500 italic text-sm')
            return

        with self.mqtt_grid:
            for idx, server in enumerate(servers):
                self._build_mqtt_card(idx, server)

    def _build_mqtt_card(self, idx, server):
        with ui.card().classes('bg-[#0d1b2a] border border-amber-900/50 p-0 w-80 overflow-hidden'):
            with ui.row().classes('w-full justify-between items-center px-3 py-2 bg-amber-900/10'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('dns', color='amber-500').classes('text-xs')
                    ui.label(server.get('name', 'Unknown')).classes('text-[11px] font-bold text-amber-200 uppercase')
                
                with ui.row().classes('gap-0 no-wrap'):
                    ui.button(icon='edit', on_click=lambda i=idx: self.mqtt_dialog(i)) \
                        .props('flat dense size=sm color=blue-300')
                    ui.button(icon='delete', on_click=lambda i=idx: self.delete_mqtt(i)) \
                        .props('flat dense size=sm color=red-400')

            with ui.column().classes('p-3 gap-1'):
                self._info_row('HOST', server.get('url'))
                self._info_row('PORT', server.get('port'))
                self._info_row('TOPIC', server.get('topic'))
                
                user = server.get('username')
                if user:
                    self._info_row('USER', user)
                    ui.label('Password saved securely').classes('text-[9px] text-gray-600 italic mt-1')

    def _info_row(self, label, value):
        with ui.row().classes('w-full justify-between items-center'):
            ui.label(label).classes('text-[9px] text-gray-500 font-bold')
            ui.label(str(value)).classes('text-[10px] text-gray-300 font-mono')

    def mqtt_dialog(self, index=None):
        servers = app.storage.user.get('mqtt_servers', [])
        is_edit = index is not None
        curr = servers[index] if is_edit else {
            'name': '', 'url': '', 'port': '1883', 'topic': '#', 'username': '', 'password': ''
        }

        with ui.dialog() as diag, ui.card().classes('bg-[#0a192f] border border-amber-800 w-96'):
            ui.label('MQTT BROKER CONFIG').classes('text-amber-400 font-bold mb-2')
            
            name = ui.input('Display Name', value=curr['name']).props('dark dense outlined').classes('w-full')
            
            with ui.row().classes('w-full gap-2'):
                url = ui.input('Host / IP', value=curr['url']).props('dark dense outlined').classes('grow')
                port = ui.input('Port', value=curr['port']).props('dark dense outlined').classes('w-20')
            
            topic = ui.input('Default Topic', value=curr['topic']).props('dark dense outlined').classes('w-full')
            
            ui.separator().classes('bg-amber-900/30 my-2')
            
            user = ui.input('Username', value=curr['username']).props('dark dense outlined').classes('w-full')
            pwd = ui.input('Password', value=curr['password'], password=True).props('dark dense outlined').classes('w-full')

            def save():
                data = {
                    'name': name.value, 'url': url.value, 'port': port.value,
                    'topic': topic.value, 'username': user.value, 'password': pwd.value
                }
                if is_edit: servers[index] = data
                else: servers.append(data)
                app.storage.user['mqtt_servers'] = servers
                self.refresh_mqtt_grid()
                diag.close()

            with ui.row().classes('w-full gap-2 mt-2'):
                ui.button('CANCEL', on_click=diag.close).props('flat color=white').classes('grow')
                ui.button('SAVE BROKER', on_click=save).props('color=amber-9').classes('grow')
        diag.open()

    def delete_mqtt(self, index):
        servers = app.storage.user.get('mqtt_servers', [])
        if 0 <= index < len(servers):
            servers.pop(index)
            app.storage.user['mqtt_servers'] = servers
            self.refresh_mqtt_grid()

class PluginsPage:
    def __init__(self):
        self.scan_path = Path.home()
        self.container = None
        self.scan_btn = None
        self.build()
        
        # Auto-scan if empty
        if not plugin_manager.cached_scan:
            ui.timer(0.1, self.run_scan, once=True)

    def build(self):
        with ui.column().classes('w-full h-[calc(100vh-50px)] gap-0 p-0'):
            # Header
            with ui.row().classes('w-full items-center bg-[#0d1b2a] px-4 py-3 border-b border-blue-900'):
                ui.label('NATIVE PLUGIN INTEGRATION').classes('text-blue-200 font-black tracking-widest text-sm')
                ui.space()
                
                # --- TUTORIAL BUTTON ---
                ui.button('DEV GUIDE', icon='school', on_click=self.open_tutorial) \
                    .props('flat dense color=purple-300')
                
                self.scan_btn = ui.button('RE-SCAN DISK', icon='refresh', on_click=self.run_scan) \
                    .props('flat dense color=blue-400')

            # List Container
            self.container = ui.column().classes('w-full p-4 gap-4 overflow-y-auto')
            self.refresh_list()

    def open_tutorial(self):
        """Opens a dialog with plugin development instructions."""
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl bg-[#0a192f] border border-blue-500 text-white'):
            with ui.row().classes('w-full items-center justify-between mb-4'):
                ui.label('PLUGIN DEVELOPER GUIDE').classes('text-xl font-black text-blue-400 tracking-widest')
                ui.button(icon='close', on_click=dialog.close).props('flat dense color=white')

            with ui.row().classes('w-full gap-8'):
                # Left Column: Rules
                with ui.column().classes('flex-1 gap-4'):
                    ui.markdown('### 📋 Strict Requirements')
                    with ui.list().props('dense dark'):
                        def rule(icon, text, subtext, color='white'):
                            with ui.item():
                                with ui.item_section().props('avatar'):
                                    ui.icon(icon, color=color)
                                with ui.item_section():
                                    ui.item_label(text).classes(f'text-{color} font-bold')
                                    ui.item_label(subtext).classes('text-gray-400 text-xs')
                        
                        rule('check', 'Variable Name', "Must define: router = APIRouter()", 'green-400')
                        rule('login', 'Entry Point', "Must define: @router.page('/index')", 'green-400')
                        rule('extension', 'File Type', "Must be a standard .py file", 'blue-300')
                    
                    ui.separator().classes('bg-blue-900')
                    
                    ui.markdown('### 🚫 Forbidden Patterns')
                    with ui.list().props('dense dark'):
                        rule('block', 'Global Pages', "Do NOT use @ui.page(...)", 'red-400')
                        rule('block', 'Root Path', "Do NOT use @router.page('/')", 'red-400')

                # Right Column: Code Template
                with ui.column().classes('flex-1'):
                    ui.markdown('### 💻 Starter Template')
                    code = """from nicegui import APIRouter, ui

# 1. Define Router (Must be named 'router')
router = APIRouter()

# 2. Define Entry Point (Must be '/index')
@router.page('/index')
def main_plugin_page():
    ui.label('My Awesome Plugin')
    
    # Use relative paths for navigation
    ui.button('Go to API', on_click=lambda: ui.navigate.to('./api/data'))

# 3. Define APIs (Optional)
@router.get('/api/data')
def get_data():
    return {"status": "ok"}
"""
                    ui.code(code, language='python').classes('w-full bg-black/50 border border-gray-700 rounded p-2 text-xs')
            
            dialog.open()

    async def run_scan(self):
        self.scan_btn.disable()
        ui.notify("Scanning for plugins...", color='blue')
        await run.io_bound(plugin_manager.scan_for_plugins, self.scan_path)
        self.refresh_list()
        self.scan_btn.enable()

    def refresh_list(self):
        self.container.clear()
        if not plugin_manager.cached_scan:
            with self.container:
                ui.label('No plugins found matching requirements.').classes('text-gray-500 italic')
                ui.button('Check Requirements', icon='help', on_click=self.open_tutorial) \
                    .props('flat dense color=blue-400')
            return

        with self.container:
            for p in plugin_manager.cached_scan:
                self._render_row(p)

    def _render_row(self, plugin):
        p_id = plugin['id']
        is_mounted = plugin_manager.is_mounted(p_id)
        
        # Main Card
        with ui.card().classes('w-full max-w-3xl bg-[#0d1b2a] border border-blue-900/50 p-0 overflow-hidden'):
            
            # Top Row: Info & Controls
            with ui.row().classes('w-full items-center justify-between p-4 bg-blue-900/10'):
                with ui.row().classes('items-center gap-4'):
                    icon_color = 'emerald-400' if is_mounted else 'gray-600'
                    ui.icon('extension', color=icon_color).classes('text-2xl')
                    
                    with ui.column().classes('gap-0'):
                        ui.label(plugin['name']).classes('text-blue-200 font-bold text-lg')
                        if is_mounted:
                            ui.label(f"Entry: /plugins/{p_id}/index").classes('text-emerald-600 font-mono text-xs')
                        else:
                            ui.label(f"Source: {plugin['filename']}").classes('text-gray-500 font-mono text-xs')

                with ui.row().classes('items-center gap-2'):
                    if is_mounted:
                        ui.label('ACTIVE').classes('text-emerald-500 font-black text-xs tracking-widest px-2')
                        ui.button('UNMOUNT', icon='link_off', on_click=lambda: self.do_unmount(p_id)) \
                            .props('outline color=red-400 dense')
                    else:
                        ui.button('MOUNT', icon='add_link', on_click=lambda: self.do_mount(plugin)) \
                            .props('outline color=blue-400 dense')

            # Bottom Row: Endpoint Links
            if is_mounted:
                routes = plugin_manager.mounted_plugins[p_id].get('routes', [])
                if routes:
                    with ui.row().classes('w-full p-3 gap-2 bg-[#050a0f] border-t border-blue-900/30 wrap'):
                        ui.label('ENDPOINTS:').classes('text-[10px] font-bold text-gray-500 my-auto mr-2')
                        
                        for route in routes:
                            is_index = '/index' in route['path']
                            is_api = '/api/' in route['path']
                            
                            color = 'emerald' if is_index else ('amber' if is_api else 'blue')
                            icon = 'home' if is_index else ('data_object' if is_api else 'link')
                            
                            ui.button(route['name'], icon=icon, on_click=lambda u=route['path']: ui.navigate.to(u)) \
                                .props(f'flat dense size=sm color={color}-400') \
                                .classes('px-2 bg-white/5 hover:bg-white/10 rounded')

    def do_mount(self, plugin):
        if plugin_manager.mount_plugin(plugin):
            ui.notify(f"Mounted {plugin['name']}", color='green')
            self.refresh_list()

    def do_unmount(self, p_id):
        plugin_manager.unmount_plugin(p_id)
        ui.notify(f"Unmounted plugin: {p_id}", color='orange')
        self.refresh_list()

class SystemLogsPage:
    def __init__(self):
        self.last_seen_index = 0
        self.build()

    def build(self):
        with ui.column().classes('w-full h-[calc(100vh-50px)] gap-0 bg-[#050a0f]'):
            # Header
            with ui.row().classes('w-full items-center bg-[#0d1b2a] px-4 py-2 border-b border-blue-900'):
                ui.icon('terminal', color='emerald-400')
                ui.label('SYSTEM STDOUT').classes('text-blue-200 font-black tracking-widest text-xs')
                ui.space()
                ui.button('CLEAR BUFFER', on_click=self.clear_logs).props('flat dense color=red-400 size=sm')

            # Log Area
            self.log_container = ui.column().classes('w-full grow p-4 font-mono text-[11px] overflow-y-auto gap-1')
            
            # Initial Load
            self.refresh_logs()
            
            # Auto-update timer
            ui.timer(1.0, self.refresh_logs)

    def refresh_logs(self):
        with self.log_container:
            with sys_logger.lock:
                # To prevent re-rendering everything, we only show current deque state
                # For a more advanced version, you'd track indices
                if len(sys_logger.logs) == 0:
                    return
                
                self.log_container.clear()
                for entry in sys_logger.logs:
                    with ui.row().classes('gap-3 no-wrap hover:bg-white/5 w-full'):
                        ui.label(entry['time']).classes('text-gray-500 shrink-0')
                        ui.label(entry['msg']).classes('text-emerald-300 break-all')

    def clear_logs(self):
        with sys_logger.lock:
            sys_logger.logs.clear()
        self.log_container.clear()
        ui.notify("Internal log buffer cleared")

class DatabasePage:
    def __init__(self):
        self.build()

    def refresh_ui_state(self):
        """Updates UI elements based on real-time database and harvester state."""
        state = db_manager.check_system_deps()
        is_running = (state == "running")
        
        # Shutdown harvester if database service stops
        if not is_running and db_manager.is_bridging():
            db_manager.stop_bridge_logic()

        self.status_label.set_text(f"STATUS: {'ONLINE' if is_running else 'OFFLINE'}")
        self.harvester_card.set_visibility(is_running)
        self.query_ui.set_visibility(is_running)
        
        # Harvester Status Toggle Logic
        bridging = db_manager.is_bridging()
        self.bridge_status.set_text('ACTIVE' if bridging else 'INACTIVE')
        self.bridge_status.props(f"color={'emerald' if bridging else 'gray'}")
        self.bridge_btn.set_text('STOP HARVESTER' if bridging else 'START HARVESTER')
        self.bridge_btn.props(f"color={'red-10' if bridging else 'amber-9'} icon={'sensors_off' if bridging else 'sensors'}")

        # Engine Controls (Restores the Purge button)
        self.controls_container.clear()
        with self.controls_container:
            if state == "missing":
                ui.button('DEPLOY', icon='rocket', on_click=self.start_initialization).props('color=emerald-9')
            else:
                label = 'STOP' if is_running else 'START'
                color = 'red-400' if is_running else 'emerald-400'
                ui.button(label, icon='power' if not is_running else 'power_off', 
                          on_click=self.stop_db_service if is_running else self.start_db_service).props(f"flat border color={color}")
                ui.button('PURGE', icon='delete_forever', on_click=self.confirm_purge).props('flat border color=red-500')

    async def toggle_bridge(self):
        """Triggers persistent harvester logic in db_manager."""
        if db_manager.is_bridging():
            db_manager.stop_bridge_logic()
            ui.notify("Harvester Stopped", color='amber-9')
        else:
            idx = self.broker_sel.value
            if idx is not None:
                broker = app.storage.user.get('mqtt_servers', [])[idx]
                if db_manager.start_persistent_bridge(broker):
                    ui.notify(f"Harvester active on {broker['url']}", color='emerald-9')
            else:
                ui.notify("Select a broker profile", color='red-10')
        self.refresh_ui_state()

    def build(self):
        with ui.column().classes('w-full p-6 gap-4'):
            # --- ENGINE CONTROLS ---
            with ui.row().classes('w-full items-center justify-between bg-blue-900/10 p-4 rounded border border-blue-900'):
                with ui.column().classes('gap-0'):
                    ui.label('POSTGRESQL ENGINE').classes('text-blue-200 font-black tracking-widest')
                    self.status_label = ui.label('').classes('text-[10px] text-gray-400')
                self.controls_container = ui.row().classes('gap-2')

            # --- MQTT HARVESTER ---
            self.harvester_card = ui.card().classes('w-full bg-[#0a192f] border border-amber-900/50 p-4')
            with self.harvester_card:
                with ui.row().classes('w-full items-center justify-between mb-2'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('hub', color='amber-500')
                        ui.label('MQTT DEBUG HARVESTER').classes('text-amber-400 font-black tracking-widest text-xs')
                    self.bridge_status = ui.badge('INACTIVE', color='gray')

                with ui.row().classes('w-full gap-4 items-end'):
                    brokers = app.storage.user.get('mqtt_servers', [])
                    options = {i: f"{b['name']} ({b['url']})" for i, b in enumerate(brokers)}
                    self.broker_sel = ui.select(options=options, label='Broker Profile').props('dark dense outlined').classes('grow')
                    self.bridge_btn = ui.button('START HARVESTER', icon='sensors', on_click=self.toggle_bridge).props('color=amber-9')

            # --- SQL & LOGGING ---
            self.setup_card = ui.card().classes('w-full bg-black border border-emerald-900 p-0 overflow-hidden').style('display:none')
            with self.setup_card:
                self.log_area = ui.log().classes('w-full h-32 text-[10px] font-mono text-emerald-500 p-2')

            self.query_ui = ui.column().classes('w-full gap-4')
            with self.query_ui:
                with ui.card().classes('w-full bg-[#0d1b2a] border border-blue-900 p-4'):
                    ui.label('SQL DEBUG CONSOLE').classes('text-[10px] font-bold text-gray-500 mb-2')
                    self.query_input = ui.textarea(placeholder='SELECT * FROM detection_debug_logs ORDER BY timestamp DESC LIMIT 10;').props('dark filled').classes('w-full font-mono')
                    with ui.row().classes('w-full justify-end'):
                        ui.button('RUN SQL', icon='play_arrow', on_click=self.run_sql).props('color=emerald-9')

            self.results_container = ui.column().classes('w-full bg-black/40 rounded border border-gray-800 p-4')
            with self.results_container:
                self.results_table_spot = ui.column().classes('w-full')
                self.results_label = ui.label('Ready for query.').classes('text-gray-600 italic text-sm')

            self.refresh_ui_state()

    async def start_bridge(self):
        idx = self.broker_sel.value
        if idx is None:
            ui.notify("Select a broker profile", color='red-10')
            return
        
        broker = app.storage.user.get('mqtt_servers', [])[idx]
        topic = broker.get('topic', 'hailo/detections') # Use topic from registered broker config

        try:
            def on_message(client, userdata, msg):
                try:
                    import json
                    detections = json.loads(msg.payload.decode())
                    for d in detections:
                        db_manager.insert_detection(broker, d)
                except Exception as e: print(f"Parse Error: {e}")

            import paho.mqtt.client as mqtt_lib
            self.bridge_client = mqtt_lib.Client(mqtt_lib.CallbackAPIVersion.VERSION2)
            if broker.get('username'):
                self.bridge_client.username_pw_set(broker['username'], broker['password'])
            
            self.bridge_client.on_message = on_message
            self.bridge_client.connect(broker['url'], int(broker['port']), 60)
            self.bridge_client.subscribe(topic)
            self.bridge_client.loop_start()
            self.is_bridging = True
            ui.notify(f"Harvesting {topic} from {broker['name']}...", color='emerald-9')
        except Exception as e:
            ui.notify(f"Bridge Failed: {e}", color='red-10')

    async def start_initialization(self):
        self.setup_card.visible = True
        self.log_area.clear()
        await db_manager.run_setup_sequence(lambda msg: self.log_area.push(msg))
        self.refresh_ui_state() # Switches UI to 'Stopped' mode after deploy

    async def start_db_service(self):
        bin_dir = await db_manager.get_pg_bin()
        cmd = [
            f"{bin_dir}/pg_ctl", "-D", str(db_manager.db_path), "-l", "/tmp/pg_log", 
            "-o", f"-p 5433 -c unix_socket_directories='{db_manager.db_path}'", 
            "start"
        ]
        await asyncio.create_subprocess_exec(*cmd)
        await asyncio.sleep(1.5) # Wait for PID file creation
        self.refresh_ui_state()
        ui.notify("PostgreSQL Online", color='emerald-9')

    async def stop_db_service(self):
        self.setup_card.visible = True
        await db_manager.stop_database(lambda msg: self.log_area.push(msg))
        self.refresh_ui_state()
        ui.notify("PostgreSQL Offline", color='amber-9')

    async def run_sql(self):
        query = self.query_input.value
        if not query.strip(): return

        try:
            conn = psycopg2.connect(
                dbname='postgres', user=db_manager.admin_user, password=db_manager.admin_pass,
                host=str(db_manager.db_path), port=5433
            )
            cursor = conn.cursor()
            cursor.execute(query)

            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                self.display_results(columns, rows)
                conn.commit()
            else:
                ui.notify("Success", color='emerald')
                conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            ui.notify(f"SQL Error: {str(e)}", color='red-10')

    def display_results(self, columns, rows):
        self.results_label.visible = False
        self.results_table_spot.clear()
        with self.results_table_spot:
            col_defs = [{'name': c, 'label': c.upper(), 'field': c} for c in columns]
            row_data = [dict(zip(columns, r)) for r in rows]
            ui.table(columns=col_defs, rows=row_data).props('dark flat bordered').classes('w-full font-mono')

    def confirm_purge(self):
        """Confirmation popup for the Purge action."""
        with ui.dialog() as diag, ui.card().classes('bg-[#0a192f] border border-red-900 p-6'):
            ui.label('FACTORY RESET DATABASE?').classes('text-red-500 font-black text-lg')
            ui.label('This will stop the service and delete ALL data in ~/RemoteUtilsDatabase. This cannot be undone.').classes('text-gray-400 text-sm')
            with ui.row().classes('w-full justify-end mt-4 gap-2'):
                ui.button('CANCEL', on_click=diag.close).props('flat color=white')
                ui.button('CONFIRM PURGE', on_click=lambda: self.run_purge_sequence(diag)).props('color=red-10')
        diag.open()

    async def run_purge_sequence(self, dialog):
        dialog.close()
        self.setup_card.visible = True
        self.query_ui.visible = False
        
        def update_log(msg):
            self.log_area.push(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

        await db_manager.purge_database(update_log)
        ui.notify("Database Wiped", type='warning')
        ui.navigate.reload() # Refresh UI to show "Run Full Deployment" button again

    def execute(self, sql):
        # Placeholder for your psycopg2 logic
        ui.notify(f"Executing: {sql[:20]}...")

# --- Main Application Controller ---

# Global instance
sys_logger = LogCapture()
sys.stdout = sys_logger
sys.stderr = sys_logger

class PiManagerApp:
    def __init__(self):

        self.setup_routes()
        self.args = self._parse_args()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cli', nargs=3, metavar=('IP', 'USER', 'PWD'))
        parser.add_argument('--discover', nargs='+', metavar='USER PWD [NET]')
        parser.add_argument('--deploy', action='store_true')
        parser.add_argument('--destroy', action='store_true')
        parser.add_argument('--reboot', action='store_true')
        parser.add_argument('--gui', action='store_true')
        return parser.parse_args()

    def setup_routes(self):
        @ui.page('/')
        def index():
            ui.colors(primary='#1a237e', secondary='#0d1b2a', accent='#3f51b5', dark='#0a192f')
            if not app.storage.user.get('authenticated', False):
                self.show_login()
            else:
                DashboardLayout()
                FileBrowser()

        @ui.page('/resources') 
        def resources_page():
            if not app.storage.user.get('authenticated', False):
                ui.navigate.to('/')
            else:
                DashboardLayout()
                ResourcesManager()
        
        @ui.page('/plugins')
        def plugins_page():
             if not app.storage.user.get('authenticated', False):
                ui.navigate.to('/')
             else:
                DashboardLayout()
                PluginsPage()

        @app.get('/camera_proxy/{idx}')
        async def camera_proxy(idx: int):
            all_cameras = ResourcesManager.get_all_cameras() 
            
            if idx < 0 or idx >= len(all_cameras):
                return {"error": "Invalid index"}
            
            raw_url = all_cameras[idx]['url']
            source = int(raw_url) if raw_url.isdigit() else raw_url

            def generate_frames():
                while True:
                    frame_bytes = camera_manager.get_frame(source)
                    if frame_bytes:
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        time.sleep(0.1) 
                    time.sleep(0.03) 

            return StreamingResponse(generate_frames(), 
                                    media_type='multipart/x-mixed-replace; boundary=frame')
        
        @app.get('/status')
        def system_status():
            try:
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f: temp = round(int(f.read()) / 1000, 1)
            except: temp = None
            return {
                "cpu_percent": psutil.cpu_percent(),
                "ram_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "cpu_temp": temp,
                "timestamp": time.time()
            }
        
        @ui.page('/logs')
        def logs_page():
            if not app.storage.user.get('authenticated', False):
                ui.navigate.to('/')
            else:
                DashboardLayout()
                SystemLogsPage()
        
        @ui.page('/database')
        def database_route():
            if not app.storage.user.get('authenticated', False):
                ui.navigate.to('/')
            else:
                DashboardLayout()
                DatabasePage()

    def show_login(self):
        with ui.column().classes('w-full h-screen items-center justify-center bg-[#050a0f]'):
            with ui.card().classes('shadow-24 q-pa-lg border border-blue-900').style('background-color: #0a192f; width: 360px'):
                ui.label('SECURE PI ACCESS').classes('text-h6 text-white font-bold tracking-widest text-center w-full q-mb-md')
                u = ui.input('Username').classes('w-full').props('dark color=blue-400 outlined')
                p = ui.input('Password', password=True).classes('w-full').props('dark color=blue-400 outlined')
                async def handle_login():
                    if PiAuth.verify(u.value, p.value):
                        app.storage.user.update({'authenticated': True, 'username': u.value})
                        ui.navigate.reload()
                    else: ui.notify('AUTHENTICATION FAILED', color='red-10', icon='security')
                ui.button('LOGIN', on_click=handle_login).classes('w-full q-mt-md py-2').props('color=blue-800 elevated')

    def run_cli_logic(self):
        target_ips, user, pwd = [], None, None
        
        if self.args.discover:
            user, pwd = self.args.discover[0], self.args.discover[1]
            net = self.args.discover[2] if len(self.args.discover) > 2 else None
            target_ips = NetworkScanner(user, pwd, net).discover()
        elif self.args.cli:
            target_ips, user, pwd = [self.args.cli[0]], self.args.cli[1], self.args.cli[2]
        else:
            return False 

        for ip in target_ips:
            ctrl = RemoteController(ip, user, pwd)
            if self.args.destroy: ctrl.destroy()
            if self.args.deploy: ctrl.deploy()
            if self.args.reboot: ctrl.reboot()
        return True

    def start(self):
        is_cli_task = self.run_cli_logic()
        
        if not is_cli_task or self.args.gui:
            global gpio_manager
            gpio_manager = GPIOManager()

            ui.run(
                port=8080, 
                title="Pi Manager", 
                dark=True, 
                reload=False, 
                storage_secret='midnight_blue_heartbeat_secret',
                on_air="dJ8RUiKa7vRAHEjx"
            )

# --- Entry Point ---
if __name__ in {"__main__", "__mp_main__"}:
    pi_app = PiManagerApp()
    pi_app.start()