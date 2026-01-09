import sys
import os
import subprocess
import argparse
import socket
import concurrent.futures
import ipaddress
import time
import shutil
from typing import List, Optional
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
from starlette.responses import StreamingResponse
from collections import deque
from datetime import datetime
from api_integration import get_api_integration

# --- Dependency Management ---

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
mqtt_client = deps.auto_install("paho-mqtt", import_name="paho.mqtt.client") # <--- ADD THIS
ui, app = ng.ui, ng.app

# --- Logic Layer ---
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
        """Returns a single JPEG encoded frame from the requested source."""
        if source not in self.streams:
            # Open only if not already open
            cap = cv2.VideoCapture(source)
            self.streams[source] = cap
        
        cap = self.streams[source]
        success, frame = cap.read()
        if not success:
            return None
            
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buffer.tobytes()

camera_manager = GlobalCameraManager()

class GPIOManager:
    """Handles GPIO interactions with safety fallback for non-Pi environments."""
    _setup_done = False
    
    # Standard 40-Pin Header Map (Physical Pin -> BCM GPIO)
    # Types: 'pwr' (3.3/5V), 'gnd', 'gpio'
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
                f"[Service]\nExecStart=/usr/bin/python3 {self.remote_path} --gui\n"
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

# --- UI Layer ---

class SystemHeader:
    """Consolidated Header with combined Menu/User button and telemetry tooltips."""
    def __init__(self, drawer):
        self.drawer = drawer
        self.username = app.storage.user.get('username', 'USER').upper()
        self.cpu = self.ram = self.tmp = self.pwr = None
        self.build()

    def _metric_tack(self, icon, tooltip_text):
        with ui.column().classes('items-center gap-0 px-1 w-10 cursor-help'):
            ui.icon(icon, color='blue-300').classes('text-sm')
            lbl = ui.label('--').classes('text-[9px] font-bold text-white uppercase tracking-tighter')
            ui.tooltip(tooltip_text) 
            return lbl

    def build(self):
        with ui.header(elevated=True).classes('items-center py-1 px-4').style('background-color: #0a192f; border-bottom: 2px solid #1a237e'):
            # Grouped Username and Menu into one unified button
            with ui.button(on_click=lambda: self.drawer.toggle()).props('flat color=white dense no-caps').classes('q-pa-xs'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('menu')
                    ui.label(self.username).classes('text-sm font-black tracking-widest text-blue-200')
            
            ui.space()
            with ui.row().classes('items-center gap-1'):
                self.cpu = self._metric_tack('memory', 'CPU Load')
                self.ram = self._metric_tack('storage', 'RAM Usage')
                self.tmp = self._metric_tack('thermostat', 'Core Temp')
                self.pwr = self._metric_tack('bolt', 'Power Status')
        
        ui.timer(2.0, self.refresh_metrics)

    def refresh_metrics(self):
        self.cpu.set_text(f"{psutil.cpu_percent():.0f}%")
        self.ram.set_text(f"{psutil.virtual_memory().percent:.0f}%")
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                self.tmp.set_text(f"{int(f.read()) / 1000:.0f}°")
        except: self.tmp.set_text("N/A")
        self.pwr.set_text("AC")

class DashboardLayout:
    """A reusable layout wrapper that provides the Header and Sidebar without extra titles."""
    def __init__(self):
        ui.query('.q-page').classes('p-0')
        self.drawer = self.setup_sidebar()
        self.header = SystemHeader(self.drawer)

    def setup_sidebar(self):
        """Standardized sidebar with consistent left-aligned icons and text."""
        with ui.left_drawer().style('background-color: #0d1b2a; color: white').classes('column no-wrap p-0') as dr:
            
            # Common style for sidebar buttons
            btn_props = 'flat no-caps color=white dense'
            btn_classes = 'w-full justify-start px-4 py-3 text-xs font-medium hover:bg-blue-900/30 rounded-none'

            # --- Top Section: Navigation ---
            with ui.column().classes('grow w-full gap-0'):
                ui.label('SYSTEM').classes('text-[10px] font-bold px-5 py-3 text-blue-400 tracking-widest opacity-60')
                
                ui.button('File Manager', icon='folder_open', on_click=lambda: ui.navigate.to('/')) \
                    .props(btn_props).classes(btn_classes)
                
                ui.button('Camera Monitor', icon='videocam', on_click=lambda: ui.navigate.to('/cameras')) \
                    .props(btn_props).classes(btn_classes)

                ui.button('MQTT Inspector', icon='hub', on_click=lambda: ui.navigate.to('/mqtt')) \
                    .props(btn_props).classes(btn_classes)
                
                ui.button('GPIO Control', icon='settings_input_component', on_click=lambda: ui.navigate.to('/gpio')) \
                    .props(btn_props).classes(btn_classes)

                ui.button('API Navigator', icon='api', on_click=lambda: ui.navigate.to('/api-control')) \
                    .props(btn_props).classes(btn_classes + ' text-emerald-400')
            
            # --- Bottom Section: System Utilities ---
            with ui.column().classes('w-full pb-4 gap-0'):
                ui.separator().classes('bg-blue-900/30 mb-2')
                
                ui.button('Reboot Host', icon='restart_alt', on_click=self.confirm_reboot) \
                    .props(btn_props).classes(btn_classes + ' text-red-400 hover:bg-red-900/20')

                ui.button('Logout Session', icon='logout', on_click=self.logout) \
                    .props(btn_props).classes(btn_classes + ' text-blue-300')
                    
        return dr

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
        """Initial construction of the navigator container."""
        with ui.column().classes('w-full h-full gap-0 border-r border-blue-900/40'):
            self.header_row = ui.row().classes('w-full items-center bg-[#0d1b2a] px-2 py-1 border-b border-blue-900/50')
            self.content_area = ui.column().classes('w-full grow overflow-hidden bg-[#050a0f] gap-0')
            self.refresh_ui()

    def refresh_ui(self):
        """Swaps UI between Browser and Shell modes."""
        self.header_row.clear()
        self.content_area.clear()
        
        with self.content_area:
            self._setup_persistent_dialogs()

        if self.mode == "browser":
            self._build_browser_interface()
        else:
            self._build_shell_interface()

    def _build_browser_interface(self):
        """Constructs the standard file browser header with tooltips and Workspace Selector."""
        with self.header_row:
            with ui.row().classes('gap-1 items-center'):
                # Workspace Selector Button
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
                # Full system access starts at ~/ but allows going up to /
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
            # Boundary is always / for system mode, or the folder itself for workspaces
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
                # Fallback to Full Access mode
                self.set_workspace(Path("/"), ui_home=Path.home())
                self.open_workspace_manager() 
        except Exception as e: ui.notify(f"Error: {e}", color='red-10')

    def _build_shell_interface(self):
        """Integrated terminal view with a seamless, high-contrast console look."""
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
        """Downloads files or zips folders."""
        try:
            if item.is_file(): ui.download(str(item))
            else:
                target = shutil.make_archive(os.path.join(tempfile.gettempdir(), item.name), 'zip', item)
                ui.download(target, filename=f"{item.name}.zip")
        except Exception as e: ui.notify(f"Error: {e}", color='red-10')

    def handle_copy(self, item: Path):
        """Copies item to the opposite pane."""
        dest_pane = self.parent.right_pane if self.side_label == "LEFT" else self.parent.left_pane
        dest_dir = dest_pane.current_path
        dest_file = dest_dir / item.name
        
        try:
            if item.is_dir(): 
                shutil.copytree(str(item), str(dest_file), dirs_exist_ok=True)
            else: 
                shutil.copy2(str(item), str(dest_file))
            ui.notify("COPIED")
            dest_pane.refresh_file_list()
        except Exception as e: 
            ui.notify(f"Failed: {e}", color='red-10')

    def handle_move(self, item: Path):
        """Moves item to the opposite pane."""
        dest_pane = self.parent.right_pane if self.side_label == "LEFT" else self.parent.left_pane
        dest_dir = dest_pane.current_path
        dest_file = dest_dir / item.name
        
        try:
            shutil.move(str(item), str(dest_file))
            self.parent.left_pane.refresh_file_list()
            self.parent.right_pane.refresh_file_list()
        except Exception as e: ui.notify(f"Failed: {e}", color='red-10')

    def handle_delete_confirm(self, item: Path):
        """Confirms and executes deletion."""
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
        """Restored missing dialog setup method."""
        with ui.dialog() as self.upload_dialog, ui.card().classes('bg-[#0a192f] border border-blue-800'):
            ui.label(f'UPLOAD TO: {self.side_label}').classes('text-blue-200 text-xs font-bold')
            ui.upload(on_upload=self.handle_upload, multiple=True, max_file_size=500_000_000).classes('w-80').props('dark')
            ui.button('CLOSE', on_click=self.upload_dialog.close).props('flat color=white').classes('w-full')

    def refresh_file_list(self):
        """Populates file rows."""
        if not hasattr(self, 'list_area') or self.list_area is None: return
        self.list_area.clear()
        try:
            items = sorted(list(self.current_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
            with self.list_area:
                for item in items: self._build_row(item)
        except Exception as e:
            with self.list_area: ui.label(f"DENIED").classes('text-red-500 text-[10px] p-4')

    def _build_row(self, item: Path):
        """Builds a file row with the new View button."""
        with ui.row().classes('w-full items-center hover:bg-blue-900/20 px-3 py-0.5 border-b border-blue-900/10 group'):
            icon = 'folder' if item.is_dir() else 'description'
            ui.icon(icon, color='blue-400' if item.is_dir() else 'slate-500').classes('text-sm')
            lbl = ui.label(item.name).classes('grow text-[12px] cursor-pointer text-gray-300 truncate')
            
            if item.is_dir(): 
                lbl.on('click', lambda: self.navigate_to(item))
                lbl.tooltip('Open Folder')

            is_protected = item.name == "RemoteUtils.py" and item.parent == Path.home()

            with ui.row().classes('opacity-0 group-hover:opacity-100 transition-opacity gap-0'):
                # NEW: Viewer Button for files
                if item.is_file():
                    ui.button(icon='visibility', on_click=lambda: MediaViewer(item).open()).props('flat color=blue-200 dense size=xs').tooltip('View')

                if item.suffix in ['.py', '.sh']:
                    ui.button(icon='play_arrow', on_click=lambda: ExecutionDialog(item, self.current_path).open()).props('flat color=emerald-400 dense size=xs').tooltip('Run Script')
                
                ui.button(icon='download', on_click=lambda: self.handle_download(item)).props('flat color=blue-200 dense size=xs').tooltip('Download')
                ui.button(icon='content_copy', on_click=lambda: self.handle_copy(item)).props('flat color=amber-300 dense size=xs').tooltip('Copy to Other Pane')
                
                if not is_protected:
                    ui.button(icon='input', on_click=lambda: self.handle_move(item)).props('flat color=orange-400 dense size=xs').tooltip('Move to Other Pane')
                    ui.button(icon='delete', on_click=lambda: self.handle_delete_confirm(item)).props('flat color=red-500 dense size=xs').tooltip('Delete')
                else:
                    ui.icon('lock', color='blue-900').classes('text-xs self-center px-2').tooltip('Protected System File')

    def navigate_to(self, p: Path): self.current_path = p; self.refresh_ui()
    def go_back(self):
        # HARD BOUNDARY: Never go back past the designated boundary (/ or workspace root)
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

class FileBrowser:
    def __init__(self):
        with ui.row().classes('w-full h-[calc(100vh-50px)] gap-0 no-wrap flex-col md:flex-row bg-[#050a0f]'):
            with ui.column().classes('w-full md:w-1/2 h-1/2 md:h-full border-b md:border-b-0 md:border-r border-blue-900/40 overflow-hidden'):
                self.left_pane = FileNavigator("LEFT", self)
            with ui.column().classes('w-full md:w-1/2 h-1/2 md:h-full overflow-hidden'):
                self.right_pane = FileNavigator("RIGHT", self)

class ScriptInspector:
    # List of common argument names that likely refer to a camera source
    CAMERA_KEYWORDS = {'camera', 'source', 'input', 'cam'}

    @staticmethod
    def _get_val(node):
        """Safely extracts the value from an AST node."""
        try:
            # First try standard literal evaluation (works for strings, numbers, lists, tuples)
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            # Fallback for variables (e.g., argparse.REMAINDER) or objects
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return node.attr
            elif hasattr(node, 'id'): # Catch-all for older python versions
                return node.id
            return None

    @staticmethod
    def extract_args(item: Path):
        if item.suffix != '.py': return None
        args_found = {'positional': [], 'flags': []}
        try:
            tree = ast.parse(item.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument':
                    arg = ScriptInspector._parse_arg_node(node)
                    if not arg: continue

                    # Detect if this argument is likely a camera reference
                    clean_name = re.sub(r'^--?', '', arg['actual_name']).lower()
                    arg['is_camera_arg'] = clean_name in ScriptInspector.CAMERA_KEYWORDS
                    
                    if arg['is_flag']: args_found['flags'].append(arg)
                    else: args_found['positional'].append(arg)
            return args_found if any(args_found.values()) else None
        except Exception as e:
            print(f"Error parsing script {item.name}: {e}")
            return None

    @staticmethod
    def _parse_arg_node(node):
        # Extract the argument names (e.g., "-f", "--file")
        names = []
        for a in node.args:
            val = ScriptInspector._get_val(a)
            if isinstance(val, str) and val.startswith('-'):
                names.append(val)
        
        # If no flag names found, check if it's a positional argument defined by string
        if not names:
            for a in node.args:
                val = ScriptInspector._get_val(a)
                if isinstance(val, str):
                    names.append(val)

        if not names: return None
        
        actual_name = names[-1] # Use the last one (usually the long flag --verbose)
        
        # Extract keyword arguments
        kwargs = {}
        for k in node.keywords:
            val = ScriptInspector._get_val(k.value)
            
            # Handle list/tuple values specifically if literal_eval didn't catch them
            if k.arg in ['choices', 'metavar'] and isinstance(k.value, (ast.List, ast.Tuple)):
                if val is None:
                    val = [ScriptInspector._get_val(elt) for elt in k.value.elts]
            
            kwargs[k.arg] = val

        is_flag = actual_name.startswith('-')
        formatted = re.sub(r'^--?', '', actual_name).replace('-', ' ').replace('_', ' ').title()
        
        return {
            'actual_name': actual_name,
            'formatted_name': formatted,
            'help': kwargs.get('help', ''),
            'type': kwargs.get('type', 'str'),
            'action': kwargs.get('action', 'store'),
            'default': kwargs.get('default', None),
            'choices': kwargs.get('choices', None),
            'nargs': kwargs.get('nargs', None),
            'metavar': kwargs.get('metavar', None), # <--- THIS WAS MISSING
            'is_flag': is_flag
        }
       
class ExecutionDialog(ui.dialog):
    def __init__(self, item: Path, current_dir: Path):
        super().__init__()
        self.item = item
        self.current_dir = current_dir
        self.process = None
        self.arg_elements = {}
        self.log_container = None 
        self.log_scroll = None
        
        # MQTT State Management
        # Structure: { topic: {'client': client_obj, 'visible': bool} }
        self.mqtt_states = {} 
        self.topic_rows = {}       # {topic: [ui.row, ui.row, ...]}
        
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
        with self, ui.card().classes('bg-[#0a192f] border border-emerald-900 p-4 w-full max-w-[500px] overflow-hidden') as self.card:
            ui.label(f'RUN: {self.item.name}').classes('text-emerald-400 font-black text-lg mb-2')
            with ui.scroll_area().classes('w-full h-64 pr-2'):
                if detected:
                    self._build_section("REQUIRED PARAMETERS", detected['positional'], is_optional=False)
                    self._build_section("OPTIONAL FLAGS", detected['flags'], is_optional=True)
                else:
                    self.arg_elements['raw'] = {'input': ui.input('Manual Args').props('dark outline').classes('w-full'), 'type': 'raw'}
            with ui.row().classes('w-full justify-end mt-4 gap-2'):
                ui.button('CANCEL', on_click=self.close).props('flat color=white')
                ui.button('EXECUTE', on_click=self.assemble_and_run).props('color=emerald-9')

    def _build_section(self, label, args_list, is_optional):
        if not args_list: return
        ui.label(label).classes('text-[10px] text-emerald-600 font-bold mt-4 mb-2 tracking-widest')
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

            with ui.column().classes('w-full gap-1 mb-2'):
                with ui.row().classes('w-full items-start gap-2 no-wrap'):
                    cb = None
                    if is_optional: 
                        cb = ui.checkbox().props('dark dense color=emerald').style('margin-top: 12px')
                    
                    ui.label(arg['formatted_name']).classes('text-xs text-gray-300 w-24 truncate mt-3').tooltip(arg['help'])
                    
                    input_container = ui.column().classes('grow gap-1')
                    self.arg_elements[arg['actual_name']] = {
                        'check': cb, 
                        'container': input_container, 
                        'inputs': [], 
                        'type': 'flag' if is_optional else 'positional', 
                        'meta': arg
                    }
                    
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
                            .classes('mt-1')
                if cb: input_container.bind_visibility_from(cb, 'value')

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
        if arg.get('is_camera_arg'):
            all_cams = CameraView.get_all_cameras()
            if all_cams:
                options = {c['url']: f"📷 {c['name']}" for c in all_cams}
                return ui.select(options=options, value=all_cams[0]['url'], label=label).props('dark dense outline').classes('grow')
        if arg['choices']:
            opts = {val: str(val) for val in arg['choices']}
            return ui.select(options=opts, value=arg['default'] or arg['choices'][0], label=label) \
                     .props('dark dense outline use-input').classes('grow')
        default_val = str(arg['default']) if arg['default'] is not None else ''
        return ui.input(label=label, value=default_val).props('dark dense outline').classes('grow')

    def assemble_and_run(self):
        if 'raw' in self.arg_elements: 
            final_args = self.arg_elements['raw']['input'].value
        else:
            cmd_parts = []
            for name, el in self.arg_elements.items():
                values = []
                for inp in el['inputs']:
                    val = inp.value
                    if val is None or (isinstance(val, str) and val.strip() == ""): continue
                    if isinstance(val, list): values.extend([str(v) for v in val])
                    else:
                        try:
                            if el['meta'].get('type') == 'int': val = int(float(val))
                        except (ValueError, TypeError): pass
                        values.append(str(val))
                if not values: continue
                if el['type'] == 'positional': cmd_parts.extend(values)
                elif el['type'] == 'flag' and el['check'].value:
                    if el['meta']['action'] in ['store_true', 'store_false']: cmd_parts.append(name)
                    else:
                        cmd_parts.append(name)
                        cmd_parts.extend(values)
            final_args = " ".join(filter(None, cmd_parts))
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

    # --- MQTT MANAGEMENT ---
    def start_mqtt_listener(self, host, port, topic, user, password):
        try:
            import paho.mqtt.client as mqtt_lib
            context_client = ui.context.client

            # Persist existing visibility preference if re-subscribing
            is_visible = True
            if topic in self.mqtt_states:
                is_visible = self.mqtt_states[topic].get('visible', True)
                # Ensure old client is cleaned up if it exists
                old_client = self.mqtt_states[topic].get('client')
                if old_client:
                    try: old_client.loop_stop(); old_client.disconnect()
                    except: pass
            
            try:
                client = mqtt_lib.Client(mqtt_lib.CallbackAPIVersion.VERSION2)
            except AttributeError:
                client = mqtt_lib.Client()

            if user: client.username_pw_set(user, password)
            
            # --- FIX STARTS HERE ---
            # Automatically enable TLS for port 8883
            if int(port) == 8883:
                client.tls_set()
            # --- FIX ENDS HERE ---

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
            
            # Store client AND visibility state
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
            # CHANGED: Do NOT delete the entry. Just remove the client reference.
            # This keeps the 'visible' state and allows toggling to work on old logs.
            self.mqtt_states[topic]['client'] = None

    def stop_all_listeners(self):
        # Iterate over keys safely
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
        
        # Check visibility from persistent state
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

    # --- LOG PARSING & RENDERING ---
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
                        # Pass event 'e' to lambda to handle click event properly
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

    async def run_process(self, args_str: str):
        abs_path = str(self.item.absolute())
        cmd = ['python3', abs_path] if self.item.suffix == '.py' else ['bash', abs_path]
        if args_str: cmd.extend(args_str.split())
        
        try:
            self.append_log_line(f"--- Launching: {' '.join(cmd)} ---\n")
            self.process = await asyncio.create_subprocess_exec(
                *cmd, 
                stdout=asyncio.subprocess.PIPE, 
                stderr=asyncio.subprocess.STDOUT, 
                cwd=str(self.current_dir)
            )
            
            while True:
                line_bytes = await self.process.stdout.readline()
                if not line_bytes: break
                line = line_bytes.decode().rstrip()
                self.append_log_line(line)
            
            await self.process.wait()
            self.append_log_line(f"\n[DONE] Exit Code: {self.process.returncode}")
        except Exception as e:
            if self.log_container: self.append_log_line(f"\n[SYSTEM ERROR] {str(e)}")
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

class CameraView:
    @staticmethod
    def get_all_cameras():
        """Returns a combined list of local and remote cameras without touching the UI."""
        # 1. Discover local hardware
        local = []
        for i in [0, 1]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                local.append({'name': f'PI-CAM {i}', 'url': str(i), 'is_local': True})
                cap.release()
        
        # 2. Get remote cameras from storage
        remote = app.storage.user.get('cameras', [])
        return local + remote
    
    def __init__(self):
        if 'cameras' not in app.storage.user:
            app.storage.user['cameras'] = []
        
        self.camera_stats = {}
        self.build()

    def discover_local_cameras(self):
        """Checks for /dev/video devices and returns them as camera objects."""
        found = []
        # Common local device indices to check
        for i in [0, 1]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                found.append({
                    'name': f'PI-CAMERA {i}' if i == 0 else f'USB-CAM {i}',
                    'url': str(i), # We use the integer index as the URL string
                    'is_local': True
                })
                cap.release()
        return found

    def build(self):
        with ui.column().classes('w-full p-6 gap-6'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('CAMERA NETWORK').classes('text-blue-200 text-xl font-black tracking-tighter')
                ui.button('REGISTER REMOTE', icon='add', on_click=lambda: self.camera_dialog()) \
                    .props('flat color=emerald-400 border')

            self.grid = ui.row().classes('w-full gap-4')
            self.refresh_grid()

    def refresh_grid(self):
        self.grid.clear()
        
        # FIX: Call get_all_cameras() fresh. 
        # This automatically merges Local + Remote (Storage) correctly.
        all_cameras = self.get_all_cameras()
        
        if not all_cameras:
            with self.grid:
                ui.label('No cameras detected.').classes('text-gray-500 italic mt-10 w-full text-center')
            return

        with self.grid:
            for idx, cam in enumerate(all_cameras):
                self._build_camera_card(idx, cam)

    def _build_camera_card(self, idx, cam):
        is_local = cam.get('is_local', False)
        stats = self.camera_stats.setdefault(idx, {'res': 'Detecting...', 'online': False})
        
        with ui.card().classes('bg-[#0d1b2a] border border-blue-900 p-0 w-80 overflow-hidden group'):
            # --- Header ---
            with ui.row().classes('w-full justify-between items-center px-3 py-2 bg-blue-900/20'):
                with ui.column().classes('gap-0'):
                    ui.label(cam['name']).classes('text-[10px] font-bold text-blue-300 uppercase')
                    # Local Indicator
                    if is_local:
                        ui.label('HARDWARE DIRECT').classes('text-[7px] text-amber-500 font-black')
                
                with ui.row().classes('gap-0 no-wrap'):
                    ui.button(icon='fullscreen', on_click=lambda: self.preview_modal(idx, cam)) \
                        .props('flat dense size=sm color=amber-400')
                    
                    if not is_local:
                        ui.button(icon='edit', on_click=lambda: self.camera_dialog(idx - len(self.local_cameras))) \
                            .props('flat dense size=sm color=blue-300')
                        ui.button(icon='delete', on_click=lambda: self.delete_camera(idx - len(self.local_cameras))) \
                            .props('flat dense size=sm color=red-400')
                    else:
                        ui.icon('memory', color='blue-800').classes('p-2')

            # Preview
            ui.interactive_image(f'/camera_proxy/{idx}').classes('w-full h-48 bg-black')
            
            # Backend check
            threading.Thread(target=self.verify_camera_connection, args=(idx, cam['url']), daemon=True).start()

    def process_detection(self, idx, event):
        """Extracts resolution from the proxy stream metadata."""
        try:
            # Check if event.args['target'] exists (common in interactive_image loads)
            target = event.args.get('target', {})
            width = target.get('naturalWidth', 0)
            height = target.get('naturalHeight', 0)
            
            if width > 0:
                self.camera_stats[idx].update({'res': f"{width}x{height}", 'online': True})
            else:
                # If width is 0, browser loaded the 'broken' icon
                self.mark_offline(idx)
        except:
            self.mark_offline(idx)

    def verify_camera_connection(self, idx, url):
        """Backend check using OpenCV to support RTSP and get real resolution."""
        try:
            # Attempt to open the stream (RTSP, HTTP, etc.)
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                self.camera_stats[idx]['online'] = True
                self.camera_stats[idx]['res'] = f"{width}x{height}"
                cap.release()
            else:
                self.mark_offline(idx)
        except Exception:
            self.mark_offline(idx)

    def preview_modal(self, idx, cam_data):
        """Full-screen preview using the proxy with a structured header."""
        proxy_url = f'/camera_proxy/{idx}'
        
        # Extract details safely from the dictionary
        camera_name = cam_data.get('name', f'Camera {idx}')
        camera_source = cam_data.get('url', 'Local Device')

        with ui.dialog() as diag, ui.card().classes('bg-black border border-blue-900 p-0 overflow-hidden').style('width: 90vw;'):
            # --- Structured Header ---
            with ui.row().classes('w-full items-center justify-between bg-[#0d1b2a] px-4 py-2 border-b border-blue-900'):
                with ui.column().classes('gap-0'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('videocam', color='red-500').classes('text-xs animate-pulse')
                        ui.label(f"LIVE: {camera_name}").classes('text-blue-200 font-mono text-sm font-bold tracking-wider')
                    
                    # Display the URL/Source in a smaller, truncated font
                    ui.label(camera_source).classes('text-[10px] text-gray-500 font-mono truncate max-w-[60vw]')
                
                ui.button(icon='close', on_click=diag.close).props('flat color=white dense')
            
            # --- Video Feed ---
            ui.image(proxy_url).classes('w-full bg-black')
        diag.open()

    def camera_dialog(self, index=None):
        is_edit = index is not None
        cameras = app.storage.user.get('cameras', [])
        curr = cameras[index] if is_edit else {'name': '', 'url': ''}

        with ui.dialog() as diag, ui.card().classes('bg-[#0a192f] border border-blue-800 w-96'):
            ui.label('CONFIG CAMERA').classes('text-blue-200 font-bold mb-2')
            name_input = ui.input('Camera Name', value=curr['name']).props('dark dense outlined').classes('w-full mb-2')
            url_input = ui.input('Source URL', value=curr['url']).props('dark dense outlined').classes('w-full mb-4')
            
            async def save():
                data = {'name': name_input.value, 'url': url_input.value}
                if is_edit: cameras[index] = data
                else: cameras.append(data)
                app.storage.user['cameras'] = cameras
                self.refresh_grid()
                diag.close()

            with ui.row().classes('w-full gap-2'):
                ui.button('CANCEL', on_click=diag.close).props('flat color=white').classes('grow')
                ui.button('SAVE', on_click=save).props('color=emerald-9').classes('grow')
        diag.open()

    def delete_camera(self, index):
        cameras = app.storage.user.get('cameras', [])
        cameras.pop(index)
        app.storage.user['cameras'] = cameras
        if index in self.camera_stats: del self.camera_stats[index]
        self.refresh_grid()

class MQTTInspector:
    def __init__(self):
        self.client = None
        self.is_connected = False
        self.messages = deque(maxlen=200) # Increased buffer
        self.container = None
        
        # Default defaults
        self.conn_details = {
            'host': 'mqtt.portabo.cz', 
            'port': '1883',
            'topic': '#',
            'user': '',
            'pass': ''
        }
        
        self.build()

    def build(self):
        # Root container that switches content completely
        self.container = ui.column().classes('w-full h-[calc(100vh-50px)] p-0 m-0 bg-[#050a0f]')
        self.render_login()
        
        # Keep timer running but valid only when connected (Aggressive 100ms refresh)
        self.refresh_timer = ui.timer(0.1, self.update_log_ui)

    def render_login(self):
        """Phase 1: Configuration Form"""
        self.container.clear()
        # Center alignment for login
        self.container.classes('items-center justify-center')
        self.container.classes(remove='items-start justify-start')

        with self.container:
            with ui.card().classes('bg-[#0a192f] border border-blue-800 w-96 p-6 shadow-2xl'):
                with ui.row().classes('w-full items-center justify-center mb-4 gap-2'):
                    ui.icon('hub', color='amber-400').classes('text-3xl')
                    ui.label('MQTT CONSOLE').classes('text-amber-400 text-xl font-black tracking-widest')
                
                # Connection Form
                self.host_in = ui.input('Broker Host / IP', value=self.conn_details['host']).props('dark dense outlined').classes('w-full')
                
                with ui.row().classes('w-full gap-2'):
                    self.port_in = ui.input('Port', value=self.conn_details['port']).props('dark dense outlined').classes('grow')
                    self.topic_in = ui.input('Topic', value=self.conn_details['topic']).props('dark dense outlined').classes('grow')
                
                ui.separator().classes('bg-blue-900/50 my-2')
                
                self.user_in = ui.input('Username', value=self.conn_details['user']).props('dark dense outlined').classes('w-full')
                self.pass_in = ui.input('Password', value=self.conn_details['pass'], password=True).props('dark dense outlined').classes('w-full')
                
                self.connect_btn = ui.button('ESTABLISH CONNECTION', on_click=self.attempt_connect) \
                    .props('color=emerald-9 icon=login').classes('w-full mt-4 py-2 font-bold')

    def render_listener(self):
        """Phase 2: Full Screen Message Stream"""
        self.container.clear()
        # Full width alignment for stream
        self.container.classes('items-start justify-start')
        self.container.classes(remove='items-center justify-center')
        
        with self.container:
            # Sticky Header Bar
            with ui.row().classes('w-full items-center justify-between bg-[#0d1b2a] px-4 py-2 border-b border-blue-900 h-12 shrink-0'):
                with ui.row().classes('items-center gap-2'):
                    ui.spinner('dots', size='sm', color='emerald-400')
                    ui.label(f"LISTENING: {self.conn_details['host']}").classes('text-emerald-400 font-mono font-bold text-xs')
                    ui.label(f"[{self.conn_details['topic']}]").classes('text-gray-500 font-mono text-[10px]')
                
                with ui.row().classes('gap-2'):
                    ui.button('CLEAR', on_click=self.clear_log).props('flat color=blue-300 dense size=sm icon=delete_sweep')
                    ui.button('DISCONNECT', on_click=self.disconnect).props('color=red-9 dense size=sm icon=logout')

            # Scrollable Message Area
            with ui.scroll_area().classes('w-full grow bg-black/50 p-4') as self.scroll_area:
                self.msg_container = ui.column().classes('w-full gap-1')

    async def attempt_connect(self):
        """Async wrapper to prevent UI freeze during DNS/Auth checks."""
        self.connect_btn.props('loading')
        
        # Save inputs to memory
        self.conn_details.update({
            'host': self.host_in.value.strip(),
            'port': self.port_in.value.strip(),
            'topic': self.topic_in.value.strip(),
            'user': self.user_in.value.strip(),
            'pass': self.pass_in.value.strip()
        })

        try:
            # Run the blocking network call in a separate thread
            await asyncio.get_running_loop().run_in_executor(None, self._connect_sync)
            
            self.is_connected = True
            ui.notify(f"Connected to {self.conn_details['host']}", color='emerald')
            self.render_listener()
            
        except Exception as e:
            ui.notify(f"Connection Failed: {e}", color='red-10', timeout=0, close_button='OK')
        finally:
            if not self.is_connected:
                self.connect_btn.props(remove='loading')

    def _connect_sync(self):
        """Blocking connection logic."""
        self.client = mqtt_client.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_message = self.on_mqtt_message
        
        if self.conn_details['user']:
            self.client.username_pw_set(self.conn_details['user'], self.conn_details['pass'])
        
        # Paho handles both IP and Hostname resolution here
        self.client.connect(self.conn_details['host'], int(self.conn_details['port']), 10)
        self.client.subscribe(self.conn_details['topic'])
        self.client.loop_start()

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        self.is_connected = False
        self.messages.clear()
        ui.notify("Disconnected", color='amber')
        self.render_login()

    def on_mqtt_message(self, client, userdata, msg):
        try: payload = msg.payload.decode()
        except: payload = f"<Binary: {len(msg.payload)} bytes>"
        
        self.messages.appendleft({
            'time': datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'topic': msg.topic,
            'payload': payload,
            'qos': msg.qos
        })

    def update_log_ui(self):
        """Aggressive 100ms refresh that only runs if connected."""
        if not self.is_connected or not hasattr(self, 'msg_container'): return
        
        self.msg_container.clear()
        with self.msg_container:
            for msg in self.messages:
                with ui.row().classes('w-full items-start gap-3 p-1 border-b border-blue-900/20 hover:bg-white/5 no-wrap font-mono text-[11px] group'):
                    ui.label(msg['time']).classes('text-gray-500 min-w-[70px] select-none')
                    ui.label(msg['topic']).classes('text-amber-500 font-bold min-w-[120px] max-w-[200px] break-all')
                    ui.label(msg['payload']).classes('text-emerald-300 break-all grow')
                    ui.badge(f"Q{msg['qos']}", color='blue-900').props('rounded').classes('text-[9px]')

    def clear_log(self):
        self.messages.clear()
        self.msg_container.clear()

class APINavigator:
    """
    Comprehensive API control interface with live status monitoring
    """
    def __init__(self, api_integration):
        self.api = api_integration
        self.status_timer = None
        self.build()
        
    def build(self):
        with ui.column().classes('w-full h-[calc(100vh-50px)] bg-[#050a0f] p-6 gap-6 overflow-y-auto'):
            # Header
            with ui.row().classes('w-full items-center justify-between mb-4'):
                with ui.row().classes('items-center gap-3'):
                    ui.icon('api', color='emerald-400').classes('text-3xl')
                    ui.label('API NAVIGATOR').classes('text-emerald-400 text-xl font-black tracking-widest')
                ui.button('REFRESH STATUS', icon='refresh', on_click=self.refresh_all_status) \
                    .props('flat color=blue-400')
            # Status Overview Card
            with ui.card().classes('bg-[#0a192f] border border-blue-900 p-4 w-full'):
                ui.label('SYSTEM STATUS').classes('text-blue-300 font-bold text-sm mb-3')
                with ui.row().classes('w-full gap-4'):
                    # Pipeline Status
                    with ui.column().classes('gap-1'):
                        ui.label('AI Pipeline').classes('text-[10px] text-gray-500 uppercase')
                        self.pipeline_status = ui.label('Checking...').classes('text-emerald-400 font-mono text-sm')
                    # FPS Counter
                    with ui.column().classes('gap-1'):
                        ui.label('FPS').classes('text-[10px] text-gray-500 uppercase')
                        self.fps_label = ui.label('--').classes('text-amber-400 font-mono text-sm')
                    # Object Count
                    with ui.column().classes('gap-1'):
                        ui.label('Tracked Objects').classes('text-[10px] text-gray-500 uppercase')
                        self.count_label = ui.label('--').classes('text-blue-400 font-mono text-sm')
            # Servo Control Section
            with ui.card().classes('bg-[#0a192f] border border-emerald-900 p-4 w-full'):
                ui.label('SERVO CONTROL').classes('text-emerald-300 font-bold text-sm mb-3')
                with ui.row().classes('w-full gap-4 items-end'):
                    self.angle_input = ui.number('Angle (degrees)', value=0, min=-180, max=180) \
                        .props('dark outlined dense').classes('grow')
                    self.speed_input = ui.number('Speed (%)', value=50, min=0, max=100) \
                        .props('dark outlined dense').classes('w-32')
                    ui.button('MOVE', icon='rotate_right', on_click=self.move_servo) \
                        .props('color=emerald-9').classes('px-6')
                # Quick Angle Presets
                with ui.row().classes('w-full gap-2 mt-2'):
                    ui.label('Quick Angles:').classes('text-xs text-gray-500 mr-2')
                    for angle in [-90, -45, 0, 45, 90]:
                        ui.button(f'{angle}°', on_click=lambda a=angle: self.quick_angle(a)) \
                            .props('flat dense size=sm color=blue-400')
            # Camera Stream Section
            with ui.card().classes('bg-[#0a192f] border border-blue-900 p-4 w-full'):
                ui.label('CAMERA STREAM').classes('text-blue-300 font-bold text-sm mb-3')
                with ui.row().classes('w-full gap-4'):
                    ui.button('Open HLS Stream', icon='videocam', on_click=lambda: ui.navigate.to('/camera/stream/hls', new_tab=True)) \
                        .props('color=blue-9').classes('grow')
                    ui.button('Open Raw Stream', icon='video_library', on_click=lambda: ui.navigate.to('/camera/stream/raw', new_tab=True)) \
                        .props('flat color=blue-400').classes('grow')
            # API Endpoints Reference
            with ui.card().classes('bg-[#0a192f] border border-gray-800 p-4 w-full'):
                ui.label('API ENDPOINTS').classes('text-gray-400 font-bold text-sm mb-3')
                endpoints = [
                    {'method': 'POST', 'path': '/api/servo/rotate', 'desc': 'Rotate servo to angle'},
                    {'method': 'GET', 'path': '/api/pipeline/status', 'desc': 'Get AI pipeline status'},
                    {'method': 'GET', 'path': '/api/camera/stream', 'desc': 'Camera video stream'},
                    {'method': 'POST', 'path': '/api/pipeline/start', 'desc': 'Start AI processing'},
                    {'method': 'POST', 'path': '/api/pipeline/stop', 'desc': 'Stop AI processing'},
                ]
                for ep in endpoints:
                    with ui.row().classes('w-full items-center gap-3 py-2 border-b border-gray-800/50 hover:bg-white/5 font-mono'):
                        ui.badge(ep['method'], color='emerald' if ep['method'] == 'GET' else 'amber') \
                            .props('rounded').classes('text-[9px] w-14')
                        ui.label(ep['path']).classes('text-blue-400 text-xs grow')
                        ui.label(ep['desc']).classes('text-gray-500 text-[10px]')
            # Log Output
            with ui.card().classes('bg-black border border-gray-900 p-0 w-full'):
                with ui.row().classes('w-full items-center justify-between bg-[#0d1b2a] px-4 py-2 border-b border-gray-900'):
                    ui.label('API LOG').classes('text-gray-400 font-mono text-xs font-bold')
                    ui.button(icon='delete_sweep', on_click=lambda: self.log_area.clear()) \
                        .props('flat dense size=sm color=gray-500')
                self.log_area = ui.log().classes('w-full h-48 text-emerald-400 font-mono text-[11px] bg-black')
        # Start auto-refresh timer
        self.status_timer = ui.timer(2.0, self.refresh_all_status)

    async def move_servo(self):
        try:
            angle = int(self.angle_input.value)
            speed = int(self.speed_input.value)
            self.log_area.push(f'> Moving servo to {angle}° at {speed}% speed...')
            result = await self.api.servo.rotate_degrees(angle, speed)
            self.log_area.push(f'✓ Servo moved successfully')
            ui.notify(f'Servo moved to {angle}°', color='positive')
        except Exception as e:
            self.log_area.push(f'✗ Error: {str(e)}')
            ui.notify(f'Error: {str(e)}', color='negative')

    def quick_angle(self, angle):
        self.angle_input.value = angle
        
    async def refresh_all_status(self):
        try:
            if self.api.pipeline:
                fps = self.api.pipeline.fps_actual
                total = self.api.pipeline.counter.get_total_count()        
                self.pipeline_status.text = 'ACTIVE'
                self.pipeline_status.classes('text-emerald-400')
                self.fps_label.text = f'{fps:.1f}'
                self.count_label.text = str(total)
            else:
                self.pipeline_status.text = 'INACTIVE'
                self.pipeline_status.classes('text-red-400')
                self.fps_label.text = '--'
                self.count_label.text = '--'
        except Exception as e:
            self.log_area.push(f'Status check error: {str(e)}')
        
class GPIOPage:
    def __init__(self):
        self.refresh_timer = None
        self.pin_elements = {} # Map to update UI without full redraw
        self.build()

    def build(self):
        with ui.column().classes('w-full h-[calc(100vh-50px)] bg-[#050a0f] p-6 items-center'):
            # Header
            with ui.row().classes('w-full max-w-4xl items-center justify-between mb-6'):
                with ui.row().classes('items-center gap-3'):
                    ui.icon('settings_input_component', color='emerald-400').classes('text-3xl')
                    with ui.column().classes('gap-0'):
                        ui.label('GPIO INTERFACE').classes('text-emerald-400 text-xl font-black tracking-widest')
                        ui.label('BCM MODE ACTIVE').classes('text-blue-400 text-[10px] font-bold')
                
                # Simulation Warning
                if gpio_manager.simulated:
                    ui.label('⚠️ SIMULATION MODE').classes('text-amber-400 font-bold border border-amber-400 px-2 py-1 rounded text-xs animate-pulse')

            # The 40-Pin Board Layout
            with ui.card().classes('bg-[#0a192f] border border-blue-900 p-4 rounded-xl shadow-2xl'):
                with ui.row().classes('gap-8'):
                    # Left Column (Odd Pins)
                    with ui.column().classes('gap-1'):
                        for pin_num in range(1, 41, 2):
                            self._render_pin_row(pin_num, align_right=True)
                    
                    # Center Divider (Physical Pin Numbers)
                    with ui.column().classes('gap-1 items-center justify-around h-full py-2'):
                        for pin_num in range(1, 41, 2):
                            with ui.row().classes('gap-4 items-center'):
                                ui.label(str(pin_num)).classes('text-gray-500 font-mono text-[10px] w-4 text-center')
                                ui.label(str(pin_num+1)).classes('text-gray-500 font-mono text-[10px] w-4 text-center')

                    # Right Column (Even Pins)
                    with ui.column().classes('gap-1'):
                        for pin_num in range(2, 42, 2):
                            self._render_pin_row(pin_num, align_right=False)

        # Poll for input changes every 500ms
        self.refresh_timer = ui.timer(0.5, self.update_states)

    def _render_pin_row(self, pin_num, align_right):
        data = GPIOManager.PIN_MAP[pin_num]
        bcm = data['bcm']
        
        row_classes = 'items-center gap-2 p-1 rounded min-w-[280px] h-8 transition-colors '
        row_classes += 'justify-end' if align_right else 'justify-start'
        
        # Determine styling based on type
        if data['type'] == 'pwr':
            bg_class = 'bg-red-900/20 border border-red-900/30'
            text_color = 'text-red-400'
            icon = 'bolt'
        elif data['type'] == 'gnd':
            bg_class = 'bg-gray-800/50 border border-gray-700/30'
            text_color = 'text-gray-500'
            icon = 'grounding' # or minimize
        else: # GPIO
            bg_class = 'bg-blue-900/20 border border-blue-800/30 hover:bg-blue-800/30'
            text_color = 'text-blue-300'
            icon = 'circle'

        with ui.row().classes(row_classes + ' ' + bg_class) as row:
            # Logic for Left-aligned (Odd) vs Right-aligned (Even) visual flow
            
            elements = []
            
            # 1. Label Section
            label_div = ui.label(data['label']).classes(f'text-[10px] font-bold font-mono {text_color} uppercase tracking-tight')
            
            # 2. Controls (Only for GPIO)
            controls_div = ui.row().classes('gap-1 items-center')
            status_indicator = None
            
            if data['type'] == 'gpio':
                # Mode Toggle (IN/OUT)
                mode_switch = ui.switch(on_change=lambda e, b=bcm: self.toggle_mode(b, e.value)) \
                    .props('dense size=xs color=emerald').classes('scale-75')
                mode_switch.tooltip('Toggle IN/OUT')
                
                # State Toggle (High/Low) - Visibility depends on mode
                state_btn = ui.button().props('flat dense size=xs icon=power_settings_new') \
                    .on('click', lambda e, b=bcm: self.toggle_state(b))
                
                # Status Light
                status_indicator = ui.icon('circle', size='xs').classes('text-gray-600 transition-colors duration-300')
                
                with controls_div:
                    if align_right: 
                        status_indicator.move(controls_div)
                        state_btn.move(controls_div)
                        mode_switch.move(controls_div)
                    else:
                        mode_switch.move(controls_div)
                        state_btn.move(controls_div)
                        status_indicator.move(controls_div)

                # Store refs for updates
                self.pin_elements[bcm] = {
                    'indicator': status_indicator,
                    'btn': state_btn,
                    'switch': mode_switch,
                    'state': 0,
                    'is_out': False
                }

            # Layout Ordering based on side
            if align_right:
                controls_div.move(row)
                label_div.move(row)
            else:
                label_div.move(row)
                controls_div.move(row)

    def toggle_mode(self, bcm, is_out):
        mode = 'OUT' if is_out else 'IN'
        gpio_manager.set_pin_mode(bcm, mode)
        
        el = self.pin_elements[bcm]
        el['is_out'] = is_out
        
        # Update UI visibility
        el['btn'].visible = is_out
        if not is_out:
            el['btn'].props('color=gray-600')

    def toggle_state(self, bcm):
        el = self.pin_elements[bcm]
        new_val = 1 if el['state'] == 0 else 0
        gpio_manager.set_pin_value(bcm, new_val)
        self.update_single_pin(bcm)

    def update_states(self):
        """Polls hardware for current states."""
        for bcm, el in self.pin_elements.items():
            self.update_single_pin(bcm)

    def update_single_pin(self, bcm):
        val = gpio_manager.get_pin_status(bcm)
        el = self.pin_elements[bcm]
        el['state'] = val
        
        # Visual updates
        color = 'emerald-400' if val else 'gray-700'
        shadow = 'drop-shadow-[0_0_5px_rgba(52,211,153,0.8)]' if val else ''
        
        el['indicator'].classes(remove='text-emerald-400 text-gray-700 drop-shadow-[0_0_5px_rgba(52,211,153,0.8)]')
        el['indicator'].classes(f'text-{color} {shadow}')
        
        # Button color if output
        if el['is_out']:
             btn_color = 'emerald-400' if val else 'red-400'
             el['btn'].props(f'color={btn_color}')

# --- Main Application Controller ---
class SystemWatchdog(threading.Thread):
    """Monitors the local web server and reboots if it becomes unresponsive."""
    def __init__(self, port=8080, startup_delay=120, check_interval=60):
        super().__init__(daemon=True)
        self.port = port
        self.startup_delay = startup_delay
        self.check_interval = check_interval
        self.running = True

    def run(self):
        print(f"[*] Watchdog active. Waiting {self.startup_delay}s for system startup...")
        time.sleep(self.startup_delay)
        
        while self.running:
            if not self._check_health():
                print("[!] Health check failed. Retrying in 10s...")
                time.sleep(10)
                if not self._check_health():
                    print("[!!!] CRITICAL FAILURE. REBOOTING SYSTEM.")
                    subprocess.run(['sudo', 'reboot'])
                    self.running = False
            time.sleep(self.check_interval)

    def _check_health(self):
        try:
            r = requests.get(f"https://europe.on-air.io/projectaria/RPiManager/{self.port}/status", timeout=5)
            # Must be 200 OK and return valid JSON
            return r.status_code == 200 and isinstance(r.json(), dict)
        except Exception:
            return False

class PiManagerApp:
    def __init__(self):
        self.api = get_api_integration()
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
                # Initialize the shared layout (Sidebar + Header)
                DashboardLayout()
                # Mount the File Browser into the remaining page space
                FileBrowser()

        @ui.page('/cameras')
        def cameras_page():
            if not app.storage.user.get('authenticated', False):
                ui.navigate.to('/')
            else:
                # Initialize the shared layout
                DashboardLayout()
                # Mount the Camera View
                CameraView()
        
        # In PiManagerApp.setup_routes
        @app.get('/camera_proxy/{idx}')
        async def camera_proxy(idx: int):
            # Fetch data without instantiating UI classes
            all_cameras = CameraView.get_all_cameras() 
            
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
                        time.sleep(0.1) # Wait for camera to recover
                    time.sleep(0.03) # Limit to ~30 FPS to save CPU

            return StreamingResponse(generate_frames(), 
                                    media_type='multipart/x-mixed-replace; boundary=frame')

        @ui.page('/mqtt')
        def mqtt_page():
            if not app.storage.user.get('authenticated', False):
                ui.navigate.to('/')
            else:
                DashboardLayout()
                MQTTInspector()
        
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

        @ui.page('/gpio')
        def gpio_page():
            if not app.storage.user.get('authenticated', False):
                ui.navigate.to('/')
            else:
                DashboardLayout()
                GPIOPage()
        
        @ui.page('/api-control')
        def api_control_page():
            if not app.storage.user.get('authenticated', False):
                ui.navigate.to('/')
            else:
                DashboardLayout()
                APIControlPanel(self.api)
        
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
            # watchdog = SystemWatchdog()
            # watchdog.start()

            global gpio_manager
            gpio_manager = GPIOManager()

            ui.run(
                port=8080, 
                title="Pi Manager", 
                dark=True, 
                reload=False, 
                storage_secret='midnight_blue_heartbeat_secret'
            )

# --- Entry Point ---

if __name__ in {"__main__", "__mp_main__"}:
    pi_app = PiManagerApp()
    pi_app.start()