import os
import sys
import time
import shutil
import socket
import threading
import tempfile
import subprocess
import re
import ftplib
import getpass
import tempfile
import ipaddress
import argparse
import hashlib
import ast
import cv2
import base64
import io
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Callable, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# [Data]
# CLI.Name = "imang"
# CLI.Password = "imang"
# CLI.Host = "192.168.37.205"
# [Data End]

# Global variable for lazy loading NiceGUI
ui = None

# --- Core Business Logic ---

class SystemMonitor:
    """Handles hardware metrics and power interactions."""
    def __init__(self):
        import psutil
        psutil.cpu_percent(interval=None)

    def get_metrics(self):
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        # Temperature Logic
        temp = 42.0 
        try:
            res = subprocess.check_output(['vcgencmd', 'measure_temp'], stderr=subprocess.DEVNULL).decode('utf-8')
            temp = float(re.search(r'\d+\.\d+', res).group())
        except: pass
            
        # Voltage & Health Logic (Targeting the 5V rail status)
        volts = 5.0
        low_power = False
        try:
            # Check throttled state for Under-Voltage (Bit 0)
            res_t = subprocess.check_output(['vcgencmd', 'get_throttled'], stderr=subprocess.DEVNULL).decode('utf-8')
            # Bit 0: Under-voltage detected (currently active)
            throttle_hex = int(res_t.split('=')[1], 16)
            low_power = bool(throttle_hex & 0x1) 
            
            # If low power is detected, show the 'sagged' voltage (e.g., 4.6V)
            if low_power:
                volts = 4.6
            else:
                # Optionally get core voltage just to verify system activity
                res_v = subprocess.check_output(['vcgencmd', 'measure_volts', 'core'], stderr=subprocess.DEVNULL).decode('utf-8')
                v_core = float(re.search(r'\d+\.\d+', res_v).group())
                if v_core > 0.5: volts = 5.05 # Nominal 5V display
        except: 
            volts = 5.0 
                
        return {"cpu": cpu, "ram": ram, "disk": disk, "temp": temp, "volts": volts, "low_power": low_power}

    def get_ip_address(self):
        """Returns the first non-localhost IP address used for network access."""
        try:
            # This is the most reliable way to find the 'reachable' IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Use a dummy address to find the outbound route
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if not ip.startswith('127.'):
                return ip
        except Exception:
            pass

        try:
            # Fallback: Find the first non-localhost IPv4 address across all interfaces
            import psutil
            addrs = psutil.net_if_addrs()
            # Sort keys to prioritize eth0 over wlan0 if both are present
            for interface in sorted(addrs.keys()):
                for addr in addrs[interface]:
                    if addr.family == socket.AF_INET and not addr.address.startswith('127.'):
                        return addr.address
        except:
            pass
            
        return "127.0.0.1"

class FTPService:
    """Handles FTP connections and file transfers."""
    def __init__(self):
        self.connected = False
        self.host = ""
        self.user = ""
        self.port = 21
        self.ftp: Optional[ftplib.FTP] = None
    
    def connect(self, host: str, user: str, password: str, port: str = "21") -> Tuple[bool, str]:
        try:
            clean_host = host.replace('ftp://', '').replace('ftps://', '').split('/')[0].split(':')[0]
            target_port = int(port)
            self.ftp = ftplib.FTP()
            self.ftp.connect(clean_host, target_port, timeout=15)
            self.ftp.login(user, password)
            self.ftp.set_pasv(True) 
            self.host, self.user, self.port = clean_host, user, target_port
            self.connected = True
            return True, "Success"
        except Exception as e:
            self.connected = False
            return False, str(e)

    def disconnect(self):
        if self.ftp:
            try: self.ftp.quit()
            except: 
                try: self.ftp.close()
                except: pass
        self.connected = False
        self.ftp = None

    def list_files(self, path):
        if not self.connected or not self.ftp: return []
        files = []
        try:
            self.ftp.cwd(path)
            items = []
            self.ftp.dir(items.append)
            for line in items:
                parts = line.split(maxsplit=8)
                if len(parts) < 9: continue
                perms, _, _, _, size, _, _, _, name = parts
                if name in ['.', '..']: continue
                files.append({"name": name, "is_dir": perms.startswith('d'), "size": int(size)})
        except Exception as e:
            print(f"FTP List Error: {e}")
        return sorted(files, key=lambda x: (not x['is_dir'], x['name']))

    def get_file_size(self, remote_path: str) -> int:
        try:
            size = self.ftp.size(remote_path)
            if size is not None: return size
        except: pass
        
        try:
            parent = os.path.dirname(remote_path) or "/"
            filename = os.path.basename(remote_path)
            lines = []
            self.ftp.dir(parent, lines.append)
            for line in lines:
                if filename in line:
                    parts = line.split()
                    if len(parts) >= 8 and parts[-1] == filename:
                        return int(parts[4])
        except: pass
        return 0

    def download_file(self, remote_path: str, local_path: str, progress_callback: Optional[Callable[[float], None]] = None):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        total_size = self.get_file_size(remote_path)
        downloaded = 0
        
        # Inside FTPService.download_file
        def handle_chunk(chunk):
            nonlocal downloaded
            lf.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total_size > 0:
                progress_callback(min(downloaded / total_size, 1.0))
                # Optional: time.sleep(0) allows thread switching

        with open(local_path, 'wb') as lf:
            path = "/" + remote_path.lstrip('/')
            self.ftp.retrbinary(f"RETR {path}", handle_chunk, blocksize=32768)
            
        if progress_callback:
            progress_callback(1.0)

    def upload_file(self, local_path: str, remote_path: str, progress_callback: Optional[Callable[[float], None]] = None):
        """Uploads a local file to the FTP server."""
        total_size = os.path.getsize(local_path)
        uploaded = 0

        def handle_chunk(chunk):
            nonlocal uploaded
            uploaded += len(chunk)
            if progress_callback and total_size > 0:
                progress_callback(min(uploaded / total_size, 1.0))

        with open(local_path, 'rb') as f:
            path = "/" + remote_path.lstrip('/')
            self.ftp.storbinary(f"STOR {path}", f, blocksize=32768, callback=handle_chunk)

    def ensure_dir(self, remote_dir: str):
        """Ensures a remote directory exists (mkdir -p)."""
        parts = remote_dir.strip('/').split('/')
        current = ""
        for part in parts:
            current += "/" + part
            try:
                self.ftp.mkd(current)
            except:
                pass # Already exists or no permission

class DeploymentManager:
    def __init__(self, ftp_service: FTPService):
        self.ftp = ftp_service
        self.temp_dir = None
        self.process = None

    def extract_argparse_args(self, script_content: str) -> Dict[str, List[Dict]]:
        """Statically parses argparse with support for positional, optional, and nargs."""
        results = {'positional': [], 'optional': []}
        try:
            tree = ast.parse(script_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument':
                    arg_info = {'name': '', 'flags': [], 'default': None, 'help': '', 'type': 'str', 'nargs': 1}
                    
                    # Identify Positional vs Optional
                    for arg in node.args:
                        if isinstance(arg, ast.Constant):
                            val = str(arg.value)
                            if val.startswith('-'):
                                arg_info['flags'].append(val)
                            else:
                                arg_info['name'] = val

                    # Parse Keywords (help, default, nargs, etc.)
                    for kw in node.keywords:
                        if kw.arg == 'default':
                            arg_info['default'] = ast.literal_eval(kw.value) if isinstance(kw.value, (ast.Constant, ast.List)) else None
                        elif kw.arg == 'help':
                            arg_info['help'] = kw.value.value if isinstance(kw.value, ast.Constant) else ""
                        elif kw.arg == 'nargs':
                            arg_info['nargs'] = kw.value.value if isinstance(kw.value, ast.Constant) else 1
                        elif kw.arg == 'action' and isinstance(kw.value, ast.Constant):
                            if kw.value.value in ['store_true', 'store_false']:
                                arg_info['type'] = 'bool'

                    # Final Classification
                    if not arg_info['flags']:
                        results['positional'].append(arg_info)
                    else:
                        arg_info['name'] = arg_info['flags'][-1] # Primary flag
                        results['optional'].append(arg_info)
        except Exception as e:
            print(f"AST Parse Error: {e}")
        return results
    
    def prepare_deployment(self, remote_script_path: str, on_file_list: Callable, on_file_progress: Callable):
        self.temp_dir = tempfile.mkdtemp()
        remote_script_path = "/" + remote_script_path.lstrip('/')
        local_script_path = os.path.join(self.temp_dir, remote_script_path.lstrip('/'))
        
        # 1. Download main script to parse it
        self.ftp.download_file(remote_script_path, local_script_path, 
                            lambda p: on_file_progress(remote_script_path, p))
        
        with open(local_script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 2. Extract Argparse arguments
        discovered_args = self.extract_argparse_args(content)
        
        # 3. Parse [Include] dependencies (existing logic)
        deps = []
        include_block = re.search(r'# \[Include\](.*?)# \[Include End\]', content, re.DOTALL)
        if include_block:
            deps = re.findall(r'#\s*-\s*(\S+)', include_block.group(1))

        # 4. Sync dependencies
        for dep in deps:
            dep_path = "/" + dep.lstrip('/')
            local_dep_path = os.path.join(self.temp_dir, dep_path.lstrip('/'))
            self.ftp.download_file(dep_path, local_dep_path, 
                                   lambda p, d=dep_path: on_file_progress(d, p))
        
        # RETURN discovered_args so the UI can render them
        return self.temp_dir, local_script_path, discovered_args
    
    def run_script(self, local_script_path: str, on_output: Callable, args: List[str] = None):
        cwd = os.path.dirname(local_script_path)
        def run():
            try:
                # Build the full command list
                full_cmd = [sys.executable, "-u", local_script_path]
                if args:
                    full_cmd.extend(args)
                
                # Start the process
                self.process = subprocess.Popen(
                    full_cmd, 
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                    text=True, cwd=cwd, bufsize=1
                )
                
                # Stream the output
                for line in self.process.stdout: 
                    on_output(line.strip())
                    
                rc = self.process.wait()
                on_output(f"\n[Finished] Process exited with code {rc}")
            except Exception as e: 
                on_output(f"[Error] Failed to launch: {e}")
                
        threading.Thread(target=run, daemon=True).start()
        
    def cleanup(self):
        if self.process and self.process.poll() is None: self.process.terminate()
        if self.temp_dir and os.path.exists(self.temp_dir): shutil.rmtree(self.temp_dir, ignore_errors=True)

# --- UI Components ---

class HeaderComponent:
    def __init__(self, app_state):
        self.app_state = app_state
        self.dialog_settings = None
        self.dialog_cameras = None # New dialog
        self.dialog_settings = None
        self.history = {k: deque([0.0]*30, maxlen=30) for k in ['cpu', 'ram', 'disk', 'temp', 'volts']}
        self.current_width = 1200  
        self.selected_metric = 'cpu' 
        self.metrics_container = None
        self.charts = {} 

    def get_mode(self, width):
        """Helper to determine the layout mode based on breakpoints."""
        if width > 1200: return 'desktop'
        if width > 650: return 'tablet'
        return 'mobile'

    def render(self):
        ui.add_head_html('''
            <script>
                const notifyResize = () => {
                    if (window.emitEvent) {
                        emitEvent("update_width", {width: window.innerWidth});
                    }
                };
                window.addEventListener("resize", notifyResize);
                setTimeout(notifyResize, 500);
            </script>
        ''')
        
        ui.on('update_width', lambda e: self.handle_resize(e.args['width']))

        # Main Header Structure
        with ui.header().classes('bg-gray-950 text-white shadow-xl items-center h-24 p-0 px-6 no-wrap'):
            # 1. Identity Section (Always Visible)
            with ui.row().classes('items-center no-wrap grow-0 mr-4'):
                with ui.column().classes('cursor-pointer gap-0 min-w-[150px]').on('click', self.open_settings):
                    ui.label().bind_text_from(self.app_state, 'user').classes('text-xl font-black text-cyan-400 leading-none tracking-tight')
                    ui.label(f"Host: {self.app_state['host_ip']}").classes('text-[10px] text-gray-600 uppercase font-bold mt-1')
                    ui.label('Client: ...').bind_text_from(ui.context.client, 'ip', backward=lambda x: f"Client: {x}").classes('text-[9px] text-gray-500 font-bold uppercase')
                    
                    with ui.row().classes('items-center gap-1 mt-1'):
                        ui.element('div').classes('w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_#22c55e]') \
                            .bind_visibility_from(self.app_state, 'ftp_connected')
                        ui.label('FTP Online').classes('text-[9px] font-black uppercase text-green-500') \
                            .bind_visibility_from(self.app_state, 'ftp_connected')
                        
                        ui.element('div').classes('w-1.5 h-1.5 rounded-full bg-red-500 shadow-[0_0_8px_#ef4444]') \
                            .bind_visibility_from(self.app_state, 'ftp_connected', backward=lambda x: not x)
                        ui.label('FTP Offline').classes('text-[9px] font-black uppercase text-red-500') \
                            .bind_visibility_from(self.app_state, 'ftp_connected', backward=lambda x: not x)

            self.metrics_container = ui.row().classes('flex-1 justify-end items-center gap-4 overflow-hidden h-full no-wrap')
            self.refresh_metrics_layout()

        self.build_settings_popup()
        ui.timer(2.0, self.update_history)

    def handle_resize(self, width):
        # Breakpoint optimization: Only redraw if we switch modes
        if self.get_mode(self.current_width) != self.get_mode(width):
            self.current_width = width
            self.refresh_metrics_layout()
        else:
            self.current_width = width

    def update_history(self):
        # Update internal buffers
        for k in self.history:
            val = self.app_state['metrics'].get(k, 0.0)
            self.history[k].append(float(val))
        
        # Update Charts
        active_keys = list(self.charts.keys())
        for key in active_keys:
            chart = self.charts.get(key)
            if chart:
                chart.options['series'][0]['data'] = list(self.history[key])
                chart.update()

        # Dynamic Danger Theme for Voltage
        if hasattr(self, 'volts_card'):
            is_low = self.app_state['metrics'].get('low_power', False)
            if is_low:
                # Apply Red Pulsing Theme
                self.volts_card.classes(replace='bg-red-950/80 border-red-500 animate-pulse w-32 h-16 rounded-md border shadow-inner transition-all duration-500')
                self.volts_label.classes(replace='text-[9px] font-mono font-bold text-red-400')
            else:
                # Restore Normal Cyan Theme
                self.volts_card.classes(replace='bg-black/40 border-gray-800/50 w-32 h-16 rounded-md border shadow-inner transition-all duration-500')
                self.volts_label.classes(replace='text-[9px] font-mono font-bold text-cyan-300')
                
    def refresh_metrics_layout(self):
        if not self.metrics_container: return
        self.metrics_container.clear()
        self.charts.clear() 
        
        m_list = [
            ('cpu', '#06b6d4', 'memory', '%'), 
            ('ram', '#a855f7', 'developer_board', '%'), 
            ('disk', '#f97316', 'storage', '%'), 
            ('temp', '#ef4444', 'thermostat', '°C'),
            ('volts', '#eab308', 'bolt', 'V')
        ]
        
        mode = self.get_mode(self.current_width)
        
        with self.metrics_container:
            if mode == 'desktop':
                for k, c, i, u in m_list:
                    self.charts[k] = self.render_metric_box(k, c, u)
            
            elif mode == 'tablet':
                with ui.row().classes('gap-3 items-center no-wrap'):
                    for k, c, i, u in m_list:
                        if k == self.selected_metric:
                            self.charts[k] = self.render_metric_box(k, c, u, is_inline=True)
                        else:
                            self.render_mini_item(k, c, i, u)

            else: # Mobile
                active_cfg = next(m for m in m_list if m[0] == self.selected_metric)
                with ui.row().classes('w-full justify-end px-2').on('click', self.cycle_metric):
                    self.charts[active_cfg[0]] = self.render_metric_box(active_cfg[0], active_cfg[1], active_cfg[3], is_narrow=True)

    def render_metric_box(self, key, color, unit, is_inline=False, is_narrow=False):
        width = 'w-44' if is_inline else 'w-full max-w-[280px]' if is_narrow else 'w-32'
        is_volts = (key == 'volts')
        
        # Define base classes
        base_classes = f'{width} h-16 gap-0 items-center justify-center rounded-md border shadow-inner transition-all duration-500 bg-black/40 border-gray-800/50'
        
        # Create the column
        card = ui.column().classes(base_classes)
        
        # Store the card in a dictionary if it's the voltage card so we can find it later
        if is_volts:
            self.volts_card = card

        with card:
            with ui.row().classes('w-full justify-between items-baseline px-2 no-wrap'):
                ui.label(key.upper()).classes('text-[8px] font-black text-gray-500')
                precision = ".1f" if key in ['volts', 'temp'] else ".0f"
                
                # Text label
                lbl = ui.label().bind_text_from(self.app_state['metrics'], key, backward=lambda x: f"{x:{precision}}{unit}")
                lbl.classes('text-[9px] font-mono font-bold text-cyan-300')
                if is_volts:
                    self.volts_label = lbl
            
            chart = ui.echart(self.get_chart_options(key, color)).classes('w-full h-10')
            return chart
    
    def render_mini_item(self, key, color_hex, icon, unit):
        with ui.column().classes('items-center gap-0 w-10 cursor-pointer hover:bg-white/5 rounded p-1 transition-all').on('click', lambda: self.set_active(key)):
            ui.icon(icon).style(f'color: {color_hex}').classes('text-base mb-0.5')
            with ui.row().classes('gap-0 items-baseline no-wrap'):
                precision = ".1f" if key in ['volts', 'temp'] else ".0f"
                ui.label().bind_text_from(self.app_state['metrics'], key, backward=lambda x: f"{x:{precision}}").classes('text-[8px] font-black')
            
            with ui.element('div').classes('w-full h-1 bg-gray-800 rounded-full mt-0.5 overflow-hidden'):
                ui.linear_progress(show_value=False).bind_value_from(self.app_state['metrics'], key, 
                    backward=lambda x: (x/100 if key not in ['volts', 'temp'] else (x/85 if key=='temp' else (x/1.4 if x < 2.0 else x/5.5)))) \
                    .props(f'color={color_hex} track-color=transparent').classes('h-full')

    def get_chart_options(self, key, color):
        """Calculates chart scale based on the metric type."""
        data = list(self.history[key])
        
        # Updated Y-Axis scaling
        if key == 'volts':
            y_min, y_max = 0, 6.0    # Changed from 1.5 to 6.0 for 5V rail visibility
        elif key == 'temp':
            y_min, y_max = 30, 85    # Thermal range
        else:
            y_min, y_max = 0, 100    # Percentage range for CPU/RAM/Disk

        return {
            'xAxis': {'type': 'category', 'show': False},
            'yAxis': {'type': 'value', 'show': False, 'min': y_min, 'max': y_max},
            'grid': {'left': 5, 'right': 5, 'top': 5, 'bottom': 5},
            'animationDurationUpdate': 1000,
            'series': [{
                'data': data,
                'type': 'line',
                'smooth': True,
                'symbol': 'none',
                'areaStyle': {'color': color, 'opacity': 0.1},
                'lineStyle': {'color': color, 'width': 2},
            }]
        }

    def cycle_metric(self):
        keys = ['cpu', 'ram', 'disk', 'temp', 'volts']
        idx = (keys.index(self.selected_metric) + 1) % len(keys)
        self.set_active(keys[idx])

    def set_active(self, key):
        self.selected_metric = key
        self.refresh_metrics_layout()

    def open_settings(self): self.dialog_settings.open()

    def build_settings_popup(self):
        with ui.dialog() as self.dialog_settings, ui.card().classes('bg-gray-900 border border-gray-800 text-white min-w-[320px]'):
            ui.label('SYSTEM CONTROL').classes('text-xs font-black text-gray-500 mb-2 tracking-widest')
            with ui.column().classes('w-full gap-2'):
                ui.button('REBOOT', on_click=lambda: os.system('sudo reboot'), color='red-500').props('unelevated').classes('w-full font-bold')
                ui.button('SHUTDOWN', on_click=lambda: os.system('sudo shutdown now'), color='red-900').props('unelevated').classes('w-full font-bold')
                ui.separator().classes('bg-gray-800 my-2')
                ui.button('RESTART APP', on_click=lambda: ui.run_javascript('location.reload()'), color='gray-700').props('unelevated').classes('w-full font-bold')
                btn_update = ui.button('UPDATE APP', color='cyan-600').classes('w-full font-bold')
                btn_update.bind_enabled_from(self.app_state, 'ftp_connected')
                btn_update.on('click', lambda: ui.notify("Update sequence started..."))

    def build_camera_popup(self):
        """Creates the UI for managing saved RTSP cameras."""
        with ui.dialog() as self.dialog_cameras, ui.card().classes('bg-gray-900 border border-gray-800 text-white min-w-[400px]'):
            ui.label('CAMERA REGISTRY').classes('text-xs font-black text-cyan-500 mb-4 tracking-[0.2em]')
            
            # List existing cameras
            camera_list_container = ui.column().classes('w-full gap-2 mb-4')
            
            def refresh_cam_list():
                camera_list_container.clear()
                with camera_list_container:
                    for idx, cam in enumerate(self.app_state['cameras']):
                        with ui.row().classes('w-full items-center bg-black/40 p-2 rounded border border-gray-800'):
                            ui.icon('sensors').classes('text-cyan-600')
                            with ui.column().classes('flex-1 gap-0'):
                                ui.label(cam['name']).classes('text-[10px] font-bold')
                                ui.label(cam['url']).classes('text-[8px] text-gray-500 truncate w-48')
                            
                            ui.button(icon='play_circle', on_click=lambda c=cam: CameraStreamPopup(c).open()) \
                                .props('flat dense round color=green-500')
                            
                            def delete_cam(i=idx):
                                self.app_state['cameras'].pop(i)
                                refresh_cam_list()
                            ui.button(icon='delete', on_click=delete_cam).props('flat dense round color=red-900')

            refresh_cam_list()

            ui.separator().classes('bg-gray-800')
            
            # Add New Camera Section
            ui.label('ADD NEW STREAM').classes('text-[9px] font-bold text-gray-500 mt-2')
            name_input = ui.input('Camera Name').props('dark dense filled')
            url_input = ui.input('RTSP/Stream URL').props('dark dense filled')
            
            def add_camera():
                if name_input.value and url_input.value:
                    self.app_state['cameras'].append({'name': name_input.value, 'url': url_input.value})
                    name_input.value = ''; url_input.value = ''
                    refresh_cam_list()
                    ui.notify('Camera Saved', type='positive')

            ui.button('REGISTER CAMERA', on_click=add_camera).classes('w-full mt-2')
        
    def update_volts_chart_color(self, chart):
        # Change the line color dynamically if voltage drops
        is_low = self.app_state['metrics'].get('low_power', False)
        new_color = '#ef4444' if is_low else '#eab308'
        chart.options['series'][0]['lineStyle']['color'] = new_color
        chart.options['series'][0]['areaStyle']['color'] = new_color
        chart.update()

class FileBrowserComponent:
    def __init__(self, title, is_remote, app_state, ftp_service, deploy_mgr):
        self.title, self.is_remote, self.app_state, self.ftp, self.deploy_mgr = title, is_remote, app_state, ftp_service, deploy_mgr
        self.path = "/" if is_remote else os.getcwd()
        self.content_container = None
        self.path_label = None
        self.list_area = None 
        self.target_browser: Optional['FileBrowserComponent'] = None

    def render(self):
        with ui.card().classes('w-full h-[600px] flex flex-col bg-gray-950 border border-gray-800 p-0 overflow-hidden shadow-2xl'):
            with ui.row().classes('w-full bg-gray-900/50 p-2 items-center gap-2 border-b border-gray-800'):
                ui.icon('folder' if not self.is_remote else 'cloud').classes('text-cyan-400 ml-1')
                ui.label(self.title).classes('font-black text-[10px] tracking-widest uppercase text-gray-500')
                self.path_label = ui.label(self.path).classes('font-mono text-[9px] bg-black/50 px-2 py-1 rounded text-cyan-200 truncate flex-1 mx-2')
                
                with ui.row().classes('gap-1 items-center mr-1'):
                    ui.button(icon='refresh', on_click=self.refresh_list).props('flat dense round color=cyan-400 size=sm')
                    ui.button(icon='home', on_click=lambda: self.navigate('/' if self.is_remote else os.path.expanduser('~'))).props('flat dense round color=gray-400 size=sm')
                    ui.button(icon='arrow_upward', on_click=lambda: self.navigate('..')).props('flat dense round color=gray-400 size=sm')
                    if not self.is_remote:
                        ui.button(icon='terminal', on_click=self.toggle_console).props('flat dense round color=cyan-400 size=sm')
                    if self.is_remote:
                        ui.button(icon='logout', on_click=self.handle_disconnect).props('flat round color=red-400 dense size=sm').bind_visibility_from(self.app_state, 'ftp_connected')

            self.content_container = ui.column().classes('flex-1 w-full overflow-hidden gap-0')
            self.refresh_ui()

    def refresh_ui(self):
        if not self.content_container: return
        self.content_container.clear()
        with self.content_container:
            if self.is_remote and not self.app_state['ftp_connected']:
                self.render_ftp_login()
            else:
                self.render_browser_content()

    def render_ftp_login(self):
        with ui.column().classes('w-full h-full items-center justify-center p-4 bg-gray-950'):
            with ui.column().classes('w-full max-w-[320px] items-stretch gap-3 p-6 rounded-lg bg-gray-900/30 border border-gray-800/50'):
                with ui.column().classes('items-center w-full mb-2'):
                    ui.icon('lan').classes('text-4xl text-cyan-900')
                    ui.label("CLIENT FTP CONFIGURATION").classes('text-[9px] font-black text-gray-500 tracking-widest mt-2')
                
                with ui.row().classes('w-full gap-2'):
                    # HOST INPUT with internal autofill for Client IP
                    with ui.input('FTP HOST').classes('flex-1').props('dark filled dense color=cyan') as host_input:
                        with host_input.add_slot('append'):
                            ui.button(icon='auto_fix_high', 
                                      on_click=lambda: host_input.set_value(ui.context.client.ip)) \
                                .props('flat dense round size=sm color=cyan-400') \
                                .tooltip('Autofill My IP')
                    
                    # FIXED PORT: Initialized with value='21' instead of props
                    port_input = ui.input('PORT', value='21').classes('w-20').props('dark filled dense color=cyan')
                
                user_input = ui.input('USERNAME').classes('w-full').props('dark filled dense color=cyan')
                pwd_input = ui.input('PASSWORD', password=True).classes('w-full').props('dark filled dense color=cyan')
                
                btn = ui.button('ESTABLISH CONNECTION').classes('w-full bg-cyan-900 font-bold mt-2 text-xs py-2 shadow-lg transition-all active:scale-95')
                with btn:
                    spinner = ui.spinner(size='sm', color='white').classes('hidden ml-2')

                client = ui.context.client
                def handle_connect():
                    # Basic validation to prevent empty strings causing int() errors
                    if not port_input.value:
                        ui.notify("Port is required", type='warning')
                        return
                        
                    btn.disable(); spinner.classes(remove='hidden'); btn.text = "CONNECTING..."
                    def thread_task():
                        # This now correctly passes '21' if untouched
                        success, message = self.ftp.connect(host_input.value, user_input.value, pwd_input.value, port_input.value)
                        with client:
                            if success: 
                                self.app_state['ftp_connected'] = True
                                self.refresh_ui()
                            else: 
                                ui.notify(f'Connection failed: {message}', type='negative')
                                btn.enable(); spinner.classes(add='hidden'); btn.text = "ESTABLISH CONNECTION"
                    threading.Thread(target=thread_task, daemon=True).start()
                btn.on('click', handle_connect)
        
    def render_browser_content(self):
        self.list_area = ui.scroll_area().classes('flex-1 w-full p-2 bg-gray-950/20')
        self.refresh_list()
        if not self.is_remote:
            self.console_ui = ui.column().classes('w-full h-0 bg-black transition-all overflow-hidden border-t border-gray-800')
            with self.console_ui:
                self.con_log = ui.log().classes('w-full h-32 font-mono text-[9px] text-green-500/80 p-2 bg-black m-0 overflow-y-auto border-none')
                self.cmd_input = ui.input(placeholder='CMD >').classes('w-full px-2 py-1 bg-black border-t border-gray-900/50').props('dark dense borderless standout=false font-mono').style('font-size: 9px; color: #10b981; font-family: monospace;')
                self.cmd_input.on('keydown.enter', self.run_cmd)

    def refresh_list(self):
        if not self.list_area: return
        self.list_area.clear()
        if self.path_label: self.path_label.text = self.path
        files = []
        try:
            if self.is_remote: files = self.ftp.list_files(self.path)
            else:
                for e in os.scandir(self.path):
                    files.append({"name": e.name, "is_dir": e.is_dir(), "size": e.stat().st_size})
                files.sort(key=lambda x: (not x['is_dir'], x['name']))
        except Exception as e:
            ui.notify(f"List Error: {e}")

        with self.list_area:
            if not files:
                ui.label("Empty folder or access denied").classes('text-gray-600 italic text-[10px] text-center w-full mt-4')
            for f in files:
                with ui.row().classes('w-full items-center p-1 px-2 hover:bg-white/5 rounded group transition-colors'):
                    ic = 'folder' if f['is_dir'] else 'insert_drive_file'
                    clr = 'text-yellow-600/80' if f['is_dir'] else 'text-blue-500/80'
                    ui.icon(ic).classes(f'{clr} mr-2 text-sm')
                    ui.label(f['name']).classes('flex-1 text-xs text-gray-300 cursor-pointer truncate').on('click', lambda n=f['name'], d=f['is_dir']: self.on_item_click(n, d))
                    
                    with ui.row().classes('opacity-0 group-hover:opacity-100 transition-opacity gap-1'):
                        # Action: Transfer (Copy to the other side)
                        ui.button(icon='content_copy', on_click=lambda n=f['name'], d=f['is_dir']: self.transfer_item(n, d)) \
                            .props('flat dense round size=xs color=cyan-500').classes('hover:scale-110')
                        
                        ui.button(icon='delete', on_click=lambda n=f['name']: self.delete_item(n)).props('flat dense round size=xs color=red-600')
                        if self.is_remote and not f['is_dir'] and f['name'].endswith(('.py', '.sh')):
                            ui.button('DEPLOY', on_click=lambda n=f['name']: self.deploy(n)).props('unelevated dense size=xs color=green-800').classes('text-[8px] font-black px-1')

    def transfer_item(self, name, is_dir):
        """Copies item from current browser to the other browser's path."""
        if not self.target_browser:
            ui.notify("Sync error: Target browser not detected", type='negative')
            return
        
        # Guard for remote browser connectivity
        if self.is_remote and not self.app_state['ftp_connected']:
            ui.notify("Remote browser not connected", type='warning')
            return
        if not self.is_remote and not self.app_state['ftp_connected']:
            ui.notify("Cannot transfer: Remote side is offline", type='warning')
            return

        source_full = os.path.join(self.path, name) if not self.is_remote else (self.path.rstrip('/') + '/' + name).replace('//', '/')
        target_path = self.target_browser.path
        dest_full = os.path.join(target_path, name) if self.is_remote else (target_path.rstrip('/') + '/' + name).replace('//', '/')

        ui.notify(f"Transferring {name}...", icon='sync', color='cyan-900')

        def run_transfer():
            try:
                if self.is_remote:
                    # Remote -> Local (Download)
                    if is_dir:
                        self._download_recursive(source_full, dest_full)
                    else:
                        self.ftp.download_file(source_full, dest_full)
                else:
                    # Local -> Remote (Upload)
                    if is_dir:
                        self._upload_recursive(source_full, dest_full)
                    else:
                        self.ftp.upload_file(source_full, dest_full)
                
                ui.notify(f"Transfer complete: {name}", type='positive')
                self.target_browser.refresh_list()
            except Exception as e:
                ui.notify(f"Transfer Error: {str(e)}", type='negative')

        threading.Thread(target=run_transfer, daemon=True).start()

    def _download_recursive(self, remote_path, local_path):
        """Helper to download directories."""
        os.makedirs(local_path, exist_ok=True)
        items = self.ftp.list_files(remote_path)
        for item in items:
            r_item = (remote_path.rstrip('/') + '/' + item['name']).replace('//', '/')
            l_item = os.path.join(local_path, item['name'])
            if item['is_dir']:
                self._download_recursive(r_item, l_item)
            else:
                self.ftp.download_file(r_item, l_item)

    def _upload_recursive(self, local_path, remote_path):
        """Helper to upload directories."""
        self.ftp.ensure_dir(remote_path)
        for item in os.scandir(local_path):
            l_item = item.path
            r_item = (remote_path.rstrip('/') + '/' + item.name).replace('//', '/')
            if item.is_dir():
                self._upload_recursive(l_item, r_item)
            else:
                self.ftp.upload_file(l_item, r_item)

    def on_item_click(self, name, is_dir):
        if is_dir:
            if self.is_remote: self.path = (self.path.rstrip('/') + '/' + name).replace('//', '/')
            else: self.path = os.path.join(self.path, name)
            self.refresh_list()
        else:
            if not self.is_remote: ViewerPopup(os.path.join(self.path, name), name.split('.')[-1]).open()

    def navigate(self, target):
        if target == '..':
            if self.is_remote: 
                parts = self.path.rstrip('/').split('/')
                self.path = "/".join(parts[:-1]) or "/"
            else: self.path = str(Path(self.path).parent)
        else: self.path = target
        self.refresh_list()

    def delete_item(self, name):
        try:
            if not self.is_remote:
                target = os.path.join(self.path, name)
                if os.path.isdir(target): shutil.rmtree(target)
                else: os.remove(target)
            else:
                try: self.ftp.ftp.delete(name)
                except: self.ftp.ftp.rmd(name)
            self.refresh_list()
        except Exception as e: ui.notify(f"Error: {e}")

    def toggle_console(self): 
        if not hasattr(self, 'console_ui'): return
        self.console_ui.classes(remove='h-0', add='h-48') if 'h-0' in self.console_ui.classes else self.console_ui.classes(remove='h-48', add='h-0')

    def run_cmd(self, e):
        cmd = self.cmd_input.value
        self.cmd_input.value = ''
        self.con_log.push(f"root@pi:{os.path.basename(self.path)}$ {cmd}")
        def worker():
            try:
                out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, cwd=self.path).decode()
                for l in out.splitlines(): self.con_log.push(l)
            except Exception as ex: self.con_log.push(str(ex))
        threading.Thread(target=worker, daemon=True).start()

    def handle_disconnect(self):
        self.ftp.disconnect(); self.app_state['ftp_connected'] = False
        self.list_area = None; self.refresh_ui()

    def deploy(self, name): 
        ConsolePopup(f"{self.path}/{name}", self.deploy_mgr, self.app_state).open()

class CameraRegistryComponent:
    def __init__(self, app_state):
        self.app_state = app_state
        self.active_url = None
        # UI Element placeholders
        self.preview_placeholder = None
        self.stream_info = None
        self.active_name = None
        self.active_url_label = None

    def render(self):
        with ui.card().classes('w-full bg-gray-950 border border-gray-800 p-0 overflow-hidden shadow-2xl mb-6'):
            with ui.row().classes('w-full bg-gray-900/50 p-3 items-center border-b border-gray-800'):
                ui.icon('videocam').classes('text-cyan-400 ml-1')
                ui.label('CAMERA REGISTRY').classes('font-black text-[10px] tracking-widest uppercase text-gray-500')
            
            with ui.grid().classes('w-full p-4 grid-cols-1 md:grid-cols-2 gap-6'):
                # Left: Management
                with ui.column().classes('gap-3'):
                    ui.label('REGISTER NEW STREAM').classes('text-[9px] font-bold text-gray-500 mb-1')
                    name_in = ui.input('Camera Name').props('dark filled dense color=cyan').classes('w-full')
                    url_in = ui.input('RTSP / Stream URL').props('dark filled dense color=cyan').classes('w-full')
                    
                    with ui.row().classes('w-full gap-2 mt-2'):
                        ui.button('ADD CAMERA', on_click=lambda: self.add_camera(name_in, url_in)) \
                            .classes('flex-1 bg-cyan-900 font-bold text-xs')

                    self.list_area = ui.column().classes('w-full mt-4 gap-2')
                    self.render_list()

                # Right: Preview Area (Placeholder State)
                with ui.column().classes('w-full aspect-video bg-black rounded-lg border border-gray-800 items-center justify-center relative overflow-hidden'):
                    self.preview_placeholder = ui.column().classes('items-center gap-2')
                    with self.preview_placeholder:
                        ui.icon('construction').classes('text-6xl text-gray-800')
                        ui.label('PREVIEW NOT IMPLEMENTED').classes('text-[10px] text-gray-600 font-black tracking-widest')
                        ui.label('RTSP decoding disabled for stability').classes('text-[8px] text-gray-700 font-bold')
                    
                    self.stream_info = ui.column().classes('w-full h-full hidden relative bg-gray-900/30 p-6 items-center justify-center')
                    with self.stream_info:
                        ui.icon('videocam_off').classes('text-4xl text-cyan-900 mb-2')
                        self.active_name = ui.label('').classes('text-[10px] text-white font-bold uppercase tracking-widest')
                        self.active_url_label = ui.label('').classes('text-[8px] text-cyan-400 font-mono truncate w-full text-center')
                        ui.button('COPY RTSP URL', on_click=self.copy_url).classes('mt-4 text-[9px] bg-cyan-950')

    def set_preview(self, cam):
        """Activates the 'Selected' view for a camera without attempting a video stream."""
        self.active_url = cam['url']
        self.preview_placeholder.set_visibility(False)
        self.stream_info.set_visibility(True)
        self.active_name.text = cam['name'].upper()
        self.active_url_label.text = cam['url']
        ui.notify(f"Selected: {cam['name']}", color='cyan-900')

    def add_camera(self, name_el, url_el):
        if name_el.value and url_el.value:
            self.app_state['cameras'].append({'name': name_el.value, 'url': url_el.value})
            name_el.value = ''; url_el.value = ''
            self.render_list()

    def render_list(self):
        self.list_area.clear()
        with self.list_area:
            if not self.app_state['cameras']:
                ui.label('No cameras registered').classes('text-gray-600 italic text-[10px] py-4 text-center w-full')
            for idx, cam in enumerate(self.app_state['cameras']):
                with ui.row().classes('w-full items-center p-2 bg-gray-900/30 rounded border border-gray-800'):
                    ui.label(cam['name']).classes('text-xs font-bold text-gray-300 flex-1 truncate')
                    ui.button(icon='info', on_click=lambda c=cam: self.set_preview(c)).props('flat dense color=cyan')
                    ui.button(icon='delete', on_click=lambda i=idx: self.remove_camera(i)).props('flat dense color=red-900')

    def copy_url(self):
        if self.active_url:
            ui.run_javascript(f'navigator.clipboard.writeText("{self.active_url}")')
            ui.notify('RTSP URL Copied', type='positive')

    def remove_camera(self, index):
        self.app_state['cameras'].pop(index)
        self.render_list()

class ViewerPopup:
    def __init__(self, path, ext):
        self.path, self.ext, self.dialog = path, ext.lower(), ui.dialog()

    def _format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0: return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def open(self):
        # Fetch local file metadata
        try:
            stats = os.stat(self.path)
            file_size = self._format_size(stats.st_size)
            mod_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            full_path = os.path.abspath(self.path)
        except:
            file_size = "Unknown"; mod_time = "Unknown"; full_path = self.path

        with self.dialog, ui.card().classes('bg-gray-950 w-[90vw] h-[80vh] border border-gray-800 p-0 overflow-hidden flex flex-col'):
            # Primary Header
            with ui.row().classes('w-full justify-between items-center p-4 bg-gray-900 border-b border-gray-800'):
                with ui.column().classes('gap-0'):
                    ui.label(os.path.basename(self.path)).classes('font-black text-cyan-400 uppercase text-sm tracking-tight')
                    ui.label("MEDIA PREVIEW & METADATA").classes('text-[8px] text-gray-500 font-bold tracking-widest mt-0.5')
                ui.button(icon='close', on_click=self.dialog.close).props('flat round color=white size=sm')
            
            # Info Bar (Metadata)
            with ui.row().classes('w-full bg-black/40 px-4 py-2 items-center gap-6 border-b border-gray-800/50'):
                # Size
                with ui.row().classes('items-center gap-1.5'):
                    ui.icon('storage').classes('text-xs text-gray-500')
                    ui.label(file_size).classes('text-[10px] font-mono text-cyan-100')
                # Modified Date
                with ui.row().classes('items-center gap-1.5'):
                    ui.icon('event').classes('text-xs text-gray-500')
                    ui.label(mod_time).classes('text-[10px] font-mono text-gray-400')
                # Path (Flexible/Truncated)
                with ui.row().classes('items-center gap-1.5 flex-1 overflow-hidden'):
                    ui.icon('link').classes('text-xs text-gray-500')
                    ui.label(full_path).classes('text-[10px] font-mono text-gray-500 truncate')

            # Content Area
            with ui.scroll_area().classes('flex-1 w-full p-4 bg-gray-950'):
                if self.ext in ['jpg', 'png', 'jpeg', 'webp']: 
                    ui.image(self.path).classes('max-w-full rounded-lg shadow-2xl border border-gray-800 mx-auto')
                elif self.ext in ['mp4', 'webm', 'ogg', 'avi', 'mkv']: 
                    ui.video(self.path).classes('w-full rounded-lg shadow-2xl border border-gray-800')
                else:
                    try:
                        with open(self.path, 'r', encoding='utf-8', errors='ignore') as f: 
                            ui.markdown(f"```python\n{f.read()}\n```").classes('text-xs')
                    except: 
                        with ui.column().classes('w-full h-full items-center justify-center py-20 opacity-40'):
                            ui.icon('visibility_off').classes('text-6xl text-red-500')
                            ui.label("BINARY OR UNREADABLE DATA").classes('text-red-500 font-black tracking-widest mt-4 text-xs')
        self.dialog.open()

class ConsolePopup:
    def __init__(self, remote_path, deploy_mgr, app_state): # Added app_state
        self.remote_path, self.mgr, self.dialog = remote_path, deploy_mgr, ui.dialog()
        self.app_state = app_state # Store the state
        self.sync_container = None
        self.exec_container = None
        self.con_list = None
        self.scroll_handle = None
        self.local_exec_dir = None
        self.rows = {}
        self.arg_inputs = {} 
        self.arg_toggles = {}

    def open(self):
        with self.dialog, ui.card().classes('bg-black w-[95vw] h-[90vh] border border-green-900 p-0 overflow-hidden flex flex-col'):
            # Header
            with ui.row().classes('bg-gray-900 w-full p-3 items-center justify-between border-b border-green-900/30'):
                ui.label(f"DEPLOYMENT: {os.path.basename(self.remote_path)}").classes('text-green-500 font-black text-[10px]')
                ui.button(icon='close', on_click=self.close).props('flat round color=red-500 size=sm')
            
            # Sync Panel
            self.sync_container = ui.column().classes('w-full p-6 gap-4')
            with self.sync_container:
                ui.label("ASSET SYNCHRONIZATION").classes('text-green-500 font-black text-xs tracking-widest')
                self.sync_area = ui.scroll_area().classes('w-full h-80 border border-gray-900 rounded bg-gray-900/20')

            # Terminal Panel
            self.exec_container = ui.column().classes('flex-1 w-full h-full bg-black overflow-hidden').style('display: none;')
            with self.exec_container:
                ui.label("PROCESS OUTPUT").classes('text-green-500 font-black text-[9px] px-4 py-2 bg-gray-900/50 w-full border-b border-green-900/20')
                self.scroll_handle = ui.scroll_area().classes('flex-1 w-full p-4')
                with self.scroll_handle:
                    self.con_list = ui.column().classes('w-full gap-1 font-mono text-[10px]')

        self.dialog.open()
        client = ui.context.client
        threading.Thread(target=self.start_pipeline, args=(client,), daemon=True).start()

    def start_pipeline(self, client):
        try:
            temp_dir, local_script, discovered_args = self.mgr.prepare_deployment(
                self.remote_path, 
                lambda l: self.init_sync_ui(l, client), 
                lambda f, p: self.update_sync_progress(f, p, client)
            )
            self.local_exec_dir = os.path.dirname(local_script)
            self.render_preflight_panel(discovered_args, local_script, client)
        except Exception as e:
            with client: ui.notify(f"Pipeline Error: {e}", type='negative')

    def push_line(self, line, client):
        """Forces UI update within the correct client context."""
        with client:
            if not self.con_list: return
            with self.con_list:
                ui.label(line).classes('text-green-400 font-mono text-[10px] leading-tight')
            self.scroll_handle.scroll_to(percent=1.0)

    def _render_media_card(self, path, label, icon):
        """Helper to render a thumbnail card in the console."""
        with ui.card().classes('w-48 bg-gray-900 border border-green-900/40 p-0 overflow-hidden cursor-pointer hover:border-green-400 transition-all group my-2 shadow-lg shadow-black/50').on('click', lambda: ViewerPopup(path, path.split('.')[-1]).open()):
            with ui.row().classes('w-full items-center p-1 px-2 gap-2 bg-black/40'):
                ui.icon(icon).classes('text-[10px] text-green-500')
                ui.label(label).classes('text-[8px] font-black text-green-500 tracking-widest')
                ui.label(os.path.basename(path)).classes('text-[8px] text-gray-500 flex-1 truncate text-right')
            
            if label == "IMAGE":
                ui.image(path).classes('w-full h-24 object-cover opacity-80 group-hover:opacity-100')
            else:
                with ui.element('div').classes('w-full h-24 bg-black flex items-center justify-center relative'):
                    ui.icon('play_circle').classes('text-4xl text-green-500/50 group-hover:text-green-500')
                    ui.label("CLICK TO PLAY").classes('absolute bottom-2 text-[7px] font-bold text-gray-600')

    def init_sync_ui(self, file_list, client):
        with client:
            with self.sync_area:
                self.sync_area.clear()
                self.rows.clear()
                for f in file_list:
                    with ui.row().classes('w-full items-center p-2 border-b border-gray-900/30'):
                        ui.label(os.path.basename(f['path'])).classes('text-[10px] text-gray-300 flex-1 truncate')
                        ui.label(f"{f['size']/1024:.1f} KB").classes('text-[10px] text-gray-500 w-24 text-right')
                        with ui.row().classes('w-48 justify-end items-center gap-2'):
                            self.rows[f['path']] = ui.linear_progress(value=f['progress'], show_value=False).classes('w-32 h-1.5 rounded').props('color=green-500 track-color=gray-800')
                            ui.label().bind_text_from(self.rows[f['path']], 'value', backward=lambda v: f"{int(v*100)}%").classes('text-[9px] text-gray-600 w-8 text-right font-mono')
    
    def format_name(self, name: str) -> str:
        """Converts '--first-name' to 'First Name'."""
        # Remove leading dashes and replace hyphens/underscores with spaces
        clean = name.lstrip('-').replace('-', ' ').replace('_', ' ')
        return clean.title()

    def render_arg_row(self, arg, is_optional=True):
        # Identify if this argument binds to a registry (e.g. video-url -> cameras)
        registry_key = None
        arg_name_lower = arg['name'].lower().lstrip('-')
        
        # Access the registry map from the stored app_state
        if arg_name_lower in self.app_state['registry_map']:
            registry_key = self.app_state['registry_map'][arg_name_lower]

        with ui.row().classes('w-full items-center gap-4 p-3 bg-gray-900/40 border border-gray-800/60 rounded'):
            # --- Toggle Logic ---
            if is_optional:
                if arg['type'] == 'bool':
                    self.arg_inputs[arg['name']] = ui.switch(value=arg['default'] or False).props('dark color=cyan')
                else:
                    self.arg_toggles[arg['name']] = ui.switch().props('dark color=cyan')
            else:
                ui.element('div').classes('w-10')

            # --- Label Logic ---
            with ui.column().classes('w-32 gap-0'):
                ui.label(self.format_name(arg['name'])).classes('text-[11px] font-black text-cyan-400 leading-tight uppercase')
                ui.label(arg['name']).classes('text-[9px] font-mono text-gray-600 font-bold')
            
            # --- Input/Dropdown Logic ---
            if arg['type'] != 'bool':
                # Check if we should render a camera dropdown
                if registry_key == 'cameras':
                    # Text field with integrated camera menu
                    with ui.input(value=str(arg['default'] or "")).props('dark filled dense color=cyan') \
                        .classes('flex-1 font-mono text-xs') as cam_input:
                        
                        if self.app_state['cameras']:
                            with cam_input.add_slot('append'):
                                ui.button(icon='menu_open').props('flat dense round size=sm color=cyan-400')
                                with ui.menu().classes('bg-gray-900 border border-gray-800'):
                                    for cam in self.app_state['cameras']:
                                        ui.menu_item(f"{cam['name']}", on_click=lambda c=cam: cam_input.set_value(c['url']))
                        
                        if is_optional:
                            cam_input.bind_enabled_from(self.arg_toggles[arg['name']], 'value')
                        self.arg_inputs[arg['name']] = [cam_input]
                else:
                    # Standard text input for non-camera arguments
                    n = int(arg['nargs']) if str(arg['nargs']).isdigit() else 1
                    inputs = []
                    with ui.row().classes('flex-1 gap-2'):
                        for i in range(n):
                            val = str(arg['default'][i]) if isinstance(arg['default'], list) and i < len(arg['default']) else str(arg['default'] or "")
                            field = ui.input(value=val).props('dark dense filled color=cyan').classes('flex-1 font-mono text-xs')
                            if is_optional:
                                field.bind_enabled_from(self.arg_toggles[arg['name']], 'value')
                            inputs.append(field)
                    self.arg_inputs[arg['name']] = inputs
                    
    # Inside ConsolePopup class in Utils.py
    def render_preflight_panel(self, discovered_args, local_script, client):
        with client:
            self.sync_container.clear()
            with self.sync_container:
                ui.label("MISSION CONFIGURATION").classes('text-cyan-400 font-black text-xs tracking-[0.3em] mb-4')
                
                if discovered_args['positional']:
                    for arg in discovered_args['positional']: self.render_arg_row(arg, False)
                if discovered_args['optional']:
                    for arg in discovered_args['optional']: self.render_arg_row(arg, True)

                ui.button('INITIATE SEQUENCE', on_click=lambda: self.execute_with_args(local_script, discovered_args, client)) \
                    .classes('w-full bg-cyan-950 font-black text-white mt-8 py-3')

    def execute_with_args(self, local_script, discovered_args, client):
        cmd_args = []
        for arg in discovered_args['positional']:
            widgets = self.arg_inputs.get(arg['name'], [])
            cmd_args.extend([w.value for w in widgets if hasattr(w, 'value')])
        
        for arg in discovered_args['optional']:
            name = arg['name']
            if arg['type'] == 'bool':
                if self.arg_inputs[name].value: cmd_args.append(name)
            elif name in self.arg_toggles and self.arg_toggles[name].value:
                cmd_args.append(name)
                cmd_args.extend([w.value for w in self.arg_inputs.get(name, []) if hasattr(w, 'value')])

        with client:
            self.sync_container.style('display: none;') # Use style for immediate effect
            self.exec_container.style('display: flex;')
            self.push_line(f"[System] Command: python3 {os.path.basename(local_script)} {' '.join(cmd_args)}", client)
            self.mgr.run_script(local_script, lambda line: self.push_line(line, client), args=cmd_args)
            
    # Inside ConsolePopup class
    def update_sync_progress(self, file_path, progress, client):
        with client:
            # Debug print to console if still failing: 
            # print(f"Updating {file_path} to {progress}")
            if file_path in self.rows:
                self.rows[file_path].set_value(progress)
        
    def close(self): 
        self.mgr.cleanup(); self.dialog.close()

class RPiManagerApp:
    def __init__(self):
        self.system = SystemMonitor()
        self.ftp = FTPService()
        self.deploy_mgr = DeploymentManager(self.ftp)
        try: user_val = getpass.getuser()
        except: user_val = os.environ.get('USER', 'ADMIN')
        self.state = {
            'user': user_val,
            'host_ip': self.system.get_ip_address(),
            'ftp_connected': False,
            'metrics': {'cpu': 0, 'ram': 0, 'disk': 0, 'temp': 0, 'volts': 0},
            'cameras': [],
            'registry_map': {
                'camera': 'cameras',
                'camera-url': 'cameras',
                'camera_url': 'cameras',
                'video-url': 'cameras',
                'video_url': 'cameras',
                'stream': 'cameras'
            }
        }

    def run(self):
        global ui
        from nicegui import ui as nicegui_ui
        ui = nicegui_ui

        @ui.page('/')
        def index():
            ui.colors(primary='#06b6d4', dark='#020617')
            ui.dark_mode().enable()
            HeaderComponent(self.state).render()

            with ui.column().classes('w-full p-4 lg:p-8 max-w-7xl mx-auto gap-6'):
                # 1. Collapsible Camera Registry
                with ui.expansion('CAMERA REGISTRY & ASSET BINDING', icon='videocam').classes('w-full bg-gray-950 border border-gray-800 rounded-lg text-gray-400 font-bold'):
                    self.cam_registry = CameraRegistryComponent(self.state)
                    self.cam_registry.render()

                # 2. Collapsible File Browser
                with ui.expansion('FILE SYSTEM & DEPLOYMENT', icon='folder_special').classes('w-full bg-gray-950 border border-gray-800 rounded-lg'):
                    with ui.grid().classes('w-full p-4 gap-6 grid-cols-1 lg:grid-cols-2'):
                        lb = FileBrowserComponent("SERVER", False, self.state, self.ftp, self.deploy_mgr)
                        lb.render()
                        rb = FileBrowserComponent("CLIENT", True, self.state, self.ftp, self.deploy_mgr)
                        rb.render()
                        lb.target_browser, rb.target_browser = rb, lb

            ui.timer(2.0, lambda: self.state['metrics'].update(self.system.get_metrics()))
        ui.run(title='Pi Manager', port=8080, reload=False, show=False)

class DeployCLI:
    """Handles self-modification and network discovery for Pi deployment."""

    def __init__(self):
        self.script_path = os.path.abspath(__file__)
        self.script_name = os.path.basename(self.script_path)
        self.manager_dir = os.path.dirname(self.script_path)
        self.data = self._load_data()
        self._paramiko_available = self._check_paramiko()

    def _check_paramiko(self):
        try:
            import paramiko
            return True
        except ImportError:
            return False

    def _load_data(self):
        config = {"Name": "", "Password": "", "Host": ""}
        try:
            with open(self.script_path, 'r') as f:
                content = f.read()
                data_block = re.search(r'# \[Data\](.*?)# \[Data End\]', content, re.DOTALL)
                if data_block:
                    block_text = data_block.group(1)
                    config["Name"] = self._extract(block_text, "Name")
                    config["Password"] = self._extract(block_text, "Password")
                    config["Host"] = self._extract(block_text, "Host")
        except Exception as e:
            print(f"Error reading self: {e}")
        return config

    def _extract(self, text, key):
        match = re.search(rf'# CLI\.{key} = "(.*?)"', text)
        return match.group(1) if match else ""

    def _update_file(self, updates):
        new_data = self.data.copy()
        new_data.update(updates)

        temp_path = self.script_path + ".tmp"
        try:
            with open(self.script_path, 'r') as f_in, open(temp_path, 'w') as f_out:
                in_block = False
                block_replaced = False

                for line in f_in:
                    if "# [Data]" in line and "# [Data End]" not in line and not block_replaced:
                        in_block = True
                        f_out.write(line)
                        f_out.write(f'# CLI.Name = "{new_data["Name"]}"\n')
                        f_out.write(f'# CLI.Password = "{new_data["Password"]}"\n')
                        f_out.write(f'# CLI.Host = "{new_data["Host"]}"\n')
                    elif "# [Data End]" in line and in_block:
                        in_block = False
                        block_replaced = True
                        f_out.write(line)
                    elif not in_block:
                        f_out.write(line)

            os.replace(temp_path, self.script_path)
            self.data = new_data
            print("Successfully updated internal configuration.")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"Failed to update file: {e}")

    def _get_ssh_client(self):
        if not self._paramiko_available:
            raise ImportError("Paramiko is required for this operation.")
        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self.data['Host'], 
            username=self.data['Name'], 
            password=self.data['Password'],
            timeout=5.0
        )
        return client

    def _ssh_verify(self, ip, username, password):
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(ip, username=username, password=password, timeout=2.0, banner_timeout=2.0)
            ssh.close()
            return True
        except Exception:
            return False

    def _verify_pi(self, ip, username, password):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((ip, 22)) != 0:
                return False
        
        if not self._paramiko_available:
            return True 
            
        return self._ssh_verify(ip, username, password)

    def discover(self, subnet=None):
        if not self._paramiko_available:
            print("[Warning] 'paramiko' not found. Discovery will only check if port 22 is open.")

        saved_host = self.data.get("Host")
        if saved_host:
            print(f"Checking currently saved host: {saved_host}...")
            if self._verify_pi(saved_host, self.data['Name'], self.data['Password']):
                print(f"[!] Saved host {saved_host} is active and verified. Skipping search.")
                return

        if not subnet:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                local_ip = s.getsockname()[0]
                subnet = '.'.join(local_ip.split('.')[:-1]) + '.0/24'
            except Exception:
                subnet = '192.168.1.0/24'
            finally:
                s.close()

        print(f"Starting discovery on {subnet}...")
        prefix = '.'.join(subnet.split('/')[0].split('.')[:-1]) if '/' in subnet else '.'.join(subnet.split('.')[:-1])

        found_ip = None
        ips_to_scan = [f"{prefix}.{i}" for i in range(1, 255)]
        
        max_workers = 40
        checked_count = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._verify_pi, ip, self.data['Name'], self.data['Password']): ip 
                for ip in ips_to_scan
            }
            
            for future in as_completed(futures):
                ip = futures[future]
                is_pi = future.result()
                with lock:
                    checked_count += 1
                    sys.stdout.write(f"\rProgress: {checked_count}/254 | Searching: {ip:<15}")
                    sys.stdout.flush()

                if is_pi:
                    print(f"\n[!] Verified Pi found at {ip}")
                    found_ip = ip
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
        
        if found_ip:
            self._update_file({"Host": found_ip})
        else:
            print("\nDiscovery finished. No verified Pi discovered.")

    def run_remote_pipeline(self, local_path):
        if not os.path.exists(local_path):
            print(f"Error: {local_path} not found.")
            return

        abs_local_path = os.path.abspath(local_path)
        try:
            relative_script_path = os.path.relpath(abs_local_path, self.manager_dir).replace('\\', '/')
        except ValueError:
            relative_script_path = os.path.basename(abs_local_path)

        hash_id = hashlib.md5(abs_local_path.encode()).hexdigest()[:12]
        remote_tmp_dir = f"/tmp/pi_mgr_{hash_id}"
        print(f"Initializing remote pipeline: {remote_tmp_dir}")

        try:
            client = self._get_ssh_client()
            sftp = client.open_sftp()
            client.exec_command(f"mkdir -p {remote_tmp_dir}")
            remote_main = f"{remote_tmp_dir}/{relative_script_path}"
            remote_script_dir = os.path.dirname(remote_main)
            if remote_script_dir != remote_tmp_dir:
                client.exec_command(f"mkdir -p {remote_script_dir}")
                
            print(f"Uploading script: {relative_script_path}")
            sftp.put(abs_local_path, remote_main)
            sftp.chmod(remote_main, 0o755)

            with open(abs_local_path, 'r') as f:
                content = f.read()
            
            include_block = re.search(r'# \[Include\](.*?)# \[Include End\]', content, re.DOTALL)
            if include_block:
                deps = re.findall(r'#\s*-\s*(\S+)', include_block.group(1))
                print(f"Found {len(deps)} dependencies to sync...")
                for dep in deps:
                    clean_dep_path = dep.replace('\\', '/')
                    src_dep = os.path.normpath(os.path.join(self.manager_dir, clean_dep_path))
                    dst_dep = f"{remote_tmp_dir}/{clean_dep_path}"
                    
                    if os.path.exists(src_dep):
                        remote_subdir = os.path.dirname(dst_dep)
                        client.exec_command(f"mkdir -p {remote_subdir}")
                        sftp.put(src_dep, dst_dep)
                        print(f"Synced: {clean_dep_path}")
            sftp.close()

            print(f"[Pipeline] Assets synced to Pi. Launching on {self.data['Host']}...")
            exec_cmd = f"cd {remote_tmp_dir} && python3 -u {relative_script_path}"
            stdin, stdout, stderr = client.exec_command(exec_cmd)

            def stream_output(pipe):
                for line in iter(pipe.readline, ""):
                    print(line, end="")

            threading.Thread(target=stream_output, args=(stdout,), daemon=True).start()
            threading.Thread(target=stream_output, args=(stderr,), daemon=True).start()
            
            rc = stdout.channel.recv_exit_status()
            print(f"\n[Finished] Remote process exited with code {rc}")
            print(f"Cleaning up remote path: {remote_tmp_dir}")
            client.exec_command(f"rm -rf {remote_tmp_dir}")
            client.close()
        except Exception as e:
            print(f"Remote pipeline failed: {e}")

    def deploy(self, service_name=None):
        print(f"Deploying {self.script_name} to {self.data['Host']}...")
        try:
            client = self._get_ssh_client()
            sftp = client.open_sftp()
            remote_path = f"/home/{self.data['Name']}/{self.script_name}"
            sftp.put(self.script_path, remote_path)
            sftp.chmod(remote_path, 0o755)
            sftp.close()
            print(f"File uploaded to {remote_path}")

            if service_name:
                service_content = f"""[Unit]
Description=Pi Manager Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 {remote_path}
WorkingDirectory=/home/{self.data['Name']}
StandardOutput=inherit
StandardError=inherit
Restart=always
User={self.data['Name']}

[Install]
WantedBy=multi-user.target
"""
                temp_path = f"{self.script_path}.service"
                with open(temp_path, "w") as f: f.write(service_content)
                sftp = client.open_sftp()
                sftp.put(temp_path, f"/tmp/{service_name}.service")
                sftp.close()
                os.remove(temp_path)
                client.exec_command(f"sudo mv /tmp/{service_name}.service /etc/systemd/system/")
                client.exec_command("sudo systemctl daemon-reload")
                client.exec_command(f"sudo systemctl enable {service_name}")
                client.exec_command(f"sudo systemctl start {service_name}")
            client.close()
        except Exception as e:
            print(f"Deployment failed: {e}")

    def destroy(self):
        try:
            client = self._get_ssh_client()
            stdin, stdout, stderr = client.exec_command("ls /etc/systemd/system/*.service")
            services = stdout.read().decode().splitlines()
            for s in services:
                si, so, se = client.exec_command(f"cat {s}")
                if self.script_name in so.read().decode():
                    s_name = os.path.basename(s)
                    client.exec_command(f"sudo systemctl stop {s_name}")
                    client.exec_command(f"sudo systemctl disable {s_name}")
                    client.exec_command(f"sudo rm {s}")
            client.exec_command(f"rm /home/{self.data['Name']}/{self.script_name}")
            client.close()
        except Exception as e:
            print(f"Destroy failed: {e}")

    def remote_exec(self, cmd, desc):
        try:
            client = self._get_ssh_client()
            client.exec_command(cmd)
            client.close()
        except Exception as e:
            print(f"Error: {e}")

    def status(self):
        try:
            client = self._get_ssh_client()
            cmds = {
                "CPU Usage": "top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4\"%\"}'",
                "Memory": "free -m | awk 'NR==2{printf \"Memory Usage: %s/%sMB (%.2f%%)\\n\", $3,$2,$3*100/$2 }'",
                "Disk": "df -h | awk '$NF==\"/\"{printf \"Disk Usage: %d/%dGB (%s)\\n\", $3,$2,$5}'",
                "Temp": "vcgencmd measure_temp",
                "Power": "vcgencmd get_throttled",
                "Process": f"pgrep -f {self.script_name} || echo 'Not running'"
            }
            for label, cmd in cmds.items():
                stdin, stdout, stderr = client.exec_command(cmd)
                print(f"{label}: {stdout.read().decode().strip()}")
            client.close()
        except Exception as e:
            print(f"Status failed: {e}")

    def shell(self):
        try:
            client = self._get_ssh_client()
            channel = client.invoke_shell()
            def receiver():
                while not channel.exit_status_ready():
                    if channel.recv_ready():
                        sys.stdout.write(channel.recv(1024).decode('utf-8', errors='ignore'))
                        sys.stdout.flush()
                    time.sleep(0.01)
            threading.Thread(target=receiver, daemon=True).start()
            while not channel.exit_status_ready():
                try:
                    user_input = input()
                    if user_input.lower() == 'exit': break
                    channel.send(user_input + "\n")
                except EOFError: break
            channel.close()
            client.close()
        except Exception as e: print(f"Shell failed: {e}")

    def init_hailo(self):
        try:
            client = self._get_ssh_client()
            sftp = client.open_sftp()
            sftp.put("hailo_install.sh", f"/home/{self.data['Name']}/hailo_install.sh")
            sftp.chmod(f"/home/{self.data['Name']}/hailo_install.sh", 0o755)
            sftp.close()
            stdin, stdout, stderr = client.exec_command(f"/home/{self.data['Name']}/hailo_install.sh")
            for line in iter(stdout.readline, ""): print(f"[Remote] {line}", end="")
            client.close()
        except Exception as e: print(f"Init failed: {e}")

    def download(self, path, dest=None):
        try:
            client = self._get_ssh_client()
            sftp = client.open_sftp()
            local_path = os.path.join(os.path.dirname(self.script_path), dest if dest else os.path.basename(path))
            sftp.get(f"/home/{self.data['Name']}/{path}", local_path)
            sftp.close()
            client.close()
        except Exception as e: print(f"Download failed: {e}")

    def get_value(self, key):
        key = "Host" if key.lower() == "ip" else key.capitalize()
        print(f"{key}: {self.data.get(key)}")

    def set_value(self, key, value):
        key = "Host" if key.lower() == "ip" else key.capitalize()
        if key in ["Name", "Password", "Host"]: self._update_file({key: value})

    def start_ftp_server(self, root_path, user=None, password=None):
        from pyftpdlib.authorizers import DummyAuthorizer
        from pyftpdlib.handlers import FTPHandler
        from pyftpdlib.servers import FTPServer
        authorizer = DummyAuthorizer()
        authorizer.add_user(user or self.data['Name'], password or self.data['Password'], os.path.abspath(root_path), perm="elradfmwMT")
        handler = FTPHandler
        handler.authorizer = authorizer
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            lip = s.getsockname()[0]
        except: lip = "127.0.0.1"
        finally: s.close()
        try: server = FTPServer(("0.0.0.0", 21), handler)
        except: server = FTPServer(("0.0.0.0", 2121), handler)
        server.serve_forever()

    def run(self):
        parser = argparse.ArgumentParser(description="Pi Manager CLI")
        subparsers = parser.add_subparsers(dest="command")
        subparsers.add_parser('discover').add_argument('subnet', nargs='?')
        set_p = subparsers.add_parser('set'); set_p.add_argument('key', choices=['name', 'password', 'ip']); set_p.add_argument('value')
        read_p = subparsers.add_parser('read'); read_p.add_argument('key', choices=['name', 'password', 'ip'])
        subparsers.add_parser('deploy').add_argument('service', nargs='?')
        subparsers.add_parser('destroy'); subparsers.add_parser('start'); subparsers.add_parser('kill'); subparsers.add_parser('reboot'); subparsers.add_parser('shutdown'); subparsers.add_parser('status'); subparsers.add_parser('shell'); subparsers.add_parser('init')
        subparsers.add_parser('run').add_argument('local_path')
        dl_p = subparsers.add_parser('download'); dl_p.add_argument('path'); dl_p.add_argument('dest', nargs='?')
        ftp_p = subparsers.add_parser('ftp'); ftp_p.add_argument('root_path'); ftp_p.add_argument('--credentials', nargs=2, metavar=('USER', 'PASS'))
        args = parser.parse_args()
        if args.command == 'discover': self.discover(args.subnet)
        elif args.command == 'set': self.set_value(args.key, args.value)
        elif args.command == 'read': self.get_value(args.key)
        elif args.command == 'deploy': self.deploy(args.service)
        elif args.command == 'destroy': self.destroy()
        elif args.command == 'start': self.remote_exec(f"nohup python3 ~/{self.script_name} > /dev/null 2>&1 &", "Starting script")
        elif args.command == 'kill': self.remote_exec(f"pkill -f {self.script_name}", "Killing script processes")
        elif args.command == 'reboot': self.remote_exec("sudo reboot", "Rebooting Pi")
        elif args.command == 'shutdown': self.remote_exec("sudo shutdown now", "Shutting down Pi")
        elif args.command == 'status': self.status()
        elif args.command == 'shell': self.shell()
        elif args.command == 'init': self.init_hailo()
        elif args.command == 'run': self.run_remote_pipeline(args.local_path)
        elif args.command == 'ftp':
            u, pw = args.credentials if args.credentials else (None, None)
            self.start_ftp_server(args.root_path, u, pw)
        elif args.command == 'download': self.download(args.path, args.dest)

if __name__ == "__main__":
    cli = DeployCLI()
    if len(sys.argv) > 1:
        cli.run()
    else:
        RPiManagerApp().run()