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
ui, app = ng.ui, ng.app

# --- Media Viewer UI Component ---

class MediaViewer(ui.dialog):
    """Versatile popup for viewing text, images, and videos."""
    def __init__(self, item: Path):
        super().__init__()
        self.item = item
        self.mime = mimetypes.guess_type(str(item))[0] or ""
        self.build()

    def build(self):
        with self, ui.card().classes('bg-[#050a0f] border border-blue-900 p-0 overflow-hidden').style('width: 80vw; max-width: 1000px;'):
            # Header
            with ui.row().classes('w-full items-center justify-between bg-[#0d1b2a] px-4 py-2 border-b border-blue-900'):
                ui.label(f'VIEWER: {self.item.name}').classes('text-blue-200 font-mono text-xs font-bold truncate')
                ui.button(icon='close', on_click=self.close).props('flat color=white dense size=sm')

            # Content Area
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
                    ui.button('DOWNLOAD TO VIEW', on_click=lambda: ui.download(str(self.item))).props('outline color=blue-300')

    def is_image(self):
        return self.mime.startswith('image/') or self.item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']

    def is_video(self):
        return self.mime.startswith('video/') or self.item.suffix.lower() in ['.mp4', '.webm', '.ogv']

    def is_text(self):
        return self.mime.startswith('text/') or self.item.suffix.lower() in ['.txt', '.log', '.py', '.sh', '.js', '.json', '.html', '.css', '.md', '.cfg', '.yaml']

    def get_data_url(self):
        """Encodes file to base64 for local viewing without a complex file server."""
        try:
            with open(self.item, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('ascii')
                return f"data:{self.mime};base64,{encoded}"
        except Exception:
            return ""

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

class DashboardView:
    """Refined Dashboard with simplified navigation and the Reboot action restored."""
    def __init__(self):
        ui.query('.q-page').classes('p-0')
        self.drawer = self.setup_sidebar()
        self.header = SystemHeader(self.drawer)
        self.setup_content()

    def setup_sidebar(self):
        """Restores the Reboot button to the lower utility section of the sidebar."""
        with ui.left_drawer().style('background-color: #0d1b2a; color: white').classes('column no-wrap') as dr:
            # Top Section: Navigation
            with ui.column().classes('grow w-full'):
                ui.label('SYSTEM').classes('text-[10px] font-bold q-pa-md text-blue-300 tracking-widest')
                ui.item('FILE MANAGER').props('clickable active').classes('text-white text-xs bg-blue-900/20')
            
            ui.space()

            # Bottom Section: System Utilities
            with ui.column().classes('w-full q-pa-md gap-1'):
                ui.separator().classes('bg-blue-900 opacity-50 q-mb-sm')
                
                # RESTORED: Reboot Button
                ui.button('REBOOT', icon='restart_alt', on_click=self.confirm_reboot) \
                    .props('flat color=red-4 dense') \
                    .classes('text-xs w-full justify-start') \
                    .tooltip('Restart the Host Machine')

                # Logout Button
                ui.button('LOGOUT', icon='logout', on_click=self.logout) \
                    .props('flat color=blue-300 dense') \
                    .classes('text-xs w-full justify-start') \
                    .tooltip('End Session')
        return dr

    def setup_content(self):
        """Ensures content fills the responsive browser layout."""
        with ui.column().classes('w-full h-full gap-0'):
            FileBrowser()

    def logout(self):
        """Clears authentication state and reloads."""
        app.storage.user['authenticated'] = False
        ui.navigate.reload()

    async def confirm_reboot(self):
        """Displays a critical confirmation dialog before triggering a system restart."""
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
    @staticmethod
    def extract_args(item: Path):
        if item.suffix != '.py': return None
        args_found = {'positional': [], 'flags': []}
        try:
            tree = ast.parse(item.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'add_argument':
                    arg = ScriptInspector._parse_arg_node(node)
                    if arg['is_flag']: args_found['flags'].append(arg)
                    else: args_found['positional'].append(arg)
            return args_found if any(args_found.values()) else None
        except Exception: return None

    @staticmethod
    def _parse_arg_node(node):
        names = [ast.literal_eval(a) for a in node.args if isinstance(a, (ast.Constant, ast.Str))]
        kwargs = {}
        for k in node.keywords:
            try:
                if k.arg == 'choices' and isinstance(k.value, (ast.List, ast.Tuple)):
                    kwargs[k.arg] = [ast.literal_eval(elt) for elt in k.value.elts]
                elif isinstance(k.value, (ast.Constant, ast.Str, ast.Num)):
                    kwargs[k.arg] = ast.literal_eval(k.value)
                elif isinstance(k.value, ast.Name):
                    kwargs[k.arg] = k.value.id 
            except: pass
        actual_name = names[-1]
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
            'is_flag': is_flag
        }
      
class ExecutionDialog(ui.dialog):
    def __init__(self, item: Path, current_dir: Path):
        super().__init__()
        self.item = item
        self.current_dir = current_dir
        self.process = None
        self.arg_elements = {}
        self.on_value_change(lambda e: self.handle_dialog_close(e))
        self.build_input_view()

    def handle_dialog_close(self, e):
        if not e.value: self.kill_process()

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
            with ui.row().classes('w-full items-center gap-2 mb-2 no-wrap'):
                cb = None
                if is_optional: cb = ui.checkbox().props('dark dense color=emerald')
                ui.label(arg['formatted_name']).classes('text-xs text-gray-300 w-24 truncate').tooltip(arg['help'])
                input_el = self._create_input(arg)
                if cb: input_el.bind_visibility_from(cb, 'value')
                self.arg_elements[arg['actual_name']] = {'check': cb, 'input': input_el, 'type': 'flag' if is_optional else 'positional', 'meta': arg}

    def _create_input(self, arg):
        placeholder = str(arg['default']) if arg['default'] is not None else ''
        if arg['choices']:
            opts = {val: str(val) for val in arg['choices']}
            return ui.select(options=opts, value=arg['default'] or arg['choices'][0]).props('dark dense outline').classes('grow')
        if arg['type'] in ['int', 'float']: return ui.number(placeholder=placeholder).props('dark dense outline').classes('grow')
        return ui.input(placeholder=placeholder).props('dark dense outline').classes('grow')

    def assemble_and_run(self):
        if 'raw' in self.arg_elements: 
            final_args = self.arg_elements['raw']['input'].value
        else:
            cmd_parts = []
            # Process positional and flag arguments
            for name, el in self.arg_elements.items():
                val = el['input'].value
                
                # CASTING LOGIC: Fix float-to-int conversion here
                if val is not None:
                    try:
                        if el.get('meta') and el['meta'].get('type') == 'int':
                            val = int(float(val)) # Convert to float first to handle cases like "1.0"
                    except (ValueError, TypeError):
                        pass

                if el['type'] == 'positional':
                    cmd_parts.append(str(val if val is not None else (el['meta']['default'] or '')))
                
                elif el['type'] == 'flag' and el['check'].value:
                    cmd_parts.append(name)
                    if el['meta']['action'] not in ['store_true', 'store_false']:
                        cmd_parts.append(str(val))
            
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
            self.log_area = ui.log().classes('w-full h-96 bg-black text-emerald-300 font-mono text-[11px] p-4')
        asyncio.create_task(self.run_process(args_str))

    async def run_process(self, args_str: str):
        abs_path = str(self.item.absolute())
        cmd = ['python3', abs_path] if self.item.suffix == '.py' else ['bash', abs_path]
        if args_str: cmd.extend(args_str.split())
        try:
            self.log_area.push(f"--- Launching: {' '.join(cmd)} ---\n")
            self.process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=str(self.current_dir))
            while True:
                line = await self.process.stdout.readline()
                if not line: break
                self.log_area.push(line.decode().rstrip())
            await self.process.wait()
            self.log_area.push(f"\n[DONE] Exit Code: {self.process.returncode}")
        except Exception as e:
            if self.log_area: self.log_area.push(f"\n[SYSTEM ERROR] {str(e)}")

    def rerun(self): self.kill_process(); self.build_input_view()
    def kill_process(self):
        if self.process and self.process.returncode is None:
            try: self.process.kill()
            except ProcessLookupError: pass

# --- Main Application Controller ---

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
                DashboardView()

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
            ui.run(
                port=8080, 
                title="Pi Manager", 
                dark=True, 
                reload=False, 
                storage_secret='midnight_blue_heartbeat_secret',
                on_air="pnTbvBDuc87QIzc7"
            )

# --- Entry Point ---

if __name__ in {"__main__", "__mp_main__"}:
    pi_app = PiManagerApp()
    pi_app.start()