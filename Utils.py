#!/usr/bin/env python3

import os
import sys
import posixpath

try:
    import paramiko
    import importlib
    import sys
except ImportError:
    print("paramiko is required. Install with: pip install paramiko")
    sys.exit(1)
    
class RunRemote:
    """
    Upload a local file to a Raspberry Pi and run it with Python, then remove it.

    Usage:
        python run_remote.py /path/to/local_script.py

    Hardcoded connection settings are below (change as needed).
    """

    # --- Hardcoded connection variables ---
    HOST = "192.168.37.205"     # Raspberry Pi IP
    PORT = 22
    USERNAME = "imang"            # remote user
    PASSWORD = "imang"     # remote password
    # ---------------------------------------

    REMOTE_TEMP_DIR = f"/home/{USERNAME}/.temp"
    
    @staticmethod
    def ensure_remote_dir(sftp, path):
        # create directories recursively on the remote side (posix path)
        dirs = []
        while path and path not in ("/", ""):
            dirs.append(path)
            path, _ = posixpath.split(path)
        dirs.reverse()
        for d in dirs:
            try:
                sftp.stat(d)
            except Exception:
                try:
                    sftp.mkdir(d)
                except Exception:
                    # ignore races/other errors
                    pass

    @staticmethod
    def collect_included_files(main_local_path, script_dir):
        """
        If the first line of the main file equals "# Include", collect consecutive
        next lines that start exactly with "# - " and return list of included paths
        (resolved to absolute paths). Otherwise return empty list.
        """
        includes = []
        try:
            with open(main_local_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return includes

        if not lines:
            return includes

        if lines[0].strip() != "# Include:":
            return includes

        # start from second line, collect consecutive lines starting exactly with "# - "
        for ln in lines[1:]:
            if not ln.startswith("# - "):
                break
            rel = ln[len("# - "):].strip()
            if not rel:
                continue
            if os.path.isabs(rel):
                included_path = os.path.normpath(rel)
            else:
                included_path = os.path.normpath(os.path.join(script_dir, rel))
            includes.append(included_path)
        return includes

    @staticmethod
    def map_local_to_remote(local_paths, script_dir):
        upload_map = {}
        for lp in local_paths:
            try:
                rel = os.path.relpath(lp, script_dir)
                if rel.startswith(".."):
                    remote_sub = os.path.basename(lp)
                else:
                    remote_sub = posixpath.join(*rel.split(os.sep))
            except Exception:
                remote_sub = os.path.basename(lp)
            remote_path = posixpath.join(RunRemote.REMOTE_TEMP_DIR, remote_sub)
            upload_map[lp] = remote_path
        return upload_map

    @staticmethod
    def upload_files(sftp, upload_map):
        for local_path, remote_path in upload_map.items():
            print(f"Uploading {local_path} -> {remote_path} ...")
            sftp.put(local_path, remote_path)

    @staticmethod
    def main(*args):
        if len(args) < 2:
            print("Usage: python run_remote.py relative/path/to/local_file.py (relative to this script)")
            sys.exit(1)

        arg_path = args[1]

        # Resolve the given path relative to this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isabs(arg_path):
            local_path = os.path.normpath(arg_path)
        else:
            local_path = os.path.normpath(os.path.join(script_dir, arg_path))

        if not os.path.isfile(local_path):
            print("Local file not found:", local_path)
            sys.exit(1)

        # Collect additional files if requested
        additional_files = RunRemote.collect_included_files(local_path, script_dir)

        # Build list of files to upload: main file first, then additional ones (avoid duplicates)
        upload_local_paths = [local_path]
        main_abs = os.path.normcase(os.path.abspath(local_path))
        for p in additional_files:
            if not os.path.isfile(p):
                print("Included local file not found:", p)
                sys.exit(1)
            if os.path.normcase(os.path.abspath(p)) != main_abs:
                upload_local_paths.append(p)

        # Map local -> remote paths (preserve relative path under REMOTE_TEMP_DIR)
        upload_map = RunRemote.map_local_to_remote(upload_local_paths, script_dir)

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        sftp = None

        try:
            ssh.connect(RunRemote.HOST, port=RunRemote.PORT, username=RunRemote.USERNAME, password=RunRemote.PASSWORD, timeout=10)
            sftp = ssh.open_sftp()

            # ensure all remote directories exist
            remote_dirs = set(posixpath.dirname(rp) for rp in upload_map.values())
            for d in remote_dirs:
                RunRemote.ensure_remote_dir(sftp, d)

            # upload files
            RunRemote.upload_files(sftp, upload_map)

            # Run only the specified main file
            remote_main = upload_map[local_path]
            run_cmd = f'python3 "{remote_main}" || python "{remote_main}"'
            print("Executing:", run_cmd)
            stdin, stdout, stderr = ssh.exec_command(run_cmd)

            # stream output
            for line in stdout:
                sys.stdout.write(line)
            for line in stderr:
                sys.stderr.write(line)

            exit_status = stdout.channel.recv_exit_status()
            print("Remote script exit status:", exit_status)

            # remove the uploaded files
            for rp in upload_map.values():
                try:
                    print("Removing remote file:", rp)
                    sftp.remove(rp)
                except Exception:
                    pass

        except Exception as e:
            print("Error:", e)
            # best-effort cleanup of uploaded files
            try:
                if sftp:
                    for rp in upload_map.values():
                        try:
                            sftp.remove(rp)
                        except Exception:
                            pass
            except Exception:
                pass
        finally:
            try:
                if sftp:
                    sftp.close()
            except Exception:
                pass
            try:
                ssh.close()
            except Exception:
                pass

def get_class(argument):
    # If already a class, return it
    if isinstance(argument, type):
        return argument
    if not isinstance(argument, str):
        raise TypeError("argument must be a class or string")

    arg = argument.strip()
    if not arg:
        raise ValueError("empty class argument")

    # Try to resolve simple name from this module first
    if '.' not in arg and ':' not in arg:
        obj = globals().get(arg)
        if isinstance(obj, type):
            return obj
        # fallback to builtins
        built = getattr(__builtins__, arg, None) if isinstance(__builtins__, dict) else getattr(__builtins__, arg, None)
        if isinstance(built, type):
            return built
        raise ImportError(f"Cannot resolve class name '{argument}'")

    # Split module and class (support "module:Class" or "module.Class")
    if ':' in arg:
        mod_name, cls_name = arg.split(':', 1)
    else:
        mod_name, cls_name = arg.rsplit('.', 1)

    # Import module (use already-loaded if present)
    if not mod_name:
        module = sys.modules.get(__name__)
    else:
        module = sys.modules.get(mod_name) or importlib.import_module(mod_name)

    cls = getattr(module, cls_name, None)
    if not isinstance(cls, type):
        raise TypeError(f"Resolved object '{argument}' is not a class")
    return cls

def main():
    class_name = sys.argv[1] if len(sys.argv) > 1 else None
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    print("Arguments:", class_name, args)
    if not class_name:
        print("Usage: python Utils.py <ClassName or Module:ClassName>")
        sys.exit(1)
    cl = get_class(class_name)
    if not cl:
        print(f"Class '{class_name}' not found.")
        sys.exit(1)
    print("Running command:", cl)
    cl.main(args)

if __name__ == "__main__":
    main()