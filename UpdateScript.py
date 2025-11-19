import argparse
import getpass
import os
import stat
import sys
import socket
from pathlib import Path
import paramiko

# Default fallbacks to use when no CLI values are provided — adjust as needed
DEFAULT_IP = "192.168.37.205"
DEFAULT_NAME = "imang"
DEFAULT_PASSWORD = "imang"
REMOTE_LOCATION = f"/home/{DEFAULT_NAME}/NightshadeAria/RaspberryPi"

#!/usr/bin/env python3
"""
UpdateScript.py

Upload the local RaspberryPi folder (default: this script's directory) to a Raspberry Pi over SSH/SFTP.

Usage examples:
        python UpdateScript.py --host 192.168.1.50 --user pi --remote /home/pi/project
        python UpdateScript.py --host pi.example.com --user pi --remote /opt/myapp --key /home/user/.ssh/id_rsa

This script uses Paramiko (pip install paramiko). It will try key-based auth first (agent/keys),
or use the --key file if supplied, or prompt for a password.
"""

def parse_args():
    p = argparse.ArgumentParser(description="Upload a local folder to a Raspberry Pi via SFTP")
    p.add_argument("--host", default=DEFAULT_IP, help="Raspberry Pi hostname or IP (default from DEFAULT_IP)")
    p.add_argument("--user", default=DEFAULT_NAME, help="SSH username (default: current user or DEFAULT_NAME)")
    p.add_argument("--port", type=int, default=22, help="SSH port (default: 22)")
    p.add_argument("--local", default=os.path.dirname(os.path.abspath(__file__)),
                   help="Local folder to upload (default: this script's directory)")
    p.add_argument("--remote", help="Remote destination folder on the Pi (default: /home/<user>/<local_folder_basename>)")
    p.add_argument("--key", help="Path to private key file (optional)")
    p.add_argument("--password", default=DEFAULT_PASSWORD, help="Password (optional). If omitted and no key, you'll be prompted.")
    p.add_argument("--preserve-perms", action="store_true",
                   help="Try to preserve file permissions when uploading (best-effort)")
    return p.parse_args()


def sftp_mkdirs(sftp, remote_path):
    # Create remote directories recursively (like mkdir -p)
    dirs = []
    head = remote_path
    while head not in ("", "/"):
        try:
            sftp.stat(head)
            break
        except IOError:
            dirs.append(head)
            head = os.path.dirname(head)
    for d in reversed(dirs):
        try:
            sftp.mkdir(d)
        except IOError:
            pass


def upload_directory(sftp, local_dir, remote_dir, preserve_perms=False):
    local_dir = os.path.join(os.path.abspath(local_dir), "RaspberryPi")
    if not os.path.isdir(local_dir):
        raise ValueError(f"Local path is not a directory: {local_dir}")

    for root, dirs, files in os.walk(local_dir):
        rel_root = os.path.relpath(root, local_dir)
        if rel_root == ".":
            rel_root = ""
        target_root = os.path.join(remote_dir, rel_root).replace("\\", "/")

        # ensure remote directory exists
        sftp_mkdirs(sftp, target_root)

        # set directory permissions if requested
        if preserve_perms:
            try:
                mode = os.stat(root).st_mode & 0o777
                sftp.chmod(target_root, mode)
            except Exception:
                pass

        for fname in files:
            local_path = os.path.join(root, fname)
            remote_path = os.path.join(target_root, fname).replace("\\", "/")

            # skip special files
            if stat.S_ISFIFO(os.stat(local_path).st_mode):
                continue

            # SFTP put with a simple progress callback
            def progress(transferred, total, lp=local_path):
                pct = (transferred / total * 100) if total > 0 else 0
                print(f"\rUploading {lp} -> {remote_path} {transferred}/{total} bytes ({pct:.1f}%)", end="")

            try:
                sftp.put(local_path, remote_path, callback=lambda a, b: progress(a, b))
                print()  # newline after progress
                if preserve_perms:
                    try:
                        mode = os.stat(local_path).st_mode & 0o777
                        sftp.chmod(remote_path, mode)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Failed to upload {local_path} -> {remote_path}: {e}", file=sys.stderr)


def main():
    args = parse_args()
    local_dir = args.local

    # If remote not specified, default to /home/<user>/<basename_of_local_dir>
    if not args.remote:
        base = os.path.basename(os.path.abspath(local_dir)) or "upload"
        remote_dir = REMOTE_LOCATION
    else:
        remote_dir = args.remote

    # Connect SSH
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    pkey = None
    if args.key:
        try:
            pkey = paramiko.RSAKey.from_private_key_file(os.path.expanduser(args.key))
        except Exception as e:
            print(f"Could not load key {args.key}: {e}", file=sys.stderr)
            return 1

    password = args.password
    if not password and not pkey:
        # give user chance to use password if no key provided/available
        password = getpass.getpass(f"Password for {args.user}@{args.host}: ")

    try:
        client.connect(args.host, port=args.port, username=args.user,
                       password=password, pkey=pkey, timeout=10, allow_agent=True, look_for_keys=True)
    except (paramiko.ssh_exception.AuthenticationException, socket.error) as e:
        print(f"SSH connection failed: {e}", file=sys.stderr)
        return 2

    try:
        sftp = client.open_sftp()
        # make base remote dir
        sftp_mkdirs(sftp, remote_dir)
        upload_directory(sftp, local_dir, remote_dir, preserve_perms=args.preserve_perms)
        sftp.close()
    finally:
        client.close()

    print("Upload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())