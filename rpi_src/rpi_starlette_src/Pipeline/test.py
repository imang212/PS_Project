import os
import sys
import argparse
from HailoPipeline import RTSPSource, LetterboxAdapter, HailoInference, HailoPipeline, SaveListener, MQTTListener

# [Include]
# - TryB/HailoPipeline.py
# [Include end]

def find_file_recursive(target_filename, search_roots):
    """
    Searches recursively for a specific filename starting from a list of root directories.
    Returns the first full path found, otherwise returns None.
    """
    print(f"Searching for {target_filename}...", file=sys.stderr)
    
    for root_dir in search_roots:
        # Expand user paths (e.g. ~/)
        expanded_root = os.path.expanduser(root_dir)
        
        if not os.path.exists(expanded_root):
            continue

        # Walk through the directory tree
        for dirpath, dirnames, filenames in os.walk(expanded_root):
            if target_filename in filenames:
                full_path = os.path.join(dirpath, target_filename)
                return full_path
                
    return None

def main():
    parser = argparse.ArgumentParser(description="Hailo AI Kit Demo Script", formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("--camera", required=True, help="RTSP URL of the camera source")
    parser.add_argument("--time", type=int, required=True, help="Seconds to run the pipeline")

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--save",
        metavar="PATH",
        help="Save processed video to a file at PATH"
    )
    output_group.add_argument(
        "--mqtt",
        nargs=4,
        metavar=('BROKER_IP:PORT', 'USERNAME', 'PASSWORD', 'TOPIC'),
        help="""Publish detection results to an MQTT broker.
        Arguments: BROKER_IP:PORT USERNAME PASSWORD TOPIC
        Example: --mqtt mqtt.portabo.cz:8883 videoanalyza phdA9ZNW1vfkXdJkhhbP /videoanalyza"""
    )

    args = parser.parse_args()

    # --- Resource Configuration ---
    
    # Priority list of HEF models to look for
    HEF_PRIORITY_LIST = ["yolov8n.hef", "yolov8m.hef"]
    POST_SO_FILENAME = "libyolo_hailortpp_post.so"

    # Define ROOT directories to start recursive search.
    search_roots = [
        ".",                                  # Current working directory
        "~/hailo-rpi5-examples",              # User examples folder
        "/usr/lib",                           # Common library location
        "/usr/local/lib",                     # Local library location
        "/opt/hailo",                         # Opt location
        "/usr/share/hailo",                   # Share location
        os.path.join(os.path.expanduser("~"), "tappas") # Home tappas folder if exists
    ]

    # Search for HEF (Try Nano first, then Medium)
    hef_path = None
    for hef_name in HEF_PRIORITY_LIST:
        found_path = find_file_recursive(hef_name, search_roots)
        if found_path:
            hef_path = found_path
            break # Stop looking if we found one

    # Search for Post-Process SO
    post_so_path = find_file_recursive(POST_SO_FILENAME, search_roots)

    # Validate resources
    if not hef_path:
        print(f"Error: Could not find any of {HEF_PRIORITY_LIST} anywhere in these trees: {search_roots}")
        sys.exit(1)
    else:
        print(f"Found HEF: {hef_path}")

    if not post_so_path:
        print(f"Error: Could not find '{POST_SO_FILENAME}' anywhere in these trees: {search_roots}")
        sys.exit(1)
    else:
        print(f"Found Post-Process SO: {post_so_path}")

    # --- End Resource Configuration ---

    # Setup the listener based on arguments
    if args.save:
        listener = SaveListener(args.save)
    else:
        broker_parts = args.mqtt[0].split(':')
        broker_ip = broker_parts[0]
        broker_port = int(broker_parts[1]) if len(broker_parts) > 1 else 1883 
        username = args.mqtt[1]
        password = args.mqtt[2] 
        topic = args.mqtt[3] 

        print(f'<mqtt start "{broker_ip}" "{broker_port}" "{topic}" "{username}" "{password}">', flush=True)
        listener = MQTTListener(broker=broker_ip, port=broker_port, topic=topic, username=username, password=password)

    # Initialize and run with found paths
    pipeline = HailoPipeline(RTSPSource(args.camera), LetterboxAdapter(), HailoInference(hef_path, post_so_path), listener)

    from gi.repository import GLib
    GLib.timeout_add_seconds(int(args.time), lambda: (pipeline.stop(), False)[1])
    pipeline.run()
    
    if args.save:
        print(f"<video \"{args.save}\">")
    elif args.mqtt:
        print(f'<mqtt end "{broker_ip}" "{broker_port}" "{topic}">', flush=True)

if __name__ == "__main__":
    main()