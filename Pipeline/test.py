import os
import sys

import argparse
from HailoPipeline import RTSPSource, LetterboxAdapter, HailoInference, HailoPipeline, SaveListener, MQTTListener

# [Include]
# - TryB/HailoPipeline.py
# [Include end]

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

    # Shared configuration
    HEF = "/home/nightshadearia/hailo-rpi5-examples/resources/yolov8n.hef"
    POST_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"

    # Setup the listener based on arguments
    if args.save:
        listener = SaveListener(args.save)
    else:
        broker_parts = args.mqtt[0].split(':')
        broker_ip = broker_parts[0]
        broker_port = int(broker_parts[1]) if len(broker_parts) > 1 else 1883 # Default MQTT port
        username = args.mqtt[1]
        password = args.mqtt[2] if len(args.mqtt) > 2 else None
        topic = args.mqtt[3] if len(args.mqtt) > 3 else "hailo/detections" # Default topic

        print(f'<mqtt start "{broker_ip}" "{broker_port}" "{topic}" "{username}" "{password}">', flush=True)
        listener = MQTTListener(broker=broker_ip, port=broker_port, topic=topic, username=username, password=password)

    # Initialize and run
    pipeline = HailoPipeline(RTSPSource(args.camera), LetterboxAdapter(), HailoInference(HEF, POST_SO), listener)



    from gi.repository import GLib
    GLib.timeout_add_seconds(int(args.time), lambda: (pipeline.stop(), False)[1])
    pipeline.run()
    if args.save:
        print(f"<video {args.save}>")
    elif args.mqtt:
        print(f'<mqtt end "{broker_ip}" "{broker_port}" "{topic}">', flush=True)

if __name__ == "__main__":
    main()