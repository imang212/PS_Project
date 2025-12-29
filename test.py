import os
import sys

import argparse
from HailoPipeline import RTSPSource, LetterboxAdapter, HailoInference, HailoPipeline, SaveListener, MQTTListener

# [Include]
# - TryB/HailoPipeline.py
# [Include end]

def main():
    parser = argparse.ArgumentParser(description="Hailo AI Kit Demo Script")
    parser.add_argument("--source", default="rtsp://192.168.0.105:8080/h264_ulaw.sdp", help="RTSP URL or file path")
    parser.add_argument("--mode", choices=["save", "mqtt"], default="save", help="Action: 'save' to disk or 'mqtt' upload")
    parser.add_argument("--output", default="output.mkv", help="Path for saved video")
    parser.add_argument("--time", type=int, default=10, help="Seconds to run")
    args = parser.parse_args()

    # Shared configuration
    HEF = "/home/imang/hailo-rpi5-examples/resources/models/hailo8/yolov8m.hef"
    POST_SO = "/home/imang/hailo-rpi5-examples/resources/so/libyolo_hailortpp_postprocess.so"

    # Setup the selected listener
    if args.mode == "save":
        listener = SaveListener(args.output)
    else:
        listener = MQTTListener()

    # Initialize and run
    pipeline = HailoPipeline(RTSPSource(args.source), LetterboxAdapter(), HailoInference(HEF, POST_SO), listener)
    
    from gi.repository import GLib
    GLib.timeout_add_seconds(int(args.time), lambda: (pipeline.stop(), False)[1])
    pipeline.run()

if __name__ == "__main__":
    main()