import argparse
import time
import os
import csv
import cv2
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import json
from HailoPipeline import (FileSource, LetterboxAdapter,
                           HailoInference, HailoTracker, AppSink, HailoPipeline, HailoListener)

class BenchmarkListener(HailoListener):
    def __init__(self, pipeline, run_id, detection_log, max_frames=150):
        self.pipeline = pipeline
        self.run_id = run_id
        self.detection_log = detection_log
        self.detection_log[self.run_id] = [] # Initialize the list for this run
        self.max_frames = max_frames
        self.frame_count = 0
        self.start_time = None
        self.end_time = None
        self.total_detections = 0
        self.stopping = False

    def on_data_received(self, frame, detections):
        if self.frame_count == 0:
            self.start_time = time.time()
        
        self.frame_count += 1
        self.total_detections += len(detections)

        # Save the raw detection data for this frame
        self.detection_log[self.run_id].append({
            "frame_count": self.frame_count,
            "num_detections": len(detections),
            "detections": detections # AppSink already formats this nicely as a dict
        })

        if self.frame_count >= self.max_frames and not self.stopping:
            self.stopping = True
            self.end_time = time.time()
            print(f"\nTarget frames ({self.max_frames}) reached. Sending EOS...")
            GLib.idle_add(self.pipeline.pipeline.send_event, Gst.Event.new_eos())

    def on_stop(self):
        if not self.end_time:
            self.end_time = time.time()
            print(f"\nStream ended early. Processed {self.frame_count} frames.")

    def get_results(self):
        if self.start_time and self.end_time and (self.end_time > self.start_time) and self.frame_count > 0:
            fps = (self.frame_count - 1) / (self.end_time - self.start_time)
        else:
            fps = 0.0
        avg_detections = self.total_detections / self.frame_count if self.frame_count else 0.0
        return round(fps, 2), round(avg_detections, 2)

def record_live_feed(source_input, num_frames, output_filename="temp_benchmark_video.mp4"):
    print(f"\n[Pre-Recording] Capturing {num_frames} frames from {source_input}...")
    cap = cv2.VideoCapture(int(source_input) if source_input.isdigit() else source_input)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source_input}")
        return None
    
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    if fps <= 0 or fps != fps: fps = 30.0
        
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frames_recorded = 0
    while frames_recorded < num_frames:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
        frames_recorded += 1
        if frames_recorded % 30 == 0:
            print(f"  -> Recorded {frames_recorded}/{num_frames} frames...")
            
    cap.release()
    out.release()
    print(f"[Pre-Recording] Video saved to {output_filename}\n")
    return output_filename

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--output", default="benchmark_results.csv")
    args = parser.parse_args()

    # The exact files we just downloaded
    target_hefs = ['yolov8n_h8l.hef', 'yolov8s_h8l.hef', 'yolov8m_h8l.hef', 'yolov8l_h8l.hef', 'yolov8x_h8l.hef']
    post_so_path = 'libyolo_hailortpp_post.so'

    # Verify files exist before starting
    missing = [f for f in target_hefs + [post_so_path] if not os.path.exists(f)]
    if missing:
        print(f"Error: The following required files are missing from this directory: {missing}")
        return

    # Convert to absolute paths so GStreamer's C-backend doesn't get lost
    target_hefs = [os.path.abspath(f) for f in target_hefs]
    post_so_path = os.path.abspath(post_so_path)

    # Standardize Source
    if args.source.isdigit() or args.source.startswith("rtsp://"):
        video_path = record_live_feed(args.source, args.frames)
        if not video_path: return
    else:
        video_path = os.path.abspath(args.source)

    source = FileSource(location=video_path)
    results = []
    all_detections = {}
    
    for hef_path in target_hefs:
        model_name = os.path.basename(hef_path).split('_')[0] 
        for use_tracking in [False, True]:
            for include_frame in [False, True]:
                print(f"\n--- Testing: {model_name} | Tracking: {use_tracking} | Materialize Frames: {include_frame} ---")
                
                # Create a unique ID for this run
                run_id = f"{model_name}_Track-{use_tracking}_Frames-{include_frame}"
                
                adapter = LetterboxAdapter()
                inference = HailoInference(hef_path, post_so_path)
                tracker = HailoTracker() if use_tracking else None
                sink = AppSink(include_frame=include_frame)
                
                pipeline = HailoPipeline(source, adapter, inference, sink, tracker)
                
                # --- ADD THIS LINE ---
                # Remove leaky queues so heavier models process every single frame
                pipeline.pipeline_str = pipeline.pipeline_str.replace("leaky=2 max-size-buffers=5", "")
                
                listener = BenchmarkListener(pipeline, run_id, all_detections, max_frames=args.frames)
                pipeline.add_listener(listener)
                
                pipeline.run()
                
                fps, avg_det = listener.get_results()
                print(f"Result -> FPS: {fps}, Avg Detections/Frame: {avg_det}")
                
                results.append({
                    "Model": model_name,
                    "Tracking Enabled": use_tracking,
                    "Frame Materialization": include_frame,
                    "FPS": fps,
                    "Avg Detections": avg_det
                })
                time.sleep(2) # Let Hailo hardware cool down/flush queues

    # Save Results
    with open(args.output, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # NEW: Save JSON Detection Data
    with open('benchmark_detections.json', 'w') as json_file:
        json.dump(all_detections, json_file, indent=4)

    print(f"\nBenchmarking complete! Run 'streamlit run app.py' to view results.")

if __name__ == "__main__":
    main()