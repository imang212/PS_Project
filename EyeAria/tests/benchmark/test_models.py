import os
import gi

# Fix for the PyGIWarning: Explicitly require version 1.0 before importing Gst/GLib
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

def test_hef_model(hef_path):
    print(f"\n--- Testing Model: {hef_path} ---")
    
    # We use videotestsrc to generate 10 dummy frames to push through the model
    # This completely isolates the HEF file from any camera or network issues
    pipeline_str = (
        f"videotestsrc num-buffers=10 ! "
        f"video/x-raw,format=RGB,width=640,height=640 ! "
        f"hailonet hef-path={hef_path} ! fakesink"
    )
    
    pipeline = Gst.parse_launch(pipeline_str)
    bus = pipeline.get_bus()
    pipeline.set_state(Gst.State.PLAYING)
    
    # Block and wait for either an Error or an End-Of-Stream (EOS) message
    msg = bus.timed_pop_filtered(
        Gst.CLOCK_TIME_NONE,
        Gst.MessageType.ERROR | Gst.MessageType.EOS
    )
    
    success = False
    if msg:
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"❌ [FAIL] Incompatible or broken model.")
            # Print the exact reason (e.g., the HAILO8 vs HAILO8L mismatch)
            print(f"   Reason: {err}") 
        elif msg.type == Gst.MessageType.EOS:
            print(f"✅ [PASS] Model loaded successfully and processed frames!")
            success = True
            
    # Clean up the pipeline memory
    pipeline.set_state(Gst.State.NULL)
    return success

def main():
    # Find all .hef files in the current directory
    hef_files = [f for f in os.listdir('.') if f.endswith('.hef')]
    
    if not hef_files:
        print("No .hef files found in the current directory.")
        return

    print(f"Found {len(hef_files)} .hef files. Starting diagnostics...")
    
    results = {}
    for hef in hef_files:
        results[hef] = test_hef_model(hef)
        
    print("\n" + "="*30)
    print("       DIAGNOSTIC SUMMARY")
    print("="*30)
    for hef, passed in results.items():
        status = "✅ Working (Hailo-8L Compatible)" if passed else "❌ Broken (Mismatch/Corrupt)"
        print(f"{hef.ljust(20)} : {status}")

if __name__ == "__main__":
    main()

