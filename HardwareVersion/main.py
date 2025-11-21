# main.py
import time
import logging
from pipeline.gstreamer_input_node import GStreamerInputNode
from pipeline.yolo_node import HailoYoloNode
from pipeline.pipeline_data import PipelineData

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def build_pipeline():
    # Create nodes
    input_node = GStreamerInputNode(width=640, height=640, history_size=5, filter_type="RGB")
    yolo_node = HailoYoloNode(model_path="compiled_yolovX.hef")  # put your compiled model here

    # Chain nodes: input -> yolo
    input_node.next_node = yolo_node

    # Build GStreamer elements (start camera)
    input_node.build_element()
    # Computation node may not return a Gst element; it initializes its runtime in constructor.

    return input_node  # head of pipeline

def main_loop(head_node):
    try:
        while True:
            # Create a new PipelineData for this iteration
            pdata = PipelineData(request_frame_copy=True)  # ask for Python frame copy for visualization

            # Start the chain: each node will process and forward the data
            head_node.process(pdata)

            # At this point, pdata contains:
            # - pdata.frame_copy (numpy array) if requested and available
            # - pdata.detections (list) from YOLO
            # - pdata.timestamp and meta info
            if pdata.frame_copy is not None:
                # Example: draw boxes using OpenCV (not implemented here)
                # cv2.imshow("preview", pdata.frame_copy)
                pass

            # Example: logging detections
            log.info("Detections: %s", pdata.detections)
            time.sleep(0.01)  # adapt to your required loop rate
    except KeyboardInterrupt:
        log.info("Stopping pipeline loop")

if __name__ == "__main__":
    head = build_pipeline()
    main_loop(head)
