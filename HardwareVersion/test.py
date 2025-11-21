# Include:
# - HardwareVersion/PipelineCode.py

from PipelineCode import Pipeline, Node, PipelineData, GStreamerInputNode, HailoYoloNode
import time

class IntermediatePrint(Node):
    """
    A simple debug/test node.
    It prints when data enters and when it forwards to the next node.
    No modifications are made to PipelineData.
    """

    def __init__(self, label="Intermediate"):
        super().__init__()
        self.label = label

    def _process(self, pdata):
        # Announce entry
        print(f"[{self.label}] Received data. Passing to next stage...")

        # Forward to next node
        if self.next_node:
            self.next_node.process(pdata)

        # Optionally announce completion (useful for debugging order)
        print(f"[{self.label}] Stage completed.")

pipeline = (
    Pipeline()
    .add_node(GStreamerInputNode(history_size=3))
)

for i in range(101):
    pdata = pipeline.tick(); time.sleep(0.1)
    if i % 10 == 0:print("Tick.")

print(pdata)

print("Final detections:", pdata.detections)
