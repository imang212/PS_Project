from yolov5.models.yolo import Model
import torch
from ultralytics import YOLO

model = YOLO("yolov5s.pt")  # supports older YOLOv5 weights too
model.export(format="onnx")  # directly export to ONNX
