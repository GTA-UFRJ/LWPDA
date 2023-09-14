from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

benchmark(model='yolov8n.pt', data='D:\IC\imagenet.yaml', imgsz=640, half=False)
