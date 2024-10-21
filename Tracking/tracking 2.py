from ultralytics import YOLO

model = YOLO('C:/Users/komal/University/Machine_learning/yolov8/runs/detect/train18/weights/best.pt')

results = model.track(source='C:/Users/komal/University/Machine_learning/data/Videos/MAH00088.MP4', conf=0.3, iou=0,show=True, tracker="bytetrack.yaml")