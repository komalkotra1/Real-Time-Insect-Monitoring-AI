import cv2
from ultralytics import YOLO
import torch

model = YOLO('C:/Users/komal/University/Machine_learning/yolov8/runs/detect/train27/weights/best.pt')

video_path = "C:/Users/komal/University/Machine_learning/data/Videos/MAH00088.MP4"
cap = cv2.VideoCapture(video_path)
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
print(video_fps)
combined_cls_list = []
combined_conf_list = []
combined_id_list = []
areas_list = []  # List to store areas
midpoints_list = []  # List to store midpoints
counter = 0
while cap.isOpened():
    success, frame = cap.read()
    counter += 1

    if success:
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        # Accessing and printing the tensor values
        cls_tensor = results[0].boxes.cls
        conf_tensor = results[0].boxes.conf
        id_tensor = results[0].boxes.id
        boxes_tensor = results[0].boxes.xyxy

        if cls_tensor is not None and len(cls_tensor) != 2:
            # Converting the tensor to a numpy array
            boxes_numpy = boxes_tensor.numpy()

            # Loop over each bounding box
            for box in boxes_numpy:
                # Extract coordinates
                x1, y1, x2, y2 = box[:4]
                # Calculate midpoint
                midpoint_x = (x1 + x2) / 2
                midpoint_y = (y1 + y2) / 2
                midpoint = (midpoint_x, midpoint_y)
                midpoints_list.append(midpoint)

                # Calculate area of the bounding box
                area = (x2 - x1) * (y2 - y1)
                areas_list.append(area)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
print("Areas list:", areas_list)
print("Midpoints list:", midpoints_list)
cv2.destroyAllWindows()
