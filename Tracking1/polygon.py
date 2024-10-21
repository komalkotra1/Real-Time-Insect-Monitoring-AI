import cv2
from ultralytics import YOLO
import torch
import pandas as pd
import numpy as np

model = YOLO('C:/Users/komal/University/Machine_learning/yolov8/runs/detect/train27/weights/best.pt')

video_path = "C:/Users/komal/University/Machine_learning/data/Videos/MAH00088.MP4"
cap = cv2.VideoCapture(video_path)
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
print(video_fps)
success, first_frame = cap.read()
midpoints_list=[]
cv2.imwrite("C:/Users/komal/University/Machine_learning/polygon.jpg", first_frame)

counter = 0

while cap.isOpened():
    success, frame = cap.read()
    counter += 1

    if success:
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        boxes_tensor = results[0].boxes.xyxy
        boxes_numpy = boxes_tensor.numpy()

        for i, box in enumerate(boxes_numpy):
                # coordinates
                x1, y1, x2, y2 = box[:4]
                # Calculate midpoint
                midpoint_x = (x1 + x2) / 2
                midpoint_y = (y1 + y2) / 2
                midpoint = (midpoint_x, midpoint_y)
                midpoints_list.append(midpoint)


        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        break

cap.release()
midpoints_int = [(int(midpoint[0]), int(midpoint[1])) for midpoint in midpoints_list]
arr = np.array(midpoints_int)
midpoints_int= arr.reshape((-1,1,2))
color = (0, 255, 0)  # Green color
thickness = 1

# Draw the polygon using the midpoints
cv2.polylines(first_frame, [midpoints_int], isClosed=True, color=color, thickness=thickness)
cv2.imwrite(r"C:/Users/komal/University/Machine_learning/Image_with_polygon.jpg", first_frame)
# Display the image with the polygon
cv2.imshow("Image with Polygon", first_frame)
cv2.waitKey(0)

print("time that an insect spend on the flower in seconds:", counter/video_fps)
cv2.destroyAllWindows()


