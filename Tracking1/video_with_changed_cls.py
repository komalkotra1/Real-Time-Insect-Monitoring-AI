import cv2
from ultralytics import YOLO
import torch

import pandas as pd

model = YOLO('C:/Users/komal/University/Machine_learning/yolov8/runs/detect/train27/weights/best.pt')

video_path = "C:/Users/komal/University/Machine_learning/data/Videos/MAH00088.MP4"
cap = cv2.VideoCapture(video_path)
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
print(video_fps)

output_video_path = "C:/Users/komal/University_Machine_learning/yolov8/output_video_with_changed_cls_tracking.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, video_fps, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        cls_tensor = results[0].boxes.cls
        new_cls_tensor = torch.tensor([2.0])  #  set class label to 2.0

        # Ensure new_cls_tensor has the same number of rows as the original bounding box coordinates
        new_cls_tensor = new_cls_tensor.unsqueeze(0).repeat(results[0].boxes.xyxy.size(0), 1)

        # Concatenate the modified class tensor with the original bounding box coordinates
        new_boxes = torch.cat([results[0].boxes.xyxy[:, :4], new_cls_tensor,
                               results[0].boxes.xyxy[:, 4:]], dim=1)

        # Create a new Boxes object with the modified bounding box coordinates and class tensor
        new_boxes_object = results[0].boxes.__class__(new_boxes, orig_shape=results[0].boxes.orig_shape)

        results[0].boxes = new_boxes_object

        # Write the frame to the output video
        output_video.write(annotated_frame)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()
