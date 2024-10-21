import cv2
from ultralytics import YOLO
import torch
import pandas as pd

model = YOLO('C:/Users/komal/University/Machine_learning/yolov8/runs/detect/train27/weights/best.pt')

video_path = "C:/Users/komal/University/Machine_learning/data/Videos/MAH00088.MP4"
cap = cv2.VideoCapture(video_path)
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
print(video_fps)
output_video_path = "C:/Users/komal/University/Machine_learning/yolov8/output_video_with_boxes2.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, video_fps, (int(cap.get(3)), int(cap.get(4))))

while True:
    success, frame = cap.read()

    if not success:
        break


    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()

    # Accessing and printing the tensor values
    cls_tensor = results[0].boxes.cls
    conf_tensor = results[0].boxes.conf
    id_tensor = results[0].boxes.id
    boxes_tensor = results[0].boxes.xyxy
    boxes_numpy = boxes_tensor.numpy()

    if cls_tensor is not None and id_tensor is not None and conf_tensor is not None:
        for i, box in enumerate(boxes_numpy):
            # coordinates
            x1, y1, x2, y2 = box[:4]

            # Draw box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            class_label = cls_tensor[i].item()
            if class_label ==4.0:
                class_label =2.0

            # Draw text (ID, Class, Confidence)
            text = f"ID: {id_tensor[i].item()}, Class: {class_label}, Confidence: {conf_tensor[i].item()}"
            cv2.putText(annotated_frame, text, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)

        # Write the frame to the output video
    output_video.write(annotated_frame)

    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
output_video.release()
cv2.destroyAllWindows()