import cv2
import os
from ultralytics import YOLO

model = YOLO('C:/Users/komal/University/Machine_learning/yolov10/yolov10_n/runs/detect/train/weights/best.pt')

video_path = "C:/Users/komal/University/Machine_learning/data/Videos/MAH00088.MP4"
cap = cv2.VideoCapture(video_path)
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_n.avi', fourcc, video_fps, (video_width, video_height))

counter = 0
while cap.isOpened():
    success, frame = cap.read()
    counter += 1

    if success:
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        # Display the annotated frame
        #cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release everything when finished
cap.release()
output_video.release()
cv2.destroyAllWindows()
