import os
import cv2
from ultralytics import YOLO

model = YOLO('C:/Users/komal/University/Machine_learning/yolov8/runs/detect/train20/weights/best.pt')

image_directory = "yolov8/images/train"

image_files = [f
               for f in os.listdir(image_directory)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    frame = cv2.imread(image_path)

    counter=0
    bucket=[]
    if frame is not None:
       results = model.track(frame, persist=True)
       bucket.append(results)
       counter+=1
       annotated_frame = results[0].plot()
       for i in len(bucket):
           print(bucket[i])
       cv2.imshow("YOLOv8 Tracking", annotated_frame)
       delay= 500
       if cv2.waitKey(delay) & 0xFF == ord("q"):
          break

cv2.destroyAllWindows()
