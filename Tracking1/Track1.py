import cv2
from ultralytics import YOLO


model = YOLO('C:/Users/komal/University/Machine_learning/yolov10/yolov10_x/runs/detect/train/weights/best.pt')
video_path = "C:/Users/komal/University/Machine_learning/data/Videos/MAH00088.MP4"
cap = cv2.VideoCapture(video_path)
video_fps= int(cap.get(cv2.CAP_PROP_FPS))
print(video_fps)

counter=0
while cap.isOpened():
    success, frame = cap.read()
    counter+=1

    if success:
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        #print(results[0].boxes)

        print(counter," ", results[0].boxes.id, " ",results[0].boxes.cls," ", results[0].boxes.conf )
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
