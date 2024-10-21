import cv2
from ultralytics import YOLO
import torch
import pandas as pd

model = YOLO('C:/Users/komal/University/Machine_learning/yolov8/runs/detect/train27/weights/best.pt')

video_path = "C:/Users/komal/University/Machine_learning/data/Videos/MAH00088.MP4"
cap = cv2.VideoCapture(video_path)
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
print(video_fps)
combined_cls_list = []
combined_conf_list=[]
combined_id_list=[]
midpoints_list=[]
boxes_list=[]
data_list=[]
areas_list=[]
counter = 0

#output_video_path="C:/Users/komal/University/Machine_learning/yolov8/output_video_with_tracking.avi"
fourcc= cv2.VideoWriter_fourcc(*'XVID')
#output_video= cv2.VideoWriter(output_video_path, fourcc, video_fps, (int(cap.get(3)), int(cap.get(4))))
while cap.isOpened():
    success, frame = cap.read()
    counter += 1

    if success:
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        #Write the frame to the output video
        #output_video.write(annotated_frame)

        # Accessing and printing the tensor values
        cls_tensor = results[0].boxes.cls
        conf_tensor = results[0].boxes.conf
        id_tensor = results[0].boxes.id
        boxes_tensor = results[0].boxes.xyxy
        boxes_numpy = boxes_tensor.numpy()

        if cls_tensor is not None and len(cls_tensor) !=2:
            print("Original cls Tensor:")
            print(cls_tensor)

            # Converting the tensor to a numpy array
            cls_numpy = cls_tensor.numpy()
            print("Numpy Array:")
            print(cls_numpy)

            # Converting the numpy array to a Python list
            cls_list = cls_numpy.tolist()
            print("Python List:")
            print(cls_list)
            combined_cls_list.append(cls_list)


            # Accessing the modified tensor as a numpy array
            cls_numpy_modified = cls_tensor.numpy()
            print("Modified Numpy Array:")
            print(cls_numpy_modified)

            for i, box in enumerate(boxes_numpy):
                # coordinates
                x1, y1, x2, y2 = box[:4]
                # Calculate midpoint
                midpoint_x = (x1 + x2) / 2
                midpoint_y = (y1 + y2) / 2
                midpoint = (midpoint_x, midpoint_y)
                midpoints_list.append(midpoint)

                # area of the bounding box
                area = (x2 - x1) * (y2 - y1)
                areas_list.append(area)


        if conf_tensor is not None:
            print("Original Tensor Conf:")
            print(conf_tensor)

            # Converting the tensor to a numpy array
            conf_numpy = conf_tensor.numpy()
            print("Numpy Array:")
            print(conf_numpy)

            # Converting the numpy array to a Python list
            conf_list = conf_numpy.tolist()
            print("Python List:")
            print(conf_list)
            combined_conf_list.append(conf_list)

            # Accessing the modified tensor as a numpy array
            conf_numpy_modified = conf_tensor.numpy()
            print("Modified Numpy Array:")
            print(conf_numpy_modified)

        if id_tensor is not None:
            print("Original Tensor id:")
            print(id_tensor)

            # Converting the tensor to a numpy array
            id_numpy = id_tensor.numpy()
            print("Numpy Array:")
            print(id_numpy)

            # Converting the numpy array to a Python list
            id_list = id_numpy.tolist()
            print("Python List:")
            print(id_list)
            combined_id_list.append(id_list)

            # Accessing the modified tensor as a numpy array
            id_numpy_modified = id_tensor.numpy()
            print("Modified Numpy Array:")
            print(id_numpy_modified)

            box_data = {'ID': id_numpy[i],
                        'Class': cls_numpy[i],
                        'Confidence': conf_numpy[i],
                        'Midpoint': midpoint,
                        'Area': area
                        }
            data_list.append(box_data)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        break

cap.release()
#output_video.release()  #release the videowriter object
print("list with cls:", combined_cls_list)
print("List with conf values:", combined_conf_list)
print("List with Id_list:" , combined_id_list)
print("Area list:", areas_list)
print("Midpoints list:", midpoints_list)

df= pd.DataFrame(data_list)

print("Table with all values:")
print(df)
csv_file_path = "output2.csv"
df.to_csv(csv_file_path, index=False, sep="\t")
print("Table saved to:", csv_file_path)
#claculating the mean
count_class_4 = combined_cls_list.count([4.0])
count_class_2 = combined_cls_list.count([2.0])
count_class_1 = combined_cls_list.count([1.0])
count_class_0 = combined_cls_list.count([0.0])
count_class_3= combined_cls_list.count([3.0])

# Calculate total number of detections
total_detections = len(combined_cls_list)

# Calculate percentage of each class
percentage_class_4 = (count_class_4 / total_detections) * 100
percentage_class_2 = (count_class_2 / total_detections) * 100
percentage_class_1 = (count_class_1 / total_detections) * 100
percentage_class_3 = (count_class_3 / total_detections) * 100
percentage_class_0 = (count_class_0 / total_detections) * 100


# Print the percentages
print("Percentage of class 4.0(fly):", percentage_class_4)
print("Percentage of class 2.0(wasp)", percentage_class_2)

cap.release()
#output_video.release()
cv2.destroyAllWindows()

