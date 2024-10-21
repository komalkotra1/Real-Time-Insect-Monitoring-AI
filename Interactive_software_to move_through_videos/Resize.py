import cv2
import os

video_path= r'C:\Users\komal\Downloads\Machine_learning\Videos\MAH00089.mp4'
output_path = r'C:\Users\komal\Downloads\Machine_learning\Videos\Resize_MAH00089.mp4'

output_width = 500
output_height = 250


cap = cv2.VideoCapture(video_path)


fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

while True:
    success, frame = cap.read()

    if not success:
        break


    resized_frame = cv2.resize(frame, (output_width, output_height))


    output.write(resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
output.release()

cap_output = cv2.VideoCapture(output_path)
original_frame_count = int(cap_output.get(cv2.CAP_PROP_FRAME_COUNT))
cap_output.set(cv2.CAP_PROP_FRAME_COUNT, original_frame_count)
cap_output.release()

print("Video resizing complete.")
