import cv2
import os

video_path = r'C:\Users\komal\Downloads\Machine_learning\Videos\DSC_3203.MOV'
frames_dir = r'C:\Users\komal\Downloads\Machine_learning\Videos\Frames'
os.makedirs(frames_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(video_fps)

n_values = input("Enter the value of n (comma-separated): ")
n_values = [int(n.strip()) for n in n_values.split(',')]


frame_count = 0
selected_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count in n_values:
        selected_frames.append(frame_count)

        frame_path = os.path.join(frames_dir, "frame_{}.jpg".format(frame_count))
        cv2.imwrite(frame_path, frame)

        print('Show frame:', frame_count)
        cv2.imshow('frame', frame)

    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        break

cap.release()
print('Total Frames: {}'.format(frame_count))

with open('selected_frames.txt', 'w') as file:
    for frame_index in selected_frames:
        file.write(str(frame_index) + '\n')
