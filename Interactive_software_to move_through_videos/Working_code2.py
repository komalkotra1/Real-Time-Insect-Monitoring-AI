import cv2
import os
import numpy as np

video_path = r'C:\Users\komal\Downloads\Machine_learning\Videos\MAH00088.mp4'
frames_dir = r'C:\Users\komal\Downloads\Machine_learning\Videos\Frames'
os.makedirs(frames_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(video_fps)

resize_width = int(input("Enter the desired width for resizing: "))
resize_height = int(input("Enter the desired height for resizing: "))

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    frames.append(resized_frame)

    cv2.imshow('frame', resized_frame)

    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        break

cap.release()
frames_array = np.array(frames)
print('Shape of frames array:', frames_array.shape)

i = 0
step = 1

captured_frames = []

while True:
    if i == len(frames_array):
        i = 0

    frame = frames_array[i]
    print('Show next')
    cv2.imshow('frame', frame)

    pressedKey = cv2.waitKey(40) & 0xFF
    if pressedKey == ord('b'):
        step -= 1
    if pressedKey == ord('q'):
        break
    if pressedKey == ord('s'):
        step = 0
    if pressedKey == ord('g'):
        step = 1
    if pressedKey == ord('c'):
        frame_path = os.path.join(frames_dir, "frame_{}.jpg".format(i))
        print(frame_path)
        cv2.imwrite(frame_path, frame)
        captured_frames.append(i)

    i += step

    print('step:', step)

print('Extracted Frames: {}'.format(len(captured_frames)))

cap = cv2.VideoCapture(video_path)
i=0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)
    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        break

cap.release()
if captured_frames:
    with open('captured_frames_23.txt', 'w') as file:
        for frame_index in captured_frames:
            file.write(str(frame_index) + '\n')
cv2.destroyAllWindows()
