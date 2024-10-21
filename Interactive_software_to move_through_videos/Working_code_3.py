import cv2
import os
import numpy as np

video_path = r'C:\Users\komal\Downloads\Machine_learning\Videos\MAH00085.mp4'
frames_dir = r'C:\Users\komal\Downloads\Machine_learning\Videos\Resize'
os.makedirs(frames_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(video_fps)

resize_width = int(input("Enter the desired width for resizing: "))
resize_height = int(input("Enter the desired height for resizing: "))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(frames_dir, fourcc, video_fps, (resize_width, resize_height))

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    output.write(resized_frame)
    frames.append(resized_frame)

    cv2.imshow('frame', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print('frame read successfully')
cap.release()
output.release()

cap_resize = cv2.VideoCapture(frames_dir)
original_frame_count = int(cap_resize.get(cv2.CAP_PROP_FRAME_COUNT))
cap_resize.set(cv2.CAP_PROP_FRAME_COUNT, original_frame_count)
cap_resize.release()
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

print(captured_frames)

cap = cv2.VideoCapture(video_path)

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

        frames_dir = r'C:\Users\komal\Downloads\Machine_learning\Videos\Frames'
        frame_path = os.path.join(frames_dir, "frame_{}.jpg".format(frame_count))
        cv2.imwrite(frame_path, frame)

        print('Show frame:', frame_count)
        cv2.imshow('frame', frame)

    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        break

cap.release()
print('Total Frames: {}'.format(frame_count))
