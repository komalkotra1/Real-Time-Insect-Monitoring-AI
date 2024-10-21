import cv2
import os
import numpy as np

video_path = r'C:\Users\komal\Downloads\Machine_learning\Videos\Resize_MAH00088.mp4'
frames_dir = r'C:\Users\komal\Downloads\Machine_learning\Videos\Frames'
os.makedirs(frames_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(video_fps)
frame_interval = int(video_fps /50)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

video = np.empty((frame_count, frame_height, frame_width, 3), dtype=np.uint8)

frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break
    video[frame_count] = frame
    frame_count += 1

i = 0
step = 1

captured_frames = []
delay = 100
while True:
    if i == frame_count:
        i = 0

    frame = video[i]
    print('Show next')
    cv2.imshow('frame', frame)

    pressedKey = cv2.waitKey(delay) & 0xFF

    if pressedKey == ord('+'):
        delay -=10
    elif pressedKey == ord('-'):
        delay +=10
    if pressedKey == ord('b'):
        step += -1
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
    if pressedKey == ord('r'):
        start_frame = i
        print('Start frame:', start_frame)
        range_started = True
    elif pressedKey == ord('z') and range_started:
        end_frame = i
        range_started = False
        print('End frame:', end_frame)
        frame_interval = int(video_fps / 25)
        for j in range(start_frame, end_frame, frame_interval):
            frame_path = os.path.join(frames_dir, "frame_{}.jpg".format(j))
            cv2.imwrite(frame_path, video[j])
            captured_frames.append(j)
    print('step:', step)
    i += step
cap.release()
print('Extracted Frames{}'.format(frame_count))

with open('captured_frames.txt', 'w') as file:
    for frame_index in captured_frames:
        file.write(str(frame_index) + '\n')