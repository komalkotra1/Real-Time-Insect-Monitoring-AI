import cv2
import os

video_path = r'C:\Users\komal\Downloads\Machine_learning\Videos\MAH00087.mp4'
frames_dir = r'C:\Users\komal\Downloads\Machine_learning\Videos\Frames'
os.makedirs(frames_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
print(video_fps)

frame_interval = int(video_fps / 100)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

batch_size = 1000  # Number of frames to process in each batch

start_frame = 0
end_frame = 0
range_started = False

while True:
    success, frame = cap.read()

    if not success:
        break

    if range_started and end_frame == start_frame + batch_size:
        range_started = False

    if not range_started:
        start_frame = end_frame
        end_frame = start_frame + batch_size
        if end_frame > frame_count:
            end_frame = frame_count
        range_started = True

    for i in range(start_frame, end_frame):
        frame_path = os.path.join(frames_dir, "frame_{}.jpg".format(i))
        cv2.imwrite(frame_path, frame)

cap.release()
print('Extracted Frames: {}'.format(frame_count))
