
import cv2
import os
video_path = r'C:\Users\komal\Downloads\Machine_learning\Videos\23.mp4'
frames_dir = r'C:\Users\komal\Downloads\Machine_learning\Videos\Frames\Sample_Video\3rd_run'
os.makedirs(frames_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
Video_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Video_size:', Video_size)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame_count:', frame_count)

frame_interval = int(video_fps /10)
for i in range(frame_count):
    ret, frame = cap.read()
    if i % frame_interval == 0:
        frame_path = os.path.join(frames_dir, 'frame_{}.jpg'.format(i))
        cv2.imwrite(frame_path, frame)
    # cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print('done')
pressedKey= cv2.waitKey(1)&0xFF