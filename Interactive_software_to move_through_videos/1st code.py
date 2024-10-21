# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:13:11 2023

@author: komal
"""

import cv2
import os
video_path = r'C:\Users\komal\Downloads\Machine_learning\Videos\MAH00083.mp4'
frames_dir = r'C:\Users\komal\Downloads\Machine_learning\Videos\Frames'
os.makedirs(frames_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
Video_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Video_size:', Video_size)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame_count:', frame_count)

for i in range(frame_count):
    ret, frame = cap.read()
    frame_path = os.path.join(frames_dir, 'frame_{}.jpg'.format(i))
    cv2.imwrite(frame_path, frame)
    #cv2.imwrite(os.path.join(frames_dir, 'frame_{}.jpg'.format(i), frame))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print('done')


