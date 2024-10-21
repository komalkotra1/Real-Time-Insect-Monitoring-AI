import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Load the YOLOv8 model
model = YOLO('C:/Users/komal/University/Machine_learning/yolov10//detect/train6/weights/best.pt')

# Path to the video file
video_path = "C:/Users/komal/University/Machine_learning/yolov8/videos/MAH00088.MP4"
cap = cv2.VideoCapture(video_path)

# Get the frame width, height, and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
print("Video FPS:", video_fps)

# Define video writer to save the output video
output_path = "output_with_filtered_detections.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_width, frame_height))

# Frame buffer to store 7 frames
frame_buffer = []

# Frame counter to track the frame number
frame_counter = 0

# Sliding window counter for console output
sliding_window_counter = 1

# Read and process the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        frame_counter += 1  # Update frame counter for each frame

        # Store the current frame in the buffer
        frame_buffer.append((frame_counter, frame.copy()))  # Store both frame number and the frame

        # Process when we have 7 frames
        if len(frame_buffer) == 7:
            # Track detections in each frame
            detection_counts = []

            # First, count the detections in each frame (we'll need this for the sliding window decision)
            for (frame_number, f) in frame_buffer:
                results = model.track(f, persist=True)
                current_frame_detections = len(results[0].boxes) if len(results) > 0 else 0
                detection_counts.append(current_frame_detections)

            # Sum the total number of detections in the sliding window (for the 7 frames)
            total_detections_in_window = sum(detection_counts)

            # Print the sliding window header
            print(f"\nSliding Window {sliding_window_counter}:")

            # Print detection info for each frame in the sliding window
            for i, (frame_number, f) in enumerate(frame_buffer):
                # Get the detections in the frame
                detections_in_frame = detection_counts[i]

                if detections_in_frame > 0:
                    # Access the bounding boxes of the detections in the current frame
                    results = model.track(f, persist=True)
                    boxes = results[0].boxes  # Get the bounding boxes
                    box_coords = []

                    # Collect the coordinates of each box
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get box coordinates
                        box_coords.append(f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

                    # Print the detection info with box coordinates on the same line
                    print(
                        f"  Frame {frame_number}: {detections_in_frame} insect(s) detected, Boxes: {', '.join(box_coords)}")
                else:
                    print(f"  Frame {frame_number}: None")

            # Decision: Ignore detections if there are 3 or fewer detections across all 7 frames
            if total_detections_in_window > 3:
                print(f"  Total detections in this sliding window: {total_detections_in_window}")

                # Annotate and display the frames if the total detections exceed the threshold
                for i, (frame_number, f) in enumerate(frame_buffer):
                    results_buffer = model.track(f, persist=True)
                    annotated_frame = results_buffer[0].plot()
                    out.write(annotated_frame)
                    # Optionally show annotated frame
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)

            else:
                # Ignore detections: Show and save the original frames without annotations
                print(f"  Ignoring this sliding window due to low detections.")
                for (frame_number, f) in frame_buffer:
                    out.write(f)
                    # Optionally show original frame
                    cv2.imshow("YOLOv8 Tracking", f)

            # Clear the frame buffer after processing the 7-frame window
            frame_buffer.clear()

            # Increment the sliding window counter
            sliding_window_counter += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
