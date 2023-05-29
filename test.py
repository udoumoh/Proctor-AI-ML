import cv2
import os

# Open the video file
video_path = 'cheating2.mpeg'
video = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not video.isOpened():
    print("Error opening video file")

# Initialize variables
frame_count = 0
frame_rate = 1  # Extract a frame after every 6 frames (adjust as needed)

# Loop through the video frames
while True:
    # Read the next frame from the video
    ret, frame = video.read()

    # Check if frame was successfully read
    if not ret:
        break

    # Process the frame (e.g., save, display, etc.)
    frame_count += 1

    # Extract frames based on the frame rate
    if frame_count % frame_rate == 0:
        frame_filename = os.path.join('TD2/cheating', f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Display the frame (optional)
        cv2.imshow('Frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
video.release()
cv2.destroyAllWindows()