import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture("D:/IC/datasets/imagenetVID/videos/train/")

# Loop through the frames
while True:
    # Read the next frame
    ret, frame = cap.read()

    # Check if the frame was read correctly
    if not ret:
        break

    # Convert the frame to grayscale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a new color image with the grayscale values in the green channel
    #height, width = frame.shape[:2]
    #zeros = np.zeros((height, width), dtype=np.uint8)
    #green_image = cv2.merge((zeros, gray, zeros))

    # Display the modified frame
    cv2.imshow("Frame", frame)

    # Wait for the key to be pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the video file
cap.release()
cv2.destroyAllWindows()