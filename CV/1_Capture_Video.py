# Modules normally used
#import numpy as np
import cv2

# Get a video
cap = cv2.VideoCapture(0)               # Capture from the webcam
#cap = cv2.VideoCapture("filename.avi") # Capture video file

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Out operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Video frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()