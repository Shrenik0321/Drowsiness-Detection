# Model testing on real time

import ultralytics
ultralytics.checks()
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO

model = YOLO("models/best-3.pt")
# awake_results = model('images/img3.jpg')
# for a in awake_results:
#     a.show()
# drowsy_results = model('images/img4.jpg') 
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
cap.release()
