import time

import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from class_myolo import methods as mt
import os
import time

# write e compar turn on the functions
writ = 0
compar = 1

# Help the function compare
W = 0

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
# The first frame showed in the video take more time to be show
# So, we show a random image just to initiate the camera
frame = "C:/Users/hugol/Pictures/dog.jpg"
results = model(frame)
annotated_frame = results[0].plot()

listvideo =[]
# Open the video
#dir = "gta@jetson01:/media/ssd/HugoAntunes/vid..."
#path = "gta@jetson01:/media/ssd/HugoAntunes/tests/"
dir = "C:/Users/hugol/Desktop/IC/datasets/train/"
path = "C:/Users/hugol/Desktop/IC/tests/"
videos = os.listdir(dir)

for x in range(len(videos)):
    VID = dir+videos[x]
    cap = cv.VideoCapture(VID)
    # Loop through the video frames

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        start = time.time()
        if not ret:
            break

        totalRGB = len(frame)*len(frame[0])*3
        thresh = totalRGB*0.1
        imga = frame

        # Function to compare
        try:
            boolean, valor = mt.hyolo(imgb,imga,thresh)[0],mt.hyolo(imgb,imga,thresh)[1]
            classes = (results[0].boxes.cls.tolist())
            coord = (results[0].boxes.xyxy.tolist())
            # Frames iguais, W = 1
        except: valor = 0
        imgb = frame
        
        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        end = time.time()
        listvideo += [[end - start,valor]]
# write in a txt the results
with open('first_test.txt','w') as arquivo:
    for x in range(len(listvideo)):
        arquivo.write(str(listvideo[x])+'\n')