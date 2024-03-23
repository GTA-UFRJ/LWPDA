import time

import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from class_myolo import myolo as my
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
    start = time.time()
    # Loop through the video frames

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        totalRGB = len(frame)*len(frame[0])*3
        thresh = totalRGB*0.1
        imga = frame

        try: 
            if my.compare(imgb,imga,thresh,compar):
                #frame = imgb
                classes = (results[0].boxes.cls.tolist())
                coord = (results[0].boxes.xyxy.tolist())
                # Frames iguais, W = 1
                W = 1
        except: pass

        if W == 1:
            vasco = Annotator(frame)
            for r in range (len(classes)):
                vasco.box_label(coord[r],str(int(classes[r])))
            gama = vasco.result()
            #cv.imshow("YOLOv8 Inference", gama)
            #if cv.waitKey(1) & 0xFF == ord("q"):
                #break
        else:
            # Run YOLOv8 inference on the frame
            results = model(frame)
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            imgb = frame
            # Break the loop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        #Zerar o verificador
        W = 0
    end = time.time()
    listvideo += [end - start]
    # Release the video capture object and close the display window
# write in a txt the results
with open(path+'testvideo.txt','w') as arquivo:
    for x in range(len(listvideo)):
        arquivo.write(str(listvideo[x])+'\n')