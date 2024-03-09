import time

import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from myolo import myolo as my
import os
import time

# write e compar turn on the functions
writ = 0
compar = 0

# Help the function compare
W = 0
thresh = 500000

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
# The first frame showed in the video take more time to be show
# So, we show a random image just to initiate the camera
frame = "C:/Users/hugol/Pictures/dog.jpg"
results = model(frame)
annotated_frame = results[0].plot()
cv.imshow("YOLOv8 Inference", annotated_frame)


# Open the video
VID = "C:/Users/hugol/Desktop/IC/datasets/train/ILSVRC2015_train_00005003.mp4"
vidcapture = cv.VideoCapture(VID)
fps = vidcapture.get(cv.CAP_PROP_FPS)
totalframes = vidcapture.get(cv.CAP_PROP_FRAME_COUNT)
totaltime = 0
FRvideo = 1/fps
processedframes = 0
listdelay = []
cap = cv.VideoCapture(VID)

# Loop through the video frames
start1 = time.time()
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()

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
        cv.imshow("YOLOv8 Inference", gama)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        processedframes+=1

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv.imshow("YOLOv8 Inference", annotated_frame)
        imgb = frame
        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    end = time.time()
    totaltime += (end-start)
    measure = end-start
    listdelay += [(measure-FRvideo)]
    #Zerar o verificador
    W = 0
# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()
end = time.time()

pointsy = [listdelay[0]]
for x in range(1,len(listdelay)):
    if (pointsy[x-1]+listdelay[x])<0:
        pointsy+=[0]
    else:
        pointsy += [pointsy[x-1]+listdelay[x]]
print('pointsy = ',pointsy)
pointsx = []
for x in range(len(listdelay)):
    pointsx+= [FRvideo*(x+1)]
print('pointsx = ', pointsx)