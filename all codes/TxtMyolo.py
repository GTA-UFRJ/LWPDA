# This code is responsible to writing the coordenates from each video.
# Just to automatize the process, we create one code to create the txt file
# that contains each detection from each frame.
# The txt is in this form:
# [nºclass[coord.],nºclass[coord.]
# Each line is from one frame, so line 1 -> frame 1, line 2 -> frame 2 etc.

import time
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from myolo import myolo as my
import os
from comparisons import methods as comp


writ = 1
compar = 1
W = 0

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Path to dataset

dir = "C:/Users/amoot/Desktop/IMAGENET/amostra/"
VID = (os.listdir(dir))
for tr in range(10):
    path = f"C:/Users/amoot/Desktop/GTA/PUYM/tests/histogram/txt_0.{tr}/"

    for x in range(len(VID)):
        lista = []
        file = VID[x]
        cap = cv.VideoCapture(dir + VID[x])
        video = VID[x]
        F = -1 # Contador de frames (loops)
        # Loop through the video frames
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            #totalRGB = len(frame)*len(frame[0])*3
            #thresh = totalRGB*0
            thresh = tr/10
            imga = frame
            #A cada loop, adicionar um ao contador (para ajudar na nomeação do txt)
            #name = str(video)+ '_' +str(F) #nome do arquivo txt

            try: # FUNÇÃO DE COMPARAÇÃO 
                if comp.histogram(imgb,imga,thresh):
                    frame = imgb

                    classes = (results[0].boxes.cls.tolist())
                    coord = (results[0].boxes.xyxy.tolist())
                    lista += [[classes]+[coord]]

                    # Frames iguais, W = 1
                    W = 1
            except: pass

            if W == 1: # Repetir a marcação e o frame passado
                annotated_frame = results[0].plot()
                #cv.imshow("YOLOv8 Inference", annotated_frame)
                #if cv.waitKey(1) & 0xFF == ord("q"):
                    #break
            else:
                # Run YOLOv8 inference on the frame
                results = model(frame)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                classes = (results[0].boxes.cls.tolist())
                coord = (results[0].boxes.xyxy.tolist())
                lista += [[classes]+[coord]]
                # Display the annotated frame
                #cv.imshow("YOLOv8 Inference", annotated_frame)
                imgb = frame
                # Break the loop if 'q' is pressed
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            #Zerar o verificador
            W = 0
        print(lista)
        my.write(file, lista, A = 1, path = str(path))
