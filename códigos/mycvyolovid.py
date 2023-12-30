import time
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from txt import write
from txt import compare
import os

# write e compar ligam (1) suas funções
writ = 0
compar = 1

new_frame_time,prev_frame_time = 0,0
# W é um verificador, caso o frame seja igual W = 1
W = 0
thresh = 500000

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Percorrer o diretório onde contém os arquivos de vídeo
# Open the video

VID = (os.listdir("D:/IC/datasets/imagenetVID/videos/train/"))

for x in range(len(VID)):
    file = VID[x]
    cap = cv.VideoCapture("D:/IC/datasets/imagenetVID/videos/train/" + VID[x])
    video = VID[x]
    path = "C:/Users/hugol/Desktop/IC/train-myolo/"
    F = -1 # Contador de frames (loops)
    # Loop through the video frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        imga = frame
        F +=1 #A cada loop, adicionar um ao contador (para ajudar na nomeação do txt)
        name = str(video)+ '_' +str(F) #nome do arquivo txt

        #FPS
        #new_frame_time = time.time()
        #fps = 1/(new_frame_time - prev_frame_time)
        #prev_frame_time = new_frame_time
        #fps = str(int(fps))
        #print(fps)

        try: 
            if compare(imgb,imga,thresh,compar):
                frame = imgb

                classes = (results[0].boxes.cls)
                coord = (results[0].boxes.xyxy)
                write(classes, coord, name, path,writ)

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

            classes = (results[0].boxes.cls)
            coord = (results[0].boxes.xyxy)
            write(classes, coord, name, path,writ)

            # Display the annotated frame
            #cv.imshow("YOLOv8 Inference", annotated_frame)
            imgb = frame
            # Break the loop if 'q' is pressed
            #if cv.waitKey(1) & 0xFF == ord("q"):
                #break
        #Zerar o verificador
        W = 0
# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()