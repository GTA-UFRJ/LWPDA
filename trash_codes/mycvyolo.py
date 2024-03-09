import time
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from txt import write
from txt import compare

new_frame_time,prev_frame_time = 0,0
# W é um verificador, caso o frame seja igual W = 1
W = 0
thresh = 500000

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
  print("Cannot open camera")
  exit()

F = -1 #Contador de frames (loops)
T = 1
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    video = "webcam"
    imga = frame
    F +=1 #A cada loop, adicionar um ao contador (para ajudar na nomeação do txt)
    name = str(video)+str(F) #nome do arquivo txt

    #FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    #print(fps)

    try: 
        if compare(imgb,imga,thresh):
            frame = imgb

            classes = (results[0].boxes.cls)
            coord = (results[0].boxes.xyxy)
            write(classes, coord, name)

            #Frames iguais, W = 1
            W = 1
    except: pass

    if success:
        if W == 1: # Repetir a marcação e o frame passado
            annotated_frame = results[0].plot()
            cv.imshow("YOLOv8 Inference", annotated_frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            classes = (results[0].boxes.cls)
            coord = (results[0].boxes.xyxy)
            write(classes, coord, name)

            # Display the annotated frame
            cv.imshow("YOLOv8 Inference", annotated_frame)
            imgb = frame
            # Break the loop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    #Zerar o verificador
    W = 0
# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()