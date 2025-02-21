'''
- Author: Hugo Antunes (antunes@gta.ufrj.br)

- From: Grupo de Telecomunicação e Automação (UFRJ/COPPE) - Rio de Janeiro - Brazil

- Professors: Rodrigo de Souza Couto (rodrigo@gta.ufrj.br) and 
  Pedro Henrique Cruz Caminha (cruz@gta.ufrj.br)

- Description: This code is the class used for trying to decrease a potencially delay
when using YOLO on real-time applications. The main idea is in the README file on GitHub.
'''

# Libraries
import time
import numpy as np
import cv2
from ultralytics import YOLO
import os

class lwpda():
    def compare(imgA,imgB,thresh:int):
        '''
        Compare two images (A and B) using RGB values
        thresh is the threshold which help to know when images is similar.
        '''
        x = abs((imgB-imgA))
        z = ((0 <= x) & (x <= 10)).sum()
        if z >= thresh:
            return True
        return False

    def txt(pathVideo:str,pathResult,thresh:int):
        '''
        Generate txt from each bounding box from a video and save them
        Txts are used to compare the precision
        '''

        model = YOLO('yolov8n.pt')
        cap = cv2.VideoCapture(pathVideo)
        listBB = []
        
        # Discover video name to use in txt result
        for x in range(len(pathVideo)):
            if pathVideo[-x] == '/' or pathVideo[-x] == "\\":
                nameVideo = pathVideo[-x+1:][:-4]
                break
        if '/' not in pathVideo: nameVideo = pathVideo[:-4]

        # Loop through the video frames
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            totalRGB = len(frame)*len(frame[0])*3
            thresh = totalRGB*thresh

            imga = frame

            try: 
                if lwpda.compare(imgb,imga,thresh):
                    frame = imgb

                    classes = (results[0].boxes.cls.tolist())
                    coord = (results[0].boxes.xyxy.tolist())
                    listBB += [[classes]+[coord]]

                    # Same frames flag
                    similar = 1
            except: pass

            if similar == 1: # Repeat bounding boxes from previous frame
                annotated_frame = results[0].plot()
                #cv.imshow("YOLOv8 Inference", annotated_frame)
                #if cv.waitKey(1) & 0xFF == ord("q"):
                    #break
            else: # Common processing (YOLO detection)
                # Run YOLOv8 inference on the frame
                results = model(frame)
                annotated_frame = results[0].plot()
                classes = (results[0].boxes.cls.tolist())
                coord = (results[0].boxes.xyxy.tolist())
                listBB += [[classes]+[coord]]

                # Display the annotated frame
                #cv.imshow("YOLOv8 Inference", annotated_frame)
                imgb = frame
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Reset flag
            similar = 0

        for x in range(len(listBB)):
            classe = listBB[x][0]
            coord = listBB[x][1]
            file = open(str(pathResult)+str(nameVideo)+'.txt','a')
            file.write('[')
            for y in range(len(coord)):
                file.write('['+ str(classe[y])+ ','+ str(coord[y])+ ']' + ',')
            file.write(']'+'\n')