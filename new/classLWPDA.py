'''
- Author: Hugo Antunes (antunes@gta.ufrj.br)

- From: Grupo de Telecomunicação e Automação (UFRJ/COPPE) - Rio de Janeiro - Brazil

- Professors: 
  Rodrigo de Souza Couto (rodrigo@gta.ufrj.br)
  Luís Henrique Maciel Kosmalski Costa (luish@gta.ufrj.br)
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

    def txt(pathVideo:str, pathResult:str, thresh:int):
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
        
        return

  def iou(lista1:list, lista2:list): 
        '''
        IOU is a common metric to compare the precision of the bounding boxes
        In our article, we used the original Bounding Box (YOLO precision) as Ground-Truth
        This function is based on https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        '''
    
        xA = max(lista1[0], lista2[0])
        yA = max(lista1[1], lista2[1])
        xB = min(lista1[2], lista2[2])
        yB = min(lista1[3], lista2[3])
    
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (lista1[2] - lista1[0] + 1) * (lista1[3] - lista1[1] + 1)
        boxBArea = (lista2[2] - lista2[0] + 1) * (lista2[3] - lista2[1] + 1)
    
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    
        return iou

  def timeVideos(pathDir:str, pathResult:str, thresh:int, nameTxtFile = 'testvideo'):
        '''
        Generate txt with processing time of each video
        Txts are used to compare the processing time
        pathDir is where your dataset is
        pathResult is where you would like to save the results
        thresh can be set 0 to 100
        '''
        
        # Load the YOLOv8 model
        model = YOLO('yolov8n.pt')
    
        # The first frame showed in the video take more time to be showed
        # So, we process a random image just to initiate the camera and the model
        frame = "C:/Users/hugol/Pictures/dog.jpg"
        results = model(frame)
        annotated_frame = results[0].plot()
        timeVideos =[]
        videos = os.listdir(pathDir)
        
        for x in range(len(videos)):
            pathVideo = pathDir+videos[x]
            cap = cv.VideoCapture(pathVideo)

            # Counting the time to process each video
            start = time.time()
          
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
                        #frame = imgb
                        classes = (results[0].boxes.cls.tolist())
                        coord = (results[0].boxes.xyxy.tolist())
                        
                        # Same frames flag
                        similar = 1
                except: pass

                # If the frames are similar, repeat bounding boxes
                if similar == 1:
                    vasco = Annotator(frame)
                    for r in range (len(classes)):
                        vasco.box_label(coord[r],str(int(classes[r])))
                    gama = vasco.result()
                    #cv.imshow("YOLOv8 Inference", gama)
                    #if cv.waitKey(1) & 0xFF == ord("q"):
                        #break
              
                # Else, process the new frame
                else:
                    # Run YOLOv8 inference on the frame
                    results = model(frame)
                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()
                    imgb = frame
                    # Break the loop if 'q' is pressed
                    if cv.waitKey(1) & 0xFF == ord("q"):
                        break
                # Reset flag
                similar = 0
            
            end = time.time()
            timeVideos += [end - start]

        # write in a txt the results
        if pathResul[-1] != '/': pathResult = pathResult + '/'
        
        with open(pathResult + nameTxtFile + '.txt', 'w') as arquivo:
            for x in range(len(timeVideos)):
                arquivo.write(str(timeVideos[x])+'\n')
        return














