'''
- Author: Hugo Antunes (antunes@gta.ufrj.br)

- From: Grupo de Telecomunicação e Automação (UFRJ/COPPE) - Rio de Janeiro - Brazil

- Professors:
  Rodrigo de Souza Couto (rodrigo@gta.ufrj.br)
  Luís Henrique Maciel Kosmalski Costa (luish@gta.ufrj.br)
  Pedro Henrique Cruz Caminha (cruz@gta.ufrj.br)
 
- Description: This code is the class used for trying to decrease a potencially delay
when using Object Detection on real-time applications. The main idea is in the README file on GitHub.
'''

# Libraries
import time
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import os
from ultralytics.utils.plotting import Annotator

class lwpda():

    def isSimilar(actualFrame, previousFrame, threshold:int) -> bool:
        '''
        Compare two images (A and B) using RGB values
        Threshold assists us to know when images is similar.
        '''
        if previousFrame is None: return False 
        x = abs((previousFrame-actualFrame))
        z = ((0 <= x) & (x <= 10)).sum()
        if z >= threshold:
            actualFrame = previousFrame
            return True
        return False

    def knowVideoName(pathVideo:str) -> str:
        '''
        Function to determine the txt name of the video that will be saved
        '''
        for x in range(len(pathVideo)):
            if pathVideo[-x] == '/' or pathVideo[-x] == "\\":
                name = pathVideo[-x+1:][:-4]
                break
        if '/' not in pathVideo: name = pathVideo[:-4]
        return name

    def repeatDetectionsPreviousFrame(actualFrame, results):
        '''
        Repeating annotations from the last processed frame
        '''

        # Saving detections to write in txt later
        classes, coordenates = (results[0].boxes.cls.tolist()), (results[0].boxes.xyxy.tolist())
        boundingBoxes += [[classes]+[coordenates]]

        # Repeating annotations
        annotations = Annotator(actualFrame)
        for r in range (len(classes)):
            annotations.box_label(coordenates[r],str(int(classes[r])))

        return annotations.result()

    def writingBoundingBoxes(boundingBoxes: list, pathResult: str, txtName: str) -> None:
        '''
        Write txts from the bounding boxes to a especifc path
        '''

        for x in range(len(boundingBoxes)):
            classe = boundingBoxes[x][0]
            coord = boundingBoxes[x][1]
            file = open(str(pathResult)+str(txtName)+'.txt','a')
            file.write('[')
            for y in range(len(coord)):
                file.write('['+ str(classe[y])+ ','+ str(coord[y])+ ']' + ',')
            file.write(']'+'\n')

    def processingActualFrame(model, actualFrame, previousFrame, annotatedFrame):
        '''
        Processing Yolo and saving detections to write it on txts
        '''

        # Processing the model
        results = model(actualFrame)
        annotatedFrame = results[0].plot()

        # Saving detections to write in txt later
        classes, coordenates = (results[0].boxes.cls.tolist()), (results[0].boxes.xyxy.tolist())
        boundingBoxes += [[classes]+[coordenates]]
        previousFrame = actualFrame

        return results

    def calculatingDetectionsTxt(pathVideo:str, threshold:int) -> list:
        '''
        Generate each bounding box from a video
        Threshold is in a interval from 0 to 100
        Bounding Boxes are used to calculating mAP
        '''

        model = YOLO('yolov8n.pt')
        cap = cv.VideoCapture(pathVideo)
        boundingBoxes = []
        previousFrame = None

        # Loop through the video frames
        while True:
            # Read a frame from the video
            ret, actualFrame = cap.read()
            if not ret:
                break

            totalRGB = len(actualFrame)*len(actualFrame[0])*3
            threshold = totalRGB*threshold/100

            if previousFrame is not None and lwpda.isSimilar(previousFrame, actualFrame, threshold): 
                # Repeat bounding boxes from previous frame
                annotatedFrame = lwpda.repeatDetectionsPreviousFrame(actualFrame, results)

            else: 
                # Send actual frame to YOLO process
                results = lwpda.processingActualFrame(model, actualFrame, previousFrame, annotatedFrame)

            # Display the annotated frame
            cv.imshow("YOLOv8 Inference with LWPDA", annotatedFrame)

        return boundingBoxes

    def writingDetections(pathDir:str, pathResult:str , threshold:int) -> None:
        allVideos = os.listdir(pathDir)
        for video in allVideos:
            txtName = lwpda.knowVideoName(video)
            boundingBoxes = lwpda.calculatingDetectionsTxt(pathDir+video, threshold)
            lwpda.writingBoundingBoxes(boundingBoxes, pathResult, txtName)

    def iou(detectionA:list, detectionB:list) -> float: 
        '''
        IOU is a common metric to compare the precision of the bounding boxes
        In our article, we used the original Bounding Box (YOLO precision) as Ground-Truth
        This function is based on https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        '''
    
        xA = max(detectionA[0], detectionB[0])
        yA = max(detectionA[1], detectionB[1])
        xB = min(detectionA[2], detectionB[2])
        yB = min(detectionA[3], detectionB[3])
    
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (detectionA[2] - detectionA[0] + 1) * (detectionA[3] - detectionA[1] + 1)
        boxBArea = (detectionB[2] - detectionB[0] + 1) * (detectionB[3] - detectionB[1] + 1)
    
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    
        return iou

    def calculatingVideoProcessTime(pathVideo:str, threshold:int) -> float:
            '''
            Calculate time process of a video using LWPDA
            '''
    
            model = YOLO('yolov8n.pt')

            # The first frame showed in the video take more time to be showed
            # So, we process a random image just to initiate the camera and the model
            randomImage = "C:/Users/hugol/Pictures/dog.jpg"
            results = model(randomImage)
            annotatedFrame = results[0].plot()

            cap = cv.VideoCapture(pathVideo)
            previousFrame = None

            # Loop through the video frames
            start = time.time()

            while True:
                # Read a frame from the video
                ret, actualFrame = cap.read()
                if not ret:
                    break

                totalRGB = len(actualFrame)*len(actualFrame[0])*3
                threshold = totalRGB*threshold/100

                if previousFrame is not None and lwpda.isSimilar(previousFrame, actualFrame, threshold): 
                    # Repeat bounding boxes from previous frame
                    annotatedFrame = lwpda.repeatDetectionsPreviousFrame(actualFrame, results)

                else: 
                    # Send actual frame to YOLO process
                    results = lwpda.processingActualFrame(model, actualFrame, previousFrame, annotatedFrame)

            end = time.time()
            return end - start

    def writingVideoTimes(videoTimes: list, pathResult: str, txtName: str) -> None:
        '''
        Write video times in txts to a especifc path
        '''
        file = open(str(pathResult)+str(txtName)+'.txt','a')
        for time in range(len(videoTimes)):
            file.write(time+'\n')
        
        file.close()

    def timeVideos(pathDir:str, pathResult:str, threshold:int, txtName = 'videoTime') -> None:
        '''
        Generate txt with processing time of each video
        Txts are used to compare the processing time
        pathDir is where your dataset is
        pathResult is where you would like to save the results
        threshold can be set 0 to 100
        '''
        # Open directory with all videos
        allVideos = os.listdir(pathDir)
        videoTimes = []
        for video in allVideos:
            txtName = lwpda.knowVideoName(video)

            # Process the video
            videoTimes += lwpda.calculatingVideoProcessTime(video, threshold)
        
        lwpda.writeVideoTimes(videoTimes, pathResult, txtName)

    def CalculatingFramesProcessTime(pathVideo:str, threshold:int) -> list:
        '''
        Calculate process time of each frame from a video using LWPDA
        '''

        # Start the model
        model = YOLO('yolov8n.pt')

        # The first frame showed in the video take more time to be showed (overhead)
        # So, we process a random image just to initiate the camera and the model
        randomImage = "C:/Users/hugol/Pictures/dog.jpg"
        results = model(randomImage)
        annotatedFrame = results[0].plot()

        cap = cv.VideoCapture(pathVideo)
        previousFrame = None

        # Loop through the video frames
        framesTimes = []

        while True:
            start = time.time()
            # Read a frame from the video
            ret, actualFrame = cap.read()
            if not ret:
                break

            totalRGB = len(actualFrame)*len(actualFrame[0])*3
            threshold = totalRGB*threshold/100

            if previousFrame is not None and lwpda.isSimilar(previousFrame, actualFrame, threshold): 
                # Repeat bounding boxes from previous frame
                annotatedFrame = lwpda.repeatDetectionsPreviousFrame(actualFrame, results)

            else: 
                # Send actual frame to YOLO process
                results = lwpda.processingActualFrame(model, actualFrame, previousFrame, annotatedFrame)

            # Calculating frame process time
            end = time.time()
            framesTimes += [end-start]
        
        return framesTimes
    
    def writingFrameTimes(framesTimes: list, pathResult: str, txtName: str) -> None:
        '''
        Write txts from each frame time processing to a especifc path
        '''

        for x in range(len(framesTimes)):
            file = open(str(pathResult)+str(txtName)+'.txt','a')
            file.write(f'[{framesTimes[x]}]\n')

    def timeFrames(pathDir:str, pathResult:str , threshold:int) -> None:
        allVideos = os.listdir(pathDir)
        for video in allVideos:
            txtName = lwpda.knowVideoName(video)
            frameTimes = lwpda.calculatingDetectionsTxt(pathDir+video, threshold)
            lwpda.writingFrameTimes(frameTimes, pathResult, txtName)




