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
from shapely.geometry import Polygon

class lwpda():
    def __init__(self, model, threshold = 0, verbose = True, show = True):
        self.model = model
        self.verbose = verbose
        self.show = show
        self.threshold = threshold
        print(f'Threshold set to: {self.threshold}')
        return

    def isSimilar(self, actualFrame, previousFrame, threshold) -> bool:
        '''
        Compare two images (A and B) using RGB values
        Threshold assists us to know when images is similar.
        '''
        if previousFrame is None: return False 
        x = abs((previousFrame-actualFrame))
        z = ((0 <= x) & (x <= 10)).sum()
        return (z >= threshold)

    def knowVideoName(self, pathVideo:str) -> str:
        '''
        Function to determine the txt name of the video that will be saved
        '''
        for x in range(len(pathVideo)):
            if pathVideo[-x] == '/' or pathVideo[-x] == "\\":
                name = pathVideo[-x+1:][:-4]
                break
        if '/' not in pathVideo: name = pathVideo[:-4]
        return name

    def writingDetections(self, pathDir:str, pathResult:str) -> None:

        allVideos = os.listdir(pathDir)
        videoFiles = [file for file in allVideos if file.endswith((".mp4", ".avi"))]

        for video in videoFiles:
            txtName = lwpda.knowVideoName(self, video)
            boundingBoxes, masks = lwpda.calculatingDetectionsTxt(self, pathDir+'/'+video)
            lwpda.writingBoundingBoxes(self, boundingBoxes, pathResult, txtName)
            lwpda.writingMasks(self, masks, pathResult, txtName+'masks')

    def knowVideoName(self, pathVideo:str) -> str:
        '''
        Function to determine the txt name of the video that will be saved
        '''
        for x in range(len(pathVideo)):
            if pathVideo[-x] == '/' or pathVideo[-x] == "\\":
                name = pathVideo[-x+1:][:-4]
                break
        if '/' not in pathVideo: name = pathVideo[:-4]
        return name

    def calculatingDetectionsTxt(self, pathVideo:str) -> list:
        '''
        Generate each bounding box from a video
        Threshold is in a interval from 0 to 100
        Bounding Boxes are used to calculating mAP
        '''

        model = YOLO(self.model)
        cap = cv.VideoCapture(pathVideo)
        boundingBoxes = []
        masks = []
        previousFrame = None

        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        totalRGB = width * height * 3
        dinamicThreshold = totalRGB * self.threshold / 100

        # Loop through the video frames
        while True:
            # Read a frame from the video
            ret, actualFrame = cap.read()
            if not ret:
                break

            if lwpda.isSimilar(self, actualFrame, previousFrame, dinamicThreshold): 
                # Repeat bounding boxes from previous frame
                annotatedFrame = lwpda.repeatDetectionsPreviousFrame(self, actualFrame, results, boundingBoxes, masks)

            else: 
                # Send actual frame to YOLO process
                results = lwpda.processingActualFrame(self, model, actualFrame, boundingBoxes, masks)
                annotatedFrame = results[0].plot()
                previousFrame = actualFrame

            if self.show == True:
                # Display the annotated frame
                cv.imshow(f"{self.model} Inference with LWPDA", annotatedFrame)
                cv.waitKey(int(1000/(cap.get(cv.CAP_PROP_FPS))))

        return boundingBoxes, masks

    def repeatDetectionsPreviousFrame(self, actualFrame, results, boundingBoxes = None, masks = None):
        '''
        Repeating annotations from the last processed frame
        '''
        # Debug
        if self.verbose: print('Repeating detections')

        result = results[0]
        classes, coordenates = (result.boxes.cls.tolist()), (result.boxes.xyxy.tolist())
        # Saving detections to write in txt later
        if boundingBoxes is not None:
            boundingBoxes += [[classes]+[coordenates]]

        if masks is not None and result.masks is not None:
            masksCoordenates = (result.masks.xy)
            masksCoordenates = [mask.tolist() for mask in masksCoordenates]
            masks += [[classes]+[masksCoordenates]]

        # Plotting 
        annotations = results[0].plot(img = actualFrame)

        return annotations

    def processingActualFrame(self, model, actualFrame, boundingBoxes = None, masks = None) -> list:
        '''
        Processing Yolo and saving detections to write it on txts
        '''
        if self.verbose: print('Processing frame')
        # Processing the model
        results = model(actualFrame, verbose = self.verbose)
        result = results[0]

        # Saving detections to write in txt later
        if masks is not None and result.masks is not None:
            classes, masksCoordenates = (result.boxes.cls.tolist()), (result.masks.xy)
            masksCoordenates = [mask.tolist() for mask in masksCoordenates]
            masks += [[classes]+[masksCoordenates]]

        if boundingBoxes is not None:
            classes, coordenates = (result.boxes.cls.tolist()), (result.boxes.xyxy.tolist())
            boundingBoxes += [[classes]+[coordenates]]

        return results

    def writingBoundingBoxes(self, boundingBoxes: list, pathResult: str, txtName: str) -> None:
            '''
            Write txts from the bounding boxes to a especifc path
            '''
            with open(str(pathResult)+'/'+str(txtName)+'.txt','w') as file:
                for x in range(len(boundingBoxes)):
                    file.write(str(boundingBoxes[x])+'\n')
                file.close()

    def writingMasks(self, masks: list, pathResult: str, txtName: str) -> None:
            '''
            Write txts from the bounding boxes to a especifc path
            '''
            with open(str(pathResult)+'/'+str(txtName)+'.txt','w') as file:
                for (classe, mask) in masks:
                    file.write(str([classe, mask])+'\n')
                file.close()

    def timeVideos(self, pathDir:str, pathResult:str, txtName = 'videoTime') -> None:
        '''
        Generate txt with processing time of each video
        Txts are used to compare the processing time
        pathDir is where your dataset is
        pathResult is where you would like to save the results
        threshold can be set 0 to 100
        '''
        # Open directory with all videos
        allVideos = os.listdir(pathDir)
        videoFiles = [file for file in allVideos if file.endswith((".mp4", ".avi"))]
        videoTimes = []
        for video in videoFiles:
            # Process the video
            videoTimes += [lwpda.calculatingVideoProcessTime(self, pathDir+video)]
        
        lwpda.writingVideoTimes(self, videoTimes, pathResult, txtName)

    def calculatingVideoProcessTime(self, pathVideo:str) -> float:
            '''
            Calculate time process of a video using LWPDA
            '''
    
            model = YOLO(self.model)

            # The first frame showed in the video take more time to be showed
            # So, we process a random image just to initiate the camera and the model
            randomImage = "C:/Users/hugol/Pictures/dog.jpg"
            results = model(randomImage, verbose = False)
            annotatedFrame = results[0].plot()
            
            cap = cv.VideoCapture(pathVideo)
            previousFrame = None

            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            totalRGB = width * height * 3
            dinamicThreshold = totalRGB * self.threshold / 100

            # Loop through the video frames
            start = time.time()

            while True:
                # Read a frame from the video
                ret, actualFrame = cap.read()
                if not ret:
                    break

                if lwpda.isSimilar(self, actualFrame, previousFrame, dinamicThreshold): 
                    # Repeat bounding boxes from previous frame
                    annotatedFrame = lwpda.repeatDetectionsPreviousFrame(self, actualFrame, results)

                else: 
                    # Send actual frame to YOLO process
                    results = lwpda.processingActualFrame(self, model, actualFrame)
                    annotatedFrame = results[0].plot()
                    previousFrame = actualFrame
            
                if self.show == True:
                    # Display the annotated frame
                    cv.imshow(f"{self.model} Inference with LWPDA", annotatedFrame)
                    cv.waitKey(int(1000/(cap.get(cv.CAP_PROP_FPS))))

            end = time.time()
            return end - start

    def writingVideoTimes(self, videoTimes: list, pathResult: str, txtName: str) -> None:
        '''
        Write video times in txts to a especifc path
        '''
        file = open(str(pathResult)+str(txtName)+'.txt','w')
        for time in videoTimes:
            file.write(str(time)+'\n')
        
        file.close()

    def timeFrames(self, pathDir:str, pathResult:str) -> None:

        allVideos = os.listdir(pathDir)
        videoFiles = [file for file in allVideos if file.endswith((".mp4", ".avi"))]
        for video in videoFiles:
            txtName = lwpda.knowVideoName(self, video)
            frameTimes = lwpda.calculatingFramesProcessTime(self, pathDir+video)
            lwpda.writingFrameTimes(self, frameTimes, pathResult, txtName)

    def calculatingFramesProcessTime(self, pathVideo:str) -> list:
        '''
        Calculate process time of each frame from a video using LWPDA
        '''

        # Start the model
        model = YOLO(self.model)

        # The first frame showed in the video take more time to be showed (overhead)
        # So, we process a random image just to initiate the camera and the model
        randomImage = "C:/Users/hugol/Pictures/dog.jpg"
        results = model(randomImage, verbose = False)
        annotatedFrame = results[0].plot()

        cap = cv.VideoCapture(pathVideo)
        previousFrame = None

        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        totalRGB = width * height * 3
        dinamicThreshold = totalRGB * self.threshold / 100

        # Loop through the video frames
        framesTimes = []

        while True:
            start = time.time()
            # Read a frame from the video
            ret, actualFrame = cap.read()
            if not ret:
                break

            if previousFrame is not None and lwpda.isSimilar(self, actualFrame, previousFrame, dinamicThreshold): 
                # Repeat bounding boxes from previous frame
                annotatedFrame = lwpda.repeatDetectionsPreviousFrame(self, actualFrame, results)

            else: 
                # Send actual frame to YOLO process
                results = lwpda.processingActualFrame(self, model, actualFrame)
                annotatedFrame = results[0].plot()
                previousFrame = actualFrame
            
            if self.show == True:
                # Display the annotated frame
                cv.imshow(f"{self.model} Inference with LWPDA", annotatedFrame)
                cv.waitKey(int(1000/(cap.get(cv.CAP_PROP_FPS))))

            # Calculating frame process time
            end = time.time()
            framesTimes += [end-start]
        
        return framesTimes
    
    def writingFrameTimes(self, framesTimes: list, pathResult: str, txtName: str) -> None:
        '''
        Write txts from each frame time processing to a especifc path
        '''
        file = open(str(pathResult)+str(txtName)+'.txt','w')
        for x in range(len(framesTimes)):
            file.write(str(framesTimes[x]) +'\n')

    def iou(self, detectionA:list, detectionB:list) -> float: 
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

    def iouSegmentation(self, maskA: list, maskB: list) -> float:

        # Create polygons from points
        polygon1 = Polygon(maskA)
        polygon2 = Polygon(maskB)

        # Calculate intersection and union areas
        intersectionArea = polygon1.intersection(polygon2).area
        unionArea = polygon1.union(polygon2).area
        # Compute IoU
        if unionArea > 0: return intersectionArea/unionArea
        return 0

