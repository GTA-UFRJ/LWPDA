'''
- Author: Hugo Antunes (antunes@gta.ufrj.br)

- From: Grupo de Teleinformática e Automação (UFRJ/COPPE) - Rio de Janeiro - Brazil

- Professors:
  Rodrigo de Souza Couto (rodrigo@gta.ufrj.br)
  Luís Henrique Maciel Kosmalski Costa (luish@gta.ufrj.br)
  Pedro Henrique Cruz Caminha (cruz@gta.ufrj.br)
 
- Description: This code is the class used for trying to decrease a potencially delay
when using Object Detection on real-time applications. The main idea is in the README file on GitHub.
'''

# Libraries
import time
import ast
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

    def fullExperiments(self, pathDirVideos:str, pathResult:str) -> None:
        
        for threshold in range(11):
            lwpda.createPathResults(self, pathResult)
            classe = lwpda(self.model, threshold*10, False, False)
            classe.experiments(pathDirVideos, pathResult)

    def createPathResults(self, pathResult):
        path = pathResult+f'/{self.model}/'
        if not os.path.exists(path):
            os.makedirs(path)
        for x in range(11):
            tempPath = path+f'{x}/'
            if not os.path.exists(tempPath):
                os.makedirs(tempPath)
            if not os.path.exists(tempPath+'bb/'):
                os.makedirs(tempPath+'bb/')
            if not os.path.exists(tempPath+'videos/'):
                os.makedirs(tempPath+'videos/')
            if not os.path.exists(tempPath+'frames/'):
                os.makedirs(tempPath+'frames/')

    def experiments(self, pathDir:str, pathResult:str) -> None:

        allVideos = os.listdir(pathDir)
        videoFiles = [file for file in allVideos if file.endswith((".mp4", ".avi"))]
        videoTimes = []

        for video in videoFiles:
            txtName = lwpda.knowVideoName(self, video)
            results = lwpda.calculatingDetectionsTxt(self, pathDir+'/'+video)
            boundingBoxes, masks = results[0], results[1]
            videoTime, framesTimes = results[2], results[3]
            
            lwpda.writingFrameTimes(self, framesTimes, pathResult+'frames/', txtName)
            
            videoTimes += [videoTime]
    
            lwpda.writingBoundingBoxes(self, boundingBoxes, pathResult+'bb/', txtName)

            lwpda.writingMasks(self, masks, pathResult+'bb/', txtName+'masks')
            
        lwpda.writingVideoTimes(self, videoTimes, pathResult+'videos/', txtName)

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
        startVideo = time.time()

        framesTimes = []

        # Loop through the video frames
        while True:
            startFrame = time.time()
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
                
            endFrame = time.time()
            framesTimes += [endFrame-startFrame]
            
        endVideo = time.time()
        videoTime = endVideo - startVideo
        
        return boundingBoxes, masks, videoTime, framesTimes

    def repeatDetectionsPreviousFrame(self, actualFrame, results, boundingBoxes = None, masks = None) -> list:
        '''
        Repeating annotations from the last processed frame
        '''
        # Debug
        if self.verbose: print('Repeating detections')

        result = results[0]
        classes, coordenates = ((result.boxes.cls.tolist()), result.boxes.conf.tolist()), (result.boxes.xyxy.tolist())
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
            classes, masksCoordenates = ((result.boxes.cls.tolist()), result.boxes.conf.tolist()), (result.masks.xy)
            masksCoordenates = [mask.tolist() for mask in masksCoordenates]
            masks += [[classes]+[masksCoordenates]]

        if boundingBoxes is not None:
            classes, coordenates = ((result.boxes.cls.tolist()), result.boxes.conf.tolist()), (result.boxes.xyxy.tolist())
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

           # O mAP é a média das APs de todas as classes.
        mean_ap = sum(all_aps) / len(all_aps)
        return mean_ap

    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        return None
    except (ValueError, SyntaxError):
        print(f"Erro: O conteúdo do arquivo '{file_path}' não é um dicionário Python válido.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return None

if __name__ == "__main__":
    # Este código só roda quando você executa 'python LWPDA.py'

    # Exemplo de como orquestrar um experimento
    # model_path = 'yolov8n.pt'
    # processor = LWPDA(model=model_path, threshold=30)
    # processor.fullExperiments(pathDirVideos='path/to/videos', pathResult='path/to/results')

    # Loop para calcular o mAP
    for x in range(11):
        fileName = f'{x}.txt'
        mean_average_precision = calculate_map_from_file(fileName)

        if mean_average_precision is not None:
            print("\n" + "="*40)
            print(f"O Mean Average Precision (mAP) de {fileName} final é: {mean_average_precision:.4f}")
            print("="*40)
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
        try:
            intersectionArea = polygon1.intersection(polygon2).area
            unionArea = polygon1.union(polygon2).area
        except:
            unionArea = 0
            print('Erro encontrado')
        # Compute IoU
        if unionArea > 0: return intersectionArea/unionArea
        return 0

    def loadData(self, filepath: str) -> list:
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                classes, boxes = eval(line)[0], eval(line)[1]
                data.append((classes, boxes))
        return data

    def evaluateMAP(self, dictionary: dict, groundTruths: str, predictions: str, iouThreshold=0.5) -> dict:
        groundTruths = lwpda.loadData(self, groundTruths)
        predictions = lwpda.loadData(self, predictions)

        for gt, pred in zip(groundTruths, predictions):
            gtClasses, gtBoxes = gt
            gtClasses, gtConfiability = gtClasses

            predClasses, predBoxes = pred
            predClasses, predConfiability = predClasses

            # Calculating False Positives (FP) and True Positives (TP)
            lwpda.calculatingTPFP(self, dictionary, gtClasses, gtBoxes, gtConfiability,
                                   predClasses, predBoxes, predConfiability, iouThreshold)

            # Calculating False Negatives (FN)
            lwpda.calculatingFN(self, dictionary, gtClasses, gtBoxes, gtConfiability,
                                   predClasses, predBoxes, predConfiability, iouThreshold)
        return dictionary

    def addingFalseNegative (dictionary: dict, classe: float) -> None:
        if classe not in dictionary.keys():
            dictionary.update({classe: [[None],['FN']]})
        else:
            dictionary[classe][0] += [None]
            dictionary[classe][1] += ['FN']
    
    def addingFalsePositive (dictionary: dict, classe: float, confiability: float) -> None:
        if classe not in dictionary.keys():
            dictionary.update({classe: [[confiability],['FP']]})
        else:
            dictionary[classe][0] += [confiability]
            dictionary[classe][1] += ['FP']

    def addingTruePositive (dictionary: dict, classe: float, confiability: float) -> None:
        if classe not in dictionary.keys():
            dictionary.update({classe: [[confiability],['TP']]})
        else:
            dictionary[classe][0] += [confiability]
            dictionary[classe][1] += ['TP']

    def calculatingTPFP(self, dictionary: dict, gtClasses: list, gtBoxes: list, gtConfiability: list, predClasses: list, predBoxes: list, predConfiability: list, iouThreshold: float) -> None:

        matchedGT = set()

        for predCls, predBox, predConf in zip(predClasses, predBoxes, predConfiability):
                matchFound = False
                maxIOU = 0
                classe = predCls
                confiability = predConf
                # Seeing ground-truth and what match with predictions
                for index, (gtCls, gtBox, gtConf) in enumerate(zip(gtClasses, gtBoxes, gtConfiability)):
                    if index in matchedGT: # Index already matched
                        continue
                    if predCls == gtCls and lwpda.iou(self, predBox, gtBox) >= iouThreshold and lwpda.iou(self, predBox, gtBox) > maxIOU:
                        idx = index
                        matchFound = True
                        maxIOU = lwpda.iou(self, predBox, gtBox)
                        
                if not matchFound:
                    lwpda.addingFalsePositive(dictionary, classe, confiability)

                else:
                    lwpda.addingTruePositive(dictionary, classe, confiability)
                    matchedGT.add(idx)

    def calculatingFN(self, dictionary: dict, gtClasses: list, gtBoxes: list, gtConfiability: list, predClasses: list, predBoxes: list, predConfiability: list, iouThreshold: float) -> None:
        matchedPred = set()

        for gtCls, gtBox, gtConf in zip(gtClasses, gtBoxes, gtConfiability):
                matchFound = False
                maxIOU = 0
                classe = gtCls
                # Seeing predictions and what match with predictions
                for index, (predCls, predBox, predConf) in enumerate(zip(predClasses, predBoxes, predConfiability)):
                    if index in matchedPred: # Index already matched
                        continue
                    if predCls == gtCls and lwpda.iou(self, predBox, gtBox) >= iouThreshold and lwpda.iou(self, predBox, gtBox) > maxIOU:
                        idx = index
                        matchFound = True
                        maxIOU = lwpda.iou(self, predBox, gtBox)
                        
                if not matchFound:
                    lwpda.addingFalseNegative(dictionary, classe)
                else:
                    matchedPred.add(idx)

    def dictMAP(self, groundTruthDirectory: str, predictionsDirectory: str, iouThreshold: float) -> dict:
        '''
        Create a dictionary that help to managing the results
        Dictionary for a directory
        This function is going to be used in mAP function
        '''
        groundTruthlist = os.listdir(groundTruthDirectory)
        predictionsList = os.listdir(predictionsDirectory)
        if len(predictionsList) != len(groundTruthlist): raise "Exception"
        dictionary = {}
        n = 0
        for video in groundTruthlist:
            if 'mask' in video:
                if self.verbose: print('Processing video:',n)
                lwpda.evaluateMAPSegmentation(self, dictionary, groundTruthDirectory+video, predictionsDirectory+video, iouThreshold)
                n+=1

        return dictionary

    def evaluateMAPSegmentation(self, dictionary: dict, groundTruths: str, predictions: str, iouThreshold=0.5) -> dict:
        '''
        Create (or update) a dictionary for each video in directory
        '''
        groundTruths = lwpda.loadData(self, groundTruths)
        predictions = lwpda.loadData(self, predictions)

        for gt, pred in zip(groundTruths, predictions):
            gtClasses, gtBoxes = gt
            gtClasses, gtConfiability = gtClasses

            predClasses, predBoxes = pred
            predClasses, predConfiability = predClasses

            # Calculating False Positives (FP) and True Positives (TP)
            lwpda.calculatingTPFPSegmentation(self, dictionary, gtClasses, gtBoxes, gtConfiability,
                                   predClasses, predBoxes, predConfiability, iouThreshold)

            # Calculating False Negatives (FN)
            lwpda.calculatingFNSegmentation(self, dictionary, gtClasses, gtBoxes, gtConfiability,
                                   predClasses, predBoxes, predConfiability, iouThreshold)
        return dictionary

    def calculatingTPFPSegmentation(self, dictionary: dict, gtClasses: list, gtBoxes: list, gtConfiability: list, predClasses: list, predBoxes: list, predConfiability: list, iouThreshold: float) -> None:

        matchedGT = set()

        for predCls, predBox, predConf in zip(predClasses, predBoxes, predConfiability):
                matchFound = False
                maxIOU = 0
                classe = predCls
                confiability = predConf
                # Seeing ground-truth and what match with predictions
                for index, (gtCls, gtBox, gtConf) in enumerate(zip(gtClasses, gtBoxes, gtConfiability)):
                    if index in matchedGT: # Index already matched
                        continue
                    if predCls == gtCls and lwpda.iouSegmentation(self, predBox, gtBox) >= iouThreshold and lwpda.iouSegmentation(self, predBox, gtBox) > maxIOU:
                        idx = index
                        matchFound = True
                        maxIOU = lwpda.iouSegmentation(self, predBox, gtBox)
                        
                if not matchFound:
                    lwpda.addingFalsePositive(dictionary, classe, confiability)

                else:
                    lwpda.addingTruePositive(dictionary, classe, confiability)
                    matchedGT.add(idx)

    def calculatingFNSegmentation(self, dictionary: dict, gtClasses: list, gtBoxes: list, gtConfiability: list, predClasses: list, predBoxes: list, predConfiability: list, iouThreshold: float) -> None:
        matchedPred = set()

        for gtCls, gtBox, gtConf in zip(gtClasses, gtBoxes, gtConfiability):
                matchFound = False
                maxIOU = 0
                classe = gtCls
                # Seeing predictions and what match with predictions
                for index, (predCls, predBox, predConf) in enumerate(zip(predClasses, predBoxes, predConfiability)):
                    if index in matchedPred: # Index already matched
                        continue
                    if predCls == gtCls and lwpda.iouSegmentation(self, predBox, gtBox) >= iouThreshold and lwpda.iouSegmentation(self, predBox, gtBox) > maxIOU:
                        idx = index
                        matchFound = True
                        maxIOU = lwpda.iouSegmentation(self, predBox, gtBox)
                        
                if not matchFound:
                    lwpda.addingFalseNegative(dictionary, classe)
                else:
                    matchedPred.add(idx)

def calculate_average_precision(confidences, labels):
    """
    Calcula o Average Precision (AP) para uma única classe.

    Args:
        confidences (list): Lista de scores de confiança.
        labels (list): Lista de rótulos ('TP', 'FP', 'FN').

    Returns:
        float: O valor do Average Precision (AP).
    """
    # Combina confianças e rótulos, e ordena em ordem decrescente pela confiança.
    # 'None' é tratado como a menor confiança possível (-1.0) para ir para o fim da lista.
    sorted_detections = sorted(
        [(c if c is not None else -1.0, l) for c, l in zip(confidences, labels)],
        key=lambda x: x[0],
        reverse=True
    )

    # O número total de amostras positivas é a soma de Verdadeiros Positivos (TP)
    # e Falsos Negativos (FN).
    total_positives = labels.count('TP') + labels.count('FN')

    # Se não houver amostras positivas na verdade fundamental (ground truth), a AP é 0.
    if total_positives == 0:
        return 0.0

    acc_tp = 0  # Acumulador de Verdadeiros Positivos
    acc_fp = 0  # Acumulador de Falsos Positivos
    precision_sum = 0.0

    # Itera sobre as detecções ordenadas para calcular a soma das precisões nos TPs.
    for _, label in sorted_detections:
        if label == 'TP':
            acc_tp += 1
            # Calcula a precisão no ponto atual: TP / (TP + FP)
            precision = acc_tp / (acc_tp + acc_fp)
            # Soma a precisão a cada TP encontrado.
            precision_sum += precision
        elif label == 'FP':
            acc_fp += 1
        # FNs não são processados no loop, apenas no cálculo inicial do total_positives.

    # A AP é a soma das precisões dividida pelo número total de amostras positivas.
    return precision_sum / total_positives

def calculate_map_from_file(file_path):
    """
    Lê um arquivo de dicionário e calcula o mean Average Precision (mAP).

    Args:
        file_path (str): O caminho para o arquivo .txt contendo o dicionário.

    Returns:
        float: O valor do mAP, ou None em caso de erro.
    """
    try:
        # Abre e lê o conteúdo completo do arquivo.
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Converte a string lida do arquivo em um dicionário Python.
        data_dict = ast.literal_eval(content)
        
        all_aps = []
        #print("Calculando Average Precision (AP) para cada classe...")
        
        # Itera sobre cada classe no dicionário.
        for class_id, values in data_dict.items():
            if len(values) >= 2:
                confidences, labels = values[0], values[1]
                # Calcula a AP para a classe atual.
                ap = calculate_average_precision(confidences, labels)
                all_aps.append(ap)
                #print(f"  - AP para a Classe {class_id}: {ap:.4f}")
            
        # Se nenhuma classe foi encontrada, o mAP é 0.
        if not all_aps:
            return 0.0
            
        # O mAP é a média das APs de todas as classes.
        mean_ap = sum(all_aps) / len(all_aps)
        return mean_ap

    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        return None
    except (ValueError, SyntaxError):
        print(f"Erro: O conteúdo do arquivo '{file_path}' não é um dicionário Python válido.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return None

if __name__ == "__main__":
    # Este código só roda quando você executa 'python LWPDA.py'
    
    # Exemplo de como orquestrar um experimento
    # model_path = 'yolov8n.pt'
    # processor = LWPDA(model=model_path, threshold=30)
    # processor.fullExperiments(pathDirVideos='path/to/videos', pathResult='path/to/results')

    # Loop para calcular o mAP
    for x in range(11):
        fileName = f'{x}.txt'
        mean_average_precision = calculate_map_from_file(fileName)

        if mean_average_precision is not None:
            print("\n" + "="*40)
            print(f"O Mean Average Precision (mAP) de {fileName} final é: {mean_average_precision:.4f}")
            print("="*40)
