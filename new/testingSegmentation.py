import cv2
import numpy as np
from ultralytics import YOLO

# Carregar modelo de segmentação do YOLO
model = YOLO("yolov8n-seg.pt")  # Certifique-se de ter o modelo correto
media = 'C:/Users/hugol/Desktop/Dataset ImageNetVID/ILSVRC2015_train_00005003.mp4'
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

# Captura de vídeo
cap = cv2.VideoCapture(media)
previousFrame = None

while cap.isOpened():
    ret, actualFrame = cap.read()
    if not ret:
        break

    if previousFrame is None:
        totalPixels = len(actualFrame[0])*len(actualFrame)*3
        threshold = totalPixels*1

    if isSimilar(actualFrame, previousFrame, threshold):
        print('parecidos')
        # Repeat segmentation from previous frame
        result = results[0]
        if result.masks == None: break
        masks = result.masks.xy  # Mask coordenates
        for mask in masks:
            mask = np.array(mask, dtype=np.int32)
            cv2.polylines(actualFrame, [mask], True, (0, 255, 0), 2)

    else:
        # Process YOLO Segmentation
        print('processando')
        previousFrame = actualFrame
        results = model(actualFrame,verbose=False)
        # Desenhar resultados na imagem
        if results[0].masks == None: break
        masks = results[0].masks.xy  # Coordenadas das máscaras
        im = results[0].plot(boxes = False)
        # im.show()
        # for mask in masks:
        #     mask = np.array(mask, dtype=np.int32)
        #     cv2.polylines(actualFrame, [mask], True, (0, 255, 0), 2)  # Contorno das máscaras

    # Show result
    cv2.imshow("Segmentação YOLO", im)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()