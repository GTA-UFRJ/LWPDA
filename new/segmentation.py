import cv2
import numpy as np
from ultralytics import YOLO

# Carregar modelo de segmentação do YOLO
model = YOLO("yolov8n-seg.pt")  # Certifique-se de ter o modelo correto

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
cap = cv2.VideoCapture("car.mp4")
previousFrame = None

while cap.isOpened():
    ret, actualFrame = cap.read()
    if not ret:
        break

    if previousFrame is None:
        totalPixels = len(actualFrame[0])*len(actualFrame)*3
        threshold = totalPixels*0.3

    if isSimilar(actualFrame, previousFrame, threshold):
        print('parecidos')
        # Repeat segmentation from previous frame
        for result in results:
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
        print(type(results))
        # Desenhar resultados na imagem
        for result in results:
            if result.masks == None: break
            masks = result.masks.xy  # Coordenadas das máscaras
            for mask in masks:
                mask = np.array(mask, dtype=np.int32)
                cv2.polylines(actualFrame, [mask], True, (0, 255, 0), 2)  # Contorno das máscaras

    # Show result
    cv2.imshow("Segmentação YOLO", actualFrame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()