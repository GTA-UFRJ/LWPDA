import cv2 as cv
import numpy as np
from ultralytics import solutions

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

# Inicializar captura de vídeo
cap = cv.VideoCapture("D:/Arquivos/Desktop/LWPDA/videoTest.mp4")
assert cap.isOpened(), "Erro ao ler o arquivo de vídeo"

# Configuração do escritor de vídeo
w, h, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))
video_writer = cv.VideoWriter("isegment_output.avi", cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Inicializar segmentação
isegment = solutions.InstanceSegmentation(
    show=True,
    model="yolov8n-seg.pt",
    verbose=False
)

# Inicializa variáveis
count = 0
imgB = None  # Variável para armazenar o último frame
last_annotated_frame = None  # Última marcação válida

# Processamento do vídeo
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Processamento do vídeo concluído ou quadro vazio.")
        break

    # Redimensionar quadro
    width, height, totalPixel = 640, 480, 640 * 480
    resized_im0 = cv.resize(im0, (width, height))

    if imgB is not None:
        if compare(resized_im0, imgB, thresh=totalPixel * 3 * 0.45):
            imgB = resized_im0
            count += 1
            print(f"Quadros semelhantes detectados: {count}")
        else: 
            results = isegment(resized_im0)
            
            
            
    imgB = resized_im0  # Atualiza o frame anterior para comparação

cap.release()
video_writer.release()
cv.destroyAllWindows()
