import time
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from txt import write

new_frame_time,prev_frame_time = 0,0
# W é um verificador, caso o frame seja igual W = 1
W = 0
thresh = 500000
def compare(imgb,imga,thresh):
  #comparar duas imagens, imgb (before), imga(after). thresh é o limiar que definiremos
  #limiar é o valor que obtivemos expirimentalmente como adequado para decidir se uma imagem é ou não similar
  #além disso, a similaridade de pixels também deveria ser definida (por mim)
  #retorna True or False
  #comp,width, pix = len(imga[0]),len(imga), width*comp
  x = abs((imgb-imga))
  z = ((0 <= x) & (x <= 10)).sum()
  #print(z)
  if z >= thresh:
    return True
  return False

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

results = model.predict('/media/antunes/Professor Antunes/IC/datasets/imagenetVID/vidvrd-videos-part1/vidvrd-videos-part1/ILSVRC2015_train_00005003.mp4', save=True, imgsz=320, conf=0.5)