# This code is responsible to writing the coordenates from each video.
# Just to automatize the process, we create one code to create the txt file
# that contains each detection from each frame.
# The txt is in this form:
# [nºclass[coord.],nºclass[coord.]
# Each line is from one frame, so line 1 -> frame 1, line 2 -> frame 2 etc.

from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from myolo import myolo as my
import os

model = YOLO('yolov8n.pt')
#name = '/IC/datasets/imagenetVID/vidvrd-videos-part1/vidvrd-videos-part1/ILSVRC2015_train_00005003.mp4'
#namevid = 'ILSVRC2015_train_00005003'
z = "C:/Users/hugol/Desktop/IC/datasets/train/"
x = os.listdir(z) # lista com os nomes dos arquivos
for video in range(len(x)):
    lista = []
    name = z+x[video]
    results = model(name, stream = True)  # results list
#for r in results:
    #im_array = r.plot()  # plot a BGR numpy array of predictions
    #im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #im.show()  # show image

    #im.save('results.jpg')  # save image
    namevid = x[video]
    for r in results:
        classes = r.boxes.cls.tolist()
        coord = r.boxes.xyxy.tolist()
        lista+=[[classes]+[coord]]
        #print(classes,type(classes),coord,type(coord))
        #print(len(lista))
    print(lista)
    my.write(namevid,lista,path = "C:/Users/hugol/Desktop/IC/datasets/testando/")