from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from myolo import myolo as my
import os

model = YOLO('yolov8n.pt')
name = '/IC/datasets/imagenetVID/vidvrd-videos-part1/vidvrd-videos-part1/ILSVRC2015_train_00005003.mp4'
#namevid = 'ILSVRC2015_train_00005003'
z = "D:/IC/datasets/imagenetVID/videos/train/"
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
    my.write(namevid,lista,path = "D:/IC/txt_yolo/")