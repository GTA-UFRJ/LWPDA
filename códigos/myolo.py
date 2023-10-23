import math
import warnings
from pathlib import Path

import re
import time
import cv2 as cv
import os
from ultralytics import YOLO

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings

class myolo():
    def write(classe,coord, name, path = "",A = 1):
            # Exemplo:
            # classes = (results[0].boxes.cls)
            # coord = (results[0].boxes.xyxy)
            # write(classes, coord, name, path,writ)

        if A == 0: return # A = 0, desliga a criação dos txts
        coord = coord
        classe = classe
        file = open(str(path)+str(name)+'.txt','w')
        for x in range(len(coord)):
            file.write(str(classe[x])+ ' '+ str(coord[x])+ ' \n')
        file.close()
        with open(str(path)+str(name)+'.txt', 'r+') as f:
            content = f.read()
            f.seek(0)
            f.truncate()
            f.write(content.replace('tensor(','').replace('.)','    ').replace('])','').replace('[','').replace(',',''))

    def compare(imgb,imga,thresh, A =1):

        if A == 0: return # A = 0 desliga a comparação

        # comparar duas imagens, imgb (before), imga(after). thresh é o limiar que definiremos
        # limiar é o valor que obtivemos expirimentalmente como adequado para decidir se uma imagem é ou não similar
        # além disso, a similaridade de pixels também deveria ser definida (por mim)
        # retorna True or False
        # comp, width, pix = len(imga[0]),len(imga), width*comp

        x = abs((imgb-imga))
        z = ((0 <= x) & (x <= 10)).sum()
        #print(z)
        if z >= thresh:
            return True
        return False
    #(os.listdir("D:/IC/datasets/imagenetVID/videos/train/"))
    def box_iou(file1, file2): # From pyimagesearch.com (not me)
        # Calculate intersection-over-union (IoU) of boxes.
        # -1 -> False negative
        # -2 -> False positive
        # File A -> Ground Truth
        result_iou = []
        Falses = []
        fileA = open(file1, 'r')
        fileB = open(file1, 'r')
        strinA = fileA.readlines()
        strinB = fileB.readlines()
        classA = []
        classB = []
        print(strinA,strinB)
        for x in range(len(strinA)): # Modifying strings from text file
            strinA[x] = ' '.join(strinA[x].split())
            print(strinA)
            for y in range(len(strinA[x])):
                if strinA[x][y] == ' ':
                    classA += [strinA[x][:y]]
                    strinA[x] = strinA[x][:y] + ',' + strinA[x][y+1:]
            strinA[x] = strinA[x][:-3]
            for z in range(len(strinA[x])):
                if strinA[x][z] == ',':
                    print(strinA[x][:z])
                    strinA[x] += [int(strinA[x][:z])]

            print(strinA)
        x,y,z = 0,0,0

        for x in range(len(strinB)): # Modifying strings from text file
            strinB[x] = ' '.join(strinB[x].split())
            print(strinB)
            for y in range(len(strinB[x])):
                if strinB[x][y] == ' ':
                    classB += [strinB[x][:y]]
                    strinB[x] = strinB[x][:y] + ',' + strinB[x][y+1:]
            strinB[x] = strinB[x][:-3]
            print(strinB)

        if len(classA) >= len(classB): # Calcular os falsos positivos/negativos
            try:
                for a in range(len(classA)):
                    if classA[a] != classB[a]:
                        Falses+=-1
            except: pass
        else:
            try:
                for b in range(len(classA)):
                    if classA[b] != classB[b]:
                        Falses+=-2
            except: pass

        x,y,z = 0,0,0

        for x in range(len(strinA)):

            if classA[x] == classB[x]:
                xA = max(strinA[x][0], strinB[x][0])
                yA = max(strinA[x][1], strinB[x][1])
                xB = min(strinA[x][2], strinB[x][2])
                yB = min(strinA[x][3], strinB[x][3])
                # compute the area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (strinA[x][2] - strinA[x][0] + 1) * (strinA[x][3] - strinA[x][1] + 1)
                boxBArea = (strinB[x][2] - strinB[x][0] + 1) * (strinB[x][3] - strinB[x][1] + 1)
                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
                result_iou += [iou]
                print(result_iou)
            else: pass

        # return the intersection over union value
        return (result_iou, Falses)

    def auto_iou(path1:str, path2:str, pathtosave:str): 
        # path1 - ground truth; path2 - test -> both strings
        # pathtosave -> where you want to save the txts data
        # return a list with iou for each txt file
        # path have to be the path to the file with all txt files
        test = (os.listdir(path2))
        gt = (os.listdir(path1))

        if len(gt) != len(test): raise IndexError # number of txts must to be equals
        for x in range(len(gt)):
            print(len(gt))
            file1 = path1+'/'+str(gt[x])
            file2 = path2+'/'+str(test[x])
            iou = myolo.box_iou(file1,file2)
            file = open(str(pathtosave)+str(gt[x]),'w') # '       ' '     '
        return






#VID = (os.listdir("D:/IC/datasets/imagenetVID/videos/train/"))
#file1 = 'C:/Users/hugol/Desktop/IC/GTY.testes - 500000/train-myolo/ILSVRC2015_train_00005003.mp4_0.txt'
#file2 = 'C:/Users/hugol/Desktop/IC/GTY.testes - 500000/train-myolo/ILSVRC2015_train_00005003.mp4_0.txt'
#myolo.box_iou(file1,file2)

#file2 = open('C:/Users/hugol/Desktop/IC/GTY.testes - 500000/train-myolo/ILSVRC2015_train_00005003.mp4_0.txt','r')
#strin = file2.readlines()
#print(strin)
#print(len(strin))
#if type(strin[0][1]) == int: print('yes')
#print(type(eval(strin[0][1])),int(strin[0][0]))
#if type(eval(strin[0][0])) == int: print('yes')
#strin[0] = strin[0][1:]
#print(strin)
#myolo.auto_iou('C:/Users/hugol/Desktop/IC/GTY.testes - 500000/train-yolo','C:/Users/hugol/Desktop/IC/GTY.testes - 500000/train-myolo', 'C:/Users/hugol/Desktop/IC/GTY.testes - 500000/iou')
#print(os.listdir('D:\IC\códigos'))
x = '9.1177e+02'
print(float(x))