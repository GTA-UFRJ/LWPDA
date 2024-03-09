#Libs
import os
import numpy as np
import math
import re
from myolo import myolo as my
import matplotlib.pyplot as plt

path1 = "C:/Users/hugol/Desktop/IC/txt_yolo/" # Ground Truth
path2 = "C:/Users/hugol/Desktop/IC/txt_myolo_500000/" # Tests

#txtyolo = "C:/Users/hugol/Desktop/IC/txt_yolo/ILSVRC2015_train_00005003.mp4.txt"
#txtmyolo = "C:/Users/hugol/Desktop/IC/txt_myolo_500000/ILSVRC2015_train_00005003.mp4.txt"
def dicts(txtyolo:str, txtmyolo:str, mAPIOU, on=1): 
    # This function will take all the IOUs and False Positives and False Negatives
    # The input is the txts files with the coordinates
    # Auxiliar function to the mAP function (below)
    if on == 0: return
    classes = {}
    classesIOU = {}
    fileA = open(txtyolo, 'r')
    fileB = open(txtmyolo, 'r')
    x = fileA.readlines()
    y = fileB.readlines()
    parciais = []
    for lines in range(len(x)): #lines = frames
        if x[lines] and y[lines] != '[]\n':
            a = x[lines][:-4]+']]'
            b = y[lines][:-4]+']]'
        a = x[lines][:-1]
        b = y[lines][:-1]
        a = eval(a)
        b = eval(b)
        for obj in range(len(a)):
            v = int(a[obj][0])
            classesIOU['classe'+str(v)] = []
            classes['classe'+str(v)] = []

        for obj in range(len(b)):
            v = int(b[obj][0])
            classesIOU['classe'+str(v)] = []
            classes['classe'+str(v)] = []

    lines = 0
    for lines in range(len(x)): #lines = frames
        a = eval(x[lines])
        b = eval(y[lines])
        A = a.copy()
        B = b.copy()
        for classeYOLO in range(len(A)):
            parciais = []
            classeA = int(a[classeYOLO][0])
            name = 'classe'+str(classeA)
            for classeMYOLO in range(len(B)):
                classeB = int(B[classeMYOLO][0])
                coordA = A[classeYOLO][1]
                coordB = B[classeMYOLO][1]
                if classeA == classeB:
                    iou = my.iou(coordA,coordB)
                    parciais += [iou]
            if parciais != []:
                classesIOU[name] += [max(parciais)]
                indexB = parciais.index(max(parciais))
                b = B[:indexB]+B[indexB+1:]
            else: 
                classesIOU[name]+=['FN']

    lines = 0
    classeMYOLO = 0
    classeYOLO = 0
    for lines in range(len(x)): #lines = frames
        a = eval(x[lines])
        b = eval(y[lines])
        A = a.copy()
        B = b.copy()
        for classeMYOLO in range(len(B)):
            parciais = []
            classeB = int(B[classeMYOLO][0])
            coordB = B[classeMYOLO][1]
            name = 'classe'+str(classeB)
            for classeYOLO in range(len(A)):
                classeA = int(A[classeYOLO][0])
                coordA = A[classeYOLO][1]
                if classeB == classeA:
                    iou = my.iou(coordB,coordA)
                    parciais += [iou]
            if parciais == []:
                classes[name] += ['FP']
            if parciais != []:
                classes[name] += [max(parciais)]
                indexA = parciais.index(max(parciais))
                A = A[:indexA]+A[indexA+1:]

    dict1 = classesIOU
    dict2 = classes
    result = dict1.copy()
    # classes + intersection(classes,classesIOU)

    for key, value in dict2.items():
        if key in result:
            # Se a chave já existe, transforma o valor em uma lista e adiciona o novo valor
            if not isinstance(result[key], list):
                result[key] = result[key]
            result[key] += value
        else:
            # Se a chave não existe, apenas copia o valor
            result[key] = value
    return result

def mAP(yolo, myolo, mAPIOU,a=1):
    result = {}
    if a == 0: return
    for a in range(len(yolo)):
        dict2 = dicts(path1+str(yolo[a]),path2+str(myolo[a]),0.5)
        for key, value in dict2.items():
            if key in result:
                # merging the dicts for every video
                if not isinstance(result[key], list):
                    result[key] = result[key]
                result[key] += value
            else: result[key] = value
    strings = []
    floats = []
    # sorting the dict
    for key in result.items():
        for a in range(len(result[key[0]])):
            if type(result[key[0]][a]) == str:
                strings += [result[key[0]][a]]
            else: floats += [result[key[0]][a]]
        floats.sort(reverse=True)
        result[key[0]] = floats+strings
        floats = []
        strings = []

    # Calculating mAP for all the videos
    for key in result.items():
        b = key[0]
        for a in range(len(result[b])):
            x = result[b][a]
            if type(x) == float: 
                if x >= mAPIOU:
                    result[b][a] = 'TP'
                else: result[b][a] = 'FN'
    FN = 0
    FP = 0
    TP = 0
    precision_values = []
    recall_values = []
    for key in result.items():
        b = key[0]
        kek = result[b]
        print(kek)
        for a in range(len(result[b])):
            if kek[a] == 'TP':
                TP +=1
                try:
                    precision = TP/(TP + FP)
                    recall = TP/(TP + FN)
                except ZeroDivisionError:
                    precision = TP/(TP + FP+1)
                    recall = TP/(TP + FN+1)

            if kek[a] == 'FN':
                FN +=1
                try:
                    precision = TP/(TP + FP)
                    recall = TP/(TP + FN)
                except ZeroDivisionError:
                    precision = TP/(TP + FP+1)
                    recall = TP/(TP + FN+1)

            if kek[a] == 'FP':
                FP += 1
                try:
                    precision = TP/(TP + FP)
                    recall = TP/(TP + FN)
                except ZeroDivisionError:
                    precision = TP/(TP + FP+1)
                    recall = TP/(TP + FN+1)
            print(precision_values)
            print(recall_values)
            precision_values.append(precision)
            recall_values.append(recall)
        plt.scatter(recall_values, precision_values)
        plt.ylabel('Precision'+' '+str(key[0]))
        plt.xlabel('Recall'+' '+str(key[0]))
        plt.show()
        precision_values,recall_values,FP,TP,FN = [],[],0,0,0

myolo = os.listdir(path2)
yolo = os.listdir(path1)

mAP(yolo,myolo, 0.5)