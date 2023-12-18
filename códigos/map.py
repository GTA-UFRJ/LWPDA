#Libs
import os
import numpy as np
import math
import re
from myolo import myolo as my

path1 = "C:/Users/hugol/Desktop/IC/txt_yolo/" # Ground Truth
path2 = "C:/Users/hugol/Desktop/IC/txt_myolo_500000/" # Tests

#txtyolo = "C:/Users/hugol/Desktop/IC/txt_yolo/ILSVRC2015_train_00005003.mp4.txt"
#txtmyolo = "C:/Users/hugol/Desktop/IC/txt_myolo_500000/ILSVRC2015_train_00005003.mp4.txt"
def mAP(txtyolo:str, txtmyolo:str, mAPIOU, on=1):
    if on == 0: return
    classes = {}
    classesIOU = {}
    fileA = open(txtyolo, 'r')
    fileB = open(txtmyolo, 'r')
    x = fileA.readlines()
    y = fileB.readlines()
    parciais = []
    for linhas in range(len(x)): #linhas = frames
        #print('a = ',a)
        #print(x[linhas])
        #print(type(x[linhas]))
        #print(y[linhas])
        if x[linhas] and y[linhas] != '[]\n':
            a = x[linhas][:-4]+']]'
            b = y[linhas][:-4]+']]'
        a = x[linhas][:-1]
        b = y[linhas][:-1]
        a = eval(a)
        b = eval(b)
        #print(a)
        #print(b)
        for obj in range(len(a)):
            v = int(a[obj][0])
            classesIOU['classe'+str(v)] = []
            classes['classe'+str(v)] = []

        for obj in range(len(b)):
            v = int(b[obj][0])
            classesIOU['classe'+str(v)] = []
            classes['classe'+str(v)] = []

    linhas = 0
    for linhas in range(len(x)): #linhas = frames
        a = eval(x[linhas])
        b = eval(y[linhas])
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
                #print(classesIOU)
                indexB = parciais.index(max(parciais))
                b = B[:indexB]+B[indexB+1:]
            else: 
                classesIOU[name]+=['FN']

    #print(classesIOU)
    linhas = 0
    classeMYOLO = 0
    classeYOLO = 0
    for linhas in range(len(x)): #linhas = frames
        a = eval(x[linhas])
        b = eval(y[linhas])
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
    #print(classes)

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

myolo = os.listdir(path2)
yolo = os.listdir(path1)
result = {}

for a in range(len(yolo)):
    dict2 = mAP(path1+str(yolo[a]),path2+str(myolo[a]),0.5)
    for key, value in dict2.items():
        if key in result:
            # Se a chave já existe, transforma o valor em uma lista e adiciona o novo valor
            if not isinstance(result[key], list):
                result[key] = result[key]
            result[key] += value
        else:
            # Se a chave não existe, apenas copia o valor
            result[key] = value

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
mAPIOU = 0.5
for key in result.items():
    b = key[0]
    #print(result[b])
    for a in range(len(result[b])):
        x = result[b][a]
        if type(x) == float: 
            if x >= mAPIOU:
                result[b][a] = 'TP'
            else: result[b][a] = 'FN'

print(result)