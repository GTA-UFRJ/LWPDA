from myolo import myolo as my
import os

path1 = "D:/IC/txt_yolo/" # Ground Truth
path2 = "D:/IC/txt_myolo_500000/" # Tests
myolo = os.listdir(path1)
yolo = os.listdir(path2)
wheresave = "D:/IC/iou_500000/"
vidname = 'ILSVRC2015_train_00005005.mp4.txt'

def write_iou(wheresave:str, vidname:str, path1:str, path2:str, on = "1" ):
    # path1 => Ground truth
    # path2 = > Tests
    if on == 0: return
    save = open(wheresave+vidname+'testando', 'a')
    pathfile1, pathfile2 = path1+vidname, path2+vidname
    fileA = open(pathfile1, 'r')
    fileB = open(pathfile2, 'r')
    x = fileA.readlines()
    y = fileB.readlines()
    resultados = []
    parciais = []
    FN = 0 # False Negative
    FP = 0 # False positive
    for linhas in range(len(x)): #linhas = frames
        #print('a = ',a)
        print(x[linhas])
        print(type(x[linhas]))
        print(y[linhas])
        if x[linhas] and y[linhas] != '[]\n':
            a = x[linhas][:-4]+']]'
            b = y[linhas][:-4]+']]'
        a = x[linhas][:-1]
        b = y[linhas][:-1]
        a = eval(a)
        b = eval(b)
        
        for classeA in range(len(a)):
            for classeB in range(len(b)):
                iou = my.iou(a[classeA][1],b[classeB][1])
                parciais +=[iou]
            if parciais == []:
                resultados += [[]] #Tem que mudar
            else:
                resultados+= [max(parciais)]
                indexB = parciais.index(max(parciais))
                #print(b)
                b = b[:indexB]+b[indexB+1:]
            parciais = []
        save.write(str(resultados)+'\n')

for x in range(len(yolo)):
    vidname = yolo[x]
    my.write_iou(wheresave,vidname,path1,path2)