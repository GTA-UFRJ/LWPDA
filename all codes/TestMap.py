# This code is responsible to calculate the mean average precision (mAP)

#Libs
from myolo import myolo as my
import numpy as np
import scipy.stats as st
import statistics


# Txt files paths (folders)
path1 = "C:/Users/hugol/Desktop/IC/tests/txt_Myolo1.0/" # Ground Truth
path2 = "C:/Users/hugol/Desktop/IC/tests/txt_Myolo1.0/" # Tests

#txtyolo = "C:/Users/hugol/Desktop/IC/txt_yolo/ILSVRC2015_train_00005003.mp4.txt"
#txtmyolo = "C:/Users/hugol/Desktop/IC/txt_myolo_500000/ILSVRC2015_train_00005003.mp4.txt"
def dicts(txtyolo:str, txtmyolo:str, on=1): 
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
    output = 0
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
    AP = []
    for key in result.items():
        TPFN = 0
        b = key[0]
        kek = result[b]
        for a in range(len(result[b])):
            x = result[b][a]
            if x == 'TP' or x == 'FN':
                TPFN +=1
        
        FN = 0
        FP = 0
        TP = 0
        precision_values = []
        recall_values = []

        for a in range(len(result[b])):
            if kek[a] == 'TP':
                TP +=1
                try:
                    precision = TP/(TP + FP)
                    recall = TP/(TPFN)
                except ZeroDivisionError:
                    precision = TP/(TP + FP+1)
                    recall = TP/(TPFN)

            if kek[a] == 'FN':
                FN +=1
                try:
                    precision = TP/(TP + FP)
                    recall = TP/(TPFN)
                except ZeroDivisionError:
                    precision = TP/(TP + FP+1)
                    recall = TP/(TPFN)

            if kek[a] == 'FP':
                FP += 1
                try:
                    precision = TP/(TP + FP)
                    recall = TP/(TPFN)
                except ZeroDivisionError:
                    precision = TP/(TP + FP+1)
                    recall = TP/(TPFN)
                    
            precision_values.append(precision)
            recall_values.append(recall)
        #print(recall_values)
        #print (precision_values)
        #plt.scatter(recall_values, precision_values)
        #plt.ylabel('Precision'+' '+str(key[0]))
        #plt.xlabel('Recall'+' '+str(key[0]))
        #plt.show()
        AP += [recall_values[-1]]
        precision_values,recall_values,FP,TP,FN = [],[],0,0,0
    
    for x in range(len(AP)):
        output += AP[x]

    output = output/len(AP)
    data = AP
    confidence_interval = st.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
    DP = statistics.stdev(AP)
    #intervalocheck = (output-0.95(dp/))
    return AP, confidence_interval, DP

# Results

# threshold   //   mAP   //                        confidence                         //   Desvio Padrão   // Descartados (%)
#     0%      ->     4,7%  (0.024893596737432917, 0.0699503592870) -> (04,7% +/- 2,3%)  0.0992561440388874      99.76%
#    10%      ->    30,1%  (0.24593526786019249, 0.35759508995040) -> (30,1% +/- 5,6%)  0.2459769135106025      98.24%
#    20%      ->    48,4%  (0.4291668899511522, 0.540075328631902) -> (48,4% +/- 5,5%)  0.2443216809617406      90.54%
#    30%      ->    64,5%  (0.5995159960785145, 0.690802205443680) -> (64,5% +/- 4,5%)  0.2010956099104638      79.47%
#    40%      ->    78,9%  (0.7586892733817621, 0.819490468649709) -> (78,9% +/- 3,0%)  0.1339397651706934      66.34%
#    50%      ->    88,0%  (0.8568094862121882, 0.904783038245631) -> (88,0% +/- 2,4%)  0.1056815785519726      50.69%
#    60%      ->    93,5%  (0.9174974940259211, 0.954164013662243) -> (93,5% +/- 1,8%)  0.0807731658575566      35.41%
#    70%      ->    97,1%  (0.961755943857658, 0.9806153432417544) -> (97,1% +/- 5,9%)  0.0415456228061660      23.37%
#    80%      ->    98,6%  (0.9814480848465654, 0.991693423201021) -> (98,6% +/- 0,5%)  0.0225695927068986      14.28%
#    90%      ->    99,6%  (0.994556826437707, 0.9976782589985121) -> (99,6% +/- 0,2%)  0.0068762454808318      06.78%
#    100%     ->    100%   