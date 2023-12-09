def write_iou(wheresave:str, vidname:str, path1:str, path2:str, on = "1" ):

    # path1 => ground-truth
    # path2 => testing
    # as linhas dos txts com os BBs foram considerados assim:
    # [[class1[coord1]],[classe2[coord2]],...] a primeira linha indica o primeiro frame
    # [[class1[coord1]],[classe2[coord2]],...] a segunda linha indica o segundo frame...

    if on == 0: return

    save = open(wheresave+vidname, 'a')
    pathfile1, pathfile2 = path1+vidname, path2+vidname
    fileA = open(pathfile1, 'r')
    fileB = open(pathfile2, 'r')
    x = fileA.readlines()
    y = fileB.readlines()
    resultados = [] # lista dos resultados definitivos
    parciais = [] # lista para armazenar todos os iou, para cada classe encontrada

    for linhas in range(len(x)): #linhas = frames
        if x[linhas] and y[linhas] != '[]\n':
            a = x[linhas][:-4]+']]'
            b = y[linhas][:-4]+']]'
        a = x[linhas][:-1]
        b = y[linhas][:-1]
        a = eval(a)
        b = eval(b)
        for classeA in range(len(a)):
            for classeB in range(len(b)): #loopings para percorrer as listas
                iou = myolo.iou(a[classeA][1],b[classeB][1])
                parciais +=[iou]

            if parciais == []:
                resultados += [[]]
            else:
                resultados+= [max(parciais)]
                indexB = parciais.index(max(parciais))
                #print(b)
                b = b[:indexB]+b[indexB+1:]
            parciais = []
        save.write(str(resultados)+'\n')