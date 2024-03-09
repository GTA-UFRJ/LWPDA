'''
- Author: Hugo Antunes (antunes@gta.ufrj.br)

- From: Grupo de Telecomunicação e Automação (UFRJ/COPPE) - Rio de Janeiro - Brazil

- Professors: Rodrigo de Souza Couto (rodrigo@gta.ufrj.br) and 
  Pedro Henrique Cruz Caminha (cruz@gta.ufrj.br)

- Description: This code is the class used for trying to decrease a potencially delay
when using YOLO on real-time applications. The main idea is in the README file on GitHub.
'''

# Libraries
import cv2 as cv
from ultralytics import YOLO


# Class
class myolo():
    def write(namevid:str,lista, A = 1, path = ""):
            # Exemplo:
            # classes = (results[0].boxes.cls)
            # coord = (results[0].boxes.xyxy)
            # write(classes, coord, name, path,writ)
        if A == 0: return # A = 0, desliga a criação dos txts
        #model = YOLO('yolov8n.pt')
        #results = model(name, stream = True)
        #print('results = ',results)
        for x in range(len(lista)):
            classe = lista[x][0]
            coord = lista[x][1]
            file = open(str(path)+str(namevid)+'.txt','a')
            file.write('[')
            for y in range(len(coord)):
                file.write('['+ str(classe[y])+ ','+ str(coord[y])+ ']' + ',')
            file.write(']'+'\n')
      
    def compare(imgb,imga,thresh, A):

        if A == 0: return # A = 0 desliga a comparação
        # Compare two images, imgb(before), imga(after)
        # Thresh is the threshold that measure how similar the images must be
        # We sum every value of RGB (3) of every pixel (640x480)
        # So, the max value (if the images is literally the same) is 3x640x480 = 921000
        # Return True (The images are similar) or False
        # comp,width, pix = len(imga[0]),len(imga), width*comp

        x = abs((imgb-imga))
        z = ((0 <= x) & (x <= 10)).sum()
        if z >= thresh:
            return True
        return False
    #(os.listdir("D:/IC/datasets/imagenetVID/videos/train/"))

    def cam(thresh, compare = 0, write = 0):
        W = 0 # Variável auxiliar
        V = 0 # V -> verificador para auxiliar a função compare
        F = -1 # F -> contador de frames para auxiliar a função write
        # Load the YOLOv8 model
        model = YOLO('yolov8n.pt')
        cap = cv.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            video = "webcam"
            imga = frame
            F +=1 #A cada loop, adicionar um ao contador (para ajudar na nomeação do txt)
            name = str(video)+str(F) #nome do arquivo txt

            try: 
                if myolo.compare(imgb,imga,thresh, compare) and compare:
                    #frame = imgb
                    coord = (results[0].boxes.xyxy)
                    myolo.write(classes, coord, name, write)

                    #Frames iguais -> W = 1
                    W = 1
            except: pass

            if success:
                if W == 1: # Repetir a marcação e o frame passado
                    annotated_frame = imgb.plot(img = imga)
                    cv.imshow("YOLOv8 Inference", annotated_frame)
                    if cv.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    # Run YOLOv8 inference on the frame
                    results = model(frame)

                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()

                    classes = (results[0].boxes.cls)
                    coord = (results[0].boxes.xyxy)
                    myolo.write(classes, coord, name, write)

                    # Display the annotated frame
                    cv.imshow("YOLOv8 Inference", annotated_frame)
                    imgb = frame
                    # Break the loop if 'q' is pressed
                    if cv.waitKey(1) & 0xFF == ord("q"):
                        break
            #Zerar o verificador
            W = 0
            cap.release()
            #cv.destroyAllWindows()

    def iou(lista1,lista2): #listas que têm as coordenadas dos BBs!
        # https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        xA = max(lista1[0], lista2[0])
        yA = max(lista1[1], lista2[1])
        xB = min(lista1[2], lista2[2])
        yB = min(lista1[3], lista2[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (lista1[2] - lista1[0] + 1) * (lista1[3] - lista1[1] + 1)
        boxBArea = (lista2[2] - lista2[0] + 1) * (lista2[3] - lista2[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

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
            if x[linhas] and y[linhas] != '[]\n': # Caso em que o frame identifica algo
                a = x[linhas][:-4]+']]'
                b = y[linhas][:-4]+']]'
            a = x[linhas][:-1]
            b = y[linhas][:-1]
            a = eval(a)
            b = eval(b)
            for classeA in range(len(a)): 
                # para selecionar a classe de A => a[classeA][0]
                # para selecionar as coordenadas da classe A de A => a[classeA][1]
                for classeB in range(len(b)): #loopings para percorrer as listas
                    iou = myolo.iou(a[classeA][1],b[classeB][1])
                    parciais +=[iou]

                if parciais == []:
                    resultados += [[]] #Tem que mudar para 1.00 -> certo.
                else:
                    resultados+= [max(parciais)]
                    indexB = parciais.index(max(parciais))
                    #print(b)
                    b = b[:indexB]+b[indexB+1:]
                parciais = []
            save.write(str(resultados)+'\n')
            resultados = []
            return