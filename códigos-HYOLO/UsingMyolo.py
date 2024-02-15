# Libraries

import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from myolo import myolo as my
from random import randint

# If you want to use webcam, VID = int(0)
VID = "C:/Users/hugol/Desktop/IC/datasets/train/ILSVRC2015_train_00005003.mp4"
VID = 0
# Turn on or turn off the function compare
compar = 1
# Threshold of the function compare
threshold = 300000

def Cam(VID,threshold,compar):

    # This code is to using the compare function of the class Myolo
    # You can use in a video, just changing the input of cv.VideoCapture
    # This dict is use to change the color of the classes detecteds
    dict = {0: ['person',(randint(0,255),randint(0,255),randint(0,255))], 1: ['bicycle',(randint(0,255),randint(0,255),randint(0,255))], 2: ['car',(randint(0,255),randint(0,255),randint(0,255))], 3: ['motorcycle',(randint(0,255),randint(0,255),randint(0,255))], 4: ['airplane',(randint(0,255),randint(0,255),randint(0,255))], 5: ['bus',(randint(0,255),randint(0,255),randint(0,255))], 6: ['train',(randint(0,255),randint(0,255),randint(0,255))], 7: ['truck',(randint(0,255),randint(0,255),randint(0,255))], 8: ['boat',(randint(0,255),randint(0,255),randint(0,255))], 9: ['traffic light',(randint(0,255),randint(0,255),randint(0,255))], 10: ['fire hydrant',(randint(0,255),randint(0,255),randint(0,255))], 11: ['stop sign',(randint(0,255),randint(0,255),randint(0,255))], 12: ['parking meter',(randint(0,255),randint(0,255),randint(0,255))], 13: ['bench',(randint(0,255),randint(0,255),randint(0,255))], 14: ['bird',(randint(0,255),randint(0,255),randint(0,255))], 15: ['cat',(randint(0,255),randint(0,255),randint(0,255))], 16: ['dog',(randint(0,255),randint(0,255),randint(0,255))], 17: ['horse',(randint(0,255),randint(0,255),randint(0,255))], 18: ['sheep',(randint(0,255),randint(0,255),randint(0,255))], 19: ['cow',(randint(0,255),randint(0,255),randint(0,255))], 20: ['elephant',(randint(0,255),randint(0,255),randint(0,255))], 21: ['bear',(randint(0,255),randint(0,255),randint(0,255))], 22: ['zebra',(randint(0,255),randint(0,255),randint(0,255))], 23: ['giraffe',(randint(0,255),randint(0,255),randint(0,255))], 24: ['backpack',(randint(0,255),randint(0,255),randint(0,255))], 25: ['umbrella',(randint(0,255),randint(0,255),randint(0,255))], 26: ['handbag',(randint(0,255),randint(0,255),randint(0,255))], 27: ['tie',(randint(0,255),randint(0,255),randint(0,255))], 28: ['suitcase',(randint(0,255),randint(0,255),randint(0,255))], 29: ['frisbee',(randint(0,255),randint(0,255),randint(0,255))], 30: ['skis',(randint(0,255),randint(0,255),randint(0,255))], 31: ['snowboard',(randint(0,255),randint(0,255),randint(0,255))], 32: ['sports ball',(randint(0,255),randint(0,255),randint(0,255))], 33: ['kite',(randint(0,255),randint(0,255),randint(0,255))], 34: ['baseball bat',(randint(0,255),randint(0,255),randint(0,255))], 35: ['baseball glove',(randint(0,255),randint(0,255),randint(0,255))], 36: ['skateboard',(randint(0,255),randint(0,255),randint(0,255))], 37: ['surfboard',(randint(0,255),randint(0,255),randint(0,255))], 38: ['tennis racket',(randint(0,255),randint(0,255),randint(0,255))], 39: ['bottle',(randint(0,255),randint(0,255),randint(0,255))], 40: ['wine glass',(randint(0,255),randint(0,255),randint(0,255))], 41: ['cup',(randint(0,255),randint(0,255),randint(0,255))], 42: ['fork',(randint(0,255),randint(0,255),randint(0,255))], 43: ['knife',(randint(0,255),randint(0,255),randint(0,255))], 44: ['spoon',(randint(0,255),randint(0,255),randint(0,255))], 45: ['bowl',(randint(0,255),randint(0,255),randint(0,255))], 46: ['banana',(randint(0,255),randint(0,255),randint(0,255))], 47: ['apple',(randint(0,255),randint(0,255),randint(0,255))], 48: ['sandwich',(randint(0,255),randint(0,255),randint(0,255))], 49: ['orange',(randint(0,255),randint(0,255),randint(0,255))], 50: ['broccoli',(randint(0,255),randint(0,255),randint(0,255))], 51: ['carrot',(randint(0,255),randint(0,255),randint(0,255))], 52: ['hot dog',(randint(0,255),randint(0,255),randint(0,255))], 53: ['pizza',(randint(0,255),randint(0,255),randint(0,255))], 54: ['donut',(randint(0,255),randint(0,255),randint(0,255))], 55: ['cake',(randint(0,255),randint(0,255),randint(0,255))], 56: ['chair',(randint(0,255),randint(0,255),randint(0,255))], 57: ['couch',(randint(0,255),randint(0,255),randint(0,255))], 58: ['potted plant',(randint(0,255),randint(0,255),randint(0,255))], 59: ['bed',(randint(0,255),randint(0,255),randint(0,255))], 60: ['dining table',(randint(0,255),randint(0,255),randint(0,255))], 61: ['toilet',(randint(0,255),randint(0,255),randint(0,255))], 62: ['tv',(randint(0,255),randint(0,255),randint(0,255))], 63: ['laptop',(randint(0,255),randint(0,255),randint(0,255))], 64: ['mouse',(randint(0,255),randint(0,255),randint(0,255))], 65: ['remote',(randint(0,255),randint(0,255),randint(0,255))], 66: ['keyboard',(randint(0,255),randint(0,255),randint(0,255))], 67: ['cell phone',(randint(0,255),randint(0,255),randint(0,255))], 68: ['microwave',(randint(0,255),randint(0,255),randint(0,255))], 69: ['oven',(randint(0,255),randint(0,255),randint(0,255))], 70: ['toaster',(randint(0,255),randint(0,255),randint(0,255))], 71: ['sink',(randint(0,255),randint(0,255),randint(0,255))], 72: ['refrigerator',(randint(0,255),randint(0,255),randint(0,255))], 73: ['book',(randint(0,255),randint(0,255),randint(0,255))], 74: ['clock',(randint(0,255),randint(0,255),randint(0,255))], 75: ['vase',(randint(0,255),randint(0,255),randint(0,255))], 76: ['scissors',(randint(0,255),randint(0,255),randint(0,255))], 77: ['teddy bear',(randint(0,255),randint(0,255),randint(0,255))], 78: ['hair drier',(randint(0,255),randint(0,255),randint(0,255))], 79: ['toothbrush',(randint(0,255),randint(0,255),randint(0,255))]}

    # W is a auxiliary variable, thresh is the threshold of the function compare
    W = 0

    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    # The first frame showed in the video take more time to be show
    # So, we show a random image just to initiate the camera


    # Percorrer o diretório onde contém os arquivos de vídeo
    # Open the video
    cap = cv.VideoCapture(VID)

    # Loop through the video frames

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        imga = frame

        try: 
            if my.compare(imgb,imga,threshold,compar):
                #frame = imgb
                classes = (results[0].boxes.cls.tolist())
                coord = (results[0].boxes.xyxy.tolist())
                # Same frames -> W = 1
                W = 1
        except: pass

        if W == 1: # Repeat the annotations of the last frame
            vasco = Annotator(frame)
            for r in range (len(classes)):
                vasco.box_label(coord[r],str(dict[int(classes[r])][0]),dict[int(classes[r])][1])
            gama = vasco.result()
            cv.imshow("YOLOv8 Inference", gama)
            if cv.waitKey(1) & 0xFF == ord("q"):
                return
        else:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv.imshow("YOLOv8 Inference", annotated_frame)
            imgb = frame
            # Break the loop if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord("q"):
                return
        W = 0
    # Release the video capture object and close the display window
    cap.release()
    cv.destroyAllWindows()

Cam(VID,threshold,compar)