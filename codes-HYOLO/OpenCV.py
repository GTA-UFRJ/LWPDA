import cv2 as cv
import os
dir = 'C:/Users/hugol/Desktop/IC/datasets/vidvrd-videos-part1/'
VID = (os.listdir(dir))

totaltime, totalframes = 0,0



for x in range(len(VID)):
    v = cv.VideoCapture(dir+VID[x])
    totalframes += v.get(cv.CAP_PROP_FRAME_COUNT)
    fps = v.get(cv.CAP_PROP_FPS)
    totalNoFrames = v.get(cv.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames / fps
    totaltime += durationInSeconds
    print(totalframes,totaltime)

print(totalframes,totaltime)