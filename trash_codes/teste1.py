from myolo import cam
import cv2 as cv

while True:
    myolo.cam(500000)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break