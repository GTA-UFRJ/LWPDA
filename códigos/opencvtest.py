import cv2 as cv
import numpy as np
import sys

cap = cv.VideoCapture(0)
ret, frame = cap.read()
print(ret)
print(cap.read())
print(cap.isOpened())
print(cap.read())
a = frame
a = a.astype('int32')

ret, frame = cap.read()
print(ret)
print(cap.read())
print(cap.isOpened())
print(cap.read())
b = frame
b = b.astype('int32')

print('b = ',b,type(b))
print('a =',a,type(a))
print(a+300)
x = abs((a-b))
z = ((0 <= x) & (x < 10)).sum()
print(z)
#y = x/a
print('x =',x,type(x))
#print('y =',y,type(y))
#y = np.average(y, axis = 2 ,keepdims=True)
#print(y,len(y),len(y[0]))

#w = ((0 <= y) & (y < 1)).sum()
#print(w)
#y = np.average(x, axis=2, keepdims=True)
#y = np.average(y, axis=1)
#print('y =', y)
#print('len y = ',len(y[0]))
#print(a[0][0],b[0][0])
#w = a[479][639]/b[479][639]
#w = np.average(w)
while True:
    cv.imshow('frame',frame)
    if cv.waitKey(1) == ord('q'):
        break
#x = cap.read()
#print(len(x[1]),len(x[1][479]))
#pix = 480*640
#print(pix)
#Primeira coordenada "x[1]" representa o eixo y(altura) da imagem, já a segunda "x[1][0]" representa o eixo x(comprimento)