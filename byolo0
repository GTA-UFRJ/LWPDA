import time
import numpy as np
import cv2 as cv
import time
new_frame_time,prev_frame_time = 0,0

thresh = 450000
def compare(imgb,imga,thresh):
  #comparar duas imagens, imgb (before), imga(after). thresh é o limiar que definiremos
  #limiar é o valor que obtivemos expirimentalmente como adequado para decidir se uma imagem é ou não similar
  #além disso, a similaridade de pixels também deveria ser definida (por mim)
  #retorna True or False
  #comp,width, pix = len(imga[0]),len(imga), width*comp
  x = abs((imgb-imga))
  z = ((0 <= x) & (x <= 10)).sum()
  #print(z)
  if z >= thresh:
    return True
  return False

#Abrir a câmera
cap = cv.VideoCapture(0)
if not cap.isOpened():
  print("Cannot open camera")
  exit()
while True:
  #time.sleep(0.15) #-> delay
  # Capture frame-by-frame
  ret, frame = cap.read()
  imga = frame
  # if frame is read correctly ret is True
  new_frame_time = time.time()
  fps = 1/(new_frame_time - prev_frame_time)
  prev_frame_time = new_frame_time
  fps = str(int(fps))
  print(fps)
  if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break
  try: 
    if compare(imgb,imga,thresh):
      frame = imgb
  except: pass
      

  # Our operations on the frame come here
  #gray = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
  #print(len(frame), "x", len(frame[0]), ", ", len(frame[0][0]))
  # Display the resulting frame
  cv.imshow('frame', frame)
  imgb = frame
  #print(frame[0][0])
  if cv.waitKey(1) == ord('q'):
    break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
