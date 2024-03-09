import re
import os

def write(classe,coord, name, path = "",A = 1):
    if A == 0: return
    coord = coord
    classe = classe
    file = open(str(path)+str(name)+'.txt','w')
    for x in range(len(coord)):
        file.write(str(classe[x])+ ' '+ str(coord[x])+ ' \n')
    file.close()
    with open(str(path)+str(name)+'.txt', 'r+') as f:
        content = f.read()
        f.seek(0)
        f.truncate()
        f.write(content.replace('tensor(','').replace('.)','    ').replace('])','').replace('[','').replace(',',''))

def compare(imgb,imga,thresh, A =1):
  if A == 0: return
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
VID = (os.listdir("D:/IC/datasets/imagenetVID/videos/train/"))