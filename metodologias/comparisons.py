import numpy as np 
import cv2 as cv
from skimage import metrics
import imagehash
from PIL import Image
import os
import time


"""
    Todas as imagens PRECISAM estar do mesmo tamanho. Os vídeos não mudam de dimensionamento, então as funções nao possuem resize.
    As funções ainda não possuem um threshold, mas terão em breve.
"""

class methods():
    def mean_squared_error(image1, image2,t):
        # quanto mais próximos de 0, mais similar
        image1,image2 = cv.imread(image1),cv.imread(image2)
        return np.mean((image1 - image2) ** 2, dtype=np.float64)
    def histogram(image1,image2,t):
        # Maior similaridade = 1 (figuras iguais) -> quanto mais próximo de 1, mais similares
        image1 = cv.imread(image1)
        image2 = cv.imread(image2)
        hist_img1 = cv.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_img1[255, 255, 255] = 0 #ignore all white pixels
        cv.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        hist_img2 = cv.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_img2[255, 255, 255] = 0  #ignore all white pixels
        cv.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        metric_val = cv.compareHist(hist_img1, hist_img2, cv.HISTCMP_CORREL)
        if metric_val <= t: return True
        return False
    def ssim(image1,image2,t):
        # quanto mais próximos de 1, mais similar
        image1 = cv.imread(image1)
        image2 = cv.imread(image2)
        # Convert images to grayscale
        image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        # Calculate SSIM
        ssim_score = metrics.structural_similarity(image1_gray, image2_gray,full=True)
        if ssim_score[0] >= t: return True
        return False
    def imagehash(image1,image2,t):
        hash1 = imagehash.average_hash(Image.open(image1))
        hash2 = imagehash.average_hash(Image.open(image2))
        diff = hash1 - hash2
        print('dif = ',diff)

    
    def hyolo(imgb,imga,thresh):
        z = 0
        x = abs((imgb-imga))
        ch = (0 <= x) & (x <= 10)
        z += np.sum(ch)
        if z >= thresh:
            return True
        return False

s = time.time()
x = methods.imagehash('C:/Users/amoot/Desktop/BetterYolo/metodologias/image1.jpg','C:/Users/amoot/Desktop/BetterYolo/metodologias/image2.png',0)
print('Similaridade = ',x)
print(time.time()-s)