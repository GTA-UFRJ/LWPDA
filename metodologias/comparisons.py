import numpy as np 
import cv2 as cv
from skimage import metrics
import imagehash
from PIL import Image

class methods():
    def mse(image1,image2):
        image1 = cv.imread(image1)
        image2 = cv.imread(image2)
        image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        h, w = image1.shape
        diff = cv.subtract(image1, image2)
        err = np.sum(diff**2)
        mse = err/(float(h*w))
        return mse, diff
    def histogram(image1,image2):
        image1 = cv.imread(image1)
        image2 = cv.imread(image2)
        hist_img1 = cv.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_img1[255, 255, 255] = 0 #ignore all white pixels
        cv.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        hist_img2 = cv.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_img2[255, 255, 255] = 0  #ignore all white pixels
        cv.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        metric_val = cv.compareHist(hist_img1, hist_img2, cv.HISTCMP_CORREL)
        return metric_val
    def ssim(image1,image2):
        image1 = cv.imread(image1)
        image2 = cv.imread(image2)
        image2 = cv.resize(image2, (image1.shape[1], image1.shape[0]), interpolation = cv.INTER_AREA)
        print(image1.shape, image2.shape)
        # Convert images to grayscale
        image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        # Calculate SSIM
        ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
        return ssim_score
    def imagehash(image1,image2,t):
        # Convert cvImg from OpenCV format to PIL format
        image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
        image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
        # Get the average hashes of both images
        hash0 = imagehash.average_hash(image1)
        hash1 = imagehash.average_hash(image2)
        t = 5  # Can be changed according to what works best for your images
        hashDiff = hash0 - hash1  # Finds the distance between the hashes of images
        if hashDiff < t:
            return True
        return False
    