import numpy as np
import os
import shutil


path1 = 'C:/Users/amoot/Desktop/IMAGENET/vidvrd-videos-part1'
path2 = 'C:/Users/amoot/Desktop/IMAGENET/vidvrd-videos-part2'
l = os.listdir(path1)
p = os.listdir(path2)
dst = 'C:/Users/amoot/Desktop/IMAGENET/amostra/'
l1 = [332, 112, 303, 179, 173, 151, 197, 221, 41, 365, 300, 90, 165, 348, 49, 424, 462, 283, 113, 33, 491, 64, 210, 81, 416, 148, 155, 147, 139, 309]
l2 = [42, 80, 17, 204, 279, 225, 479, 262, 180, 401, 470, 70, 179, 425, 217, 210, 21, 416, 47, 59, 114, 271, 37, 364, 250, 26, 65, 263, 266, 441]
name1 = []
name2 = []
for x in range(len(l1)):
    src = path2+'/'+p[l1[x]]
    shutil.copy(src, dst)
    src = path1+'/'+l[l2[x]]
    shutil.copy(src, dst)
    name2 += [l2[x]]

print(name1,name2)
shutil.copyfile(src, dst)