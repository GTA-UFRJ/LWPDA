from LWPDA import lwpda
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon
from shapely.geometry import Polygon
a = lwpda('yolov8n-seg.pt', threshold=100)
#a.writingDetections('C:/Users/hugol/Desktop/videosTest/', 'C:/Users/hugol/Desktop/videosTest/')

file = open('C:/Users/hugol/Desktop/videosTest/ILSVRC2015_train_00005003masks.txt', 'r')
b = eval(file.readline())
maskA = b[1][0]
for x in range(70):
    b = eval(file.readline())
maskB = b[1][0]

fig = plt.figure(1, dpi=90)
ax = fig.add_subplot(121)
maskA = Polygon(maskA)
maskB = Polygon(maskB)
plot_polygon(maskA, ax=ax, add_points=False, color='RED', alpha=0.2)
plot_polygon(maskB, ax=ax, add_points=False, color='BLUE', alpha=0.2)

c = maskA.intersection(maskB)
plot_polygon(c, ax=ax, add_points=False, color='GRAY', alpha=1)

ax.set_title('a.intersection(b)')

# #2
# ax = fig.add_subplot(122)

# plot_polygon(maskA, ax=ax, add_points=False, color=GRAY, alpha=0.2)
# plot_polygon(maskB, ax=ax, add_points=False, color=GRAY, alpha=0.2)

# c = maskA.symmetric_difference(maskB)
# plot_polygon(c, ax=ax, add_points=False, color=BLUE, alpha=0.5)

# ax.set_title('a.symmetric_difference(b)')

# set_limits(ax, -1, 4, -1, 3)

plt.show()

print(a.iouSegmentation(maskA, maskB))


