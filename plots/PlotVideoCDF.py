import numpy as np
import matplotlib.pyplot as plt
for x in range(0,10):
    listvideos0 = []
    if x == 10:
        x = 'yolo'
    file0 = open('C:/Users/hugol/Desktop/IC/tests/videos/testvideo'+ str(x) +'.txt', 'r')
    file1 = open('C:/Users/hugol/Desktop/IC/tests/videos/testvideo'+ 'yolo' +'.txt', 'r')
    while True:
        line0 = file0.readline()
        line1 = file1.readline()
        if not line0:
            break
        listvideos0 += [abs((float(line0) - float(line1))/float(line1))]
    data0 = listvideos0 
    count, bins_count0 = np.histogram(data0, bins=1000)
    pdf0 = count / sum(count)
    cdf0 = np.cumsum(pdf0)
    if x == 'yolo': plt.plot(bins_count0[1:], cdf0, label='YOLO')
    else: plt.plot(bins_count0[1:], cdf0, label=str(x)+'0%')

font1 = {'family':'serif','color':'black','size':20}
font2 = {'family':'serif','color':'black','size':15}

plt.title("Tempo relativo de processamento", fontdict = font1)
plt.xlabel("Diferença relativa do processamento/video (s)", fontdict = font2)
plt.ylabel("Videos totais", fontdict = font2)
plt.legend(title='Limiares')
plt.show()