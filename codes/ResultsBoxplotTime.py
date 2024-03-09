import os
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Code for analyse the results from video time processing

# Just change the paths
dir = 'C:/Users/hugol/Desktop/IC/tests/videos/'
yolo = open('C:/Users/hugol/Desktop/IC/tests/videos/testvideoyolo.txt','r')
VID = (os.listdir(dir))
times = []
intervalos = []
dados = []
for y in range(len(VID)):
    time = 0 
    file = open(dir+VID[y])
    list0=[]
    yolo = open('C:/Users/hugol/Desktop/IC/tests/videos/testvideoyolo.txt','r')
    for x in range(500):
        line = eval(file.readline())
        timeyolo = eval(yolo.readline())
        time += float(line)
        list0 += [abs(float(line)- timeyolo)/timeyolo]
    times += [time]
    vascudo = st.t.interval(confidence=0.95, df=len(list0)-1, loc=np.mean(list0), scale=st.sem(list0))
    vasco = np.mean(list0)
    dagama = (vascudo[1] - vascudo[0])/2
    dados += [list0]
    intervalos += [str(vasco)+'$\mathcal{\pm}$'+str(dagama)]


#print(dados)
#print(times,len(times))
#len(times)
#print(VID)
intervalos += [str(vasco)+'$\mathcal{\pm}$'+str(dagama)]
print(intervalos, len(intervalos))

fig, ax = plt.subplots()
font1 = {'family':'serif','color':'black','size':20}
font2 = {'family':'serif','color':'black','size':13}
plt.plot()
#plt.title("Tempo de processamento por quadro", fontdict = font1)
plt.xlabel("Limiar", fontdict = font2)
plt.ylabel("Diferenca relativa do tempo de\n processamento por vídeo", fontdict = font2)
ax.boxplot(dados)
ax.set_xticklabels(['00%','10%','20%','30%','40%','50%','60%','70%','80%','90%','YOLO'])
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
# Mostrar o gráfico
#plt.show()
plt.savefig('boxplottime.pdf')