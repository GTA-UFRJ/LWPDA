import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib.backends.backend_pdf import PdfPages

totaltime =[]
intervalosyolo, intervalosmyolo, porcentagens = [], [], []
for x in range(0,11):
    listframes = []
    totalframes, totalyolo, totalmyolo = [],[],[]
    yolo, discard, timeyolo, timemyolo = 0,0,0,0
    discard,yolo = 0,0

    if x == 10:
        x = 'yolo'
        file0 = open('C:/Users/hugol/Desktop/IC/tests/frames1/newtestframesyolo.txt', 'r')
    else: file0 = open('C:/Users/hugol/Desktop/IC/tests/frames1/newtestframes0.'+ str(x) +'.txt', 'r')
    for y in range(500):
        line0 = eval(file0.readline())
        for z in range(len(line0[0])):
            if line0[1][z] == "Y": 
                yolo+=1
                totalyolo += [line0[0][z]]
                timeyolo += line0[0][z]
            else: 
                discard+=1
                totalmyolo += [line0[0][z]]
                timemyolo += line0[0][z]
            listframes += [line0[0][z]*1000]
    porcentagens += [discard/(yolo+discard)]
    intervalosyolo += [st.t.interval(confidence=0.95, df=len(totalyolo)-1, loc=np.mean(totalyolo), scale=st.sem(totalyolo))]
    intervalosmyolo += [st.t.interval(confidence=0.95, df=len(totalmyolo)-1, loc=np.mean(totalmyolo), scale=st.sem(totalmyolo))]
    vasco = intervalosmyolo[0][1] -intervalosmyolo[0][0] 
    totaltime += [listframes]
    data0 = listframes 
    count, bins_count0 = np.histogram(data0, bins=1000)
    pdf0 = count / sum(count)
    cdf0 = np.cumsum(pdf0)
    if x == 'yolo': plt.plot(bins_count0[1:], cdf0, label='YOLO')
    else: plt.plot(bins_count0[1:], cdf0, label=str(x)+'0%')


#dados = totaltime
#fig, ax = plt.subplots()
#plt.plot()

font1 = {'family':'serif','color':'black','size':20}
font2 = {'family':'serif','color':'black','size':15}
#plt.xlabel("Limiar", fontdict = font2)
#plt.ylabel("mAP relativo", fontdict = font2)
#ax.boxplot(dados)
#ax.set_xticklabels(['00%','10%','20%','30%','40%','50%','60%','70%','80%','90%','YOLO'])
#plt.plot()
#plt.title("Tempo de processamento por quadro", fontdict = font1)
plt.xlabel("Tempo de processamento por quadro (ms)", fontdict = font2)
plt.ylabel("Proporção de quadros", fontdict = font2)
plt.legend(title='Limiares')
#plt.show()
plt.savefig('line_plot.pdf')



# Tempo médio gasto com o descarte  ->  02.575 +/- 0.021 (ms) -> threshold  00%
# Tempo médio gasto com o YOLO      ->  32.288 +/- 0.034 (ms) -> yolo puro