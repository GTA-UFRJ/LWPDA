import matplotlib.pyplot as plt

# Code for analyse the mAP Results

# Just change the paths
arquivo = open('C:/Users/hugol/Desktop/IC/tests/mAP.txt','r')
dados = eval(arquivo.readline())
fig, ax = plt.subplots()
font1 = {'family':'serif','color':'black','size':20}
font2 = {'family':'serif','color':'black','size':13}
plt.plot()
#plt.title("Tempo de processamento por quadro", fontdict = font1)
plt.xlabel("Limiar", fontdict = font2)
plt.ylabel("mAP relativo", fontdict = font2)
ax.boxplot(dados)
ax.set_xticklabels(['00%','10%','20%','30%','40%','50%','60%','70%','80%','90%'])
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
# Mostrar o gráfico
#plt.show()
plt.savefig('boxplotmAP.pdf')