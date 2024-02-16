import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt 

totalframes, totalyolo, totalmyolo = [],[],[]
yolo, discard, timeyolo, timemyolo = 0,0,0,0

file = open('C:/Users/hugol/Desktop/IC/tests/frames1/newtestframes0.4.txt', 'r')
file = open('C:/Users/hugol/Desktop/IC/tests/frames1/newtestframesyolo.txt', 'r')

for y in range(500):

    line = eval(file.readline())

    for x in range(len(line[0])):
        totalframes += [float(line[0][x])]
        if line[1][x] == 'Y': 
            yolo +=1
            totalyolo += [line[0][x]]
            timeyolo += line[0][x]

        if line[1][x] == 'D': 
            discard +=1
            totalmyolo += [line[0][x]]
            timemyolo += line[0][x]

timeframes = yolo + discard

confidence_interval = st.t.interval(confidence=0.95, df=len(totalyolo)-1, loc=np.mean(totalyolo), scale=st.sem(totalyolo))
print(confidence_interval)

#matplotlib inline 
  
# No of Data points 
#N = 500
  
# initializing random values 
data = totalframes 

# getting data of the histogram 
count, bins_count = np.histogram(data, bins=1000)

pdf = count / sum(count) 
  

cdf = np.cumsum(pdf) 
  
# plotting PDF and CDF 
#plt.plot(bins_count[1:], pdf, color="red", label="PDF") 
plt.plot(bins_count[1:], cdf, label="CDF") 
#plt.legend() 
plt.show()