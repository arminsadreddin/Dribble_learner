import matplotlib.pyplot as plt
import numpy as np

file  = open('loss.txt', 'r')
i = 1
data = []
for line in file:
    i+=1
    data.append(float(line))
plt.plot(data)
plt.title('Loss_PLOT')
plt.ylabel('loss')
plt.xlabel('time')
plt.show()