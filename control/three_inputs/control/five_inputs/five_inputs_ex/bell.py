import numpy as np
import matplotlib.pyplot as plt
a=2.5
b=2
c=-.18
x = []
y = -5
for i in range(10000):
    x.append(y)
    y += 0.001
#y  =1. / (1. + np.abs((x - b) / a) ** (2 * c))
y = np.zeros(10000)

for i in range(10000):
    y[i] = 1. / (1. + np.abs((x[i] - b) / a) ** (2 * c))
plt.plot(x,y)
plt.show()
