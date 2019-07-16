# -*- coding: utf-8 -*-
import sys
import os
import matplotlib.pyplot as plt
import numpy as np



x=np.linspace(0,2,10)
y=np.sin(x)
dx = x[1]-x[0]
der=np.gradient(y,dx)


plt.figure(1)
plt.plot(x,y,c="b")
plt.plot(x,der,c="r")
plt.plot(x,np.cos(x),c="g")



print("real val start point = ",np.cos(x)[0])
print("grad numpy val start point = ",der[0])

print("real val end point = ",np.cos(x)[len(x)-1])
print("grad numpy val end point = ",der[len(x)-1])


plt.show()
