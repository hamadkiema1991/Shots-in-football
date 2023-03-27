# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 17:53:31 2023

@author: hamad
"""

import numpy as np
import matplotlib.pyplot as plt



x= np.arange(0, 40)
y=np.arange(-20,20)
X, Y = np.meshgrid(x, y)
def f(x, y):
    beta=np.arctan(7.32*x /(x**2 + y**2 - (7.32/2)**2))
    d=np.sqrt(x**2 + y**2)
    d3=abs(np.power(d,3)-16.5)
    # probability of free kick from shot
    pgoal_shot=(1/(1+np.exp(3.985-4.879*beta)))*100
    # probability of free kick from scross
    pgoal_cross=(1/(1+np.exp(3.0936-0.00005872*d3)))*100
    
    return pgoal_shot-pgoal_cross

Z= f(X,Y)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z,[0], colors='black')
ax.clabel(CS, CS.levels,inline=True,  fontsize=10)
#  draw penalty area
plt.plot([0,16.5],[-20,-20],'--', alpha=0.5, color='black')
plt.plot([0,16.5],[20,20],'--', alpha=0.5,color='black')
plt.plot([16.5,16.5],[-20,20],'--',alpha=0.5, color='black')
# 
plt.annotate('shoot',(17,0) )
plt.annotate('cross',(23,0) )
# 
plt.xlabel('x[m]')
plt.ylabel('y[m]')