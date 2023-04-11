# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:25:12 2023

@author: hamad
"""


import numpy as np
import matplotlib.pyplot as plt
import pitch
import matplotlib.patches as mpatches

# Make an example a plot to show the distance and angle of shooting  
# create of pitch
(fig,ax) = pitch.createGoalMouth()

# plotting axis of angle
plt.plot((30,42),(0,25),color='black',linewidth=2.0)
plt.plot((42,37.5),(25,0),color='black',linewidth=2.0)

# plotting distance to the goal line
plt.plot((34,34),(25,0),'--', color='black',alpha=0.7,label='d:distance to goal line')
#plotting c
plt.plot((34,42),(25,25),'--',color='black',alpha=0.7,label='c:distance to centre')

#plotting distance of shooting
plt.plot((42,34),(25,0),'--',color='black',alpha=0.7,label='g: shooting distance')

#plotting the arrow
plt.arrow(40, 21, 5, 5)

#annotate theta
plt.annotate('theta',(45,26))

#annotate c
plt.annotate('c',(36,26) )

#annotate d
plt.annotate('d',(30,14) )

#annotate g
plt.annotate('g',(37,7) )

#draw of arc
pac = mpatches.Arc((42,28),height=15,width=15,angle=260,theta1=355,theta2=6,linewidth=3.0, label='theta:shooting angle')
ax.add_patch(pac)
plt.xlim((0,66))
plt.ylim((-3,35))
plt.legend()
plt.savefig('../figures/angle_distance_freeKick.png')

#Make a contour that show when you have to shot or cross
# axis dimensions
x= np.arange(0, 40)
y=np.arange(-20,20)

X, Y = np.meshgrid(x, y)
# Make a function tha compute the different between probability to shot and to cross from a free kick
def f(x, y):
    beta=np.arctan(7.32*x /(x**2 + y**2 - (7.32/2)**2))
    #distance of shooting
    d=np.sqrt(x**2 + y**2)
    
    # ajusted distance
    d3=abs(np.power(d,3)-16.5)
    
    # probability of free kick from shot(see fit_model.py)
    pgoal_shot=(1/(1+np.exp(3.985-4.879*beta)))*100
    
    # probability of free kick from cross(see fit_model.py)
    pgoal_cross=(1/(1+np.exp(3.0936-0.00005872*d3)))*100
    
    return pgoal_shot-pgoal_cross
#plotting
Z= f(X,Y)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z,[0], colors='black')
ax.clabel(CS, CS.levels,inline=True,  fontsize=10)

#  draw penalty area
plt.plot([0,16.5],[-20,-20],'--', alpha=0.5, color='black')
plt.plot([0,16.5],[20,20],'--', alpha=0.5,color='black')
plt.plot([16.5,16.5],[-20,20],'--',alpha=0.5, color='black')

# annotate shoot
plt.annotate('shoot',(17,0) )

#annotate cross
plt.annotate('cross',(23,0) )

plt.xlabel('x[m]')
plt.ylabel('y[m]')

plt.savefig('../figures/shot_cross_analytic_contour.png')