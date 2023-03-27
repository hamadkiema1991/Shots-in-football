# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:11:36 2023

@author: hamad
"""

import FCPython 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# create of pitch
(fig,ax) = FCPython.createGoalMouth()

# plotting axis of angle
plt.plot((29,42),(0,25),color='black',linewidth=2.0)
plt.plot((42,36),(25,0),color='black',linewidth=2.0)
# plotting d
plt.plot((32.5,32.5),(25,0),'--', color='black',alpha=0.7,label='d:distance to goal line')
#plotting c
plt.plot((32.5,42),(25,25),'--',color='black',alpha=0.7,label='c:distance to centre')
#plotting g
plt.plot((42,32.5),(25,0),'--',color='black',alpha=0.7,label='g: shooting distance')
#plotting the arrow
plt.arrow(40, 18, 5, 5)
#annotate theta
plt.annotate('theta',(45,23))
#annotate c
plt.annotate('c',(36,26) )
#annotate d
plt.annotate('d',(30,14) )
#annotate g
plt.annotate('g',(36,7) )
#draw of arc
pac = mpatches.Arc((40,26),height=15,width=15,angle=260,theta1=355,theta2=15,linewidth=3.0, label='theta:shooting angle')
ax.add_patch(pac)
plt.xlim((0,66))
plt.ylim((-3,35))
plt.legend()

