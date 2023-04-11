# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 20:21:11 2023

@author: hamad
"""

import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
def createGoalMouth():
    pitch = VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
    fig, ax = pitch.draw()
    plt.xlim((0,68))
    plt.ylim((0,60))
    plt.gca().set_aspect('equal', adjustable='box')
    return fig,ax