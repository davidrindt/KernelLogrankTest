#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:49:14 2021

@author: rindt
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 35
plt.rcParams['axes.titlesize'] = 35
plt.rcParams['font.size'] = 35
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = "\\usepackage{subdepth}"


def survival_scatter_plot(x, z, d, filename=None, save=False):
    select=np.where(d==1)
    unselect=np.where(d==0)
    plt.scatter(x[select],z[select],c='navy',label='observed')
    plt.scatter(x[unselect],z[unselect],facecolors='none', edgecolors='r',label='censored')
#    plt.legend((l1,l2),('observed','censored'))
    plt.xlabel('covariate')
    plt.ylabel('time')
    plt.gcf().subplots_adjust(left=0.138  ,bottom=0.175)
    if save == True:
        plt.savefig(filename)
    plt.show()
    return None


if __name__ == '__main__':
    # Generate some data
    local_state = np.random.RandomState(1)
    n = 200
    dim = 10
    x = local_state.normal(loc=0, scale=1,size=n)
    c = np.zeros(n)
    for v in range(n):
        c[v] = local_state.exponential(np.exp(x[v]/8))
    t = local_state.exponential(.6,size=n)
    d = np.int64(c > t)
    z = np.minimum(t,c)
    
    # Make the plot
    survival_scatter_plot(x, z, d)