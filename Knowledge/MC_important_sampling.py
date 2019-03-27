#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
important sampling
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def p(x):
 #standard normal
    mu=0
    sigma=1
    return 1/(math.pi*2)**0.5/sigma*np.exp(-(x-mu)**2/2/sigma**2)

#uniform proposal distribution on [-4,4]
def q(x, a, b):
        return np.array([1/abs(b - a) for i in np.arange(x.shape[0])])
    
#draw N samples that conform to q(x), and then draw M from then that approximately conform to p(x)
N=1000
M=1000

x = np.random.uniform(-10, 10, N)
w_x = p(x) / q(x,-10,10)
w_x = w_x / sum(w_x)

w_xc = np.cumsum(w_x) #used for uniform quantile inverse
# resample from x with replacement with probability of w_x
X=np.array([])
for i in range(M):
    u = np.random.rand()
    X = np.hstack((X,x[w_xc>u][0]))

plt.hist(X,bins=100,normed=True)
plt.title('Sampling Importance Resampling')

