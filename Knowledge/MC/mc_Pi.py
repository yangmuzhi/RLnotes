#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo 方法
"""
import numpy as np
import matplotlib.pyplot as plt
# 计算 Pi
N = 100
width = 10
r = 5
def rand_sample(N):
    return np.random.uniform(0,5,N), np.random.uniform(0,5,N)

def estimate(N):
    x, y = rand_sample(N)
    ratio = np.sum(np.sqrt(x**2 + y**2) <= r) / N
    Pi = ratio * width**2 / r**2
    return Pi
N = [1,10,100,1e4,1e5,1e6,1e7]
Pi = []
for i in N:
    Pi.append(estimate(int(i)))
_ = [print("experiment nums: {}\t".format(int(i)), 
      "estimate Pi is {} \n".format(p)) for i,p in zip(N,Pi)]

