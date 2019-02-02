#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a3c 训练cartpole
"""

import gym
from A3C.A3C import A3C
import numpy as np
import matplotlib.pyplot as plt
from utils.net import simple_net
from tqdm import tqdm
import sys

env = gym.make('CartPole-v0')
a3c = A3C(state_shape=4, n_action=2, net=simple_net)
from multiprocessing import cpu_count
cpu_count()
eps = int(sys.argv[1])

a3c.trainAsy('CartPole-v0', eps)


len(a3c.cum_r)
plt.plot(range(len(a3c.cum_r)), np.array(a3c.cum_r))

#####
#用训后模型测试
def play(N=200):
    r = []
    tqdm_e = tqdm(range(N))
    for i in tqdm_e:
        state = env.reset()
        cum_r = 0
        done = False
        while not done:
            state = state.reshape(-1,4)
            action = np.argmax(a3c.actor.action_prob(state))
            state, reward, done, _ = env.step(action)
            cum_r += reward
        r.append(cum_r)
    plt.plot(range(len(r)), np.array(r))
play(200)          

