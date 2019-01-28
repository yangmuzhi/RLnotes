#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 08:49:24 2019

@author: muzhi
"""

import gym
from A3C.A3C import A3C
from A2C.A2C import A2C
import numpy as np
import matplotlib.pyplot as plt
from utils.net import simple_net
import gym
env = gym.make('CartPole-v0')
a3c = A3C(state_shape=4, n_action=2, net=simple_net)
#a3c.train(env,100)
a3c.trainAsy('CartPole-v0', 50)




"""
len(a3c.cum_r)
plt.plot(range(len(a3c.cum_r)), np.array(a3c.cum_r))


#####
def play(N=100):
    r = []
    for i in range(N):
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
play(100)          
"""