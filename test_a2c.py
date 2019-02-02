#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

a2c 训练cartpole

"""


import gym
import sys
from A2C.A2C import A2C
from keras.layers import Input, Flatten
from A2C.Actor import Actor
from A2C.Critic import Critic
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from utils.net import simple_net
from tqdm import tqdm

a2c = A2C(state_shape=4, n_action=2, net=simple_net)
a2c.actor.model.summary()
env = gym.make("CartPole-v0")
eps = int(sys.argv[1])

a2c.train(env, eps)
len(a2c.cum_r)

plt.plot(np.arange(len(a2c.cum_r)), np.array(a2c.cum_r))



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
            action = np.argmax(a2c.actor.action_prob(state))
            state, reward, done, _ = env.step(action)
            cum_r += reward
        r.append(cum_r)
    plt.plot(range(len(r)), np.array(r))
play(200)          

