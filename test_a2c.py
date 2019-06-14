#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

a2c 训练cartpole

"""


import gym
import sys
from keras_version.A2C.A2C import A2C
from keras.layers import Input, Flatten
from keras_version.A2C.Actor import Actor
from keras_version.A2C.Critic import Critic
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from utils.net import simple_net
from tqdm import tqdm
import time

a2c = A2C(state_shape=84, n_action=2, net=simple_net)
a2c.actor.model.summary()
from Env.snake_env import Snakes_subsonic
env = Snakes_subsonic()
# env = gym.make("CartPole-v0")
# eps = int(sys.argv[1])
eps = 2000
a2c.train(env, eps)

len(a2c.cum_r)

plt.plot(np.arange(len(a2c.cum_r)), np.array(a2c.cum_r))
plt.show()
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
            state = state[np.newaxis,:]
            action = a2c.actor.explore(state)
            state, reward, done, _ = env.step(action)
            cum_r += reward
            env.render()
            time.sleep(0.01)

        r.append(cum_r)
    plt.plot(range(len(r)), np.array(r))
    plt.show()

a2c.actor.actprob

play(200)
