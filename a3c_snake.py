#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a3c play snake
"""

from snake_env import Snakes_subsonic
from A3C.A3C import A3C
from utils.net import  simple_net
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
env = Snakes_subsonic()
state = env.reset()

a3c = A3C(state_shape=84, n_action=3, net=simple_net,
          model_path='model/a3c/snake_demo')

batch_size = 32
eps = 2000

a3c.trainAsy(Snakes_subsonic, eps, use_gym=False)
plt.plot(range(len(a3c.cum_r)),a3c.cum_r)

def play(N=200):
    r = []
    tqdm_e = tqdm(range(N))
    for i in tqdm_e:

        state = env.reset()

        cum_r = 0
        done = False
        while not done:
            env.render()
            state = state.reshape(-1,84)
            action = np.argmax(dqn.agent.q_eval(state))
            state, reward, done, _ = env.step(action)
            cum_r += reward
        r.append(cum_r)
    plt.plot(range(len(r)), np.array(r))
play(10)
