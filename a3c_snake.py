#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a3c play snake
"""

from Env.snake_env import Snakes_subsonic
from keras_version.A3C.A3C import A3C
from keras_version.A2C.A2C import A2C
from utils.net import  simple_net, conv_shared
# import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
env = Snakes_subsonic()
state = env.reset()

# env.render()


#
# a3c = A3C(state_shape=2184, n_action=3, net=simple_net,
#           model_path='model/a3c/snake_demo')
a2c = A2C(state_shape=env.ground.shape, n_action=3, net=simple_net,
          model_path='model/a2c/snake_demo')

batch_size = 128
eps = 10000
#
# a3c.trainAsy(Snakes_subsonic, eps, use_gym=False)
# plt.plot(range(len(a3c.cum_r)),a3c.cum_r)
# plt.show()
#
a2c.train(env, eps)

# a2c.actor.model.summary()
# a2c.critic.model.summary()

def play(N=200):
    r = []
    tqdm_e = tqdm(range(N))
    for i in tqdm_e:

        state = env.reset()

        cum_r = 0
        done = False
        while not done:
            env.render()
            state = state[np.newaxis,...]
            action = np.argmax(a2c.actor.explore(state))
            state, reward, done, _ = env.step(action)
            cum_r += reward
        r.append(cum_r)
    env.close()
    # plt.plot(range(len(r)), np.array(r))
    plt.show()

play(10)
