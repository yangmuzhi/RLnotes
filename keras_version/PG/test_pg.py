#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test
"""

from PG.pg import PG
from Env.snake_env import Snakes_subsonic
from utils.net import simple_net
import numpy as np
import matplotlib.pyplot as plt

env = Snakes_subsonic()
s = env.reset()

pg = PG(env, state_shape=s.shape, 
                n_actions=3, net=simple_net)
pg.train(500)
plt.plot(np.arange(len(pg.reward_his)), pg.reward_his)


s = s.reshape(-1,*s.shape)
pg.act(s)

d = False

while not d:
    s = env.reset()
    s_ = s.reshape(-1,*s.shape)
    a = pg.act(s_)
    s_next, r, d, info = env.step(a)
    # ob : state, reward, done, action, next_state
    pg.sample_pool.add_to_buffer((s,r,d,a,s_next))

s,r,d,a,s_next = pg.sample_pool.get_sample(shuffle=False)


    
