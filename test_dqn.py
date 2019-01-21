#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:25:10 2019

@author: muzhi
"""

import gym
from dqn.deepqn import DQN

dqn = DQN(state_shape=4, n_action=2)
dqn.agent.q_eval_net.summary()
env = gym.make("CartPole-v0")

dqn.train(env, 500, 256)


