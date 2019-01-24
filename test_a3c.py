#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 08:49:24 2019

@author: muzhi
"""
import gym
from A3C.A3C import A3C
from A2C.A2C import A2C
env = gym.make('CartPole-v0')
a3c = A3C(state_shape=4, n_action=2)
a3c.train(env,1000)


