#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:12:27 2019

@author: muzhi
"""


import gym
import sys
from A2C.A2C import A2C
from keras.layers import Input, Flatten
from A2C.Actor import Actor
from A2C.Critic import Critic
from keras.utils import to_categorical

a2c = A2C(state_shape=4, n_action=2)
a2c.actor.model.summary()
env = gym.make("CartPole-v0")

a2c.train(env, 500)













