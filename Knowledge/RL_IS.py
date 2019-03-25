#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

mc important sampling from maze

Env: maze from "morvanzhou.github.io"
"""

import numpy as np
from Env.maze_env import Maze
from collections import defaultdict


class MC_IS_wrapper:

    """MC Important Sampling
    """

    def __init__(self, env, n_actions=4, discount=0.9):

        self.env = env
        self.n_actions = n_actions

        self.Q = defaultdict(lambda: np.random.uniform(0,1,n_actions))
        self.C = defaultdict(lambda: np.zeros(n_actions))
        self.discount = discount
        self.behavoir_policy = 1 / self.n_actions
        self.sard = []
        self.e = e

    def e_greedy_explore(self, s):
        """探索
        return state, action
        """
        a = np.random.choice(np.arange(len(self.env.action_space)))
        state_next, reward, done = env.step(a)
        return state_next, a,reward, done

    def sampling(self):
        self.sard = []
        s = self.env.reset()
        d = False
        while not d:
            s_next, a, r, d = self.e_greedy_explore(s)
            self.sard.append((s, a, r, d))
            s = s_next

    def update(self, episode = 100):
        """
        update Q using important sampling
        1. Sampling
        2. reversed sard
        """
        for i in range(100):

            W = 1.0
            G = 0.0
            self.sampling()
            for s,a,r,d in reversed(self.sard):
                G = self.discount * G + r
                self.C[str(s)][a] = self.C[str(s)][a] + W
                self.Q[str(s)][a] = self.Q[str(s)][a] + W / self.C[str(s)][a] * (G - self.Q[str(s)][a])
                if not (np.argmax(self.Q[str(s)]) == a):
                    break
                W *= 1 / self.behavoir_policy


# init env
env = Maze()
mc = MC_IS_wrapper(env = env)

mc.update(1000)
