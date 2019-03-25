#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q learning is the off policy version of TD control
"""


from Env.maze_env import Maze
import numpy as np
from collections import defaultdict


class table_qlearning:
    """ qleanring  using table
        n_actions: agent动作的数量
        learning_rate:  Q table 更新率
        discount factor: 0-1， 折扣因子
        epsilon: 0-1，探索概率（越大越不去探索）

    """

    ##init
    def __init__(
            self, n_actions,
            env,
            learning_rate=0.01,
            discount_factor=0.9,
            epsilon=0.9,
            ):

        self.env = env
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self.obs = []
        self.episode = 0
        self.reward_his = defaultdict(lambda: 0)

    ##e-greedy choose action
    def e_greedy_choose_action(self, state):

        if np.random.uniform() < self.epsilon:
            # 当出现最大有多个情况时，应该随机选择
            action_value = self.Q[str(state)]
            action = np.random.choice(
                        np.where(action_value == max(action_value))[0])
        else:
            # random choose an action
            action = np.random.randint(self.n_actions)

        return action

    def update(self, obs):
        # s,s_next,a,r,d
        s = obs[0]
        s_next = obs[1]
        a = obs[2]
        r = obs[3]
        d = obs[4]
        # Bellman equations
        # Q(s_t) = Q(s_t) + alpha * (r + gamma * max[Q(s_t+1)] - Q(s_t))
        self.Q[str(s)] += self.learning_rate * (r +
                        self.discount_factor * max(self.Q[str(s_next)])
                        - self.Q[str(s)][a])

    def train(self, epsiode=10):
        d = False
        s = self.env.reset()
        for i in range(epsiode):
            cum_r = 0
            while not d:
                a = self.e_greedy_choose_action(s)
                s_next, r, d = self.env.step(a)
                self.update((s,s_next,a,r,d))
                s = s_next
                cum_r += r
        self.reward_his[i] = cum_r


if __name__ == "__main__":
    env = Maze()
    q_learning = table_qlearning(
            n_actions = 4,
            env = env,
            epsilon=0.8,
            )
    s = env.reset()
    q_learning.train(10)
    q_learning.Q
