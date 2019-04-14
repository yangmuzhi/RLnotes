"""
Kuhn Poker
Counterfactual Regret Minimization (with chance sampling)
"""

import numpy as np
from collections import defaultdict

# d = defaultdict(lambda: np.ones(2) / 2)

class Agent:

    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.policy = defaultdict(lambda: np.ones(self.num_actions) / self.num_actions)
        self.cum_regrets = defaultdict(lambda: np.zeros(self.num_actions))
        self.cum_strategy = defaultdict(lambda: np.zeros(self.num_actions))

    def choose_action(self):
        pass

    def get_value(self):
        pass

    def update(self):
        pass

class Kuhn_Poker(self):
    """
    注意这是两人的环境
    """

    def __init__(self):
        pass

    def step(self):
        while True:
            yield

class CFR_warpper:

    def __init__(self):
        pass
