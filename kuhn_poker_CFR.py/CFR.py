"""

"""

from collections import defaultdict
import numpy as np

"""
每个agent会自己生成一个树

type: str chance , terminate
parent:
child: list
value: numeric

"""


Tree = defaultdict(lambda:{"player":None,
                            "type": None,
                            "parent": None,
                            "child": None,
                            "value": 0,
                            "valid_action": [0,1],
                            # "info_set": None
                            })

class Agent:

    def __init__(self, num_actions, Tree):
        self.num_actions = num_actions
        self.policy = defaultdict(lambda: np.ones(self.num_actions) / self.num_actions)
        self.cum_regrets = defaultdict(lambda: np.zeros(self.num_actions))
        self.cum_strategy = defaultdict(lambda: np.zeros(self.num_actions))
        self.Tree = Tree

    def choose_action(self):
        return np.random.randint(2)

    def update_tree(self, node):
        pass


class CFR_warpper:

    def __init__(self, env, agents):
        self.env = env
        self.agent0 = agents[0]
        self.agent1 = agents[1]

    def is_terminate(self):
        pass

    def get_utilties(self):
        pass

    def get_action(self):
        pass

    def step(self):
        pass

    def run(self, node):
        if is_terminate(node):
            return get_utilties(node)
        else:
            action = self.get_action(node)
            node = self.step(action)

    def train(self, epsoides=10):
        node = self.reset()
        self.run(node)
