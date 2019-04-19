"""

"""

from collections import defaultdict
from Env.kuhn_poker_env import Kuhn_Poker
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
                            "action": None,
                            "card": None,
                            "info": None,
                            })


class Agent:

    def __init__(self, name, num_actions, Tree):
        self.name = name
        self.num_actions = num_actions
        self.policy = defaultdict(lambda: np.ones(self.num_actions) / self.num_actions)
        self.cum_regrets = defaultdict(lambda: np.zeros(self.num_actions))
        self.cum_strategy = defaultdict(lambda: np.zeros(self.num_actions))
        self.Tree = Tree
        self.node_id = 1

    def _build_chance_node(self, card):
        if not (card in )

    def _search(self, state):


    def choose_action(self, node):
        if not (node in self.Tree):
            np.random.randint(self.num_actions)
        else:
            pass
        return a

    def update_tree(self, node):
        # 更新node 和  value
        pass


class CFR_warpper:

    def __init__(self, env, agents):
        self.env = env
        # self.agent0 = agents[0]
        # self.agent1 = agents[1]
        self.agents = agents

    def is_terminate(self):
        pass

    def is_chance(self):
        pass

    def get_utilties(self):
        pass

    def run_to_end(self):
        api = self.env.start_play()
        next(api)
        d = False
        player = 0
        self.env.card_in_hand[0]
        while not d:
            a = self.agents[player % 2].choose_action()
            h, d, r = api.send(a)
            self.agents[player % 2].update_tree(h,d,r)
            player += 1

    def train(self, epsoides=10, shuffle_players=True):
        for i in range(epsoides):
            if shuffle_players:
                np.random.shuffle(self.agents)
            self.run_to_end()
