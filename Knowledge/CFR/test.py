from Env.kuhn_poker_env import Kuhn_Poker
import numpy as np
from Knowledge.CFR.agent import Tree
from copy import deepcopy

env = Kuhn_Poker()


def play():
    h = []
    for i in range(10):
        api = env.start_play()
        d = False
        _ = next(api)
        while not d:
            a = np.random.randint(2)
            h_dict,d,r = api.send(a)
        # h.append(list(h_dict.values()))
        h.append(h_dict)
    return h
h = play()

his = [list(i.values()) for i in h]
len(his)
his[0][0]

his[0][0]["action"]
his[0][0]["done"]
his[0][0]["player"]
his[0][0]["value"]

tree = Tree()

tree.generate()
