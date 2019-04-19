"""

Kuhn Poker:is a simple 3-card poker game .
Two players each ante 1 chip,
i.e. bet 1 chip blind into the pot before the deal.
Three cards, marked with numbers 1, 2, and 3, are shuffled,
and one card is dealt to each player and held as private information.
Play alternates starting with player 1. On a turn, a player may either pass or bet.
A player that bets places an additional chip into the pot. When a player passes after a bet,
the opponent takes all chips in the pot. When there are two successive passes
 or two successive bets, both players reveal their cards, and the player with
 the higher card takes all chips in the pot.

# 注意这是两人的环境

"""

import numpy as np
from collections import defaultdict
CARDS = ["1", "2", "3"]
# 0, 1
VALID_ACTION = ["PASS", "BET"]

TREE = defaultdict(lambda:{"type": None,
                            "parent": None,
                            "child": None,
                            "value": 0,
                            })

class Kuhn_Poker:

    def __init__(self):
        self.cards = np.array([1,2,3])

    def _shuffle(self):
        np.random.shuffle(self.cards)

    def _end_info(self):

        if self.card_in_hand[0] < self.card_in_hand[1]:
            r = 0, self.pot
        else:
            r = self.pot, 0

        return r

    def _check_end(self):
        if self.round < 1:
            return 0, False

        elif self.round == 1:
            a = np.array([ self.h_dict[i]["action"] for i in range(len(self.h_dict))])
            # 两个人都放弃
            if (a == 0).all():
                d = True
                r = self._end_info()
            elif a[0] == 1:
                d = True
                r = self._end_info()
            else:
                return 0, False

        elif self.round == 2:
            d = True
            r = self._end_info()

        return r,d

    def step(self, action):


        self.h_dict[self.round]["player"] = self.round % 2
        self.h_dict[self.round]["action"] = action
        self.pot += action

        r, d = self._check_end()
        self.h_dict[self.round]["value"] = r
        self.h_dict[self.round]["done"] = d
        self.round += 1

        return r, d

    def reset(self):
        self._shuffle()
        self.round = 0
        self.pot = 2
        self.card_in_hand = self.cards[:2]
        self.h_dict = defaultdict(lambda: { "player":None, "action":None,
                                           "value":0, "done":False})

    def start_play(self):
        self.reset()
        d = False

        while True:
            action = yield self.h_dict, d, None
            r, d = self.step(action)
            if d:
                yield self.h_dict, d, self._end_info()
