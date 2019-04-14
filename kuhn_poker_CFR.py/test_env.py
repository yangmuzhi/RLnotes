from Env.kuhn_poker_env import Kuhn_Poker
import numpy as np
env = Kuhn_Poker()
api = env.start_play()
next(api)

env.card_in_hand
api.send(0)
api.send(1)
