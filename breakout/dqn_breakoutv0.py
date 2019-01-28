"""
测试dqn breakoutv0
"""

import gym
import sys
from utils.net import conv_breakout
from dqn.deepqn import DQN

eps = int(sys.argv[1])


env = gym.make("Breakout-v0")
state = env.reset()
state_shape = state.shape 

dqn = DQN(state_shape=state_shape, n_action=2, net=conv_breakout)
dqn.agent.q_eval_net.summary()

dqn.train(env, eps, 256)