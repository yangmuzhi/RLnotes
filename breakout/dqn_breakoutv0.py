"""
测试dqn breakoutv0
"""

import gym
import sys
from utils.net import conv_breakout
from dqn.deepqn import DQN
import os
eps = int(sys.argv[1])


env = gym.make("Breakout-v0")
state = env.reset()
state_shape = state.shape

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

dqn = DQN(state_shape=state_shape, n_action=4, net=conv_breakout)
dqn.agent.q_eval_net.summary()

dqn.train(env, eps, 256)
