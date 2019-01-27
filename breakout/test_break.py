"""
用来实现测试breakout
"""

import gym
from A3C.A3C import A3C
from A2C.A2C import A2C
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from utils.net import conv_shared
env = gym.make("Breakout-v0")
state = env.reset()
state_shape = state.shape 
a3c = A3C(state_shape=state_shape, n_action=4, net=conv_shared, n_threads=2)
a3c.n_threads

a3c.train(env,2)


"""

a3c.trainAsy(env,2)

#state_newaxis = state[np.newaxis,:]
#for i in range(10):
#    next_state, reward, done, _ =env.step(0)
#    #next_state = next_state[np.newaxis,:]
#    action_onehot = to_categorical(0, 4)
#    ob = (state, reward, done, action_onehot, next_state)
#   a3c.sampling_pool.add_to_buffer(ob)



state = env.reset()
cum_r = 0
done = False
while not done:
    state_newaxis = state[np.newaxis,:]
    action = a3c.actor.explore(state_newaxis)
    next_state,reward,done, _ = env.step(action)
    action_onehot = to_categorical(action, a3c.n_action)
    ob = (state, reward, done, action_onehot, next_state)
    a3c.sampling_pool.add_to_buffer(ob)
    state = next_state
    cum_r += reward

state, reward, done, action, next_state = \
            a3c.sampling_pool.get_sample(shuffle=False)
value = a3c.critic.value(state)
next_value = a3c.critic.value(next_state)
td_error = reward + a3c.gamma * next_value - value
a3c.actor_update([state, action, td_error])
a3c.critic_update([state, value])

"""


