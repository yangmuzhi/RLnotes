"""
agent use keras
"""

import numpy as np
np.random.uniform(1130,1300)
import random
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Flatten

class Agent:
    """

    """
    def __init__(self,state_shape,n_action,
                lr, epsilon, net):
        self.state_shape = state_shape
        self.n_action = n_action
        self.q_eval_net = self.build_net(net)
        self.q_eval_net.compile(Adam(lr), 'mse')
        self.epsilon = epsilon
        self.epsilon_decay = 0.95
        self.q_target_net = self.build_net(net)

    def build_net(self, net):
        if isinstance(self.state_shape, int):
            inputs = Input((self.state_shape,))
        else:
            inputs = Input(self.state_shape)
        x = net(inputs)
        x = Dense(self.n_action, activation='linear')(x)
        return Model(inputs, x)

    def update(self, state, q):
        """ agent更新
        """
        self.q_eval_net.fit(state, q, epochs=1,verbose=0)

    def e_greedy_action(self, state):
        if np.random.uniform() <= self.epsilon:
            self.epsilon *= self.epsilon_decay
            return random.randrange(self.n_action)
        else :
            self.epsilon *= self.epsilon_decay
            return np.argmax(self.q_eval(state)[0])
