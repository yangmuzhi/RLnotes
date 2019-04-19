import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import RMSprop


class Actor:
    """

    """
    def __init__(self,state_shape,n_action,
                lr, share_net):
        self.state_shape = state_shape
        self.n_action = n_action
        self.lr = 1e-4
        self.model = self.build_net(share_net)
        self.action = K.placeholder(shape=(None, self.n_action))
        self.td_error = K.placeholder(shape=(None,1))

        self.opt = RMSprop(lr=self.lr)

    def build_net(self,share_net):
        """
        """
        x = Dense(128, activation='relu')(share_net.output)
        x = Dense(self.n_action, activation='softmax')(x)
        return Model(share_net.input, x)

    def update(self):
        """
        """
        weighted_actions = K.sum(self.action * self.model.output, axis=1)
        loss = - K.sum(K.log(weighted_actions + 1e-10) * K.stop_gradient(self.td_error))
        updates = self.opt.get_updates(self.model.trainable_weights, [], loss)
        return K.function([self.model.input, self.action, self.td_error], [], updates=updates)

    def action_prob(self, state):
        """  给出动作的概率
        """
        return self.model.predict(state)

    def explore(self, state):
        """
        """
        prob = self.action_prob(state)
        return np.random.choice(np.arange(self.n_action), p=prob.ravel())
