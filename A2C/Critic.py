import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop

class Critic:
    """

    """
    def __init__(self,state_shape,output_shape,
                lr, share_net):
        self.state_shape = state_shape
        self.output_shape = output_shape
        self.lr = lr
        self.model = self.build_net(share_net)
        self.state_value = K.placeholder(shape=(None,))
        self.opt = RMSprop(lr=self.lr)

    def build_net(self, share_net):
        """
        """
        x = Dense(64, activation='relu')(share_net.output)
        x = Dense(1, activation='linear')(x)
        return Model(share_net.input, x)

    def update(self):
        """
        """
        critic_loss = K.mean(K.square(self.state_value - self.model.output))
        updates = self.opt.get_updates(self.model.trainable_weights, [], critic_loss)

        return K.function([self.model.input, self.state_value], [], updates=updates)

    def fit(self, state, state_value):
        """ Perform one epoch of training
        """
        self.model.fit(state, state_value, epochs=1, verbose=0)

    def value(self, state):
        """ Critic Value Prediction
        """
        return self.model.predict(state)
