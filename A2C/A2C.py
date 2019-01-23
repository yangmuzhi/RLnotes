
import numpy as np
from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten
from utils.sample_buffer import Sampling_Pool
from A2C.Actor import Actor
from A2C.Critic import Critic

class A2C:
    """
    """
    def __init__(self, state_shape, n_action):
        self.state_shape = state_shape
        self.n_action = n_action
        self.lr = 1e-4
        self.gamma = 0.9
        share_net = self.build_share_net()
        self.critic = Critic(self.state_shape,1,
                self.lr, share_net)
        self.actor = Actor(self.state_shape,self.n_action,
                self.lr, share_net)
        self.cum_r = []
        self.sampling_pool = Sampling_Pool()
        self.actor_update = self.actor.update()
        self.critic_update = self.critic.update()

    def build_share_net(self):
        # 暂时使用一个简单的网络, 后面应该做一个net.py
        # actor 和 critic 共享一个net
        inputs = Input((self.state_shape,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        return Model(inputs, x)

    def update(self):
        state, reward, done, action, next_state = \
            self.sampling_pool.get_sample(shuffle=False)
        state = state.reshape(-1,self.state_shape)
        next_state = next_state.reshape(-1,self.state_shape)
        value = self.critic.value(state)
        next_value = self.critic.value(next_state)
        td_error = reward + self.gamma * next_value - value
        self.actor_update([state, action, td_error])
        self.critic_update([state, value])

    def train(self, env, episode):

        tqdm_e = tqdm(range(episode))
        for i in tqdm_e:
            state = env.reset()
            cum_r = 0
            done = False
            state = state.reshape(-1,self.state_shape)
            while not done:
                action = self.actor.explore(state)
                next_state,reward,done, _ = env.step(action)
                next_state = next_state.reshape(-1,self.state_shape)
                action_onehot = to_categorical(action, self.n_action)
                ob = (state, reward, done, action_onehot, next_state)
                self.sampling_pool.add_to_buffer(ob)
                state = next_state
                cum_r += reward

            self.update()
            self.cum_r.append(cum_r)
            self.sampling_pool.clear()

            tqdm_e.set_description("Score: " + str(cum_r))
            tqdm_e.refresh()
