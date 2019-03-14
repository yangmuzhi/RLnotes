from snake_env import Snakes_subsonic
from dqn.deepqn import DQN
from utils.net import  simple_net
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
env = Snakes_subsonic()
state = env.reset()

dqn = DQN(state_shape=84, n_action=3, net=simple_net)
dqn.agent.q_eval_net.summary()

batch_size = 32
eps = 1000

dqn.train(env, eps, batch_size)
plt.plot(range(len(dqn.cum_r)),dqn.cum_r)

def play(N=200):
    r = []
    tqdm_e = tqdm(range(N))
    for i in tqdm_e:

        state = env.reset()

        cum_r = 0
        done = False
        while not done:
            env.render()
            state = state.reshape(-1,84)
            action = np.argmax(dqn.agent.q_eval(state))
            state, reward, done, _ = env.step(action)
            cum_r += reward
        r.append(cum_r)
    plt.plot(range(len(r)), np.array(r))
play(10)
