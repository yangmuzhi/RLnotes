import gym
import random
import sys

env = gym.make("CartPole-v0")
env.action_space.n
### a random action
def random_action(n_action):
    return random.choice(range(n_action))
env.reset()
while True:
    state = env.render()
    action = random_action(2)
    state, reward, done, _ = env.step(action)
    if done :
        break
env.close()
