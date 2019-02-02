import gym
import random
import sys

env = gym.make("CartPole-v0")
env.action_space.n
### a random action
def random_action(n_action):
    return random.choice(range(n_action))
env.reset()
done = False
while not done:
    action = random_action(2)
    state, reward, done, _ = env.step(action)
    print(reward)
env.close()
