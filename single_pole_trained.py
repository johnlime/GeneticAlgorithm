import gym

env = gym.make('CartPole-v0')

import pickle
f = open('progress.pkl', 'rb')
obj = pickle.load(f)
grl = obj

observation = env.reset()
while True:
    env.render()
    action = grl.gene_pool[3].output(observation)
    observation, reward, done, info = env.step(action)
