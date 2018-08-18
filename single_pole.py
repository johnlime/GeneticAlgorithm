import gym
from rlga import GeneticRL

env = gym.make('CartPole-v0')
grl = GeneticRL(population_size=100, input_size=4, hidden_size=8, output_size=2)

for generation in range(80):
    for individual in range(grl.pop_size):
        observation = env.reset()
        value = 0
        for t in range(1000):
            env.render()
            action = grl.gene_pool[individual].output(observation)
            observation, reward, done, info = env.step(action)
            value += reward #Calculating value of each session of each individual agent
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                value -= 10 #Penalty for losing
                break
            elif t==999:
                print("Episode made to the end")
        grl.fitness[individual] = value
    grl.best_of()
    if generation < 19:
        grl.optimizer(2, 5)
    else:
        print("End of Experiment")
        continue

import pickle
object = grl.best_network
filehandler = open('bestnnagent.pkl', 'wb')
pickle.dump(object, filehandler, protocol=pickle.HIGHEST_PROTOCOL)

object2 = grl
filehandler2 = open('progress.pkl', 'wb')
pickle.dump(object2, filehandler2, protocol=pickle.HIGHEST_PROTOCOL)
