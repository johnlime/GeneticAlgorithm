"""
This is a Numpy module for Reinforcement Learning with Genetic Algorithm Optimizer
"""
import numpy as np

class NonBiasedNN:
    def __init__(self, input_size=None, hidden_size=None, output_size=None):
        self.W = np.random.rand(hidden_size, input_size)
        self.U = np.random.rand(output_size, hidden_size)

    def output(self, input_vector):
        """
        Input: (self.input_size, 1) Numpy input vector
        Output: (self.output_size, 1) Numpy output vector
        """
        hidden_output = np.amax(np.stack((np.dot(self.W, input_vector), np.zeros(np.shape(self.W)[0]))), axis=0)
        nn_output = np.amax(np.stack((np.dot(self.U, hidden_output), np.zeros(np.shape(self.U)[0]))), axis=0)
        softmax_prob = np.exp(nn_output)/np.sum(np.exp(nn_output))
        return np.argmax(softmax_prob)

class GeneticRL:
    def __init__(self, population_size=100, input_size=None, hidden_size=None, output_size=None):
        self.pop_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #Gene Pool
        self.gene_pool= []
        for i in range(self.pop_size):
            individual_agent = NonBiasedNN(self.input_size, self.hidden_size, self.output_size)
            self.gene_pool.append(individual_agent)
        self.fitness = np.empty(self.pop_size)
