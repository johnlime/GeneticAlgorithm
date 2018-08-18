"""
This is a Numpy module for Reinforcement Learning with Genetic Algorithm Optimizer
"""
import numpy as np
import random

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
        self.gene_length = self.input_size*self.hidden_size + self.hidden_size*self.output_size

        #Gene Pool (Stored as set of weight matrices)
        self.gene_pool= []
        for i in range(self.pop_size):
            individual_agent = NonBiasedNN(self.input_size, self.hidden_size, self.output_size)
            self.gene_pool.append(individual_agent)

        #Record Data
        self.fitness = np.empty(self.pop_size)
        self.best_network = NonBiasedNN(self.input_size, self.hidden_size, self.output_size)
        self.best_fitness = 0

    def elitism(self, number_of_elites):
        elite_index = np.argsort(self.fitness)[:number_of_elites]
        elite_genes = []
        for i in range(np.size(elite_index)):
            elite_genes.append(self.gene_pool[i])
        return elite_genes

    def parent_genes(self, number_of_parents):
        score = self.fitness-np.amin(self.fitness)
        survival_probability = score/np.sum(score)
        parent_genotype = []
        for choice in np.random.choice(range(self.pop_size), number_of_parents, replace=False, p=survival_probability):
            parent_genotype.append(self.gene_pool[choice])
        return parent_genotype

    def crossover(self, parent_gene_a, parent_gene_b):
        """
        INPUT: (gene_length, 1)Numpy Ndarray: 2 Parent Genes
        OUTPUT: (gene_length, 1)Numpy Ndarray: 2 Child Gene
        """
        def child(_a, _b):
            snippets = sorted(random.sample(range(0, self.gene_length), 2))
            a = _a[snippets[0]:snippets[1]]
            b = _b
            child_gene = b[:snippets[0]]
            child_gene = np.append(child_gene, a)
            child_gene = np.append(child_gene, b[snippets[1]:])
            return child_gene
        child_genotype = child(parent_gene_a, parent_gene_b)
        child_genotype = np.stack((child_genotype, child(parent_gene_b, parent_gene_a)))
        return child_genotype

    def mutation(self, parent_gene):
        """
        INPUT: (gene_length, 1)Numpy Ndarray: Parent Gene
        OUTPUT: (gene_length, 1)Numpy Ndarray: Child Gene
        """
        child_gene = parent_gene + np.random.rand(self.gene_length)
        return child_gene

    def matrix_to_gene(self, nn_object):
        gene = np.reshape(nn_object.W, self.input_size*self.hidden_size)
        gene = np.append(gene, np.reshape(nn_object.U, self.hidden_size*self.output_size))
        return gene

    def gene_to_matrix(self, gene):
        W = np.reshape(gene[:self.input_size*self.hidden_size], (self.hidden_size, self.input_size))
        U = np.reshape(gene[self.input_size*self.hidden_size:], (self.output_size, self.hidden_size))
        new_object = NonBiasedNN(self.input_size, self.hidden_size, self.output_size)
        new_object.W = W
        new_object.U = U
        return new_object

    def best_of(self):
        self.best_fitness = np.max(self.fitness)
        self.best_network = self.gene_pool[np.argsort(self.fitness)[0]]
        print("Best Fitness: ", self.best_fitness)

    def optimizer(self, number_of_elites, number_of_parents, crossover_rate = 0.7):
        nextgen = self.elitism(number_of_elites)
        parents = self.parent_genes(number_of_parents)
        #Performing Crossover
        for iter in range(int(self.pop_size*crossover_rate)):
            a_index, b_index = random.sample(range(number_of_parents), 2)
            a_gene, b_gene = self.matrix_to_gene(parents[a_index]), self.matrix_to_gene(parents[b_index])
            children = self.crossover(a_gene, b_gene)
            nextgen = np.append(nextgen, self.gene_to_matrix(children[0]))
            nextgen = np.append(nextgen, self.gene_to_matrix(children[1]))
        #Performing Mutation
        for iter in range(self.pop_size - number_of_elites - int(self.pop_size * crossover_rate)):
            chosen_parent = random.sample(range(number_of_parents), 1)[0]
            chosen_gene = self.matrix_to_gene(parents[chosen_parent])
            mutant = self.mutation(chosen_gene)
            nextgen = np.append(nextgen, self.gene_to_matrix(mutant))
        self.gene_pool = nextgen
