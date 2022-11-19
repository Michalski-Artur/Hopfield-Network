import numpy as np

from learning_rules import LearningRuleType


class HopfieldNetwork:
    # create network with set of patterns. Patterns -> array of arrays? Single image is going to be represented by single array?
    # number_of_neurons correspond to the size of the single pattern?
    def __init__(self, patterns):
        self.memory = patterns
        self.number_of_neurons = patterns.shape[1]
        self.neurons_state = np.random.randint(-2, 2, (self.number_of_neurons, 1))
        self.weights = np.zeros((self.number_of_neurons, self.number_of_neurons))

    def learning(self, learning_rule: LearningRuleType):
        self.weights = learning_rule(self.memory)
        # to ensure convergence, remember that matrix also has to be symmetric
        np.fill_diagonal(self.weights, 0)

    def set_initial_neurons_state(self, neurons_state):
        self.neurons_state = neurons_state

    def update_state(self, number_of_neurons_to_update):
        for neuron in range(number_of_neurons_to_update):
            random_index = np.random.randint(0, self.number_of_neurons)
            self.neurons_state[random_index] = self.__threshold_state_change_function(random_index)

    def __activation_function(self, neuron_index):
        return np.dot(self.weights[neuron_index, :], self.neurons_state)

    def __threshold_state_change_function(self, neuron_index):
        return 1 if self.__activation_function(neuron_index) > 0 else -1

