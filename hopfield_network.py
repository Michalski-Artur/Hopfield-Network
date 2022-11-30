import numpy as np
from data_visualizer import DataVisualizer

from learning_rules import LearningRuleType
from itertools import cycle

class HopfieldNetwork:
    # create network with set of patterns. Patterns -> array of arrays? Single image is going to be represented by single array?
    # number_of_neurons correspond to the size of the single pattern?
    def __init__(self, patterns, is_update_synchronous):
        self.memory = patterns
        self.number_of_neurons = patterns.shape[1]
        self.neurons_state = np.zeros((self.number_of_neurons, 1))
        self.weights = np.zeros((self.number_of_neurons, self.number_of_neurons))
        self.is_update_synchronous = is_update_synchronous
        self.asynchronous_update_iterator = cycle(range(self.number_of_neurons))

    def learning(self, learning_rule: LearningRuleType):
        self.weights = learning_rule(self.memory)

    def predict(self, test_sample, number_of_neurons_to_update = 1, iterations_limit = None, data_visualizer: DataVisualizer = None):
        self.__set_initial_neurons_state(test_sample)
        iter = 0
        while iterations_limit is None or iter < iterations_limit:
            self.__update_state(number_of_neurons_to_update)
            if data_visualizer is not None:
                is_last_iteration = iterations_limit is not None and iter == iterations_limit - 1
                if not data_visualizer.visualize_step(self.neurons_state, iter, is_last_iteration):
                    break
            iter += 1

    def __set_initial_neurons_state(self, neurons_state):
        self.neurons_state = neurons_state

    def __update_state(self, number_of_neurons_to_update=1):
        self.__synchronous_state_update() if self.is_update_synchronous else self.__asynchronous_state_update(number_of_neurons_to_update)

    def __synchronous_state_update(self):
        updated_states = []
        for neuron_index in range(self.number_of_neurons):
            updated_states.append(self.__threshold_state_change_function(neuron_index))
        self.neurons_state = np.array(updated_states)

    def __asynchronous_state_update(self, number_of_neurons_to_update):
        for _ in range(number_of_neurons_to_update):
            index_of_neuron_to_update = next(self.asynchronous_update_iterator)
            self.neurons_state[index_of_neuron_to_update] = self.__threshold_state_change_function(index_of_neuron_to_update)

    def __activation_function(self, neuron_index):
        return np.dot(self.weights[neuron_index, :], self.neurons_state)

    def __threshold_state_change_function(self, neuron_index):
        return 1 if self.__activation_function(neuron_index) >= 0 else -1

