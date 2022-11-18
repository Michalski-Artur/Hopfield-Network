# imports
import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork:  # network class
    # initialize network variables and memory
    def __init__(self, input):

        # patterns for network training / retrieval
        self.memory = np.array(input)
        # single vs. multiple memories
        if self.memory.size > 1:
            self.n = self.memory.shape[1]
        else:
            self.n = len(self.memory)
        # network construction
        self.state = np.random.randint(-2, 2, (self.n, 1))  # state vector
        self.weights = np.zeros((self.n, self.n))  # weights vector
        self.energies = []  # container for tracking of energy

    def network_learning(self):  # learn the pattern / patterns
        self.weights = (1 / self.memory.shape[0]) * self.memory.T @ self.memory  # hebbian learning
        np.fill_diagonal(self.weights, 0)

    def update_network_state(self, n_update):  # update network
        for neuron in range(n_update):  # update n neurons randomly
            self.rand_index = np.random.randint(0, self.n)  # pick a random neuron in the state vector
            # Compute activation for randomly indexed neuron
            self.index_activation = np.dot(self.weights[self.rand_index, :],
                                           self.state)
            # threshold function for binary state change
            if self.index_activation < 0:
                self.state[self.rand_index] = -1
            else:
                self.state[self.rand_index] = 1

    def compute_energy(self):  # compute energy
        self.energy = -0.5 * np.dot(np.dot(self.state.T, self.weights), self.state)
        self.energies.append(self.energy)