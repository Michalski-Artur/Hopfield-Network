import numpy as np

from data_reader import DataReader
from data_visualizer import DataVisualizer
from hopfield_network import HopfieldNetwork
from learning_rules import LearningRules

# Initialize seed for random number generator
np.random.seed(1)

# Hopfield network parameters
learning_rule = LearningRules.hebb
is_update_synchronous = True

# Read collection of patterns from file
path_to_file = 'data/large-25x25.csv'
single_pattern_size = (25, 25)
patterns = DataReader.read_data(path_to_file)

# Create noised pattern
noised_patterns = patterns.copy()
noise_level = 0.1
for i in range(noised_patterns.shape[0]):
    for j in range(noised_patterns.shape[1]):
        if np.random.random() < noise_level:
            noised_patterns[i, j] = -patterns[i, j]

# Create network and initialize memory with collection of patterns
hopfield_network = HopfieldNetwork(patterns, is_update_synchronous)

# Learn network using one of supported learning rules
hopfield_network.learning(learning_rule)

for i in range(patterns.shape[0]):
    # Set network state to the noised image
    hopfield_network.set_initial_neurons_state(noised_patterns[i])
    # Visualize original pattern and convergence
    DataVisualizer.visualize_pattern(noised_patterns[i].reshape(single_pattern_size))
    DataVisualizer.visualize_convergence(hopfield_network, single_pattern_size)


