import numpy as np
import random

from data_reader import DataReader
from data_visualizer import DataVisualizer
from hopfield_network import HopfieldNetwork
from learning_rules import LearningRules

# Hopfield network parameters
learning_rule = LearningRules.oja
is_update_synchronous = True

# Read collection of patterns from file
path_to_file = 'data/large-25x25.csv'
single_pattern_size = (25, 25)
patterns = DataReader.read_data(path_to_file)

# Create noised pattern
index_of_pattern_to_noise = np.random.randint(len(patterns))
noise_scale = 0.05
pattern_to_noise = patterns[index_of_pattern_to_noise]
indexes_to_flip = random.sample(range(1, len(pattern_to_noise)), int(noise_scale*len(pattern_to_noise)))
noised_pattern = np.array(pattern_to_noise)
for i in indexes_to_flip:
    noised_pattern[i] *= -1

# Create network and initialize memory with collection of patterns
hopfield_network = HopfieldNetwork(patterns, is_update_synchronous)

# Learn network using one of supported learning rules
hopfield_network.learning(learning_rule)

# Set network state to the noised image
hopfield_network.set_initial_neurons_state(noised_pattern)

# Visualize original pattern and convergence
DataVisualizer.visualize_pattern(pattern_to_noise.reshape(single_pattern_size))
DataVisualizer.visualize_convergence(hopfield_network, single_pattern_size)
