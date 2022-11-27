import numpy as np

from data_reader import DataReader
from data_visualizer import DataVisualizer
from hopfield_network import HopfieldNetwork
from learning_rules import LearningRules

# Initialize seed for random number generator
np.random.seed(1)

# Hopfield network parameters
NUMBER_OF_NEURONS_TO_UPDATE = 16
MAX_ITERATIONS = 15
LEARNING_RULE = LearningRules.oja
IS_UPDATE_SYNCHRONOUS = True

# Read collection of patterns from file
paths = ['data/animals-14x9.csv',
'data/large-25x25.csv',
'data/large-25x25.plus.csv',
'data/large-25x50.csv',
'data/letters-14x20.csv',
'data/letters-abc-8x12.csv',
'data/OCRA-12x30-cut.csv',
'data/small-7x7.csv']


sizes = [(14, 9), (25, 25), (25, 25), (25, 50), (14, 20), (8, 12), (12, 30), (7, 7)]

for path, size in zip(paths, sizes):
    patterns = DataReader.read_data(path)
    output_path = path.replace('data', 'results').replace('.csv', '')

    # Create noised pattern
    noised_patterns = patterns.copy()
    noise_level = 0.1
    for i in range(noised_patterns.shape[0]):
        for j in range(noised_patterns.shape[1]):
            if np.random.random() < noise_level:
                noised_patterns[i, j] = -patterns[i, j]

    # Create network and initialize memory with collection of patterns
    hopfield_network = HopfieldNetwork(patterns, IS_UPDATE_SYNCHRONOUS)
    # Learn network using one of supported learning rules
    hopfield_network.learning(LEARNING_RULE)

    for i in range(patterns.shape[0]):
        # Set network state to the noised image
        data_visualizer = DataVisualizer(size, patterns[i], noised_patterns[i], f'Sample {i} from {path}', output_path+f'_sample{i}_oja')
        hopfield_network.run(noised_patterns[i], NUMBER_OF_NEURONS_TO_UPDATE, MAX_ITERATIONS, data_visualizer)

