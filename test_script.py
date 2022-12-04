import numpy as np

from data_reader import DataReader
from data_visualizer import DataVisualizer
from hopfield_network import HopfieldNetwork
from learning_rules import LearningRules

# Initialize seed for random number generator
np.random.seed(1)

# Hopfield network parameters
NUMBER_OF_NEURONS_TO_UPDATE = 32
MAX_ITERATIONS = 15
IS_UPDATE_SYNCHRONOUS = True
NOISE_LEVEL = 0.0

# Read collection of patterns from file
paths = [
'data/animals-14x9.csv',
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
    for i in range(noised_patterns.shape[0]):
        for j in range(noised_patterns.shape[1]):
            if np.random.random() < NOISE_LEVEL:
                noised_patterns[i, j] = -patterns[i, j]

    #OJA
    # Create network and initialize memory with collection of patterns
    hopfield_network = HopfieldNetwork(patterns.copy(), IS_UPDATE_SYNCHRONOUS)
    hopfield_network.learning(LearningRules.oja)
    for i in range(patterns.shape[0]):
        # Set network state to the noised image
        input = noised_patterns[i].copy()
        data_visualizer = DataVisualizer(size, patterns[i], input, f'Sample {i+1} from {path}, rule: Oja', output_path+f'_sample{i+1}_oja')
        hopfield_network.predict(input, NUMBER_OF_NEURONS_TO_UPDATE, MAX_ITERATIONS, data_visualizer)

    # HEBB
    hopfield_network = HopfieldNetwork(patterns.copy(), IS_UPDATE_SYNCHRONOUS)
    hopfield_network.learning(LearningRules.hebb)
    for i in range(patterns.shape[0]):
        # Set network state to the noised image
        input = noised_patterns[i].copy()
        data_visualizer = DataVisualizer(size, patterns[i], input, f'Sample {i+1} from {path}, rule: Hebb', output_path+f'_sample{i+1}_hebb')
        hopfield_network.predict(input, NUMBER_OF_NEURONS_TO_UPDATE, MAX_ITERATIONS, data_visualizer)