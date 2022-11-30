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

paths = ['data/custom_tests.csv']
sizes = [(150, 150)]

for path, size in zip(paths, sizes):
    patterns = DataReader.read_data(path)
    output_path = path.replace('data', 'results').replace('.csv', '')

    # Create noised pattern
    noised_patterns = patterns.copy()
    for i in range(noised_patterns.shape[0]):
        for j in range(noised_patterns.shape[1]):
            if np.random.random() < NOISE_LEVEL:
                noised_patterns[i, j] = -patterns[i, j]

    # HEBB
    hopfield_network = HopfieldNetwork(patterns.copy(), IS_UPDATE_SYNCHRONOUS)
    hopfield_network.learning(LearningRules.hebb)
    for i in range(patterns.shape[0]):
        # Set network state to the noised image
        input = noised_patterns[i].copy()
        data_visualizer = DataVisualizer(size, patterns[i], input, f'Sample {i+1} from {path}, rule: Hebb', output_path+f'_sample{i+1}_hebb')
        hopfield_network.predict(input, NUMBER_OF_NEURONS_TO_UPDATE, MAX_ITERATIONS, data_visualizer)