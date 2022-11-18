import numpy as np

from data_reader import DataReader
from hopfield_network import HopfieldNetwork
from learning_rules import LearningRules

path_to_file = 'data/small-7x7.csv'
patterns = DataReader.read_data(path_to_file)

hopfield_network = HopfieldNetwork(patterns)
hopfield_network.learning(LearningRules.hebb)
hopfield_network.update_state(1)


