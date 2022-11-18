import numpy as np

from hopfield_network import HopfieldNetwork
from learning_rules import LearningRules

patterns = np.array([1, 1, 1, 1, -1, -1])

hopfield_network = HopfieldNetwork(patterns)
hopfield_network.learning(LearningRules.hebb)
hopfield_network.update_state(1)
