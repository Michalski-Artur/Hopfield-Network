from typing import Callable
from math_utils import MathUtils
import numpy as np

LearningRuleType = Callable[[np.ndarray], np.ndarray]

MAX_ITERATIONS = 50
# LEARNING_COEFF = 0.1

class LearningRules:

    @staticmethod
    def hebb(patterns: np.ndarray):
        weights = patterns.T @ patterns / patterns.shape[0]
        np.fill_diagonal(weights, 0)
        return weights

    @staticmethod
    def oja(patterns: np.ndarray):
        weights = LearningRules.hebb(patterns)
        # factor = LEARNING_COEFF / patterns.shape[1] ** 2
        #weights = np.zeros((patterns.shape[1], patterns.shape[1])) + 1 #Maybe better ? Or initialize with random?
        prev_dif = np.inf
        prev_weights = weights
        for iter in range(MAX_ITERATIONS):
            print(f'Learning patterns with Oja\'s rule: iteration {iter + 1}/{MAX_ITERATIONS}...')
            delta_weights = ((patterns.T @ patterns) / patterns.shape[0] * weights -
                             weights.T * ((patterns.T @ patterns) / patterns.shape[0]) * weights * weights / 2) / 4
            weights = weights + delta_weights
            diff = np.linalg.norm(delta_weights)
            print(f'Delta = {diff}')
            if diff < 1e-5:
                print(f'Learning converged after {iter} iterations')
                break
            if diff - prev_dif > 1:
                print (f'Learning diverged after {iter} iterations')
                weights = prev_weights
                break
            prev_dif = diff
            prev_weights = weights
        return weights - 1
