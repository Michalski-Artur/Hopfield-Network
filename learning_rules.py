from typing import Callable
from math_utils import MathUtils
import numpy as np

LearningRuleType = Callable[[np.ndarray], np.ndarray]

MAX_ITERATIONS = 5000
LEARNING_COEFF = 1e-3

class LearningRules:

    @staticmethod
    def hebb(patterns: np.ndarray):
        weights = (1 / patterns.shape[0]) * patterns.T @ patterns
        np.fill_diagonal(weights, 0)
        return weights

    @staticmethod
    def oja(patterns: np.ndarray):
        weights = LearningRules.hebb(patterns)
        for iter in range(MAX_ITERATIONS):
            print(f'Learning patterns with Oja\'s rule: iteration {iter + 1}/{MAX_ITERATIONS}...')
            delta = np.zeros_like(weights)
            for pattern in patterns:
                y_vector = np.sign(np.dot(weights, pattern.T))
                delta += LEARNING_COEFF * (np.outer(y_vector, pattern) - np.square(y_vector) * weights)
            weights += delta
            diff = np.linalg.norm(delta)
            print(f'Delta = {diff}')
            if diff < 1e-6:
                print (f'Learning converged after {iter} iterations')
                break
        return weights
