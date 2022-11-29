from typing import Callable
import numpy as np

from math_utils import MathUtils

LearningRuleType = Callable[[np.ndarray], np.ndarray]

MAX_ITERATIONS = 100
LEARNING_COEFF = 1e-6

class LearningRules:

    @staticmethod
    def hebb(patterns: np.ndarray):
        weights = patterns.T @ patterns / patterns.shape[0]
        np.fill_diagonal(weights, 0)
        return weights

    @staticmethod
    def oja(patterns: np.ndarray):
        weights = LearningRules.hebb(patterns)
        for iter in range(MAX_ITERATIONS):
            print(f'Learning patterns with Oja\'s rule: iteration {iter + 1}/{MAX_ITERATIONS}...')
            delta = np.zeros_like(weights)
            for pattern in patterns:
                factor = LEARNING_COEFF / (patterns.shape[0] * patterns.shape[1])
                y_vector = MathUtils.sign(np.dot(weights, pattern))
                delta += factor * (np.dot(y_vector, pattern) - np.square(y_vector) * weights)
            weights += delta
            diff = np.linalg.norm(delta)
            print(f'Delta = {diff}')
            if diff < 1e-6:
                print (f'Learning converged after {iter} iterations')
                break
        return weights
