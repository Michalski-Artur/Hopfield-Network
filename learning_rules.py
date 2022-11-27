from typing import Callable
from math_utils import MathUtils
import numpy as np

LearningRuleType = Callable[[np.ndarray], np.ndarray]


class LearningRules:

    @staticmethod
    def hebb(patterns: np.ndarray):
        weights = (1 / patterns.shape[0]) * patterns.T @ patterns
        np.fill_diagonal(weights, 0)
        return weights

    @staticmethod
    def oja(patterns: np.ndarray):
        weights = LearningRules.hebb(patterns)
        max_iter = 10
        factor = patterns.shape[1] * patterns.shape[1]
        prev_dif = np.inf
        prev_weights = weights
        for iter in range(max_iter):
            print(f'Learning patterns with Oja\'s rule: iteration {iter + 1}/{max_iter}...')
            delta_weights = np.zeros(weights.shape)
            y_array = weights @ patterns.T
            for i in range(patterns.shape[1]):
                for j in range(patterns.shape[1]):
                    if i == j:
                        continue
                    delta_weights[i, j] = y_array[i, :] @ (patterns[:, j] - weights[i, j] * y_array[i, :]).T
            delta_weights = delta_weights / factor
            weights = weights + delta_weights
            diff = np.linalg.norm(delta_weights)
            print(f'Delta = {diff}')
            if diff < 1e-5:
                print (f'Learning converged after {iter} iterations')
                break
            if diff - prev_dif > 1e-5:
                print (f'Learning diverged after {iter} iterations')
                weights = prev_weights
                break
            prev_dif = diff
            prev_weights = weights
        return weights
